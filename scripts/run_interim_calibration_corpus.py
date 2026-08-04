"""Drive a deterministic calibration corpus through the interim EXL3 server.

The script does not load model weights. It loads only the tokenizer from the
local interim serve directory, verifies that the live capture manifest names
the interim EXL3 hybrid as its teacher, and then sends pretokenized completion
requests. A content hash assigns whole JSONL records to disjoint folds, so a
separate training and validation capture cannot share a source document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

INTERIM_SOURCE = "interim_exl3_3p09_hybrid"
DEFAULT_MODEL_DIR = Path("/models/Kimi-K3-EXL3-3p09-serve")


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    weight: float
    max_prompt_tokens: int | None = None
    record_source: str | None = None


@dataclass(frozen=True)
class PlannedRequest:
    source: str
    line: int
    document_hash: str
    prompt_hash: str
    prompt_tokens: tuple[int, ...]

    @property
    def tokens(self) -> int:
        return len(self.prompt_tokens)


def _parse_source(value: str) -> SourceSpec:
    value, record_separator, record_source = value.partition("::")
    if record_separator and not record_source:
        raise argparse.ArgumentTypeError("record source filter must not be empty")
    path_text, separator, weight_text = value.rpartition("=")
    if not separator:
        return SourceSpec(Path(value), 1.0, record_source=record_source or None)
    weight_text, cap_separator, cap_text = weight_text.partition("@")
    try:
        weight = float(weight_text)
    except ValueError:
        return SourceSpec(Path(value), 1.0, record_source=record_source or None)
    if weight <= 0:
        raise argparse.ArgumentTypeError("source weight must be positive")
    max_prompt_tokens = None
    if cap_separator:
        try:
            max_prompt_tokens = int(cap_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "source token cap must be an integer"
            ) from exc
        if max_prompt_tokens <= 0:
            raise argparse.ArgumentTypeError("source token cap must be positive")
    return SourceSpec(
        Path(path_text), weight, max_prompt_tokens, record_source or None
    )


def _content_hash(raw: str) -> str:
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_excluded_documents(
    reports: list[Path],
) -> tuple[set[str], set[str], list[dict[str, object]]]:
    """Authenticate prior reports and collect document and prompt hashes."""

    document_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    identities: list[dict[str, object]] = []
    for raw_path in reports:
        path = raw_path.resolve()
        report = json.loads(path.read_text())
        if report.get("kind") != "kquant_interim_calibration_corpus_run":
            raise ValueError(f"{path} is not a calibration corpus report")
        documents = report.get("documents")
        if not isinstance(documents, list):
            raise ValueError(f"{path} has no document inventory")
        report_hashes: set[str] = set()
        report_prompt_hashes: set[str] = set()
        for document in documents:
            value = document.get("document_hash") if isinstance(document, dict) else None
            if not isinstance(value, str) or len(value) != 32:
                raise ValueError(f"{path} contains an invalid document hash")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(
                    f"{path} contains a non-hex document hash"
                ) from exc
            report_hashes.add(value)
            prompt_hash = (
                document.get("prompt_hash") if isinstance(document, dict) else None
            )
            if prompt_hash is not None:
                if not isinstance(prompt_hash, str) or len(prompt_hash) != 32:
                    raise ValueError(f"{path} contains an invalid prompt hash")
                try:
                    bytes.fromhex(prompt_hash)
                except ValueError as exc:
                    raise ValueError(
                        f"{path} contains a non-hex prompt hash"
                    ) from exc
                report_prompt_hashes.add(prompt_hash)
        identities.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "documents": len(report_hashes),
                "prompts": len(report_prompt_hashes),
            }
        )
        document_hashes.update(report_hashes)
        prompt_hashes.update(report_prompt_hashes)
    return document_hashes, prompt_hashes, identities


def _fold_selected(
    document_hash: str,
    *,
    modulus: int,
    index: int,
    mode: str,
) -> bool:
    bucket = int(document_hash, 16) % modulus
    selected = bucket == index
    return selected if mode == "include" else not selected


def _allocate_quotas(total: int, sources: list[SourceSpec]) -> list[int]:
    weight_sum = sum(source.weight for source in sources)
    quotas = [int(total * source.weight / weight_sum) for source in sources]
    for index in range(total - sum(quotas)):
        quotas[index % len(quotas)] += 1
    return quotas


def _record_tokens(row: dict, tokenizer: TokenizerLike) -> list[int]:
    if isinstance(row.get("prompt"), str):
        return list(tokenizer.encode(row["prompt"], add_special_tokens=False))
    messages = row.get("messages")
    if not isinstance(messages, list):
        # The locally prepared GLMFlash corpora use ``conversations`` for the
        # same OpenAI-style role/content sequence.  Accept it directly so the
        # K3 planner can reuse those authenticated source mixtures without a
        # lossy rewrite or another multi-gigabyte copy.
        messages = row.get("conversations")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "record must contain a string prompt or non-empty messages/conversations"
        )
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("messages must be JSON objects")
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )


def _source_requests(
    source: SourceSpec,
    tokenizer: TokenizerLike,
    *,
    token_quota: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    fold_modulus: int,
    fold_index: int,
    fold_mode: str,
    seed: int,
    excluded_document_hashes: set[str],
    excluded_prompt_hashes: set[str],
) -> list[PlannedRequest]:
    candidates: list[tuple[str, int, str]] = []
    seen_document_hashes = set(excluded_document_hashes)
    with source.path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            if source.record_source is not None:
                try:
                    source_row = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(source_row, dict):
                    continue
                if source_row.get("source") != source.record_source:
                    continue
            document_hash = _content_hash(raw)
            if document_hash in seen_document_hashes:
                continue
            if _fold_selected(
                document_hash,
                modulus=fold_modulus,
                index=fold_index,
                mode=fold_mode,
            ):
                candidates.append((raw, line_number, document_hash))
                # Source JSONL files can contain repeated records.  A request
                # epoch is the statistical document unit, so retain only the
                # first occurrence before shuffling and quota selection.
                seen_document_hashes.add(document_hash)
    random.Random(seed ^ int(_content_hash(str(source.path)), 16)).shuffle(candidates)

    requests = []
    seen_prompt_hashes = set(excluded_prompt_hashes)
    used = 0
    for raw, line_number, document_hash in candidates:
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                continue
            tokens = _record_tokens(row, tokenizer)
        except (TypeError, ValueError, KeyError):
            continue
        if len(tokens) < min_prompt_tokens:
            continue
        remaining = token_quota - used
        if remaining <= 0:
            break
        tokens = tokens[: min(max_prompt_tokens, remaining)]
        if not tokens:
            break
        prompt_bytes = b"".join(int(token).to_bytes(4, "little") for token in tokens)
        prompt_hash = hashlib.blake2b(prompt_bytes, digest_size=16).hexdigest()
        if prompt_hash in seen_prompt_hashes:
            continue
        seen_prompt_hashes.add(prompt_hash)
        requests.append(
            PlannedRequest(
                source=str(source.path.resolve()),
                line=line_number,
                document_hash=document_hash,
                prompt_hash=prompt_hash,
                prompt_tokens=tuple(tokens),
            )
        )
        used += len(tokens)
    if used != token_quota:
        raise ValueError(
            f"{source.path} supplied {used:,} usable tokens, expected {token_quota:,}"
        )
    return requests


def build_plan(
    sources: list[SourceSpec],
    tokenizer: TokenizerLike,
    *,
    target_tokens: int,
    max_prompt_tokens: int,
    min_prompt_tokens: int,
    fold_modulus: int,
    fold_index: int,
    fold_mode: str,
    seed: int,
    excluded_document_hashes: set[str] | None = None,
    excluded_prompt_hashes: set[str] | None = None,
) -> list[PlannedRequest]:
    excluded = set(excluded_document_hashes or ())
    excluded_prompts = set(excluded_prompt_hashes or ())
    quotas = _allocate_quotas(target_tokens, sources)
    plan = []
    for offset, (source, quota) in enumerate(zip(sources, quotas, strict=True)):
        source_plan = _source_requests(
            source,
            tokenizer,
            token_quota=quota,
            max_prompt_tokens=source.max_prompt_tokens or max_prompt_tokens,
            min_prompt_tokens=min_prompt_tokens,
            fold_modulus=fold_modulus,
            fold_index=fold_index,
            fold_mode=fold_mode,
            seed=seed + offset * 1_000_003,
            excluded_document_hashes=excluded,
            excluded_prompt_hashes=excluded_prompts,
        )
        plan.extend(source_plan)
        # Also prevent the same raw document from being selected through two
        # different source inventories.
        excluded.update(request.document_hash for request in source_plan)
        excluded_prompts.update(request.prompt_hash for request in source_plan)
    random.Random(seed).shuffle(plan)
    if sum(request.tokens for request in plan) != target_tokens:
        raise AssertionError("planned corpus does not match the requested token budget")
    return plan


def _validate_interim_directory(model_dir: Path) -> Path:
    root = model_dir.resolve()
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root} is not an interim EXL3 serve tree; missing {missing}")

    exl3_shards = [
        root / f"exl3-layer-{layer:05d}.safetensors" for layer in range(1, 93)
    ]
    missing_shards = [path.name for path in exl3_shards if not path.is_file()]
    if missing_shards:
        raise ValueError(f"{root} is missing EXL3 layer shards: {missing_shards[:3]}")
    artifact_roots = {path.resolve().parent for path in exl3_shards}
    if len(artifact_roots) != 1:
        raise ValueError(f"{root} mixes EXL3 shards from multiple artifacts")
    artifact_root = artifact_roots.pop()
    artifact_required = ("kquant_exl3_manifest.json", "allocation-exl3.json")
    artifact_missing = [
        name for name in artifact_required if not (artifact_root / name).is_file()
    ]
    if artifact_missing:
        raise ValueError(
            f"EXL3 artifact {artifact_root} is missing metadata: {artifact_missing}"
        )
    manifest = json.loads((artifact_root / "kquant_exl3_manifest.json").read_text())
    if manifest.get("kind") != "kquant_exl3_artifact":
        raise ValueError(f"{artifact_root} has an unsupported EXL3 manifest")
    return root


def _read_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def _wait_for_capture_manifest(capture_dir: Path, timeout: float) -> dict:
    path = capture_dir / "manifest.json"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2)
    raise TimeoutError(f"capture manifest did not appear: {last_error}")


def _validate_live_capture(
    capture_dir: Path,
    model_dir: Path,
    *,
    timeout: float,
    allow_complete: bool = False,
) -> dict:
    manifest = _wait_for_capture_manifest(capture_dir, timeout)
    if manifest.get("kind") != "kquant_vllm_b12x_capture":
        raise ValueError("live server does not expose a KQuant capture manifest")
    if manifest.get("source") != INTERIM_SOURCE:
        raise ValueError(
            f"capture source is {manifest.get('source')!r}, expected {INTERIM_SOURCE!r}"
        )
    teacher = Path(str(manifest.get("teacher_checkpoint", ""))).resolve()
    if teacher != model_dir:
        raise ValueError(f"capture teacher is {teacher}, expected interim {model_dir}")
    if manifest.get("complete") and not allow_complete:
        raise ValueError("capture is already finalized")
    return manifest


def _post_completion(
    base_url: str,
    *,
    model: str,
    prompt_tokens: tuple[int, ...],
    timeout: float,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": list(prompt_tokens),
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"completion request failed ({error.code}): {detail}"
        ) from error


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _plan_report(
    args: argparse.Namespace,
    plan: list[PlannedRequest],
    excluded_reports: list[dict[str, object]],
    excluded_document_hashes: set[str],
    excluded_prompt_hashes: set[str],
) -> dict:
    return {
        "kind": "kquant_interim_calibration_corpus_run",
        "schema_version": 1,
        "constraint": "interim EXL3 hybrid only; official model not loaded",
        "model_dir": str(args.model_dir.resolve()),
        "capture_dir": str(args.capture_dir.resolve()),
        "sources": [
            {
                "path": str(source.path.resolve()),
                "weight": source.weight,
                "max_prompt_tokens": (
                    source.max_prompt_tokens or args.max_prompt_tokens
                ),
                "record_source": source.record_source,
            }
            for source in args.source
        ],
        "fold": {
            "modulus": args.fold_modulus,
            "index": args.fold_index,
            "mode": args.fold_mode,
            "unit": "whole JSONL record by content hash",
        },
        "excluded_corpus_reports": excluded_reports,
        "excluded_document_hashes": sorted(excluded_document_hashes),
        "excluded_prompt_hashes": sorted(excluded_prompt_hashes),
        "seed": args.seed,
        "target_tokens": args.target_tokens,
        "planned_tokens": sum(request.tokens for request in plan),
        "planned_requests": len(plan),
        "documents": [
            {
                key: value
                for key, value in asdict(request).items()
                if key != "prompt_tokens"
            }
            | {"tokens": request.tokens}
            for request in plan
        ],
        "completed_requests": 0,
        "reported_prompt_tokens": 0,
        "reported_completion_tokens": 0,
        "finalized": False,
    }


def _resume_report(path: Path, planned: dict) -> dict:
    previous = json.loads(path.read_text())
    immutable = (
        "kind",
        "schema_version",
        "constraint",
        "model_dir",
        "capture_dir",
        "sources",
        "fold",
        "excluded_corpus_reports",
        "excluded_document_hashes",
        "excluded_prompt_hashes",
        "seed",
        "target_tokens",
        "planned_tokens",
        "planned_requests",
        "documents",
    )
    mismatched = [key for key in immutable if previous.get(key) != planned.get(key)]
    if mismatched:
        raise ValueError(f"resume report does not match this plan: {mismatched}")
    completed = previous.get("completed_requests")
    if (
        not isinstance(completed, int)
        or not 0 <= completed <= planned["planned_requests"]
    ):
        raise ValueError("resume report has an invalid completed request count")
    return previous


def run(args: argparse.Namespace) -> dict:
    model_dir = _validate_interim_directory(args.model_dir)
    for source in args.source:
        if not source.path.is_file():
            raise FileNotFoundError(source.path)
    (
        excluded_hashes,
        excluded_prompt_hashes,
        excluded_reports,
    ) = _load_excluded_documents(args.exclude_report)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    plan = build_plan(
        args.source,
        tokenizer,
        target_tokens=args.target_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        min_prompt_tokens=args.min_prompt_tokens,
        fold_modulus=args.fold_modulus,
        fold_index=args.fold_index,
        fold_mode=args.fold_mode,
        seed=args.seed,
        excluded_document_hashes=excluded_hashes,
        excluded_prompt_hashes=excluded_prompt_hashes,
    )
    planned_report = _plan_report(
        args,
        plan,
        excluded_reports,
        excluded_hashes,
        excluded_prompt_hashes,
    )
    if args.resume:
        if not args.report.is_file():
            raise FileNotFoundError(f"resume report does not exist: {args.report}")
        report = _resume_report(args.report, planned_report)
    else:
        if args.report.exists():
            raise FileExistsError(
                f"report already exists (use --resume if appropriate): {args.report}"
            )
        report = planned_report
        _write_report(args.report, report)
    if args.dry_run:
        return report

    _wait_for_server(args.base_url, args.health_timeout)
    completed_before_connect = report["completed_requests"] == len(plan)
    sentinel_recorded = "finalize_sentinel_created_before_request" in report
    live_manifest = _validate_live_capture(
        args.capture_dir,
        model_dir,
        timeout=args.health_timeout,
        allow_complete=completed_before_connect and sentinel_recorded,
    )
    report["capture_manifest_before_run"] = live_manifest
    if live_manifest.get("complete"):
        report["capture_manifest_after_run"] = live_manifest
        report["finalized"] = True
        _write_report(args.report, report)
        return report
    models = _read_json(f"{args.base_url}/v1/models", timeout=args.request_timeout)
    served_ids = {
        entry.get("id") for entry in models.get("data", []) if isinstance(entry, dict)
    }
    if args.model not in served_ids:
        raise ValueError(f"server exposes {sorted(served_ids)}, not {args.model!r}")

    start = report["completed_requests"]
    if (
        start == len(plan)
        and args.finalize_file is not None
        and "finalize_sentinel_created_before_request" not in report
    ):
        raise ValueError("all requests completed without a finalize sentinel")

    for index in range(start, len(plan)):
        planned = plan[index]
        if args.finalize_file is not None and index == len(plan) - 1:
            sentinel_recorded = (
                report.get("finalize_sentinel_created_before_request") == index
            )
            if args.finalize_file.exists() and not sentinel_recorded:
                raise FileExistsError(args.finalize_file)
            if not args.finalize_file.exists():
                args.finalize_file.parent.mkdir(parents=True, exist_ok=True)
                args.finalize_file.touch()
                report["finalize_sentinel_created_before_request"] = index
                _write_report(args.report, report)
        response = _post_completion(
            args.base_url,
            model=args.model,
            prompt_tokens=planned.prompt_tokens,
            timeout=args.request_timeout,
        )
        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        if prompt_tokens != planned.tokens:
            raise ValueError(
                f"request {index} reported {prompt_tokens} prompt tokens, "
                f"planned {planned.tokens}"
            )
        report["completed_requests"] = index + 1
        report["reported_prompt_tokens"] += prompt_tokens
        report["reported_completion_tokens"] += int(usage.get("completion_tokens", 0))
        report["last_completed_prompt_hash"] = planned.prompt_hash
        _write_report(args.report, report)

    if args.finalize_file is not None:
        deadline = time.monotonic() + args.finalize_timeout
        while time.monotonic() < deadline:
            manifest = _wait_for_capture_manifest(args.capture_dir, 5)
            if manifest.get("complete"):
                report["capture_manifest_after_run"] = manifest
                report["finalized"] = True
                break
            time.sleep(2)
        else:
            raise TimeoutError("capture did not finalize after the sentinel request")
    _write_report(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=_parse_source,
        action="append",
        required=True,
        metavar="JSONL[=WEIGHT[@MAX_TOKENS][::RECORD_SOURCE]]",
        help=(
            "JSONL source, optional mixture weight, and optional per-source "
            "prompt cap/source-field filter; for example "
            "hybrid.jsonl=0.2@4096::ultrachat"
        ),
    )
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--min-prompt-tokens", type=int, default=64)
    parser.add_argument("--fold-modulus", type=int, default=8)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--fold-mode", choices=("include", "exclude"), required=True)
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[],
        help="exclude every document named by a prior corpus report; repeatable",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="kimi-k3-exl3")
    parser.add_argument("--health-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--finalize-timeout", type=float, default=900)
    parser.add_argument("--finalize-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")
    if not 0 <= args.fold_index < args.fold_modulus:
        parser.error("--fold-index must be in [0, fold-modulus)")
    if not 0 < args.min_prompt_tokens <= args.max_prompt_tokens:
        parser.error("prompt token limits are invalid")
    return args


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
