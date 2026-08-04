"""Exact input-suite contract for the Kimi-K3 interim-teacher proxy study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


TEACHER_PROXY_SUITE_SCHEMA_VERSION = 1
TEACHER_PROXY_SUITE_KIND = "kquant_teacher_proxy_input_suite"
CALIBRATION_CORPUS_KIND = "kquant_interim_calibration_corpus_run"


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


@dataclass(frozen=True)
class LoadedTeacherProxySuite:
    path: Path
    file_sha256: str
    manifest: dict[str, Any]
    input_ids: torch.Tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(values: list[int] | tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("token IDs must be non-negative integers")
        if value >= 2**32:
            raise ValueError("token ID does not fit the canonical uint32 encoding")
        digest.update(value.to_bytes(4, "little"))
    return digest.hexdigest()


def suite_token_hash(document_hashes: list[str]) -> str:
    if not document_hashes:
        raise ValueError("teacher-proxy suite must contain documents")
    return hashlib.sha256("\n".join(document_hashes).encode("ascii")).hexdigest()


def parse_layer_list(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("layers must be comma-separated integers") from exc
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("layers must be a nonempty list of non-negative integers")
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicates")
    return tuple(sorted(layers))


def _record_tokens(row: dict[str, Any], tokenizer: TokenizerLike) -> list[int]:
    if isinstance(row.get("prompt"), str):
        return list(tokenizer.encode(row["prompt"], add_special_tokens=False))
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("record must contain a string prompt or non-empty messages")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("messages must be JSON objects")
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )


def _content_hash(raw: str) -> str:
    return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()


def _prompt_hash(tokens: list[int]) -> str:
    payload = b"".join(int(token).to_bytes(4, "little") for token in tokens)
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _read_source_lines(
    requested: dict[Path, set[int]],
) -> dict[tuple[Path, int], str]:
    result: dict[tuple[Path, int], str] = {}
    for source, line_numbers in requested.items():
        with source.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if line_number in line_numbers:
                    value = raw.strip()
                    if not value:
                        raise ValueError(
                            f"{source}:{line_number}: selected line is empty"
                        )
                    result[(source, line_number)] = value
        missing = sorted(
            line for line in line_numbers if (source, line) not in result
        )
        if missing:
            raise ValueError(f"{source}: selected line numbers are missing: {missing}")
    return result


def _weighted_counts(total: int, sources: list[dict[str, Any]]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("document count must be positive")
    if not sources:
        raise ValueError("corpus report has no sources")
    weights: list[float] = []
    names: list[str] = []
    for source in sources:
        name = str(Path(str(source.get("path", ""))).resolve())
        weight = source.get("weight")
        if not name or not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("corpus source path/weight is invalid")
        names.append(name)
        weights.append(float(weight))
    weight_sum = sum(weights)
    raw = [total * weight / weight_sum for weight in weights]
    counts = [int(value) for value in raw]
    order = sorted(
        range(len(names)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return dict(zip(names, counts, strict=True))


def _tokenizer_fingerprint(root: Path) -> dict[str, str]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    )
    files = {
        name: file_sha256(root / name)
        for name in names
        if (root / name).is_file()
    }
    if not files:
        raise ValueError(f"{root}: no tokenizer files found")
    return files


def build_teacher_proxy_suite(
    *,
    corpus_report_path: Path,
    tokenizer_root: Path,
    tokenizer: TokenizerLike,
    document_count: int,
    tokens_per_document: int,
    trace_layers: tuple[int, ...],
) -> dict[str, Any]:
    """Reconstruct and bind a source-stratified subset of a captured corpus."""
    corpus_report_path = corpus_report_path.resolve()
    tokenizer_root = tokenizer_root.resolve()
    report = json.loads(corpus_report_path.read_text())
    if report.get("kind") != CALIBRATION_CORPUS_KIND:
        raise ValueError(f"{corpus_report_path}: unsupported corpus report kind")
    if report.get("schema_version") != 1:
        raise ValueError(f"{corpus_report_path}: unsupported corpus schema")
    if report.get("completed_requests") != report.get("planned_requests"):
        raise ValueError("corpus report is incomplete")
    if report.get("reported_prompt_tokens") != report.get("planned_tokens"):
        raise ValueError("corpus report token accounting is incomplete")
    if not report.get("finalized"):
        raise ValueError("corpus capture is not finalized")
    if tokens_per_document <= 0:
        raise ValueError("tokens per document must be positive")
    if not trace_layers or tuple(sorted(set(trace_layers))) != trace_layers:
        raise ValueError("trace layers must be sorted and unique")

    sources = report.get("sources")
    documents = report.get("documents")
    if not isinstance(sources, list) or not isinstance(documents, list):
        raise ValueError("corpus report sources/documents are malformed")
    quotas = _weighted_counts(document_count, sources)
    eligible: dict[str, list[tuple[int, dict[str, Any]]]] = {
        source: [] for source in quotas
    }
    for report_index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError("corpus document descriptor is malformed")
        source = str(Path(str(document.get("source", ""))).resolve())
        tokens = document.get("tokens")
        if (
            source in eligible
            and isinstance(tokens, int)
            and tokens >= tokens_per_document
        ):
            eligible[source].append((report_index, document))

    chosen: list[tuple[int, dict[str, Any]]] = []
    for source, quota in quotas.items():
        if len(eligible[source]) < quota:
            raise ValueError(
                f"source {source} has {len(eligible[source])} eligible documents, "
                f"requires {quota}"
            )
        chosen.extend(eligible[source][:quota])
    chosen.sort(key=lambda item: item[0])
    if len(chosen) != document_count:
        raise AssertionError("source-stratified selection count does not close")

    requested: dict[Path, set[int]] = {}
    for unused_index, document in chosen:
        source = Path(str(document["source"])).resolve()
        line = int(document["line"])
        requested.setdefault(source, set()).add(line)
    source_lines = _read_source_lines(requested)

    suite_documents: list[dict[str, Any]] = []
    token_hashes: list[str] = []
    for suite_index, (report_index, descriptor) in enumerate(chosen):
        source = Path(str(descriptor["source"])).resolve()
        line = int(descriptor["line"])
        raw = source_lines[(source, line)]
        if _content_hash(raw) != descriptor.get("document_hash"):
            raise ValueError(f"{source}:{line}: document hash differs from report")
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line}: source row is not an object")
        planned_tokens = _record_tokens(row, tokenizer)[: int(descriptor["tokens"])]
        if len(planned_tokens) != int(descriptor["tokens"]):
            raise ValueError(f"{source}:{line}: reconstructed token count differs")
        if _prompt_hash(planned_tokens) != descriptor.get("prompt_hash"):
            raise ValueError(f"{source}:{line}: reconstructed prompt hash differs")
        input_ids = planned_tokens[:tokens_per_document]
        token_hash = token_ids_sha256(input_ids)
        token_hashes.append(token_hash)
        suite_documents.append(
            {
                "suite_index": suite_index,
                "corpus_report_index": report_index,
                "source": str(source),
                "line": line,
                "document_hash": descriptor["document_hash"],
                "captured_prompt_hash": descriptor["prompt_hash"],
                "captured_prompt_tokens": int(descriptor["tokens"]),
                "input_tokens": len(input_ids),
                "input_ids_sha256": token_hash,
                "input_ids": input_ids,
            }
        )

    return {
        "kind": TEACHER_PROXY_SUITE_KIND,
        "schema_version": TEACHER_PROXY_SUITE_SCHEMA_VERSION,
        "purpose": (
            "document-clustered official-source versus interim-3p09 teacher "
            "proxy validation; no source text is embedded"
        ),
        "corpus_report": {
            "path": str(corpus_report_path),
            "sha256": file_sha256(corpus_report_path),
            "kind": report["kind"],
            "fold": report.get("fold"),
            "seed": report.get("seed"),
        },
        "tokenizer": {
            "root": str(tokenizer_root),
            "files_sha256": _tokenizer_fingerprint(tokenizer_root),
        },
        "selection": {
            "method": (
                "first eligible documents in captured report order, "
                "stratified by source weight"
            ),
            "document_count": document_count,
            "tokens_per_document": tokens_per_document,
            "source_document_counts": quotas,
        },
        "batch_shape": [document_count, tokens_per_document],
        "trace_layers": list(trace_layers),
        "end_layer": max(trace_layers) + 1,
        "suite_token_hash_sha256": suite_token_hash(token_hashes),
        "documents": suite_documents,
    }


def load_teacher_proxy_suite(path: Path) -> LoadedTeacherProxySuite:
    path = path.resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("kind") != TEACHER_PROXY_SUITE_KIND:
        raise ValueError(f"{path}: unsupported teacher-proxy suite kind")
    if manifest.get("schema_version") != TEACHER_PROXY_SUITE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported teacher-proxy suite schema")
    corpus_report = manifest.get("corpus_report")
    if not isinstance(corpus_report, dict):
        raise ValueError(f"{path}: corpus provenance is missing")
    corpus_path = Path(str(corpus_report.get("path", ""))).resolve()
    if not corpus_path.is_file() or file_sha256(corpus_path) != corpus_report.get(
        "sha256"
    ):
        raise ValueError(f"{path}: corpus report provenance differs")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError(f"{path}: tokenizer provenance is missing")
    tokenizer_root = Path(str(tokenizer.get("root", ""))).resolve()
    tokenizer_files = tokenizer.get("files_sha256")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise ValueError(f"{path}: tokenizer file hashes are missing")
    for name, expected_hash in tokenizer_files.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"{path}: tokenizer filename is unsafe")
        tokenizer_path = tokenizer_root / name
        if not tokenizer_path.is_file() or file_sha256(tokenizer_path) != expected_hash:
            raise ValueError(f"{path}: tokenizer file {name} differs")
    shape = manifest.get("batch_shape")
    documents = manifest.get("documents")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in shape
        )
        or not isinstance(documents, list)
        or len(documents) != shape[0]
    ):
        raise ValueError(f"{path}: invalid batch shape/documents")
    rows: list[list[int]] = []
    token_hashes: list[str] = []
    document_hashes: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict) or document.get("suite_index") != index:
            raise ValueError(f"{path}: document {index} has invalid identity")
        values = document.get("input_ids")
        if (
            not isinstance(values, list)
            or len(values) != shape[1]
            or document.get("input_tokens") != shape[1]
        ):
            raise ValueError(f"{path}: document {index} token count differs")
        actual_hash = token_ids_sha256(values)
        if actual_hash != document.get("input_ids_sha256"):
            raise ValueError(f"{path}: document {index} token hash differs")
        document_hash = document.get("document_hash")
        if not isinstance(document_hash, str) or not document_hash:
            raise ValueError(f"{path}: document {index} source hash is invalid")
        if document_hash in document_hashes:
            raise ValueError(f"{path}: duplicate source document {document_hash}")
        document_hashes.add(document_hash)
        rows.append(values)
        token_hashes.append(actual_hash)
    if suite_token_hash(token_hashes) != manifest.get("suite_token_hash_sha256"):
        raise ValueError(f"{path}: suite token hash differs")
    trace_layers = manifest.get("trace_layers")
    if (
        not isinstance(trace_layers, list)
        or not trace_layers
        or any(
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer < 0
            for layer in trace_layers
        )
        or trace_layers != sorted(set(trace_layers))
        or manifest.get("end_layer") != max(trace_layers) + 1
    ):
        raise ValueError(f"{path}: trace layer contract is invalid")
    return LoadedTeacherProxySuite(
        path=path,
        file_sha256=file_sha256(path),
        manifest=manifest,
        input_ids=torch.tensor(rows, dtype=torch.long),
    )
