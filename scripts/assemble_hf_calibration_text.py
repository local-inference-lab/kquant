#!/usr/bin/env python3
"""Stream a bounded, revision-pinned Hugging Face text calibration shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


def _parse_minimum(value: str) -> tuple[str, float]:
    field, separator, threshold = value.rpartition("=")
    if not separator or not field:
        raise argparse.ArgumentTypeError("minimum must be FIELD=VALUE")
    try:
        return field, float(threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("minimum value must be numeric") from exc


def _eligible(record: dict, minima: Iterable[tuple[str, float]]) -> bool:
    for field, threshold in minima:
        value = record.get(field)
        if not isinstance(value, (int, float)) or float(value) < threshold:
            return False
    return True


def _output_record(
    record: dict,
    *,
    text_field: str,
    max_chars: int,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    provenance_fields: list[str],
) -> tuple[dict, str] | None:
    text = record.get(text_field)
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    text = text[:max_chars]
    content_hash = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
    provenance = {
        "dataset": dataset,
        "revision": revision,
        "config": config,
        "split": split,
        "content_hash": content_hash,
    }
    for field in provenance_fields:
        value = record.get(field)
        if value is None or isinstance(value, (str, int, float, bool)):
            provenance[field] = value
    return {"prompt": text, "provenance": provenance}, content_hash


def assemble(args: argparse.Namespace) -> dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the Hugging Face datasets package is required for source assembly"
        ) from exc

    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output.exists():
        raise FileExistsError(output)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        revision=args.revision,
    )
    if args.shuffle_buffer > 1:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    output_hash = hashlib.sha256()
    seen_hashes: set[str] = set()
    scanned = 0
    selected = 0
    selected_chars = 0
    rejected_minimum = 0
    rejected_text = 0
    rejected_duplicate = 0
    try:
        with temporary.open("wb") as handle:
            for record in dataset:
                scanned += 1
                if not isinstance(record, dict):
                    rejected_text += 1
                    continue
                if not _eligible(record, args.minimum):
                    rejected_minimum += 1
                    continue
                built = _output_record(
                    record,
                    text_field=args.text_field,
                    max_chars=args.max_chars,
                    dataset=args.dataset,
                    revision=args.revision,
                    config=args.config,
                    split=args.split,
                    provenance_fields=args.provenance_field,
                )
                if built is None:
                    rejected_text += 1
                    continue
                output_record, content_hash = built
                if len(output_record["prompt"]) < args.min_chars:
                    rejected_text += 1
                    continue
                if content_hash in seen_hashes:
                    rejected_duplicate += 1
                    continue
                seen_hashes.add(content_hash)
                encoded = (
                    json.dumps(output_record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode()
                handle.write(encoded)
                output_hash.update(encoded)
                selected += 1
                selected_chars += len(output_record["prompt"])
                if selected >= args.target_documents:
                    break
            if selected != args.target_documents:
                raise ValueError(
                    f"source yielded {selected} documents, expected "
                    f"{args.target_documents}"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest: dict[str, object] = {
        "kind": "kquant_hf_calibration_text_shard",
        "schema_version": 1,
        "dataset": args.dataset,
        "revision": args.revision,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "minimum": dict(args.minimum),
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "scanned_documents": scanned,
        "selected_documents": selected,
        "selected_chars": selected_chars,
        "rejected_minimum": rejected_minimum,
        "rejected_text": rejected_text,
        "rejected_duplicate": rejected_duplicate,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": output_hash.hexdigest(),
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--target-documents", type=int, required=True)
    parser.add_argument("--min-chars", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=65536)
    parser.add_argument("--minimum", type=_parse_minimum, action="append", default=[])
    parser.add_argument(
        "--provenance-field",
        action="append",
        default=[],
        help="copy a scalar upstream identity/metadata field; repeatable",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.target_documents <= 0:
        parser.error("target documents must be positive")
    if not 0 < args.min_chars <= args.max_chars:
        parser.error("character limits are invalid")
    if args.shuffle_buffer <= 0:
        parser.error("shuffle buffer must be positive")
    return args


def main() -> None:
    manifest = assemble(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
