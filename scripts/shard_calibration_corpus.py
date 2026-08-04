#!/usr/bin/env python3
"""Split a mixed calibration JSONL into provenance-preserving source shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


_SAFE_LABEL = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_label(value: str) -> str:
    result = _SAFE_LABEL.sub("-", value).strip("-.")
    if not result:
        raise ValueError(f"source label has no safe filename characters: {value!r}")
    return result


def shard_corpus(
    input_path: Path,
    output_dir: Path,
    *,
    field: str,
    labels: list[str],
) -> dict[str, object]:
    input_path = input_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if len(labels) != len(set(labels)):
        raise ValueError("source labels must be unique")
    filenames = {label: f"{_safe_label(label)}.jsonl" for label in labels}
    if len(filenames.values()) != len(set(filenames.values())):
        raise ValueError("source labels collide after filename normalization")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    final_paths = {label: output_dir / name for label, name in filenames.items()}
    for path in final_paths.values():
        if path.exists():
            raise FileExistsError(path)

    temporary_paths = {
        label: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for label, path in final_paths.items()
    }
    handles = {
        label: path.open("wb") for label, path in temporary_paths.items()
    }
    input_hash = hashlib.sha256()
    output_hashes = {label: hashlib.sha256() for label in labels}
    counts = {label: 0 for label in labels}
    byte_counts = {label: 0 for label in labels}
    invalid_records = 0
    unselected_records = 0
    total_records = 0
    try:
        with input_path.open("rb") as source:
            for raw in source:
                input_hash.update(raw)
                total_records += 1
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    invalid_records += 1
                    continue
                if not isinstance(record, dict):
                    invalid_records += 1
                    continue
                label = record.get(field)
                if label not in handles:
                    unselected_records += 1
                    continue
                handles[label].write(raw)
                output_hashes[label].update(raw)
                counts[label] += 1
                byte_counts[label] += len(raw)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        for label, temporary in temporary_paths.items():
            temporary.replace(final_paths[label])
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise

    manifest: dict[str, object] = {
        "kind": "kquant_calibration_source_shards",
        "schema_version": 1,
        "input": str(input_path),
        "input_sha256": input_hash.hexdigest(),
        "field": field,
        "total_records": total_records,
        "invalid_records": invalid_records,
        "unselected_records": unselected_records,
        "shards": {
            label: {
                "path": str(final_paths[label]),
                "records": counts[label],
                "bytes": byte_counts[label],
                "sha256": output_hashes[label].hexdigest(),
            }
            for label in labels
        },
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", default="source")
    parser.add_argument("--label", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = shard_corpus(
        args.input, args.output_dir, field=args.field, labels=args.label
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
