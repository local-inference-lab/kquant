"""Reference validation and workspace helpers for Kimi-K3 KLD captures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from kquant.correctness import sha256_file
from kquant.kld_gate import (
    CANONICAL_REFERENCE_MANIFEST_SHA256,
    CANONICAL_SUITE_MANIFEST_SHA256,
    CANONICAL_SUITE_TOKEN_HASH,
    CANONICAL_TOOL_SHA256,
    SuiteWindow,
    validate_logits_manifest,
    validate_suite_manifest,
    validate_token_ids,
)


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_payload_path(directory: Path, record: Mapping[str, Any]) -> Path:
    filename = record.get("file")
    if not isinstance(filename, str) or not filename:
        raise ValueError("KLD manifest payload filename must be nonempty")
    relative = Path(filename)
    if relative.name != filename or relative.is_absolute():
        raise ValueError(f"unsafe KLD payload filename: {filename!r}")
    return directory / relative


def validate_reference_root(
    reference_root: Path,
    *,
    rehash_payloads: bool = False,
) -> dict[str, Any]:
    """Validate the pinned suite, tools, token files, and reference payloads."""

    root = reference_root.resolve()
    suite_path = root / "suite-manifest.json"
    reference_dir = root / "ref"
    reference_manifest_path = reference_dir / "manifest.json"
    if sha256_file(suite_path) != CANONICAL_SUITE_MANIFEST_SHA256:
        raise ValueError("reference suite manifest is not the pinned canonical file")
    if sha256_file(reference_manifest_path) != CANONICAL_REFERENCE_MANIFEST_SHA256:
        raise ValueError("reference logit manifest is not the pinned canonical file")
    for filename, expected_hash in CANONICAL_TOOL_SHA256.items():
        path = root / "tools" / filename
        if sha256_file(path) != expected_hash:
            raise ValueError(f"reference tool is not canonical: {filename}")

    suite = load_json_object(suite_path)
    suite_hash, windows = validate_suite_manifest(
        suite, expected_suite_hash=CANONICAL_SUITE_TOKEN_HASH
    )
    for window in windows:
        token_path = root / window.token_file
        validate_token_ids(
            json.loads(token_path.read_text(encoding="utf-8")), window=window
        )

    manifest = load_json_object(reference_manifest_path)
    records = validate_logits_manifest(
        manifest,
        label="reference",
        suite_hash=suite_hash,
        expected_windows=windows,
    )
    total_size = 0
    for record in records:
        path = _safe_payload_path(reference_dir, record)
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(record["size_bytes"])
        if path.stat().st_size != expected_size:
            raise ValueError(f"reference logit size differs for {path.name}")
        if rehash_payloads and sha256_file(path) != record["sha256"]:
            raise ValueError(f"reference logit SHA-256 differs for {path.name}")
        total_size += expected_size
    return {
        "root": str(root),
        "suite_hash_sha256": suite_hash,
        "windows": len(windows),
        "total_size_bytes": total_size,
        "payloads_rehashed": rehash_payloads,
    }


def prepare_suite_workspace(reference_root: Path, workspace: Path) -> Path:
    """Create a writable suite view without mutating the canonical dataset."""

    root = reference_root.resolve()
    destination = workspace.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("suite-manifest.json", "tokens"):
        source = root / name
        if not source.exists():
            raise FileNotFoundError(source)
        (destination / name).symlink_to(source, target_is_directory=source.is_dir())
    return destination


def validate_finalized_manifest(
    manifest_path: Path,
    *,
    suite_hash: str,
    windows: tuple[SuiteWindow, ...],
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    manifest = load_json_object(manifest_path)
    if not isinstance(manifest.get("runtime"), dict):
        raise ValueError(f"{label} manifest lacks serving runtime provenance")
    return validate_logits_manifest(
        manifest,
        label=label,
        suite_hash=suite_hash,
        expected_windows=windows,
    )
