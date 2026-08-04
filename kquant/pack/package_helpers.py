"""Shared filesystem helpers for publishing the current QSRT serve package."""

from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors import safe_open


DEFAULT_NONEXPERT = Path("/models/Kimi-K3-mxfp8-nonexpert")
IGNORED_DENSE_LAYERS = (
    "kv_b_proj",
    "g_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "vision_tower",
    "mm_projector",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _link(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve(strict=True))


def _scan_nonexpert_weight_map(source: Path) -> tuple[dict[str, str], int, list[str]]:
    files = sorted(source.glob("00-nonexpert-*.safetensors"))
    if not files:
        raise ValueError(f"no MXFP8 non-expert shards found in {source}")
    weight_map: dict[str, str] = {}
    total = 0
    names: list[str] = []
    for path in files:
        names.append(path.name)
        total += path.stat().st_size
        with safe_open(path, framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                if ".block_sparse_moe.experts." in tensor_name:
                    raise ValueError(
                        f"non-expert overlay contains routed expert {tensor_name}"
                    )
                if tensor_name in weight_map:
                    raise ValueError(
                        f"non-expert tensor appears in multiple shards: {tensor_name}"
                    )
                weight_map[tensor_name] = path.name
    if not weight_map:
        raise ValueError("MXFP8 non-expert overlay contains no tensors")
    return weight_map, total, names


def _nonexpert_weight_map(
    source: Path, destination: Path
) -> tuple[dict[str, str], int, list[str]]:
    weight_map, total, names = _scan_nonexpert_weight_map(source)
    for name in names:
        _link(source / name, destination / name)
    return weight_map, total, names


def _auxiliary_sources(snapshot: Path) -> list[Path]:
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        "hf_quant_config.json",
        "quant_config.json",
    }
    return [
        source
        for source in sorted(snapshot.iterdir())
        if source.name not in excluded and source.suffix != ".safetensors"
    ]


def _require_exact_link(link: Path, source: Path) -> None:
    if not link.is_symlink():
        raise ValueError(f"serve package entry is not a symlink: {link}")
    if link.resolve(strict=True) != source.resolve(strict=True):
        raise ValueError(f"serve package link {link.name} has the wrong target")
