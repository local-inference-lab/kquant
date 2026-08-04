"""Atomic writer for calibration-statistics bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from kquant import constants as C


SCHEMA_VERSION = 2


def _validate(arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        if value.shape[:2] != (C.NUM_MOE_LAYERS, C.NUM_EXPERTS) and value.shape[
            :1
        ] != (C.NUM_MOE_LAYERS,):
            raise ValueError(
                f"array {name} has shape {value.shape}; leading dimensions must "
                f"be [{C.NUM_MOE_LAYERS}, {C.NUM_EXPERTS}] or "
                f"[{C.NUM_MOE_LAYERS}]"
            )


def save_stats_bundle(
    path: str | Path,
    arrays: dict[str, np.ndarray | torch.Tensor],
    producer: str,
    extra_manifest: dict | None = None,
) -> Path:
    """Write tensors and provenance atomically to one ``.kqstats`` directory."""

    path = Path(path)
    if not path.name.endswith(".kqstats"):
        path = path.with_name(path.name + ".kqstats")
    path.mkdir(parents=True, exist_ok=True)
    values = {
        name: (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        for name, value in arrays.items()
    }
    _validate(values)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "model": C.MODEL_ID,
        "revision": C.REVISION,
        "keys": sorted(values),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    arrays_path = path / "arrays.safetensors"
    arrays_tmp = path / f".{arrays_path.name}.tmp"
    manifest_path = path / "manifest.json"
    manifest_tmp = path / f".{manifest_path.name}.tmp"
    save_file(values, str(arrays_tmp))
    manifest_tmp.write_text(json.dumps(manifest, indent=1))
    arrays_tmp.replace(arrays_path)
    manifest_tmp.replace(manifest_path)
    return path
