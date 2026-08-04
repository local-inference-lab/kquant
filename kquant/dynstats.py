"""The negotiated stats-bundle contract.

A bundle is a directory ``<name>.kqstats/`` containing:
  manifest.json        - schema_version, producer, model/revision, free-form
                         provenance (corpus, token_count, conventions)
  arrays.safetensors   - float arrays whose leading dims are
                         [NUM_MOE_LAYERS, NUM_EXPERTS] (= [92, 896]),
                         optionally with trailing dims (e.g. per-matrix [.., 3]).

kquant's own static producers (stats/distortion) emit this same container, so
rankers consume static and dynamic signals uniformly.  Schema v2 adds the
gate-square-weighted activation moments emitted by the vLLM/B12X calibration
path.  Unknown keys are preserved and ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file, save_file

from kquant import constants as C

SCHEMA_VERSION = 2


@dataclass
class StatsBundle:
    manifest: dict
    arrays: dict[str, np.ndarray]
    path: Path | None = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def keys(self):
        return self.arrays.keys()


def validate(arrays: dict[str, np.ndarray]) -> None:
    for k, a in arrays.items():
        if a.shape[:2] != (C.NUM_MOE_LAYERS, C.NUM_EXPERTS) and a.shape[:1] != (
            C.NUM_MOE_LAYERS,
        ):
            raise ValueError(
                f"array {k} has shape {a.shape}; leading dims must be "
                f"[{C.NUM_MOE_LAYERS}, {C.NUM_EXPERTS}] or [{C.NUM_MOE_LAYERS}]"
            )


def save_bundle(
    path: str | Path,
    arrays: dict[str, np.ndarray | torch.Tensor],
    producer: str,
    extra_manifest: dict | None = None,
) -> Path:
    path = Path(path)
    if not path.name.endswith(".kqstats"):
        path = path.with_name(path.name + ".kqstats")
    path.mkdir(parents=True, exist_ok=True)
    np_arrays = {
        k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        for k, v in arrays.items()
    }
    validate(np_arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "model": C.MODEL_ID,
        "revision": C.REVISION,
        "keys": sorted(np_arrays),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    arrays_path = path / "arrays.safetensors"
    arrays_tmp = path / f".{arrays_path.name}.tmp"
    manifest_path = path / "manifest.json"
    manifest_tmp = path / f".{manifest_path.name}.tmp"
    save_file(np_arrays, str(arrays_tmp))
    manifest_tmp.write_text(json.dumps(manifest, indent=1))
    arrays_tmp.replace(arrays_path)
    manifest_tmp.replace(manifest_path)
    return path


def load_bundle(path: str | Path) -> StatsBundle:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version", 0) > SCHEMA_VERSION:
        raise ValueError(f"bundle {path} has newer schema than this kquant")
    arrays = load_file(str(path / "arrays.safetensors"))
    return StatsBundle(manifest=manifest, arrays=dict(arrays), path=path)


@dataclass
class ComposedStats:
    """Union view over several bundles; later bundles win on key conflicts."""

    bundles: list[StatsBundle] = field(default_factory=list)

    def get(self, key: str) -> np.ndarray | None:
        for b in reversed(self.bundles):
            if key in b:
                return b[key]
        return None

    def require(self, key: str) -> np.ndarray:
        a = self.get(key)
        if a is None:
            have = sorted({k for b in self.bundles for k in b.keys()})
            raise KeyError(f"required stats key {key!r} not found; have {have}")
        return a
