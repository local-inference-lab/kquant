"""Resumable exact-byte index for the QSRT X4T endpoint."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kquant import constants as C
from kquant.source_weights import OfficialMXFP4Store
from kquant.x4t import (
    X4T_DATA_OFFSET,
    X4T_EXPERTS_PER_LAYER,
    X4T_MATRIX_ORDER,
    X4T_RECORD_ALIGNMENT,
    X4T_VERSION,
    x4t_matrix_storage_bytes,
)


X4T_COST_INDEX_KIND = "kquant_kimi_k3_qsrt_x4t_cost_index"
X4T_COST_INDEX_SCHEMA_VERSION = 1
X4T_COST_LAYER_KIND = "kquant_kimi_k3_qsrt_x4t_cost_layer"
X4T_COST_COMPLETION_KIND = "kquant_kimi_k3_qsrt_x4t_cost_completion"
X4T_COST_MANIFEST_FILENAME = "x4t-cost-manifest.json"
X4T_COST_COMPLETION_FILENAME = "x4t-cost-completion.json"


def x4t_cost_layer_filename(layer: int) -> str:
    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T cost layer must be a Kimi-K3 MoE layer")
    return f"x4t-cost-layer-{layer:05d}.json"


def _canonical_json(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=1, sort_keys=True))
    temporary.replace(path)


def x4t_cost_manifest(store: OfficialMXFP4Store) -> dict:
    return {
        "kind": X4T_COST_INDEX_KIND,
        "schema_version": X4T_COST_INDEX_SCHEMA_VERSION,
        "source_root": str(store.root),
        "source_revision": store.revision,
        "codec": {
            "name": "X4T",
            "version": X4T_VERSION,
            "matrix_order": list(X4T_MATRIX_ORDER),
            "scale_codec": "x4t-adjacent-pair-fixed-stream-v1",
            "scale_tile_rows": 16,
            "record_alignment_bytes": X4T_RECORD_ALIGNMENT,
            "layer_fixed_bytes": X4T_DATA_OFFSET,
            "exact_mxfp4_reconstruction": True,
        },
        "layers": list(C.MOE_LAYERS),
        "experts_per_layer": X4T_EXPERTS_PER_LAYER,
    }


def prepare_x4t_cost_index(
    destination: str | Path,
    *,
    store: OfficialMXFP4Store,
    resume: bool,
) -> Path:
    destination = Path(destination)
    manifest = x4t_cost_manifest(store)
    path = destination / X4T_COST_MANIFEST_FILENAME
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        if not path.is_file() or json.loads(path.read_text()) != manifest:
            raise ValueError("existing X4T cost index has a different build contract")
    else:
        destination.mkdir(parents=True)
        _atomic_json(path, manifest)
    return destination


def build_x4t_cost_layer(
    store: OfficialMXFP4Store,
    layer: int,
) -> dict:
    """Scan only scale tensors and return exact aligned X4T record costs."""

    if layer not in C.MOE_LAYERS:
        raise ValueError("X4T cost layer must be a Kimi-K3 MoE layer")
    costs = np.empty(
        (X4T_EXPERTS_PER_LAYER, len(X4T_MATRIX_ORDER)), dtype=np.int64
    )
    seen = np.zeros_like(costs, dtype=np.bool_)
    matrix_ids = {matrix: index for index, matrix in enumerate(X4T_MATRIX_ORDER)}
    for expert, matrix, scale in store.iter_layer_scale_planes(
        layer, matrices=X4T_MATRIX_ORDER
    ):
        matrix_id = matrix_ids[matrix]
        if seen[expert, matrix_id]:
            raise ValueError(
                f"duplicate X4T cost source for layer {layer} expert {expert} {matrix}"
            )
        costs[expert, matrix_id] = x4t_matrix_storage_bytes(matrix, scale)
        seen[expert, matrix_id] = True
    if not bool(seen.all()):
        missing = np.argwhere(~seen)[0]
        raise ValueError(
            "X4T cost source is incomplete at layer "
            f"{layer} expert {int(missing[0])} "
            f"{X4T_MATRIX_ORDER[int(missing[1])]}"
        )
    matrix_bytes = costs.tolist()
    expert_bytes = costs.sum(axis=1).tolist()
    document = {
        "kind": X4T_COST_LAYER_KIND,
        "schema_version": X4T_COST_INDEX_SCHEMA_VERSION,
        "source_revision": store.revision,
        "layer": layer,
        "matrix_order": list(X4T_MATRIX_ORDER),
        "matrix_storage_bytes": matrix_bytes,
        "expert_storage_bytes": expert_bytes,
        "all_x4t_sidecar_bytes": X4T_DATA_OFFSET + sum(expert_bytes),
    }
    validate_x4t_cost_layer(document, source_revision=store.revision)
    return document


def validate_x4t_cost_layer(
    document: dict,
    *,
    source_revision: str,
) -> tuple[np.ndarray, np.ndarray]:
    if document.get("kind") != X4T_COST_LAYER_KIND:
        raise ValueError("X4T cost layer kind is invalid")
    if document.get("schema_version") != X4T_COST_INDEX_SCHEMA_VERSION:
        raise ValueError("X4T cost layer schema is unsupported")
    if document.get("source_revision") != source_revision:
        raise ValueError("X4T cost layer source revision disagrees with its index")
    layer = document.get("layer")
    if isinstance(layer, bool) or layer not in C.MOE_LAYERS:
        raise ValueError("X4T cost layer ID is invalid")
    if document.get("matrix_order") != list(X4T_MATRIX_ORDER):
        raise ValueError("X4T cost matrix order is invalid")
    matrix = np.asarray(document.get("matrix_storage_bytes"), dtype=np.int64)
    expert = np.asarray(document.get("expert_storage_bytes"), dtype=np.int64)
    if matrix.shape != (X4T_EXPERTS_PER_LAYER, len(X4T_MATRIX_ORDER)):
        raise ValueError("X4T matrix cost array has the wrong shape")
    if expert.shape != (X4T_EXPERTS_PER_LAYER,):
        raise ValueError("X4T expert cost array has the wrong shape")
    if np.any(matrix <= 0) or np.any(matrix % X4T_RECORD_ALIGNMENT):
        raise ValueError("X4T matrix costs must be positive aligned byte counts")
    if not np.array_equal(expert, matrix.sum(axis=1)):
        raise ValueError("X4T expert costs do not close against matrix costs")
    if document.get("all_x4t_sidecar_bytes") != X4T_DATA_OFFSET + int(expert.sum()):
        raise ValueError("X4T layer sidecar byte accounting does not close")
    return matrix, expert


def write_x4t_cost_layer(destination: str | Path, document: dict) -> Path:
    path = Path(destination) / x4t_cost_layer_filename(int(document["layer"]))
    if path.exists():
        raise FileExistsError(path)
    _atomic_json(path, document)
    return path


@dataclass(frozen=True)
class X4TCostIndex:
    root: Path
    manifest: dict
    matrix_storage_bytes: np.ndarray
    expert_storage_bytes: np.ndarray
    content_sha256: str

    def __post_init__(self) -> None:
        if self.matrix_storage_bytes.shape != (
            C.NUM_MOE_LAYERS,
            C.NUM_EXPERTS,
            len(X4T_MATRIX_ORDER),
        ):
            raise ValueError("X4T matrix cost index shape is invalid")
        if self.expert_storage_bytes.shape != (
            C.NUM_MOE_LAYERS,
            C.NUM_EXPERTS,
        ):
            raise ValueError("X4T expert cost index shape is invalid")


def finalize_x4t_cost_index(destination: str | Path) -> dict:
    destination = Path(destination)
    manifest = json.loads((destination / X4T_COST_MANIFEST_FILENAME).read_text())
    if manifest.get("kind") != X4T_COST_INDEX_KIND:
        raise ValueError("X4T cost manifest kind is invalid")
    layer_hashes: dict[str, str] = {}
    total = 0
    for layer in C.MOE_LAYERS:
        path = destination / x4t_cost_layer_filename(layer)
        document = json.loads(path.read_text())
        _, expert = validate_x4t_cost_layer(
            document,
            source_revision=str(manifest["source_revision"]),
        )
        raw = _canonical_json(document)
        layer_hashes[str(layer)] = hashlib.sha256(raw).hexdigest()
        total += X4T_DATA_OFFSET + int(expert.sum())
    digest = hashlib.sha256(_canonical_json(layer_hashes)).hexdigest()
    completion = {
        "kind": X4T_COST_COMPLETION_KIND,
        "schema_version": X4T_COST_INDEX_SCHEMA_VERSION,
        "source_revision": manifest["source_revision"],
        "layers": layer_hashes,
        "all_x4t_container_bytes": total,
        "content_sha256": digest,
    }
    _atomic_json(destination / X4T_COST_COMPLETION_FILENAME, completion)
    return completion


def load_x4t_cost_index(destination: str | Path) -> X4TCostIndex:
    destination = Path(destination).resolve()
    manifest = json.loads((destination / X4T_COST_MANIFEST_FILENAME).read_text())
    completion = json.loads((destination / X4T_COST_COMPLETION_FILENAME).read_text())
    if (
        manifest.get("kind") != X4T_COST_INDEX_KIND
        or manifest.get("schema_version") != X4T_COST_INDEX_SCHEMA_VERSION
    ):
        raise ValueError("X4T cost manifest is unsupported")
    if (
        completion.get("kind") != X4T_COST_COMPLETION_KIND
        or completion.get("schema_version") != X4T_COST_INDEX_SCHEMA_VERSION
        or completion.get("source_revision") != manifest.get("source_revision")
    ):
        raise ValueError("X4T cost completion is unsupported or mismatched")
    matrices = np.empty(
        (C.NUM_MOE_LAYERS, C.NUM_EXPERTS, len(X4T_MATRIX_ORDER)),
        dtype=np.int64,
    )
    experts = np.empty((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.int64)
    layer_hashes: dict[str, str] = {}
    total = 0
    for row, layer in enumerate(C.MOE_LAYERS):
        document = json.loads(
            (destination / x4t_cost_layer_filename(layer)).read_text()
        )
        matrix, expert = validate_x4t_cost_layer(
            document,
            source_revision=str(manifest["source_revision"]),
        )
        if int(document["layer"]) != layer:
            raise ValueError("X4T cost layer filename and payload disagree")
        matrices[row] = matrix
        experts[row] = expert
        layer_hashes[str(layer)] = hashlib.sha256(
            _canonical_json(document)
        ).hexdigest()
        total += X4T_DATA_OFFSET + int(expert.sum())
    digest = hashlib.sha256(_canonical_json(layer_hashes)).hexdigest()
    if completion.get("layers") != layer_hashes or completion.get("content_sha256") != digest:
        raise ValueError("X4T cost index content digest does not close")
    if completion.get("all_x4t_container_bytes") != total:
        raise ValueError("X4T cost index aggregate bytes do not close")
    return X4TCostIndex(
        root=destination,
        manifest=manifest,
        matrix_storage_bytes=matrices,
        expert_storage_bytes=experts,
        content_sha256=digest,
    )
