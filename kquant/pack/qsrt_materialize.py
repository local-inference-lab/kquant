"""Authenticate and materialize the TP-independent QSRT artifact.

Every MoE layer is represented by a canonical balanced-atom QSRT file and an
exact X4T file. Tensor parallelism is a serving-time view over those files;
neither the build contract nor the serialized layer names a TP size.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from kquant.pack.qsrt_allocation import (
    HIGH_TIER_STORAGE,
    QSRT_ALLOCATION_KIND,
    QSRT_ALLOCATION_SCHEMA_VERSION,
    choose_qsrt_lagrangian,
    qsrt_total_container_bytes,
    qsrt_trellis_layer_bytes,
)
from kquant.pack.qsrt_atoms import (
    QSRTAtomLayerReader,
    QSRTAtomLayerSpec,
    assemble_candidate_atoms,
    candidate_layer_path,
    layer_filename,
)
from kquant.pack.qsrt_candidates import candidate_tensor_name
from kquant.pack.qsrt_pool import QSRTCandidatePool
from kquant.pack.qsrt_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
    load_qsrt_validation_scores,
)
from kquant.pack.x4t_index import (
    X4T_COST_COMPLETION_FILENAME,
    X4T_COST_MANIFEST_FILENAME,
    X4TCostIndex,
)
from kquant.qsrt import (
    EXPERTS_PER_LAYER,
    FORMAT_X4T,
    PHASE1_MODE_IDS,
    ExpertFormatSpec,
)
from kquant.qsrt_storage import QSRTLayerLayout
from kquant.source_weights import OfficialMXFP4Store
from kquant.x4t import (
    X4T_LAYER_FIXED_BYTES,
    X4T_MATRIX_ORDER,
    X4TLayerReader,
    x4t_layer_path,
)


QSRT_ARTIFACT_KIND = "kquant_kimi_k3_qsrt_artifact"
QSRT_ARTIFACT_SCHEMA_VERSION = 2
QSRT_BUILD_KIND = "kquant_kimi_k3_qsrt_materialization_request"
QSRT_BUILD_SCHEMA_VERSION = 2
QSRT_BUILD_FILENAME = "qsrt-build.json"
QSRT_MANIFEST_FILENAME = "qsrt-manifest.json"
QSRT_ALLOCATION_COPY_FILENAME = "allocation-qsrt.json"
QSRT_CANDIDATE_MANIFEST_COPY_FILENAME = "qsrt-candidate-manifest.json"
QSRT_CANDIDATE_COMPLETION_COPY_FILENAME = "qsrt-candidate-completion.json"
QSRT_X4T_MANIFEST_COPY_FILENAME = "qsrt-x4t-cost-manifest.json"
QSRT_X4T_COMPLETION_COPY_FILENAME = "qsrt-x4t-cost-completion.json"
QSRT_CLOSURE_KIND = "kquant_kimi_k3_qsrt_layer_closure"
QSRT_CLOSURE_SCHEMA_VERSION = 2
QSRT_CLOSURE_SUFFIX = ".closure.json"


@dataclass(frozen=True)
class QSRTMaterializationPlan:
    layers: tuple[QSRTAtomLayerSpec, ...]
    x4t_layer_bytes: tuple[int, ...]
    qsrt_atom_bytes: int
    x4t_bytes: int
    total_container_bytes: int
    x4t_experts: int
    compressed_experts: int

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.x4t_layer_bytes):
            raise ValueError("QSRT layer specs and X4T byte ledger disagree")

    def x4t_bytes_for_layer(self, layer: int) -> int:
        for index, spec in enumerate(self.layers):
            if spec.layer == layer:
                return self.x4t_layer_bytes[index]
        raise ValueError(f"QSRT plan has no layer {layer}")


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, document: dict) -> None:
    _atomic_bytes(
        path,
        (json.dumps(document, indent=1, sort_keys=True) + "\n").encode(),
    )


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "file": path.name,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _allocation_damage(meta: dict, pool: QSRTCandidatePool) -> np.ndarray:
    metric = meta.get("damage_metric")
    weighting = meta.get("damage_weighting")
    provenance = meta.get("damage_provenance")
    if metric == pool.damage_metric:
        if weighting != pool.damage_weighting or provenance != pool.damage_provenance:
            raise ValueError("QSRT candidate damage provenance drifted")
        return pool.damage
    if metric != VALIDATION_DAMAGE_METRIC or weighting != VALIDATION_DAMAGE_WEIGHTING:
        raise ValueError("QSRT allocation damage metric is unsupported")
    if not isinstance(provenance, dict):
        raise ValueError("QSRT validation damage is missing its provenance")
    try:
        validation_root = Path(str(provenance["validation_scores"])).resolve()
    except KeyError as exc:
        raise ValueError("QSRT validation provenance has no score set") from exc
    validation = load_qsrt_validation_scores(validation_root, pool)
    expected = {
        "validation_scores": str(validation.root),
        "validation_score_set_sha256": validation.content_sha256,
        "validation_capture": validation.manifest["validation_capture"],
        "validation_report": validation.manifest["validation_report"],
        "validation_documents": validation.manifest["validation_documents"],
        "selection_data_used": False,
    }
    if provenance != expected:
        raise ValueError("QSRT validation damage provenance drifted")
    return validation.damage


def validate_qsrt_materialization_allocation(
    document: dict,
    pool: QSRTCandidatePool,
    x4t_index: X4TCostIndex,
) -> QSRTMaterializationPlan:
    """Authenticate the complete allocation and TP-free byte ledger."""

    if document.get("kind") != QSRT_ALLOCATION_KIND:
        raise ValueError("allocation is not a QSRT allocation")
    if document.get("schema_version") != QSRT_ALLOCATION_SCHEMA_VERSION:
        raise ValueError("QSRT allocation schema is unsupported")
    meta = document.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("QSRT allocation is missing its meta object")
    expected_meta = {
        "codec": "QSRT",
        "high_tier_storage": HIGH_TIER_STORAGE,
        "candidate_pool_content_sha256": pool.content_sha256,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(PHASE1_MODE_IDS),
        "x4t_cost_index_content_sha256": x4t_index.content_sha256,
        "source_revision": pool.manifest.get("source_revision"),
        "allocation_optimality": "exact-for-reported-lagrange-multiplier",
        "format_histogram_before_x4t": pool.format_histogram,
    }
    for name, expected in expected_meta.items():
        if meta.get(name) != expected:
            raise ValueError(
                f"QSRT allocation meta {name} mismatch: "
                f"{meta.get(name)!r} != {expected!r}"
            )
    if pool.codebook != CODEBOOK_SQG_XOR_CHEB_T12 or pool.mode_ids != PHASE1_MODE_IDS:
        raise ValueError("QSRT materialization requires sqg_xor_cheb_t12 and R0/R1/R2")
    if x4t_index.manifest.get("source_revision") != pool.manifest.get(
        "source_revision"
    ):
        raise ValueError("QSRT pool and X4T index source revisions disagree")
    if Path(str(meta.get("candidate_pool"))).resolve() != pool.root:
        raise ValueError("QSRT allocation names a different candidate pool")
    if Path(str(meta.get("x4t_cost_index"))).resolve() != x4t_index.root:
        raise ValueError("QSRT allocation names a different X4T cost index")
    damage = _allocation_damage(meta, pool)

    layers = document.get("layers")
    expected_layer_keys = {str(layer) for layer in C.MOE_LAYERS}
    if not isinstance(layers, dict) or set(layers) != expected_layer_keys:
        raise ValueError("QSRT allocation must contain exactly 92 MoE layers")
    mask = np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.bool_)
    specs: list[QSRTAtomLayerSpec] = []
    layer_x4t_bytes: list[int] = []
    total_atoms = 0
    total_x4t = 0
    universe = tuple(range(EXPERTS_PER_LAYER))
    for row, layer in enumerate(C.MOE_LAYERS):
        entry = layers[str(layer)]
        if not isinstance(entry, dict):
            raise ValueError(f"QSRT layer {layer} must be an object")
        x4t_raw = entry.get("x4t")
        compressed_raw = entry.get("compressed")
        codes_raw = entry.get("format_codes")
        if not isinstance(x4t_raw, list) or not isinstance(compressed_raw, list):
            raise ValueError(f"QSRT layer {layer} tiers must be lists")
        x4t = tuple(_plain_int(value, f"layer {layer} X4T expert") for value in x4t_raw)
        compressed = tuple(
            _plain_int(value, f"layer {layer} compressed expert")
            for value in compressed_raw
        )
        if x4t != tuple(sorted(x4t)) or compressed != tuple(sorted(compressed)):
            raise ValueError(f"QSRT layer {layer} tiers are not canonical")
        if (
            len(set(x4t)) != len(x4t)
            or len(set(compressed)) != len(compressed)
            or set(x4t).intersection(compressed)
            or tuple(sorted(x4t + compressed)) != universe
        ):
            raise ValueError(f"QSRT layer {layer} tiers do not form a partition")
        if not isinstance(codes_raw, list) or len(codes_raw) != EXPERTS_PER_LAYER:
            raise ValueError(f"QSRT layer {layer} format table has the wrong length")
        formats: list[ExpertFormatSpec] = []
        x4t_set = set(x4t)
        for expert, raw_code in enumerate(codes_raw):
            code = _plain_int(raw_code, f"layer {layer} format code")
            if not 0 <= code <= 0xFF:
                raise ValueError(f"QSRT layer {layer} format code is out of range")
            format_spec = ExpertFormatSpec.from_code(code)
            if expert in x4t_set:
                if code != FORMAT_X4T:
                    raise ValueError(f"QSRT layer {layer} X4T expert is not exact")
            elif (
                format_spec.is_x4t
                or format_spec.r13 != int(pool.selected_r13[row, expert])
                or format_spec.r2 != int(pool.selected_r2[row, expert])
            ):
                raise ValueError(
                    f"QSRT layer {layer} expert {expert} does not match its "
                    "selected trellis candidate"
                )
            formats.append(format_spec)
        mask[row, list(x4t)] = True
        layout = QSRTLayerLayout(len(compressed), len(x4t))
        expected_atoms = qsrt_trellis_layer_bytes(len(x4t))
        expected_x4t = X4T_LAYER_FIXED_BYTES + int(
            x4t_index.expert_storage_bytes[row, mask[row]].sum()
        )
        if expected_atoms != layout.disk_bytes:
            raise AssertionError("QSRT atom byte accounting drifted")
        if (
            _plain_int(entry.get("qsrt_atom_layer_bytes"), "qsrt_atom_layer_bytes")
            != expected_atoms
        ):
            raise ValueError(f"QSRT layer {layer} atom bytes do not close")
        if (
            _plain_int(entry.get("x4t_layer_bytes"), "x4t_layer_bytes")
            != expected_x4t
        ):
            raise ValueError(f"QSRT layer {layer} X4T bytes do not close")
        specs.append(QSRTAtomLayerSpec(layer, tuple(formats), compressed, x4t))
        layer_x4t_bytes.append(expected_x4t)
        total_atoms += expected_atoms
        total_x4t += expected_x4t

    lagrange_lambda = _finite(
        meta.get("lagrange_lambda_damage_per_byte"), "QSRT Lagrange multiplier"
    )
    expected = choose_qsrt_lagrangian(
        damage,
        x4t_index.expert_storage_bytes,
        lagrange_lambda=lagrange_lambda,
        target_container_bytes=meta.get("target_container_bytes"),
    )
    if not np.array_equal(mask, expected.x4t_mask):
        raise ValueError("QSRT X4T set is not optimal for its reported multiplier")
    total_bytes = qsrt_total_container_bytes(mask, x4t_index.expert_storage_bytes)
    if total_bytes != total_atoms + total_x4t:
        raise AssertionError("QSRT materialization byte accounting drifted")
    integer_fields = {
        "x4t_experts": int(mask.sum()),
        "compressed_experts": int(mask.size - mask.sum()),
        "container_bytes": total_bytes,
    }
    for name, actual in integer_fields.items():
        if _plain_int(meta.get(name), f"QSRT meta {name}") != actual:
            raise ValueError(f"QSRT allocation meta {name} does not close")
    target = meta.get("target_container_bytes")
    expected_slack = None if target is None else int(target) - total_bytes
    if meta.get("budget_slack_bytes") != expected_slack:
        raise ValueError("QSRT allocation budget slack does not close")
    total_damage = float(damage.sum())
    avoided = float(damage[mask].sum())
    float_fields = {
        "realized_x4t_fraction": int(mask.sum()) / mask.size,
        "effective_bits_per_original_expert_weight": (
            total_bytes
            * 8
            / (C.NUM_MOE_LAYERS * C.NUM_EXPERTS * 3 * 3072 * 3584)
        ),
        "all_compressed_damage": total_damage,
        "x4t_damage_avoided": avoided,
        "remaining_compressed_damage": total_damage - avoided,
    }
    for name, actual in float_fields.items():
        stored = _finite(meta.get(name), f"QSRT meta {name}")
        if not math.isclose(stored, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"QSRT allocation meta {name} does not close")
    return QSRTMaterializationPlan(
        layers=tuple(specs),
        x4t_layer_bytes=tuple(layer_x4t_bytes),
        qsrt_atom_bytes=total_atoms,
        x4t_bytes=total_x4t,
        total_container_bytes=total_bytes,
        x4t_experts=int(mask.sum()),
        compressed_experts=int(mask.size - mask.sum()),
    )


def qsrt_layer_closure_filename(layer: int) -> str:
    if layer not in C.MOE_LAYERS:
        raise ValueError("QSRT closure layer must lie in 1..92")
    return f"{layer_filename(layer)}{QSRT_CLOSURE_SUFFIX}"


def validate_qsrt_layer_pair(
    atom_path: str | Path,
    x4t_path: str | Path,
    spec: QSRTAtomLayerSpec,
    *,
    expected_x4t_bytes: int,
) -> dict[str, int | str]:
    """Validate one complete TP-free layer file pair."""

    atom_path = Path(atom_path)
    x4t_path = Path(x4t_path)
    with QSRTAtomLayerReader(atom_path) as atoms:
        if atoms.header.layer != spec.layer or atoms.header.layout != spec.layout:
            raise ValueError("QSRT atom layer disagrees with its materialization plan")
        expected_formats = tuple(value.name for value in spec.formats)
        if atoms.formats != expected_formats:
            raise ValueError("QSRT atom layer format table drifted")
    x4t = X4TLayerReader(x4t_path)
    if x4t.layer != spec.layer or x4t.file_bytes != expected_x4t_bytes:
        raise ValueError("QSRT X4T layer disagrees with its materialization plan")
    expected_matrices = len(spec.x4t) * len(X4T_MATRIX_ORDER)
    if x4t.matrix_count != expected_matrices:
        raise ValueError("QSRT X4T matrix count disagrees with its tier")
    exact = set(spec.x4t)
    for expert in range(C.NUM_EXPERTS):
        for matrix in X4T_MATRIX_ORDER:
            if x4t.has(expert, matrix) != (expert in exact):
                raise ValueError(
                    f"layer {spec.layer} X4T inventory drifted at {expert} {matrix}"
                )
    return {
        "atom_file": atom_path.name,
        "atom_disk_bytes": atom_path.stat().st_size,
        "x4t_file": x4t_path.name,
        "x4t_disk_bytes": x4t.file_bytes,
        "compressed_experts": len(spec.compressed),
        "x4t_experts": len(spec.x4t),
        "layer_container_bytes": atom_path.stat().st_size + x4t.file_bytes,
    }


def validate_qsrt_layer_payloads(
    atom_path: str | Path,
    x4t_path: str | Path,
    spec: QSRTAtomLayerSpec,
    candidate_root: str | Path,
    store: OfficialMXFP4Store,
    *,
    expected_x4t_bytes: int,
) -> dict[str, int | str]:
    """Compare every atom with its candidate and every X4T tensor with source."""

    structural = validate_qsrt_layer_pair(
        atom_path, x4t_path, spec, expected_x4t_bytes=expected_x4t_bytes
    )
    candidate_path = candidate_layer_path(candidate_root, spec.layer)
    with (
        QSRTAtomLayerReader(atom_path) as atom_reader,
        safe_open(candidate_path, framework="pt", device="cpu") as candidates,
    ):
        expected_shared: dict[str, torch.Tensor] = {}
        for compressed_slot, expert in enumerate(spec.compressed):
            tensors = {
                matrix: {
                    part: candidates.get_tensor(
                        candidate_tensor_name(spec.layer, expert, matrix, part)
                    )
                    for part in ("trellis", "suh", "svh")
                }
                for matrix in C.EXPERT_MATRICES
            }
            expected, shared = assemble_candidate_atoms(
                layer=spec.layer,
                expert=expert,
                format_spec=spec.formats[expert],
                tensors=tensors,
            )
            for name, value in shared.items():
                reference = expected_shared.setdefault(name, value)
                if not torch.equal(reference, value):
                    raise ValueError(
                        f"layer {spec.layer} candidate shared scale {name} drifted"
                    )
            if not torch.equal(atom_reader.read_expert_atoms(compressed_slot), expected):
                raise ValueError(
                    f"layer {spec.layer} expert {expert} atom bytes drifted"
                )
        actual_shared = dict(
            zip(
                ("w1.suh", "w3.suh", "w2.svh"),
                atom_reader.shared_scales,
                strict=True,
            )
        )
        if spec.compressed:
            if set(expected_shared) != set(actual_shared):
                raise AssertionError("QSRT shared-scale inventory drifted")
            for name, value in expected_shared.items():
                if not torch.equal(actual_shared[name], value):
                    raise ValueError(
                        f"layer {spec.layer} shared scale {name} drifted"
                    )
        elif any(bool(torch.count_nonzero(value)) for value in actual_shared.values()):
            raise ValueError(
                f"layer {spec.layer} all-X4T shared scale section is nonzero"
            )
    x4t_reader = X4TLayerReader(x4t_path)
    if spec.x4t:
        with store.open_layer(spec.layer, experts=spec.x4t) as source:
            for expert in spec.x4t:
                for matrix in X4T_MATRIX_ORDER:
                    actual = x4t_reader.read(expert, matrix)
                    expected = source.load_packed_matrix(spec.layer, expert, matrix)
                    if not torch.equal(actual.packed, expected.packed) or not torch.equal(
                        actual.scale, expected.scale
                    ):
                        raise ValueError(
                            f"layer {spec.layer} expert {expert} {matrix} X4T drifted"
                        )
    return {
        **structural,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(spec.compressed),
        "x4t_experts_verified": len(spec.x4t),
    }


def _validate_full_qsrt_closure(
    metadata: object,
    spec: QSRTAtomLayerSpec,
    structural: dict[str, int | str],
) -> dict[str, int | str]:
    expected = {
        **structural,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(spec.compressed),
        "x4t_experts_verified": len(spec.x4t),
    }
    if metadata != expected:
        raise ValueError(
            f"layer {spec.layer} closure does not describe complete source coverage"
        )
    return expected


def write_qsrt_layer_closure_receipt(
    atom_path: str | Path,
    x4t_path: str | Path,
    spec: QSRTAtomLayerSpec,
    metadata: dict[str, int | str],
    *,
    expected_x4t_bytes: int,
    build_document: dict,
) -> Path:
    atom_path = Path(atom_path)
    x4t_path = Path(x4t_path)
    structural = validate_qsrt_layer_pair(
        atom_path, x4t_path, spec, expected_x4t_bytes=expected_x4t_bytes
    )
    closure = _validate_full_qsrt_closure(metadata, spec, structural)
    receipt = {
        "kind": QSRT_CLOSURE_KIND,
        "schema_version": QSRT_CLOSURE_SCHEMA_VERSION,
        "layer": spec.layer,
        "build_sha256": _canonical_json_sha256(build_document),
        "atom_identity": _file_identity(atom_path),
        "x4t_identity": _file_identity(x4t_path),
        "closure": closure,
    }
    path = atom_path.with_name(qsrt_layer_closure_filename(spec.layer))
    _atomic_json(path, receipt)
    return path


def load_qsrt_layer_closure_receipt(
    atom_path: str | Path,
    x4t_path: str | Path,
    spec: QSRTAtomLayerSpec,
    *,
    expected_x4t_bytes: int,
    build_document: dict,
) -> dict[str, int | str]:
    atom_path = Path(atom_path)
    x4t_path = Path(x4t_path)
    receipt = _read_json(atom_path.with_name(qsrt_layer_closure_filename(spec.layer)))
    expected = {
        "kind": QSRT_CLOSURE_KIND,
        "schema_version": QSRT_CLOSURE_SCHEMA_VERSION,
        "layer": spec.layer,
        "build_sha256": _canonical_json_sha256(build_document),
        "atom_identity": _file_identity(atom_path),
        "x4t_identity": _file_identity(x4t_path),
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"layer {spec.layer} closure receipt {name} drifted")
    structural = validate_qsrt_layer_pair(
        atom_path, x4t_path, spec, expected_x4t_bytes=expected_x4t_bytes
    )
    return _validate_full_qsrt_closure(receipt.get("closure"), spec, structural)


def qsrt_materialization_build_document(
    *,
    pool: QSRTCandidatePool,
    x4t_index: X4TCostIndex,
    allocation_path: str | Path,
    official_root: str | Path,
    official_revision: str,
    plan: QSRTMaterializationPlan,
) -> dict:
    if pool.content_sha256 is None:
        raise ValueError("QSRT materialization requires a sealed candidate pool")
    allocation_path = Path(allocation_path).resolve()
    candidate_manifest = pool.root / "qsrt-candidate-manifest.json"
    candidate_completion = pool.root / "qsrt-candidate-completion.json"
    x4t_manifest = x4t_index.root / X4T_COST_MANIFEST_FILENAME
    x4t_completion = x4t_index.root / X4T_COST_COMPLETION_FILENAME
    for path in (
        allocation_path,
        candidate_manifest,
        candidate_completion,
        x4t_manifest,
        x4t_completion,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    allocation_meta = _read_json(allocation_path).get("meta")
    if not isinstance(allocation_meta, dict):
        raise ValueError("QSRT allocation is missing its meta object")
    return {
        "kind": QSRT_BUILD_KIND,
        "schema_version": QSRT_BUILD_SCHEMA_VERSION,
        "codec": "QSRT",
        "storage_schema": "kquant_kimi_k3_qsrt_atoms_v1",
        "codec_contract": {
            "trellis_codebook": pool.codebook,
            "trellis_modes": list(PHASE1_MODE_IDS),
            "separate_r13_r2": True,
            "high_tier": "X4T",
        },
        "source_model": C.MODEL_ID,
        "source_revision": official_revision,
        "official_checkpoint": str(Path(official_root).resolve()),
        "candidate_pool": str(pool.root),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "candidate_completion_sha256": _sha256(candidate_completion),
        "candidate_pool_content_sha256": pool.content_sha256,
        "candidate_damage_metric": pool.damage_metric,
        "candidate_damage_weighting": pool.damage_weighting,
        "x4t_cost_index": str(x4t_index.root),
        "x4t_manifest_sha256": _sha256(x4t_manifest),
        "x4t_completion_sha256": _sha256(x4t_completion),
        "x4t_cost_index_content_sha256": x4t_index.content_sha256,
        "allocation_source": str(allocation_path),
        "allocation_sha256": _sha256(allocation_path),
        "allocation_damage_metric": allocation_meta.get("damage_metric"),
        "allocation_damage_weighting": allocation_meta.get("damage_weighting"),
        "allocation_damage_provenance": allocation_meta.get("damage_provenance"),
        "qsrt_atom_bytes": plan.qsrt_atom_bytes,
        "x4t_bytes": plan.x4t_bytes,
        "container_bytes": plan.total_container_bytes,
        "x4t_experts": plan.x4t_experts,
        "compressed_experts": plan.compressed_experts,
        "layer_count": len(plan.layers),
        "materialization_status": "reference-bytes; serving gates are separate",
        "byte_order": "little",
    }


def prepare_qsrt_destination(
    destination: str | Path,
    *,
    build_document: dict,
    allocation_path: str | Path,
    pool: QSRTCandidatePool,
    x4t_index: X4TCostIndex,
    resume: bool,
) -> Path:
    destination = Path(destination).resolve()
    sources = {
        QSRT_ALLOCATION_COPY_FILENAME: Path(allocation_path).resolve(),
        QSRT_CANDIDATE_MANIFEST_COPY_FILENAME: (
            pool.root / "qsrt-candidate-manifest.json"
        ),
        QSRT_CANDIDATE_COMPLETION_COPY_FILENAME: (
            pool.root / "qsrt-candidate-completion.json"
        ),
        QSRT_X4T_MANIFEST_COPY_FILENAME: x4t_index.root / X4T_COST_MANIFEST_FILENAME,
        QSRT_X4T_COMPLETION_COPY_FILENAME: (
            x4t_index.root / X4T_COST_COMPLETION_FILENAME
        ),
    }
    build_path = destination / QSRT_BUILD_FILENAME
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        if not build_path.is_file() or _read_json(build_path) != build_document:
            raise ValueError("QSRT resume destination build contract does not match")
        for name, source in sources.items():
            copy = destination / name
            if not copy.is_file() or copy.read_bytes() != source.read_bytes():
                raise ValueError(f"QSRT resume metadata copy {name} drifted")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    staging.mkdir()
    try:
        _atomic_json(staging / QSRT_BUILD_FILENAME, build_document)
        for name, source in sources.items():
            _atomic_bytes(staging / name, source.read_bytes())
        _fsync_directory(staging)
        staging.replace(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if staging.is_dir():
            for child in staging.iterdir():
                if child.is_file():
                    child.unlink()
            staging.rmdir()
        raise
    return destination


def qsrt_artifact_manifest(
    *,
    build_document: dict,
    plan: QSRTMaterializationPlan,
    layers: Sequence[dict[str, int | str]],
) -> dict:
    if len(layers) != len(plan.layers):
        raise ValueError("a final QSRT manifest requires every MoE layer")
    by_layer: dict[str, dict] = {}
    atom_bytes = 0
    x4t_bytes = 0
    for spec, expected_x4t, metadata in zip(
        plan.layers, plan.x4t_layer_bytes, layers, strict=True
    ):
        if metadata.get("atom_disk_bytes") != spec.layout.disk_bytes:
            raise ValueError(f"layer {spec.layer} atom byte ledger drifted")
        if metadata.get("x4t_disk_bytes") != expected_x4t:
            raise ValueError(f"layer {spec.layer} X4T byte ledger drifted")
        atom_bytes += spec.layout.disk_bytes
        x4t_bytes += expected_x4t
        by_layer[str(spec.layer)] = {
            **metadata,
            "qsrt_atoms": layer_filename(spec.layer),
            "x4t": x4t_layer_path(Path("."), spec.layer).name,
            "layout": spec.layout.to_manifest(),
        }
    if (
        atom_bytes != plan.qsrt_atom_bytes
        or x4t_bytes != plan.x4t_bytes
        or atom_bytes + x4t_bytes != plan.total_container_bytes
    ):
        raise AssertionError("QSRT final artifact byte ledger does not close")
    return {
        "kind": QSRT_ARTIFACT_KIND,
        "schema_version": QSRT_ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "build": QSRT_BUILD_FILENAME,
        "build_sha256": _canonical_json_sha256(build_document),
        "allocation": QSRT_ALLOCATION_COPY_FILENAME,
        "codec": "QSRT",
        "storage_schema": "kquant_kimi_k3_qsrt_atoms_v1",
        "source_model": build_document["source_model"],
        "source_revision": build_document["source_revision"],
        "candidate_pool_content_sha256": build_document[
            "candidate_pool_content_sha256"
        ],
        "x4t_cost_index_content_sha256": build_document[
            "x4t_cost_index_content_sha256"
        ],
        "trellis_codebook": build_document["codec_contract"]["trellis_codebook"],
        "trellis_modes": list(PHASE1_MODE_IDS),
        "qsrt_atom_bytes": atom_bytes,
        "x4t_bytes": x4t_bytes,
        "container_bytes": atom_bytes + x4t_bytes,
        "x4t_experts": plan.x4t_experts,
        "compressed_experts": plan.compressed_experts,
        "layer_count": len(plan.layers),
        "layers": by_layer,
    }


def write_qsrt_artifact_manifest(destination: str | Path, document: dict) -> Path:
    path = Path(destination) / QSRT_MANIFEST_FILENAME
    if path.exists():
        if _read_json(path) != document:
            raise ValueError("existing QSRT manifest disagrees with the build")
        return path
    _atomic_json(path, document)
    return path


__all__ = [
    "QSRT_ALLOCATION_COPY_FILENAME",
    "QSRT_ARTIFACT_KIND",
    "QSRT_ARTIFACT_SCHEMA_VERSION",
    "QSRT_BUILD_FILENAME",
    "QSRT_BUILD_KIND",
    "QSRT_BUILD_SCHEMA_VERSION",
    "QSRT_CANDIDATE_COMPLETION_COPY_FILENAME",
    "QSRT_CANDIDATE_MANIFEST_COPY_FILENAME",
    "QSRT_MANIFEST_FILENAME",
    "QSRTMaterializationPlan",
    "QSRT_X4T_COMPLETION_COPY_FILENAME",
    "QSRT_X4T_MANIFEST_COPY_FILENAME",
    "load_qsrt_layer_closure_receipt",
    "prepare_qsrt_destination",
    "qsrt_artifact_manifest",
    "qsrt_layer_closure_filename",
    "qsrt_materialization_build_document",
    "validate_qsrt_layer_pair",
    "validate_qsrt_layer_payloads",
    "validate_qsrt_materialization_allocation",
    "write_qsrt_artifact_manifest",
    "write_qsrt_layer_closure_receipt",
]
