"""Materialize selected TP12 mixed-EXL candidates into fixed layer slabs.

The candidate pool is deliberately allocation-independent.  This module joins
one validated allocation with those candidate bytes and the original packed
MXFP4 keep tier.  It never instantiates or dequantizes the official model:
candidate tensors and official matrices are consumed one at a time and placed
directly into the exact offsets owned by :class:`TP12LayerLayout`.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_MCG, EXL3_CODEBOOKS
from kquant.mixed_exl3 import (
    EXPERTS_PER_LAYER,
    EXPERT_RANK_MXFP4_BYTES,
    FORMAT_MXFP4,
    FORMAT_SECTION_BYTES,
    FORMAT_TABLE_BYTES,
    INTERMEDIATE_CHANNELS,
    KEEP_STORAGE_EXTERNAL_X4,
    KEEP_STORAGE_INLINE_MXFP4,
    LATENT_CHANNELS,
    EXPERT_RANK_TRELLIS_BYTES,
    EXPERT_TRELLIS_BYTES,
    LAYER_HEADER_BYTES,
    LOCAL_SCALE_VECTOR_BYTES,
    LOGICAL_CANDIDATE_SCHEMAS,
    MATRIX_TRELLIS_BYTES,
    PAIR_BYTES,
    RANK_MATRIX_WEIGHTS,
    RANK_MXFP4_MATRIX_BYTES,
    SCALE_BYTES,
    SCHEMA,
    SHARED_SCALE_SECTION_BYTES,
    TP_SIZE,
    ExpertFormatSpec,
    TP12LayerHeader,
    TP12LayerLayout,
    pack_tp12_format_section,
    pack_tp12_shared_scale_section,
    tp12_logical_pair_index,
    unpack_tp12_format_section,
    unpack_tp12_rank_scale_section,
    unpack_tp12_shared_scale_section,
)
from kquant.pack.mixed_allocation import (
    ALLOCATION_KIND,
    ALLOCATION_SCHEMA_VERSION,
    MixedCandidatePool,
    allocation_optimality,
    choose_mixed_allocation,
)
from kquant.pack.mixed_candidates import (
    CANDIDATE_POOL_SCHEMA_VERSION,
    candidate_tensor_name,
)
from kquant.pack.mixed_mode_validation import (
    MODE_VALIDATION_SUMMARY_KIND,
    MODE_VALIDATION_SUMMARY_SCHEMA_VERSION,
    load_mixed_mode_validation_summary,
)
from kquant.pack.mixed_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
    MixedValidationScores,
    load_mixed_validation_scores,
)
from kquant.source_weights import OfficialMXFP4Store, PackedMXFP4Matrix
from kquant.teacher_proxy_gate import validate_teacher_proxy_gate_report
from kquant.tp12_performance_gate import validate_tp12_performance_report
from kquant.x4 import X4LayerReader, X4LayerWriter


MATERIALIZED_ARTIFACT_KIND = "kquant_tp12_mixed_exl3_artifact"
MATERIALIZED_ARTIFACT_SCHEMA_VERSION = 4
MATERIALIZATION_BUILD_KIND = "kquant_tp12_mixed_exl3_materialization_request"
MATERIALIZATION_BUILD_SCHEMA_VERSION = 4
MATERIALIZATION_BUILD_FILENAME = "mixed_exl3_tp12_build.json"
MATERIALIZED_MANIFEST_FILENAME = "mixed_exl3_tp12_manifest.json"
MATERIALIZED_ALLOCATION_FILENAME = "allocation-mixed-exl3-tp12.json"
MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME = (
    "mixed-exl3-mode-validation-summary.json"
)
MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME = (
    "mixed-exl3-teacher-proxy-summary.json"
)
MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME = (
    "mixed-exl3-tp12-performance-summary.json"
)
MATERIALIZED_LAYER_PREFIX = "mixed-exl3-tp12-layer-"
MATERIALIZED_CLOSURE_KIND = "kquant_tp12_mixed_exl3_layer_closure"
MATERIALIZED_CLOSURE_SCHEMA_VERSION = 1
MATERIALIZED_CLOSURE_SUFFIX = ".closure.json"
MXFP4_MATRIX_COMPONENT_ORDER = ("weight_packed", "weight_scale")


@dataclass(frozen=True)
class LayerMaterializationSpec:
    """Validated, canonical materialization decision for one decoder layer."""

    layer: int
    formats: tuple[ExpertFormatSpec, ...]
    compressed: tuple[int, ...]
    kept: tuple[int, ...]
    layout: TP12LayerLayout
    codebook: str = CODEBOOK_MCG

    def __post_init__(self) -> None:
        if self.layer not in C.MOE_LAYERS:
            raise ValueError("materialized layer must be a K3 MoE layer")
        if self.codebook not in EXL3_CODEBOOKS:
            raise ValueError("materialized layer codebook is unsupported")
        if len(self.formats) != EXPERTS_PER_LAYER:
            raise ValueError(
                f"materialized layer must describe {EXPERTS_PER_LAYER} experts"
            )
        universe = tuple(range(EXPERTS_PER_LAYER))
        if (
            self.compressed != tuple(sorted(self.compressed))
            or self.kept != tuple(sorted(self.kept))
            or len(set(self.compressed)) != len(self.compressed)
            or len(set(self.kept)) != len(self.kept)
            or set(self.compressed).intersection(self.kept)
            or tuple(sorted(self.compressed + self.kept)) != universe
        ):
            raise ValueError("materialized layer tiers must be a canonical partition")
        expected_compressed = tuple(
            expert
            for expert, format_spec in enumerate(self.formats)
            if not format_spec.is_mxfp4
        )
        expected_kept = tuple(
            expert
            for expert, format_spec in enumerate(self.formats)
            if format_spec.is_mxfp4
        )
        if self.compressed != expected_compressed or self.kept != expected_kept:
            raise ValueError("materialized layer tiers disagree with its format table")
        if self.layout != TP12LayerLayout.from_formats(
            self.formats,
            keep_storage=self.layout.keep_storage,
        ):
            raise ValueError("materialized layer layout disagrees with its format table")

    @property
    def format_codes(self) -> tuple[int, ...]:
        return tuple(value.code for value in self.formats)


@dataclass(frozen=True)
class MaterializationPlan:
    """Complete allocation after closing it against its candidate pool."""

    layers: tuple[LayerMaterializationSpec, ...]
    total_container_bytes: int
    retained_experts: int
    compressed_experts: int


@dataclass(frozen=True)
class TP12CompressedRankPayload:
    """One rank's compressed tier in B12X's public pair-decoder contract.

    The disk slab is expert-major with ``w1, w3, w2`` pairs.  B12X consumes
    projection-major FC1 pairs, a separate FC2 plane, and local rotations in
    ``gate_svh, up_svh, down_suh`` order.  Constructing this value is the only
    format-to-runtime transpose required by the reference loader.
    """

    rank: int
    expert_ids: torch.Tensor
    w13_trellis: torch.Tensor
    w2_trellis: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    intermediate_rotations: torch.Tensor
    down_svh: torch.Tensor
    fc1_pair_modes: torch.Tensor
    fc2_pair_modes: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= self.rank < TP_SIZE:
            raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
        count = int(self.expert_ids.numel())
        words = PAIR_BYTES // torch.int16.itemsize
        expected = (
            ("expert_ids", self.expert_ids, torch.int32, (count,)),
            ("w13_trellis", self.w13_trellis, torch.int16, (2, count, words)),
            ("w2_trellis", self.w2_trellis, torch.int16, (count, words)),
            ("gate_suh", self.gate_suh, torch.float16, (1, LATENT_CHANNELS)),
            ("up_suh", self.up_suh, torch.float16, (1, LATENT_CHANNELS)),
            (
                "intermediate_rotations",
                self.intermediate_rotations,
                torch.float16,
                (count, 3 * (INTERMEDIATE_CHANNELS // TP_SIZE)),
            ),
            ("down_svh", self.down_svh, torch.float16, (1, LATENT_CHANNELS)),
            ("fc1_pair_modes", self.fc1_pair_modes, torch.int32, (count,)),
            ("fc2_pair_modes", self.fc2_pair_modes, torch.int32, (count,)),
        )
        device = self.expert_ids.device
        for name, value, dtype, shape in expected:
            if value.dtype != dtype or tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must be contiguous {dtype} {shape}, got "
                    f"{value.dtype} {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError(f"{name} must be on {device}, got {value.device}")
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if count:
            ids = self.expert_ids.to(device="cpu", dtype=torch.int64).tolist()
            if ids != sorted(ids) or len(set(ids)) != count:
                raise ValueError("expert_ids must be strictly increasing")
            if ids[0] < 0 or ids[-1] >= EXPERTS_PER_LAYER:
                raise ValueError("expert_ids contain an out-of-range expert")
        for name, modes in (
            ("fc1_pair_modes", self.fc1_pair_modes),
            ("fc2_pair_modes", self.fc2_pair_modes),
        ):
            if not bool(torch.all((modes == 0) | (modes == 1))):
                raise ValueError(f"{name} values must be 0=P33 or 1=P24")
        for name, scale in (
            ("gate_suh", self.gate_suh),
            ("up_suh", self.up_suh),
            ("intermediate_rotations", self.intermediate_rotations),
            ("down_svh", self.down_svh),
        ):
            if not bool(torch.all(torch.isfinite(scale))):
                raise ValueError(f"{name} contains non-finite values")

    @property
    def num_experts(self) -> int:
        return int(self.expert_ids.numel())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TP12CompressedRankPayload":
        """Move the complete runtime payload to one device."""

        target = torch.device(device)
        values = {
            name: getattr(self, name).to(target, non_blocking=non_blocking)
            for name in (
                "expert_ids",
                "w13_trellis",
                "w2_trellis",
                "gate_suh",
                "up_suh",
                "intermediate_rotations",
                "down_svh",
                "fc1_pair_modes",
                "fc2_pair_modes",
            )
        }
        return TP12CompressedRankPayload(rank=self.rank, **values)

    def b12x_prepare_kwargs(self) -> dict[str, torch.Tensor]:
        """Arguments consumed by ``sparkinfer.moe.fused_moe.prepare_weights``."""

        return {
            "w1_fp4": self.w13_trellis,
            "w2_fp4": self.w2_trellis,
            "gate_suh": self.gate_suh,
            "up_suh": self.up_suh,
            "intermediate_rotations": self.intermediate_rotations,
            "down_svh": self.down_svh,
            "trellis_fc1_pair_modes": self.fc1_pair_modes,
            "trellis_fc2_pair_modes": self.fc2_pair_modes,
        }


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, indent=1, sort_keys=True).encode("utf-8"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _allocation_validation_scores(
    meta: dict,
    pool: MixedCandidatePool,
) -> MixedValidationScores:
    """Resolve and authenticate the untouched-validation score set."""

    if meta.get("damage_metric") != VALIDATION_DAMAGE_METRIC:
        raise ValueError("production materialization requires validation damage")
    if meta.get("damage_weighting") != VALIDATION_DAMAGE_WEIGHTING:
        raise ValueError("allocation validation damage weighting drifted")
    provenance = meta.get("damage_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("allocation validation damage has no provenance")
    try:
        validation_root = Path(str(provenance["validation_scores"])).resolve()
    except KeyError as exc:
        raise ValueError(
            "allocation validation provenance has no score set"
        ) from exc
    validation = load_mixed_validation_scores(validation_root, pool)
    expected_provenance = {
        "validation_scores": str(validation.root),
        "validation_score_set_sha256": validation.content_sha256,
        "validation_capture": validation.manifest["validation_capture"],
        "validation_report": validation.manifest["validation_report"],
        "validation_documents": validation.manifest["validation_documents"],
        "selection_data_used": False,
    }
    if provenance != expected_provenance:
        raise ValueError("allocation validation damage provenance drifted")
    return validation


def _allocation_damage(meta: dict, pool: MixedCandidatePool) -> np.ndarray:
    """Resolve and authenticate the damage source named by an allocation."""

    metric = meta.get("damage_metric")
    weighting = meta.get("damage_weighting")
    provenance = meta.get("damage_provenance")
    if metric == pool.damage_metric:
        if weighting != pool.damage_weighting:
            raise ValueError("allocation candidate-pool damage weighting drifted")
        if provenance != pool.damage_provenance:
            raise ValueError("allocation candidate-pool damage provenance drifted")
        return pool.damage
    if metric != VALIDATION_DAMAGE_METRIC:
        raise ValueError(f"allocation damage metric is unsupported: {metric!r}")
    return _allocation_validation_scores(meta, pool).damage


def validate_materialization_mode_gate(
    summary_path: str | Path,
    allocation: dict,
    pool: MixedCandidatePool,
) -> dict[str, object]:
    """Require a reproducible matched-R0 pass for the allocation's score set."""

    meta = allocation.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("allocation is missing its meta object")
    validation = _allocation_validation_scores(meta, pool)
    summary_path = Path(summary_path).resolve()
    summary, scores = load_mixed_mode_validation_summary(
        summary_path,
        pool,
        validation,
        require_pass=True,
    )
    return {
        "kind": MODE_VALIDATION_SUMMARY_KIND,
        "schema_version": MODE_VALIDATION_SUMMARY_SCHEMA_VERSION,
        "status": "pass",
        "summary_sha256": _sha256(summary_path),
        "candidate_pool_content_sha256": pool.content_sha256,
        "selected_validation_score_set_sha256": validation.content_sha256,
        "mode_validation_scores": str(scores.root),
        "mode_validation_score_set_sha256": scores.content_sha256,
        "bootstrap_replicates": summary["bootstrap_replicates"],
        "bootstrap_seed": summary["bootstrap_seed"],
        "audited_assignments": int(scores.audited.sum()),
    }


def validate_materialization_teacher_proxy_gate(
    summary_path: str | Path,
) -> dict[str, object]:
    """Require the preregistered official/interim calibration-proxy pass."""

    return validate_teacher_proxy_gate_report(summary_path, require_pass=True)


def validate_materialization_performance_gate(
    summary_path: str | Path,
    pool: MixedCandidatePool,
) -> dict[str, object]:
    """Require isolated and natural-route TP12 kernel acceptance."""

    receipt = validate_tp12_performance_report(
        summary_path,
        expected_candidate_pool=pool.root,
        require_pass=True,
    )
    return {
        **receipt,
        "candidate_pool_content_sha256": pool.content_sha256,
    }


def _validate_allocation_decision(
    document: dict,
    meta: dict,
    pool: MixedCandidatePool,
) -> None:
    """Recompute the keep decision and every damage-ledger scalar."""

    damage = _allocation_damage(meta, pool)
    keep_mask = np.zeros(
        (C.NUM_MOE_LAYERS, C.NUM_EXPERTS),
        dtype=np.bool_,
    )
    for row, layer in enumerate(C.MOE_LAYERS):
        keep_mask[row, document["layers"][str(layer)]["keep"]] = True

    raw_target = _require_plain_int(
        meta.get("raw_target_keep_count"),
        "allocation meta raw_target_keep_count",
    )
    target_bytes = meta.get("target_container_bytes")
    if target_bytes is None:
        expected = choose_mixed_allocation(damage, keep_count=raw_target)
    else:
        target_bytes = _require_plain_int(
            target_bytes,
            "allocation meta target_container_bytes",
        )
        expected = choose_mixed_allocation(
            damage,
            target_container_bytes=target_bytes,
        )
    if not np.array_equal(keep_mask, expected.keep_mask):
        raise ValueError(
            "allocation keep tier is not the deterministic optimum for its "
            "authenticated damage source"
        )

    integer_fields = {
        "raw_target_keep_count": expected.raw_target_keep_count,
        "alignment_repair_swaps": expected.alignment_repair_swaps,
    }
    for name, actual in integer_fields.items():
        if _require_plain_int(meta.get(name), f"allocation meta {name}") != actual:
            raise ValueError(f"allocation meta {name} does not close")

    retained_damage = float(damage[keep_mask].sum())
    compressed_damage = float(damage[~keep_mask].sum())
    total_damage = float(damage.sum())
    parameters = C.NUM_MOE_LAYERS * C.NUM_EXPERTS * 3 * 3072 * 3584
    float_fields = {
        "realized_keep_fraction": expected.retained_experts / keep_mask.size,
        "effective_bits_per_original_expert_weight": (
            expected.container_bytes * 8 / parameters
        ),
        "all_compressed_damage": total_damage,
        "retained_damage_avoided": retained_damage,
        "remaining_compressed_damage": compressed_damage,
        "alignment_repair_damage_cost": expected.alignment_repair_damage_cost,
    }
    for name, actual in float_fields.items():
        stored = _require_finite_number(meta.get(name), f"allocation meta {name}")
        if not math.isclose(stored, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"allocation meta {name} does not close")
    if meta.get("format_histogram_before_keep") != pool.format_histogram:
        raise ValueError("allocation pre-keep format histogram does not close")
    if meta.get("allocation_optimality") != allocation_optimality(expected):
        raise ValueError("allocation optimality claim does not close")


def validate_materialization_allocation(
    document: dict,
    pool: MixedCandidatePool,
) -> MaterializationPlan:
    """Close every allocation byte and mode against a validated candidate pool."""

    if document.get("kind") != ALLOCATION_KIND:
        raise ValueError("allocation is not a TP12 mixed-EXL allocation")
    if document.get("schema_version") != ALLOCATION_SCHEMA_VERSION:
        raise ValueError("allocation schema version is unsupported")
    meta = document.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("allocation is missing its meta object")
    expected_meta = {
        "candidate_schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "candidate_pool_content_sha256": pool.content_sha256,
        "candidate_logical_trellis_schema": pool.trellis_schema,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(pool.mode_ids),
        "source_revision": pool.manifest.get("source_revision"),
        "tp_size": TP_SIZE,
    }
    for name, expected in expected_meta.items():
        if meta.get(name) != expected:
            raise ValueError(
                f"allocation meta {name} mismatch: {meta.get(name)!r} != {expected!r}"
            )
    try:
        allocation_pool = Path(str(meta["candidate_pool"])).resolve()
    except KeyError as exc:
        raise ValueError("allocation does not identify its candidate pool") from exc
    if allocation_pool != pool.root:
        raise ValueError(
            f"allocation candidate pool {allocation_pool} != {pool.root}"
        )

    layers = document.get("layers")
    expected_layer_keys = {str(layer) for layer in C.MOE_LAYERS}
    if not isinstance(layers, dict) or set(layers) != expected_layer_keys:
        raise ValueError("allocation must contain exactly the 92 K3 MoE layers")

    universe = tuple(range(EXPERTS_PER_LAYER))
    specs: list[LayerMaterializationSpec] = []
    retained = 0
    compressed_count = 0
    total_bytes = 0
    for row, layer in enumerate(C.MOE_LAYERS):
        entry = layers[str(layer)]
        if not isinstance(entry, dict):
            raise ValueError(f"allocation layer {layer} must be an object")
        keep_raw = entry.get("keep")
        compressed_raw = entry.get("compressed")
        codes_raw = entry.get("format_codes")
        if not isinstance(keep_raw, list) or not isinstance(compressed_raw, list):
            raise ValueError(f"allocation layer {layer} tiers must be lists")
        keep = tuple(
            _require_plain_int(value, f"layer {layer} keep expert")
            for value in keep_raw
        )
        compressed = tuple(
            _require_plain_int(value, f"layer {layer} compressed expert")
            for value in compressed_raw
        )
        if keep != tuple(sorted(keep)) or compressed != tuple(sorted(compressed)):
            raise ValueError(f"allocation layer {layer} tiers are not canonical")
        if (
            len(set(keep)) != len(keep)
            or len(set(compressed)) != len(compressed)
            or set(keep).intersection(compressed)
            or tuple(sorted(keep + compressed)) != universe
        ):
            raise ValueError(f"allocation layer {layer} tiers do not form a partition")
        if not isinstance(codes_raw, list) or len(codes_raw) != EXPERTS_PER_LAYER:
            raise ValueError(
                f"allocation layer {layer} must contain {EXPERTS_PER_LAYER} format codes"
            )
        codes = tuple(
            _require_plain_int(value, f"layer {layer} format code")
            for value in codes_raw
        )
        if any(not 0 <= code <= 0xFF for code in codes):
            raise ValueError(f"allocation layer {layer} has an out-of-range format code")
        formats: list[ExpertFormatSpec] = []
        keep_set = set(keep)
        for expert, code in enumerate(codes):
            try:
                format_spec = ExpertFormatSpec.from_code(code)
            except ValueError as exc:
                raise ValueError(
                    f"allocation layer {layer} expert {expert} has invalid format code"
                ) from exc
            if expert in keep_set:
                if code != FORMAT_MXFP4:
                    raise ValueError(
                        f"allocation layer {layer} kept expert {expert} is not MXFP4"
                    )
            else:
                selected_r13 = int(pool.selected_r13[row, expert])
                selected_r2 = int(pool.selected_r2[row, expert])
                if (
                    format_spec.is_mxfp4
                    or format_spec.r13 not in pool.mode_ids
                    or format_spec.r2 not in pool.mode_ids
                    or format_spec.r13 != selected_r13
                    or format_spec.r2 != selected_r2
                ):
                    raise ValueError(
                        f"allocation layer {layer} expert {expert} format does not "
                        "match its selected phase-1 candidate"
                    )
            formats.append(format_spec)

        layout = TP12LayerLayout(len(compressed), len(keep))
        if TP12LayerLayout.from_formats(formats) != layout:
            raise AssertionError("format-derived TP12 layout disagrees with tier counts")
        specs.append(
            LayerMaterializationSpec(
                layer=layer,
                formats=tuple(formats),
                compressed=compressed,
                kept=keep,
                layout=layout,
                codebook=pool.codebook,
            )
        )
        retained += len(keep)
        compressed_count += len(compressed)
        total_bytes += layout.disk_bytes

    expected_total = C.NUM_MOE_LAYERS * C.NUM_EXPERTS
    if retained + compressed_count != expected_total:
        raise AssertionError("materialization assignment accounting drifted")
    for name, actual in (
        ("retained_experts", retained),
        ("compressed_experts", compressed_count),
        ("container_bytes", total_bytes),
    ):
        if _require_plain_int(meta.get(name), f"allocation meta {name}") != actual:
            raise ValueError(
                f"allocation meta {name} does not close: {meta.get(name)!r} != {actual}"
            )
    _validate_allocation_decision(document, meta, pool)
    return MaterializationPlan(
        layers=tuple(specs),
        total_container_bytes=total_bytes,
        retained_experts=retained,
        compressed_experts=compressed_count,
    )


def _validate_candidate_tensor(
    tensor: torch.Tensor,
    *,
    matrix: str,
    part: str,
) -> torch.Tensor:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        tensor = tensor.detach().cpu().contiguous()
    if part == "trellis":
        expected_dtype = torch.int16
        expected_values = MATRIX_TRELLIS_BYTES // torch.int16.itemsize
    else:
        expected_dtype = torch.float16
        if (matrix in ("w1", "w3") and part == "suh") or (
            matrix == "w2" and part == "svh"
        ):
            expected_values = LATENT_CHANNELS
        else:
            expected_values = INTERMEDIATE_CHANNELS
    if tensor.dtype != expected_dtype or tensor.ndim != 1 or tensor.numel() != expected_values:
        raise ValueError(
            f"{matrix}.{part} has {tensor.dtype} {tuple(tensor.shape)}, expected "
            f"{expected_dtype} ({expected_values},)"
        )
    if part != "trellis" and not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{matrix}.{part} contains non-finite scales")
    return tensor


def candidate_trellis_pair(
    trellis: torch.Tensor,
    logical_pair: int,
) -> torch.Tensor:
    """Return one logical P24/P33 pair from a candidate matrix payload."""

    trellis = _validate_candidate_tensor(trellis, matrix="w1", part="trellis")
    if (
        isinstance(logical_pair, bool)
        or not isinstance(logical_pair, int)
        or not 0 <= logical_pair < TP_SIZE
    ):
        raise ValueError(f"logical_pair must be in 0..{TP_SIZE - 1}")
    words = PAIR_BYTES // torch.int16.itemsize
    result = trellis.narrow(0, logical_pair * words, words)
    if not result.is_contiguous():
        result = result.contiguous()
    return result


def candidate_local_scale(
    scale: torch.Tensor,
    logical_pair: int,
) -> torch.Tensor:
    """Return one logical 256-neuron slice of a candidate scale vector."""

    if scale.dtype != torch.float16 or scale.ndim != 1 or scale.numel() != INTERMEDIATE_CHANNELS:
        raise ValueError(
            f"local scale must be torch.float16 ({INTERMEDIATE_CHANNELS},)"
        )
    if not bool(torch.all(torch.isfinite(scale))):
        raise ValueError("local scale contains non-finite values")
    if (
        isinstance(logical_pair, bool)
        or not isinstance(logical_pair, int)
        or not 0 <= logical_pair < TP_SIZE
    ):
        raise ValueError(f"logical_pair must be in 0..{TP_SIZE - 1}")
    width = INTERMEDIATE_CHANNELS // TP_SIZE
    result = scale.narrow(0, logical_pair * width, width)
    if not result.is_contiguous():
        result = result.contiguous()
    if result.numel() * result.element_size() != LOCAL_SCALE_VECTOR_BYTES:
        raise AssertionError("local scale byte accounting drifted")
    return result


def _validate_packed_source(raw: PackedMXFP4Matrix, matrix: str) -> None:
    if matrix not in C.EXPERT_MATRICES:
        raise ValueError(f"unknown expert matrix: {matrix}")
    out_features, in_features = C.EXPERT_SHAPES[matrix]
    expected_packed = (out_features, in_features // 2)
    expected_scale = (out_features, in_features // C.MXFP4_BLOCK)
    if raw.packed.dtype != torch.uint8 or tuple(raw.packed.shape) != expected_packed:
        raise ValueError(
            f"{matrix}.weight_packed must be torch.uint8 {expected_packed}"
        )
    if raw.scale.dtype != torch.uint8 or tuple(raw.scale.shape) != expected_scale:
        raise ValueError(
            f"{matrix}.weight_scale must be torch.uint8 {expected_scale}"
        )
    if raw.packed.device.type != "cpu" or raw.scale.device.type != "cpu":
        raise ValueError("official packed source tensors must be on CPU")


def mxfp4_rank_matrix_components(
    raw: PackedMXFP4Matrix,
    matrix: str,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard one official matrix as packed codes followed by E8M0 scales."""

    _validate_packed_source(raw, matrix)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < TP_SIZE:
        raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
    width = INTERMEDIATE_CHANNELS // TP_SIZE
    if matrix in ("w1", "w3"):
        packed = raw.packed.narrow(0, rank * width, width).contiguous()
        scale = raw.scale.narrow(0, rank * width, width).contiguous()
    else:
        packed_width = width // 2
        scale_width = width // C.MXFP4_BLOCK
        packed = raw.packed.narrow(1, rank * packed_width, packed_width).contiguous()
        scale = raw.scale.narrow(1, rank * scale_width, scale_width).contiguous()
    expected_packed_bytes = RANK_MATRIX_WEIGHTS // 2
    expected_scale_bytes = RANK_MATRIX_WEIGHTS // C.MXFP4_BLOCK
    if packed.numel() != expected_packed_bytes or scale.numel() != expected_scale_bytes:
        raise AssertionError("TP12 MXFP4 shard byte accounting drifted")
    if packed.numel() + scale.numel() != RANK_MXFP4_MATRIX_BYTES:
        raise AssertionError("TP12 MXFP4 matrix byte accounting drifted")
    return packed, scale


def _byte_view(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu":
        raise ValueError("only CPU tensors can be written to an artifact")
    if not tensor.is_contiguous():
        raise ValueError("artifact tensors must be contiguous")
    return memoryview(tensor.numpy()).cast("B")


def pwrite_exact(descriptor: int, payload: bytes | bytearray | memoryview, offset: int) -> None:
    """Write a complete byte buffer at an explicit offset, handling short writes."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    view = memoryview(payload).cast("B")
    cursor = 0
    while cursor < len(view):
        written = os.pwrite(descriptor, view[cursor:], offset + cursor)
        if written <= 0:
            raise OSError("pwrite made no forward progress")
        cursor += written


def _pread_exact(descriptor: int, count: int, offset: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        piece = os.pread(descriptor, count - len(result), offset + len(result))
        if not piece:
            raise ValueError("artifact ended before the expected offset")
        result.extend(piece)
    return bytes(result)


def _discard_stale_partial(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"materialization partial is still owned by a live writer: {path}"
            ) from exc
        path.unlink()
    finally:
        os.close(descriptor)


def _tensor_from_bytes(
    payload: bytes | bytearray,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape).clone()
    if not value.is_contiguous():
        value = value.contiguous()
    return value


class TP12LayerReader:
    """Bounded-memory reference reader for one materialized TP12 layer slab."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            self.header = TP12LayerHeader.from_bytes(
                _pread_exact(descriptor, LAYER_HEADER_BYTES, 0)
            )
            if self.path.stat().st_size != self.header.layout.disk_bytes:
                raise ValueError("materialized layer size disagrees with its header")
            format_section = torch.frombuffer(
                bytearray(
                    _pread_exact(
                        descriptor,
                        FORMAT_SECTION_BYTES,
                        self.header.format_offset,
                    )
                ),
                dtype=torch.uint8,
            )
            unpack_tp12_format_section(format_section)
            self.formats = tuple(
                ExpertFormatSpec.from_code(code)
                for code in format_section[:FORMAT_TABLE_BYTES].tolist()
            )
            if TP12LayerLayout.from_formats(
                self.formats,
                keep_storage=self.header.layout.keep_storage,
            ) != self.header.layout:
                raise ValueError("format table and layer-header tier counts disagree")
            slots: list[int] = []
            compressed_slot = 0
            kept_slot = 0
            for format_spec in self.formats:
                if format_spec.is_mxfp4:
                    slots.append(kept_slot)
                    kept_slot += 1
                else:
                    slots.append(compressed_slot)
                    compressed_slot += 1
            self.slots = tuple(slots)
            self.compressed_experts = tuple(
                expert
                for expert, format_spec in enumerate(self.formats)
                if not format_spec.is_mxfp4
            )
            self.kept_experts = tuple(
                expert
                for expert, format_spec in enumerate(self.formats)
                if format_spec.is_mxfp4
            )
            shared_section = torch.frombuffer(
                bytearray(
                    _pread_exact(
                        descriptor,
                        SHARED_SCALE_SECTION_BYTES,
                        self.header.shared_scale_offset,
                    )
                ),
                dtype=torch.uint8,
            )
            self.shared_scales = unpack_tp12_shared_scale_section(shared_section)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor: int | None = descriptor

    @property
    def layout(self) -> TP12LayerLayout:
        return self.header.layout

    def close(self) -> None:
        if self._descriptor is not None:
            descriptor = self._descriptor
            self._descriptor = None
            os.close(descriptor)

    def __enter__(self) -> "TP12LayerReader":
        if self._descriptor is None:
            raise RuntimeError("materialized TP12 layer reader is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def _pread(self, count: int, offset: int) -> bytes:
        if self._descriptor is None:
            raise RuntimeError("materialized TP12 layer reader is closed")
        return _pread_exact(self._descriptor, count, offset)

    def _slot(self, expert: int, *, kept: bool) -> int:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError("expert must be an integer")
        if not 0 <= expert < EXPERTS_PER_LAYER:
            raise ValueError(f"expert must be in 0..{EXPERTS_PER_LAYER - 1}")
        actual_kept = self.formats[expert].is_mxfp4
        if actual_kept != kept:
            expected = "MXFP4" if kept else "compressed"
            raise ValueError(f"expert {expert} is not in the {expected} tier")
        return self.slots[expert]

    def read_compressed_rank(self, rank: int) -> TP12CompressedRankPayload:
        """Read one rank directly into the bounded B12X pair-decoder layout."""

        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < TP_SIZE:
            raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
        count = self.layout.compressed_experts
        words = PAIR_BYTES // torch.int16.itemsize
        w13 = torch.empty((2, count, words), dtype=torch.int16)
        w2 = torch.empty((count, words), dtype=torch.int16)
        rank_offset = self.layout.rank_offset(rank)
        for slot in range(count):
            raw = _tensor_from_bytes(
                self._pread(
                    EXPERT_RANK_TRELLIS_BYTES,
                    rank_offset + slot * EXPERT_RANK_TRELLIS_BYTES,
                ),
                dtype=torch.int16,
                shape=(3, words),
            )
            w13[:, slot].copy_(raw[:2])
            w2[slot].copy_(raw[2])

        scale_section = torch.frombuffer(
            bytearray(
                self._pread(
                    self.layout.rank_scale_section_bytes,
                    self.layout.rank_scale_offset(rank),
                )
            ),
            dtype=torch.uint8,
        )
        local_scales = unpack_tp12_rank_scale_section(scale_section, self.layout)
        intermediate_rotations = local_scales.reshape(count, -1).contiguous()
        expert_ids = torch.tensor(self.compressed_experts, dtype=torch.int32)
        r13 = torch.tensor(
            [self.formats[expert].r13 for expert in self.compressed_experts],
            dtype=torch.int32,
        )
        r2 = torch.tensor(
            [self.formats[expert].r2 for expert in self.compressed_experts],
            dtype=torch.int32,
        )
        logical_pairs = torch.tensor(
            [
                tp12_logical_pair_index(self.header.layer, expert, rank)
                for expert in self.compressed_experts
            ],
            dtype=torch.int32,
        )
        return TP12CompressedRankPayload(
            rank=rank,
            expert_ids=expert_ids,
            w13_trellis=w13,
            w2_trellis=w2,
            gate_suh=self.shared_scales[0].reshape(1, -1).clone(),
            up_suh=self.shared_scales[1].reshape(1, -1).clone(),
            intermediate_rotations=intermediate_rotations,
            down_svh=self.shared_scales[2].reshape(1, -1).clone(),
            fc1_pair_modes=(r13 > logical_pairs).to(torch.int32).contiguous(),
            fc2_pair_modes=(r2 > logical_pairs).to(torch.int32).contiguous(),
        )

    def read_compressed_matrix(
        self, expert: int, matrix: str
    ) -> dict[str, torch.Tensor]:
        """Reassemble one logical candidate payload from its twelve rank slices."""

        if matrix not in C.EXPERT_MATRICES:
            raise ValueError(f"unknown expert matrix: {matrix}")
        slot = self._slot(expert, kept=False)
        trellis_raw = bytearray(MATRIX_TRELLIS_BYTES)
        local_raw = bytearray(INTERMEDIATE_CHANNELS * torch.float16.itemsize)
        for rank in range(TP_SIZE):
            logical_pair = tp12_logical_pair_index(
                self.header.layer,
                expert,
                rank,
            )
            pair_begin = logical_pair * PAIR_BYTES
            trellis_raw[pair_begin : pair_begin + PAIR_BYTES] = self._pread(
                PAIR_BYTES,
                self.layout.trellis_pair_offset(rank, slot, matrix),
            )
            local_begin = logical_pair * LOCAL_SCALE_VECTOR_BYTES
            local_raw[local_begin : local_begin + LOCAL_SCALE_VECTOR_BYTES] = self._pread(
                LOCAL_SCALE_VECTOR_BYTES,
                self.layout.local_scale_offset(rank, slot, matrix),
            )
        trellis = _tensor_from_bytes(
            trellis_raw,
            dtype=torch.int16,
            shape=(MATRIX_TRELLIS_BYTES // torch.int16.itemsize,),
        )
        local = _tensor_from_bytes(
            local_raw,
            dtype=torch.float16,
            shape=(INTERMEDIATE_CHANNELS,),
        )
        shared_index = {"w1": 0, "w3": 1, "w2": 2}[matrix]
        shared = self.shared_scales[shared_index].clone()
        if matrix in ("w1", "w3"):
            parts = {"trellis": trellis, "suh": shared, "svh": local}
        else:
            parts = {"trellis": trellis, "suh": local, "svh": shared}
        for part, value in parts.items():
            _validate_candidate_tensor(value, matrix=matrix, part=part)
        return parts

    def read_kept_matrix(self, expert: int, matrix: str) -> PackedMXFP4Matrix:
        """Reassemble one official MXFP4 matrix from codes-then-scales slices."""

        if self.layout.keep_storage != KEEP_STORAGE_INLINE_MXFP4:
            raise ValueError("external X4 matrices must be read from their sidecar")
        if matrix not in C.EXPERT_MATRICES:
            raise ValueError(f"unknown expert matrix: {matrix}")
        slot = self._slot(expert, kept=True)
        packed_chunks: list[torch.Tensor] = []
        scale_chunks: list[torch.Tensor] = []
        packed_bytes = RANK_MATRIX_WEIGHTS // 2
        scale_bytes = RANK_MATRIX_WEIGHTS // C.MXFP4_BLOCK
        local_width = INTERMEDIATE_CHANNELS // TP_SIZE
        if matrix in ("w1", "w3"):
            packed_shape = (local_width, LATENT_CHANNELS // 2)
            scale_shape = (local_width, LATENT_CHANNELS // C.MXFP4_BLOCK)
            concatenate_dimension = 0
        else:
            packed_shape = (LATENT_CHANNELS, local_width // 2)
            scale_shape = (LATENT_CHANNELS, local_width // C.MXFP4_BLOCK)
            concatenate_dimension = 1
        for rank in range(TP_SIZE):
            payload = self._pread(
                RANK_MXFP4_MATRIX_BYTES,
                self.layout.kept_matrix_offset(rank, slot, matrix),
            )
            packed_chunks.append(
                _tensor_from_bytes(
                    payload[:packed_bytes],
                    dtype=torch.uint8,
                    shape=packed_shape,
                )
            )
            scale_chunks.append(
                _tensor_from_bytes(
                    payload[packed_bytes : packed_bytes + scale_bytes],
                    dtype=torch.uint8,
                    shape=scale_shape,
                )
            )
        result = PackedMXFP4Matrix(
            packed=torch.cat(packed_chunks, dim=concatenate_dimension).contiguous(),
            scale=torch.cat(scale_chunks, dim=concatenate_dimension).contiguous(),
        )
        _validate_packed_source(result, matrix)
        return result


def layer_filename(layer: int) -> str:
    if layer not in C.MOE_LAYERS:
        raise ValueError("K3 MoE layer must be in 1..92")
    return f"{MATERIALIZED_LAYER_PREFIX}{layer:05d}.bin"


def layer_closure_filename(layer: int) -> str:
    return f"{layer_filename(layer)}{MATERIALIZED_CLOSURE_SUFFIX}"


def expected_logical_source_bytes(spec: LayerMaterializationSpec) -> int:
    """Bytes compared while proving one slab against both source tiers."""

    compressed_matrix_bytes = (
        EXPERT_TRELLIS_BYTES
        + 3 * (LATENT_CHANNELS + INTERMEDIATE_CHANNELS) * SCALE_BYTES
    )
    kept_expert_bytes = TP_SIZE * EXPERT_RANK_MXFP4_BYTES
    return (
        len(spec.compressed) * compressed_matrix_bytes
        + len(spec.kept) * kept_expert_bytes
    )


def _canonical_json_sha256(document: dict) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _layer_file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "file": path.name,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _validate_full_closure_metadata(
    metadata: object,
    spec: LayerMaterializationSpec,
    structural: dict[str, int | str],
) -> dict[str, int | str]:
    expected = {
        **structural,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(spec.compressed),
        "kept_experts_verified": len(spec.kept),
        "logical_source_bytes_verified": expected_logical_source_bytes(spec),
    }
    if metadata != expected:
        raise ValueError(
            f"layer {spec.layer} closure metadata does not describe a full "
            "bit-exact source comparison"
        )
    return expected


def write_layer_closure_receipt(
    path: str | Path,
    spec: LayerMaterializationSpec,
    metadata: dict[str, int | str],
    *,
    build_document: dict,
) -> Path:
    """Persist a crash-safe receipt for the source comparison done at write time."""

    path = Path(path)
    structural = validate_materialized_layer(path, spec)
    closure = _validate_full_closure_metadata(metadata, spec, structural)
    receipt = {
        "kind": MATERIALIZED_CLOSURE_KIND,
        "schema_version": MATERIALIZED_CLOSURE_SCHEMA_VERSION,
        "layer": spec.layer,
        "build_sha256": _canonical_json_sha256(build_document),
        "slab_identity": _layer_file_identity(path),
        "closure": closure,
    }
    receipt_path = path.with_name(layer_closure_filename(spec.layer))
    _atomic_json(receipt_path, receipt)
    return receipt_path


def load_layer_closure_receipt(
    path: str | Path,
    spec: LayerMaterializationSpec,
    *,
    build_document: dict,
) -> dict[str, int | str]:
    """Load a receipt only when it still names this exact immutable slab."""

    path = Path(path)
    receipt_path = path.with_name(layer_closure_filename(spec.layer))
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = _read_json(receipt_path)
    expected_scalars = {
        "kind": MATERIALIZED_CLOSURE_KIND,
        "schema_version": MATERIALIZED_CLOSURE_SCHEMA_VERSION,
        "layer": spec.layer,
        "build_sha256": _canonical_json_sha256(build_document),
        "slab_identity": _layer_file_identity(path),
    }
    for name, expected in expected_scalars.items():
        if receipt.get(name) != expected:
            raise ValueError(
                f"layer {spec.layer} closure receipt {name} no longer matches"
            )
    structural = validate_materialized_layer(path, spec)
    return _validate_full_closure_metadata(
        receipt.get("closure"),
        spec,
        structural,
    )


def candidate_layer_path(root: Path, layer: int) -> Path:
    return root / "candidates" / f"mixed-exl3-layer-{layer:05d}.safetensors"


def validate_materialized_layer(
    path: str | Path,
    spec: LayerMaterializationSpec,
) -> dict[str, int | str]:
    """Validate fixed sections, exact size, and every alignment padding range."""

    path = Path(path)
    stat = path.stat()
    if stat.st_size != spec.layout.disk_bytes:
        raise ValueError(
            f"{path} has {stat.st_size} bytes, expected {spec.layout.disk_bytes}"
        )
    descriptor = os.open(path, os.O_RDONLY)
    try:
        header = TP12LayerHeader.from_bytes(
            _pread_exact(descriptor, LAYER_HEADER_BYTES, 0)
        )
        if (
            header.layer != spec.layer
            or header.layout != spec.layout
            or header.codebook != spec.codebook
        ):
            raise ValueError("materialized layer header disagrees with its allocation")
        format_section = torch.frombuffer(
            bytearray(
                _pread_exact(
                    descriptor,
                    FORMAT_SECTION_BYTES,
                    header.format_offset,
                )
            ),
            dtype=torch.uint8,
        )
        if unpack_tp12_format_section(format_section) != tuple(
            value.name for value in spec.formats
        ):
            raise ValueError("materialized format table disagrees with its allocation")
        shared_section = torch.frombuffer(
            bytearray(
                _pread_exact(
                    descriptor,
                    SHARED_SCALE_SECTION_BYTES,
                    header.shared_scale_offset,
                )
            ),
            dtype=torch.uint8,
        )
        unpack_tp12_shared_scale_section(shared_section)
        for rank in range(TP_SIZE):
            padding = spec.layout.rank_scale_padding_bytes
            if not padding:
                continue
            padding_offset = (
                spec.layout.rank_scale_offset(rank)
                + spec.layout.rank_scale_payload_bytes
            )
            if any(_pread_exact(descriptor, padding, padding_offset)):
                raise ValueError(f"rank {rank} has nonzero local-scale padding")
    finally:
        os.close(descriptor)
    return {
        "file": path.name,
        "disk_bytes": spec.layout.disk_bytes,
        "compressed_experts": len(spec.compressed),
        "kept_experts": len(spec.kept),
        "codebook": spec.codebook,
    }


def _requested_tier(
    requested: Sequence[int] | None,
    available: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    if requested is None:
        return available
    values = tuple(requested)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError(f"{name} expert IDs must be integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} expert IDs must be unique")
    if not set(values).issubset(available):
        raise ValueError(f"{name} payload closure requested an expert outside its tier")
    return values


def _require_bit_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    if torch.equal(actual, expected):
        return
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"{name} geometry changed in the materialized artifact: "
            f"{actual.dtype} {tuple(actual.shape)} != "
            f"{expected.dtype} {tuple(expected.shape)}"
        )
    mismatched = int(torch.count_nonzero(actual != expected))
    raise ValueError(
        f"{name} is not bit-exact in the materialized artifact "
        f"({mismatched}/{actual.numel()} values differ)"
    )


def validate_materialized_layer_payloads(
    path: str | Path,
    spec: LayerMaterializationSpec,
    candidate_root: str | Path,
    store: OfficialMXFP4Store,
    *,
    compressed_experts: Sequence[int] | None = None,
    kept_experts: Sequence[int] | None = None,
    x4_path: str | Path | None = None,
) -> dict[str, int | str]:
    """Prove candidate and official-source bytes survived slab placement exactly."""

    metadata = validate_materialized_layer(path, spec)
    compressed = _requested_tier(
        compressed_experts, spec.compressed, name="compressed"
    )
    kept = _requested_tier(kept_experts, spec.kept, name="kept")
    logical_bytes = 0
    x4_reader: X4LayerReader | None = None
    if spec.layout.keep_storage == KEEP_STORAGE_EXTERNAL_X4:
        if x4_path is None:
            raise ValueError("external-X4 closure requires its layer sidecar")
        x4_reader = X4LayerReader(x4_path)
        if x4_reader.layer != spec.layer:
            raise ValueError("X4 sidecar layer disagrees with its trellis slab")
        if x4_reader.record_count != len(spec.kept) * len(C.EXPERT_MATRICES):
            raise ValueError("X4 sidecar record count disagrees with its allocation")
    elif x4_path is not None:
        raise ValueError("inline-MXFP4 closure must not supply an X4 sidecar")
    with TP12LayerReader(path) as reader:
        if reader.header.layer != spec.layer or reader.formats != spec.formats:
            raise ValueError("materialized reader disagrees with the allocation spec")
        if compressed:
            candidate_path = candidate_layer_path(Path(candidate_root), spec.layer)
            with safe_open(candidate_path, framework="pt", device="cpu") as handle:
                for expert in compressed:
                    for matrix in C.EXPERT_MATRICES:
                        actual = reader.read_compressed_matrix(expert, matrix)
                        for part in ("trellis", "suh", "svh"):
                            name = candidate_tensor_name(
                                spec.layer, expert, matrix, part
                            )
                            try:
                                expected = _validate_candidate_tensor(
                                    handle.get_tensor(name),
                                    matrix=matrix,
                                    part=part,
                                )
                            except KeyError as exc:
                                raise ValueError(
                                    f"candidate layer {spec.layer} expert {expert} "
                                    f"is missing {matrix}.{part}"
                                ) from exc
                            _require_bit_exact(
                                actual[part],
                                expected,
                                name=(
                                    f"layer {spec.layer} expert {expert} "
                                    f"{matrix}.{part}"
                                ),
                            )
                            logical_bytes += expected.numel() * expected.element_size()
        for expert in kept:
            for matrix in C.EXPERT_MATRICES:
                if x4_reader is None:
                    actual = reader.read_kept_matrix(expert, matrix)
                else:
                    x4_record = x4_reader.read(expert, matrix)
                    actual = PackedMXFP4Matrix(
                        packed=x4_record.packed,
                        scale=x4_record.scale,
                    )
                expected = store.load_packed_matrix(spec.layer, expert, matrix)
                for part in MXFP4_MATRIX_COMPONENT_ORDER:
                    actual_part = (
                        actual.packed if part == "weight_packed" else actual.scale
                    )
                    expected_part = (
                        expected.packed if part == "weight_packed" else expected.scale
                    )
                    _require_bit_exact(
                        actual_part,
                        expected_part,
                        name=f"layer {spec.layer} expert {expert} {matrix}.{part}",
                    )
                    logical_bytes += expected_part.numel() * expected_part.element_size()
    return {
        **metadata,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(compressed),
        "kept_experts_verified": len(kept),
        "logical_source_bytes_verified": logical_bytes,
    }


def _write_candidate_expert(
    descriptor: int,
    handle,
    spec: LayerMaterializationSpec,
    *,
    expert: int,
    compressed_slot: int,
    shared_reference: dict[str, torch.Tensor],
) -> None:
    shared_part = {"w1": "suh", "w3": "suh", "w2": "svh"}
    local_part = {"w1": "svh", "w3": "svh", "w2": "suh"}
    for matrix in C.EXPERT_MATRICES:
        names = {
            part: candidate_tensor_name(spec.layer, expert, matrix, part)
            for part in ("trellis", "suh", "svh")
        }
        try:
            tensors = {
                part: _validate_candidate_tensor(
                    handle.get_tensor(name), matrix=matrix, part=part
                )
                for part, name in names.items()
            }
        except KeyError as exc:
            raise ValueError(
                f"candidate layer {spec.layer} expert {expert} is incomplete"
            ) from exc

        shared_key = f"{matrix}.{shared_part[matrix]}"
        shared = tensors[shared_part[matrix]]
        reference = shared_reference.get(shared_key)
        if reference is None:
            shared_reference[shared_key] = shared.clone()
        elif not torch.equal(shared, reference):
            difference = (shared.float() - reference.float()).abs()
            raise ValueError(
                f"layer {spec.layer} expert {expert} {shared_key} is not bit-exact "
                f"with the layer-shared transform (max_abs={float(difference.max())})"
            )

        for rank in range(TP_SIZE):
            logical_pair = tp12_logical_pair_index(
                spec.layer,
                expert,
                rank,
            )
            pair = candidate_trellis_pair(tensors["trellis"], logical_pair)
            pwrite_exact(
                descriptor,
                _byte_view(pair),
                spec.layout.trellis_pair_offset(rank, compressed_slot, matrix),
            )
            local = candidate_local_scale(
                tensors[local_part[matrix]],
                logical_pair,
            )
            pwrite_exact(
                descriptor,
                _byte_view(local),
                spec.layout.local_scale_offset(rank, compressed_slot, matrix),
            )


def _write_kept_expert(
    descriptor: int,
    store: OfficialMXFP4Store,
    spec: LayerMaterializationSpec,
    *,
    expert: int,
    kept_slot: int,
) -> None:
    for matrix in C.EXPERT_MATRICES:
        raw = store.load_packed_matrix(spec.layer, expert, matrix)
        for rank in range(TP_SIZE):
            packed, scale = mxfp4_rank_matrix_components(raw, matrix, rank)
            offset = spec.layout.kept_matrix_offset(rank, kept_slot, matrix)
            pwrite_exact(descriptor, _byte_view(packed), offset)
            pwrite_exact(
                descriptor,
                _byte_view(scale),
                offset + packed.numel(),
            )


def materialize_layer(
    candidate_root: str | Path,
    store: OfficialMXFP4Store,
    destination: str | Path,
    spec: LayerMaterializationSpec,
    *,
    discard_partial: bool = False,
    preallocate: bool = True,
    x4_destination: str | Path | None = None,
) -> dict[str, int | str]:
    """Write one layer atomically with bounded source and candidate residency."""

    candidate_root = Path(candidate_root).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    external_x4 = spec.layout.keep_storage == KEEP_STORAGE_EXTERNAL_X4
    if external_x4 and x4_destination is None:
        raise ValueError("SQRT-C materialization requires an X4 sidecar destination")
    if not external_x4 and x4_destination is not None:
        raise ValueError("inline-MXFP4 materialization cannot write an X4 sidecar")
    x4_path = None if x4_destination is None else Path(x4_destination)
    if x4_path is not None and x4_path.exists():
        raise FileExistsError(x4_path)
    candidate_path = candidate_layer_path(candidate_root, spec.layer)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        if not discard_partial:
            raise FileExistsError(
                f"incomplete materialization exists: {partial}; resume with discard_partial"
            )
        _discard_stale_partial(partial)

    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    x4_writer: X4LayerWriter | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if preallocate and hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, spec.layout.disk_bytes)
        else:
            os.ftruncate(descriptor, spec.layout.disk_bytes)
        header = TP12LayerHeader(spec.layer, spec.layout, codebook=spec.codebook)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        format_section = pack_tp12_format_section(spec.formats)
        pwrite_exact(descriptor, _byte_view(format_section), header.format_offset)

        shared_reference: dict[str, torch.Tensor] = {}
        if spec.compressed:
            with safe_open(candidate_path, framework="pt", device="cpu") as handle:
                for slot, expert in enumerate(spec.compressed):
                    _write_candidate_expert(
                        descriptor,
                        handle,
                        spec,
                        expert=expert,
                        compressed_slot=slot,
                        shared_reference=shared_reference,
                    )
            expected_shared = {"w1.suh", "w3.suh", "w2.svh"}
            if set(shared_reference) != expected_shared:
                raise AssertionError("layer-shared scale collection did not close")
            shared_section = pack_tp12_shared_scale_section(
                shared_reference["w1.suh"],
                shared_reference["w3.suh"],
                shared_reference["w2.svh"],
            )
        else:
            shared_section = torch.zeros(
                SHARED_SCALE_SECTION_BYTES, dtype=torch.uint8
            )
        pwrite_exact(
            descriptor,
            _byte_view(shared_section),
            header.shared_scale_offset,
        )

        if external_x4:
            assert x4_path is not None
            x4_writer = X4LayerWriter(x4_path, layer=spec.layer)
            for expert in spec.kept:
                for matrix in C.EXPERT_MATRICES:
                    raw = store.load_packed_matrix(spec.layer, expert, matrix)
                    x4_writer.add(expert, matrix, raw.packed, raw.scale)
            x4_writer.close()
        else:
            for slot, expert in enumerate(spec.kept):
                _write_kept_expert(
                    descriptor,
                    store,
                    spec,
                    expert=expert,
                    kept_slot=slot,
                )
        os.fsync(descriptor)
        metadata = validate_materialized_layer_payloads(
            partial,
            spec,
            candidate_root,
            store,
            x4_path=x4_path,
        )
    except BaseException:
        if x4_writer is not None:
            x4_writer.abort()
        if x4_path is not None:
            x4_path.unlink(missing_ok=True)
        os.close(descriptor)
        partial.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    partial.replace(destination)
    _fsync_directory(destination.parent)
    return metadata


def materialization_build_document(
    *,
    pool: MixedCandidatePool,
    allocation_path: Path,
    mode_validation_summary_path: Path,
    teacher_proxy_summary_path: Path,
    performance_summary_path: Path,
    official_root: Path,
    official_revision: str,
    plan: MaterializationPlan,
) -> dict:
    manifest_path = pool.root / "mixed_exl3_candidate_manifest.json"
    if pool.content_sha256 is None:
        raise ValueError("materialization requires a sealed candidate pool")
    if pool.trellis_schema not in LOGICAL_CANDIDATE_SCHEMAS:
        raise ValueError(
            "materialization requires an explicit supported logical candidate schema"
        )
    allocation = _read_json(allocation_path)
    allocation_meta = allocation.get("meta")
    if not isinstance(allocation_meta, dict):
        raise ValueError("allocation is missing its meta object")
    mode_validation_summary_path = mode_validation_summary_path.resolve()
    teacher_proxy_summary_path = teacher_proxy_summary_path.resolve()
    performance_summary_path = performance_summary_path.resolve()
    mode_validation_gate = validate_materialization_mode_gate(
        mode_validation_summary_path,
        allocation,
        pool,
    )
    teacher_proxy_gate = validate_materialization_teacher_proxy_gate(
        teacher_proxy_summary_path
    )
    performance_gate = validate_materialization_performance_gate(
        performance_summary_path,
        pool,
    )
    return {
        "kind": MATERIALIZATION_BUILD_KIND,
        "schema_version": MATERIALIZATION_BUILD_SCHEMA_VERSION,
        "format_schema": SCHEMA,
        "tp_size": TP_SIZE,
        "source_model": C.MODEL_ID,
        "source_revision": official_revision,
        "official_checkpoint": str(official_root.resolve()),
        "candidate_pool": str(pool.root),
        "candidate_manifest_sha256": _sha256(manifest_path),
        "candidate_pool_content_sha256": pool.content_sha256,
        "candidate_logical_trellis_schema": pool.trellis_schema,
        "candidate_codebook": pool.codebook,
        "candidate_mode_ids": list(pool.mode_ids),
        "allocation_source": str(allocation_path.resolve()),
        "allocation_sha256": _sha256(allocation_path),
        "allocation_damage_metric": allocation_meta.get("damage_metric"),
        "allocation_damage_provenance": allocation_meta.get(
            "damage_provenance"
        ),
        "mode_validation_summary_source": str(mode_validation_summary_path),
        "mode_validation_summary_sha256": _sha256(
            mode_validation_summary_path
        ),
        "mode_validation_gate": mode_validation_gate,
        "teacher_proxy_summary_source": str(teacher_proxy_summary_path),
        "teacher_proxy_summary_sha256": _sha256(teacher_proxy_summary_path),
        "teacher_proxy_gate": teacher_proxy_gate,
        "performance_summary_source": str(performance_summary_path),
        "performance_summary_sha256": _sha256(performance_summary_path),
        "performance_gate": performance_gate,
        "container_bytes": plan.total_container_bytes,
        "retained_experts": plan.retained_experts,
        "compressed_experts": plan.compressed_experts,
        "layer_count": len(plan.layers),
        "mxfp4_matrix_component_order": list(MXFP4_MATRIX_COMPONENT_ORDER),
        "byte_order": "little",
    }


def prepare_destination(
    destination: str | Path,
    *,
    build_document: dict,
    allocation_path: Path,
    mode_validation_summary_path: Path,
    teacher_proxy_summary_path: Path,
    performance_summary_path: Path,
    resume: bool,
) -> Path:
    """Create or strictly reopen a materialization destination."""

    destination = Path(destination).resolve()
    build_path = destination / MATERIALIZATION_BUILD_FILENAME
    allocation_copy = destination / MATERIALIZED_ALLOCATION_FILENAME
    mode_summary_copy = (
        destination / MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME
    )
    teacher_proxy_copy = (
        destination / MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME
    )
    performance_copy = destination / MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME
    if destination.exists():
        if not resume:
            raise FileExistsError(
                f"destination {destination} exists; use --resume only for "
                "the identical build"
            )
        if not build_path.is_file() or _read_json(build_path) != build_document:
            raise ValueError("resume destination build contract does not match")
        if (
            not allocation_copy.is_file()
            or allocation_copy.read_bytes() != allocation_path.read_bytes()
        ):
            raise ValueError("resume destination allocation copy does not match")
        if (
            not mode_summary_copy.is_file()
            or mode_summary_copy.read_bytes()
            != mode_validation_summary_path.read_bytes()
        ):
            raise ValueError(
                "resume destination mode-validation summary does not match"
            )
        if (
            not teacher_proxy_copy.is_file()
            or teacher_proxy_copy.read_bytes()
            != teacher_proxy_summary_path.read_bytes()
        ):
            raise ValueError(
                "resume destination teacher-proxy summary does not match"
            )
        if (
            not performance_copy.is_file()
            or performance_copy.read_bytes() != performance_summary_path.read_bytes()
        ):
            raise ValueError(
                "resume destination TP12 performance summary does not match"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        staging.mkdir()
        try:
            _atomic_json(
                staging / MATERIALIZATION_BUILD_FILENAME,
                build_document,
            )
            _atomic_bytes(
                staging / MATERIALIZED_ALLOCATION_FILENAME,
                allocation_path.read_bytes(),
            )
            _atomic_bytes(
                staging / MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME,
                mode_validation_summary_path.read_bytes(),
            )
            _atomic_bytes(
                staging / MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
                teacher_proxy_summary_path.read_bytes(),
            )
            _atomic_bytes(
                staging / MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
                performance_summary_path.read_bytes(),
            )
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


def artifact_manifest(
    *,
    build_document: dict,
    plan: MaterializationPlan,
    layers: Sequence[dict[str, int | str]],
) -> dict:
    if len(layers) != len(plan.layers):
        raise ValueError("a final manifest requires every materialized layer")
    by_layer = {
        str(spec.layer): {
            **metadata,
            "layout": spec.layout.to_manifest(),
        }
        for spec, metadata in zip(plan.layers, layers, strict=True)
    }
    if (
        sum(int(value["disk_bytes"]) for value in by_layer.values())
        != plan.total_container_bytes
    ):
        raise AssertionError("final layer byte ledger does not close")
    return {
        "kind": MATERIALIZED_ARTIFACT_KIND,
        "schema_version": MATERIALIZED_ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        **{
            name: build_document[name]
            for name in (
                "format_schema",
                "tp_size",
                "source_model",
                "source_revision",
                "official_checkpoint",
                "candidate_pool",
                "candidate_manifest_sha256",
                "candidate_pool_content_sha256",
                "candidate_logical_trellis_schema",
                "candidate_codebook",
                "candidate_mode_ids",
                "allocation_source",
                "allocation_sha256",
                "allocation_damage_metric",
                "allocation_damage_provenance",
                "mode_validation_summary_source",
                "mode_validation_summary_sha256",
                "mode_validation_gate",
                "teacher_proxy_summary_source",
                "teacher_proxy_summary_sha256",
                "teacher_proxy_gate",
                "performance_summary_source",
                "performance_summary_sha256",
                "performance_gate",
                "container_bytes",
                "retained_experts",
                "compressed_experts",
                "mxfp4_matrix_component_order",
                "byte_order",
            )
        },
        "allocation": MATERIALIZED_ALLOCATION_FILENAME,
        "mode_validation_summary": (
            MATERIALIZED_MODE_VALIDATION_SUMMARY_FILENAME
        ),
        "teacher_proxy_summary": MATERIALIZED_TEACHER_PROXY_SUMMARY_FILENAME,
        "performance_summary": MATERIALIZED_PERFORMANCE_SUMMARY_FILENAME,
        "layers": by_layer,
    }


def write_artifact_manifest(destination: Path, document: dict) -> Path:
    path = destination / MATERIALIZED_MANIFEST_FILENAME
    if path.exists():
        if _read_json(path) != document:
            raise ValueError(
                "existing final artifact manifest disagrees with the build"
            )
        return path
    _atomic_json(path, document)
    return path
