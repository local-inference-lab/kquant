"""Production-facing heterogeneous Kimi-K3 QSRT encoder wrapper.

The kquant encoder backend owns the dense-H LDLQ traversal. This module owns
the Kimi-K3 semantics around it: one common intermediate-neuron order, the
logical rate axis of each expert matrix, complete-record repacking into the
balanced P24/P33 layout, and closure against the stored FP16 scale vectors.

The returned reconstruction is in the source checkpoint's canonical neuron
order.  The returned tensors are in physical serving order and contain one
fixed-rate trellis payload plus ``suh``/``svh``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import torch

from kquant.exl3_reference import (
    CODEBOOK_QSRT_E4M3,
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3,
    CODEBOOK_SQG_NORMAL_E4M3,
    QSRT_CODEBOOKS,
    decode_qsrt_weight,
)
from kquant.sqg_e4m3 import (
    qsrt_e4m3_bytes,
    sqg_cheb_normal_e4m3_bytes,
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_k2_eight_stratum_e4m3_bytes_from_rank_lut,
)

from kquant.qsrt import (
    CONTEXT_GROUP_CHANNELS,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    ModeSpec,
    ExpertFormatSpec,
    PackedQSRTTrellis,
    RECORD_CHANNELS,
    SCHEMA,
    RateAxis,
    decode_qsrt_exl3_weight,
    expand_group_order,
    importance_group_order,
    matrix_rate_axis,
    pack_qsrt_trellis,
    record_repack_order,
    repack_encoded_records,
    repack_rate_axis,
    resolve_mode,
    record_bits,
    storage_group_order,
    unpack_qsrt_trellis_edges,
    unpack_qsrt_trellis_states,
)
from kquant.ldlq import SIGMA_REG, make_shared_h
from kquant.tp_simulator import comparison_metrics


MATRICES = ("w1", "w3", "w2")
Layout = Literal["importance_ordered", "qsrt_pair_prefix_reuse"]
MIXED_LAYOUTS: tuple[Layout, ...] = (
    "importance_ordered",
    "qsrt_pair_prefix_reuse",
)

# Candidate-search policy rather than one matrix traversal.  The guarded
# policy keeps the established importance-ordered traversal for w1/w3 and for
# the R0 w2 control, while encoding the R1/R2 w2 counterfactuals in the
# branchable prefix layout.  Thus the selector can never accept a layout-
# induced regression as its baseline merely to obtain encoder throughput.
SearchLayout = Literal["importance_ordered", "qsrt_guarded_reuse"]
MIXED_SEARCH_LAYOUTS: tuple[SearchLayout, ...] = (
    "importance_ordered",
    "qsrt_guarded_reuse",
)

# R0--R2 differ only in the two least-important donor records and the two
# most-important recipient records. Keeping those four records at the low-K
# end of the offline LDLQ traversal leaves a 20-record K3 suffix. LDLQ walks K
# backwards, so the suffix can be encoded once and its exact error-feedback
# state can be fanned out into all three candidates. The serving permutation is
# unchanged: selected records are still repacked into the ordinary QSRT
# P24/P33 order after encoding.
QSRT_PREFIX_REUSE_MAX_MODE = 2


def _qsrt_pair_prefix_reuse_group_order(
    block_contexts: torch.Tensor,
    spec: ModeSpec,
) -> torch.Tensor:
    """Put nested donor/recipient pairs next to each other for trie reuse.

    LDLQ traverses the returned record order backwards.  R1's pair is
    therefore visited first and is common to every shifted candidate; each
    subsequent pair introduces only one additional live trie state. For
    R1/R2 this reduces the variable-region traversal while preserving every
    candidate exactly.
    """

    if spec.context_count != INTERMEDIATE_CHANNELS // RECORD_CHANNELS:
        raise ValueError(
            "qsrt_pair_prefix_reuse requires one context per 128-channel record"
        )
    ordered = importance_group_order(block_contexts, spec).reshape(
        spec.context_count, -1
    )
    r = QSRT_PREFIX_REUSE_MAX_MODE
    paired = tuple(
        record
        for transfer in range(r, 0, -1)
        for record in (transfer - 1, spec.context_count - transfer)
    )
    middle = tuple(range(r, spec.context_count - r))
    return ordered.index_select(
        0,
        torch.tensor(paired + middle, dtype=torch.long, device=ordered.device),
    ).flatten()


@dataclass(frozen=True)
class QSRTMatrixPlan:
    """Pure layout plan for one matrix encode.

    ``encoder_permutation`` is the low-to-high order used during LDLQ.
    ``physical_permutation`` is the common P24/P33 order stored for serving.
    ``record_repack_order`` maps the former to the latter without splitting a
    128-channel record.
    """

    matrix: str
    mode: ModeSpec
    rate_axis: RateAxis
    layout: Layout
    encoder_permutation: torch.Tensor
    physical_permutation: torch.Tensor
    record_repack_order: torch.Tensor
    encoder_tile_bits: tuple[int, ...]
    physical_tile_bits: tuple[int, ...]
    encoder_pair_bpw: tuple[float, ...]
    physical_pair_bpw: tuple[float, ...]


@dataclass(frozen=True)
class QSRTTransformSeeds:
    """Independent deterministic sign streams for one matrix transform.

    ``input_sign`` produces ``su`` and ``output_sign`` produces ``sv`` in the
    local EXL encoder.  Canonical QSRT storage requires the residual-side
    stream to be layer-shared (w1/w3 input, w2 output), while the opposite
    intermediate-side stream may be selected per expert.
    """

    input_sign: int
    output_sign: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.input_sign, self.output_sign)
        ):
            raise ValueError("QSRT transform seeds must be nonnegative integers")


def default_qsrt_transform_seeds(layer: int, matrix: str) -> QSRTTransformSeeds:
    """Return the established production transform streams unchanged."""

    if not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must be in 1..92")
    if matrix not in MATRICES:
        raise ValueError(f"unknown expert matrix: {matrix}")
    input_sign = layer * 1_000_000 + MATRICES.index(matrix)
    return QSRTTransformSeeds(input_sign, input_sign + 499_979)


def qsrt_transform_seed_draw(
    layer: int,
    matrix: str,
    *,
    residual_draw: int = 0,
    intermediate_draw: int = 0,
    expert: int | None = None,
) -> QSRTTransformSeeds:
    """Derive a reproducible layer-shared/private transform candidate.

    Draw zero is bit-for-bit the established production seed.  Nonzero
    residual draws alter w1/w3 input signs and w2 output signs and therefore
    must be selected once per layer.  Nonzero intermediate draws alter the
    opposite side and are salted by expert ID.
    """

    if any(
        isinstance(draw, bool) or not isinstance(draw, int) or draw < 0
        for draw in (residual_draw, intermediate_draw)
    ):
        raise ValueError("rotation draw indices must be nonnegative integers")
    if intermediate_draw and (
        expert is None
        or isinstance(expert, bool)
        or not isinstance(expert, int)
        or not 0 <= expert < 896
    ):
        raise ValueError("a nonzero intermediate draw requires an expert in 0..895")
    base = default_qsrt_transform_seeds(layer, matrix)
    residual_offset = residual_draw * 10_000_019
    private_offset = 0
    if intermediate_draw:
        assert expert is not None
        private_offset = intermediate_draw * 100_000_007 + expert * 1_000_003
    if matrix == "w2":
        return QSRTTransformSeeds(
            base.input_sign + private_offset,
            base.output_sign + residual_offset,
        )
    return QSRTTransformSeeds(
        base.input_sign + residual_offset,
        base.output_sign + private_offset,
    )


@dataclass
class QSRTMatrixEncoding:
    """One closed physical payload and its canonical-order reconstruction."""

    reconstruction: torch.Tensor
    tensors: dict[str, torch.Tensor]
    packed: PackedQSRTTrellis
    plan: QSRTMatrixPlan
    coding: dict[str, object]


@dataclass
class QSRTMatrixCandidate:
    """Unpacked candidate used for fast rate-grid scoring.

    ``reconstruction`` is restored to canonical checkpoint coordinates for
    functional scoring.  The equivalent EXL-oriented reconstruction is
    derived only for the selected candidate during finalization.  Retaining
    both copies for every R candidate consumed hundreds of MiB per expert and
    needlessly constrained the offline batch size.  Packing and the expensive
    reference state roundtrip are likewise deferred until selection.
    """

    reconstruction: torch.Tensor
    encoded: torch.Tensor
    tensors: dict[str, torch.Tensor]
    plan: QSRTMatrixPlan
    proxy: float
    transform_seeds: QSRTTransformSeeds
    global_scale: float = 1.0


@dataclass
class QSRTExpertEncoding:
    """Three matrices encoded under one exact physical neuron permutation."""

    format: ExpertFormatSpec
    matrices: dict[str, QSRTMatrixEncoding]
    physical_permutation: torch.Tensor
    coding: dict[str, object]


def _pair_bpw(tile_bits: tuple[int, ...]) -> tuple[float, ...]:
    if len(tile_bits) != INTERMEDIATE_CHANNELS // 16:
        raise ValueError("rate map must cover the 3072-channel tile axis")
    return tuple(
        sum(tile_bits[begin : begin + 16]) / 16
        for begin in range(0, len(tile_bits), 16)
    )


def plan_qsrt_matrix(
    block_contexts: torch.Tensor,
    mode: ModeSpec | str | int,
    *,
    matrix: str,
    layout: Layout = "importance_ordered",
) -> QSRTMatrixPlan:
    """Plan one rate-shifted traversal and its canonical paired-record repack."""

    spec = resolve_mode(mode)
    rate_axis = matrix_rate_axis(matrix)
    if layout == "importance_ordered":
        encoder_group_order = importance_group_order(block_contexts, spec)
    elif layout == "qsrt_pair_prefix_reuse":
        if spec.mode_id > QSRT_PREFIX_REUSE_MAX_MODE:
            raise ValueError("qsrt_pair_prefix_reuse supports only R0 through R2")
        encoder_group_order = _qsrt_pair_prefix_reuse_group_order(
            block_contexts, spec
        )
    else:
        raise ValueError(f"unsupported context layout: {layout}")

    physical_group_order = storage_group_order(block_contexts, spec)
    encoder_permutation = expand_group_order(encoder_group_order)
    physical_permutation = expand_group_order(physical_group_order)
    repack_order = record_repack_order(
        encoder_group_order, physical_group_order
    )

    channel_contexts = block_contexts.repeat_interleave(CONTEXT_GROUP_CHANNELS)
    encoder_contexts = channel_contexts.index_select(0, encoder_permutation)
    encoder_tiles = encoder_contexts.reshape(-1, 16)
    if not bool(torch.all(encoder_tiles == encoder_tiles[:, :1])):
        raise ValueError("context permutation does not produce homogeneous EXL tiles")
    encoder_tile_bits = tuple(
        spec.context_bits[int(context)]
        for context in encoder_tiles[:, 0].cpu().tolist()
    )

    physical_contexts = channel_contexts.index_select(0, physical_permutation)
    physical_tiles = physical_contexts.reshape(-1, 16)
    if not bool(torch.all(physical_tiles == physical_tiles[:, :1])):
        raise ValueError("physical permutation does not produce homogeneous EXL tiles")
    physical_tile_bits = tuple(
        spec.context_bits[int(context)]
        for context in physical_tiles[:, 0].cpu().tolist()
    )
    expected_physical_bits = tuple(
        bits for bits in record_bits(spec) for _ in range(8)
    )
    if physical_tile_bits != expected_physical_bits:
        raise ValueError("physical contexts do not match the QSRT mode schedule")

    return QSRTMatrixPlan(
        matrix=matrix,
        mode=spec,
        rate_axis=rate_axis,
        layout=layout,
        encoder_permutation=encoder_permutation,
        physical_permutation=physical_permutation,
        record_repack_order=repack_order,
        encoder_tile_bits=encoder_tile_bits,
        physical_tile_bits=physical_tile_bits,
        encoder_pair_bpw=_pair_bpw(encoder_tile_bits),
        physical_pair_bpw=_pair_bpw(physical_tile_bits),
    )


def _validate_source_and_hessian(
    source: torch.Tensor, hessian: torch.Tensor, matrix: str
) -> None:
    expected_source = (
        (LATENT_CHANNELS, INTERMEDIATE_CHANNELS)
        if matrix == "w2"
        else (INTERMEDIATE_CHANNELS, LATENT_CHANNELS)
    )
    expected_hessian = (
        (INTERMEDIATE_CHANNELS, INTERMEDIATE_CHANNELS)
        if matrix == "w2"
        else (LATENT_CHANNELS, LATENT_CHANNELS)
    )
    if source.ndim != 2 or tuple(source.shape) != expected_source:
        raise ValueError(
            f"{matrix} source has shape {tuple(source.shape)}, expected {expected_source}"
        )
    if hessian.ndim != 2 or tuple(hessian.shape) != expected_hessian:
        raise ValueError(
            f"{matrix} Hessian has shape {tuple(hessian.shape)}, expected {expected_hessian}"
        )
    if source.dtype != torch.float32:
        raise TypeError("QSRT source must use torch.float32")
    if not bool(torch.all(torch.isfinite(source))):
        raise ValueError("QSRT source contains non-finite values")
    if not bool(torch.all(torch.isfinite(hessian))):
        raise ValueError("QSRT Hessian contains non-finite values")


def _load_quantizer_module(codebook: str = CODEBOOK_QSRT_E4M3) -> ModuleType:
    if codebook not in QSRT_CODEBOOKS:
        raise ValueError(f"unsupported QSRT codebook: {codebook!r}")
    from kquant.exl3_loader import load_qsrt_encoder
    from kquant.sqg_quantizer import install_sqg_quantizer

    root = Path(
        os.environ.get("KQUANT_EXLLAMAV3_ROOT", "/home/luke/projects/exllamav3")
    )
    quantizer_module = load_qsrt_encoder(root)
    install_sqg_quantizer(quantizer_module)
    return quantizer_module


def make_qsrt_shared_h(
    block_contexts: torch.Tensor,
    *,
    matrix: str,
    mode: ModeSpec | str | int,
    hessian: torch.Tensor,
    device: torch.device,
    layout: Layout = "importance_ordered",
) -> dict[str, object]:
    """Prepare one reusable LDL factorization for several rate candidates.

    All canonical ``R_r`` modes use the same low-to-high neuron traversal for
    a fixed context map.  The expensive Hessian preparation can therefore be
    shared across the phase-1 ladder.  The attached signature makes accidental
    reuse with another matrix, permutation, or Hessian fail closed.
    """

    if device.type != "cuda":
        raise ValueError("native QSRT quantization requires a CUDA device")
    if matrix not in MATRICES:
        raise ValueError(f"unknown expert matrix: {matrix}")
    expected_dimension = INTERMEDIATE_CHANNELS if matrix == "w2" else LATENT_CHANNELS
    if hessian.ndim != 2 or tuple(hessian.shape) != (
        expected_dimension,
        expected_dimension,
    ):
        raise ValueError(
            f"{matrix} Hessian has shape {tuple(hessian.shape)}, expected "
            f"{(expected_dimension, expected_dimension)}"
        )
    if not bool(torch.all(torch.isfinite(hessian))):
        raise ValueError("QSRT Hessian contains non-finite values")
    contexts = block_contexts.to(device=device, dtype=torch.long)
    plan = plan_qsrt_matrix(contexts, mode, matrix=matrix, layout=layout)
    encoder_hessian = hessian
    if matrix == "w2":
        permutation = plan.encoder_permutation.to(device=hessian.device)
        encoder_hessian = hessian.index_select(0, permutation).index_select(
            1, permutation
        )
    shared_h: dict[str, object] = make_shared_h(
        expected_dimension, device, encoder_hessian
    )
    shared_h["_kquant_qsrt_matrix"] = matrix
    shared_h["_kquant_qsrt_layout"] = layout
    shared_h["_kquant_qsrt_permutation"] = (
        plan.encoder_permutation.detach().cpu().contiguous()
    )
    shared_h["_kquant_source_hessian_ptr"] = int(hessian.data_ptr())
    return shared_h


def _validate_qsrt_shared_h(
    shared_h: dict[str, object],
    *,
    matrix: str,
    layout: Layout,
    plan: QSRTMatrixPlan,
    hessian: torch.Tensor,
    device: torch.device,
) -> None:
    if shared_h.get("_kquant_qsrt_matrix") != matrix:
        raise ValueError("QSRT shared Hessian belongs to another matrix")
    if shared_h.get("_kquant_qsrt_layout") != layout:
        raise ValueError("QSRT shared Hessian belongs to another layout")
    if shared_h.get("_kquant_source_hessian_ptr") != int(hessian.data_ptr()):
        raise ValueError("QSRT shared Hessian belongs to another source Hessian")
    permutation = shared_h.get("_kquant_qsrt_permutation")
    if not isinstance(permutation, torch.Tensor) or not torch.equal(
        permutation, plan.encoder_permutation.detach().cpu()
    ):
        raise ValueError("QSRT shared Hessian belongs to another neuron order")
    error_h = shared_h.get("error_H")
    expected_dimension = INTERMEDIATE_CHANNELS if matrix == "w2" else LATENT_CHANNELS
    if (
        not isinstance(error_h, torch.Tensor)
        or tuple(error_h.shape) != (expected_dimension, expected_dimension)
        or error_h.device != device
    ):
        raise ValueError("QSRT shared Hessian has an invalid error_H tensor")


def _qsrt_quant_args(
    plan: QSRTMatrixPlan,
    *,
    matrix: str,
    layer: int,
    device: torch.device,
    shared_scale_scope: object | None,
    codebook: str = CODEBOOK_QSRT_E4M3,
    sqg_e4m3_luts_by_bits: Mapping[int, torch.Tensor | None] | None = None,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    transform_seeds: QSRTTransformSeeds | None = None,
    g_scale_override: float | None = None,
) -> dict[str, object]:
    if codebook not in QSRT_CODEBOOKS:
        raise ValueError(
            f"unsupported QSRT codebook {codebook!r}; expected one of {QSRT_CODEBOOKS}"
        )
    seeds = transform_seeds or default_qsrt_transform_seeds(layer, matrix)
    quant_args: dict[str, object] = {
        # Preserve the established K3 regularization and scale search.  The
        # heterogeneous map controls only the trellis operation.
        "K": 3,
        # These are internal kquant encoder-backend keys, not serialized QSRT
        # format names.
        "mixed_rate_axis": plan.rate_axis,
        "mixed_tile_bits": plan.encoder_tile_bits,
        "seed": seeds.input_sign,
        "sv_seed": seeds.output_sign,
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": bool(ldlq_tf32),
        "tailbite_context": int(tailbite_context),
    }
    if (
        sqg_e4m3_luts_by_bits is None
        and codebook == CODEBOOK_SQG_NORMAL_E4M3
    ):
        quant_args["sqg_e4m3_mode"] = "normal"
    else:
        if sqg_e4m3_luts_by_bits is None:
            if codebook == CODEBOOK_QSRT_E4M3:
                sqg_e4m3_luts_by_bits = {
                    bits: qsrt_e4m3_bytes(bits) for bits in (2, 3, 4)
                }
            elif codebook == CODEBOOK_SQG_CHEB_NORMAL_E4M3:
                sqg_e4m3_luts_by_bits = {
                    bits: sqg_cheb_normal_e4m3_bytes(bits) for bits in (2, 3, 4)
                }
            elif codebook == CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3:
                sqg_e4m3_luts_by_bits = {
                    bits: sqg_cheb_normal_e4m3_bytes(bits) for bits in (2, 3, 4)
                }
                if matrix == "w2":
                    sqg_e4m3_luts_by_bits[2] = (
                        sqg_k2_eight_stratum_e4m3_bytes_from_rank_lut(
                            sqg_cheb_normal_rank_e4m3_bytes(), history_bit=4
                        )
                    )
            else:
                raise AssertionError("QSRT codebook dispatch is incomplete")
        if set(sqg_e4m3_luts_by_bits) != {2, 3, 4}:
            raise ValueError("rate-specific SQG LUTs must define K2, K3, and K4")
        quant_args["sqg_e4m3_luts_by_bits"] = dict(sqg_e4m3_luts_by_bits)
    if matrix in ("w1", "w3"):
        quant_args["shared_input_scales_key"] = (shared_scale_scope, matrix)
        quant_args["g_scale_into_sv"] = True
    if g_scale_override is not None:
        if (
            isinstance(g_scale_override, bool)
            or not isinstance(g_scale_override, (int, float))
            or not math.isfinite(float(g_scale_override))
            or float(g_scale_override) <= 0.0
        ):
            raise ValueError("g_scale_override must be positive and finite")
        quant_args["g_scale_override"] = float(g_scale_override)
    return quant_args


def _encoder_weight(
    source: torch.Tensor,
    hessian: torch.Tensor,
    plan: QSRTMatrixPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = plan.encoder_permutation
    if plan.matrix == "w2":
        weight = source.index_select(1, permutation).T.contiguous()
        hessian_permutation = permutation.to(device=hessian.device)
        encoder_hessian = hessian.index_select(
            0, hessian_permutation
        ).index_select(1, hessian_permutation)
    else:
        weight = source.index_select(0, permutation).T.contiguous()
        encoder_hessian = hessian
    return weight, encoder_hessian


def quantize_qsrt_matrix_candidates_batched(
    sources: dict[str, torch.Tensor],
    block_contexts: torch.Tensor,
    *,
    modes: tuple[ModeSpec, ...],
    hessians: dict[str, torch.Tensor],
    layer: int,
    device: torch.device,
    quantizer_module: ModuleType | object | None = None,
    shared_scale_scope: object | None = None,
    layout: Layout = "importance_ordered",
    codebook: str = CODEBOOK_QSRT_E4M3,
    sqg_e4m3_luts_by_bits: Mapping[int, torch.Tensor | None] | None = None,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    transform_seeds: Mapping[str, QSRTTransformSeeds] | None = None,
    intermediate_conditioning: torch.Tensor | None = None,
    g_scale_override: float | None = None,
) -> dict[str, dict[int, QSRTMatrixCandidate]]:
    """Encode same-shape matrices and all requested R modes in one batch.

    The low-level encoder performs one preparation per source matrix and one
    grouped heterogeneous-rate LDLQ traversal.  Returned candidates remain unpacked so the
    caller can score a complete ``(r13, r2)`` grid before closing only the
    selected physical payload.
    """

    return quantize_qsrt_matrix_expert_batch(
        [sources],
        [block_contexts],
        modes_by_expert=[modes],
        hessians_by_expert=[hessians],
        layer=layer,
        device=device,
        quantizer_module=quantizer_module,
        shared_scale_scope=shared_scale_scope,
        layout=layout,
        codebook=codebook,
        sqg_e4m3_luts_by_bits=sqg_e4m3_luts_by_bits,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        transform_seeds_by_expert=[transform_seeds or {}],
        intermediate_conditioning_by_expert=[intermediate_conditioning],
        g_scale_overrides_by_expert=[g_scale_override],
    )[0]


def quantize_qsrt_matrix_expert_batch(
    sources_by_expert: list[dict[str, torch.Tensor]],
    block_contexts_by_expert: list[torch.Tensor],
    *,
    modes_by_expert: list[tuple[ModeSpec, ...]],
    hessians_by_expert: list[dict[str, torch.Tensor]],
    layer: int,
    device: torch.device,
    quantizer_module: ModuleType | object | None = None,
    shared_scale_scope: object | None = None,
    layout: Layout = "importance_ordered",
    codebook: str = CODEBOOK_QSRT_E4M3,
    sqg_e4m3_luts_by_bits: Mapping[int, torch.Tensor | None] | None = None,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    transform_seeds_by_expert: list[Mapping[str, QSRTTransformSeeds]] | None = None,
    intermediate_conditioning_by_expert: list[torch.Tensor | None] | None = None,
    g_scale_overrides_by_expert: list[float | None] | None = None,
) -> list[dict[str, dict[int, QSRTMatrixCandidate]]]:
    """Batch the same projection family across several routed experts.

    Experts retain independent permutations, Hessians, transforms, LDLQ
    feedback, and candidate grids.  Flattening them into one low-level batch
    only fills otherwise underoccupied trellis launches.  This is especially
    important for R1/R2 replacement records, where one expert contributes far
    fewer tiles than an SM120 device can execute concurrently.
    """

    expert_count = len(sources_by_expert)
    if (
        expert_count <= 0
        or len(block_contexts_by_expert) != expert_count
        or len(modes_by_expert) != expert_count
        or len(hessians_by_expert) != expert_count
    ):
        raise ValueError("expert-batched QSRT inputs must have equal nonzero lengths")
    if transform_seeds_by_expert is None:
        transform_seeds_by_expert = [{} for _ in range(expert_count)]
    elif len(transform_seeds_by_expert) != expert_count:
        raise ValueError("expert-batched transform seed maps have the wrong length")
    if intermediate_conditioning_by_expert is None:
        intermediate_conditioning_by_expert = [None] * expert_count
    elif len(intermediate_conditioning_by_expert) != expert_count:
        raise ValueError("expert-batched conditioning profiles have the wrong length")
    if g_scale_overrides_by_expert is None:
        g_scale_overrides_by_expert = [None] * expert_count
    elif len(g_scale_overrides_by_expert) != expert_count:
        raise ValueError("expert-batched global-scale overrides have the wrong length")

    entries: list[
        tuple[int, str, torch.Tensor, tuple[ModeSpec, ...], torch.Tensor | None]
    ] = []
    plans: list[dict[int, QSRTMatrixPlan]] = []
    weights: list[torch.Tensor] = []
    shared_hessians: list[dict[str, object]] = []
    quant_args_groups: list[list[dict[str, object]]] = []
    common_shape: tuple[int, ...] | None = None

    residual_side_seeds: dict[str, int] = {}
    for expert_index, (
        sources,
        contexts_raw,
        modes,
        hessians,
        seed_overrides,
        conditioning_raw,
        g_scale_override,
    ) in enumerate(
        zip(
            sources_by_expert,
            block_contexts_by_expert,
            modes_by_expert,
            hessians_by_expert,
            transform_seeds_by_expert,
            intermediate_conditioning_by_expert,
            g_scale_overrides_by_expert,
        )
    ):
        if not sources or set(sources) != set(hessians):
            raise ValueError(
                "batched QSRT sources and Hessians must share nonempty keys"
            )
        if not modes or len({mode.mode_id for mode in modes}) != len(modes):
            raise ValueError("batched QSRT modes must be nonempty and unique")
        matrices = [matrix for matrix in MATRICES if matrix in sources]
        if set(matrices) != set(sources):
            raise ValueError("batched QSRT sources contain an unknown matrix")
        shapes = {tuple(source.shape) for source in sources.values()}
        if len(shapes) != 1:
            raise ValueError("one QSRT batch may contain only same-shape matrices")
        shape = next(iter(shapes))
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise ValueError("expert-batched QSRT matrices must share one shape")

        contexts = contexts_raw.to(device=device, dtype=torch.long)
        conditioning = None
        if conditioning_raw is not None:
            conditioning = conditioning_raw.to(device=device, dtype=torch.float32)
            if conditioning.ndim != 1 or conditioning.numel() != INTERMEDIATE_CHANNELS:
                raise ValueError("intermediate conditioning must cover 3072 channels")
            if not bool(torch.all(torch.isfinite(conditioning))) or bool(
                torch.any(conditioning <= 0)
            ):
                raise ValueError("intermediate conditioning must be finite and positive")
        for matrix in matrices:
            source = sources[matrix]
            hessian = hessians[matrix]
            _validate_source_and_hessian(source, hessian, matrix)
            if source.device != device:
                raise ValueError(
                    f"batched QSRT source {expert_index}.{matrix} is not on {device}"
                )
            if matrix in ("w1", "w3") and shared_scale_scope is None:
                raise ValueError(f"{matrix} requires a shared_scale_scope")
            unknown_seed_matrices = set(seed_overrides) - set(MATRICES)
            if unknown_seed_matrices:
                raise ValueError(
                    "transform seed map contains unknown matrices: "
                    f"{sorted(unknown_seed_matrices)}"
                )
            seeds = seed_overrides.get(matrix) or default_qsrt_transform_seeds(
                layer, matrix
            )
            if not isinstance(seeds, QSRTTransformSeeds):
                raise TypeError("transform seed overrides must use QSRTTransformSeeds")
            residual_seed = (
                seeds.output_sign if matrix == "w2" else seeds.input_sign
            )
            previous_residual_seed = residual_side_seeds.setdefault(
                matrix, residual_seed
            )
            if previous_residual_seed != residual_seed:
                side = "output" if matrix == "w2" else "input"
                raise ValueError(
                    f"{matrix} {side}-sign seed must be layer-shared within a batch"
                )
            matrix_plans = {
                mode.mode_id: plan_qsrt_matrix(
                    contexts, mode, matrix=matrix, layout=layout
                )
                for mode in modes
            }
            reference = matrix_plans[modes[0].mode_id]
            for plan in matrix_plans.values():
                if not torch.equal(
                    plan.encoder_permutation, reference.encoder_permutation
                ):
                    raise ValueError("rate candidates changed the encoder neuron order")
                if not torch.equal(
                    plan.physical_permutation, reference.physical_permutation
                ):
                    raise ValueError("rate candidates changed the physical neuron order")
            weight, _ = _encoder_weight(source, hessian, reference)
            encoder_conditioning = None
            if conditioning is not None and matrix in ("w1", "w3"):
                encoder_conditioning = conditioning.index_select(
                    0, reference.encoder_permutation
                )
                blocks = encoder_conditioning.reshape(-1, 128)
                if not bool(torch.all(blocks == blocks[:, :1])):
                    raise ValueError(
                        "folded conditioning must be constant in each Hadamard block"
                    )
                weight = weight / encoder_conditioning.unsqueeze(0)
            entries.append(
                (expert_index, matrix, source, modes, encoder_conditioning)
            )
            plans.append(matrix_plans)
            weights.append(weight)
            shared_hessians.append(
                make_qsrt_shared_h(
                    contexts,
                    matrix=matrix,
                    mode=modes[0],
                    hessian=hessian,
                    device=device,
                    layout=layout,
                )
            )
            quant_args_groups.append(
                [
                    _qsrt_quant_args(
                        matrix_plans[mode.mode_id],
                        matrix=matrix,
                        layer=layer,
                        device=device,
                        shared_scale_scope=shared_scale_scope,
                        codebook=codebook,
                        sqg_e4m3_luts_by_bits=sqg_e4m3_luts_by_bits,
                        ldlq_tf32=ldlq_tf32,
                        tailbite_context=tailbite_context,
                        transform_seeds=seeds,
                        g_scale_override=g_scale_override,
                    )
                    for mode in modes
                ]
            )

    module = (
        _load_quantizer_module(codebook)
        if quantizer_module is None
        else quantizer_module
    )
    batch_api = getattr(module, "quantize_qsrt_batch", None)
    if batch_api is None:
        raise ValueError("QSRT encoder backend lacks mixed-rate batch support")
    raw_groups = batch_api(
        weights,
        shared_hessians,
        quant_args_groups,
        return_weight_q=True,
    )
    if len(raw_groups) != len(entries):
        raise ValueError("QSRT batch returned the wrong source count")

    result: list[dict[str, dict[int, QSRTMatrixCandidate]]] = [
        {} for _ in range(expert_count)
    ]
    finite_checks: list[torch.Tensor] = []
    finite_labels: list[tuple[int, str, int]] = []
    for entry_index, (
        (expert_index, matrix, source, modes, encoder_conditioning),
        raw_group,
    ) in enumerate(
        zip(entries, raw_groups)
    ):
        if len(raw_group) != len(modes):
            raise ValueError("QSRT batch returned the wrong candidate count")
        matrix_result: dict[int, QSRTMatrixCandidate] = {}
        for mode, raw in zip(modes, raw_group):
            plan = plans[entry_index][mode.mode_id]
            encoder_weight = raw["weight_q"]
            if not isinstance(encoder_weight, torch.Tensor):
                raise ValueError("QSRT batch omitted its reconstruction")
            candidate_suh = raw["suh"]
            candidate_svh = raw["svh"]
            if encoder_conditioning is not None:
                original_svh = candidate_svh.float()
                candidate_svh = (
                    original_svh * encoder_conditioning
                ).to(torch.float16)
                ratio = torch.ones_like(original_svh)
                nonzero = original_svh != 0
                ratio[nonzero] = (
                    candidate_svh.float()[nonzero] / original_svh[nonzero]
                )
                encoder_weight = encoder_weight * ratio.unsqueeze(0)
            reconstruction = torch.empty_like(source)
            if matrix == "w2":
                reconstruction[:, plan.encoder_permutation] = encoder_weight.T
            else:
                reconstruction[plan.encoder_permutation] = encoder_weight.T
            # Defer this device scalar read until every candidate has queued
            # its check.  Reading one boolean here serialized hundreds of
            # otherwise independent reconstructions in an R0/R1/R2 batch.
            finite_checks.append(torch.all(torch.isfinite(reconstruction)))
            finite_labels.append((expert_index, matrix, mode.mode_id))
            proxy = float(raw["proxy"])
            if not math.isfinite(proxy):
                raise ValueError("QSRT batch produced a non-finite proxy")
            matrix_result[mode.mode_id] = QSRTMatrixCandidate(
                reconstruction=reconstruction,
                encoded=raw["encoded"],
                tensors={"suh": candidate_suh, "svh": candidate_svh},
                plan=plan,
                proxy=proxy,
                transform_seeds=QSRTTransformSeeds(
                    input_sign=int(
                        quant_args_groups[entry_index][0]["seed"]
                    ),
                    output_sign=int(
                        quant_args_groups[entry_index][0]["sv_seed"]
                    ),
                ),
                global_scale=float(raw.get("g_scale", 1.0)),
            )
        result[expert_index][matrix] = matrix_result
    finite = torch.stack(finite_checks).cpu()
    if not bool(torch.all(finite)):
        first = int(torch.nonzero(~finite, as_tuple=False)[0])
        expert_index, matrix, mode_id = finite_labels[first]
        raise ValueError(
            "QSRT batch produced a non-finite reconstruction for "
            f"expert-index {expert_index} {matrix} R{mode_id}"
        )
    return result


def finalize_qsrt_matrix_candidate(
    candidate: QSRTMatrixCandidate,
    *,
    layer: int,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_QSRT_E4M3,
    tailbite_context: int = 128,
) -> QSRTMatrixEncoding:
    """Pack and independently decode one selected rate-shifted candidate."""

    plan = candidate.plan
    physical = repack_encoded_records(
        candidate.encoded, plan.record_repack_order, plan.rate_axis
    )
    packed = pack_qsrt_trellis(
        physical,
        plan.mode,
        rate_axis=plan.rate_axis,
        schema=logical_trellis_schema,
        # The selected-candidate closure immediately below reconstructs and
        # checks every state in one batched pass.  Repeating that same
        # 256-step recurrence separately for all 24 records here was pure
        # duplicate work in the all-expert encoder.
        validate_states=False,
    )
    edge_symbols = unpack_qsrt_trellis_edges(packed)
    rate_bits = torch.tensor(
        plan.physical_tile_bits,
        device=physical.device,
        dtype=torch.int64,
    )
    masks = (1 << rate_bits) - 1
    masks = masks[:, None, None] if plan.rate_axis == "k" else masks[None, :, None]
    edge_closes = torch.all(
        edge_symbols.to(torch.int64) == (physical.to(torch.int64) & masks)
    )
    decoded_states = unpack_qsrt_trellis_states(packed)
    state_closes = torch.all(decoded_states == physical.to(torch.int16))
    edge_ok, state_ok = torch.stack((edge_closes, state_closes)).cpu().tolist()
    if not edge_ok:
        raise ValueError(
            "QSRT reference edge unpack failed closure for "
            f"layer {layer} {plan.matrix} R{plan.mode.mode_id} "
            f"({plan.rate_axis}-rate, {plan.layout})"
        )
    if not state_ok:
        mismatch = decoded_states != physical.to(torch.int16)
        mismatch_count = int(torch.count_nonzero(mismatch).cpu())
        first = tuple(
            int(value)
            for value in torch.nonzero(mismatch, as_tuple=False)[0].cpu().tolist()
        )
        expected = int(physical[first].to(torch.int32).cpu())
        observed = int(decoded_states[first].to(torch.int32).cpu())
        raise ValueError(
            "QSRT state reconstruction failed exact closure for "
            f"layer {layer} {plan.matrix} R{plan.mode.mode_id} "
            f"({plan.rate_axis}-rate, {plan.layout}): {mismatch_count} states "
            f"differ; first at {first}, encoded={expected}, decoded={observed}"
        )

    tensors = {
        "trellis": packed.payload,
        "suh": candidate.tensors["suh"],
        "svh": candidate.tensors["svh"],
    }
    if plan.rate_axis == "k":
        tensors["suh"] = repack_rate_axis(
            tensors["suh"], plan.record_repack_order, 0
        )
    else:
        tensors["svh"] = repack_rate_axis(
            tensors["svh"], plan.record_repack_order, 0
        )
    tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
    decoded_physical = decode_qsrt_weight(
        decoded_states,
        tensors["suh"],
        tensors["svh"],
        rate_axis=plan.rate_axis,
        tile_bits=plan.physical_tile_bits,
        codebook=codebook,
    )
    # Recreate the EXL-oriented selected reconstruction exactly from the
    # canonical scoring tensor.  Keeping this second 44 MiB tensor alive for
    # every R candidate substantially increased peak memory; only the winner
    # needs it for physical-order closure.
    if plan.matrix == "w2":
        encoder_weight = candidate.reconstruction.index_select(
            1, plan.encoder_permutation
        ).T.contiguous()
    else:
        encoder_weight = candidate.reconstruction.index_select(
            0, plan.encoder_permutation
        ).T.contiguous()
    encoder_physical = repack_rate_axis(
        encoder_weight,
        plan.record_repack_order,
        0 if plan.rate_axis == "k" else 1,
    )
    stored_decode_vs_encoder = comparison_metrics(
        encoder_physical, decoded_physical
    )
    if not all(math.isfinite(value) for value in stored_decode_vs_encoder.values()):
        raise ValueError("stored QSRT decode produced non-finite values")

    reconstruction = torch.empty_like(candidate.reconstruction)
    physical_permutation = plan.physical_permutation
    if plan.matrix == "w2":
        reconstruction[:, physical_permutation] = decoded_physical.T
    else:
        reconstruction[physical_permutation] = decoded_physical.T
    scoring_vs_stored = comparison_metrics(candidate.reconstruction, reconstruction)
    if not all(math.isfinite(value) for value in scoring_vs_stored.values()):
        raise ValueError("selected QSRT scoring reconstruction did not close")

    index_bits = tensors["trellis"].numel() * tensors["trellis"].element_size() * 8
    scale_bits = sum(
        tensors[name].numel() * tensors[name].element_size() * 8
        for name in ("suh", "svh")
    )
    if index_bits != reconstruction.numel() * 3:
        raise ValueError("QSRT payload is not exactly three trellis bpw")
    coding: dict[str, object] = {
        "proxy": candidate.proxy,
        "scale_bits": scale_bits,
        "index_bits": index_bits,
        "matrix": plan.matrix,
        "codebook": codebook,
        "tailbite_context": int(tailbite_context),
        "mode": plan.mode.name,
        "mode_id": plan.mode.mode_id,
        "rate_axis": plan.rate_axis,
        "layout": plan.layout,
        "encoder_pair_trellis_bpw": list(plan.encoder_pair_bpw),
        "stored_pair_trellis_bpw": list(plan.physical_pair_bpw),
        "postpack_moves_complete_128_channel_records": True,
        "reference_edge_roundtrip": True,
        "reference_state_roundtrip": True,
        "stored_decode_vs_encoder": stored_decode_vs_encoder,
        "scoring_reconstruction_vs_stored": scoring_vs_stored,
        "trellis_descriptor": packed.descriptor.to_manifest(),
        "permutation_is_128_context_aligned": True,
        "scale_tensor_shapes": {
            name: list(tensors[name].shape) for name in ("suh", "svh")
        },
        "shared_hidden_scale_tensor": "svh" if plan.matrix == "w2" else "suh",
        "expert_local_intermediate_scale_tensor": (
            "suh" if plan.matrix == "w2" else "svh"
        ),
        "transform_seeds": {
            "input_sign": candidate.transform_seeds.input_sign,
            "output_sign": candidate.transform_seeds.output_sign,
        },
    }
    return QSRTMatrixEncoding(
        reconstruction=reconstruction,
        tensors=tensors,
        packed=packed,
        plan=plan,
        coding=coding,
    )


def quantize_qsrt_matrix(
    source: torch.Tensor,
    block_contexts: torch.Tensor,
    *,
    matrix: str,
    mode: ModeSpec | str | int,
    hessian: torch.Tensor,
    layer: int,
    device: torch.device,
    layout: Layout = "importance_ordered",
    quantizer_module: ModuleType | object | None = None,
    shared_scale_scope: object | None = None,
    shared_h_data: dict[str, object] | None = None,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_QSRT_E4M3,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    transform_seeds: QSRTTransformSeeds | None = None,
) -> QSRTMatrixEncoding:
    """Run dense-H heterogeneous LDLQ and close the stored QSRT payload.

    ``shared_scale_scope`` is required for ``w1``/``w3`` because their hidden
    ``suh`` vector is stored once per layer/matrix.  All experts and all mode
    candidates belonging to one layer artifact must use the same scope.
    """

    if not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must be in 1..92")
    if device.type != "cuda":
        raise ValueError("native QSRT quantization requires a CUDA device")
    if source.device != device:
        raise ValueError(
            f"QSRT source must be resident on {device}, got {source.device}"
        )
    _validate_source_and_hessian(source, hessian, matrix)
    if matrix in ("w1", "w3") and shared_scale_scope is None:
        raise ValueError(
            f"{matrix} requires a shared_scale_scope for layer-shared suh"
        )
    if shared_scale_scope is not None:
        try:
            hash(shared_scale_scope)
        except TypeError as exc:
            raise TypeError("shared_scale_scope must be hashable") from exc

    contexts = block_contexts.to(device=source.device, dtype=torch.long)
    plan = plan_qsrt_matrix(contexts, mode, matrix=matrix, layout=layout)
    permutation = plan.encoder_permutation
    if matrix == "w2":
        weight = source.index_select(1, permutation).T.contiguous()
        hessian_permutation = permutation.to(device=hessian.device)
        encoder_hessian = hessian.index_select(
            0, hessian_permutation
        ).index_select(1, hessian_permutation)
    else:
        weight = source.index_select(0, permutation).T.contiguous()
        encoder_hessian = hessian

    if shared_h_data is None:
        shared_h = make_shared_h(weight.shape[0], device, encoder_hessian)
    else:
        _validate_qsrt_shared_h(
            shared_h_data,
            matrix=matrix,
            layout=layout,
            plan=plan,
            hessian=hessian,
            device=device,
        )
        shared_h = shared_h_data
    seeds = transform_seeds or default_qsrt_transform_seeds(layer, matrix)
    quant_args = _qsrt_quant_args(
        plan,
        matrix=matrix,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        codebook=codebook,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        transform_seeds=seeds,
    )

    packed_holder: dict[str, PackedQSRTTrellis] = {}

    def pack_qsrt(encoded: torch.Tensor, _quant_args: dict) -> torch.Tensor:
        physical = repack_encoded_records(
            encoded, plan.record_repack_order, plan.rate_axis
        )
        packed = pack_qsrt_trellis(
            physical,
            plan.mode,
            rate_axis=plan.rate_axis,
            schema=logical_trellis_schema,
        )
        edge_symbols = unpack_qsrt_trellis_edges(packed)
        rate_bits = torch.tensor(
            plan.physical_tile_bits,
            device=physical.device,
            dtype=torch.int64,
        )
        masks = (1 << rate_bits) - 1
        masks = masks[:, None, None] if plan.rate_axis == "k" else masks[None, :, None]
        if not torch.equal(
            edge_symbols.to(torch.int64), physical.to(torch.int64) & masks
        ):
            raise ValueError("QSRT reference edge unpack failed closure")
        if not torch.equal(
            unpack_qsrt_trellis_states(packed), physical.to(torch.int16)
        ):
            raise ValueError("QSRT state reconstruction failed exact closure")
        packed_holder["packed"] = packed
        return packed.payload

    quant_args["pack_trellis_fn"] = pack_qsrt
    module = (
        _load_quantizer_module(codebook)
        if quantizer_module is None
        else quantizer_module
    )
    quantized, proxy, raw_tensors = module.quantize_qsrt(
        weight,
        shared_h,
        quant_args,
        True,
        progress_str="",
    )
    packed = packed_holder.get("packed")
    if packed is None:
        raise ValueError("QSRT packer was not called; fallback encoding is unsupported")

    tensors = {
        name: raw_tensors[name]
        for name in ("trellis", "suh", "svh")
    }
    if plan.rate_axis == "k":
        tensors["suh"] = repack_rate_axis(
            tensors["suh"], plan.record_repack_order, 0
        )
    else:
        tensors["svh"] = repack_rate_axis(
            tensors["svh"], plan.record_repack_order, 0
        )
    for name, tensor in tensors.items():
        if not tensor.is_contiguous():
            tensors[name] = tensor.contiguous()

    decoded_physical = decode_qsrt_exl3_weight(
        packed,
        tensors["suh"],
        tensors["svh"],
        codebook=codebook,
    )
    encoder_physical = repack_rate_axis(
        quantized,
        plan.record_repack_order,
        0 if plan.rate_axis == "k" else 1,
    )
    stored_decode_vs_encoder = comparison_metrics(
        encoder_physical, decoded_physical
    )
    if not all(math.isfinite(value) for value in stored_decode_vs_encoder.values()):
        raise ValueError("stored QSRT decode produced non-finite values")

    reconstruction = torch.empty_like(source)
    physical_permutation = plan.physical_permutation
    if matrix == "w2":
        reconstruction[:, physical_permutation] = decoded_physical.T
        if not torch.equal(
            reconstruction.index_select(1, physical_permutation),
            decoded_physical.T,
        ):
            raise ValueError("w2 physical permutation failed exact matrix closure")
    else:
        reconstruction[physical_permutation] = decoded_physical.T
        if not torch.equal(
            reconstruction.index_select(0, physical_permutation),
            decoded_physical.T,
        ):
            raise ValueError(
                f"{matrix} physical permutation failed exact matrix closure"
            )

    index_bits = tensors["trellis"].numel() * tensors["trellis"].element_size() * 8
    scale_bits = sum(
        tensors[name].numel() * tensors[name].element_size() * 8
        for name in ("suh", "svh")
    )
    if index_bits != source.numel() * 3:
        raise ValueError("QSRT payload is not exactly three trellis bpw")
    if not torch.equal(tensors["trellis"], packed.payload):
        raise ValueError("quantizer returned a different trellis payload")

    coding: dict[str, object] = {
        "proxy": float(proxy),
        "scale_bits": scale_bits,
        "index_bits": index_bits,
        "matrix": matrix,
        "codebook": codebook,
        "tailbite_context": int(tailbite_context),
        "mode": plan.mode.name,
        "mode_id": plan.mode.mode_id,
        "rate_axis": plan.rate_axis,
        "layout": plan.layout,
        "encoder_pair_trellis_bpw": list(plan.encoder_pair_bpw),
        "stored_pair_trellis_bpw": list(plan.physical_pair_bpw),
        "postpack_moves_complete_128_channel_records": True,
        "reference_edge_roundtrip": True,
        "reference_state_roundtrip": True,
        "stored_decode_vs_encoder": stored_decode_vs_encoder,
        "trellis_descriptor": packed.descriptor.to_manifest(),
        "permutation_is_128_context_aligned": True,
        "scale_tensor_shapes": {
            name: list(tensors[name].shape) for name in ("suh", "svh")
        },
        "shared_hidden_scale_tensor": "svh" if matrix == "w2" else "suh",
        "expert_local_intermediate_scale_tensor": (
            "suh" if matrix == "w2" else "svh"
        ),
        "transform_seeds": {
            "input_sign": seeds.input_sign,
            "output_sign": seeds.output_sign,
        },
    }
    return QSRTMatrixEncoding(
        reconstruction=reconstruction,
        tensors=tensors,
        packed=packed,
        plan=plan,
        coding=coding,
    )


def quantize_qsrt_expert(
    sources: dict[str, torch.Tensor],
    block_contexts: torch.Tensor,
    *,
    format_spec: ExpertFormatSpec,
    h13: torch.Tensor,
    h2: torch.Tensor,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
    quantizer_module: ModuleType | object | None = None,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_QSRT_E4M3,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    transform_seeds: Mapping[str, QSRTTransformSeeds] | None = None,
) -> QSRTExpertEncoding:
    """Encode all three expert matrices with one physical neuron order."""

    if format_spec.is_x4t:
        raise ValueError("quantize_qsrt_expert requires a compressed format")
    if set(sources) != set(MATRICES):
        raise ValueError(f"sources must contain exactly {MATRICES}")
    assert format_spec.r13 is not None and format_spec.r2 is not None
    modes = {
        "w1": resolve_mode(format_spec.r13),
        "w3": resolve_mode(format_spec.r13),
        "w2": resolve_mode(format_spec.r2),
    }
    hessians = {"w1": h13, "w3": h13, "w2": h2}
    encodings = {
        matrix: quantize_qsrt_matrix(
            sources[matrix],
            block_contexts,
            matrix=matrix,
            mode=modes[matrix],
            hessian=hessians[matrix],
            layer=layer,
            device=device,
            quantizer_module=quantizer_module,
            shared_scale_scope=shared_scale_scope,
            logical_trellis_schema=logical_trellis_schema,
            codebook=codebook,
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
            transform_seeds=(transform_seeds or {}).get(matrix),
        )
        for matrix in MATRICES
    }
    physical_permutation = encodings["w1"].plan.physical_permutation
    for matrix in ("w3", "w2"):
        if not torch.equal(
            encodings[matrix].plan.physical_permutation, physical_permutation
        ):
            raise ValueError("expert matrices do not share one physical permutation")

    trellis_bytes = sum(
        encoding.tensors["trellis"].numel()
        * encoding.tensors["trellis"].element_size()
        for encoding in encodings.values()
    )
    scale_bytes = sum(
        encoding.tensors[name].numel() * encoding.tensors[name].element_size()
        for encoding in encodings.values()
        for name in ("suh", "svh")
    )
    return QSRTExpertEncoding(
        format=format_spec,
        matrices=encodings,
        physical_permutation=physical_permutation,
        coding={
            "format": format_spec.name,
            "format_code": format_spec.code,
            "r13": format_spec.r13,
            "r2": format_spec.r2,
            "common_physical_permutation": True,
            "trellis_bytes": trellis_bytes,
            "raw_scale_bytes": scale_bytes,
            "matrices": {
                matrix: encoding.coding
                for matrix, encoding in encodings.items()
            },
        },
    )
