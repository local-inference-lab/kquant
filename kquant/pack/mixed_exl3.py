"""Production-facing TP12 heterogeneous EXL3 encoder wrapper.

The low-level EXL encoder owns the dense-H LDLQ traversal.  This module owns
the Kimi-K3 semantics around it: one common intermediate-neuron order, the
logical rate axis of each expert matrix, complete-record repacking into the
TP12 P24/P33 layout, and closure against the stored FP16 scale vectors.

The returned reconstruction is in the source checkpoint's canonical neuron
order.  The returned tensors are in physical serving order and contain one
fixed-rate trellis payload plus ``suh``/``svh``.  The MCG multiplier is a
format-level constant and is therefore not duplicated per matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch

from kquant.exl3_reference import (
    CODEBOOK_MCG,
    CODEBOOK_MUL1_E4M3,
    CODEBOOK_SQG_NORMAL_E4M3,
    EXL3_CODEBOOKS,
    decode_mixed_exl3_weight,
)

from kquant.mixed_exl3 import (
    CONTEXT_GROUP_CHANNELS,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    ModeSpec,
    ExpertFormatSpec,
    PackedTP12Trellis,
    RECORD_CHANNELS,
    SCHEMA,
    RateAxis,
    decode_tp12_exl3_weight,
    expand_group_order,
    importance_group_order,
    matrix_rate_axis,
    pack_tp12_trellis,
    record_repack_order,
    repack_encoded_records,
    repack_rate_axis,
    resolve_mode,
    tp12_record_bits,
    tp12_storage_group_order,
    unpack_tp12_trellis_edges,
    unpack_tp12_trellis_states,
)
from kquant.pack.exl3 import SIGMA_REG, make_shared_h
from kquant.tp_simulator import comparison_metrics


MATRICES = ("w1", "w3", "w2")
Layout = Literal[
    "importance_ordered",
    "tp12_balanced",
    "r05_prefix_reuse",
    "r05_pair_prefix_reuse",
]
MIXED_LAYOUTS: tuple[Layout, ...] = (
    "importance_ordered",
    "tp12_balanced",
    "r05_prefix_reuse",
    "r05_pair_prefix_reuse",
)

# Candidate-search policy rather than one matrix traversal.  The guarded
# policy keeps the established importance-ordered traversal for w1/w3 and for
# the R0 w2 control, while encoding only the R1--R5 w2 counterfactuals in the
# branchable prefix layout.  Thus the selector can never accept a layout-
# induced regression as its baseline merely to obtain encoder throughput.
SearchLayout = Literal[
    "importance_ordered",
    "tp12_balanced",
    "r05_prefix_reuse",
    "r05_guarded_reuse",
]
MIXED_SEARCH_LAYOUTS: tuple[SearchLayout, ...] = (
    *MIXED_LAYOUTS,
    "r05_guarded_reuse",
)

# R0--R5 differ only in the five least-important donor records and the five
# most-important recipient records.  Keeping those ten records at the low-K
# end of the offline LDLQ traversal leaves a 14-record K3 suffix.  LDLQ walks
# K backwards, so the suffix can be encoded once and its exact error-feedback
# state can be fanned out into all six candidates.  The serving permutation is
# unchanged: selected records are still repacked into the ordinary TP12
# P24/P33 order after encoding.
R05_PREFIX_REUSE_MAX_MODE = 5


def _r05_prefix_reuse_group_order(
    block_contexts: torch.Tensor,
    spec: ModeSpec,
) -> torch.Tensor:
    if spec.context_count != INTERMEDIATE_CHANNELS // RECORD_CHANNELS:
        raise ValueError("r05_prefix_reuse requires one context per 128-channel record")
    ordered = importance_group_order(block_contexts, spec).reshape(
        spec.context_count, -1
    )
    r = R05_PREFIX_REUSE_MAX_MODE
    # Preserve the original low-to-high order inside each importance band.
    # This perturbs the established LDLQ coordinate order much less than an
    # interleaved rate-threshold trie; the low-level encoder still reuses the
    # common suffix and branches incrementally where the nested maps permit.
    record_order = (
        tuple(range(r))
        + tuple(range(spec.context_count - r, spec.context_count))
        + tuple(range(r, spec.context_count - r))
    )
    return ordered.index_select(
        0,
        torch.tensor(record_order, dtype=torch.long, device=ordered.device),
    ).flatten()


def _r05_pair_prefix_reuse_group_order(
    block_contexts: torch.Tensor,
    spec: ModeSpec,
) -> torch.Tensor:
    """Put nested donor/recipient pairs next to each other for trie reuse.

    LDLQ traverses the returned record order backwards.  R1's pair is
    therefore visited first and is common to every shifted candidate; each
    subsequent pair introduces only one additional live trie state.  For
    R1--R5 this reduces the variable-region traversal from 40 to 30 record
    equivalents without approximating or pruning any candidate.
    """

    if spec.context_count != INTERMEDIATE_CHANNELS // RECORD_CHANNELS:
        raise ValueError(
            "r05_pair_prefix_reuse requires one context per 128-channel record"
        )
    ordered = importance_group_order(block_contexts, spec).reshape(
        spec.context_count, -1
    )
    r = R05_PREFIX_REUSE_MAX_MODE
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
class MixedMatrixPlan:
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
    mixed_tile_bits: tuple[int, ...]
    physical_tile_bits: tuple[int, ...]
    encoder_tp12_rank_bpw: tuple[float, ...]
    physical_tp12_rank_bpw: tuple[float, ...]


@dataclass
class MixedMatrixEncoding:
    """One closed physical payload and its canonical-order reconstruction."""

    reconstruction: torch.Tensor
    tensors: dict[str, torch.Tensor]
    packed: PackedTP12Trellis
    plan: MixedMatrixPlan
    coding: dict[str, object]


@dataclass
class MixedMatrixCandidate:
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
    plan: MixedMatrixPlan
    proxy: float


@dataclass
class MixedExpertEncoding:
    """Three matrices encoded under one exact physical neuron permutation."""

    format: ExpertFormatSpec
    matrices: dict[str, MixedMatrixEncoding]
    physical_permutation: torch.Tensor
    coding: dict[str, object]


def _rank_bpw(tile_bits: tuple[int, ...]) -> tuple[float, ...]:
    if len(tile_bits) != INTERMEDIATE_CHANNELS // 16:
        raise ValueError("rate map must cover the 3072-channel tile axis")
    return tuple(
        sum(tile_bits[begin : begin + 16]) / 16
        for begin in range(0, len(tile_bits), 16)
    )


def plan_mixed_matrix(
    block_contexts: torch.Tensor,
    mode: ModeSpec | str | int,
    *,
    matrix: str,
    layout: Layout = "importance_ordered",
) -> MixedMatrixPlan:
    """Plan one mixed-rate traversal and its fixed TP12 physical repack."""

    spec = resolve_mode(mode)
    rate_axis = matrix_rate_axis(matrix)
    if layout == "importance_ordered":
        encoder_group_order = importance_group_order(block_contexts, spec)
    elif layout == "tp12_balanced":
        encoder_group_order = tp12_storage_group_order(block_contexts, spec)
    elif layout == "r05_prefix_reuse":
        if spec.mode_id > R05_PREFIX_REUSE_MAX_MODE:
            raise ValueError("r05_prefix_reuse supports only R0 through R5")
        encoder_group_order = _r05_prefix_reuse_group_order(block_contexts, spec)
    elif layout == "r05_pair_prefix_reuse":
        if spec.mode_id > R05_PREFIX_REUSE_MAX_MODE:
            raise ValueError("r05_pair_prefix_reuse supports only R0 through R5")
        encoder_group_order = _r05_pair_prefix_reuse_group_order(
            block_contexts, spec
        )
    else:
        raise ValueError(f"unsupported context layout: {layout}")

    physical_group_order = tp12_storage_group_order(block_contexts, spec)
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
    mixed_tile_bits = tuple(
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
        bits for bits in tp12_record_bits(spec) for _ in range(8)
    )
    if physical_tile_bits != expected_physical_bits:
        raise ValueError("physical contexts do not match the TP12 mode schedule")

    return MixedMatrixPlan(
        matrix=matrix,
        mode=spec,
        rate_axis=rate_axis,
        layout=layout,
        encoder_permutation=encoder_permutation,
        physical_permutation=physical_permutation,
        record_repack_order=repack_order,
        mixed_tile_bits=mixed_tile_bits,
        physical_tile_bits=physical_tile_bits,
        encoder_tp12_rank_bpw=_rank_bpw(mixed_tile_bits),
        physical_tp12_rank_bpw=_rank_bpw(physical_tile_bits),
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
        raise TypeError("mixed EXL3 source must use torch.float32")
    if not bool(torch.all(torch.isfinite(source))):
        raise ValueError("mixed EXL3 source contains non-finite values")
    if not bool(torch.all(torch.isfinite(hessian))):
        raise ValueError("mixed EXL3 Hessian contains non-finite values")


def _load_quantizer_module(codebook: str = CODEBOOK_MCG) -> ModuleType:
    from exllamav3.modules.quant.exl3_lib import quantize as quantizer_module

    if codebook == CODEBOOK_SQG_NORMAL_E4M3:
        from kquant.sqg_quantizer import install_sqg_quantizer

        install_sqg_quantizer(quantizer_module)
    return quantizer_module


def make_mixed_shared_h(
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
        raise ValueError("native mixed EXL3 quantization requires a CUDA device")
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
        raise ValueError("mixed EXL3 Hessian contains non-finite values")
    contexts = block_contexts.to(device=device, dtype=torch.long)
    plan = plan_mixed_matrix(contexts, mode, matrix=matrix, layout=layout)
    encoder_hessian = hessian
    if matrix == "w2":
        permutation = plan.encoder_permutation.to(device=hessian.device)
        encoder_hessian = hessian.index_select(0, permutation).index_select(
            1, permutation
        )
    shared_h: dict[str, object] = make_shared_h(
        expected_dimension, device, encoder_hessian
    )
    shared_h["_kquant_mixed_matrix"] = matrix
    shared_h["_kquant_mixed_layout"] = layout
    shared_h["_kquant_mixed_permutation"] = (
        plan.encoder_permutation.detach().cpu().contiguous()
    )
    shared_h["_kquant_source_hessian_ptr"] = int(hessian.data_ptr())
    return shared_h


def _validate_mixed_shared_h(
    shared_h: dict[str, object],
    *,
    matrix: str,
    layout: Layout,
    plan: MixedMatrixPlan,
    hessian: torch.Tensor,
    device: torch.device,
) -> None:
    if shared_h.get("_kquant_mixed_matrix") != matrix:
        raise ValueError("mixed shared Hessian belongs to another matrix")
    if shared_h.get("_kquant_mixed_layout") != layout:
        raise ValueError("mixed shared Hessian belongs to another layout")
    if shared_h.get("_kquant_source_hessian_ptr") != int(hessian.data_ptr()):
        raise ValueError("mixed shared Hessian belongs to another source Hessian")
    permutation = shared_h.get("_kquant_mixed_permutation")
    if not isinstance(permutation, torch.Tensor) or not torch.equal(
        permutation, plan.encoder_permutation.detach().cpu()
    ):
        raise ValueError("mixed shared Hessian belongs to another neuron order")
    error_h = shared_h.get("error_H")
    expected_dimension = INTERMEDIATE_CHANNELS if matrix == "w2" else LATENT_CHANNELS
    if (
        not isinstance(error_h, torch.Tensor)
        or tuple(error_h.shape) != (expected_dimension, expected_dimension)
        or error_h.device != device
    ):
        raise ValueError("mixed shared Hessian has an invalid error_H tensor")


def _mixed_quant_args(
    plan: MixedMatrixPlan,
    *,
    matrix: str,
    layer: int,
    device: torch.device,
    shared_scale_scope: object | None,
    codebook: str = CODEBOOK_MCG,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> dict[str, object]:
    if codebook not in EXL3_CODEBOOKS:
        raise ValueError(
            f"unsupported EXL3 codebook {codebook!r}; expected one of {EXL3_CODEBOOKS}"
        )
    transform_seed = layer * 1_000_000 + MATRICES.index(matrix)
    quant_args: dict[str, object] = {
        # Preserve the established K3 regularization and scale search.  The
        # heterogeneous map controls only the trellis operation.
        "K": 3,
        "mixed_rate_axis": plan.rate_axis,
        "mixed_tile_bits": plan.mixed_tile_bits,
        "seed": transform_seed,
        "sv_seed": transform_seed + 499_979,
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "ldlq_tf32": bool(ldlq_tf32),
        "tailbite_context": int(tailbite_context),
    }
    if codebook == CODEBOOK_MCG:
        quant_args["mcg"] = True
    elif codebook == CODEBOOK_MUL1_E4M3:
        quant_args["mul1_e4m3"] = True
    elif codebook == CODEBOOK_SQG_NORMAL_E4M3:
        quant_args["sqg_e4m3_mode"] = "normal"
    else:  # pragma: no cover - guarded by EXL3_CODEBOOKS above
        raise AssertionError(f"unhandled EXL3 codebook: {codebook}")
    if matrix in ("w1", "w3"):
        quant_args["shared_input_scales_key"] = (shared_scale_scope, matrix)
        quant_args["g_scale_into_sv"] = True
    return quant_args


def _encoder_weight(
    source: torch.Tensor,
    hessian: torch.Tensor,
    plan: MixedMatrixPlan,
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


def quantize_mixed_matrix_candidates_batched(
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
    codebook: str = CODEBOOK_MCG,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> dict[str, dict[int, MixedMatrixCandidate]]:
    """Encode same-shape matrices and all requested R modes in one batch.

    The low-level encoder performs one preparation per source matrix and one
    grouped mixed-LDLQ traversal.  Returned candidates remain unpacked so the
    caller can score a complete ``(r13, r2)`` grid before closing only the
    selected physical payload.
    """

    return quantize_mixed_matrix_expert_batch(
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
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
    )[0]


def quantize_mixed_matrix_expert_batch(
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
    codebook: str = CODEBOOK_MCG,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> list[dict[str, dict[int, MixedMatrixCandidate]]]:
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
        raise ValueError("expert-batched mixed inputs must have equal nonzero lengths")

    entries: list[tuple[int, str, torch.Tensor, tuple[ModeSpec, ...]]] = []
    plans: list[dict[int, MixedMatrixPlan]] = []
    weights: list[torch.Tensor] = []
    shared_hessians: list[dict[str, object]] = []
    quant_args_groups: list[list[dict[str, object]]] = []
    common_shape: tuple[int, ...] | None = None

    for expert_index, (sources, contexts_raw, modes, hessians) in enumerate(
        zip(
            sources_by_expert,
            block_contexts_by_expert,
            modes_by_expert,
            hessians_by_expert,
        )
    ):
        if not sources or set(sources) != set(hessians):
            raise ValueError(
                "batched mixed sources and Hessians must share nonempty keys"
            )
        if not modes or len({mode.mode_id for mode in modes}) != len(modes):
            raise ValueError("batched mixed modes must be nonempty and unique")
        matrices = [matrix for matrix in MATRICES if matrix in sources]
        if set(matrices) != set(sources):
            raise ValueError("batched mixed sources contain an unknown matrix")
        shapes = {tuple(source.shape) for source in sources.values()}
        if len(shapes) != 1:
            raise ValueError("one mixed batch may contain only same-shape matrices")
        shape = next(iter(shapes))
        if common_shape is None:
            common_shape = shape
        elif shape != common_shape:
            raise ValueError("expert-batched mixed matrices must share one shape")

        contexts = contexts_raw.to(device=device, dtype=torch.long)
        for matrix in matrices:
            source = sources[matrix]
            hessian = hessians[matrix]
            _validate_source_and_hessian(source, hessian, matrix)
            if source.device != device:
                raise ValueError(
                    f"batched mixed source {expert_index}.{matrix} is not on {device}"
                )
            if matrix in ("w1", "w3") and shared_scale_scope is None:
                raise ValueError(f"{matrix} requires a shared_scale_scope")
            matrix_plans = {
                mode.mode_id: plan_mixed_matrix(
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
            entries.append((expert_index, matrix, source, modes))
            plans.append(matrix_plans)
            weights.append(weight)
            shared_hessians.append(
                make_mixed_shared_h(
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
                    _mixed_quant_args(
                        matrix_plans[mode.mode_id],
                        matrix=matrix,
                        layer=layer,
                        device=device,
                        shared_scale_scope=shared_scale_scope,
                        codebook=codebook,
                        ldlq_tf32=ldlq_tf32,
                        tailbite_context=tailbite_context,
                    )
                    for mode in modes
                ]
            )

    module = (
        _load_quantizer_module(codebook)
        if quantizer_module is None
        else quantizer_module
    )
    batch_api = getattr(module, "quantize_exl3_mixed_batch", None)
    if batch_api is None:
        raise ValueError("EXL encoder lacks quantize_exl3_mixed_batch")
    raw_groups = batch_api(
        weights,
        shared_hessians,
        quant_args_groups,
        return_weight_q=True,
    )
    if len(raw_groups) != len(entries):
        raise ValueError("mixed EXL batch returned the wrong source count")

    result: list[dict[str, dict[int, MixedMatrixCandidate]]] = [
        {} for _ in range(expert_count)
    ]
    finite_checks: list[torch.Tensor] = []
    finite_labels: list[tuple[int, str, int]] = []
    for entry_index, ((expert_index, matrix, source, modes), raw_group) in enumerate(
        zip(entries, raw_groups)
    ):
        if len(raw_group) != len(modes):
            raise ValueError("mixed EXL batch returned the wrong candidate count")
        matrix_result: dict[int, MixedMatrixCandidate] = {}
        for mode, raw in zip(modes, raw_group):
            plan = plans[entry_index][mode.mode_id]
            encoder_weight = raw["weight_q"]
            if not isinstance(encoder_weight, torch.Tensor):
                raise ValueError("mixed EXL batch omitted its reconstruction")
            reconstruction = torch.empty_like(source)
            if matrix == "w2":
                reconstruction[:, plan.encoder_permutation] = encoder_weight.T
            else:
                reconstruction[plan.encoder_permutation] = encoder_weight.T
            # Defer this device scalar read until every candidate has queued
            # its check.  Reading one boolean here serialized hundreds of
            # otherwise independent reconstructions in an R0--R5 batch.
            finite_checks.append(torch.all(torch.isfinite(reconstruction)))
            finite_labels.append((expert_index, matrix, mode.mode_id))
            proxy = float(raw["proxy"])
            if not math.isfinite(proxy):
                raise ValueError("mixed EXL batch produced a non-finite proxy")
            matrix_result[mode.mode_id] = MixedMatrixCandidate(
                reconstruction=reconstruction,
                encoded=raw["encoded"],
                tensors={"suh": raw["suh"], "svh": raw["svh"]},
                plan=plan,
                proxy=proxy,
            )
        result[expert_index][matrix] = matrix_result
    finite = torch.stack(finite_checks).cpu()
    if not bool(torch.all(finite)):
        first = int(torch.nonzero(~finite, as_tuple=False)[0])
        expert_index, matrix, mode_id = finite_labels[first]
        raise ValueError(
            "mixed EXL batch produced a non-finite reconstruction for "
            f"expert-index {expert_index} {matrix} R{mode_id}"
        )
    return result


def finalize_mixed_matrix_candidate(
    candidate: MixedMatrixCandidate,
    *,
    layer: int,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_MCG,
    tailbite_context: int = 128,
) -> MixedMatrixEncoding:
    """Pack and independently decode one selected mixed-rate candidate."""

    plan = candidate.plan
    physical = repack_encoded_records(
        candidate.encoded, plan.record_repack_order, plan.rate_axis
    )
    packed = pack_tp12_trellis(
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
    edge_symbols = unpack_tp12_trellis_edges(packed)
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
    decoded_states = unpack_tp12_trellis_states(packed)
    state_closes = torch.all(decoded_states == physical.to(torch.int16))
    edge_ok, state_ok = torch.stack((edge_closes, state_closes)).cpu().tolist()
    if not edge_ok:
        raise ValueError(
            "mixed trellis reference edge unpack failed closure for "
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
            "mixed trellis state reconstruction failed exact closure for "
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
    decoded_physical = decode_mixed_exl3_weight(
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
        raise ValueError("stored mixed-EXL decode produced non-finite values")

    reconstruction = torch.empty_like(candidate.reconstruction)
    physical_permutation = plan.physical_permutation
    if plan.matrix == "w2":
        reconstruction[:, physical_permutation] = decoded_physical.T
    else:
        reconstruction[physical_permutation] = decoded_physical.T
    scoring_vs_stored = comparison_metrics(candidate.reconstruction, reconstruction)
    if not all(math.isfinite(value) for value in scoring_vs_stored.values()):
        raise ValueError("selected mixed-EXL scoring reconstruction did not close")

    index_bits = tensors["trellis"].numel() * tensors["trellis"].element_size() * 8
    scale_bits = sum(
        tensors[name].numel() * tensors[name].element_size() * 8
        for name in ("suh", "svh")
    )
    if index_bits != reconstruction.numel() * 3:
        raise ValueError("TP12 mixed payload is not exactly three trellis bpw")
    transform_seed = layer * 1_000_000 + MATRICES.index(plan.matrix)
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
        "tp12_rank_trellis_bpw": list(plan.encoder_tp12_rank_bpw),
        "postpack_tp12_rank_trellis_bpw": list(plan.physical_tp12_rank_bpw),
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
            "input_sign": transform_seed,
            "output_sign": transform_seed + 499_979,
        },
    }
    return MixedMatrixEncoding(
        reconstruction=reconstruction,
        tensors=tensors,
        packed=packed,
        plan=plan,
        coding=coding,
    )


def quantize_mixed_matrix(
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
    codebook: str = CODEBOOK_MCG,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> MixedMatrixEncoding:
    """Run dense-H heterogeneous LDLQ and close the stored TP12 payload.

    ``shared_scale_scope`` is required for ``w1``/``w3`` because their hidden
    ``suh`` vector is stored once per layer/matrix.  All experts and all mode
    candidates belonging to one layer artifact must use the same scope.
    """

    if not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must be in 1..92")
    if device.type != "cuda":
        raise ValueError("native mixed EXL3 quantization requires a CUDA device")
    if source.device != device:
        raise ValueError(
            f"mixed EXL3 source must be resident on {device}, got {source.device}"
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
    plan = plan_mixed_matrix(contexts, mode, matrix=matrix, layout=layout)
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
        _validate_mixed_shared_h(
            shared_h_data,
            matrix=matrix,
            layout=layout,
            plan=plan,
            hessian=hessian,
            device=device,
        )
        shared_h = shared_h_data
    transform_seed = layer * 1_000_000 + MATRICES.index(matrix)
    quant_args = _mixed_quant_args(
        plan,
        matrix=matrix,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        codebook=codebook,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
    )

    packed_holder: dict[str, PackedTP12Trellis] = {}

    def pack_mixed(encoded: torch.Tensor, _quant_args: dict) -> torch.Tensor:
        physical = repack_encoded_records(
            encoded, plan.record_repack_order, plan.rate_axis
        )
        packed = pack_tp12_trellis(
            physical,
            plan.mode,
            rate_axis=plan.rate_axis,
            schema=logical_trellis_schema,
        )
        edge_symbols = unpack_tp12_trellis_edges(packed)
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
            raise ValueError("mixed trellis reference edge unpack failed closure")
        if not torch.equal(
            unpack_tp12_trellis_states(packed), physical.to(torch.int16)
        ):
            raise ValueError("mixed trellis state reconstruction failed exact closure")
        packed_holder["packed"] = packed
        return packed.payload

    quant_args["pack_trellis_fn"] = pack_mixed
    module = (
        _load_quantizer_module(codebook)
        if quantizer_module is None
        else quantizer_module
    )
    quantized, proxy, raw_tensors = module.quantize_exl3(
        weight,
        shared_h,
        quant_args,
        True,
        progress_str="",
    )
    packed = packed_holder.get("packed")
    if packed is None:
        raise ValueError("mixed packer was not called; fallback encoding is unsupported")

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

    decoded_physical = decode_tp12_exl3_weight(
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
        raise ValueError("stored mixed-EXL decode produced non-finite values")

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
        raise ValueError("TP12 mixed payload is not exactly three trellis bpw")
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
        "tp12_rank_trellis_bpw": list(plan.encoder_tp12_rank_bpw),
        "postpack_tp12_rank_trellis_bpw": list(plan.physical_tp12_rank_bpw),
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
            "input_sign": transform_seed,
            "output_sign": transform_seed + 499_979,
        },
    }
    return MixedMatrixEncoding(
        reconstruction=reconstruction,
        tensors=tensors,
        packed=packed,
        plan=plan,
        coding=coding,
    )


def quantize_mixed_expert(
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
    codebook: str = CODEBOOK_MCG,
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> MixedExpertEncoding:
    """Encode all three expert matrices with one physical neuron order."""

    if format_spec.is_mxfp4:
        raise ValueError("quantize_mixed_expert requires a compressed format")
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
        matrix: quantize_mixed_matrix(
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
    return MixedExpertEncoding(
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
