"""Offline-streamed all-expert candidates for the QSRT codec."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Literal, Protocol, Sequence

import torch
import torch.nn.functional as F

from kquant.capture import LayerSamples
from kquant.exl3_reference import CODEBOOK_QSRT_E4M3
from kquant.qsrt_candidates import (
    PERMUTATION_POLICIES,
    PermutationPolicy,
    RatePairSelection,
    RequestPartition,
    RoutedRows,
    build_expert_hessians,
    deterministic_expert_seed,
    functional_sse_by_request,
    permutation_policy_contexts,
    select_expert_rows,
    select_phase1_rate_pair,
)
from kquant.qsrt import (
    SCHEMA,
    ExpertFormatSpec,
    PHASE1_MODE_IDS,
    RATE_TRANSFER_MODES,
    RECORDS_PER_EXPERT,
)
from kquant.pack.qsrt_encoder import (
    Layout,
    MATRICES,
    QSRTExpertEncoding,
    QSRTMatrixCandidate,
    QSRTMatrixEncoding,
    QSRTTransformSeeds,
    SearchLayout,
    finalize_qsrt_matrix_candidate,
    quantize_qsrt_matrix_candidates_batched,
    quantize_qsrt_matrix_expert_batch,
)
from kquant.tp_simulator import situ


CANDIDATE_POOL_KIND = "kquant_kimi_k3_qsrt_candidate_pool"
CANDIDATE_POOL_SCHEMA_VERSION = 6
OFFICIAL_SOURCE_DAMAGE_METRIC = "official_source_excess_sse"
HessianPolicy = Literal["captured_blend", "identity"]
HESSIAN_POLICIES: tuple[HessianPolicy, ...] = ("captured_blend", "identity")
FOLDED_SCALE_GRID: tuple[float, ...] = (0.85, 0.925, 1.0, 1.075, 1.15)


class MatrixStore(Protocol):
    """Read-only source that materializes one official matrix per call."""

    def load_matrix(
        self,
        layer: int,
        expert: int,
        matrix: str,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor: ...


@dataclass
class QSRTCandidateEncoding:
    """Selected payload plus the complete phase-1 selection evidence."""

    expert: QSRTExpertEncoding
    selection: RatePairSelection
    block_contexts: torch.Tensor
    block_scores: torch.Tensor
    context_basis: str
    covariance: dict[str, object]
    evaluated_modes: tuple[int, ...]
    evaluated_formats: tuple[tuple[int, int], ...]
    fit_sse: dict[tuple[int, int], torch.Tensor]
    confirmation_sse: dict[tuple[int, int], torch.Tensor]
    fit_reference_energy: torch.Tensor
    confirmation_reference_energy: torch.Tensor
    fit_counts: torch.Tensor
    confirmation_counts: torch.Tensor
    mode_coding: dict[tuple[int, int], dict[str, object]]

    def metadata(self) -> dict[str, object]:
        return {
            "selection": asdict(self.selection),
            "context_basis": self.context_basis,
            "covariance": self.covariance,
            "evaluated_modes": list(self.evaluated_modes),
            "mode_coding": {
                f"R{r13}/R{r2}": value
                for (r13, r2), value in self.mode_coding.items()
            },
        }


def _source_middle(
    gate_projection: torch.Tensor, up_projection: torch.Tensor
) -> torch.Tensor:
    return situ(gate_projection, up_projection)


def _folded_intermediate_conditioning(
    block_scores: torch.Tensor,
    block_contexts: torch.Tensor,
    *,
    power: float,
) -> torch.Tensor | None:
    """Derive a metadata-free per-record output conditioning profile.

    The source rows are divided by this profile before w1/w3 quantization and
    the exact inverse is folded into their existing expert-local ``svh``
    vectors.  Factors are constant over each final 128-channel Hadamard block,
    so the fold commutes with the transform and requires no new runtime tensor.
    """

    if not math.isfinite(power):
        raise ValueError("folded scale power must be finite")
    if power == 0.0:
        return None
    if block_scores.ndim != 1 or block_scores.numel() != 3072 // 4:
        raise ValueError("folded conditioning requires 768 group scores")
    if not bool(torch.all(torch.isfinite(block_scores))) or bool(
        torch.any(block_scores < 0)
    ):
        raise ValueError("folded conditioning scores must be finite and nonnegative")
    contexts = block_contexts.to(device=block_scores.device, dtype=torch.long)
    group_energy = block_scores.float()
    record_energy = torch.zeros(
        RECORDS_PER_EXPERT, device=block_scores.device, dtype=torch.float32
    ).scatter_add_(0, contexts, group_energy.float())
    counts = torch.bincount(contexts, minlength=RECORDS_PER_EXPERT).float()
    record_rms = (record_energy / counts).clamp_min(1e-30).sqrt()
    geometric_mean = record_rms.log().mean().exp()
    # A smaller folded scale magnifies important rows inside the trellis
    # search and shrinks their absolute error again when folded back into svh.
    # This is an output-metric conditioning heuristic, not a new runtime scale.
    desired = (record_rms / geometric_mean).pow(-power)
    grid = torch.tensor(FOLDED_SCALE_GRID, device=desired.device)
    selected = grid[(desired[:, None] - grid[None, :]).abs().argmin(dim=1)]
    return selected.index_select(0, contexts).repeat_interleave(4).contiguous()


def _load_source_matrix(
    store: MatrixStore,
    layer: int,
    expert: int,
    matrix: str,
    device: torch.device,
) -> torch.Tensor:
    source = store.load_matrix(layer, expert, matrix, device=device)
    if source.device != device:
        raise ValueError(
            f"official source store returned {source.device}, expected {device}"
        )
    if source.dtype != torch.float32:
        raise ValueError(
            f"official source store returned {source.dtype}, expected torch.float32"
        )
    return source.contiguous()


def _split_rows(rows: RoutedRows, requests: dict[int, str]) -> torch.Tensor:
    allowed = torch.tensor(
        sorted(requests), dtype=rows.request_steps.dtype, device=rows.request_steps.device
    )
    return torch.isin(rows.request_steps, allowed)


def _empty_metrics(requests: dict[int, str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(requests)
    return (
        torch.zeros(count, dtype=torch.float64),
        torch.zeros(count, dtype=torch.float64),
        torch.zeros(count, dtype=torch.int64),
    )


@dataclass(frozen=True)
class _DeferredFunctionalRows:
    """Static routed-metric state shared by every candidate for one fold."""

    device_mask: torch.Tensor
    reference: torch.Tensor
    route_weights: torch.Tensor
    request_indices: torch.Tensor
    reference_energy: torch.Tensor
    counts: torch.Tensor
    request_count: int


def _prepare_deferred_functional_rows(
    reference: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    mask: torch.Tensor,
    requests: dict[int, str],
    *,
    device: torch.device,
) -> _DeferredFunctionalRows:
    """Prepare invariant metric terms without changing their arithmetic."""

    device_mask = mask.to(device=device)
    selected_reference = reference[device_mask].double()
    selected_gates = gates[device_mask]
    selected_steps = request_steps[mask]
    ordered_steps = sorted(map(int, requests))
    positions = {step: index for index, step in enumerate(ordered_steps)}
    try:
        request_indices = torch.tensor(
            [positions[int(step)] for step in selected_steps.cpu().tolist()],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(
            f"metric row has unmapped request epoch {exc.args[0]}"
        ) from exc
    route_weights = selected_gates.double().square().cpu()
    row_reference = (
        selected_reference.square().sum(dim=1).cpu() * route_weights
    )
    reference_energy = torch.zeros(len(ordered_steps), dtype=torch.float64)
    counts = torch.zeros(len(ordered_steps), dtype=torch.int64)
    reference_energy.scatter_add_(0, request_indices, row_reference)
    counts.scatter_add_(
        0,
        request_indices,
        torch.ones_like(request_indices, dtype=torch.int64),
    )
    return _DeferredFunctionalRows(
        device_mask=device_mask,
        reference=selected_reference,
        route_weights=route_weights,
        request_indices=request_indices,
        reference_energy=reference_energy,
        counts=counts,
        request_count=len(ordered_steps),
    )


def _defer_functional_row_sse(
    plan: _DeferredFunctionalRows,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Compute per-row SSE on CUDA and defer its host synchronization."""

    delta = candidate[plan.device_mask].double() - plan.reference
    return delta.square().sum(dim=1)


def _finish_deferred_functional_sse(
    plan: _DeferredFunctionalRows,
    keys: Sequence[tuple[int, int]],
    row_sse: Sequence[torch.Tensor],
) -> dict[tuple[int, int], torch.Tensor]:
    """Transfer all candidate rows once, then reproduce the CPU reductions."""

    if len(keys) != len(row_sse):
        raise ValueError("deferred functional metric keys and rows differ")
    if not keys:
        return {}
    rows_cpu = torch.stack(tuple(row_sse)).cpu()
    result: dict[tuple[int, int], torch.Tensor] = {}
    for key, values in zip(keys, rows_cpu):
        sse = torch.zeros(plan.request_count, dtype=torch.float64)
        sse.scatter_add_(
            0,
            plan.request_indices,
            values * plan.route_weights,
        )
        result[key] = sse
    return result


def _mode_evidence(encodings: dict[str, QSRTMatrixEncoding]) -> dict[str, object]:
    trellis_bytes = sum(
        value.tensors["trellis"].numel()
        * value.tensors["trellis"].element_size()
        for value in encodings.values()
    )
    scale_bytes = sum(
        value.tensors[name].numel() * value.tensors[name].element_size()
        for value in encodings.values()
        for name in ("suh", "svh")
    )
    return {
        "trellis_bytes": trellis_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes_before_layer_deduplication": trellis_bytes + scale_bytes,
        "matrices": {
            matrix: {
                "proxy": float(value.coding["proxy"]),
                "mode": value.coding["mode"],
                "trellis_descriptor": value.coding["trellis_descriptor"],
            }
            for matrix, value in encodings.items()
        },
    }


def _candidate_mode_evidence(encodings: dict[str, object]) -> dict[str, object]:
    trellis_bytes = sum(
        value.reconstruction.numel() * 3 // 8 for value in encodings.values()
    )
    scale_bytes = sum(
        value.tensors[name].numel() * value.tensors[name].element_size()
        for value in encodings.values()
        for name in ("suh", "svh")
    )
    return {
        "trellis_bytes": trellis_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes_before_layer_deduplication": trellis_bytes + scale_bytes,
        "matrices": {
            matrix: {
                "proxy": float(value.proxy),
                "mode": value.plan.mode.name,
                "mode_id": value.plan.mode.mode_id,
                "rate_axis": value.plan.rate_axis,
            }
            for matrix, value in encodings.items()
        },
    }


def _candidate_middle_by_r13(
    inputs: torch.Tensor,
    upstream: dict[str, dict[int, QSRTMatrixCandidate]],
    evaluated_modes: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Replay exact candidate reconstructions through Kimi's SiTU."""

    return {
        int(r13): _source_middle(
            F.linear(inputs, upstream["w1"][int(r13)].reconstruction),
            F.linear(inputs, upstream["w3"][int(r13)].reconstruction),
        )
        for r13 in evaluated_modes
    }


def _conditional_h2_by_r13(
    inputs: torch.Tensor,
    gates: torch.Tensor,
    fit_mask_device: torch.Tensor,
    middle_by_r13: dict[int, torch.Tensor],
    *,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    device: torch.device,
    hessian_policy: HessianPolicy,
    have_local_support: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, object]]:
    """Build one support-aware down-projection covariance per upstream mode."""

    h2_by_r13: dict[int, torch.Tensor] = {}
    evidence_by_r13: dict[str, object] = {}
    identity_h2 = torch.eye(
        global_h2.shape[0], dtype=torch.float32, device=device
    )
    for r13, middle in middle_by_r13.items():
        if hessian_policy == "identity" or not have_local_support:
            h2_by_r13[r13] = identity_h2
            evidence_by_r13[f"R{r13}"] = {
                "h2_shrinkage_policy": "identity_fallback",
                "h2_local_alpha": 0.0,
                "h2_effective_sample_size": 0.0,
                "h2_identity_scale": 1.0,
            }
            continue
        _, h2, evidence = build_expert_hessians(
            inputs[fit_mask_device],
            gates[fit_mask_device],
            middle[fit_mask_device],
            global_h13=global_h13,
            global_h2=global_h2,
            device=device,
        )
        h2_by_r13[r13] = h2
        evidence_by_r13[f"R{r13}"] = evidence
    return h2_by_r13, evidence_by_r13


def _quantize_conditional_down_grid(
    source_w2_by_expert: Sequence[torch.Tensor],
    block_contexts_by_expert: Sequence[torch.Tensor],
    mode_specs_by_expert: Sequence[tuple[object, ...]],
    h2_by_r13_by_expert: Sequence[dict[int, torch.Tensor]],
    *,
    layer: int,
    device: torch.device,
    shared_scale_scope: object,
    quantizer_module: object | None,
    codebook: str,
    layout: SearchLayout,
    ldlq_tf32: bool,
    tailbite_context: int,
    transform_seeds_by_expert: Sequence[
        Mapping[str, QSRTTransformSeeds]
    ] | None = None,
) -> list[dict[int, dict[int, QSRTMatrixCandidate]]]:
    """Encode the exhaustive ``w2[r13, r2]`` grid in flattened GPU batches."""

    count = len(source_w2_by_expert)
    if not (
        count
        == len(block_contexts_by_expert)
        == len(mode_specs_by_expert)
        == len(h2_by_r13_by_expert)
    ):
        raise ValueError("conditional down-grid inputs have inconsistent lengths")
    if transform_seeds_by_expert is None:
        transform_seeds_by_expert = [{} for _ in range(count)]
    elif len(transform_seeds_by_expert) != count:
        raise ValueError("conditional down-grid transform seeds have the wrong length")

    flat_keys: list[tuple[int, int]] = []
    flat_sources: list[dict[str, torch.Tensor]] = []
    flat_contexts: list[torch.Tensor] = []
    flat_modes: list[tuple[object, ...]] = []
    flat_hessians: list[dict[str, torch.Tensor]] = []
    flat_transform_seeds: list[Mapping[str, QSRTTransformSeeds]] = []
    for expert_index, (source, contexts, modes, h2_by_r13) in enumerate(
        zip(
            source_w2_by_expert,
            block_contexts_by_expert,
            mode_specs_by_expert,
            h2_by_r13_by_expert,
        )
    ):
        mode_ids = tuple(int(mode.mode_id) for mode in modes)
        h2_bases = tuple(h2_by_r13)
        if h2_bases != mode_ids:
            raise ValueError("conditional H2 modes differ from upstream modes")
        for r13 in h2_bases:
            flat_keys.append((expert_index, r13))
            flat_sources.append({"w2": source})
            flat_contexts.append(contexts)
            flat_modes.append(modes)
            flat_hessians.append({"w2": h2_by_r13[r13]})
            flat_transform_seeds.append(transform_seeds_by_expert[expert_index])

    guarded_reuse = layout == "qsrt_guarded_reuse"
    if guarded_reuse:
        flat_candidates = quantize_qsrt_matrix_expert_batch(
            flat_sources,
            flat_contexts,
            modes_by_expert=[(modes[0],) for modes in flat_modes],
            hessians_by_expert=flat_hessians,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout="importance_ordered",
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
            transform_seeds_by_expert=flat_transform_seeds,
        )
        shifted_indices = [
            index for index, modes in enumerate(flat_modes) if len(modes) > 1
        ]
        if shifted_indices:
            shifted = quantize_qsrt_matrix_expert_batch(
                [flat_sources[index] for index in shifted_indices],
                [flat_contexts[index] for index in shifted_indices],
                modes_by_expert=[flat_modes[index][1:] for index in shifted_indices],
                hessians_by_expert=[flat_hessians[index] for index in shifted_indices],
                layer=layer,
                device=device,
                shared_scale_scope=shared_scale_scope,
                quantizer_module=quantizer_module,
                codebook=codebook,
                layout="qsrt_pair_prefix_reuse",
                ldlq_tf32=ldlq_tf32,
                tailbite_context=tailbite_context,
                transform_seeds_by_expert=[
                    flat_transform_seeds[index] for index in shifted_indices
                ],
            )
            for index, extra in zip(shifted_indices, shifted):
                flat_candidates[index]["w2"].update(extra["w2"])
    else:
        flat_candidates = quantize_qsrt_matrix_expert_batch(
            flat_sources,
            flat_contexts,
            modes_by_expert=flat_modes,
            hessians_by_expert=flat_hessians,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout=layout,
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
            transform_seeds_by_expert=flat_transform_seeds,
        )

    result: list[dict[int, dict[int, QSRTMatrixCandidate]]] = [
        {} for _ in range(count)
    ]
    for (expert_index, r13), candidates in zip(flat_keys, flat_candidates):
        result[expert_index][r13] = candidates["w2"]
    return result


def encode_phase1_expert(
    store: MatrixStore,
    samples: LayerSamples,
    *,
    layer: int,
    expert: int,
    partition: RequestPartition,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    fallback_block_contexts: torch.Tensor,
    fallback_block_scores: torch.Tensor,
    device: torch.device,
    shared_scale_scope: object,
    mode_ids: Sequence[int] = PHASE1_MODE_IDS,
    quantizer_module: object | None = None,
    min_fit_documents: int = 6,
    min_confirmation_documents: int = 4,
    bootstrap_replicates: int = 2_000,
    minimum_improvement: float = 0.0,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_QSRT_E4M3,
    layout: SearchLayout = "importance_ordered",
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    hessian_policy: HessianPolicy = "captured_blend",
    permutation_policy: PermutationPolicy = "h2_reverse",
    folded_scale_power: float = 0.0,
    transform_seeds: Mapping[str, QSRTTransformSeeds] | None = None,
) -> QSRTCandidateEncoding:
    """Encode and conservatively select one official-source expert.

    The official model is never instantiated.  ``w1`` and ``w3`` are streamed
    together so their same-shape R candidates can share one batched traversal;
    they are released before ``w2`` is loaded.
    """

    if device.type != "cuda":
        raise ValueError("QSRT candidate encoding requires CUDA")
    if not 1 <= layer <= 92 or not 0 <= expert < 896:
        raise ValueError("invalid K3 layer/expert assignment")
    modes = tuple(int(mode) for mode in mode_ids)
    if not modes or modes[0] != 0 or len(set(modes)) != len(modes):
        raise ValueError("mode_ids must be unique and begin with R0")
    if any(not 0 <= mode < len(RATE_TRANSFER_MODES) for mode in modes):
        raise ValueError("mode_ids contain an unsupported transfer count")
    if hessian_policy not in HESSIAN_POLICIES:
        raise ValueError(f"unsupported Hessian policy: {hessian_policy}")
    if permutation_policy not in PERMUTATION_POLICIES:
        raise ValueError(f"unsupported permutation policy: {permutation_policy}")

    all_rows = select_expert_rows(samples, expert, partition.all)
    fit_mask = _split_rows(all_rows, partition.fit)
    confirmation_mask = _split_rows(all_rows, partition.confirmation)
    fit_documents = int(torch.unique(all_rows.request_steps[fit_mask]).numel())
    confirmation_documents = int(
        torch.unique(all_rows.request_steps[confirmation_mask]).numel()
    )
    inputs = all_rows.inputs.float().to(device=device)
    gates = all_rows.gates.float().to(device=device)

    source_w1 = _load_source_matrix(store, layer, expert, "w1", device)
    source_w3 = _load_source_matrix(store, layer, expert, "w3", device)
    if all_rows.rows:
        source_gate = F.linear(inputs, source_w1)
        source_up = F.linear(inputs, source_w3)
        source_middle = _source_middle(source_gate, source_up)
        del source_gate, source_up
    else:
        source_middle = torch.empty(
            (0, global_h2.shape[0]), dtype=torch.float32, device=device
        )

    have_local_support = fit_documents >= min_fit_documents
    fit_mask_device = fit_mask.to(device=device)
    if have_local_support:
        block_contexts, block_scores = permutation_policy_contexts(
            source_middle[fit_mask_device],
            gates[fit_mask_device],
            policy=permutation_policy,
        )
        h13 = global_h13
        covariance: dict[str, object] = {
            "gate_square_sum": float(
                gates[fit_mask_device].square().double().sum()
            ),
            "h13_local_alpha": 0.0,
        }
        context_basis = (
            f"official_source_post_situ_fit_documents:{permutation_policy}"
        )
        covariance = {
            **covariance,
            "fit_rows": int(fit_mask.sum()),
            "fit_documents": fit_documents,
        }
    else:
        block_contexts = fallback_block_contexts.clone()
        block_scores = fallback_block_scores.clone()
        h13 = global_h13
        context_basis = "fallback_context_order_no_local_support"
        covariance = {
            "fit_rows": int(fit_mask.sum()),
            "fit_documents": fit_documents,
            "gate_square_sum": float(gates[fit_mask_device].square().sum()),
            "h13_local_alpha": 0.0,
        }

    can_confirm = confirmation_documents >= min_confirmation_documents
    evaluated_modes = modes if have_local_support and can_confirm else (0,)
    evaluated_formats = tuple(
        (r13, r2) for r13 in evaluated_modes for r2 in evaluated_modes
    )
    block_contexts_device = block_contexts.to(device=device, dtype=torch.long)
    intermediate_conditioning = _folded_intermediate_conditioning(
        block_scores.to(device=device),
        block_contexts_device,
        power=folded_scale_power,
    )
    if intermediate_conditioning is not None:
        covariance["folded_scale_power"] = float(folded_scale_power)
        covariance["folded_scale_min"] = float(intermediate_conditioning.min())
        covariance["folded_scale_max"] = float(intermediate_conditioning.max())
        context_basis += "+folded_128_channel_conditioning"

    mode_specs = tuple(RATE_TRANSFER_MODES[mode] for mode in evaluated_modes)
    guarded_reuse = layout == "qsrt_guarded_reuse"
    if guarded_reuse and any(mode > 2 for mode in evaluated_modes):
        raise ValueError("qsrt_guarded_reuse supports only R0 through R2")
    upstream_layout: Layout = "importance_ordered" if guarded_reuse else layout
    upstream_candidates = quantize_qsrt_matrix_candidates_batched(
        {"w1": source_w1, "w3": source_w3},
        block_contexts_device,
        modes=mode_specs,
        hessians={"w1": h13, "w3": h13},
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        codebook=codebook,
        layout=upstream_layout,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        transform_seeds=transform_seeds,
        intermediate_conditioning=intermediate_conditioning,
    )
    del source_w1, source_w3

    middle_by_r13 = _candidate_middle_by_r13(
        inputs, upstream_candidates, evaluated_modes
    )
    h2_by_r13, h2_evidence = _conditional_h2_by_r13(
        inputs,
        gates,
        fit_mask_device,
        middle_by_r13,
        global_h13=global_h13,
        global_h2=global_h2,
        device=device,
        hessian_policy=hessian_policy,
        have_local_support=have_local_support,
    )
    covariance.update(
        {
            "basis": (
                "identity"
                if hessian_policy == "identity"
                else (
                    "decoded_candidate_post_situ_adaptive_shrinkage"
                    if have_local_support
                    else "identity_support_fallback"
                )
            ),
            "h2_by_r13": h2_evidence,
        }
    )
    context_basis += "+decoded_candidate_post_situ_h2"

    source_w2 = _load_source_matrix(store, layer, expert, "w2", device)
    if all_rows.rows:
        reference_output = F.linear(source_middle, source_w2)
    else:
        reference_output = torch.empty(
            (0, global_h13.shape[0]), dtype=torch.float32, device=device
        )
    down_candidates = _quantize_conditional_down_grid(
        [source_w2],
        [block_contexts_device],
        [mode_specs],
        [h2_by_r13],
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        codebook=codebook,
        layout=layout,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        transform_seeds_by_expert=[transform_seeds or {}],
    )[0]
    fit_sse: dict[tuple[int, int], torch.Tensor] = {}
    confirmation_sse: dict[tuple[int, int], torch.Tensor] = {}
    fit_reference = fit_counts = confirmation_reference = confirmation_counts = None
    mode_coding: dict[tuple[int, int], dict[str, object]] = {}
    for rate_pair in evaluated_formats:
        r13, r2 = rate_pair
        candidate_output = F.linear(
            middle_by_r13[r13], down_candidates[r13][r2].reconstruction
        )
        fit_metric = (
            functional_sse_by_request(
                reference_output[fit_mask.to(device=device)],
                candidate_output[fit_mask.to(device=device)],
                gates[fit_mask.to(device=device)],
                all_rows.request_steps[fit_mask],
                partition.fit,
            )
            if int(fit_mask.sum())
            else _empty_metrics(partition.fit)
        )
        confirmation_metric = (
            functional_sse_by_request(
                reference_output[confirmation_mask.to(device=device)],
                candidate_output[confirmation_mask.to(device=device)],
                gates[confirmation_mask.to(device=device)],
                all_rows.request_steps[confirmation_mask],
                partition.confirmation,
            )
            if int(confirmation_mask.sum())
            else _empty_metrics(partition.confirmation)
        )
        fit_sse[rate_pair] = fit_metric[0]
        confirmation_sse[rate_pair] = confirmation_metric[0]
        if fit_reference is None:
            fit_reference, fit_counts = fit_metric[1], fit_metric[2]
            confirmation_reference, confirmation_counts = (
                confirmation_metric[1],
                confirmation_metric[2],
            )
        elif not (
            torch.equal(fit_reference, fit_metric[1])
            and torch.equal(fit_counts, fit_metric[2])
            and torch.equal(confirmation_reference, confirmation_metric[1])
            and torch.equal(confirmation_counts, confirmation_metric[2])
        ):
            raise ValueError("mode candidates changed metric support or reference energy")
        mode_coding[rate_pair] = _candidate_mode_evidence(
            {
                "w1": upstream_candidates["w1"][r13],
                "w3": upstream_candidates["w3"][r13],
                "w2": down_candidates[r13][r2],
            }
        )
        del candidate_output

    del source_w2

    assert fit_reference is not None and fit_counts is not None
    assert confirmation_reference is not None and confirmation_counts is not None
    selection = select_phase1_rate_pair(
        fit_sse,
        confirmation_sse,
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        modes=evaluated_formats,
        min_fit_documents=min_fit_documents,
        min_confirmation_documents=min_confirmation_documents,
        minimum_improvement=minimum_improvement,
        bootstrap_replicates=bootstrap_replicates,
        seed=deterministic_expert_seed(layer, expert),
    )
    selected_r13, selected_r2 = selection.selected
    selected_candidates = {
        "w1": upstream_candidates["w1"][selected_r13],
        "w3": upstream_candidates["w3"][selected_r13],
        "w2": down_candidates[selected_r13][selected_r2],
    }
    selected_encodings = {
        matrix: finalize_qsrt_matrix_candidate(
            value,
            layer=layer,
            logical_trellis_schema=logical_trellis_schema,
            codebook=codebook,
            tailbite_context=tailbite_context,
        )
        for matrix, value in selected_candidates.items()
    }
    physical_permutation = selected_encodings["w1"].plan.physical_permutation
    for matrix in ("w3", "w2"):
        if not torch.equal(
            selected_encodings[matrix].plan.physical_permutation,
            physical_permutation,
        ):
            raise ValueError("selected matrices do not share one physical permutation")
    expert_encoding = QSRTExpertEncoding(
        format=ExpertFormatSpec.compressed(selected_r13, selected_r2),
        matrices=selected_encodings,
        physical_permutation=physical_permutation,
        coding={
            "format": ExpertFormatSpec.compressed(
                selected_r13, selected_r2
            ).name,
            "r13": selected_r13,
            "r2": selected_r2,
            "codebook": codebook,
            "tailbite_context": int(tailbite_context),
            **_mode_evidence(selected_encodings),
        },
    )
    return QSRTCandidateEncoding(
        expert=expert_encoding,
        selection=selection,
        block_contexts=block_contexts.detach().cpu().to(torch.uint8).contiguous(),
        block_scores=block_scores.detach().cpu().float().contiguous(),
        context_basis=context_basis,
        covariance=covariance,
        evaluated_modes=evaluated_modes,
        evaluated_formats=evaluated_formats,
        fit_sse=fit_sse,
        confirmation_sse=confirmation_sse,
        fit_reference_energy=fit_reference,
        confirmation_reference_energy=confirmation_reference,
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        mode_coding=mode_coding,
    )


def encode_phase1_expert_batch(
    store: MatrixStore,
    samples: LayerSamples,
    *,
    layer: int,
    experts: Sequence[int],
    partition: RequestPartition,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    fallback_block_contexts: torch.Tensor,
    fallback_block_scores: torch.Tensor,
    device: torch.device,
    shared_scale_scope: object,
    mode_ids: Sequence[int] = PHASE1_MODE_IDS,
    quantizer_module: object | None = None,
    min_fit_documents: int = 6,
    min_confirmation_documents: int = 4,
    bootstrap_replicates: int = 2_000,
    minimum_improvement: float = 0.0,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_QSRT_E4M3,
    layout: SearchLayout = "importance_ordered",
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
    hessian_policy: HessianPolicy = "captured_blend",
    permutation_policy: PermutationPolicy = "h2_reverse",
    folded_scale_power: float = 0.0,
    transform_seeds_by_expert: Mapping[
        int, Mapping[str, QSRTTransformSeeds]
    ] | None = None,
) -> list[QSRTCandidateEncoding]:
    """Encode several experts while batching their independent trellis work.

    This preserves the single-expert calibration, dense-H, candidate scoring,
    and selection contracts.  Only the execution schedule changes: all
    same-shape w1/w3 projections in the chunk are submitted together, followed
    by all w2 projections.  The larger tile batches fill SM120 much more
    effectively than one expert's R1/R2 replacement records.
    """

    expert_ids = tuple(int(expert) for expert in experts)
    if not expert_ids or len(set(expert_ids)) != len(expert_ids):
        raise ValueError("expert batch must be nonempty and unique")
    if device.type != "cuda":
        raise ValueError("QSRT candidate encoding requires CUDA")
    if not 1 <= layer <= 92 or any(not 0 <= expert < 896 for expert in expert_ids):
        raise ValueError("invalid K3 layer/expert assignment")
    modes = tuple(int(mode) for mode in mode_ids)
    if not modes or modes[0] != 0 or len(set(modes)) != len(modes):
        raise ValueError("mode_ids must be unique and begin with R0")
    if any(not 0 <= mode < len(RATE_TRANSFER_MODES) for mode in modes):
        raise ValueError("mode_ids contain an unsupported transfer count")
    if hessian_policy not in HESSIAN_POLICIES:
        raise ValueError(f"unsupported Hessian policy: {hessian_policy}")
    if permutation_policy not in PERMUTATION_POLICIES:
        raise ValueError(f"unsupported permutation policy: {permutation_policy}")

    prepared: list[dict[str, object]] = []
    upstream_sources: list[dict[str, torch.Tensor]] = []
    upstream_contexts: list[torch.Tensor] = []
    upstream_modes: list[tuple[object, ...]] = []
    upstream_hessians: list[dict[str, torch.Tensor]] = []
    upstream_conditioning: list[torch.Tensor | None] = []
    seed_maps = [
        (transform_seeds_by_expert or {}).get(expert, {}) for expert in expert_ids
    ]

    for expert in expert_ids:
        all_rows = select_expert_rows(samples, expert, partition.all)
        fit_mask = _split_rows(all_rows, partition.fit)
        confirmation_mask = _split_rows(all_rows, partition.confirmation)
        fit_documents = int(torch.unique(all_rows.request_steps[fit_mask]).numel())
        confirmation_documents = int(
            torch.unique(all_rows.request_steps[confirmation_mask]).numel()
        )
        inputs = all_rows.inputs.float().to(device=device)
        gates = all_rows.gates.float().to(device=device)

        source_w1 = _load_source_matrix(store, layer, expert, "w1", device)
        source_w3 = _load_source_matrix(store, layer, expert, "w3", device)
        if all_rows.rows:
            source_gate = F.linear(inputs, source_w1)
            source_up = F.linear(inputs, source_w3)
            source_middle = _source_middle(source_gate, source_up)
            del source_gate, source_up
        else:
            source_middle = torch.empty(
                (0, global_h2.shape[0]), dtype=torch.float32, device=device
            )

        have_local_support = fit_documents >= min_fit_documents
        fit_mask_device = fit_mask.to(device=device)
        if bool(fit_mask.any()):
            block_contexts, block_scores = permutation_policy_contexts(
                source_middle[fit_mask_device],
                gates[fit_mask_device],
                policy=permutation_policy,
            )
            context_basis = (
                f"official_source_post_situ_fit_documents:{permutation_policy}"
            )
        else:
            block_contexts = fallback_block_contexts.clone()
            block_scores = fallback_block_scores.clone()
            context_basis = "identity_logical_order_no_routed_fit_rows"

        if have_local_support:
            h13 = global_h13
            covariance: dict[str, object] = {
                "gate_square_sum": float(
                    gates[fit_mask_device].square().double().sum()
                ),
                "h13_local_alpha": 0.0,
            }
            covariance = {
                **covariance,
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
            }
        else:
            h13 = global_h13
            covariance = {
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "gate_square_sum": float(
                    gates[fit_mask.to(device=device)].square().sum()
                ),
                "h13_local_alpha": 0.0,
            }

        can_confirm = confirmation_documents >= min_confirmation_documents
        evaluated_modes = modes if have_local_support and can_confirm else (0,)
        mode_specs = tuple(RATE_TRANSFER_MODES[mode] for mode in evaluated_modes)
        contexts_device = block_contexts.to(device=device, dtype=torch.long)
        intermediate_conditioning = _folded_intermediate_conditioning(
            block_scores.to(device=device),
            contexts_device,
            power=folded_scale_power,
        )
        if intermediate_conditioning is not None:
            covariance["folded_scale_power"] = float(folded_scale_power)
            covariance["folded_scale_min"] = float(
                intermediate_conditioning.min()
            )
            covariance["folded_scale_max"] = float(
                intermediate_conditioning.max()
            )
            context_basis += "+folded_128_channel_conditioning"
        prepared.append(
            {
                "expert": expert,
                "all_rows": all_rows,
                "fit_mask": fit_mask,
                "confirmation_mask": confirmation_mask,
                "inputs": inputs,
                "gates": gates,
                "source_middle": source_middle,
                "block_contexts": block_contexts,
                "block_scores": block_scores,
                "context_basis": context_basis,
                "covariance": covariance,
                "evaluated_modes": evaluated_modes,
                "evaluated_formats": tuple(
                    (r13, r2)
                    for r13 in evaluated_modes
                    for r2 in evaluated_modes
                ),
            }
        )
        upstream_sources.append({"w1": source_w1, "w3": source_w3})
        upstream_contexts.append(contexts_device)
        upstream_modes.append(mode_specs)
        upstream_hessians.append({"w1": h13, "w3": h13})
        upstream_conditioning.append(intermediate_conditioning)

    guarded_reuse = layout == "qsrt_guarded_reuse"
    if guarded_reuse and any(mode > 2 for mode in modes):
        raise ValueError("qsrt_guarded_reuse supports only R0 through R2")
    upstream_layout: Layout = "importance_ordered" if guarded_reuse else layout
    upstream_candidates = quantize_qsrt_matrix_expert_batch(
        upstream_sources,
        upstream_contexts,
        modes_by_expert=upstream_modes,
        hessians_by_expert=upstream_hessians,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        codebook=codebook,
        layout=upstream_layout,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        transform_seeds_by_expert=seed_maps,
        intermediate_conditioning_by_expert=upstream_conditioning,
    )
    del upstream_sources, upstream_hessians

    down_sources: list[torch.Tensor] = []
    conditional_h2s: list[dict[int, torch.Tensor]] = []
    for item, upstream in zip(prepared, upstream_candidates):
        expert = int(item["expert"])
        source_w2 = _load_source_matrix(store, layer, expert, "w2", device)
        source_middle = item["source_middle"]
        inputs = item["inputs"]
        gates = item["gates"]
        fit_mask = item["fit_mask"]
        evaluated_modes = item["evaluated_modes"]
        assert isinstance(source_middle, torch.Tensor)
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(gates, torch.Tensor)
        assert isinstance(fit_mask, torch.Tensor)
        middle_by_r13 = _candidate_middle_by_r13(
            inputs, upstream, evaluated_modes
        )
        h2_by_r13, h2_evidence = _conditional_h2_by_r13(
            inputs,
            gates,
            fit_mask.to(device=device),
            middle_by_r13,
            global_h13=global_h13,
            global_h2=global_h2,
            device=device,
            hessian_policy=hessian_policy,
            have_local_support=(
                int(item["covariance"]["fit_documents"]) >= min_fit_documents
            ),
        )
        item["middle_by_r13"] = middle_by_r13
        item["covariance"].update(
            {
                "basis": (
                    "identity"
                    if hessian_policy == "identity"
                    else (
                        "decoded_candidate_post_situ_adaptive_shrinkage"
                        if int(item["covariance"]["fit_documents"])
                        >= min_fit_documents
                        else "identity_support_fallback"
                    )
                ),
                "h2_by_r13": h2_evidence,
            }
        )
        item["context_basis"] = (
            str(item["context_basis"]) + "+decoded_candidate_post_situ_h2"
        )
        all_rows = item["all_rows"]
        if all_rows.rows:
            reference_output = F.linear(source_middle, source_w2)
        else:
            reference_output = torch.empty(
                (0, global_h13.shape[0]), dtype=torch.float32, device=device
            )
        item["reference_output"] = reference_output
        down_sources.append(source_w2)
        conditional_h2s.append(h2_by_r13)

    down_candidates = _quantize_conditional_down_grid(
        down_sources,
        upstream_contexts,
        upstream_modes,
        conditional_h2s,
        layer=layer,
        device=device,
        shared_scale_scope=shared_scale_scope,
        quantizer_module=quantizer_module,
        codebook=codebook,
        layout=layout,
        ldlq_tf32=ldlq_tf32,
        tailbite_context=tailbite_context,
        transform_seeds_by_expert=seed_maps,
    )
    del down_sources, conditional_h2s

    results: list[QSRTCandidateEncoding] = []
    for item, upstream, down in zip(
        prepared, upstream_candidates, down_candidates
    ):
        expert = int(item["expert"])
        all_rows = item["all_rows"]
        fit_mask = item["fit_mask"]
        confirmation_mask = item["confirmation_mask"]
        inputs = item["inputs"]
        gates = item["gates"]
        source_middle = item["source_middle"]
        reference_output = item["reference_output"]
        evaluated_modes = item["evaluated_modes"]
        evaluated_formats = item["evaluated_formats"]
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(gates, torch.Tensor)
        assert isinstance(source_middle, torch.Tensor)
        assert isinstance(reference_output, torch.Tensor)

        fit_plan = _prepare_deferred_functional_rows(
            reference_output,
            gates,
            all_rows.request_steps,
            fit_mask,
            partition.fit,
            device=device,
        )
        confirmation_plan = _prepare_deferred_functional_rows(
            reference_output,
            gates,
            all_rows.request_steps,
            confirmation_mask,
            partition.confirmation,
            device=device,
        )
        deferred_fit_sse: list[torch.Tensor] = []
        deferred_confirmation_sse: list[torch.Tensor] = []
        format_order: list[tuple[int, int]] = []
        mode_coding: dict[tuple[int, int], dict[str, object]] = {}
        middle_by_r13 = item["middle_by_r13"]
        for rate_pair in evaluated_formats:
            r13, r2 = rate_pair
            candidate_output = F.linear(
                middle_by_r13[r13], down[r13][r2].reconstruction
            )
            deferred_fit_sse.append(
                _defer_functional_row_sse(fit_plan, candidate_output)
            )
            deferred_confirmation_sse.append(
                _defer_functional_row_sse(confirmation_plan, candidate_output)
            )
            format_order.append(rate_pair)
            mode_coding[rate_pair] = _candidate_mode_evidence(
                {
                    "w1": upstream["w1"][r13],
                    "w3": upstream["w3"][r13],
                    "w2": down[r13][r2],
                }
            )
            del candidate_output

        fit_sse = _finish_deferred_functional_sse(
            fit_plan, format_order, deferred_fit_sse
        )
        confirmation_sse = _finish_deferred_functional_sse(
            confirmation_plan, format_order, deferred_confirmation_sse
        )
        fit_reference = fit_plan.reference_energy
        fit_counts = fit_plan.counts
        confirmation_reference = confirmation_plan.reference_energy
        confirmation_counts = confirmation_plan.counts
        selection = select_phase1_rate_pair(
            fit_sse,
            confirmation_sse,
            fit_counts=fit_counts,
            confirmation_counts=confirmation_counts,
            modes=evaluated_formats,
            min_fit_documents=min_fit_documents,
            min_confirmation_documents=min_confirmation_documents,
            minimum_improvement=minimum_improvement,
            bootstrap_replicates=bootstrap_replicates,
            seed=deterministic_expert_seed(layer, expert),
        )
        selected_r13, selected_r2 = selection.selected
        selected_candidates = {
            "w1": upstream["w1"][selected_r13],
            "w3": upstream["w3"][selected_r13],
            "w2": down[selected_r13][selected_r2],
        }
        selected_encodings = {
            matrix: finalize_qsrt_matrix_candidate(
                value,
                layer=layer,
                logical_trellis_schema=logical_trellis_schema,
                codebook=codebook,
                tailbite_context=tailbite_context,
            )
            for matrix, value in selected_candidates.items()
        }
        physical_permutation = selected_encodings["w1"].plan.physical_permutation
        for matrix in ("w3", "w2"):
            if not torch.equal(
                selected_encodings[matrix].plan.physical_permutation,
                physical_permutation,
            ):
                raise ValueError(
                    "selected matrices do not share one physical permutation"
                )
        expert_encoding = QSRTExpertEncoding(
            format=ExpertFormatSpec.compressed(selected_r13, selected_r2),
            matrices=selected_encodings,
            physical_permutation=physical_permutation,
            coding={
                "format": ExpertFormatSpec.compressed(
                    selected_r13, selected_r2
                ).name,
                "r13": selected_r13,
                "r2": selected_r2,
                "codebook": codebook,
                "tailbite_context": int(tailbite_context),
                **_mode_evidence(selected_encodings),
            },
        )
        results.append(
            QSRTCandidateEncoding(
                expert=expert_encoding,
                selection=selection,
                block_contexts=item["block_contexts"]
                .detach()
                .cpu()
                .to(torch.uint8)
                .contiguous(),
                block_scores=item["block_scores"].detach().cpu().float().contiguous(),
                context_basis=str(item["context_basis"]),
                covariance=item["covariance"],
                evaluated_modes=evaluated_modes,
                evaluated_formats=evaluated_formats,
                fit_sse=fit_sse,
                confirmation_sse=confirmation_sse,
                fit_reference_energy=fit_reference,
                confirmation_reference_energy=confirmation_reference,
                fit_counts=fit_counts,
                confirmation_counts=confirmation_counts,
                mode_coding=mode_coding,
            )
        )
    return results


def candidate_tensor_name(
    layer: int, expert: int, matrix: str, part: str
) -> str:
    if matrix not in MATRICES or part not in ("trellis", "suh", "svh"):
        raise ValueError("invalid QSRT candidate tensor component")
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts."
        f"{expert}.{matrix}.qsrt_{part}"
    )


def selected_candidate_tensors(
    layer: int, expert: int, candidate: QSRTCandidateEncoding
) -> dict[str, torch.Tensor]:
    """Copy one selected physical payload to CPU for layer-shard assembly."""

    tensors: dict[str, torch.Tensor] = {}
    for matrix, encoding in candidate.expert.matrices.items():
        for part in ("trellis", "suh", "svh"):
            tensors[candidate_tensor_name(layer, expert, matrix, part)] = (
                encoding.tensors[part].detach().cpu().contiguous()
            )
    return tensors
