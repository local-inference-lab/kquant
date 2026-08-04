"""Offline-streamed all-expert candidates for the TP12 mixed-EXL codec."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from kquant.capture import LayerSamples
from kquant.exl3_reference import CODEBOOK_MCG
from kquant.mixed_candidates import (
    RatePairSelection,
    RequestPartition,
    RoutedRows,
    activation_block_contexts,
    build_expert_hessians,
    deterministic_expert_seed,
    functional_sse_by_request,
    select_expert_rows,
    select_phase1_rate_pair,
)
from kquant.mixed_exl3 import (
    SCHEMA,
    ExpertFormatSpec,
    PHASE1_MODE_IDS,
    RATE_TRANSFER_MODES,
)
from kquant.pack.mixed_exl3 import (
    Layout,
    MATRICES,
    MixedExpertEncoding,
    MixedMatrixEncoding,
    SearchLayout,
    finalize_mixed_matrix_candidate,
    quantize_mixed_matrix_candidates_batched,
    quantize_mixed_matrix_expert_batch,
)
from kquant.tp_simulator import situ


CANDIDATE_POOL_KIND = "kquant_tp12_mixed_exl3_candidate_pool"
CANDIDATE_POOL_SCHEMA_VERSION = 5
OFFICIAL_SOURCE_DAMAGE_METRIC = "official_source_excess_sse"


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
class MixedCandidateEncoding:
    """Selected payload plus the complete phase-1 selection evidence."""

    expert: MixedExpertEncoding
    selection: RatePairSelection
    block_contexts: torch.Tensor
    block_scores: torch.Tensor
    context_basis: str
    covariance: dict[str, float | str | int]
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


def _mode_evidence(encodings: dict[str, MixedMatrixEncoding]) -> dict[str, object]:
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
    codebook: str = CODEBOOK_MCG,
    layout: SearchLayout = "importance_ordered",
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> MixedCandidateEncoding:
    """Encode and conservatively select one official-source expert.

    The official model is never instantiated.  ``w1`` and ``w3`` are streamed
    together so their same-shape R candidates can share one batched traversal;
    they are released before ``w2`` is loaded.
    """

    if device.type != "cuda":
        raise ValueError("phase-1 mixed candidate encoding requires CUDA")
    if not 1 <= layer <= 92 or not 0 <= expert < 896:
        raise ValueError("invalid K3 layer/expert assignment")
    modes = tuple(int(mode) for mode in mode_ids)
    if not modes or modes[0] != 0 or len(set(modes)) != len(modes):
        raise ValueError("mode_ids must be unique and begin with R0")
    if any(not 0 <= mode < len(RATE_TRANSFER_MODES) for mode in modes):
        raise ValueError("mode_ids contain an unsupported transfer count")

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
    if have_local_support:
        block_contexts, block_scores = activation_block_contexts(
            source_middle[fit_mask.to(device=device)],
            gates[fit_mask.to(device=device)],
        )
        h13, h2, covariance = build_expert_hessians(
            inputs[fit_mask.to(device=device)],
            gates[fit_mask.to(device=device)],
            source_middle[fit_mask.to(device=device)],
            global_h13=global_h13,
            global_h2=global_h2,
            device=device,
        )
        context_basis = "official_source_post_situ_fit_documents"
        covariance = {
            **covariance,
            "basis": "expert_local_fixed_shrinkage",
            "fit_rows": int(fit_mask.sum()),
            "fit_documents": fit_documents,
        }
    else:
        block_contexts = fallback_block_contexts.clone()
        block_scores = fallback_block_scores.clone()
        h13, h2 = global_h13, global_h2
        context_basis = "layer_global_interim_post_situ_fit_documents"
        covariance = {
            "basis": "layer_global_support_fallback",
            "fit_rows": int(fit_mask.sum()),
            "fit_documents": fit_documents,
            "gate_square_sum": float(gates[fit_mask.to(device=device)].square().sum()),
            "h13_local_alpha": 0.0,
            "h2_local_alpha": 0.0,
        }

    can_confirm = confirmation_documents >= min_confirmation_documents
    evaluated_modes = modes if have_local_support and can_confirm else (0,)
    evaluated_formats = tuple(
        (r13, r2) for r13 in evaluated_modes for r2 in evaluated_modes
    )
    block_contexts_device = block_contexts.to(device=device, dtype=torch.long)

    mode_specs = tuple(RATE_TRANSFER_MODES[mode] for mode in evaluated_modes)
    guarded_reuse = layout == "r05_guarded_reuse"
    if guarded_reuse and any(mode > 5 for mode in evaluated_modes):
        raise ValueError("r05_guarded_reuse supports only R0 through R5")
    upstream_layout: Layout = "importance_ordered" if guarded_reuse else layout
    upstream_candidates = quantize_mixed_matrix_candidates_batched(
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
    )
    del source_w1, source_w3

    source_w2 = _load_source_matrix(store, layer, expert, "w2", device)
    if all_rows.rows:
        reference_output = F.linear(source_middle, source_w2)
    else:
        reference_output = torch.empty(
            (0, global_h13.shape[0]), dtype=torch.float32, device=device
        )
    if guarded_reuse and len(mode_specs) > 1:
        # Preserve the exact established R0 traversal.  Only the shifted w2
        # candidates use the reordered traversal needed for prefix/trie reuse;
        # all reconstructions are restored to canonical coordinates before the
        # common functional selector compares them.
        down_candidates = quantize_mixed_matrix_candidates_batched(
            {"w2": source_w2},
            block_contexts_device,
            modes=(mode_specs[0],),
            hessians={"w2": h2},
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout="importance_ordered",
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
        )
        shifted = quantize_mixed_matrix_candidates_batched(
            {"w2": source_w2},
            block_contexts_device,
            modes=mode_specs[1:],
            hessians={"w2": h2},
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout="r05_pair_prefix_reuse",
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
        )
        down_candidates["w2"].update(shifted["w2"])
    else:
        down_candidates = quantize_mixed_matrix_candidates_batched(
            {"w2": source_w2},
            block_contexts_device,
            modes=mode_specs,
            hessians={"w2": h2},
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout="importance_ordered" if guarded_reuse else layout,
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
        )
    del source_w2

    fit_sse: dict[tuple[int, int], torch.Tensor] = {}
    confirmation_sse: dict[tuple[int, int], torch.Tensor] = {}
    fit_reference = fit_counts = confirmation_reference = confirmation_counts = None
    mode_coding: dict[tuple[int, int], dict[str, object]] = {}
    middle_by_r13 = {
        r13: _source_middle(
            F.linear(inputs, upstream_candidates["w1"][r13].reconstruction),
            F.linear(inputs, upstream_candidates["w3"][r13].reconstruction),
        )
        for r13 in evaluated_modes
    }
    for rate_pair in evaluated_formats:
        r13, r2 = rate_pair
        candidate_output = F.linear(
            middle_by_r13[r13], down_candidates["w2"][r2].reconstruction
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
                "w2": down_candidates["w2"][r2],
            }
        )
        del candidate_output

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
        "w2": down_candidates["w2"][selected_r2],
    }
    selected_encodings = {
        matrix: finalize_mixed_matrix_candidate(
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
    expert_encoding = MixedExpertEncoding(
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
    return MixedCandidateEncoding(
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
    codebook: str = CODEBOOK_MCG,
    layout: SearchLayout = "importance_ordered",
    ldlq_tf32: bool = False,
    tailbite_context: int = 128,
) -> list[MixedCandidateEncoding]:
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
        raise ValueError("phase-1 mixed candidate encoding requires CUDA")
    if not 1 <= layer <= 92 or any(not 0 <= expert < 896 for expert in expert_ids):
        raise ValueError("invalid K3 layer/expert assignment")
    modes = tuple(int(mode) for mode in mode_ids)
    if not modes or modes[0] != 0 or len(set(modes)) != len(modes):
        raise ValueError("mode_ids must be unique and begin with R0")
    if any(not 0 <= mode < len(RATE_TRANSFER_MODES) for mode in modes):
        raise ValueError("mode_ids contain an unsupported transfer count")

    prepared: list[dict[str, object]] = []
    upstream_sources: list[dict[str, torch.Tensor]] = []
    upstream_contexts: list[torch.Tensor] = []
    upstream_modes: list[tuple[object, ...]] = []
    upstream_hessians: list[dict[str, torch.Tensor]] = []

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
        if have_local_support:
            fit_mask_device = fit_mask.to(device=device)
            block_contexts, block_scores = activation_block_contexts(
                source_middle[fit_mask_device],
                gates[fit_mask_device],
            )
            h13, h2, covariance = build_expert_hessians(
                inputs[fit_mask_device],
                gates[fit_mask_device],
                source_middle[fit_mask_device],
                global_h13=global_h13,
                global_h2=global_h2,
                device=device,
            )
            context_basis = "official_source_post_situ_fit_documents"
            covariance = {
                **covariance,
                "basis": "expert_local_fixed_shrinkage",
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
            }
        else:
            block_contexts = fallback_block_contexts.clone()
            block_scores = fallback_block_scores.clone()
            h13, h2 = global_h13, global_h2
            context_basis = "layer_global_interim_post_situ_fit_documents"
            covariance = {
                "basis": "layer_global_support_fallback",
                "fit_rows": int(fit_mask.sum()),
                "fit_documents": fit_documents,
                "gate_square_sum": float(
                    gates[fit_mask.to(device=device)].square().sum()
                ),
                "h13_local_alpha": 0.0,
                "h2_local_alpha": 0.0,
            }

        can_confirm = confirmation_documents >= min_confirmation_documents
        evaluated_modes = modes if have_local_support and can_confirm else (0,)
        mode_specs = tuple(RATE_TRANSFER_MODES[mode] for mode in evaluated_modes)
        contexts_device = block_contexts.to(device=device, dtype=torch.long)
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
                "h2": h2,
            }
        )
        upstream_sources.append({"w1": source_w1, "w3": source_w3})
        upstream_contexts.append(contexts_device)
        upstream_modes.append(mode_specs)
        upstream_hessians.append({"w1": h13, "w3": h13})

    guarded_reuse = layout == "r05_guarded_reuse"
    if guarded_reuse and any(mode > 5 for mode in modes):
        raise ValueError("r05_guarded_reuse supports only R0 through R5")
    upstream_layout: Layout = "importance_ordered" if guarded_reuse else layout
    upstream_candidates = quantize_mixed_matrix_expert_batch(
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
    )
    del upstream_sources, upstream_hessians

    down_sources: list[dict[str, torch.Tensor]] = []
    down_hessians: list[dict[str, torch.Tensor]] = []
    for item in prepared:
        expert = int(item["expert"])
        source_w2 = _load_source_matrix(store, layer, expert, "w2", device)
        source_middle = item["source_middle"]
        assert isinstance(source_middle, torch.Tensor)
        all_rows = item["all_rows"]
        if all_rows.rows:
            reference_output = F.linear(source_middle, source_w2)
        else:
            reference_output = torch.empty(
                (0, global_h13.shape[0]), dtype=torch.float32, device=device
            )
        item["reference_output"] = reference_output
        down_sources.append({"w2": source_w2})
        down_hessians.append({"w2": item["h2"]})

    if guarded_reuse:
        r0_modes = [(mode_specs[0],) for mode_specs in upstream_modes]
        down_candidates = quantize_mixed_matrix_expert_batch(
            down_sources,
            upstream_contexts,
            modes_by_expert=r0_modes,
            hessians_by_expert=down_hessians,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout="importance_ordered",
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
        )
        shifted_indices = [
            index for index, mode_specs in enumerate(upstream_modes)
            if len(mode_specs) > 1
        ]
        if shifted_indices:
            shifted_candidates = quantize_mixed_matrix_expert_batch(
                [down_sources[index] for index in shifted_indices],
                [upstream_contexts[index] for index in shifted_indices],
                modes_by_expert=[
                    upstream_modes[index][1:] for index in shifted_indices
                ],
                hessians_by_expert=[
                    down_hessians[index] for index in shifted_indices
                ],
                layer=layer,
                device=device,
                shared_scale_scope=shared_scale_scope,
                quantizer_module=quantizer_module,
                codebook=codebook,
                layout="r05_pair_prefix_reuse",
                ldlq_tf32=ldlq_tf32,
                tailbite_context=tailbite_context,
            )
            for index, shifted in zip(shifted_indices, shifted_candidates):
                down_candidates[index]["w2"].update(shifted["w2"])
    else:
        down_candidates = quantize_mixed_matrix_expert_batch(
            down_sources,
            upstream_contexts,
            modes_by_expert=upstream_modes,
            hessians_by_expert=down_hessians,
            layer=layer,
            device=device,
            shared_scale_scope=shared_scale_scope,
            quantizer_module=quantizer_module,
            codebook=codebook,
            layout=layout,
            ldlq_tf32=ldlq_tf32,
            tailbite_context=tailbite_context,
        )
    del down_sources, down_hessians

    results: list[MixedCandidateEncoding] = []
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
        middle_by_r13 = {
            r13: _source_middle(
                F.linear(inputs, upstream["w1"][r13].reconstruction),
                F.linear(inputs, upstream["w3"][r13].reconstruction),
            )
            for r13 in evaluated_modes
        }
        for rate_pair in evaluated_formats:
            r13, r2 = rate_pair
            candidate_output = F.linear(
                middle_by_r13[r13], down["w2"][r2].reconstruction
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
                    "w2": down["w2"][r2],
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
            "w2": down["w2"][selected_r2],
        }
        selected_encodings = {
            matrix: finalize_mixed_matrix_candidate(
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
        expert_encoding = MixedExpertEncoding(
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
            MixedCandidateEncoding(
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
        raise ValueError("invalid mixed candidate tensor component")
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts."
        f"{expert}.{matrix}.mixed_exl3_{part}"
    )


def selected_candidate_tensors(
    layer: int, expert: int, candidate: MixedCandidateEncoding
) -> dict[str, torch.Tensor]:
    """Copy one selected physical payload to CPU for layer-shard assembly."""

    tensors: dict[str, torch.Tensor] = {}
    for matrix, encoding in candidate.expert.matrices.items():
        for part in ("trellis", "suh", "svh"):
            tensors[candidate_tensor_name(layer, expert, matrix, part)] = (
                encoding.tensors[part].detach().cpu().contiguous()
            )
    return tensors
