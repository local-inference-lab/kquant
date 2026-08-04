"""Calibration and conservative mode selection for TP12 QSRT candidates.

This module contains the data-facing part of the phase-1 encoder contract.  It
does not load model weights and it never reads the external validation corpus.
Callers provide one captured layer and stream official expert matrices through
the encoder separately.

The selection split is deliberately asymmetric:

* fit documents determine the neuron order and expert-local covariance blends;
* confirmation documents choose among the already-encoded fixed-rate formats
  and must also clear a paired bootstrap lower-bound gate against R0; and
* insufficient support deterministically falls back to R0 and layer-global
  covariance.

This separation matters because the fit rows also define the encoder Hessian.
Using their reconstruction error to choose a mode can reward a mode for
overfitting that fitted geometry.  The confirmation rows are never used by
the encoder and are therefore the correct mode-selection population.  A
separate external capture remains untouched for representative validation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from kquant.candidate_hessian import shrink_covariance, weighted_covariance
from kquant.capture import LayerSamples
from kquant.qsrt import (
    CONTEXT_GROUP_CHANNELS,
    INTERMEDIATE_CHANNELS,
    PHASE1_H13_EXPERT_LOCAL_ALPHA,
    PHASE1_H2_EXPERT_LOCAL_ALPHA,
    PHASE1_MODE_IDS,
    RECORDS_PER_EXPERT,
)


def rank_contexts(scores: torch.Tensor, contexts: int) -> torch.Tensor:
    """Assign finite scalar scores to equal-population ordered contexts."""

    if scores.ndim != 1 or not torch.is_floating_point(scores):
        raise ValueError("context scores must be a floating-point vector")
    if contexts <= 0 or contexts > scores.numel():
        raise ValueError("invalid number of contexts")
    if not torch.all(torch.isfinite(scores)):
        raise ValueError("context scores must be finite")
    order = torch.argsort(scores, stable=True)
    ordered_contexts = torch.div(
        torch.arange(scores.numel(), device=scores.device) * contexts,
        scores.numel(),
        rounding_mode="floor",
    )
    result = torch.empty_like(order)
    result[order] = ordered_contexts
    return result


PHASE1_CONFIRMATION_MODULUS = 4
PHASE1_CONFIRMATION_INDEX = 0
PHASE1_MIN_FIT_DOCUMENTS = 6
PHASE1_MIN_CONFIRMATION_DOCUMENTS = 4
PHASE1_BOOTSTRAP_REPLICATES = 2_000
PHASE1_SELECTION_SEED = 20_260_801


@dataclass(frozen=True)
class RequestPartition:
    """Document-disjoint request epochs used by the offline encoder."""

    fit: dict[int, str]
    confirmation: dict[int, str]

    @property
    def all(self) -> dict[int, str]:
        return dict(sorted({**self.fit, **self.confirmation}.items()))


@dataclass(frozen=True)
class RoutedRows:
    """Captured input rows on which one expert was selected."""

    inputs: torch.Tensor
    gates: torch.Tensor
    request_steps: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def documents(self) -> int:
        return int(torch.unique(self.request_steps).numel())


@dataclass(frozen=True)
class ModeSelection:
    """One fit-proposed and independently confirmation-gated shared mode."""

    proposed_r: int
    selected_r: int
    accepted: bool
    reason: str
    fit_documents: int
    confirmation_documents: int
    confirmation_relative_improvement: float | None
    confirmation_ci95: tuple[float | None, float | None]
    bootstrap_replicates_valid: int


@dataclass(frozen=True)
class RatePairSelection:
    """Confirmation-selected and bootstrap-gated ``(r13, r2)`` decision."""

    proposed_r13: int
    proposed_r2: int
    selected_r13: int
    selected_r2: int
    accepted: bool
    reason: str
    fit_documents: int
    confirmation_documents: int
    confirmation_relative_improvement: float | None
    confirmation_ci95: tuple[float | None, float | None]
    bootstrap_replicates_valid: int

    @property
    def proposed(self) -> tuple[int, int]:
        return self.proposed_r13, self.proposed_r2

    @property
    def selected(self) -> tuple[int, int]:
        return self.selected_r13, self.selected_r2


def request_documents(
    report: Mapping[str, object], *, deduplicate: bool = False
) -> dict[int, str]:
    """Resolve corpus request epochs from a finalized report.

    ``deduplicate=True`` preserves the original request-step keys while
    retaining only the first epoch for each content hash.  It exists so an
    already completed capture can be audited without statistically weighting
    repeated source records twice.  New corpus plans reject duplicates before
    capture.
    """

    if report.get("kind") != "kquant_interim_calibration_corpus_run":
        raise ValueError("corpus report has the wrong kind")
    if not bool(report.get("finalized")):
        raise ValueError("corpus report is not finalized")
    raw_documents = report.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("corpus report has no documents")
    expected = len(raw_documents)
    for key in ("planned_requests", "completed_requests"):
        if int(report.get(key, expected)) != expected:
            raise ValueError(f"corpus report {key} disagrees with documents")
    result: dict[int, str] = {}
    seen: set[str] = set()
    for request_step, value in enumerate(raw_documents, start=1):
        if not isinstance(value, Mapping) or not value.get("document_hash"):
            raise ValueError("corpus report has a document without a hash")
        document_hash = str(value["document_hash"])
        if document_hash in seen:
            if deduplicate:
                continue
            raise ValueError("corpus report document hashes must be unique")
        seen.add(document_hash)
        result[request_step] = document_hash
    return result


def partition_requests(
    requests: Mapping[int, str],
    *,
    modulus: int = PHASE1_CONFIRMATION_MODULUS,
    confirmation_index: int = PHASE1_CONFIRMATION_INDEX,
) -> RequestPartition:
    """Split request epochs by a stable whole-document content hash."""

    if modulus < 2 or not 0 <= confirmation_index < modulus:
        raise ValueError("invalid fit/confirmation partition")
    if not requests:
        raise ValueError("request mapping must be nonempty")
    fit: dict[int, str] = {}
    confirmation: dict[int, str] = {}
    for raw_step, raw_document in sorted(requests.items()):
        step = int(raw_step)
        document = str(raw_document)
        if step < 0 or not document:
            raise ValueError("request epochs and document hashes must be valid")
        digest = hashlib.blake2b(document.encode(), digest_size=8).digest()
        target = (
            confirmation
            if int.from_bytes(digest, "little") % modulus == confirmation_index
            else fit
        )
        target[step] = document
    if not fit or not confirmation:
        raise ValueError("document partition produced an empty side")
    return RequestPartition(fit=fit, confirmation=confirmation)


def select_expert_rows(
    samples: LayerSamples,
    expert: int,
    requests: Mapping[int, str],
) -> RoutedRows:
    """Select captured token rows routed to ``expert`` in ``requests``."""

    if not 0 <= int(expert) < 896:
        raise ValueError("expert must be in 0..895")
    if not requests:
        raise ValueError("request mapping must be nonempty")
    matches = samples.input_experts == int(expert)
    selected = matches.any(dim=1)
    steps = torch.bitwise_right_shift(samples.input_observations, 32)
    allowed = torch.tensor(
        sorted(map(int, requests)), dtype=steps.dtype, device=steps.device
    )
    selected &= torch.isin(steps, allowed)
    gates = (samples.input_gates[selected] * matches[selected]).sum(dim=1)
    return RoutedRows(
        inputs=samples.input_values[selected],
        gates=gates,
        request_steps=steps[selected],
    )


def activation_block_contexts(
    middle: torch.Tensor,
    gates: torch.Tensor,
    *,
    contexts: int = RECORDS_PER_EXPERT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank four-neuron groups by gate-square-weighted post-SiTU energy."""

    if (
        middle.ndim != 2
        or middle.shape[1] != INTERMEDIATE_CHANNELS
        or gates.ndim != 1
        or middle.shape[0] != gates.shape[0]
    ):
        raise ValueError("invalid post-SiTU ranking sample shapes")
    if not middle.shape[0]:
        raise ValueError("post-SiTU ranking samples must be nonempty")
    if contexts <= 0 or INTERMEDIATE_CHANNELS % (
        contexts * CONTEXT_GROUP_CHANNELS
    ):
        raise ValueError("contexts must evenly partition four-neuron groups")
    if not bool(torch.all(torch.isfinite(middle))) or not bool(
        torch.all(torch.isfinite(gates))
    ):
        raise ValueError("post-SiTU ranking samples must be finite")
    weights = gates.float().square()
    denominator = weights.double().sum()
    if denominator <= 0:
        raise ValueError("post-SiTU ranking weights must have positive mass")
    energy = (
        middle.float().square().mul(weights[:, None]).sum(dim=0).double()
        / denominator
    )
    scores = energy.reshape(-1, CONTEXT_GROUP_CHANNELS).mean(dim=1).float()
    return rank_contexts(scores, contexts), scores


def layer_fallback_contexts(
    samples: LayerSamples,
    requests: Mapping[int, str],
    *,
    contexts: int = RECORDS_PER_EXPERT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a layer-global post-SiTU order for unsupported experts."""

    steps = torch.bitwise_right_shift(samples.mid_observations, 32)
    allowed = torch.tensor(
        sorted(map(int, requests)), dtype=steps.dtype, device=steps.device
    )
    selected = torch.isin(steps, allowed)
    if not torch.count_nonzero(selected):
        raise ValueError("layer has no fit middle samples")
    # mid_weights are already the applied gate squared for each routed row.
    gates = samples.mid_weights[selected].float().clamp_min(0).sqrt()
    return activation_block_contexts(
        samples.mid_values[selected], gates, contexts=contexts
    )


def build_expert_hessians(
    inputs: torch.Tensor,
    gates: torch.Tensor,
    source_middle: torch.Tensor,
    *,
    global_h13: torch.Tensor,
    global_h2: torch.Tensor,
    device: torch.device,
    h13_alpha: float = PHASE1_H13_EXPERT_LOCAL_ALPHA,
    h2_alpha: float = PHASE1_H2_EXPERT_LOCAL_ALPHA,
    chunk_rows: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Construct the frozen source-local covariance blends for one expert."""

    if inputs.ndim != 2 or inputs.shape[0] != gates.shape[0]:
        raise ValueError("invalid expert input covariance samples")
    if source_middle.ndim != 2 or source_middle.shape[0] != inputs.shape[0]:
        raise ValueError("source middle rows do not match expert inputs")
    weights = gates.float().square()
    local_h13, h13_weight = weighted_covariance(
        inputs,
        weights,
        device=device,
        chunk_rows=chunk_rows,
    )
    local_h2, h2_weight = weighted_covariance(
        source_middle,
        weights,
        device=device,
        chunk_rows=chunk_rows,
    )
    if not math.isclose(h13_weight, h2_weight, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("H13 and H2 expert covariance weights differ")
    return (
        shrink_covariance(global_h13, local_h13, h13_alpha),
        shrink_covariance(global_h2, local_h2, h2_alpha),
        {
            "gate_square_sum": h13_weight,
            "h13_local_alpha": float(h13_alpha),
            "h2_local_alpha": float(h2_alpha),
        },
    )


def functional_sse_by_request(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    gates: torch.Tensor,
    request_steps: torch.Tensor,
    requests: Mapping[int, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return routed SSE, reference energy, and row counts by request epoch."""

    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("reference and candidate outputs must be equal 2-D shapes")
    rows = reference.shape[0]
    if gates.ndim != 1 or request_steps.ndim != 1 or gates.numel() != rows or request_steps.numel() != rows:
        raise ValueError("functional metric metadata has the wrong shape")
    ordered_steps = sorted(map(int, requests))
    positions = {step: index for index, step in enumerate(ordered_steps)}
    try:
        indices = torch.tensor(
            [positions[int(step)] for step in request_steps.cpu().tolist()],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"metric row has unmapped request epoch {exc.args[0]}") from exc
    route_weights = gates.double().square().cpu()
    delta = candidate.double() - reference.double()
    row_sse = delta.square().sum(dim=1).cpu() * route_weights
    row_reference = reference.double().square().sum(dim=1).cpu() * route_weights
    sse = torch.zeros(len(ordered_steps), dtype=torch.float64)
    energy = torch.zeros_like(sse)
    counts = torch.zeros(len(ordered_steps), dtype=torch.int64)
    sse.scatter_add_(0, indices, row_sse)
    energy.scatter_add_(0, indices, row_reference)
    counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int64))
    return sse, energy, counts


def _bootstrap_improvement(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    support: torch.Tensor,
    *,
    replicates: int,
    seed: int,
) -> tuple[float | None, tuple[float | None, float | None], int]:
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("bootstrap SSE vectors must be equal one-dimensional shapes")
    if support.dtype != torch.bool or support.shape != baseline.shape:
        raise ValueError("bootstrap support mask has the wrong shape")
    base = baseline[support].double().cpu().numpy()
    trial = candidate[support].double().cpu().numpy()
    if base.size == 0 or not np.isfinite(base).all() or not np.isfinite(trial).all():
        return None, (None, None), 0
    baseline_total = float(base.sum())
    if baseline_total <= 0:
        return None, (None, None), 0
    observed = 1.0 - float(trial.sum()) / baseline_total
    rng = np.random.default_rng(int(seed))
    values: list[np.ndarray] = []
    remaining = int(replicates)
    while remaining:
        count = min(remaining, 512)
        indices = rng.integers(0, base.size, size=(count, base.size))
        sampled_base = base[indices].sum(axis=1)
        sampled_trial = trial[indices].sum(axis=1)
        valid = sampled_base > 0
        if np.any(valid):
            values.append(1.0 - sampled_trial[valid] / sampled_base[valid])
        remaining -= count
    if not values:
        return observed, (None, None), 0
    samples = np.concatenate(values)
    low, high = np.quantile(samples, (0.025, 0.975)).tolist()
    return observed, (float(low), float(high)), int(samples.size)


def select_phase1_mode(
    fit_sse: Mapping[int, torch.Tensor],
    confirmation_sse: Mapping[int, torch.Tensor],
    *,
    fit_counts: torch.Tensor,
    confirmation_counts: torch.Tensor,
    mode_ids: Sequence[int] = PHASE1_MODE_IDS,
    min_fit_documents: int = PHASE1_MIN_FIT_DOCUMENTS,
    min_confirmation_documents: int = PHASE1_MIN_CONFIRMATION_DOCUMENTS,
    minimum_improvement: float = 0.0,
    bootstrap_replicates: int = PHASE1_BOOTSTRAP_REPLICATES,
    seed: int = PHASE1_SELECTION_SEED,
) -> ModeSelection:
    """Propose on fit documents and accept only with confirmation evidence."""

    modes = tuple(int(mode) for mode in mode_ids)
    if not modes or modes[0] != 0 or len(set(modes)) != len(modes):
        raise ValueError("mode_ids must be unique and begin with R0")
    if set(fit_sse) != set(modes) or set(confirmation_sse) != set(modes):
        raise ValueError("SSE tables must contain every requested mode")
    fit_support = fit_counts > 0
    confirmation_support = confirmation_counts > 0
    fit_documents = int(fit_support.sum())
    confirmation_documents = int(confirmation_support.sum())
    if fit_documents < min_fit_documents:
        return ModeSelection(
            proposed_r=0,
            selected_r=0,
            accepted=False,
            reason="insufficient fit-document support",
            fit_documents=fit_documents,
            confirmation_documents=confirmation_documents,
            confirmation_relative_improvement=None,
            confirmation_ci95=(None, None),
            bootstrap_replicates_valid=0,
        )
    proposed = min(
        modes,
        key=lambda mode: (
            float(fit_sse[mode][fit_support].double().sum()),
            mode,
        ),
    )
    if proposed == 0:
        return ModeSelection(
            proposed_r=0,
            selected_r=0,
            accepted=False,
            reason="R0 is the fit-document argmin",
            fit_documents=fit_documents,
            confirmation_documents=confirmation_documents,
            confirmation_relative_improvement=None,
            confirmation_ci95=(None, None),
            bootstrap_replicates_valid=0,
        )
    improvement, interval, valid = _bootstrap_improvement(
        confirmation_sse[0],
        confirmation_sse[proposed],
        confirmation_support,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    accepted = (
        confirmation_documents >= min_confirmation_documents
        and interval[0] is not None
        and interval[0] > minimum_improvement
    )
    if confirmation_documents < min_confirmation_documents:
        reason = "insufficient confirmation-document support"
    elif interval[0] is None:
        reason = "confirmation bootstrap has no valid baseline samples"
    elif interval[0] <= minimum_improvement:
        reason = "confirmation lower confidence bound did not clear the margin"
    else:
        reason = "confirmation lower confidence bound cleared the margin"
    return ModeSelection(
        proposed_r=proposed,
        selected_r=proposed if accepted else 0,
        accepted=accepted,
        reason=reason,
        fit_documents=fit_documents,
        confirmation_documents=confirmation_documents,
        confirmation_relative_improvement=improvement,
        confirmation_ci95=interval,
        bootstrap_replicates_valid=valid,
    )


def select_phase1_rate_pair(
    fit_sse: Mapping[tuple[int, int], torch.Tensor],
    confirmation_sse: Mapping[tuple[int, int], torch.Tensor],
    *,
    fit_counts: torch.Tensor,
    confirmation_counts: torch.Tensor,
    modes: Sequence[tuple[int, int]],
    min_fit_documents: int = PHASE1_MIN_FIT_DOCUMENTS,
    min_confirmation_documents: int = PHASE1_MIN_CONFIRMATION_DOCUMENTS,
    minimum_improvement: float = 0.0,
    bootstrap_replicates: int = PHASE1_BOOTSTRAP_REPLICATES,
    seed: int = PHASE1_SELECTION_SEED,
) -> RatePairSelection:
    """Select a coupled expert format while keeping both rate axes independent.

    ``fit_sse`` is retained in the persisted evidence and checked for support,
    but it is not used to rank formats: those same documents constructed the
    encoder Hessians and channel order.  The disjoint confirmation fold ranks
    the complete Cartesian ``(r13, r2)`` grid and then bootstrap-gates its
    winner against R0/R0.
    """

    candidates = tuple((int(r13), int(r2)) for r13, r2 in modes)
    if not candidates or candidates[0] != (0, 0) or len(set(candidates)) != len(candidates):
        raise ValueError("rate-pair candidates must be unique and begin with R0/R0")
    if set(fit_sse) != set(candidates) or set(confirmation_sse) != set(candidates):
        raise ValueError("rate-pair SSE tables must contain every requested candidate")
    fit_support = fit_counts > 0
    confirmation_support = confirmation_counts > 0
    fit_documents = int(fit_support.sum())
    confirmation_documents = int(confirmation_support.sum())

    def decision(
        proposed: tuple[int, int],
        selected: tuple[int, int],
        *,
        accepted: bool,
        reason: str,
        improvement: float | None = None,
        interval: tuple[float | None, float | None] = (None, None),
        valid: int = 0,
    ) -> RatePairSelection:
        return RatePairSelection(
            proposed_r13=proposed[0],
            proposed_r2=proposed[1],
            selected_r13=selected[0],
            selected_r2=selected[1],
            accepted=accepted,
            reason=reason,
            fit_documents=fit_documents,
            confirmation_documents=confirmation_documents,
            confirmation_relative_improvement=improvement,
            confirmation_ci95=interval,
            bootstrap_replicates_valid=valid,
        )

    if fit_documents < min_fit_documents:
        return decision(
            (0, 0),
            (0, 0),
            accepted=False,
            reason="insufficient fit-document support",
        )
    if confirmation_documents < min_confirmation_documents:
        return decision(
            (0, 0),
            (0, 0),
            accepted=False,
            reason="insufficient confirmation-document support",
        )
    proposed = min(
        candidates,
        key=lambda mode: (
            float(
                confirmation_sse[mode][confirmation_support]
                .double()
                .sum()
            ),
            mode[0] + mode[1],
            mode[0],
            mode[1],
        ),
    )
    if proposed == (0, 0):
        return decision(
            proposed,
            proposed,
            accepted=False,
            reason="R0/R0 is the confirmation-document argmin",
        )
    improvement, interval, valid = _bootstrap_improvement(
        confirmation_sse[(0, 0)],
        confirmation_sse[proposed],
        confirmation_support,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    accepted = interval[0] is not None and interval[0] > minimum_improvement
    if interval[0] is None:
        reason = "confirmation bootstrap has no valid baseline samples"
    elif interval[0] <= minimum_improvement:
        reason = "confirmation lower confidence bound did not clear the margin"
    else:
        reason = "confirmation lower confidence bound cleared the margin"
    return decision(
        proposed,
        proposed if accepted else (0, 0),
        accepted=accepted,
        reason=reason,
        improvement=improvement,
        interval=interval,
        valid=valid,
    )


def deterministic_expert_seed(layer: int, expert: int) -> int:
    """Derive a stable independent selector seed for one layer/expert cell."""

    if not 1 <= int(layer) <= 92 or not 0 <= int(expert) < 896:
        raise ValueError("invalid K3 layer/expert assignment")
    return PHASE1_SELECTION_SEED + int(layer) * 1_000_003 + int(expert) * 97
