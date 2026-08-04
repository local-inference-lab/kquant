"""Descriptive drift analysis for an interim Kimi-K3 calibration teacher.

This module deliberately does not turn a short correctness trace into a
quality gate.  It measures route and activation drift with exact trace
integrity checks, while recording that neuron-ranking and mode-selection
agreement require a document-diverse capture with routed post-SiTU values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from kquant.constants import NUM_EXPERTS
from kquant.trace_compare import PARTIAL_STAGES, TraceFormatError, TraceKey, TraceRun


TEACHER_PROXY_SCHEMA_VERSION = 1
ROUTE_IDS_STAGE = "canonical_topk_ids"
ROUTE_WEIGHTS_STAGE = "canonical_topk_weights"
POST_SITU_STAGE = "routed_post_situ"
DEFAULT_FLOAT_STAGES = (
    "moe_input",
    "router_logits",
    "routed_latent_input",
    "routed_latent_normalized",
    "routed_output_partial",
    "moe_output",
    "layer_boundary",
)


@dataclass
class _FloatMoments:
    elements: int = 0
    exact: bool = True
    reference_sum: float = 0.0
    actual_sum: float = 0.0
    reference_squared_sum: float = 0.0
    actual_squared_sum: float = 0.0
    product_sum: float = 0.0
    delta_squared_sum: float = 0.0
    absolute_delta_sum: float = 0.0
    max_abs: float = 0.0

    def add(self, reference: torch.Tensor, actual: torch.Tensor) -> None:
        if reference.shape != actual.shape:
            raise TraceFormatError(
                f"float tensor shape differs: {tuple(reference.shape)} != "
                f"{tuple(actual.shape)}"
            )
        if not reference.is_floating_point() or not actual.is_floating_point():
            raise TraceFormatError("float drift metrics require floating tensors")
        reference_float = reference.detach().to(dtype=torch.float64, device="cpu")
        actual_float = actual.detach().to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(reference_float).all():
            raise TraceFormatError("official trace contains non-finite values")
        if not torch.isfinite(actual_float).all():
            raise TraceFormatError("interim trace contains non-finite values")
        delta = actual_float - reference_float
        self.elements += reference_float.numel()
        self.exact = self.exact and torch.equal(reference, actual)
        self.reference_sum += float(reference_float.sum())
        self.actual_sum += float(actual_float.sum())
        self.reference_squared_sum += float(reference_float.square().sum())
        self.actual_squared_sum += float(actual_float.square().sum())
        self.product_sum += float((reference_float * actual_float).sum())
        self.delta_squared_sum += float(delta.square().sum())
        self.absolute_delta_sum += float(delta.abs().sum())
        if delta.numel():
            self.max_abs = max(self.max_abs, float(delta.abs().max()))

    def as_dict(self) -> dict[str, Any]:
        reference_norm = math.sqrt(max(0.0, self.reference_squared_sum))
        actual_norm = math.sqrt(max(0.0, self.actual_squared_sum))
        delta_norm = math.sqrt(max(0.0, self.delta_squared_sum))
        if reference_norm == 0:
            relative_l2 = 0.0 if actual_norm == 0 else None
        else:
            relative_l2 = delta_norm / reference_norm
        if reference_norm == 0 and actual_norm == 0:
            cosine = 1.0
        elif reference_norm == 0 or actual_norm == 0:
            cosine = 0.0
        else:
            cosine = self.product_sum / (reference_norm * actual_norm)

        pearson = _pearson_from_moments(
            self.elements,
            self.reference_sum,
            self.actual_sum,
            self.reference_squared_sum,
            self.actual_squared_sum,
            self.product_sum,
            exact=self.exact,
        )
        return {
            "elements": self.elements,
            "exact": self.exact,
            "relative_l2": relative_l2,
            "cosine": cosine,
            "pearson": pearson,
            "mean_abs": (
                self.absolute_delta_sum / self.elements if self.elements else 0.0
            ),
            "max_abs": self.max_abs,
            "reference_norm": reference_norm,
            "actual_norm": actual_norm,
        }


def _pearson_from_moments(
    count: int,
    left_sum: float,
    right_sum: float,
    left_squared_sum: float,
    right_squared_sum: float,
    product_sum: float,
    *,
    exact: bool,
) -> float:
    if count == 0:
        return 1.0
    left_centered = left_squared_sum - left_sum * left_sum / count
    right_centered = right_squared_sum - right_sum * right_sum / count
    covariance = product_sum - left_sum * right_sum / count
    # Roundoff can make a mathematically zero variance slightly negative.
    left_centered = max(0.0, left_centered)
    right_centered = max(0.0, right_centered)
    denominator = math.sqrt(left_centered * right_centered)
    if denominator == 0:
        return 1.0 if exact else 0.0
    return max(-1.0, min(1.0, covariance / denominator))


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _comparison_tensor(run: TraceRun, key: TraceKey) -> torch.Tensor:
    try:
        rank_tensors = run.tensors[key]
    except KeyError as exc:
        raise TraceFormatError(f"{run.root}: missing trace tensor {key}") from exc
    if key.stage not in PARTIAL_STAGES:
        reference_rank = run.captured_ranks[0]
        reference = rank_tensors[reference_rank]
        mismatches = [
            rank
            for rank in run.captured_ranks[1:]
            if not torch.equal(reference, rank_tensors[rank])
        ]
        if mismatches:
            raise TraceFormatError(
                f"{run.root}: replicated tensor {key} differs on ranks "
                f"{mismatches}"
            )
        return reference
    expected_ranks = tuple(range(run.world_size))
    if run.captured_ranks != expected_ranks:
        raise TraceFormatError(
            f"{run.root}: {key.stage} requires all TP ranks; "
            f"captured={run.captured_ranks}, expected={expected_ranks}"
        )
    return torch.stack(
        [rank_tensors[rank].float() for rank in run.captured_ranks]
    ).sum(dim=0)


def _stage_tensor(run: TraceRun, layer: int, stage: str, call: int) -> torch.Tensor:
    return _comparison_tensor(run, TraceKey(layer, call, stage))


def _validate_routes(ids: torch.Tensor, *, num_experts: int, label: str) -> None:
    if ids.ndim != 2 or ids.shape[1] <= 0:
        raise TraceFormatError(f"{label} route IDs must have shape [rows, top_k]")
    if ids.is_floating_point():
        raise TraceFormatError(f"{label} route IDs must be integral")
    ids_long = ids.long()
    if ids_long.numel() and (
        int(ids_long.min()) < 0 or int(ids_long.max()) >= num_experts
    ):
        raise TraceFormatError(
            f"{label} route IDs fall outside [0, {num_experts})"
        )
    if any(len(set(row.tolist())) != ids.shape[1] for row in ids_long):
        raise TraceFormatError(f"{label} contains duplicate experts in a route row")


def _sparse_gate_weights(
    ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    if ids.shape != weights.shape:
        raise TraceFormatError(
            f"route ID/weight shape differs: {tuple(ids.shape)} != "
            f"{tuple(weights.shape)}"
        )
    if not weights.is_floating_point() or not torch.isfinite(weights).all():
        raise TraceFormatError("route weights must be finite floating values")
    sparse = torch.zeros(
        (ids.shape[0], num_experts), dtype=torch.float64, device="cpu"
    )
    sparse.scatter_(1, ids.long().cpu(), weights.to(dtype=torch.float64, device="cpu"))
    return sparse


@dataclass(frozen=True)
class _MatchedPostSitu:
    official: torch.Tensor
    interim: torch.Tensor
    official_weights: torch.Tensor
    interim_weights: torch.Tensor
    expert_ids: torch.Tensor


def _match_post_situ_routes(
    official_ids: torch.Tensor,
    interim_ids: torch.Tensor,
    official_weights: torch.Tensor,
    interim_weights: torch.Tensor,
    official_values: torch.Tensor,
    interim_values: torch.Tensor,
) -> _MatchedPostSitu:
    if official_values.ndim != 3:
        raise TraceFormatError("official routed_post_situ must be rank three")
    if interim_values.ndim != 3:
        raise TraceFormatError("interim routed_post_situ must be rank three")
    expected_official = (*official_ids.shape, official_values.shape[-1])
    expected_interim = (*interim_ids.shape, interim_values.shape[-1])
    if tuple(official_values.shape) != expected_official:
        raise TraceFormatError(
            "official routed_post_situ shape does not match route IDs"
        )
    if tuple(interim_values.shape) != expected_interim:
        raise TraceFormatError(
            "interim routed_post_situ shape does not match route IDs"
        )
    if official_values.shape[-1] != interim_values.shape[-1]:
        raise TraceFormatError("official/interim post-SiTU widths differ")

    official_rows: list[torch.Tensor] = []
    interim_rows: list[torch.Tensor] = []
    official_gate_values: list[torch.Tensor] = []
    interim_gate_values: list[torch.Tensor] = []
    matched_experts: list[int] = []
    for row in range(official_ids.shape[0]):
        interim_slots = {
            int(expert_id): slot
            for slot, expert_id in enumerate(interim_ids[row].tolist())
        }
        for official_slot, expert_id_tensor in enumerate(official_ids[row]):
            expert_id = int(expert_id_tensor)
            interim_slot = interim_slots.get(expert_id)
            if interim_slot is None:
                continue
            official_rows.append(official_values[row, official_slot].detach().cpu())
            interim_rows.append(interim_values[row, interim_slot].detach().cpu())
            official_gate_values.append(
                official_weights[row, official_slot].detach().cpu()
            )
            interim_gate_values.append(
                interim_weights[row, interim_slot].detach().cpu()
            )
            matched_experts.append(expert_id)
    if not official_rows:
        raise TraceFormatError("no common routed experts for post-SiTU comparison")
    return _MatchedPostSitu(
        official=torch.stack(official_rows),
        interim=torch.stack(interim_rows),
        official_weights=torch.stack(official_gate_values).float(),
        interim_weights=torch.stack(interim_gate_values).float(),
        expert_ids=torch.tensor(matched_experts, dtype=torch.int64),
    )


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    flat = values.detach().to(dtype=torch.float64, device="cpu").flatten()
    order = sorted(range(flat.numel()), key=lambda index: (float(flat[index]), index))
    ranks = torch.empty(flat.numel(), dtype=torch.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and flat[order[stop]] == flat[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or not left.numel():
        raise ValueError("Spearman inputs must have the same nonempty shape")
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    moments = _FloatMoments()
    moments.add(left_ranks, right_ranks)
    return float(moments.as_dict()["pearson"])


def _rate_ladder_overlap(
    official_records: torch.Tensor,
    interim_records: torch.Tensor,
) -> dict[str, dict[str, float]]:
    if official_records.shape != interim_records.shape:
        raise ValueError("record score shapes differ")
    records = official_records.numel()
    result: dict[str, dict[str, float]] = {}
    for r in (1, 2, 3, 5):
        if 2 * r > records:
            continue
        official_order = official_records.argsort().tolist()
        interim_order = interim_records.argsort().tolist()
        official_donors = set(official_order[:r])
        interim_donors = set(interim_order[:r])
        official_recipients = set(official_order[-r:])
        interim_recipients = set(interim_order[-r:])
        result[f"R{r}"] = {
            "donor_record_overlap_fraction": (
                len(official_donors & interim_donors) / r
            ),
            "recipient_record_overlap_fraction": (
                len(official_recipients & interim_recipients) / r
            ),
        }
    return result


def _post_situ_ranking(
    matched: _MatchedPostSitu,
    *,
    record_channels: int = 128,
) -> dict[str, Any]:
    width = matched.official.shape[1]
    if width % record_channels:
        raise TraceFormatError(
            f"post-SiTU width {width} is not divisible by {record_channels}"
        )
    official_mass = matched.official_weights.double().square()
    interim_mass = matched.interim_weights.double().square()
    if float(official_mass.sum()) <= 0 or float(interim_mass.sum()) <= 0:
        raise TraceFormatError("post-SiTU ranking has non-positive gate mass")
    official_channel = (
        matched.official.double().square() * official_mass.unsqueeze(1)
    ).sum(dim=0) / official_mass.sum()
    interim_channel = (
        matched.interim.double().square() * interim_mass.unsqueeze(1)
    ).sum(dim=0) / interim_mass.sum()
    channel_metrics = _FloatMoments()
    channel_metrics.add(official_channel, interim_channel)
    official_records = official_channel.view(-1, record_channels).sum(dim=1)
    interim_records = interim_channel.view(-1, record_channels).sum(dim=1)
    record_metrics = _FloatMoments()
    record_metrics.add(official_records, interim_records)
    return {
        "common_routes": int(matched.official.shape[0]),
        "channel_energy": channel_metrics.as_dict(),
        "record_energy": record_metrics.as_dict(),
        "record_rank_spearman": _spearman(
            official_records, interim_records
        ),
        "rate_ladder_record_overlap": _rate_ladder_overlap(
            official_records, interim_records
        ),
    }


def align_routed_post_situ(
    topk_ids: torch.Tensor,
    values_by_expert: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Align expert-batched post-SiTU values back to ``[token, route, K]``.

    The official reference groups routes using ``topk_ids.flatten().argsort()``.
    Repeating that exact ordering is necessary: positions within one expert
    are not guaranteed to retain their original order under an unstable sort.
    """
    if topk_ids.ndim != 2 or not topk_ids.numel():
        raise ValueError("topk_ids must be a nonempty [token, route] tensor")
    flat_ids = topk_ids.reshape(-1).long()
    grouped_positions = flat_ids.argsort()
    expected_experts = sorted({int(value) for value in flat_ids.tolist()})
    if sorted(values_by_expert) != expected_experts:
        raise ValueError(
            "post-SiTU expert coverage differs from routes: "
            f"captured={sorted(values_by_expert)}, expected={expected_experts}"
        )

    feature_width: int | None = None
    output: torch.Tensor | None = None
    cursor = 0
    for expert_id in expected_experts:
        value = values_by_expert[expert_id]
        if value.ndim != 2:
            raise ValueError(
                f"expert {expert_id} post-SiTU output must be rank two"
            )
        count = int((flat_ids == expert_id).sum())
        if value.shape[0] != count:
            raise ValueError(
                f"expert {expert_id} post-SiTU rows {value.shape[0]} != {count}"
            )
        if feature_width is None:
            feature_width = int(value.shape[1])
            output = torch.empty(
                (flat_ids.numel(), feature_width),
                dtype=value.dtype,
                device=value.device,
            )
        elif value.shape[1] != feature_width:
            raise ValueError("post-SiTU feature widths differ between experts")
        assert output is not None
        if value.device != output.device or value.dtype != output.dtype:
            raise ValueError("post-SiTU dtype/device differs between experts")
        positions = grouped_positions[cursor : cursor + count]
        output[positions] = value
        cursor += count

    if output is None or feature_width is None or cursor != flat_ids.numel():
        raise ValueError("post-SiTU alignment did not consume every route")
    return output.view(*topk_ids.shape, feature_width)


def compare_teacher_proxy_traces(
    official: TraceRun,
    interim: TraceRun,
    *,
    call: int = 0,
    num_experts: int = NUM_EXPERTS,
    float_stages: tuple[str, ...] = DEFAULT_FLOAT_STAGES,
    min_expert_post_situ_routes: int = 4,
) -> dict[str, Any]:
    """Compare official and interim traces without claiming a quality pass."""
    if call < 0:
        raise ValueError("call must be non-negative")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if min_expert_post_situ_routes <= 0:
        raise ValueError("min_expert_post_situ_routes must be positive")
    official_layers = {
        key.layer
        for key in official.tensors
        if key.call == call and key.stage == ROUTE_IDS_STAGE
    }
    interim_layers = {
        key.layer
        for key in interim.tensors
        if key.call == call and key.stage == ROUTE_IDS_STAGE
    }
    if official_layers != interim_layers or not official_layers:
        raise TraceFormatError(
            "official/interim routed-layer coverage differs or is empty: "
            f"official={sorted(official_layers)}, interim={sorted(interim_layers)}"
        )
    official_post_situ_layers = {
        key.layer
        for key in official.tensors
        if key.call == call and key.stage == POST_SITU_STAGE
    }
    interim_post_situ_layers = {
        key.layer
        for key in interim.tensors
        if key.call == call and key.stage == POST_SITU_STAGE
    }
    if official_post_situ_layers != interim_post_situ_layers:
        raise TraceFormatError(
            "official/interim routed_post_situ coverage differs: "
            f"official={sorted(official_post_situ_layers)}, "
            f"interim={sorted(interim_post_situ_layers)}"
        )
    if not official_post_situ_layers <= official_layers:
        raise TraceFormatError("routed_post_situ exists outside routed layers")

    aggregate_stages = {stage: _FloatMoments() for stage in float_stages}
    aggregate_sparse_gate = _FloatMoments()
    aggregate_common_gate = _FloatMoments()
    aggregate_post_situ = _FloatMoments()
    layer_reports: list[dict[str, Any]] = []
    post_situ_layer_reports: list[dict[str, Any]] = []
    post_situ_expert_reports: list[dict[str, Any]] = []
    row_jaccards: list[float] = []
    route_assignments = 0
    common_assignments = 0
    position_matches = 0
    exact_set_rows = 0
    route_rows = 0

    for layer in sorted(official_layers):
        official_ids = _stage_tensor(official, layer, ROUTE_IDS_STAGE, call)
        interim_ids = _stage_tensor(interim, layer, ROUTE_IDS_STAGE, call)
        if official_ids.shape != interim_ids.shape:
            raise TraceFormatError(
                f"layer {layer}: route shape differs: "
                f"{tuple(official_ids.shape)} != {tuple(interim_ids.shape)}"
            )
        _validate_routes(
            official_ids, num_experts=num_experts, label=f"official layer {layer}"
        )
        _validate_routes(
            interim_ids, num_experts=num_experts, label=f"interim layer {layer}"
        )
        official_ids = official_ids.long().cpu()
        interim_ids = interim_ids.long().cpu()
        layer_common = 0
        layer_exact_rows = 0
        layer_jaccards: list[float] = []
        for official_row, interim_row in zip(official_ids, interim_ids):
            official_set = set(official_row.tolist())
            interim_set = set(interim_row.tolist())
            intersection = len(official_set & interim_set)
            union = len(official_set | interim_set)
            layer_common += intersection
            layer_exact_rows += int(official_set == interim_set)
            layer_jaccards.append(intersection / union if union else 1.0)

        layer_assignments = official_ids.numel()
        layer_position_matches = int((official_ids == interim_ids).sum())
        route_assignments += layer_assignments
        common_assignments += layer_common
        position_matches += layer_position_matches
        exact_set_rows += layer_exact_rows
        route_rows += official_ids.shape[0]
        row_jaccards.extend(layer_jaccards)

        official_weights = _stage_tensor(
            official, layer, ROUTE_WEIGHTS_STAGE, call
        )
        interim_weights = _stage_tensor(interim, layer, ROUTE_WEIGHTS_STAGE, call)
        official_sparse = _sparse_gate_weights(
            official_ids, official_weights, num_experts=num_experts
        )
        interim_sparse = _sparse_gate_weights(
            interim_ids, interim_weights, num_experts=num_experts
        )
        sparse_gate = _FloatMoments()
        sparse_gate.add(official_sparse, interim_sparse)
        aggregate_sparse_gate.add(official_sparse, interim_sparse)

        official_common: list[float] = []
        interim_common: list[float] = []
        for row_index, (official_row, interim_row) in enumerate(
            zip(official_ids, interim_ids)
        ):
            common = sorted(set(official_row.tolist()) & set(interim_row.tolist()))
            for expert_id in common:
                official_common.append(float(official_sparse[row_index, expert_id]))
                interim_common.append(float(interim_sparse[row_index, expert_id]))
        common_gate = _FloatMoments()
        if official_common:
            official_common_tensor = torch.tensor(official_common, dtype=torch.float64)
            interim_common_tensor = torch.tensor(interim_common, dtype=torch.float64)
            common_gate.add(official_common_tensor, interim_common_tensor)
            aggregate_common_gate.add(
                official_common_tensor, interim_common_tensor
            )

        layer_float: dict[str, Any] = {}
        for stage in float_stages:
            official_tensor = _stage_tensor(official, layer, stage, call)
            interim_tensor = _stage_tensor(interim, layer, stage, call)
            moments = _FloatMoments()
            moments.add(official_tensor, interim_tensor)
            aggregate_stages[stage].add(official_tensor, interim_tensor)
            layer_float[stage] = moments.as_dict()

        post_situ_report: dict[str, Any] | None = None
        if layer in official_post_situ_layers:
            matched = _match_post_situ_routes(
                official_ids,
                interim_ids,
                official_weights,
                interim_weights,
                _stage_tensor(official, layer, POST_SITU_STAGE, call),
                _stage_tensor(interim, layer, POST_SITU_STAGE, call),
            )
            direct = _FloatMoments()
            direct.add(matched.official, matched.interim)
            aggregate_post_situ.add(matched.official, matched.interim)
            post_situ_report = {
                "direct_common_route_activation": direct.as_dict(),
                **_post_situ_ranking(matched),
            }
            post_situ_layer_reports.append(
                {"layer": layer, **post_situ_report}
            )
            for expert_id in sorted(set(matched.expert_ids.tolist())):
                selected = matched.expert_ids == expert_id
                routes = int(selected.sum())
                if routes < min_expert_post_situ_routes:
                    continue
                expert_matched = _MatchedPostSitu(
                    official=matched.official[selected],
                    interim=matched.interim[selected],
                    official_weights=matched.official_weights[selected],
                    interim_weights=matched.interim_weights[selected],
                    expert_ids=matched.expert_ids[selected],
                )
                post_situ_expert_reports.append(
                    {
                        "layer": layer,
                        "expert": expert_id,
                        **_post_situ_ranking(expert_matched),
                    }
                )

        layer_reports.append(
            {
                "layer": layer,
                "token_rows": int(official_ids.shape[0]),
                "top_k": int(official_ids.shape[1]),
                "route_assignment_retention_fraction": (
                    layer_common / layer_assignments
                ),
                "route_assignment_replacement_fraction": (
                    1 - layer_common / layer_assignments
                ),
                "route_exact_set_row_fraction": (
                    layer_exact_rows / official_ids.shape[0]
                ),
                "route_position_match_fraction": (
                    layer_position_matches / layer_assignments
                ),
                "route_row_jaccard_mean": (
                    sum(layer_jaccards) / len(layer_jaccards)
                ),
                "sparse_applied_gate": sparse_gate.as_dict(),
                "common_route_gate": common_gate.as_dict(),
                "float_stages": layer_float,
                "post_situ": post_situ_report,
            }
        )

    replacement_ranked = sorted(
        (
            {
                "layer": report["layer"],
                "route_assignment_replacement_fraction": report[
                    "route_assignment_replacement_fraction"
                ],
            }
            for report in layer_reports
        ),
        key=lambda item: (
            -item["route_assignment_replacement_fraction"],
            item["layer"],
        ),
    )
    if post_situ_layer_reports:
        layer_spearman = [
            float(item["record_rank_spearman"])
            for item in post_situ_layer_reports
        ]
        expert_spearman = [
            float(item["record_rank_spearman"])
            for item in post_situ_expert_reports
        ]
        post_situ_summary: dict[str, Any] = {
            "available": True,
            "captured_layers": sorted(official_post_situ_layers),
            "direct_common_route_activation": aggregate_post_situ.as_dict(),
            "layer_record_rank_spearman": {
                "count": len(layer_spearman),
                "mean": sum(layer_spearman) / len(layer_spearman),
                "minimum": min(layer_spearman),
                "p10": _quantile(layer_spearman, 0.10),
                "median": _quantile(layer_spearman, 0.50),
            },
            "expert_record_rank_spearman": {
                "minimum_common_routes": min_expert_post_situ_routes,
                "count": len(expert_spearman),
                "mean": (
                    sum(expert_spearman) / len(expert_spearman)
                    if expert_spearman
                    else None
                ),
                "minimum": min(expert_spearman) if expert_spearman else None,
                "p10": _quantile(expert_spearman, 0.10),
                "median": _quantile(expert_spearman, 0.50),
            },
            "layers": post_situ_layer_reports,
            "experts": post_situ_expert_reports,
        }
    else:
        post_situ_summary = {
            "available": False,
            "required_stage": POST_SITU_STAGE,
            "reason": (
                "trace predates --capture-routed-post-situ; neuron-ranking "
                "agreement was not measured"
            ),
        }
    return {
        "schema_version": TEACHER_PROXY_SCHEMA_VERSION,
        "kind": "kquant_teacher_proxy_drift_anchor",
        "status": "descriptive_only",
        "scope": {
            "token_count": official.manifest.get("max_tokens"),
            "routed_layer_count": len(layer_reports),
            "call": call,
            "valid_for": [
                "small-prompt routing-drift baseline",
                "small-prompt activation-drift trajectory",
                "instrumentation smoke test",
            ],
            "not_valid_for": [
                "neuron-ranking agreement",
                "mode-selection generalization",
                "production-distribution quality",
                "accepting or rejecting the interim teacher",
            ],
        },
        "official": {
            "root": str(official.root),
            "checkpoint": official.manifest.get("checkpoint"),
            "tp_world_size": official.world_size,
        },
        "interim": {
            "root": str(interim.root),
            "checkpoint": interim.manifest.get("checkpoint"),
            "tp_world_size": interim.world_size,
        },
        "route_summary": {
            "token_layer_rows": route_rows,
            "route_assignments": route_assignments,
            "common_route_assignments": common_assignments,
            "route_assignment_retention_fraction": (
                common_assignments / route_assignments
            ),
            "route_assignment_replacement_fraction": (
                1 - common_assignments / route_assignments
            ),
            "exact_route_set_rows": exact_set_rows,
            "exact_route_set_row_fraction": exact_set_rows / route_rows,
            "route_position_match_fraction": position_matches / route_assignments,
            "row_jaccard_mean": sum(row_jaccards) / len(row_jaccards),
            "row_jaccard_p10": _quantile(row_jaccards, 0.10),
            "row_jaccard_median": _quantile(row_jaccards, 0.50),
            "row_jaccard_p90": _quantile(row_jaccards, 0.90),
            "sparse_applied_gate": aggregate_sparse_gate.as_dict(),
            "common_route_gate": aggregate_common_gate.as_dict(),
            "worst_route_replacement_layers": replacement_ranked[:10],
        },
        "float_stage_summary": {
            stage: moments.as_dict()
            for stage, moments in aggregate_stages.items()
        },
        "post_situ_summary": post_situ_summary,
        "layers": layer_reports,
    }
