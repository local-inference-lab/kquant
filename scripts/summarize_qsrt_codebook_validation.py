"""Aggregate controlled QSRT reconstruction-law validation runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


FOLDS = ("train", "confirmation", "validation")


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a JSON object")
    return payload


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _metric(candidate: Mapping[str, object], name: str) -> tuple[float, float]:
    matrices = candidate["matrix_evidence"]
    if name == "weight":
        return (
            sum(float(item["weight_squared_error"]) for item in matrices),
            sum(float(item["weight_reference_energy"]) for item in matrices),
        )
    if name == "dense_h":
        return (
            sum(float(item["captured_dense_h_numerator"]) for item in matrices),
            sum(float(item["captured_dense_h_denominator"]) for item in matrices),
        )
    fold = candidate["routed_function"][name]
    return (
        float(fold["route_weighted_squared_error"]),
        float(fold["route_weighted_reference_energy"]),
    )


def _aggregate_metric(
    candidates: Sequence[Mapping[str, object]], name: str
) -> dict[str, float]:
    pairs = [_metric(candidate, name) for candidate in candidates]
    numerator = sum(pair[0] for pair in pairs)
    denominator = sum(pair[1] for pair in pairs)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "nmse": numerator / denominator,
    }


def _paired_document_bootstrap(
    expert_evidence: Sequence[Mapping[str, object]],
    *,
    baseline: str,
    candidate: str,
    bits: int,
    fold: str,
    replicates: int,
    seed: int,
) -> dict[str, object] | None:
    totals: dict[str, dict[str, float]] = {
        baseline: defaultdict(float),
        candidate: defaultdict(float),
    }
    for evidence in expert_evidence:
        for codebook in (baseline, candidate):
            routed = evidence["candidates"][f"{codebook}:K{bits}"][
                "routed_function"
            ][fold]
            contributions = routed.get("request_contributions")
            if contributions is None:
                return None
            for item in contributions:
                document = str(item["document_hash"])
                totals[codebook][document] += float(
                    item["route_weighted_squared_error"]
                )
    documents = sorted(totals[baseline])
    if not documents or set(documents) != set(totals[candidate]):
        raise ValueError("paired codebook candidates changed document support")
    base = np.asarray([totals[baseline][item] for item in documents])
    trial = np.asarray([totals[candidate][item] for item in documents])
    if float(base.sum()) <= 0:
        raise ValueError("paired codebook baseline has non-positive SSE")
    observed = 1.0 - float(trial.sum()) / float(base.sum())
    rng = np.random.default_rng(seed)
    sampled_improvements = []
    # Bound peak memory for large document folds while retaining exact iid
    # document-cluster resampling semantics.
    for begin in range(0, replicates, 1_000):
        count = min(1_000, replicates - begin)
        indices = rng.integers(0, len(documents), size=(count, len(documents)))
        sampled_base = base[indices].sum(axis=1)
        sampled_trial = trial[indices].sum(axis=1)
        valid = sampled_base > 0
        sampled_improvements.append(
            1.0 - sampled_trial[valid] / sampled_base[valid]
        )
    samples = np.concatenate(sampled_improvements)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "documents": len(documents),
        "relative_sse_improvement": observed,
        "ci95": [float(low), float(high)],
        "probability_improvement_is_positive": float(np.mean(samples > 0.0)),
        "valid_bootstrap_replicates": int(samples.size),
    }


def summarize(
    inputs: Sequence[Path],
    *,
    baseline: str,
    candidate: str,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    experts: dict[tuple[int, int], Mapping[str, object]] = {}
    sources: list[str] = []
    skipped: dict[str, object] = {}
    provenance: Mapping[str, object] | None = None
    for path in inputs:
        payload = _read(path)
        if payload.get("kind") != "kquant_uniform_k_mxfp4_endpoint_study":
            raise ValueError(f"{path} is not a uniform-K endpoint study")
        if not payload.get("complete"):
            raise ValueError(f"{path} is incomplete")
        signature = payload["signature"]
        layer = int(signature["layer"])
        if provenance is None:
            provenance = signature["provenance"]
        elif provenance != signature["provenance"]:
            raise ValueError("input studies use different corpus provenance")
        sources.append(str(path.resolve()))
        for raw_expert, reason in payload.get("skipped", {}).items():
            skipped[f"{layer}:{raw_expert}"] = reason
        for raw_expert, evidence in payload["results"].items():
            identity = (layer, int(raw_expert))
            if identity in experts:
                existing = experts[identity]
                for key in ("support", "context", "covariance"):
                    if existing.get(key) != evidence.get(key):
                        raise ValueError(
                            f"duplicate expert {identity} changed {key}"
                        )
                old_candidates = existing["candidates"]
                new_candidates = evidence["candidates"]
                overlap = set(old_candidates) & set(new_candidates)
                if overlap:
                    raise ValueError(
                        f"duplicate expert {identity} repeats candidates "
                        f"{sorted(overlap)}"
                    )
                experts[identity] = {
                    **existing,
                    "candidates": {**old_candidates, **new_candidates},
                }
            else:
                experts[identity] = evidence
    if not experts:
        raise ValueError("no expert results found")

    rates = sorted(
        {
            int(value["bits"])
            for evidence in experts.values()
            for key, value in evidence["candidates"].items()
            if key.startswith(f"{baseline}:")
        }
    )
    metrics = ("weight", "dense_h", *FOLDS)
    rate_results: dict[str, object] = {}
    by_layer: dict[int, list[tuple[int, Mapping[str, object]]]] = defaultdict(list)
    for (layer, expert), evidence in experts.items():
        by_layer[layer].append((expert, evidence))

    for bits in rates:
        baseline_values = []
        candidate_values = []
        for identity, evidence in experts.items():
            candidates = evidence["candidates"]
            baseline_key = f"{baseline}:K{bits}"
            candidate_key = f"{candidate}:K{bits}"
            if baseline_key not in candidates or candidate_key not in candidates:
                raise ValueError(f"expert {identity} lacks a K{bits} comparison")
            baseline_values.append(candidates[baseline_key])
            candidate_values.append(candidates[candidate_key])

        comparisons = {}
        for metric in metrics:
            base = _aggregate_metric(baseline_values, metric)
            trial = _aggregate_metric(candidate_values, metric)
            per_expert = []
            for base_value, trial_value in zip(
                baseline_values, candidate_values, strict=True
            ):
                base_pair = _metric(base_value, metric)
                trial_pair = _metric(trial_value, metric)
                per_expert.append(
                    (trial_pair[0] / trial_pair[1])
                    / (base_pair[0] / base_pair[1])
                )
            comparisons[metric] = {
                "baseline": base,
                "candidate": trial,
                "candidate_over_baseline": trial["nmse"] / base["nmse"],
                "expert_wins": sum(value < 1.0 for value in per_expert),
                "expert_ties": sum(value == 1.0 for value in per_expert),
                "expert_count": len(per_expert),
                "median_expert_ratio": statistics.median(per_expert),
                "minimum_expert_ratio": min(per_expert),
                "maximum_expert_ratio": max(per_expert),
            }
            if metric in FOLDS:
                paired = _paired_document_bootstrap(
                    list(experts.values()),
                    baseline=baseline,
                    candidate=candidate,
                    bits=bits,
                    fold=metric,
                    replicates=bootstrap_replicates,
                    seed=20260803 + bits * 100 + FOLDS.index(metric),
                )
                if paired is not None:
                    comparisons[metric]["paired_document_bootstrap"] = paired

        matrix_comparisons = {}
        for matrix in ("w1", "w3", "w2"):
            ratios = []
            base_num = base_den = trial_num = trial_den = 0.0
            for base_value, trial_value in zip(
                baseline_values, candidate_values, strict=True
            ):
                base_matrix = next(
                    item
                    for item in base_value["matrix_evidence"]
                    if item["matrix"] == matrix
                )
                trial_matrix = next(
                    item
                    for item in trial_value["matrix_evidence"]
                    if item["matrix"] == matrix
                )
                base_num += float(base_matrix["captured_dense_h_numerator"])
                base_den += float(base_matrix["captured_dense_h_denominator"])
                trial_num += float(trial_matrix["captured_dense_h_numerator"])
                trial_den += float(trial_matrix["captured_dense_h_denominator"])
                ratios.append(
                    float(trial_matrix["captured_dense_h_nmse"])
                    / float(base_matrix["captured_dense_h_nmse"])
                )
            matrix_comparisons[matrix] = {
                "candidate_over_baseline": (trial_num / trial_den)
                / (base_num / base_den),
                "expert_wins": sum(value < 1.0 for value in ratios),
                "expert_count": len(ratios),
                "median_expert_ratio": statistics.median(ratios),
                "minimum_expert_ratio": min(ratios),
                "maximum_expert_ratio": max(ratios),
            }
        rate_results[str(bits)] = {
            "metrics": comparisons,
            "dense_h_by_matrix": matrix_comparisons,
        }

    curvature = {}
    if set((2, 3, 4)).issubset(rates):
        for codebook in (baseline, candidate):
            codebook_result = {}
            for metric in ("dense_h", "validation"):
                errors = {}
                for bits in (2, 3, 4):
                    selected = [
                        evidence["candidates"][f"{codebook}:K{bits}"]
                        for evidence in experts.values()
                    ]
                    errors[str(bits)] = sum(
                        _metric(value, metric)[0] for value in selected
                    )
                donor = errors["2"] - errors["3"]
                recipient = errors["3"] - errors["4"]
                codebook_result[metric] = {
                    "absolute_sse": errors,
                    "k2_to_k3_donor_cost": donor,
                    "k3_to_k4_recipient_gain": recipient,
                    "donor_cost_over_recipient_gain": donor / recipient,
                    "uniform_half_k2_half_k4_over_k3": (
                        0.5 * (errors["2"] + errors["4"]) / errors["3"]
                    ),
                }
            curvature[codebook] = codebook_result

    layer_counts = {str(layer): len(items) for layer, items in sorted(by_layer.items())}
    support_prefix = {
        "train": "fit",
        "confirmation": "confirmation",
        "validation": "validation",
    }
    support = {
        fold: {
            "rows": sum(
                int(evidence["support"][f"{support_prefix[fold]}_rows"])
                for evidence in experts.values()
            ),
            "expert_document_assignments": sum(
                int(evidence["support"][f"{support_prefix[fold]}_documents"])
                for evidence in experts.values()
            ),
        }
        for fold in FOLDS
    }
    return {
        "schema_version": 1,
        "kind": "kquant_sqg_e4m3_validation_summary",
        "baseline": baseline,
        "candidate": candidate,
        "sources": sources,
        "provenance": provenance,
        "expert_count": len(experts),
        "experts_by_layer": layer_counts,
        "skipped": skipped,
        "support": support,
        "rates": rate_results,
        "rate_curvature": curvature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--baseline", default="sqg-normal-e4m3")
    parser.add_argument("--candidate", default="sqg-cheb-normal-e4m3")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    payload = summarize(
        args.inputs,
        baseline=args.baseline,
        candidate=args.candidate,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    _write(args.output, payload)
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "experts": payload["expert_count"]}, indent=1))


if __name__ == "__main__":
    main()
