#!/usr/bin/env python3
"""Compare selection-corpus and document-disjoint QSRT keep rankings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from kquant import constants as C
from kquant.pack.qsrt_pool import (
    CERTIFIED_ALLOCATION_OPTIMALITY,
    raw_keep_allocation_optimality,
    choose_qsrt_raw_keep_allocation,
    keep_mask_from_allocation_document,
    load_qsrt_candidate_pool,
    raw_keep_container_bytes,
)
from kquant.pack.qsrt_validation import load_qsrt_validation_scores


SUMMARY_KIND = "kquant_kimi_k3_qsrt_validation_summary"
SUMMARY_SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks with stable handling of exact ties."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("rank values must be finite")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    begin = 0
    while begin < values.size:
        end = begin + 1
        while end < values.size and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2.0
        begin = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not left.size:
        raise ValueError("correlation vectors must have equal nonempty shapes")
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else None


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _correlation(_average_ranks(left), _average_ranks(right))


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    probabilities = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    quantiles = np.quantile(values, probabilities)
    return {
        f"p{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, quantiles, strict=True)
    }


def _allocation_score(keep: np.ndarray, damage: np.ndarray) -> dict[str, float | int]:
    total = float(damage.sum())
    avoided = float(damage[keep].sum())
    remaining = float(damage[~keep].sum())
    return {
        "retained_experts": int(keep.sum()),
        "total_damage": total,
        "retained_damage_avoided": avoided,
        "remaining_compressed_damage": remaining,
        "fraction_damage_avoided": avoided / total if total > 0 else 0.0,
    }


def compare_damage_rankings(
    selection_damage: np.ndarray,
    validation_damage: np.ndarray,
    *,
    target_container_bytes: int,
) -> dict[str, object]:
    """Compare two complete damage fields at one exact physical byte target."""

    selection_damage = np.asarray(selection_damage, dtype=np.float64)
    validation_damage = np.asarray(validation_damage, dtype=np.float64)
    if selection_damage.shape != validation_damage.shape or selection_damage.ndim != 2:
        raise ValueError("damage fields must be equal two-dimensional shapes")
    if (
        not np.isfinite(selection_damage).all()
        or not np.isfinite(validation_damage).all()
        or np.any(selection_damage < 0)
        or np.any(validation_damage < 0)
    ):
        raise ValueError("damage fields must be finite and non-negative")
    selection_allocation = choose_qsrt_raw_keep_allocation(
        selection_damage, target_container_bytes=target_container_bytes
    )
    validation_allocation = choose_qsrt_raw_keep_allocation(
        validation_damage, target_container_bytes=target_container_bytes
    )
    selection_keep = selection_allocation.keep_mask
    validation_keep = validation_allocation.keep_mask
    intersection = int(np.count_nonzero(selection_keep & validation_keep))
    union = int(np.count_nonzero(selection_keep | validation_keep))
    per_layer_spearman = [
        _spearman(selection_damage[row], validation_damage[row])
        for row in range(selection_damage.shape[0])
    ]
    validation_optimal = _allocation_score(validation_keep, validation_damage)
    validation_from_selection = _allocation_score(
        selection_keep, validation_damage
    )
    return {
        "target_container_bytes": int(target_container_bytes),
        "global_spearman": _spearman(selection_damage, validation_damage),
        "normalized_damage_share_pearson": _correlation(
            selection_damage / selection_damage.sum()
            if selection_damage.sum() > 0
            else selection_damage,
            validation_damage / validation_damage.sum()
            if validation_damage.sum() > 0
            else validation_damage,
        ),
        "per_layer_spearman_quantiles": _quantiles(
            np.asarray(
                [value for value in per_layer_spearman if value is not None]
            )
        ),
        "keep_set": {
            "selection_retained": int(selection_keep.sum()),
            "validation_retained": int(validation_keep.sum()),
            "intersection": intersection,
            "union": union,
            "jaccard": intersection / union if union else 1.0,
            "selection_container_bytes": selection_allocation.container_bytes,
            "validation_container_bytes": validation_allocation.container_bytes,
            "selection_optimality": raw_keep_allocation_optimality(
                selection_allocation
            ),
            "validation_optimality": raw_keep_allocation_optimality(
                validation_allocation
            ),
        },
        "selection_ranked_allocation": {
            "scored_on_selection": _allocation_score(
                selection_keep, selection_damage
            ),
            "scored_on_validation": validation_from_selection,
        },
        "validation_ranked_allocation": {
            "scored_on_selection": _allocation_score(
                validation_keep, selection_damage
            ),
            "scored_on_validation": validation_optimal,
        },
        "validation_ranking_gain_over_selection_ranking": {
            "additional_validation_damage_avoided": (
                validation_optimal["retained_damage_avoided"]
                - validation_from_selection["retained_damage_avoided"]
            ),
            "relative_remaining_damage_reduction": (
                1.0
                - validation_optimal["remaining_compressed_damage"]
                / validation_from_selection["remaining_compressed_damage"]
                if validation_from_selection["remaining_compressed_damage"] > 0
                else 0.0
            ),
        },
    }


def bootstrap_keep_stability(
    per_document_damage: np.ndarray,
    *,
    target_container_bytes: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Estimate exact-budget keep-set stability under document resampling."""

    values = np.asarray(per_document_damage, dtype=np.float64)
    if values.ndim != 3 or values.shape[:2] != (
        C.NUM_MOE_LAYERS,
        C.NUM_EXPERTS,
    ):
        raise ValueError(
            "per-document damage must have shape [92, 896, documents]"
        )
    if values.shape[2] < 2:
        raise ValueError("keep stability requires at least two documents")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("per-document damage must be finite and non-negative")
    if isinstance(replicates, bool) or replicates < 10:
        raise ValueError("bootstrap replicates must be at least 10")

    documents = values.shape[2]
    full_damage = values.sum(axis=2)
    base_allocation = choose_qsrt_raw_keep_allocation(
        full_damage,
        target_container_bytes=target_container_bytes,
    )
    base = base_allocation.keep_mask
    base_remaining = float(full_damage[~base].sum())
    inclusion_counts = np.zeros(base.shape, dtype=np.uint32)
    jaccard = np.empty(replicates, dtype=np.float64)
    churn = np.empty(replicates, dtype=np.float64)
    full_damage_regret = np.empty(replicates, dtype=np.float64)
    exact_matches = 0
    uncertified_replicates = 0
    generator = np.random.default_rng(seed)
    for replicate in range(replicates):
        multiplicity = np.bincount(
            generator.integers(0, documents, size=documents),
            minlength=documents,
        ).astype(np.float64)
        sampled_damage = np.einsum(
            "led,d->le",
            values,
            multiplicity,
            optimize=True,
        )
        sampled_allocation = choose_qsrt_raw_keep_allocation(
            sampled_damage,
            target_container_bytes=target_container_bytes,
        )
        sampled = sampled_allocation.keep_mask
        uncertified_replicates += int(
            raw_keep_allocation_optimality(sampled_allocation)
            != CERTIFIED_ALLOCATION_OPTIMALITY
        )
        inclusion_counts += sampled
        intersection = int(np.count_nonzero(base & sampled))
        union = int(np.count_nonzero(base | sampled))
        jaccard[replicate] = intersection / union if union else 1.0
        churn[replicate] = np.count_nonzero(base ^ sampled)
        full_damage_regret[replicate] = (
            float(full_damage[~sampled].sum()) - base_remaining
        )
        exact_matches += int(np.array_equal(base, sampled))

    probability = inclusion_counts.astype(np.float64) / replicates
    ambiguous = (probability > 0.1) & (probability < 0.9)
    relative_regret = (
        full_damage_regret / base_remaining
        if base_remaining > 0
        else np.zeros_like(full_damage_regret)
    )
    return {
        "documents": documents,
        "replicates": replicates,
        "seed": seed,
        "base_retained_experts": int(base.sum()),
        "base_allocation_optimality": raw_keep_allocation_optimality(
            base_allocation
        ),
        "uncertified_alignment_repair_frequency": (
            uncertified_replicates / replicates
        ),
        "exact_base_keep_set_frequency": exact_matches / replicates,
        "ambiguous_assignments_probability_10_to_90": int(ambiguous.sum()),
        "base_keep_inclusion_probability_quantiles": _quantiles(
            probability[base]
        ),
        "base_compressed_inclusion_probability_quantiles": _quantiles(
            probability[~base]
        ),
        "replicate_keep_set_jaccard_quantiles": _quantiles(jaccard),
        "replicate_assignment_churn_quantiles": _quantiles(churn),
        "full_validation_remaining_damage_regret_quantiles": _quantiles(
            full_damage_regret
        ),
        "relative_remaining_damage_regret_quantiles": _quantiles(
            relative_regret
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, object]:
    pool = load_qsrt_candidate_pool(args.candidate_pool)
    validation = load_qsrt_validation_scores(args.validation_scores, pool)
    reference_allocation = _read_json(args.target_allocation)
    target_bytes = raw_keep_container_bytes(
        keep_mask_from_allocation_document(reference_allocation)
    )
    comparison = compare_damage_rankings(
        pool.damage,
        validation.damage,
        target_container_bytes=target_bytes,
    )
    stability = bootstrap_keep_stability(
        validation.per_document_damage,
        target_container_bytes=target_bytes,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    supported = validation.routed_rows > 0
    format_support: dict[str, dict[str, object]] = {}
    for r13 in pool.manifest["mode_ids"]:
        for r2 in pool.manifest["mode_ids"]:
            selected = (pool.selected_r13 == r13) & (pool.selected_r2 == r2)
            format_support[f"R{r13}/R{r2}"] = {
            "assignments": int(selected.sum()),
            "assignments_with_validation_rows": int(
                np.count_nonzero(selected & supported)
            ),
            "validation_damage": float(validation.damage[selected].sum()),
            "validation_reference_energy": float(
                validation.reference_energy[selected].sum()
            ),
            }
    return {
        "kind": SUMMARY_KIND,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "candidate_pool": str(pool.root),
        "validation_scores": str(validation.root),
        "target_allocation": str(args.target_allocation.resolve()),
        "target_container_bytes": target_bytes,
        "support": {
            "assignments": int(supported.size),
            "assignments_with_validation_rows": int(supported.sum()),
            "assignments_without_validation_rows": int((~supported).sum()),
            "routed_row_quantiles": _quantiles(validation.routed_rows),
            "supported_document_quantiles": _quantiles(
                validation.supported_documents
            ),
        },
        "per_selected_format": format_support,
        "ranking_comparison": comparison,
        "document_bootstrap_keep_stability": stability,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--validation-scores", type=Path, required=True)
    parser.add_argument("--target-allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(args)
    _atomic_json(args.output, result)
    comparison = result["ranking_comparison"]
    print(
        f"wrote {args.output}: Spearman {comparison['global_spearman']}, "
        f"keep Jaccard {comparison['keep_set']['jaccard']:.6f}"
    )


if __name__ == "__main__":
    main()
