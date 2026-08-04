#!/usr/bin/env python3
"""Add paired matrix-cluster uncertainty to a learned-codebook experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def _group_sse(metrics: dict[str, object], name: str) -> np.ndarray:
    return np.asarray([float(row["sse"]) for row in metrics[name]["groups"]])


def _combined_group_sse(
    experiments: list[dict[str, object]], name: str
) -> np.ndarray:
    return np.concatenate(
        [_group_sse(experiment["metrics"], name) for experiment in experiments]
    )


def _ratio_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    ratios = (
        numerator[bootstrap_indices].sum(axis=1)
        / denominator[bootstrap_indices].sum(axis=1)
    )
    return {
        "aggregate_sse_ratio": float(numerator.sum() / denominator.sum()),
        "paired_matrix_bootstrap_95": [
            float(value) for value in np.quantile(ratios, (0.025, 0.975))
        ],
        "matrix_groups_better": int(np.count_nonzero(numerator < denominator)),
        "matrix_groups_total": int(numerator.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    experiments = [json.loads(path.read_text()) for path in args.experiment]
    if any(
        experiment.get("kind") != "kquant_trellis_learned_codebook_experiment"
        for experiment in experiments
    ):
        raise ValueError("an input is not a learned-codebook experiment")
    metrics = experiments[0]["metrics"]
    metric_names = set(metrics)
    if any(set(experiment["metrics"]) != metric_names for experiment in experiments):
        raise ValueError("experiments do not contain identical metric families")
    baseline_name = "procedural_mcg_fp16"
    baseline = _combined_group_sse(experiments, baseline_name)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, baseline.size, size=(args.replicates, baseline.size))

    candidates = tuple(name for name in metrics if name != baseline_name)
    versus_baseline = {
        name: _ratio_summary(_combined_group_sse(experiments, name), baseline, indices)
        for name in candidates
    }
    pairwise = {}
    for numerator, denominator in (
        ("learned_e4m3", "learned_tied_fp16"),
        ("learned_e4m3", "learned_free_fp16"),
        ("learned_tied_fp16", "learned_free_fp16"),
    ):
        pairwise[f"{numerator}_over_{denominator}"] = _ratio_summary(
            _combined_group_sse(experiments, numerator),
            _combined_group_sse(experiments, denominator),
            indices,
        )

    group_rows = [
        row
        for experiment in experiments
        for row in experiment["metrics"][baseline_name]["groups"]
    ]
    by_matrix = {}
    for matrix in sorted({str(row["matrix"]) for row in group_rows}):
        mask = np.asarray([str(row["matrix"]) == matrix for row in group_rows])
        by_matrix[matrix] = {
            name: {
                "aggregate_sse_ratio": float(
                    _combined_group_sse(experiments, name)[mask].sum()
                    / baseline[mask].sum()
                ),
                "matrix_groups_better": int(
                    np.count_nonzero(
                        _combined_group_sse(experiments, name)[mask] < baseline[mask]
                    )
                ),
                "matrix_groups_total": int(np.count_nonzero(mask)),
            }
            for name in candidates
        }

    projected = _combined_group_sse(experiments, "projected_mcg_e4m3").sum()
    learned_e4m3 = _combined_group_sse(experiments, "learned_e4m3").sum()
    tied_fp16 = _combined_group_sse(experiments, "learned_tied_fp16").sum()
    free_fp16 = _combined_group_sse(experiments, "learned_free_fp16").sum()
    gap = projected - free_fp16
    e4m3_gap_recovery = float((projected - learned_e4m3) / gap)
    tied_gap_recovery = float((projected - tied_fp16) / gap)

    payload = {
        "kind": "kquant_trellis_learned_codebook_summary",
        "schema_version": 2,
        "experiments": [str(path.resolve()) for path in args.experiment],
        "matrix_groups": int(baseline.size),
        "evaluation_coefficients": int(
            sum(
                experiment["corpus"]["evaluation_coefficients"]
                for experiment in experiments
            )
        ),
        "bootstrap": {"replicates": args.replicates, "seed": args.seed},
        "versus_procedural_mcg": versus_baseline,
        "pairwise": pairwise,
        "by_matrix": by_matrix,
        "e4m3_gap_recovery_fraction": e4m3_gap_recovery,
        "tied_fp16_gap_recovery_fraction": tied_gap_recovery,
    }
    _atomic_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
