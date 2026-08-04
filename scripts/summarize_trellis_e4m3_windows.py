#!/usr/bin/env python3
"""Compare learned E4M3 trellises across rolling-window lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


FAMILIES = (
    "procedural_mcg_fp16",
    "projected_mcg_e4m3",
    "learned_free_fp16",
    "learned_tied_fp16",
    "learned_e4m3",
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def _identity(experiment: dict[str, object]) -> tuple[tuple[int, int, str], ...]:
    rows = experiment["metrics"]["procedural_mcg_fp16"]["groups"]
    return tuple(
        (int(row["layer"]), int(row["expert"]), str(row["matrix"]))
        for row in rows
    )


def _group_sse(experiment: dict[str, object], family: str) -> np.ndarray:
    return np.asarray(
        [float(row["sse"]) for row in experiment["metrics"][family]["groups"]],
        dtype=np.float64,
    )


def _ratio_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, object]:
    samples = (
        numerator[bootstrap_indices].sum(axis=1)
        / denominator[bootstrap_indices].sum(axis=1)
    )
    return {
        "aggregate_sse_ratio": float(numerator.sum() / denominator.sum()),
        "paired_matrix_bootstrap_95": [
            float(value) for value in np.quantile(samples, (0.025, 0.975))
        ],
        "matrix_groups_better": int(np.count_nonzero(numerator < denominator)),
        "matrix_groups_total": int(numerator.size),
    }


def summarize(
    experiments: list[dict[str, object]],
    *,
    paths: list[Path],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if len(experiments) < 2:
        raise ValueError("at least two window experiments are required")
    if any(
        experiment.get("kind") != "kquant_trellis_learned_codebook_experiment"
        for experiment in experiments
    ):
        raise ValueError("an input is not a learned-codebook experiment")

    by_window: dict[int, tuple[dict[str, object], Path]] = {}
    for experiment, path in zip(experiments, paths, strict=True):
        window = int(experiment["signature"]["window_bits"])
        if window in by_window:
            raise ValueError(f"duplicate L{window} experiment")
        by_window[window] = (experiment, path)

    reference_window = max(by_window)
    reference = by_window[reference_window][0]
    reference_signature = reference["signature"]
    common_keys = (
        "source_revision",
        "layers",
        "experts",
        "matrices",
        "bits",
        "gain_tiles_per_matrix",
        "fitting_tiles_per_matrix",
        "selection_tiles_per_matrix",
        "evaluation_tiles_per_matrix",
        "ridges",
        "iterations",
        "seed",
    )
    reference_identity = _identity(reference)
    for window, (experiment, _) in by_window.items():
        if set(experiment["metrics"]) != set(FAMILIES):
            raise ValueError(f"L{window} does not contain the expected metric families")
        for key in common_keys:
            if experiment["signature"].get(key) != reference_signature.get(key):
                raise ValueError(f"L{window} differs from L{reference_window} in {key}")
        if _identity(experiment) != reference_identity:
            raise ValueError(f"L{window} evaluation groups differ from the reference")
        if int(experiment["corpus"]["evaluation_coefficients"]) != int(
            reference["corpus"]["evaluation_coefficients"]
        ):
            raise ValueError(f"L{window} evaluation coefficient count differs")

    rng = np.random.default_rng(seed)
    group_count = len(reference_identity)
    bootstrap_indices = rng.integers(
        0, group_count, size=(replicates, group_count), dtype=np.int32
    )
    reference_e4m3 = _group_sse(reference, "learned_e4m3")

    rows: dict[str, object] = {}
    for window, (experiment, path) in sorted(by_window.items()):
        procedural = _group_sse(experiment, "procedural_mcg_fp16")
        learned_e4m3 = _group_sse(experiment, "learned_e4m3")
        controls = experiment["storage_controls"]
        aliases = experiment["metrics"]["learned_e4m3"]["alias_structure"]
        right_context = aliases["right_context_distinguishability"]
        rows[str(window)] = {
            "experiment": str(path.resolve()),
            "states": int(controls.get("states", 1 << (window - 3))),
            "transitions_per_coefficient": int(
                controls.get("transitions_per_coefficient", 1 << window)
            ),
            "viterbi_work_relative_to_l16": float(
                controls.get("viterbi_work_relative_to_l16", 2.0 ** (window - 16))
            ),
            "global_e4m3_table_bytes": int(controls["global_e4m3_table_bytes"]),
            "learned_e4m3_graph": {
                "numeric_labels": int(aliases["numeric_labels"]),
                "uniform_numeric_label_entropy_bits": float(
                    aliases["uniform_numeric_label_entropy_bits"]
                ),
                "mean_distinct_outgoing_labels": float(
                    aliases["outgoing"]["mean_distinct_labels_per_state"]
                ),
                "states_with_outgoing_alias_fraction": float(
                    aliases["outgoing"]["states_with_alias_fraction"]
                ),
                "selected_edges_with_alias_available_fraction": float(
                    aliases["selected"]["edges_with_alias_available_fraction"]
                ),
                "selected_numeric_label_entropy_bits": float(
                    aliases["selected"]["numeric_label_entropy_bits"]
                ),
                "right_context_classes_at_full_memory": int(
                    right_context[-1]["right_context_classes"]
                ),
            },
            "aggregate_sse": {
                family: float(_group_sse(experiment, family).sum())
                for family in FAMILIES
            },
            "families_over_same_window_procedural": {
                family: _ratio_summary(
                    _group_sse(experiment, family), procedural, bootstrap_indices
                )
                for family in FAMILIES
                if family != "procedural_mcg_fp16"
            },
            f"learned_e4m3_over_l{reference_window}": _ratio_summary(
                learned_e4m3, reference_e4m3, bootstrap_indices
            ),
            "learned_e4m3_over_same_window_free_fp16": _ratio_summary(
                learned_e4m3,
                _group_sse(experiment, "learned_free_fp16"),
                bootstrap_indices,
            ),
        }

    best_window = min(
        by_window,
        key=lambda window: _group_sse(by_window[window][0], "learned_e4m3").sum(),
    )
    return {
        "kind": "kquant_trellis_e4m3_window_summary",
        "schema_version": 1,
        "windows": rows,
        "reference_window": reference_window,
        "best_learned_e4m3_window_by_aggregate_sse": best_window,
        "matrix_groups": group_count,
        "evaluation_coefficients": int(
            reference["corpus"]["evaluation_coefficients"]
        ),
        "bootstrap": {"replicates": replicates, "seed": seed},
        "interpretation": (
            "All windows use the same K3 path rate, source tiles, splits, and fitting "
            "budget. Window length changes finite-state sequence geometry, global table "
            "size, and exhaustive Viterbi work; it does not change stored path bpw."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    payload = summarize(
        [json.loads(path.read_text()) for path in args.experiment],
        paths=args.experiment,
        replicates=args.replicates,
        seed=args.seed,
    )
    _atomic_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
