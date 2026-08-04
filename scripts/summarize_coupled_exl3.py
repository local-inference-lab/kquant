"""Summarize document-disjoint coupled ``(r13, r2)`` expert experiments.

All mode choices are made from routed training data.  The validation fold is
used only for paired document-level evaluation.  This handles both diagonal
shared-rate studies and complete separate-rate grids.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from scripts.summarize_retained_w2_rate_transfer import (
    RecordKey,
    _bootstrap_improvement,
    _quantile,
    _read_json,
)
from scripts.summarize_retained_w2_rate_ladder import _parse_alphabets


Mode = tuple[int, int]


def _candidate(result: dict, mode: Mode) -> dict:
    matches = [
        candidate
        for candidate in result["results"]
        if (int(candidate["r13"]), int(candidate["r2"])) == mode
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one R{mode[0]}/R{mode[1]} candidate")
    return matches[0]


def _metric(candidate: dict, split: str) -> dict | None:
    metric = candidate["coupled_activation"].get(split)
    if metric is not None and metric["route_weighted_reference_energy"] <= 0:
        raise ValueError(f"{split} metric has non-positive reference energy")
    return metric


def _document_values(metric: dict | None, key: str) -> dict[str, float]:
    if metric is None:
        return {}
    values: dict[str, float] = {}
    for contribution in metric["request_contributions"]:
        document = contribution.get("document_hash")
        if not document:
            raise ValueError("coupled request contribution lacks document hash")
        values[document] = values.get(document, 0.0) + float(contribution[key])
    expected_key = {
        "route_weighted_squared_error": "route_weighted_squared_error",
        "route_weighted_reference_energy": "route_weighted_reference_energy",
        "teacher_mixture_energy": "teacher_mixture_energy",
    }[key]
    if not math.isclose(
        sum(values.values()),
        float(metric[expected_key]),
        rel_tol=2e-6,
        abs_tol=1e-12,
    ):
        raise ValueError(f"document contributions do not close for {key}")
    return values


def _argmin_map(
    records: dict[RecordKey, dict], allowed_modes: Sequence[Mode]
) -> dict[RecordKey, Mode]:
    selected = {}
    for key, result in records.items():
        scores = []
        for mode in allowed_modes:
            metric = _metric(_candidate(result, mode), "train")
            if metric is not None:
                scores.append((float(metric["route_weighted_squared_error"]), mode))
        selected[key] = min(scores)[1] if scores else (0, 0)
    return selected


def _aggregate(
    records: dict[RecordKey, dict],
    documents: list[str],
    selector: Callable[[RecordKey], Mode],
    *,
    replicates: int,
    seed: int,
) -> dict:
    baseline_sse = {document: 0.0 for document in documents}
    candidate_sse = {document: 0.0 for document in documents}
    expert_reference = {document: 0.0 for document in documents}
    mixture_reference = {document: 0.0 for document in documents}
    selected = Counter()
    supported = 0
    for key, result in records.items():
        mode = selector(key)
        selected[mode] += 1
        baseline = _metric(_candidate(result, (0, 0)), "validation")
        candidate = _metric(_candidate(result, mode), "validation")
        if baseline is None or candidate is None:
            continue
        supported += 1
        baseline_error = _document_values(
            baseline, "route_weighted_squared_error"
        )
        candidate_error = _document_values(
            candidate, "route_weighted_squared_error"
        )
        baseline_expert_reference = _document_values(
            baseline, "route_weighted_reference_energy"
        )
        candidate_expert_reference = _document_values(
            candidate, "route_weighted_reference_energy"
        )
        baseline_mixture_reference = _document_values(
            baseline, "teacher_mixture_energy"
        )
        candidate_mixture_reference = _document_values(
            candidate, "teacher_mixture_energy"
        )
        for document in documents:
            if not math.isclose(
                baseline_expert_reference.get(document, 0.0),
                candidate_expert_reference.get(document, 0.0),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ) or not math.isclose(
                baseline_mixture_reference.get(document, 0.0),
                candidate_mixture_reference.get(document, 0.0),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(f"candidate reference energies differ for {key}")
            baseline_sse[document] += baseline_error.get(document, 0.0)
            candidate_sse[document] += candidate_error.get(document, 0.0)
            expert_reference[document] += baseline_expert_reference.get(document, 0.0)
            mixture_reference[document] += baseline_mixture_reference.get(document, 0.0)
    estimate = _bootstrap_improvement(
        baseline_sse,
        candidate_sse,
        documents,
        replicates=replicates,
        seed=seed,
    )
    expert_reference_total = sum(expert_reference.values())
    mixture_reference_total = sum(mixture_reference.values())
    estimate.update(
        {
            "experts": len(records),
            "experts_with_validation_rows": supported,
            "selected_mode_histogram": {
                f"R{r13}/R{r2}": count
                for (r13, r2), count in sorted(selected.items())
            },
            "route_weighted_expert_reference_energy": expert_reference_total,
            "teacher_mixture_reference_energy": mixture_reference_total,
            "baseline_route_weighted_expert_nmse": (
                estimate["baseline_weighted_squared_error"] / expert_reference_total
                if expert_reference_total > 0
                else None
            ),
            "candidate_route_weighted_expert_nmse": (
                estimate["candidate_weighted_squared_error"] / expert_reference_total
                if expert_reference_total > 0
                else None
            ),
            "baseline_teacher_mixture_replacement_nmse": (
                estimate["baseline_weighted_squared_error"] / mixture_reference_total
                if mixture_reference_total > 0
                else None
            ),
            "candidate_teacher_mixture_replacement_nmse": (
                estimate["candidate_weighted_squared_error"] / mixture_reference_total
                if mixture_reference_total > 0
                else None
            ),
        }
    )
    return estimate


def _compare_strategies(
    records: dict[RecordKey, dict],
    documents: list[str],
    baseline_selector: Callable[[RecordKey], Mode],
    candidate_selector: Callable[[RecordKey], Mode],
    *,
    replicates: int,
    seed: int,
) -> dict:
    baseline_sse = {document: 0.0 for document in documents}
    candidate_sse = {document: 0.0 for document in documents}
    supported = 0
    for key, result in records.items():
        baseline = _metric(_candidate(result, baseline_selector(key)), "validation")
        candidate = _metric(_candidate(result, candidate_selector(key)), "validation")
        if baseline is None or candidate is None:
            continue
        supported += 1
        baseline_values = _document_values(
            baseline, "route_weighted_squared_error"
        )
        candidate_values = _document_values(
            candidate, "route_weighted_squared_error"
        )
        for document in documents:
            baseline_sse[document] += baseline_values.get(document, 0.0)
            candidate_sse[document] += candidate_values.get(document, 0.0)
    estimate = _bootstrap_improvement(
        baseline_sse,
        candidate_sse,
        documents,
        replicates=replicates,
        seed=seed,
    )
    estimate.update(
        {
            "experts": len(records),
            "experts_with_validation_rows": supported,
            "baseline": "training argmin restricted to shared r",
            "candidate": "training argmin over all available (r13,r2)",
        }
    )
    return estimate


def _support(records: dict[RecordKey, dict]) -> dict:
    train_rows = []
    train_documents = []
    validation_rows = []
    validation_documents = []
    for result in records.values():
        train = _metric(_candidate(result, (0, 0)), "train")
        validation = _metric(_candidate(result, (0, 0)), "validation")
        if train is not None:
            train_rows.append(int(train["rows"]))
            train_documents.append(int(train["documents"]))
        if validation is not None:
            validation_rows.append(int(validation["rows"]))
            validation_documents.append(int(validation["documents"]))

    def distribution(values: Sequence[float]) -> dict:
        return {
            "count": len(values),
            "minimum": min(values) if values else None,
            "median": _quantile(values, 0.5),
            "p90": _quantile(values, 0.9),
            "maximum": max(values) if values else None,
        }

    return {
        "training_rows_per_expert": distribution(train_rows),
        "training_documents_per_expert": distribution(train_documents),
        "validation_rows_per_expert": distribution(validation_rows),
        "validation_documents_per_expert": distribution(validation_documents),
    }


def _validate_and_collect(
    paths: Sequence[Path],
) -> tuple[dict[RecordKey, dict], list[Mode], Path, list[int], int]:
    records = {}
    available_modes: tuple[Mode, ...] | None = None
    validation_report: Path | None = None
    common_contract = None
    layers = set()
    closures = 0
    for path in paths:
        payload = _read_json(path)
        if payload.get("kind") != "kquant_coupled_rate_transfer_exl3_experiment":
            raise ValueError(f"{path} has the wrong experiment kind")
        if payload.get("schema_version") != 2:
            raise ValueError(f"{path} has the wrong schema")
        source = payload["source"]
        if source.get("tp_size") != 12:
            raise ValueError(f"{path} is not a TP12 experiment")
        provenance = source.get("provenance")
        if provenance is None:
            raise ValueError(f"{path} lacks document-disjoint provenance")
        candidate_validation_report = Path(provenance["validation_report"])
        contract = (
            source["checkpoint"],
            source["capture"],
            source["validation_capture"],
            source["hessians"],
            provenance["training_report"],
            provenance["validation_report"],
        )
        if common_contract not in (None, contract):
            raise ValueError("input experiments use different source contracts")
        common_contract = contract
        validation_report = candidate_validation_report
        layer = int(source["layer"])
        expert = int(source["expert"])
        key = (layer, expert)
        if key in records:
            raise ValueError(f"duplicate layer/expert result {key}")
        modes = tuple(
            sorted((int(result["r13"]), int(result["r2"])) for result in payload["results"])
        )
        if (0, 0) not in modes:
            raise ValueError(f"{path} lacks the R0/R0 control")
        if available_modes not in (None, modes):
            raise ValueError("input experiments expose different candidate grids")
        available_modes = modes
        byte_contract = None
        for result in payload["results"]:
            closure = result["physical_permutation_closure"]
            if closure["relative_l2"] > 5e-6 or closure["max_abs"] > 5e-4:
                raise ValueError(f"failed physical permutation closure for {key}")
            for matrix in result["matrix_results"]:
                if not matrix["reference_edge_roundtrip"] or not matrix[
                    "reference_state_roundtrip"
                ]:
                    raise ValueError(f"failed trellis closure for {key}")
                closures += 1
            candidate_byte_contract = (
                int(result["total_trellis_bytes"]),
                int(result["total_auxiliary_bytes"]),
                int(result["provisional_expert_mode_bits"]),
            )
            if byte_contract not in (None, candidate_byte_contract):
                raise ValueError(f"coupled modes do not have equal bytes for {key}")
            byte_contract = candidate_byte_contract
        records[key] = payload
        layers.add(layer)
    if available_modes is None or validation_report is None:
        raise ValueError("no coupled inputs")
    return records, list(available_modes), validation_report, sorted(layers), closures


def run(args: argparse.Namespace) -> dict:
    records, modes, validation_report, layers, closures = _validate_and_collect(
        args.input
    )
    validation_documents = [
        str(document["document_hash"])
        for document in _read_json(validation_report)["documents"]
    ]
    shared_modes = [mode for mode in modes if mode[0] == mode[1]]
    all_choices = _argmin_map(records, modes)
    shared_choices = _argmin_map(records, shared_modes)

    def summarize(selected_records: dict[RecordKey, dict], seed_offset: int) -> dict:
        fixed_shared = {
            f"R{r}": _aggregate(
                selected_records,
                validation_documents,
                lambda _key, mode=(r, r): mode,
                replicates=args.bootstrap_replicates,
                seed=args.seed + seed_offset + r,
            )
            for r, _ in shared_modes
        }
        alphabets = {}
        for index, alphabet in enumerate(args.mode_alphabets):
            allowed = set(alphabet)
            alphabet_modes = [
                mode for mode in modes if mode[0] in allowed and mode[1] in allowed
            ]
            alphabet_shared = [
                mode for mode in alphabet_modes if mode[0] == mode[1]
            ]
            if not alphabet_modes or not alphabet_shared:
                raise ValueError(f"mode alphabet {alphabet} has no available R0 control")
            choices = _argmin_map(selected_records, alphabet_modes)
            shared_alphabet_choices = _argmin_map(
                selected_records, alphabet_shared
            )
            alphabets["_".join(f"r{r}" for r in alphabet)] = {
                "allowed_r": list(alphabet),
                "training_argmin_shared_r": _aggregate(
                    selected_records,
                    validation_documents,
                    lambda key, choices=shared_alphabet_choices: choices[key],
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + seed_offset + 200 + index,
                ),
                "training_argmin_available_modes": _aggregate(
                    selected_records,
                    validation_documents,
                    lambda key, choices=choices: choices[key],
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + seed_offset + 220 + index,
                ),
            }
        return {
            "fixed_shared_r": fixed_shared,
            "mode_alphabets": alphabets,
            "training_argmin_shared_r": _aggregate(
                selected_records,
                validation_documents,
                lambda key: shared_choices[key],
                replicates=args.bootstrap_replicates,
                seed=args.seed + seed_offset + 100,
            ),
            "training_argmin_all_available_modes": _aggregate(
                selected_records,
                validation_documents,
                lambda key: all_choices[key],
                replicates=args.bootstrap_replicates,
                seed=args.seed + seed_offset + 101,
            ),
            "separate_vs_shared_training_argmin": _compare_strategies(
                selected_records,
                validation_documents,
                lambda key: shared_choices[key],
                lambda key: all_choices[key],
                replicates=args.bootstrap_replicates,
                seed=args.seed + seed_offset + 102,
            ),
        }

    per_layer = {}
    for offset, layer in enumerate(layers, start=1):
        layer_records = {
            key: result for key, result in records.items() if key[0] == layer
        }
        per_layer[str(layer)] = {
            "support": _support(layer_records),
            "strategies": summarize(layer_records, 1000 * offset),
        }
    result = {
        "kind": "kquant_coupled_rate_transfer_summary",
        "schema_version": 1,
        "inputs": [str(path.resolve()) for path in args.input],
        "layers": layers,
        "experts": len(records),
        "available_modes": [list(mode) for mode in modes],
        "shared_only": len(shared_modes) == len(modes),
        "matrix_candidate_closures_checked": closures,
        "equal_byte_contract_checked_per_expert": True,
        "validation_documents": len(validation_documents),
        "selection_contract": "training-only argmin; validation used only for evaluation",
        "support": _support(records),
        "strategies": summarize(records, 10_000),
        "per_layer": per_layer,
        "training_argmin_shared_mode_map": {
            f"{layer}/{expert}": list(mode)
            for (layer, expert), mode in sorted(shared_choices.items())
        },
        "training_argmin_all_mode_map": {
            f"{layer}/{expert}": list(mode)
            for (layer, expert), mode in sorted(all_choices.items())
        },
        "limitations": [
            "the selected retained experts are a small hypothesis-forming subset",
            "isolated expert self-terms omit cross-expert mixture error terms",
            "w2 uses teacher H2 rather than candidate-hhat covariance",
            "the interim teacher is a calibration proxy for later offline source-weight encoding",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True))
    temporary.replace(args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode-alphabets",
        type=_parse_alphabets,
        default=_parse_alphabets(
            "0,2;0,1,3;0,1,2,3,5;0,1,2,3,4,5,6,8"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "experts": result["experts"],
                "shared_only": result["shared_only"],
                "fixed_shared_r": {
                    mode: {
                        "relative_improvement": value["relative_improvement"],
                        "relative_improvement_ci95": value[
                            "relative_improvement_ci95"
                        ],
                    }
                    for mode, value in result["strategies"][
                        "fixed_shared_r"
                    ].items()
                },
                "training_argmin_shared_r": result["strategies"][
                    "training_argmin_shared_r"
                ],
                "training_argmin_all_available_modes": result["strategies"][
                    "training_argmin_all_available_modes"
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
