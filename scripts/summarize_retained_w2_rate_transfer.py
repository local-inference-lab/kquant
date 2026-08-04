"""Summarize document-disjoint retained-expert R0/R6 experiments.

The primary quantities are sums of gate-square-weighted routed-output squared
error, not geometric means of per-expert ratios.  Mode choices are made from
the training corpus only and evaluated on the separate document fold.  A
paired document bootstrap preserves correlation among experts and layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable
from pathlib import Path


RecordKey = tuple[int, int]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _candidate(result: dict, r: int) -> dict:
    matches = [candidate for candidate in result["candidates"] if candidate["r"] == r]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one R{r} candidate")
    return matches[0]


def _metric(candidate: dict, split: str) -> dict | None:
    value = candidate["captured_activation"].get(split)
    if value is not None and value.get("weighted_reference_energy", 0) <= 0:
        raise ValueError(f"{split} metric has non-positive reference energy")
    return value


def _document_sse(metric: dict | None) -> dict[str, float]:
    if metric is None:
        return {}
    result: dict[str, float] = {}
    for contribution in metric["request_contributions"]:
        document = contribution.get("document_hash")
        if not document:
            raise ValueError("request contribution lacks a document hash")
        result[document] = result.get(document, 0.0) + float(
            contribution["weighted_squared_error"]
        )
    if not math.isclose(
        sum(result.values()),
        float(metric["weighted_squared_error"]),
        rel_tol=2e-6,
        abs_tol=1e-12,
    ):
        raise ValueError("request contributions do not close to metric SSE")
    return result


def _document_reference(metric: dict | None) -> dict[str, float]:
    if metric is None:
        return {}
    result: dict[str, float] = {}
    for contribution in metric["request_contributions"]:
        document = contribution["document_hash"]
        result[document] = result.get(document, 0.0) + float(
            contribution["weighted_reference_energy"]
        )
    return result


def _bootstrap_improvement(
    baseline_by_document: dict[str, float],
    candidate_by_document: dict[str, float],
    documents: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict:
    baseline = [baseline_by_document.get(document, 0.0) for document in documents]
    candidate = [candidate_by_document.get(document, 0.0) for document in documents]
    baseline_total = sum(baseline)
    candidate_total = sum(candidate)
    point_ratio = candidate_total / baseline_total if baseline_total > 0 else None
    rng = random.Random(seed)
    improvements = []
    for _ in range(replicates):
        indices = [rng.randrange(len(documents)) for _ in documents]
        sampled_baseline = sum(baseline[index] for index in indices)
        if sampled_baseline <= 0:
            continue
        sampled_candidate = sum(candidate[index] for index in indices)
        improvements.append(1.0 - sampled_candidate / sampled_baseline)
    return {
        "baseline_weighted_squared_error": baseline_total,
        "candidate_weighted_squared_error": candidate_total,
        "candidate_to_baseline_sse_ratio": point_ratio,
        "relative_improvement": None if point_ratio is None else 1.0 - point_ratio,
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": len(improvements),
        "relative_improvement_ci95": [
            _quantile(improvements, 0.025),
            _quantile(improvements, 0.975),
        ],
    }


def _seed_for(base: int, layer: int, expert: int) -> int:
    digest = hashlib.blake2b(
        f"{base}:{layer}:{expert}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little")


def _mode_maps(
    records: dict[RecordKey, dict],
    training_documents: list[str],
    *,
    replicates: int,
    seed: int,
    min_training_documents: int,
    minimum_improvement: float,
) -> tuple[dict[RecordKey, int], dict[RecordKey, int], dict]:
    naive: dict[RecordKey, int] = {}
    conservative: dict[RecordKey, int] = {}
    confidence: dict[str, dict] = {}
    for (layer, expert), result in records.items():
        r0 = _metric(_candidate(result, 0), "train")
        r6 = _metric(_candidate(result, 6), "train")
        if r0 is None or r6 is None:
            naive[layer, expert] = 0
            conservative[layer, expert] = 0
            continue
        naive[layer, expert] = (
            6
            if r6["weighted_squared_error"] < r0["weighted_squared_error"]
            else 0
        )
        estimate = _bootstrap_improvement(
            _document_sse(r0),
            _document_sse(r6),
            training_documents,
            replicates=replicates,
            seed=_seed_for(seed, layer, expert),
        )
        lower = estimate["relative_improvement_ci95"][0]
        supported_documents = int(r0["documents"])
        choose_r6 = (
            supported_documents >= min_training_documents
            and lower is not None
            and lower > minimum_improvement
        )
        conservative[layer, expert] = 6 if choose_r6 else 0
        confidence[f"{layer}/{expert}"] = {
            "training_documents": supported_documents,
            "selected_r": conservative[layer, expert],
            **estimate,
        }
    return naive, conservative, confidence


def _aggregate_strategy(
    records: dict[RecordKey, dict],
    documents: list[str],
    selector: Callable[[RecordKey], int],
    *,
    replicates: int,
    seed: int,
) -> dict:
    baseline_sse = {document: 0.0 for document in documents}
    candidate_sse = {document: 0.0 for document in documents}
    reference = {document: 0.0 for document in documents}
    supported_experts = 0
    selected_r6 = 0
    for key, result in records.items():
        r0_metric = _metric(_candidate(result, 0), "validation")
        selected_r = selector(key)
        selected_metric = _metric(_candidate(result, selected_r), "validation")
        if selected_r == 6:
            selected_r6 += 1
        if r0_metric is None or selected_metric is None:
            continue
        supported_experts += 1
        r0_sse = _document_sse(r0_metric)
        chosen_sse = _document_sse(selected_metric)
        r0_reference = _document_reference(r0_metric)
        chosen_reference = _document_reference(selected_metric)
        for document in documents:
            if not math.isclose(
                r0_reference.get(document, 0.0),
                chosen_reference.get(document, 0.0),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError("candidate reference energies differ")
            baseline_sse[document] += r0_sse.get(document, 0.0)
            candidate_sse[document] += chosen_sse.get(document, 0.0)
            reference[document] += r0_reference.get(document, 0.0)
    estimate = _bootstrap_improvement(
        baseline_sse,
        candidate_sse,
        documents,
        replicates=replicates,
        seed=seed,
    )
    reference_total = sum(reference.values())
    estimate.update(
        {
            "experts": len(records),
            "experts_with_validation_rows": supported_experts,
            "experts_selected_r6": selected_r6,
            "weighted_reference_energy": reference_total,
            "baseline_weighted_output_nmse": (
                estimate["baseline_weighted_squared_error"] / reference_total
                if reference_total > 0
                else None
            ),
            "candidate_weighted_output_nmse": (
                estimate["candidate_weighted_squared_error"] / reference_total
                if reference_total > 0
                else None
            ),
        }
    )
    return estimate


def _support_and_regressions(records: dict[RecordKey, dict]) -> dict:
    train_rows = []
    train_documents = []
    validation_rows = []
    validation_documents = []
    ratios = []
    high_traffic = []
    for (layer, expert), result in records.items():
        r0 = _metric(_candidate(result, 0), "validation")
        r6 = _metric(_candidate(result, 6), "validation")
        train = _metric(_candidate(result, 0), "train")
        if train is not None:
            train_rows.append(int(train["rows"]))
            train_documents.append(int(train["documents"]))
        if r0 is None or r6 is None:
            continue
        validation_rows.append(int(r0["rows"]))
        validation_documents.append(int(r0["documents"]))
        baseline = float(r0["weighted_squared_error"])
        ratio = float(r6["weighted_squared_error"]) / baseline if baseline > 0 else math.inf
        ratios.append(ratio)
        high_traffic.append(
            (
                float(r0["weighted_reference_energy"]),
                ratio,
                layer,
                expert,
            )
        )

    def distribution(values: list[float]) -> dict:
        return {
            "count": len(values),
            "minimum": min(values) if values else None,
            "median": _quantile(values, 0.5),
            "p90": _quantile(values, 0.9),
            "p99": _quantile(values, 0.99),
            "maximum": max(values) if values else None,
        }

    high_traffic.sort(reverse=True)
    high_traffic = high_traffic[: max(1, math.ceil(len(high_traffic) * 0.1))]
    worst_high_traffic = max(high_traffic, key=lambda value: value[1]) if high_traffic else None
    return {
        "training_rows_per_expert": distribution(train_rows),
        "training_documents_per_expert": distribution(train_documents),
        "validation_rows_per_supported_expert": distribution(validation_rows),
        "validation_documents_per_supported_expert": distribution(
            validation_documents
        ),
        "r6_to_r0_validation_sse_ratio_per_supported_expert": distribution(ratios),
        "r6_better_experts": sum(ratio < 1 for ratio in ratios),
        "r6_worse_experts": sum(ratio > 1 for ratio in ratios),
        "worst_ratio_in_top_reference_energy_decile": (
            None
            if worst_high_traffic is None
            else {
                "ratio": worst_high_traffic[1],
                "layer": worst_high_traffic[2],
                "expert": worst_high_traffic[3],
                "weighted_reference_energy": worst_high_traffic[0],
            }
        ),
    }


def run(args: argparse.Namespace) -> dict:
    inputs = [_read_json(path) for path in args.input]
    records: dict[RecordKey, dict] = {}
    training_report_path: Path | None = None
    validation_report_path: Path | None = None
    layers = []
    closures = 0
    for path, payload in zip(args.input, inputs, strict=True):
        if payload.get("kind") != "kquant_retained_w2_rate_transfer_study":
            raise ValueError(f"{path} has the wrong study kind")
        if payload.get("schema_version") != 2 or not payload.get("complete"):
            raise ValueError(f"{path} is not a complete schema-v2 study")
        signature = payload["signature"]
        if signature.get("metrics_schema") != 2 or signature.get("r_values") != [0, 6]:
            raise ValueError(f"{path} does not contain the R0/R6 metric contract")
        layer = int(signature["layer"])
        layers.append(layer)
        provenance = signature["provenance"]
        candidate_training_report = Path(provenance["training_report"])
        candidate_validation_report = Path(provenance["validation_report"])
        if training_report_path not in (None, candidate_training_report):
            raise ValueError("input studies use different training reports")
        if validation_report_path not in (None, candidate_validation_report):
            raise ValueError("input studies use different validation reports")
        training_report_path = candidate_training_report
        validation_report_path = candidate_validation_report
        for expert_text, result in payload["results"].items():
            key = (layer, int(expert_text))
            if key in records:
                raise ValueError(f"duplicate layer/expert result {key}")
            for r in (0, 6):
                candidate = _candidate(result, r)
                if not candidate["reference_edge_roundtrip"] or not candidate[
                    "reference_state_roundtrip"
                ]:
                    raise ValueError(f"failed reference closure for {key} R{r}")
                closures += 1
            records[key] = result
    assert training_report_path is not None and validation_report_path is not None
    training_documents = [
        document["document_hash"]
        for document in _read_json(training_report_path)["documents"]
    ]
    validation_documents = [
        document["document_hash"]
        for document in _read_json(validation_report_path)["documents"]
    ]
    naive, conservative, confidence = _mode_maps(
        records,
        training_documents,
        replicates=args.selection_bootstrap_replicates,
        seed=args.seed,
        min_training_documents=args.min_training_documents,
        minimum_improvement=args.minimum_training_improvement,
    )

    def summaries(selected_records: dict[RecordKey, dict], seed_offset: int) -> dict:
        return {
            "uniform_r0": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda _key: 0,
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset,
            ),
            "all_r6": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda _key: 6,
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + 1,
            ),
            "training_argmin": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda key: naive[key],
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + 2,
            ),
            "training_confidence_selected": _aggregate_strategy(
                selected_records,
                validation_documents,
                lambda key: conservative[key],
                replicates=args.evaluation_bootstrap_replicates,
                seed=args.seed + seed_offset + 3,
            ),
        }

    layer_summaries = {}
    for offset, layer in enumerate(sorted(set(layers))):
        layer_records = {
            key: value for key, value in records.items() if key[0] == layer
        }
        layer_summaries[str(layer)] = {
            "support_and_regressions": _support_and_regressions(layer_records),
            "strategies": summaries(layer_records, 100 * (offset + 1)),
        }
    result = {
        "kind": "kquant_retained_w2_rate_transfer_summary",
        "schema_version": 1,
        "inputs": [str(path.resolve()) for path in args.input],
        "layers": sorted(set(layers)),
        "experts": len(records),
        "candidate_closures_checked": closures,
        "training_documents": len(training_documents),
        "validation_documents": len(validation_documents),
        "selection_rule": {
            "mode": "R6 only when the training-document bootstrap lower bound clears the margin; otherwise R0",
            "min_training_documents": args.min_training_documents,
            "minimum_training_improvement": args.minimum_training_improvement,
            "bootstrap_replicates": args.selection_bootstrap_replicates,
        },
        "limitations": [
            "retained experts are a keep-tier-biased subset",
            "mid-route sampling leaves sparse per-expert validation support",
            "metrics cover isolated gate-square-weighted expert self-terms, not routed-mixture cross terms",
            "this intermediate capture is not the planned 10M-token production calibration",
        ],
        "support_and_regressions": _support_and_regressions(records),
        "strategies": summaries(records, 10_000),
        "per_layer": layer_summaries,
        "conservative_training_confidence": confidence,
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
    parser.add_argument("--min-training-documents", type=int, default=8)
    parser.add_argument("--minimum-training-improvement", type=float, default=0.0)
    parser.add_argument("--selection-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--evaluation-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.min_training_documents < 1:
        parser.error("--min-training-documents must be positive")
    if args.selection_bootstrap_replicates < 100:
        parser.error("--selection-bootstrap-replicates must be at least 100")
    if args.evaluation_bootstrap_replicates < 100:
        parser.error("--evaluation-bootstrap-replicates must be at least 100")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "experts": result["experts"],
                "all_r6": result["strategies"]["all_r6"],
                "training_confidence_selected": result["strategies"][
                    "training_confidence_selected"
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
