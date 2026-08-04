"""Summarize held-out effects of TP12 ``w2`` record-map proposals.

For every expert, both the monotone mode and the arbitrary-map candidate are
selected using complete dense-H LDLQ training error only.  Their static
choices are then evaluated on untouched documents.  The selected arbitrary
map's matched-order R0 encode provides a second baseline, decomposing the
observed change into permutation/traversal and heterogeneous-rate effects.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.summarize_retained_w2_rate_transfer import (
    _bootstrap_improvement,
    _document_reference,
    _document_sse,
    _quantile,
    _read_json,
)


RecordKey = tuple[int, int]


def _metric(candidate: dict, split: str) -> dict:
    metric = candidate["captured_activation"].get(split)
    if metric is None or float(metric.get("weighted_reference_energy", 0.0)) <= 0:
        raise ValueError(f"candidate lacks a valid {split} metric")
    return metric


def _select_candidates(payload: dict) -> dict:
    """Select candidate identities without reading validation metrics."""

    canonical = payload["canonical_r0"]
    monotone_options = [canonical, *payload["monotone"]]
    best_monotone = min(
        monotone_options,
        key=lambda candidate: (
            float(_metric(candidate, "train")["weighted_squared_error"]),
            int(candidate["r"]),
        ),
    )

    map_options: list[tuple[dict, dict, dict | None, str]] = [
        (canonical, canonical, None, "canonical")
    ]
    map_options.extend(
        (candidate, canonical, None, "monotone")
        for candidate in payload["monotone"]
    )
    map_options.extend(
        (entry["mixed"], entry["matched_order_r0"], entry, "arbitrary")
        for entry in payload["maps"]
    )
    best_mixed, matched_r0, map_entry, map_kind = min(
        map_options,
        key=lambda option: (
            float(_metric(option[0], "train")["weighted_squared_error"]),
            int(option[0]["r"]),
            0 if option[2] is None else int(option[2]["proposal_rank"]),
        ),
    )
    return {
        "canonical": canonical,
        "monotone": best_monotone,
        "map_mixed": best_mixed,
        "map_matched_r0": matched_r0,
        "map_entry": map_entry,
        "map_kind": map_kind,
    }


def _candidate_identity(candidate: dict, map_entry: dict | None = None) -> dict:
    result = {"r": int(candidate["r"]), "mode": str(candidate["mode"])}
    if map_entry is not None:
        result.update(
            {
                "proposal_rank": int(map_entry["proposal_rank"]),
                "additive_training_delta": float(
                    map_entry["additive_training_delta"]
                ),
                "donor_records_low_to_high": map_entry[
                    "donor_records_low_to_high"
                ],
                "recipient_records_pair_order_high_to_low": map_entry[
                    "recipient_records_pair_order_high_to_low"
                ],
                "logical_record_order_low_to_high": map_entry[
                    "logical_record_order_low_to_high"
                ],
            }
        )
    return result


def _aggregate_pair(
    records: Mapping[RecordKey, dict],
    selected: Mapping[RecordKey, dict],
    documents: Sequence[str],
    *,
    baseline_name: str,
    candidate_name: str,
    replicates: int,
    seed: int,
) -> dict:
    baseline_by_document = {document: 0.0 for document in documents}
    candidate_by_document = {document: 0.0 for document in documents}
    reference_by_document = {document: 0.0 for document in documents}
    per_expert_ratios = []
    for key in records:
        baseline_metric = _metric(selected[key][baseline_name], "validation")
        candidate_metric = _metric(selected[key][candidate_name], "validation")
        baseline_sse = _document_sse(baseline_metric)
        candidate_sse = _document_sse(candidate_metric)
        baseline_reference = _document_reference(baseline_metric)
        candidate_reference = _document_reference(candidate_metric)
        baseline_total = float(baseline_metric["weighted_squared_error"])
        candidate_total = float(candidate_metric["weighted_squared_error"])
        if baseline_total > 0:
            per_expert_ratios.append(candidate_total / baseline_total)
        for document in documents:
            if not math.isclose(
                baseline_reference.get(document, 0.0),
                candidate_reference.get(document, 0.0),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"reference energy differs for layer/expert {key} on {document}"
                )
            baseline_by_document[document] += baseline_sse.get(document, 0.0)
            candidate_by_document[document] += candidate_sse.get(document, 0.0)
            reference_by_document[document] += baseline_reference.get(document, 0.0)
    result = _bootstrap_improvement(
        baseline_by_document,
        candidate_by_document,
        list(documents),
        replicates=replicates,
        seed=seed,
    )
    reference_total = sum(reference_by_document.values())
    result.update(
        {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "experts": len(records),
            "documents": len(documents),
            "weighted_reference_energy": reference_total,
            "baseline_weighted_output_nmse": (
                result["baseline_weighted_squared_error"] / reference_total
                if reference_total > 0
                else None
            ),
            "candidate_weighted_output_nmse": (
                result["candidate_weighted_squared_error"] / reference_total
                if reference_total > 0
                else None
            ),
            "per_expert_candidate_to_baseline_ratio": {
                "minimum": min(per_expert_ratios) if per_expert_ratios else None,
                "median": _quantile(per_expert_ratios, 0.5),
                "p90": _quantile(per_expert_ratios, 0.9),
                "maximum": max(per_expert_ratios) if per_expert_ratios else None,
                "better": sum(ratio < 1.0 for ratio in per_expert_ratios),
                "worse": sum(ratio > 1.0 for ratio in per_expert_ratios),
            },
        }
    )
    return result


def _validate_coding(candidate: dict, byte_contract: tuple[int, int]) -> int:
    coding = candidate["coding"]
    if (
        int(coding["index_bits"]),
        int(coding["scale_bits"]),
    ) != byte_contract:
        raise ValueError("candidate violates the exact byte contract")
    if not coding["reference_edge_roundtrip"] or not coding[
        "reference_state_roundtrip"
    ]:
        raise ValueError("candidate failed reference payload closure")
    return 1


def _validate_and_collect(
    paths: Sequence[Path],
) -> tuple[dict[RecordKey, dict], Path, dict, int]:
    records: dict[RecordKey, dict] = {}
    validation_report: Path | None = None
    common_contract = None
    study_contract = None
    closures = 0
    for path in paths:
        payload = _read_json(path)
        if payload.get("kind") != "kquant_w2_record_map_search":
            raise ValueError(f"{path} has the wrong study kind")
        if payload.get("schema_version") != 1 or not payload.get("complete"):
            raise ValueError(f"{path} is not a complete schema-v1 study")
        signature = payload["signature"]
        if signature.get("tp_size") != 12:
            raise ValueError(f"{path} is not a TP12 study")
        provenance = signature["provenance"]
        candidate_validation_report = Path(provenance["validation_report"])
        contract = (
            signature["checkpoint"],
            signature["source_checkpoint"],
            signature["source_revision"],
            signature["capture"],
            signature["validation_capture"],
            signature["hessians"],
            tuple(signature["r_values"]),
            int(signature["proposals_per_r"]),
            Path(provenance["training_report"]),
            candidate_validation_report,
        )
        if common_contract not in (None, contract):
            raise ValueError("input studies use different encoder or corpus contracts")
        common_contract = contract
        validation_report = candidate_validation_report
        study_contract = {
            "resident_teacher_checkpoint": signature["checkpoint"],
            "source_checkpoint": signature["source_checkpoint"],
            "source_revision": signature["source_revision"],
            "capture": signature["capture"],
            "validation_capture": signature["validation_capture"],
            "hessians": signature["hessians"],
            "r_values": signature["r_values"],
            "proposals_per_r": signature["proposals_per_r"],
            "tp_size": 12,
        }
        key = (int(signature["layer"]), int(signature["expert"]))
        if key in records:
            raise ValueError(f"duplicate layer/expert result {key}")
        byte_payload = payload["exact_byte_contract"]
        if not byte_payload.get("all_candidates_equal"):
            raise ValueError(f"{path} does not assert equal candidate bytes")
        byte_contract = (
            int(byte_payload["index_bits"]),
            int(byte_payload["scale_bits"]),
        )
        candidates = [payload["canonical_r0"], *payload["monotone"]]
        for entry in payload["maps"]:
            candidates.extend((entry["matched_order_r0"], entry["mixed"]))
        closures += sum(_validate_coding(candidate, byte_contract) for candidate in candidates)
        records[key] = payload
    if validation_report is None or study_contract is None:
        raise ValueError("no study inputs")
    return records, validation_report, study_contract, closures


def _summarize_subset(
    records: Mapping[RecordKey, dict],
    selected: Mapping[RecordKey, dict],
    documents: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict:
    return {
        "monotone_training_argmin_vs_canonical_r0": _aggregate_pair(
            records,
            selected,
            documents,
            baseline_name="canonical",
            candidate_name="monotone",
            replicates=replicates,
            seed=seed,
        ),
        "record_map_training_argmin_vs_canonical_r0": _aggregate_pair(
            records,
            selected,
            documents,
            baseline_name="canonical",
            candidate_name="map_mixed",
            replicates=replicates,
            seed=seed + 1,
        ),
        "selected_map_order_effect_r0_vs_canonical_r0": _aggregate_pair(
            records,
            selected,
            documents,
            baseline_name="canonical",
            candidate_name="map_matched_r0",
            replicates=replicates,
            seed=seed + 2,
        ),
        "selected_map_rate_effect_mixed_vs_matched_r0": _aggregate_pair(
            records,
            selected,
            documents,
            baseline_name="map_matched_r0",
            candidate_name="map_mixed",
            replicates=replicates,
            seed=seed + 3,
        ),
    }


def run(args: argparse.Namespace) -> dict:
    records, validation_report, study_contract, closures = _validate_and_collect(
        args.input
    )
    validation_documents = [
        str(document["document_hash"])
        for document in _read_json(validation_report)["documents"]
    ]
    selected = {key: _select_candidates(payload) for key, payload in records.items()}
    monotone_histogram = Counter(
        int(choice["monotone"]["r"]) for choice in selected.values()
    )
    map_histogram = Counter(
        int(choice["map_mixed"]["r"]) for choice in selected.values()
    )
    proposal_rank_histogram = Counter(
        int(choice["map_entry"]["proposal_rank"])
        for choice in selected.values()
        if choice["map_entry"] is not None
    )
    map_kind_histogram = Counter(choice["map_kind"] for choice in selected.values())

    per_expert = {}
    for (layer, expert), choice in sorted(selected.items()):
        canonical_validation = float(
            _metric(choice["canonical"], "validation")["weighted_squared_error"]
        )
        map_entry = choice["map_entry"]
        per_expert[f"{layer}/{expert}"] = {
            "monotone": _candidate_identity(choice["monotone"]),
            "record_map": _candidate_identity(choice["map_mixed"], map_entry),
            "record_map_family": choice["map_kind"],
            "validation_sse": {
                "canonical_r0": canonical_validation,
                "monotone": float(
                    _metric(choice["monotone"], "validation")[
                        "weighted_squared_error"
                    ]
                ),
                "record_map_matched_r0": float(
                    _metric(choice["map_matched_r0"], "validation")[
                        "weighted_squared_error"
                    ]
                ),
                "record_map_mixed": float(
                    _metric(choice["map_mixed"], "validation")[
                        "weighted_squared_error"
                    ]
                ),
            },
        }

    layers = sorted({layer for layer, _expert in records})
    layer_summaries = {}
    for offset, layer in enumerate(layers):
        layer_records = {key: value for key, value in records.items() if key[0] == layer}
        layer_selected = {key: selected[key] for key in layer_records}
        layer_summaries[str(layer)] = _summarize_subset(
            layer_records,
            layer_selected,
            validation_documents,
            replicates=args.bootstrap_replicates,
            seed=args.seed + 100 * (offset + 1),
        )

    result = {
        "kind": "kquant_w2_record_map_summary",
        "schema_version": 1,
        "study_contract": study_contract,
        "selection_contract": {
            "per_expert_candidate_selection": "complete training-fold SSE only",
            "validation_usage": "evaluation only",
            "proposal_search": (
                "top-k under independent-record slopes; not a combinatorial "
                "dense-H oracle"
            ),
        },
        "inputs": [str(path.resolve()) for path in args.input],
        "experts": len(records),
        "layers": layers,
        "reference_closed_candidates": closures,
        "selected_monotone_mode_histogram": {
            f"R{r}": monotone_histogram.get(r, 0)
            for r in sorted(set(study_contract["r_values"]) | {0})
        },
        "selected_record_map_mode_histogram": {
            f"R{r}": map_histogram.get(r, 0)
            for r in sorted(set(study_contract["r_values"]) | {0})
        },
        "selected_record_map_proposal_rank_histogram": {
            str(rank): count for rank, count in sorted(proposal_rank_histogram.items())
        },
        "selected_record_map_family_histogram": dict(sorted(map_kind_histogram.items())),
        "heldout_validation": _summarize_subset(
            records,
            selected,
            validation_documents,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        "per_layer_heldout_validation": layer_summaries,
        "per_expert": per_expert,
        "limitations": [
            "The representative experts were chosen using training support and training-selected ladder behavior; this is not a population estimate.",
            "The arbitrary-map pool is exact only under the separable proposal slopes; full dense-H encodes evaluate its candidates but do not exhaust all record maps.",
            "This is a w2-only ranking/partition study; a deployable shared permutation must next be evaluated across w1, w3, and w2 together.",
        ],
    }
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["heldout_validation"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
