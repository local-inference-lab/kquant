"""Summarize document-split candidate-specific post-SiTU Hessian studies.

Every context order and encoding Hessian is constructed from fit documents.
Shrinkage choices are selected on those fit documents, checked on untouched
training-corpus confirmation documents, and only then reported on the fully
separate validation corpus.  The primary comparison uses complete coupled
expert output, not ``w2`` weight error.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from kquant.candidate_hessian import partition_documents
from scripts.summarize_coupled_exl3 import _document_values
from scripts.summarize_retained_w2_rate_transfer import (
    _bootstrap_improvement,
    _quantile,
    _read_json,
)


RecordKey = tuple[int, int]


def _mode(payload: dict) -> dict:
    modes = payload.get("modes", [])
    if len(modes) != 1:
        raise ValueError("candidate-hhat summary expects exactly one selected mode per expert")
    return modes[0]


def _variant(payload: dict, name: str) -> dict:
    matches = [item for item in _mode(payload)["variants"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Hessian variant {name}")
    return matches[0]


def _metric(variant: dict, split: str) -> dict:
    metric = variant["coupled_activation"].get(split)
    if metric is None or float(metric.get("route_weighted_reference_energy", 0.0)) <= 0:
        raise ValueError(f"variant lacks a valid {split} metric")
    return metric


def _training_argmin(
    payload: dict, allowed_names: Sequence[str]
) -> str:
    return min(
        allowed_names,
        key=lambda name: (
            float(_metric(_variant(payload, name), "train")["route_weighted_squared_error"]),
            name,
        ),
    )


def _global_training_argmin(
    records: Mapping[RecordKey, dict], allowed_names: Sequence[str]
) -> str:
    return _global_split_argmin(records, allowed_names, "train")


def _global_split_argmin(
    records: Mapping[RecordKey, dict],
    allowed_names: Sequence[str],
    split: str,
) -> str:
    return min(
        allowed_names,
        key=lambda name: (
            sum(
                float(
                    _metric(_variant(payload, name), split)[
                        "route_weighted_squared_error"
                    ]
                )
                for payload in records.values()
            ),
            name,
        ),
    )


def _aggregate_pair(
    records: Mapping[RecordKey, dict],
    documents: Sequence[str],
    baseline_selector: Callable[[RecordKey], str],
    candidate_selector: Callable[[RecordKey], str],
    *,
    split: str = "validation",
    replicates: int,
    seed: int,
) -> dict:
    baseline_sse = {document: 0.0 for document in documents}
    candidate_sse = {document: 0.0 for document in documents}
    expert_reference = {document: 0.0 for document in documents}
    mixture_reference = {document: 0.0 for document in documents}
    baseline_histogram = Counter()
    candidate_histogram = Counter()
    per_expert_ratios = []
    for key, payload in records.items():
        baseline_name = baseline_selector(key)
        candidate_name = candidate_selector(key)
        baseline_histogram[baseline_name] += 1
        candidate_histogram[candidate_name] += 1
        baseline_metric = _metric(_variant(payload, baseline_name), split)
        candidate_metric = _metric(_variant(payload, candidate_name), split)
        baseline_values = _document_values(
            baseline_metric, "route_weighted_squared_error"
        )
        candidate_values = _document_values(
            candidate_metric, "route_weighted_squared_error"
        )
        baseline_expert_reference = _document_values(
            baseline_metric, "route_weighted_reference_energy"
        )
        candidate_expert_reference = _document_values(
            candidate_metric, "route_weighted_reference_energy"
        )
        baseline_mixture_reference = _document_values(
            baseline_metric, "teacher_mixture_energy"
        )
        candidate_mixture_reference = _document_values(
            candidate_metric, "teacher_mixture_energy"
        )
        baseline_total = float(baseline_metric["route_weighted_squared_error"])
        candidate_total = float(candidate_metric["route_weighted_squared_error"])
        if baseline_total > 0:
            per_expert_ratios.append(candidate_total / baseline_total)
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
                raise ValueError(f"reference energies differ for {key}")
            baseline_sse[document] += baseline_values.get(document, 0.0)
            candidate_sse[document] += candidate_values.get(document, 0.0)
            expert_reference[document] += baseline_expert_reference.get(document, 0.0)
            mixture_reference[document] += baseline_mixture_reference.get(document, 0.0)
    result = _bootstrap_improvement(
        baseline_sse,
        candidate_sse,
        list(documents),
        replicates=replicates,
        seed=seed,
    )
    expert_reference_total = sum(expert_reference.values())
    mixture_reference_total = sum(mixture_reference.values())
    result.update(
        {
            "experts": len(records),
            "documents": len(documents),
            "baseline_hessian_histogram": dict(sorted(baseline_histogram.items())),
            "candidate_hessian_histogram": dict(sorted(candidate_histogram.items())),
            "route_weighted_expert_reference_energy": expert_reference_total,
            "teacher_mixture_reference_energy": mixture_reference_total,
            "baseline_route_weighted_expert_nmse": (
                result["baseline_weighted_squared_error"] / expert_reference_total
                if expert_reference_total > 0
                else None
            ),
            "candidate_route_weighted_expert_nmse": (
                result["candidate_weighted_squared_error"] / expert_reference_total
                if expert_reference_total > 0
                else None
            ),
            "baseline_teacher_mixture_replacement_nmse": (
                result["baseline_weighted_squared_error"] / mixture_reference_total
                if mixture_reference_total > 0
                else None
            ),
            "candidate_teacher_mixture_replacement_nmse": (
                result["candidate_weighted_squared_error"] / mixture_reference_total
                if mixture_reference_total > 0
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


def _validate_and_collect(
    paths: Sequence[Path],
) -> tuple[dict[RecordKey, dict], list[str], Path, Path, dict, int]:
    records: dict[RecordKey, dict] = {}
    variant_names = None
    validation_report = None
    training_report = None
    common_contract = None
    study_contract = None
    closures = 0
    for path in paths:
        payload = _read_json(path)
        if payload.get("kind") != "kquant_candidate_hhat_experiment":
            raise ValueError(f"{path} has the wrong experiment kind")
        if payload.get("schema_version") != 2 or not payload.get("complete"):
            raise ValueError(f"{path} is not a complete schema-v2 experiment")
        signature = payload["signature"]
        if signature.get("tp_size") != 12:
            raise ValueError(f"{path} is not a TP12 experiment")
        provenance = signature["provenance"]
        candidate_validation_report = Path(provenance["validation_report"])
        candidate_training_report = Path(provenance["training_report"])
        contract = (
            signature["checkpoint"],
            signature["source_checkpoint"],
            signature["source_revision"],
            signature["capture"],
            signature["validation_capture"],
            signature["hessians"],
            tuple(signature["alphas"]),
            tuple(sorted(signature["document_split"].items())),
            candidate_training_report,
            candidate_validation_report,
        )
        if common_contract not in (None, contract):
            raise ValueError("input experiments use different source or covariance contracts")
        common_contract = contract
        validation_report = candidate_validation_report
        training_report = candidate_training_report
        study_contract = {
            "resident_teacher_checkpoint": signature["checkpoint"],
            "source_checkpoint": signature["source_checkpoint"],
            "source_revision": signature["source_revision"],
            "capture": signature["capture"],
            "validation_capture": signature["validation_capture"],
            "hessians": signature["hessians"],
            "alphas": signature["alphas"],
            "document_split": signature["document_split"],
            "tp_size": 12,
        }
        key = (int(signature["layer"]), int(signature["expert"]))
        if key in records:
            raise ValueError(f"duplicate layer/expert result {key}")
        mode = _mode(payload)
        names = tuple(variant["name"] for variant in mode["variants"])
        if not names or names[0] != "global_teacher":
            raise ValueError(f"{path} lacks the global-teacher control")
        if variant_names not in (None, names):
            raise ValueError("input experiments expose different Hessian variants")
        variant_names = names
        byte_payload = payload["exact_byte_contract"]
        if not byte_payload.get("all_modes_and_hessian_variants_equal"):
            raise ValueError(f"{path} lacks equal-byte closure")
        byte_contract = (
            int(byte_payload["total_trellis_bytes"]),
            int(byte_payload["total_auxiliary_bytes"]),
        )
        for matrix in (mode["w1"], mode["w3"]):
            if not matrix["reference_edge_roundtrip"] or not matrix[
                "reference_state_roundtrip"
            ]:
                raise ValueError(f"{path} failed w1/w3 reference closure")
            closures += 1
        for variant in mode["variants"]:
            if (
                int(variant["total_trellis_bytes"]),
                int(variant["total_auxiliary_bytes"]),
            ) != byte_contract:
                raise ValueError(f"{path} variant changed exact bytes")
            if not variant["w2"]["reference_edge_roundtrip"] or not variant[
                "w2"
            ]["reference_state_roundtrip"]:
                raise ValueError(f"{path} failed w2 reference closure")
            closure = variant["physical_permutation_closure"]
            if closure["relative_l2"] > 5e-6 or closure["max_abs"] > 5e-4:
                raise ValueError(f"{path} failed common permutation closure")
            closures += 1
        records[key] = payload
    if (
        variant_names is None
        or validation_report is None
        or training_report is None
        or study_contract is None
    ):
        raise ValueError("no candidate-hhat inputs")
    return (
        records,
        list(variant_names),
        training_report,
        validation_report,
        study_contract,
        closures,
    )


def _distribution(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "maximum": max(values) if values else None,
    }


def run(args: argparse.Namespace) -> dict:
    (
        records,
        names,
        training_report,
        validation_report,
        study_contract,
        closures,
    ) = _validate_and_collect(args.input)
    training_documents = [
        str(document["document_hash"])
        for document in _read_json(training_report)["documents"]
    ]
    document_split = study_contract["document_split"]
    fit_documents, confirmation_documents = partition_documents(
        training_documents,
        modulus=int(document_split["modulus"]),
        confirmation_index=int(document_split["confirmation_index"]),
    )
    if len(fit_documents) != int(document_split["fit_documents"]) or len(
        confirmation_documents
    ) != int(document_split["confirmation_documents"]):
        raise ValueError("experiment document-split counts do not reproduce")
    documents = [
        str(document["document_hash"])
        for document in _read_json(validation_report)["documents"]
    ]
    source_names = [name for name in names if name.startswith("source_expert_")]
    candidate_names = [name for name in names if name.startswith("candidate_hhat_")]
    if len(source_names) != len(candidate_names):
        raise ValueError("source and candidate shrinkage ladders differ")
    fixed_vs_global = {
        name: _aggregate_pair(
            records,
            documents,
            lambda _key: "global_teacher",
            lambda _key, name=name: name,
            replicates=args.bootstrap_replicates,
            seed=args.seed + index,
        )
        for index, name in enumerate(names)
    }
    confirmation_fixed_vs_global = {
        name: _aggregate_pair(
            records,
            confirmation_documents,
            lambda _key: "global_teacher",
            lambda _key, name=name: name,
            split="confirmation",
            replicates=args.bootstrap_replicates,
            seed=args.seed + 50 + index,
        )
        for index, name in enumerate(names)
    }
    candidate_vs_source = {}
    for index, (source_name, candidate_name) in enumerate(
        zip(source_names, candidate_names, strict=True)
    ):
        source_alpha = source_name.rsplit("_a", 1)[1]
        candidate_alpha = candidate_name.rsplit("_a", 1)[1]
        if source_alpha != candidate_alpha:
            raise ValueError("source/candidate alpha ordering differs")
        candidate_vs_source[f"alpha_{source_alpha}"] = _aggregate_pair(
            records,
            documents,
            lambda _key, name=source_name: name,
            lambda _key, name=candidate_name: name,
            replicates=args.bootstrap_replicates,
            seed=args.seed + 100 + index,
        )

    global_all_choice = _global_training_argmin(records, names)
    global_source_choice = _global_training_argmin(
        records, ["global_teacher", *source_names]
    )
    global_candidate_choice = _global_training_argmin(
        records, ["global_teacher", *candidate_names]
    )
    confirmation_all_choice = _global_split_argmin(
        records, names, "confirmation"
    )
    confirmation_source_choice = _global_split_argmin(
        records, ["global_teacher", *source_names], "confirmation"
    )
    confirmation_candidate_choice = _global_split_argmin(
        records, ["global_teacher", *candidate_names], "confirmation"
    )
    per_expert_choice = {
        key: _training_argmin(payload, names) for key, payload in records.items()
    }
    selected = {
        "confirmation_selected_for_external_validation": {
            "all_variants": {
                "selected": confirmation_all_choice,
                "heldout": _aggregate_pair(
                    records,
                    documents,
                    lambda _key: "global_teacher",
                    lambda _key: confirmation_all_choice,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + 180,
                ),
            },
            "source_ladder": {
                "selected": confirmation_source_choice,
                "heldout": _aggregate_pair(
                    records,
                    documents,
                    lambda _key: "global_teacher",
                    lambda _key: confirmation_source_choice,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + 181,
                ),
            },
            "candidate_ladder": {
                "selected": confirmation_candidate_choice,
                "heldout": _aggregate_pair(
                    records,
                    documents,
                    lambda _key: "global_teacher",
                    lambda _key: confirmation_candidate_choice,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + 182,
                ),
            },
        },
        "aggregate_training_argmin_all": {
            "selected": global_all_choice,
            "confirmation": _aggregate_pair(
                records,
                confirmation_documents,
                lambda _key: "global_teacher",
                lambda _key: global_all_choice,
                split="confirmation",
                replicates=args.bootstrap_replicates,
                seed=args.seed + 190,
            ),
            "heldout": _aggregate_pair(
                records,
                documents,
                lambda _key: "global_teacher",
                lambda _key: global_all_choice,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 200,
            ),
        },
        "aggregate_training_argmin_source_ladder": {
            "selected": global_source_choice,
            "confirmation": _aggregate_pair(
                records,
                confirmation_documents,
                lambda _key: "global_teacher",
                lambda _key: global_source_choice,
                split="confirmation",
                replicates=args.bootstrap_replicates,
                seed=args.seed + 191,
            ),
            "heldout": _aggregate_pair(
                records,
                documents,
                lambda _key: "global_teacher",
                lambda _key: global_source_choice,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 201,
            ),
        },
        "aggregate_training_argmin_candidate_ladder": {
            "selected": global_candidate_choice,
            "confirmation": _aggregate_pair(
                records,
                confirmation_documents,
                lambda _key: "global_teacher",
                lambda _key: global_candidate_choice,
                split="confirmation",
                replicates=args.bootstrap_replicates,
                seed=args.seed + 192,
            ),
            "heldout": _aggregate_pair(
                records,
                documents,
                lambda _key: "global_teacher",
                lambda _key: global_candidate_choice,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 202,
            ),
        },
        "per_expert_training_argmin_all_diagnostic": {
            "selected_histogram": dict(sorted(Counter(per_expert_choice.values()).items())),
            "confirmation": _aggregate_pair(
                records,
                confirmation_documents,
                lambda _key: "global_teacher",
                lambda key: per_expert_choice[key],
                split="confirmation",
                replicates=args.bootstrap_replicates,
                seed=args.seed + 193,
            ),
            "heldout": _aggregate_pair(
                records,
                documents,
                lambda _key: "global_teacher",
                lambda key: per_expert_choice[key],
                replicates=args.bootstrap_replicates,
                seed=args.seed + 203,
            ),
        },
    }

    per_expert = {}
    middle_nmse = []
    candidate_vs_source_covariance = []
    source_vs_global_covariance = []
    support_rows = []
    support_documents = []
    for (layer, expert), payload in sorted(records.items()):
        mode = _mode(payload)
        middle = mode["training_middle"]
        support = payload["covariance_support"]
        middle_nmse.append(float(middle["candidate_to_source_weighted_nmse"]))
        candidate_vs_source_covariance.append(
            float(
                middle["candidate_hhat_covariance_vs_source_expert"][
                    "relative_frobenius_difference"
                ]
            )
        )
        source_vs_global_covariance.append(
            float(
                middle["source_expert_covariance_vs_global"][
                    "relative_frobenius_difference"
                ]
            )
        )
        support_rows.append(int(support["fit_rows"]))
        support_documents.append(int(support["fit_documents_routed"]))
        per_expert[f"{layer}/{expert}"] = {
            "r": int(mode["r"]),
            "training_argmin": per_expert_choice[layer, expert],
            "covariance_support": support,
            "training_middle": middle,
            "validation_sse_by_hessian": {
                name: float(
                    _metric(_variant(payload, name), "validation")[
                        "route_weighted_squared_error"
                    ]
                )
                for name in names
            },
            "confirmation_sse_by_hessian": {
                name: float(
                    _metric(_variant(payload, name), "confirmation")[
                        "route_weighted_squared_error"
                    ]
                )
                for name in names
            },
        }

    result = {
        "kind": "kquant_candidate_hhat_summary",
        "schema_version": 2,
        "study_contract": study_contract,
        "inputs": [str(path.resolve()) for path in args.input],
        "experts": len(records),
        "layers": sorted({layer for layer, _expert in records}),
        "available_hessian_variants": names,
        "reference_and_permutation_closures_checked": closures,
        "selection_contract": {
            "covariance_fit": "fit-document routed rows only",
            "option_selection": "fit-document coupled-output SSE only",
            "confirmation_usage": "selection confirmation only",
            "validation_usage": "evaluation only",
        },
        "support": {
            "training_route_rows_per_expert": _distribution(support_rows),
            "training_documents_per_expert": _distribution(support_documents),
        },
        "covariance_diagnostics": {
            "candidate_middle_to_source_weighted_nmse": _distribution(middle_nmse),
            "candidate_hhat_vs_source_expert_relative_frobenius": _distribution(
                candidate_vs_source_covariance
            ),
            "source_expert_vs_global_relative_frobenius": _distribution(
                source_vs_global_covariance
            ),
        },
        "heldout_fixed_variants_vs_global": fixed_vs_global,
        "confirmation_fixed_variants_vs_global": confirmation_fixed_vs_global,
        "heldout_candidate_hhat_vs_source_expert_at_same_alpha": candidate_vs_source,
        "training_selected": selected,
        "per_expert": per_expert,
        "limitations": [
            "The seven experts were chosen using training support and prior training-fold mode behavior, not as a population sample.",
            "The local uncentered covariance has rank at most its routed-row count; fixed shrinkage controls this but does not replace production-scale support.",
            "The fit/confirmation split is internal to the 115-document training corpus; the 27-document external validation corpus remains the final study fold.",
            "The study scores isolated routed expert self-terms and does not include cross-expert mixture-error cancellation.",
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
    print(
        json.dumps(
            {
                "experts": result["experts"],
                "covariance_diagnostics": result["covariance_diagnostics"],
                "training_selected": result["training_selected"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
