"""Summarize leak-free expert-local H13 studies for ``w1``/``w3``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from kquant.candidate_hessian import partition_documents
from scripts.summarize_candidate_hhat import (
    _aggregate_pair,
    _distribution,
    _global_split_argmin,
)
from scripts.summarize_retained_w2_rate_transfer import _read_json


RecordKey = tuple[int, int]


def _metric_record(payload: dict) -> dict:
    """Expose the common variant shape used by the coupled summary helpers."""

    return {"modes": [{"variants": payload["variants"]}]}


def _validate_and_collect(
    paths: Sequence[Path],
) -> tuple[dict[RecordKey, dict], list[str], Path, Path, dict, int]:
    records: dict[RecordKey, dict] = {}
    names = None
    common_contract = None
    training_report = None
    validation_report = None
    study_contract = None
    closures = 0
    for path in paths:
        payload = _read_json(path)
        if (
            payload.get("kind") != "kquant_expert_local_h13_experiment"
            or payload.get("schema_version") != 1
            or not payload.get("complete")
        ):
            raise ValueError(f"{path} is not a complete local-H13 experiment")
        signature = payload["signature"]
        if signature.get("tp_size") != 12:
            raise ValueError(f"{path} is not TP12")
        provenance = signature["provenance"]
        candidate_training_report = Path(provenance["training_report"])
        candidate_validation_report = Path(provenance["validation_report"])
        contract = (
            signature["checkpoint"],
            signature["source_checkpoint"],
            signature["source_revision"],
            signature["capture"],
            signature["validation_capture"],
            signature["hessians"],
            tuple(signature["h13_alphas"]),
            float(signature["w2_local_alpha"]),
            tuple(sorted(signature["document_split"].items())),
            candidate_training_report,
            candidate_validation_report,
        )
        if common_contract not in (None, contract):
            raise ValueError("local-H13 inputs have different study contracts")
        common_contract = contract
        training_report = candidate_training_report
        validation_report = candidate_validation_report
        study_contract = {
            "resident_teacher_checkpoint": signature["checkpoint"],
            "source_checkpoint": signature["source_checkpoint"],
            "source_revision": signature["source_revision"],
            "capture": signature["capture"],
            "validation_capture": signature["validation_capture"],
            "hessians": signature["hessians"],
            "h13_alphas": signature["h13_alphas"],
            "w2_local_alpha": signature["w2_local_alpha"],
            "document_split": signature["document_split"],
            "tp_size": 12,
        }
        key = (int(signature["layer"]), int(signature["expert"]))
        if key in records:
            raise ValueError(f"duplicate result {key}")
        variants = payload["variants"]
        candidate_names = tuple(variant["name"] for variant in variants)
        if not candidate_names or candidate_names[0] != "global_teacher":
            raise ValueError(f"{path} lacks the global H13 control")
        if names not in (None, candidate_names):
            raise ValueError("local-H13 inputs have different alpha ladders")
        names = candidate_names
        byte_contract = payload["exact_byte_contract"]
        if not byte_contract.get("all_h13_variants_equal"):
            raise ValueError(f"{path} lacks equal-byte closure")
        expected_bytes = (
            int(byte_contract["total_trellis_bytes"]),
            int(byte_contract["total_auxiliary_bytes"]),
        )
        w2 = payload["w2"]
        if not w2["reference_edge_roundtrip"] or not w2["reference_state_roundtrip"]:
            raise ValueError(f"{path} failed w2 reference closure")
        closures += 1
        for variant in variants:
            if (
                int(variant["total_trellis_bytes"]),
                int(variant["total_auxiliary_bytes"]),
            ) != expected_bytes:
                raise ValueError(f"{path} H13 variant changed exact bytes")
            for matrix in (variant["w1"], variant["w3"]):
                if not matrix["reference_edge_roundtrip"] or not matrix[
                    "reference_state_roundtrip"
                ]:
                    raise ValueError(f"{path} failed w1/w3 reference closure")
                closures += 1
            closure = variant["physical_permutation_closure"]
            if closure["relative_l2"] > 5e-6 or closure["max_abs"] > 5e-4:
                raise ValueError(f"{path} failed common permutation closure")
        records[key] = payload
    if (
        names is None
        or training_report is None
        or validation_report is None
        or study_contract is None
    ):
        raise ValueError("no local-H13 inputs")
    return (
        records,
        list(names),
        training_report,
        validation_report,
        study_contract,
        closures,
    )


def run(args: argparse.Namespace) -> dict:
    (
        raw_records,
        names,
        training_report,
        validation_report,
        study_contract,
        closures,
    ) = _validate_and_collect(args.input)
    records = {key: _metric_record(payload) for key, payload in raw_records.items()}
    training_documents = [
        str(document["document_hash"])
        for document in _read_json(training_report)["documents"]
    ]
    split = study_contract["document_split"]
    fit_documents, confirmation_documents = partition_documents(
        training_documents,
        modulus=int(split["modulus"]),
        confirmation_index=int(split["confirmation_index"]),
    )
    if len(fit_documents) != int(split["fit_documents"]) or len(
        confirmation_documents
    ) != int(split["confirmation_documents"]):
        raise ValueError("document split does not reproduce")
    validation_documents = [
        str(document["document_hash"])
        for document in _read_json(validation_report)["documents"]
    ]

    fixed = {}
    for index, name in enumerate(names):
        fixed[name] = {
            "confirmation": _aggregate_pair(
                records,
                confirmation_documents,
                lambda _key: "global_teacher",
                lambda _key, name=name: name,
                split="confirmation",
                replicates=args.bootstrap_replicates,
                seed=args.seed + index,
            ),
            "heldout": _aggregate_pair(
                records,
                validation_documents,
                lambda _key: "global_teacher",
                lambda _key, name=name: name,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 100 + index,
            ),
        }
    confirmation_choice = _global_split_argmin(records, names, "confirmation")
    training_choice = _global_split_argmin(records, names, "train")
    selected = {
        "confirmation_argmin": {
            "selected": confirmation_choice,
            "heldout": _aggregate_pair(
                records,
                validation_documents,
                lambda _key: "global_teacher",
                lambda _key: confirmation_choice,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 200,
            ),
        },
        "fit_argmin_diagnostic": {
            "selected": training_choice,
            "confirmation": _aggregate_pair(
                records,
                confirmation_documents,
                lambda _key: "global_teacher",
                lambda _key: training_choice,
                split="confirmation",
                replicates=args.bootstrap_replicates,
                seed=args.seed + 201,
            ),
            "heldout": _aggregate_pair(
                records,
                validation_documents,
                lambda _key: "global_teacher",
                lambda _key: training_choice,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 202,
            ),
        },
    }

    h13_difference = []
    support_rows = []
    support_documents = []
    per_expert = {}
    for (layer, expert), payload in sorted(raw_records.items()):
        support = payload["covariance_support"]
        h13_difference.append(
            float(
                payload["covariance_diagnostics"]["expert_h13_vs_global"][
                    "relative_frobenius_difference"
                ]
            )
        )
        support_rows.append(int(support["fit_rows"]))
        support_documents.append(int(support["fit_documents_routed"]))
        per_expert[f"{layer}/{expert}"] = {
            "r": int(payload["r"]),
            "support": support,
            "expert_h13_vs_global": payload["covariance_diagnostics"][
                "expert_h13_vs_global"
            ],
            "confirmation_sse_by_h13": {
                variant["name"]: variant["coupled_activation"]["confirmation"][
                    "route_weighted_squared_error"
                ]
                for variant in payload["variants"]
            },
            "validation_sse_by_h13": {
                variant["name"]: variant["coupled_activation"]["validation"][
                    "route_weighted_squared_error"
                ]
                for variant in payload["variants"]
            },
        }

    result = {
        "kind": "kquant_expert_local_h13_summary",
        "schema_version": 1,
        "study_contract": study_contract,
        "inputs": [str(path.resolve()) for path in args.input],
        "experts": len(records),
        "layers": sorted({layer for layer, _expert in records}),
        "available_h13_variants": names,
        "reference_and_permutation_closures_checked": closures,
        "selection_contract": {
            "covariance_fit": "fit documents only",
            "alpha_selection": "confirmation documents only",
            "validation_usage": "evaluation only",
        },
        "support": {
            "fit_route_rows_per_expert": _distribution(support_rows),
            "fit_documents_per_expert": _distribution(support_documents),
        },
        "covariance_diagnostics": {
            "expert_h13_vs_global_relative_frobenius": _distribution(
                h13_difference
            )
        },
        "fixed_variants_vs_global": fixed,
        "selected": selected,
        "per_expert": per_expert,
        "limitations": [
            "The seven experts were selected for support and prior mode behavior, not as a population sample.",
            "The w2 source-local alpha was fixed from the preceding study, so this isolates only the incremental H13 choice.",
            "The study scores routed expert self-terms rather than cross-expert mixture-error cancellation.",
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
                "selected": result["selected"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
