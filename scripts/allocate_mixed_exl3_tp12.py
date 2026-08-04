#!/usr/bin/env python3
"""Validate and allocate a complete TP12 mixed-EXL candidate pool."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from kquant.pack.mixed_allocation import (
    CERTIFIED_ALLOCATION_OPTIMALITY,
    allocation_optimality,
    allocation_document,
    choose_mixed_allocation,
    keep_mask_from_allocation_document,
    load_mixed_candidate_pool,
    total_container_bytes,
    write_allocation,
)
from kquant.pack.mixed_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
    load_mixed_validation_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--keep-frac", type=float)
    target.add_argument("--keep-count", type=int)
    target.add_argument("--target-container-bytes", type=int)
    target.add_argument(
        "--target-allocation",
        type=Path,
        help=(
            "derive the exact TP12 expert-container budget from an existing "
            "allocation"
        ),
    )
    parser.add_argument(
        "--skip-payload-header-validation",
        action="store_true",
        help="validate metrics/ledgers but skip the 92 large safetensors headers",
    )
    parser.add_argument(
        "--validation-scores",
        type=Path,
        help=(
            "rank MXFP4 keeps by complete document-disjoint selected-candidate "
            "scores instead of selection-corpus damage"
        ),
    )
    parser.add_argument(
        "--allow-uncertified-alignment-repair",
        action="store_true",
        help=(
            "allow a deterministic greedy repair when the globally top-ranked "
            "keep set overruns layer-alignment padding; diagnostic only"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    pool = load_mixed_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=not args.skip_payload_header_validation,
    )
    if args.validation_scores is not None:
        validation = load_mixed_validation_scores(args.validation_scores, pool)
        pool = replace(
            pool,
            damage=validation.damage,
            damage_metric=VALIDATION_DAMAGE_METRIC,
            damage_weighting=VALIDATION_DAMAGE_WEIGHTING,
            damage_provenance={
                "validation_scores": str(validation.root),
                "validation_score_set_sha256": validation.content_sha256,
                "validation_capture": validation.manifest["validation_capture"],
                "validation_report": validation.manifest["validation_report"],
                "validation_documents": validation.manifest[
                    "validation_documents"
                ],
                "selection_data_used": False,
            },
        )
    target_bytes = args.target_container_bytes
    if args.target_allocation is not None:
        reference = json.loads(args.target_allocation.read_text())
        target_bytes = total_container_bytes(
            keep_mask_from_allocation_document(reference)
        )
    allocation = choose_mixed_allocation(
        pool.damage,
        keep_fraction=args.keep_frac,
        keep_count=args.keep_count,
        target_container_bytes=target_bytes,
    )
    optimality = allocation_optimality(allocation)
    if (
        optimality != CERTIFIED_ALLOCATION_OPTIMALITY
        and not args.allow_uncertified_alignment_repair
    ):
        raise SystemExit(
            "allocation requires an alignment repair without a global "
            "optimality proof; refusing to freeze a production keep tier. "
            "Implement the exact repair solver, or use "
            "--allow-uncertified-alignment-repair only for diagnostics."
        )
    document = allocation_document(pool, allocation)
    if args.target_allocation is not None:
        document["meta"]["target_allocation"] = str(
            args.target_allocation.resolve()
        )
    write_allocation(args.output, document)
    meta = document["meta"]
    print(
        f"wrote {args.output}: kept {meta['retained_experts']}, "
        f"compressed {meta['compressed_experts']}, "
        f"{meta['container_bytes']} bytes, "
        f"{meta['effective_bits_per_original_expert_weight']:.6f} bpw, "
        f"{meta['allocation_optimality']}"
    )


if __name__ == "__main__":
    main()
