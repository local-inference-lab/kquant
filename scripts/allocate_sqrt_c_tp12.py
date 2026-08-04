#!/usr/bin/env python3
"""Allocate SQRT-C trellis and exact X4 experts at an exact-byte budget."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from kquant.pack.mixed_allocation import (
    keep_mask_from_allocation_document,
    load_mixed_candidate_pool,
    total_container_bytes,
)
from kquant.pack.mixed_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
    load_mixed_validation_scores,
)
from kquant.pack.sqrt_c_allocation import (
    choose_sqrt_c_lagrangian,
    choose_sqrt_c_target,
    sqrt_c_allocation_document,
    write_sqrt_c_allocation,
)
from kquant.pack.x4_index import load_x4_cost_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--x4-cost-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-container-bytes", type=int)
    target.add_argument(
        "--target-allocation",
        type=Path,
        help=(
            "derive the exact logical TP12 expert-container byte cap from an "
            "existing allocation"
        ),
    )
    target.add_argument("--lagrange-lambda", type=float)
    parser.add_argument("--validation-scores", type=Path)
    parser.add_argument("--skip-payload-header-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    pool = load_mixed_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=not args.skip_payload_header_validation,
    )
    x4_index = load_x4_cost_index(args.x4_cost_index)
    if x4_index.manifest["source_revision"] != pool.manifest["source_revision"]:
        raise ValueError("SQRT-C candidate pool and X4 index source revisions differ")
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
    target_container_bytes = args.target_container_bytes
    if args.target_allocation is not None:
        reference = json.loads(args.target_allocation.read_text())
        target_container_bytes = total_container_bytes(
            keep_mask_from_allocation_document(reference)
        )
    if target_container_bytes is not None:
        allocation = choose_sqrt_c_target(
            pool.damage,
            x4_index.expert_storage_bytes,
            target_container_bytes=target_container_bytes,
        )
    else:
        allocation = choose_sqrt_c_lagrangian(
            pool.damage,
            x4_index.expert_storage_bytes,
            lagrange_lambda=args.lagrange_lambda,
        )
    document = sqrt_c_allocation_document(pool, x4_index, allocation)
    if args.target_allocation is not None:
        document["meta"]["target_allocation"] = str(
            args.target_allocation.resolve()
        )
    write_sqrt_c_allocation(args.output, document)
    print(json.dumps(document["meta"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
