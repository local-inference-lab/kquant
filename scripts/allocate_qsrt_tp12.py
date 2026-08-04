#!/usr/bin/env python3
"""Allocate QSRT trellis and exact X4T experts at an exact-byte budget."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from kquant.pack.qsrt_pool import load_qsrt_candidate_pool
from kquant.pack.qsrt_validation import (
    VALIDATION_DAMAGE_METRIC,
    VALIDATION_DAMAGE_WEIGHTING,
    load_qsrt_validation_scores,
)
from kquant.pack.qsrt_allocation import (
    choose_qsrt_lagrangian,
    choose_qsrt_target,
    qsrt_allocation_document,
    write_qsrt_allocation,
)
from kquant.pack.x4t_index import load_x4t_cost_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--x4t-cost-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-container-bytes", type=int)
    target.add_argument("--lagrange-lambda", type=float)
    parser.add_argument("--validation-scores", type=Path)
    parser.add_argument("--skip-payload-header-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    pool = load_qsrt_candidate_pool(
        args.candidate_pool,
        validate_payload_headers=not args.skip_payload_header_validation,
    )
    x4t_index = load_x4t_cost_index(args.x4t_cost_index)
    if x4t_index.manifest["source_revision"] != pool.manifest["source_revision"]:
        raise ValueError("QSRT candidate pool and X4T index source revisions differ")
    if args.validation_scores is not None:
        validation = load_qsrt_validation_scores(args.validation_scores, pool)
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
    if target_container_bytes is not None:
        allocation = choose_qsrt_target(
            pool.damage,
            x4t_index.expert_storage_bytes,
            target_container_bytes=target_container_bytes,
        )
    else:
        allocation = choose_qsrt_lagrangian(
            pool.damage,
            x4t_index.expert_storage_bytes,
            lagrange_lambda=args.lagrange_lambda,
        )
    document = qsrt_allocation_document(pool, x4t_index, allocation)
    write_qsrt_allocation(args.output, document)
    print(json.dumps(document["meta"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
