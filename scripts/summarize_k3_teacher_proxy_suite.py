#!/usr/bin/env python3
"""Validate and summarize a document-diverse K3 teacher-proxy trace pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kquant.correctness import write_json
from kquant.teacher_proxy_gate import (
    TeacherProxyThresholds,
    compare_teacher_proxy_suite,
)


DEFAULT_INTERIM = Path("/models/Kimi-K3-EXL3-3p09-serve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--expected-interim", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--min-expert-post-situ-routes", type=int, default=4)
    parser.add_argument("--min-route-retention", type=float, default=0.85)
    parser.add_argument("--min-gate-cosine", type=float, default=0.90)
    parser.add_argument("--min-record-spearman", type=float, default=0.75)
    parser.add_argument("--min-r3-donor-overlap", type=float, default=1 / 3)
    parser.add_argument("--min-r3-recipient-overlap", type=float, default=1 / 3)
    parser.add_argument("--min-expert-case-support", type=float, default=0.75)
    parser.add_argument("--min-expert-spearman-median", type=float, default=0.65)
    parser.add_argument("--min-expert-spearman-p10", type=float, default=0.20)
    parser.add_argument("--min-expert-r3-donor-overlap", type=float, default=0.40)
    parser.add_argument(
        "--min-expert-r3-recipient-overlap", type=float, default=0.40
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    for name in (
        "min_route_retention",
        "min_gate_cosine",
        "min_record_spearman",
        "min_r3_donor_overlap",
        "min_r3_recipient_overlap",
        "min_expert_case_support",
        "min_expert_spearman_median",
        "min_expert_spearman_p10",
        "min_expert_r3_donor_overlap",
        "min_expert_r3_recipient_overlap",
    ):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    report = compare_teacher_proxy_suite(
        suite_path=args.suite,
        official_root=args.official,
        interim_root=args.interim,
        expected_interim=args.expected_interim,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        min_expert_post_situ_routes=args.min_expert_post_situ_routes,
        thresholds=TeacherProxyThresholds(
            route_assignment_retention_fraction=args.min_route_retention,
            sparse_applied_gate_cosine=args.min_gate_cosine,
            record_rank_spearman=args.min_record_spearman,
            r3_donor_record_overlap_fraction=args.min_r3_donor_overlap,
            r3_recipient_record_overlap_fraction=args.min_r3_recipient_overlap,
            expert_case_support_fraction=args.min_expert_case_support,
            expert_record_rank_spearman_median=args.min_expert_spearman_median,
            expert_record_rank_spearman_p10=args.min_expert_spearman_p10,
            expert_r3_donor_record_overlap_fraction=(
                args.min_expert_r3_donor_overlap
            ),
            expert_r3_recipient_record_overlap_fraction=(
                args.min_expert_r3_recipient_overlap
            ),
        ),
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "next_action": report["next_action"],
                "scope": report["scope"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
