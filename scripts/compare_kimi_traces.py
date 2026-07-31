#!/usr/bin/env python
"""Validate and compare two Kimi correctness trace directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.correctness import write_json
from kquant.trace_compare import compare_traces, load_trace, select_trace_call


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-route-mismatch-fraction", type=float, default=0.0)
    parser.add_argument("--left-call", type=int)
    parser.add_argument("--right-call", type=int)
    parser.add_argument(
        "--common-keys-only",
        action="store_true",
        help="compare the shared layer/stage keys without failing on extra keys",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    left = load_trace(args.left)
    right = load_trace(args.right)
    if args.left_call is not None:
        left = select_trace_call(left, args.left_call)
    if args.right_call is not None:
        right = select_trace_call(right, args.right_call)
    report = compare_traces(
        left,
        right,
        max_relative_l2=args.max_relative_l2,
        min_cosine=args.min_cosine,
        max_route_mismatch_fraction=args.max_route_mismatch_fraction,
        allow_key_coverage_difference=args.common_keys_only,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
