#!/usr/bin/env python3
"""Measure descriptive drift between official and interim Kimi trace runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.constants import NUM_EXPERTS
from kquant.correctness import write_json
from kquant.teacher_proxy import compare_teacher_proxy_traces
from kquant.trace_compare import load_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official", type=Path)
    parser.add_argument("interim", type=Path)
    parser.add_argument("--call", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--min-expert-post-situ-routes", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_teacher_proxy_traces(
        load_trace(args.official),
        load_trace(args.interim),
        call=args.call,
        num_experts=args.num_experts,
        min_expert_post_situ_routes=args.min_expert_post_situ_routes,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        displayed = {
            "output": str(args.output.resolve()),
            "status": report["status"],
            "scope": report["scope"],
            "route_summary": report["route_summary"],
            "float_stage_summary": report["float_stage_summary"],
            "post_situ_summary": report["post_situ_summary"],
        }
    else:
        displayed = report
    print(json.dumps(displayed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
