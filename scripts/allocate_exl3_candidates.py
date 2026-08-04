#!/usr/bin/env python3
"""Materialize a new keep/EXL3 allocation from an existing candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.pack.candidates import (
    choose_allocation,
    materialize_allocation,
    score_candidates,
    write_candidate_allocation,
)
from kquant.pack.exl3 import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dir",
        "--candidates",
        dest="candidate_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--keep-frac", type=float, required=True)
    args = parser.parse_args()

    if args.dest.exists() and any(args.dest.iterdir()):
        raise SystemExit(
            f"destination {args.dest} is not empty; use a fresh allocation directory"
        )
    args.dest.mkdir(parents=True, exist_ok=True)
    scores = score_candidates(args.candidate_dir, args.stats)
    allocation = choose_allocation(scores, args.keep_frac)
    write_candidate_allocation(
        args.dest,
        allocation,
        keep_frac=args.keep_frac,
        stats_path=args.stats,
        candidate_dir=args.candidate_dir,
    )
    materialize_allocation(args.candidate_dir, args.dest, allocation)

    source_manifest_path = args.candidate_dir.parent / "kquant_exl3_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text())
        if source_manifest_path.exists()
        else {}
    )
    source_manifest.update(
        {
            "keep_frac": args.keep_frac,
            "candidate_pool": str(args.candidate_dir.resolve()),
            "stats": str(args.stats.resolve()),
        }
    )
    write_manifest(args.dest, source_manifest)
    print(f"materialized allocation in {args.dest}")


if __name__ == "__main__":
    main()
