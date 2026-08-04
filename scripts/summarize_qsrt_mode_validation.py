#!/usr/bin/env python3
"""Gate selected rate-shifted modes against matched R0 external encodes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kquant.pack.qsrt_pool import load_qsrt_candidate_pool
from kquant.pack.qsrt_mode_validation import (
    MODE_VALIDATION_SUMMARY_KIND,
    MODE_VALIDATION_SUMMARY_SCHEMA_VERSION,
    load_qsrt_mode_validation_scores,
    mode_validation_summary_document,
    paired_document_bootstrap,
    summarize_mode_validation_arrays,
)
from kquant.pack.qsrt_validation import load_qsrt_validation_scores


# Preserve the original public helper name used by experiment code and tests.
summarize_arrays = summarize_mode_validation_arrays
SUMMARY_KIND = MODE_VALIDATION_SUMMARY_KIND
SUMMARY_SCHEMA_VERSION = MODE_VALIDATION_SUMMARY_SCHEMA_VERSION


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def summarize(args: argparse.Namespace) -> dict[str, object]:
    pool = load_qsrt_candidate_pool(args.candidate_pool)
    selected_validation = load_qsrt_validation_scores(
        args.validation_scores,
        pool,
    )
    mode_validation = load_qsrt_mode_validation_scores(
        args.mode_validation_scores,
        pool,
        selected_validation,
    )
    return mode_validation_summary_document(
        pool,
        selected_validation,
        mode_validation,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--validation-scores", type=Path, required=True)
    parser.add_argument("--mode-validation-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    return args


def main() -> None:
    args = parse_args()
    result = summarize(args)
    _atomic_json(args.output, result)
    print(
        f"wrote {args.output}: {result['status']}, "
        f"{result['decision']['audited_assignments']} matched-R0 assignments"
    )


if __name__ == "__main__":
    main()
