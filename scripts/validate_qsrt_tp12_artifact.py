#!/usr/bin/env python3
"""Validate a completed TP12 Kimi-K3 QSRT artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kquant.pack.qsrt_validate import validate_qsrt_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument(
        "--verify-payloads",
        action="store_true",
        help="recompare every SQG and X4T payload with its immutable source",
    )
    parser.add_argument(
        "--skip-candidate-payload-header-validation",
        action="store_true",
        help="skip the 92 candidate payload geometry inventories",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(document, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    report = validate_qsrt_artifact(
        args.artifact,
        validate_candidate_payload_headers=(
            not args.skip_candidate_payload_header_validation
        ),
        verify_payloads=args.verify_payloads,
        official_repo_dir=args.official_repo_dir,
    )
    if args.output is not None:
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
