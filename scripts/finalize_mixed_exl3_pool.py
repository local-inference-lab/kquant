#!/usr/bin/env python3
"""Hash and seal a complete TP12 mixed-EXL candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.pack.mixed_allocation import (
    CANDIDATE_POOL_COMPLETION_FILENAME,
    candidate_pool_completion_document,
    load_mixed_candidate_pool,
    validate_candidate_pool_completion,
    write_candidate_pool_completion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_pool", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument(
        "--verify-existing-hashes",
        action="store_true",
        help="rehash every source when validating an existing completion index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.candidate_pool.resolve()
    completion_path = root / CANDIDATE_POOL_COMPLETION_FILENAME
    if completion_path.exists():
        completion = validate_candidate_pool_completion(
            root,
            verify_hashes=args.verify_existing_hashes,
        )
        print(json.dumps(completion, indent=1, sort_keys=True))
        return

    # Re-derive all selector decisions and payload headers before spending the
    # I/O needed to hash the roughly terabyte-scale pool.
    pool = load_mixed_candidate_pool(
        root,
        validate_payload_headers=True,
        require_completion=False,
    )
    if pool.trellis_schema is None:
        raise ValueError("candidate pool has no logical trellis schema")
    completion = candidate_pool_completion_document(
        root,
        jobs=args.jobs,
        logical_trellis_schema=pool.trellis_schema,
    )
    path = write_candidate_pool_completion(root, completion)
    validate_candidate_pool_completion(root, verify_hashes=False)
    total_bytes = sum(
        int(component["size"])
        for layer in completion["layers"].values()
        for component in layer.values()
    )
    print(
        f"sealed {path}: {total_bytes} bytes, "
        f"logical_schema={completion['logical_trellis_schema']}, "
        f"content_sha256={completion['content_sha256']}"
    )


if __name__ == "__main__":
    main()
