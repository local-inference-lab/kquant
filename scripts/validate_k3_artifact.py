#!/usr/bin/env python
"""Validate the complete structural contract of a Kimi-K3 EXL3 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.artifact import validate_exl3_artifact
from kquant.correctness import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--serve-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate_exl3_artifact(args.artifact, args.serve_dir)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
