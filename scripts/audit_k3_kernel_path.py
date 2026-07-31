#!/usr/bin/env python
"""Audit a K3 serving log for the requested MoE kernel path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.correctness import write_json
from kquant.kernel_audit import KERNEL_PATHS, audit_kernel_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--require", required=True, choices=KERNEL_PATHS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_kernel_path(
        args.log.read_text(errors="replace"),
        args.require,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
