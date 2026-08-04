#!/usr/bin/env python3
"""Validate a packaged fixed-slab TP12 mixed-EXL serve directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.pack.mixed_package import validate_mixed_tp12_serve_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = validate_mixed_tp12_serve_directory(parse_args().serve_dir)
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
