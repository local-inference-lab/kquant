#!/usr/bin/env python3
"""Package a complete QSRT TP12 artifact into a fresh vLLM directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.pack.package_helpers import DEFAULT_NONEXPERT
from kquant.pack.qsrt_package import package_qsrt_tp12_serve_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--x4t-tp12-source", type=Path, required=True)
    parser.add_argument("--nonexpert-source", type=Path, default=DEFAULT_NONEXPERT)
    parser.add_argument("--official-snapshot", type=Path)
    args = parser.parse_args()
    result = package_qsrt_tp12_serve_directory(
        args.artifact,
        args.destination,
        x4t_tp12_source=args.x4t_tp12_source,
        nonexpert_source=args.nonexpert_source,
        official_snapshot=args.official_snapshot,
    )
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
