#!/usr/bin/env python3
"""Package a complete SQRT-C TP12 artifact into a fresh vLLM directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.pack.mixed_package import DEFAULT_NONEXPERT
from kquant.pack.sqrt_c_package import package_sqrt_c_tp12_serve_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--nonexpert-source", type=Path, default=DEFAULT_NONEXPERT)
    parser.add_argument("--official-snapshot", type=Path)
    args = parser.parse_args()
    result = package_sqrt_c_tp12_serve_directory(
        args.artifact,
        args.destination,
        nonexpert_source=args.nonexpert_source,
        official_snapshot=args.official_snapshot,
    )
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
