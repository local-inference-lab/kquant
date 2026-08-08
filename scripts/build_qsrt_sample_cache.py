#!/usr/bin/env python3
"""Repack routed capture rows into directly addressable per-layer samples."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kquant.capture import build_layer_sample_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    output = build_layer_sample_cache(args.capture, args.output)
    print(f"built {output} in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
