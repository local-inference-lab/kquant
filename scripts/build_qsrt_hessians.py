#!/usr/bin/env python3
"""Build reusable QSRT H13 matrices and an explicit expert-local H2 prior."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kquant.capture import build_hessians


def _parse_layers(value: str) -> list[int]:
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or any(not 1 <= layer <= 92 for layer in layers):
        raise argparse.ArgumentTypeError("layers must be comma-separated decoder layers 1..92")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=_parse_layers)
    parser.add_argument("--sample-split", choices=("train", "validation", "all"), default="train")
    parser.add_argument("--request-step-min", type=int, default=1)
    parser.add_argument("--request-step-max", type=int)
    parser.add_argument("--min-rows", type=int, default=1)
    args = parser.parse_args()

    started = time.perf_counter()
    output = build_hessians(
        args.capture,
        args.output,
        layers=None if args.layers is None else [layer - 1 for layer in args.layers],
        device=args.device,
        min_rows=args.min_rows,
        sample_split=args.sample_split,
        request_step_min=args.request_step_min,
        request_step_max=args.request_step_max,
        h2_policy="identity_prior",
    )
    print(f"built {output} in {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
