#!/usr/bin/env python3
"""Measure duplicate-label state steering in MCG and projected-FP8 graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from kquant.trellis_window import (
    fp8_constrained_codebook,
    mcg_codebook,
    trellis_alias_metrics,
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-alias-structure.json"),
    )
    parser.add_argument("--window-bits", default="12,14,16")
    args = parser.parse_args()
    windows = tuple(int(value) for value in args.window_bits.split(","))

    payload: dict[str, object] = {
        "kind": "kquant_trellis_alias_analysis",
        "schema_version": 1,
        "interpretation": (
            "Outgoing aliases emit the same current numeric value while choosing "
            "different successor states. Conditional alias bits assume uniformly "
            "used outgoing branches and are a structural ceiling, not measured "
            "source entropy."
        ),
        "windows": {},
    }
    for window_bits in windows:
        mcg = mcg_codebook(window_bits, device="cpu")
        e4m3 = fp8_constrained_codebook(mcg, dtype=torch.float8_e4m3fn)
        by_rate: dict[str, object] = {}
        for bits in (2, 3, 4):
            by_rate[str(bits)] = {
                "fp16_mcg": trellis_alias_metrics(mcg, bits, window_bits),
                "e4m3_projected_mcg": trellis_alias_metrics(e4m3, bits, window_bits),
            }
        payload["windows"][str(window_bits)] = {
            "fp16_numeric_labels": int(torch.unique(mcg).numel()),
            "e4m3_numeric_labels": int(torch.unique(e4m3).numel()),
            "rates": by_rate,
        }
    _atomic_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
