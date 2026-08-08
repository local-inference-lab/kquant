#!/usr/bin/env python3
"""Authenticate and inventory the pinned Fruit annealed source checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.fruit_source import FRUIT_ANNEALED_SPEC, preflight_fruit_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = preflight_fruit_checkpoint(
        args.checkpoint,
        FRUIT_ANNEALED_SPEC,
        FRUIT_ANNEALED_SPEC.checkpoint_sha256,
    )
    print(
        json.dumps(
            evidence,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
