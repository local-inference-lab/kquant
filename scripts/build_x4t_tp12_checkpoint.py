#!/usr/bin/env python3
"""Compile canonical QSRT X4T sidecars into persistent TP12 rank shards."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kquant import constants as C
from kquant.pack.package_helpers import _atomic_json
from kquant.pack.qsrt_materialize import QSRT_ALLOCATION_COPY_FILENAME
from kquant.pack.qsrt_package import qsrt_hybrid_bit_map
from kquant.pack.x4t_tp12 import (
    X4T_TP12_MANIFEST,
    build_x4t_tp12_layer,
    load_x4t_tp12_manifest,
    write_x4t_tp12_manifest,
)


def _parse_layers(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            begin, end = map(int, item.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be unique integers/ranges")
    if any(layer not in C.MOE_LAYERS for layer in result):
        raise argparse.ArgumentTypeError("layers must lie in 1..92")
    return tuple(result)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _receipt_name(layer: int) -> str:
    return f"x4t-tp12-layer-{layer:05d}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--layers", type=_parse_layers)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    artifact = args.artifact.resolve(strict=True)
    destination = args.destination.resolve()
    if destination.exists():
        if not args.resume:
            raise FileExistsError(destination)
    else:
        destination.mkdir(parents=True)
    if (destination / X4T_TP12_MANIFEST).is_file():
        manifest = load_x4t_tp12_manifest(destination)
        print(json.dumps(manifest, indent=1, sort_keys=True))
        return

    allocation = _read_json(artifact / QSRT_ALLOCATION_COPY_FILENAME)
    qsrt_hybrid_bit_map(allocation)
    selected = C.MOE_LAYERS if args.layers is None else args.layers
    started = time.perf_counter()
    for index, layer in enumerate(selected, start=1):
        expert_ids = tuple(int(value) for value in allocation["layers"][str(layer)]["x4t"])
        rank_reports = build_x4t_tp12_layer(
            artifact,
            destination,
            layer=layer,
            expert_ids=expert_ids,
            resume=args.resume,
        )
        receipt = {
            "layer": layer,
            "expert_ids": list(expert_ids),
            "ranks": rank_reports,
            "total_bytes": sum(int(item["bytes"]) for item in rank_reports),
        }
        _atomic_json(destination / _receipt_name(layer), receipt)
        elapsed = time.perf_counter() - started
        print(
            f"layer {layer}: {len(expert_ids)} X4T experts, "
            f"{receipt['total_bytes']} bytes across 12 ranks; "
            f"{index}/{len(selected)} in {elapsed:.1f}s",
            flush=True,
        )

    layers: dict[str, dict] = {}
    for layer in C.MOE_LAYERS:
        path = destination / _receipt_name(layer)
        if not path.is_file():
            print(
                f"partial X4T TP12 checkpoint: missing layer {layer}; "
                "no completion manifest written",
                flush=True,
            )
            return
        receipt = _read_json(path)
        expected_ids = allocation["layers"][str(layer)]["x4t"]
        if receipt.get("layer") != layer or receipt.get("expert_ids") != expected_ids:
            raise ValueError(f"X4T TP12 layer {layer} receipt disagrees with allocation")
        layers[str(layer)] = receipt

    manifest = write_x4t_tp12_manifest(
        destination,
        artifact=artifact,
        layers=layers,
    )
    print(
        f"complete: {destination / X4T_TP12_MANIFEST} closes "
        f"{manifest['total_bytes']} bytes",
        flush=True,
    )


if __name__ == "__main__":
    main()
