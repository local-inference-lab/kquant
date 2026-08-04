#!/usr/bin/env python3
"""Build the resumable all-expert exact-byte index for SQRT-C X4."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from kquant import constants as C
from kquant.pack.x4_index import (
    X4_COST_COMPLETION_FILENAME,
    build_x4_cost_layer,
    finalize_x4_cost_index,
    prepare_x4_cost_index,
    write_x4_cost_layer,
    x4_cost_layer_filename,
)
from kquant.source_weights import OfficialMXFP4Store


def _parse_layers(value: str) -> tuple[int, ...]:
    if value == "all":
        return tuple(C.MOE_LAYERS)
    result: list[int] = []
    for item in value.split(","):
        if "-" in item:
            begin, end = map(int, item.split("-", 1))
            result.extend(range(begin, end + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("layers must be nonempty and unique")
    if any(layer not in C.MOE_LAYERS for layer in result):
        raise argparse.ArgumentTypeError("layers must lie in 1..92")
    return tuple(result)


def _worker(
    layer: int,
    destination: str,
    official_repo_dir: str | None,
    official_revision: str,
) -> tuple[int, int]:
    store = OfficialMXFP4Store(
        repo_dir=official_repo_dir,
        revision=official_revision,
    )
    document = build_x4_cost_layer(store, layer)
    path = write_x4_cost_layer(destination, document)
    return layer, path.stat().st_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--layers", type=_parse_layers, default=tuple(C.MOE_LAYERS))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error("--jobs must lie in 1..32")
    return args


def main() -> None:
    args = parse_args()
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    destination = prepare_x4_cost_index(
        args.dest,
        store=store,
        resume=args.resume,
    )
    pending = [
        layer
        for layer in args.layers
        if not (destination / x4_cost_layer_filename(layer)).is_file()
    ]
    started = time.time()
    repo_dir = (
        None
        if args.official_repo_dir is None
        else str(args.official_repo_dir.resolve())
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _worker,
                layer,
                str(destination),
                repo_dir,
                store.revision,
            ): layer
            for layer in pending
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            layer, json_bytes = future.result()
            print(
                f"layer {layer}: indexed {C.NUM_EXPERTS} "
                f"experts ({json_bytes} JSON bytes); {count}/{len(pending)} new "
                f"layers in {time.time() - started:.1f}s",
                flush=True,
            )
    missing = [
        layer
        for layer in C.MOE_LAYERS
        if not (destination / x4_cost_layer_filename(layer)).is_file()
    ]
    if missing:
        print(f"partial X4 cost index: {len(missing)} layers remain", flush=True)
        return
    completion = finalize_x4_cost_index(destination)
    print(
        json.dumps(
            {
                "completion": str(destination / X4_COST_COMPLETION_FILENAME),
                "content_sha256": completion["content_sha256"],
                "all_x4_container_bytes": completion[
                    "all_x4_container_bytes"
                ],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
