#!/usr/bin/env python
"""Build a larger-MXFP4 EXL3 artifact without re-encoding Trellis weights.

The new demoted set must be a subset of an existing EXL3 artifact. Layer
shards are rewritten with only that subset, existing keep shards are
hard-linked, and newly promoted experts are copied from the official source.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant import constants as C
from kquant.dynstats import ComposedStats, load_bundle
from kquant.io.hf_cache import resolve
from kquant.io.stream import load_tensor
from kquant.rank.l0 import traffic_proxy

EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
SHARD_BYTES = 8 << 30


def compute_allocation(keep_frac: float) -> dict[str, dict[str, list[int]]]:
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in the interval (0, 1]")
    stats_path = Path(__file__).resolve().parents[1] / "out/static.kqstats"
    stats = load_bundle(stats_path)
    traffic = traffic_proxy(ComposedStats([stats]), "bias")
    flat = traffic.flatten()
    n_keep = int(round(keep_frac * flat.size))
    keep_idx = np.argpartition(flat, -n_keep)[-n_keep:]
    keep_mask = np.zeros(flat.size, dtype=bool)
    keep_mask[keep_idx] = True
    keep_mask = keep_mask.reshape(traffic.shape)
    return {
        str(row + 1): {
            "keep": np.nonzero(keep_mask[row])[0].tolist(),
            "exl3": np.nonzero(~keep_mask[row])[0].tolist(),
        }
        for row in range(C.NUM_MOE_LAYERS)
    }


def _repack_layer(
    source: str,
    destination: str,
    layer: int,
    demoted: list[int],
) -> tuple[int, int, int]:
    source_path = Path(source) / f"exl3-layer-{layer:05d}.safetensors"
    destination_path = Path(destination) / source_path.name
    allowed = set(demoted)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(source_path), framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            match = EXPERT_RE.search(name)
            if match is not None and int(match.group(1)) in allowed:
                tensors[name] = handle.get_tensor(name)
    save_file(tensors, str(destination_path))

    source_errors = Path(source) / f"exl3-layer-{layer:05d}.errs.json"
    if source_errors.exists():
        errors = json.loads(source_errors.read_text())
        filtered = {
            name: value
            for name, value in errors.items()
            if (match := EXPERT_RE.search(name)) is None
            or int(match.group(1)) in allowed
        }
        (Path(destination) / source_errors.name).write_text(json.dumps(filtered))
    return layer, len(tensors), destination_path.stat().st_size


def repack_exl3_layers(
    base: Path,
    destination: Path,
    allocation: dict[str, dict[str, list[int]]],
    jobs: int,
) -> None:
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _repack_layer,
                str(base),
                str(destination),
                int(layer),
                allocation[layer]["exl3"],
            )
            for layer in sorted(allocation, key=int)
        ]
        for future in as_completed(futures):
            layer, tensors, size = future.result()
            print(
                f"layer {layer:02d}: {tensors} tensors, {size / 2**30:.2f} GiB",
                flush=True,
            )


def link_existing_keeps(base: Path, destination: Path) -> None:
    for source in sorted(base.glob("keep-mxfp4-*.safetensors")):
        os.link(source, destination / source.name)


def extract_promoted_keeps(
    base_allocation: dict[str, dict[str, list[int]]],
    allocation: dict[str, dict[str, list[int]]],
    destination: Path,
) -> int:
    cache = resolve()
    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    shard = 0
    total = 0

    def flush() -> None:
        nonlocal buffer, buffer_bytes, shard
        if not buffer:
            return
        path = destination / f"keep-mxfp4-promoted-{shard:05d}.safetensors"
        save_file(buffer, str(path))
        print(f"{path.name}: {buffer_bytes / 2**30:.2f} GiB", flush=True)
        buffer = {}
        buffer_bytes = 0
        shard += 1

    for layer in sorted(allocation, key=int):
        old_keep = set(base_allocation[layer]["keep"])
        promoted = sorted(set(allocation[layer]["keep"]) - old_keep)
        for expert in promoted:
            for matrix in ("w1", "w3", "w2"):
                base_name = (
                    f"language_model.model.layers.{layer}.block_sparse_moe."
                    f"experts.{expert}.{matrix}"
                )
                for suffix in (".weight_packed", ".weight_scale"):
                    name = base_name + suffix
                    tensor = load_tensor(cache, name)
                    buffer[name] = tensor
                    size = tensor.numel() * tensor.element_size()
                    buffer_bytes += size
                    total += size
            if buffer_bytes >= SHARD_BYTES:
                flush()
    flush()
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--keep-frac", type=float, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dest.exists() and any(args.dest.iterdir()):
        raise FileExistsError(f"destination is not empty: {args.dest}")
    args.dest.mkdir(parents=True, exist_ok=True)

    base_document = json.loads((args.base / "allocation-exl3.json").read_text())
    base_allocation = base_document["layers"]
    allocation = compute_allocation(args.keep_frac)
    for layer in allocation:
        old_keep = set(base_allocation[layer]["keep"])
        new_keep = set(allocation[layer]["keep"])
        if not old_keep <= new_keep:
            raise ValueError(
                f"layer {layer}: new keep set does not contain the base keep set"
            )

    n_keep = sum(len(value["keep"]) for value in allocation.values())
    n_exl3 = sum(len(value["exl3"]) for value in allocation.values())
    allocation_document = {
        "meta": {
            "keep_frac": args.keep_frac,
            "n_keep": n_keep,
            "n_exl3": n_exl3,
            "traffic": "bias-proxy (L0); same ranking as base artifact",
            "base_artifact": str(args.base.resolve()),
        },
        "layers": allocation,
    }
    (args.dest / "allocation-exl3.json").write_text(
        json.dumps(allocation_document)
    )
    manifest = json.loads((args.base / "kquant_exl3_manifest.json").read_text())
    manifest.update(
        {
            "keep_frac": args.keep_frac,
            "repacked_from": str(args.base.resolve()),
        }
    )
    (args.dest / "kquant_exl3_manifest.json").write_text(json.dumps(manifest))

    print(f"allocation: {n_keep} MXFP4, {n_exl3} EXL3", flush=True)
    repack_exl3_layers(args.base, args.dest, allocation, args.jobs)
    link_existing_keeps(args.base, args.dest)
    promoted_bytes = extract_promoted_keeps(
        base_allocation, allocation, args.dest
    )
    print(
        f"promoted MXFP4 payload: {promoted_bytes / 2**30:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
