"""Multi-GPU EXL3-3.0 pack driver (12 workers by default).

Parent computes the keep-set (top experts by L0 traffic proxy; kept experts
stay MXFP4 and are skipped here), shards layers across one worker per GPU,
and tracks progress via completed per-layer shards. Workers are independent
processes (CUDA_VISIBLE_DEVICES pinned) writing
  <dest>/exl3-layer-XXXXX.safetensors  (+ .errs.json)
Resume-safe TP-local builds use contract-bound per-layer completion markers;
unmarked or truncated shards can be atomically replaced.

Usage:
  parent:  python scripts/pack_exl3_12gpu.py --dest DIR [--keep-frac 0.152]
  worker:  python scripts/pack_exl3_12gpu.py --worker N --dest DIR (internal)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

NUM_GPUS = 12
TP_LOCAL_BUILD_VERSION = 1


def completion_marker(dest: Path, layer: int, build_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in build_id)
    if not safe:
        raise ValueError("build_id must contain at least one safe character")
    return dest / f".exl3-layer-{layer:05d}.{safe}.complete.json"


def _atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True))
    tmp.replace(path)


def marker_contract(
    *, layer: int, build_id: str, tp_size: int, shard_size: int, errs_size: int
) -> dict:
    local_intermediate = validate_tp_size(tp_size)
    return {
        "version": TP_LOCAL_BUILD_VERSION,
        "build_id": build_id,
        "layer": int(layer),
        "tp_size": int(tp_size),
        "local_intermediate_size": local_intermediate,
        "intermediate_hadamard_blocks": (
            [128, 64] if local_intermediate % 128 else [128]
        ),
        "bits": 3,
        "codebook": "mcg",
        "shared_su": True,
        "shard_size": int(shard_size),
        "errs_size": int(errs_size),
    }


def layer_is_complete(dest: Path, layer: int, build_id: str, tp_size: int) -> bool:
    shard = dest / f"exl3-layer-{layer:05d}.safetensors"
    errs = dest / f"exl3-layer-{layer:05d}.errs.json"
    marker = completion_marker(dest, layer, build_id)
    if not shard.is_file() or not errs.is_file() or not marker.is_file():
        return False
    try:
        actual = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected = marker_contract(
        layer=layer,
        build_id=build_id,
        tp_size=tp_size,
        shard_size=shard.stat().st_size,
        errs_size=errs.stat().st_size,
    )
    return actual == expected


def validate_tp_size(tp_size: int) -> int:
    tp_size = int(tp_size)
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if 3072 % tp_size:
        raise ValueError(f"Kimi-K3 intermediate 3072 is not divisible by TP{tp_size}")
    local_intermediate = 3072 // tp_size
    if local_intermediate % 128 not in (0, 64):
        raise ValueError(
            f"TP{tp_size} local intermediate {local_intermediate} is not "
            "H128/H64 compatible"
        )
    return local_intermediate


def compute_allocation(dest: Path, keep_frac: float) -> dict:
    import numpy as np

    from kquant import constants as C
    from kquant.dynstats import ComposedStats, load_bundle
    from kquant.rank.l0 import traffic_proxy

    st = load_bundle("out/static.kqstats")
    traffic = traffic_proxy(ComposedStats([st]), "bias")  # [92, 896]
    flat = traffic.flatten()
    n_keep = int(round(keep_frac * flat.size))
    keep_idx = np.argpartition(flat, -n_keep)[-n_keep:]
    keep_mask = np.zeros(flat.size, dtype=bool)
    keep_mask[keep_idx] = True
    keep_mask = keep_mask.reshape(traffic.shape)
    alloc = {}
    for row in range(C.NUM_MOE_LAYERS):
        layer = row + 1
        keeps = np.nonzero(keep_mask[row])[0].tolist()
        demoted = np.nonzero(~keep_mask[row])[0].tolist()
        alloc[str(layer)] = {"keep": keeps, "exl3": demoted}
    meta = {
        "keep_frac": keep_frac,
        "n_keep": int(keep_mask.sum()),
        "n_exl3": int((~keep_mask).sum()),
        "traffic": "bias-proxy (L0); re-rank on measured L1 when available",
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "allocation-exl3.json").write_text(
        json.dumps({"meta": meta, "layers": alloc})
    )
    return alloc


def worker(
    worker_id: int,
    dest: Path,
    layers: list[int] | None = None,
    tp_size: int = 1,
    temp_batch_size: int | None = None,
    replace_unmarked: bool = False,
    build_id: str | None = None,
    num_workers: int = NUM_GPUS,
) -> None:
    import torch

    from kquant.io.hf_cache import resolve
    from kquant.pack.exl3 import make_shared_h, quantize_layer, write_layer_shard

    alloc = json.loads((dest / "allocation-exl3.json").read_text())["layers"]
    if tp_size != 1 and not build_id:
        raise ValueError("TP-local quantization requires an explicit --build-id")
    if tp_size != 1 and os.environ.get("KQUANT_EXL3_SHARED_SU") != "1":
        raise ValueError(
            "TP-local K3 EXL3 requires KQUANT_EXL3_SHARED_SU=1 for the "
            "one-grid runtime contract"
        )
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= worker_id < num_workers:
        raise ValueError(
            f"worker_id must be in [0, {num_workers - 1}], got {worker_id}"
        )
    if layers is None:
        layers = [
            int(layer)
            for i, layer in enumerate(sorted(alloc, key=int))
            if i % num_workers == worker_id
        ]
    cache = resolve()
    dev = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES pins the physical GPU
    local_intermediate = validate_tp_size(tp_size)
    # w1/w3 use K=3584 and TP-local w2 uses K=3072/tp_size.  Avoid allocating
    # the unused global 3072x3072 identity on TP-local low-free-VRAM rebuilds.
    shared_h = {k: make_shared_h(k, dev) for k in {3584, local_intermediate}}
    for layer in layers:
        shard = dest / f"exl3-layer-{layer:05d}.safetensors"
        marker = completion_marker(dest, layer, build_id) if build_id else None
        complete = bool(build_id and layer_is_complete(dest, layer, build_id, tp_size))
        if shard.exists() and (not replace_unmarked or complete):
            print(f"[w{worker_id}] layer {layer}: exists, skip", flush=True)
            continue
        demoted = alloc[str(layer)]["exl3"]
        t0 = time.time()
        tensors, errs = quantize_layer(
            cache,
            layer,
            demoted,
            dev,
            shared_h,
            tp_size=tp_size,
            temp_batch_size=temp_batch_size,
        )
        write_layer_shard(dest, layer, tensors)
        errs_path = dest / f"exl3-layer-{layer:05d}.errs.json"
        _atomic_write_json(errs_path, errs)
        if marker is not None:
            _atomic_write_json(
                marker,
                marker_contract(
                    layer=layer,
                    build_id=build_id,
                    tp_size=tp_size,
                    shard_size=shard.stat().st_size,
                    errs_size=errs_path.stat().st_size,
                ),
            )
        vals = list(errs.values())
        proxy_range = (
            f"{min(vals):.4f}-{max(vals):.4f}" if vals else "no-demoted-experts"
        )
        print(
            f"[w{worker_id}] layer {layer}: {len(demoted)} experts in "
            f"{time.time() - t0:.0f}s, proxy {proxy_range}",
            flush=True,
        )
    print(f"[w{worker_id}] DONE", flush=True)


def parent(
    dest: Path,
    keep_frac: float,
    tp_size: int,
    *,
    temp_batch_size: int | None,
    replace_unmarked: bool,
    reuse_allocation: bool,
    build_id: str | None,
    num_workers: int,
) -> None:
    from kquant.pack.exl3 import write_manifest

    local_intermediate = validate_tp_size(tp_size)
    if temp_batch_size is not None and temp_batch_size <= 0:
        raise ValueError("--temp-batch-size must be positive")
    if num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    if tp_size != 1 and os.environ.get("KQUANT_EXL3_SHARED_SU") != "1":
        raise ValueError(
            "TP-local K3 EXL3 requires KQUANT_EXL3_SHARED_SU=1 for the "
            "one-grid runtime contract"
        )
    if reuse_allocation:
        allocation_path = dest / "allocation-exl3.json"
        if not allocation_path.is_file():
            raise FileNotFoundError(f"--reuse-allocation requires {allocation_path}")
        allocation_document = json.loads(allocation_path.read_text())
        alloc = allocation_document["layers"]
        existing_keep_frac = allocation_document.get("meta", {}).get("keep_frac")
        if existing_keep_frac is None:
            raise ValueError("reused allocation has no meta.keep_frac")
        keep_frac = float(existing_keep_frac)
    else:
        alloc = compute_allocation(dest, keep_frac)
    n_exl3 = sum(len(v["exl3"]) for v in alloc.values())
    print(
        f"allocation: {n_exl3} demoted experts to quantize, "
        f"{sum(len(v['keep']) for v in alloc.values())} MXFP4 keeps",
        flush=True,
    )
    manifest = {"keep_frac": keep_frac}
    if tp_size != 1:
        if not build_id:
            raise ValueError("TP-local quantization requires an explicit --build-id")
        manifest.update(
            {
                "schema_version": 3,
                "compatible_tp_sizes": [tp_size],
                "tp_local_quantization": True,
                "tp_local_intermediate_size": local_intermediate,
                "intermediate_hadamard_blocks": [128, local_intermediate % 128]
                if local_intermediate % 128
                else [128],
                "build_id": build_id,
                "build_complete": False,
                "build_version": TP_LOCAL_BUILD_VERSION,
                "shared_su": True,
            }
        )
    write_manifest(dest, manifest)
    procs = []
    for i in range(num_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        worker_args = [
            sys.executable,
            __file__,
            "--worker",
            str(i),
            "--dest",
            str(dest),
            "--tp-size",
            str(tp_size),
            "--num-workers",
            str(num_workers),
        ]
        if temp_batch_size is not None:
            worker_args.extend(["--temp-batch-size", str(temp_batch_size)])
        if replace_unmarked:
            worker_args.append("--replace-unmarked")
        if build_id is not None:
            worker_args.extend(["--build-id", build_id])
        p = subprocess.Popen(
            worker_args,
            env=env,
        )
        procs.append(p)
    rc = 0
    for p in procs:
        rc |= p.wait()
    if tp_size != 1 and build_id is not None:
        missing_markers = [
            layer
            for layer in sorted(alloc, key=int)
            if not layer_is_complete(dest, int(layer), build_id, tp_size)
        ]
        if missing_markers:
            print(
                f"incomplete build: {len(missing_markers)} layer markers "
                f"missing; sample={missing_markers[:12]}",
                flush=True,
            )
            rc = rc or 1
        if rc == 0:
            manifest["build_complete"] = True
            write_manifest(dest, manifest)
    sys.exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--keep-frac", type=float, default=0.152)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help="comma-separated explicit layer list (overrides modulo)",
    )
    ap.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="quantize routed experts as rank-local shards for this TP size",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=NUM_GPUS,
        help="number of GPU-pinned workers (use 16 on the TP16 host)",
    )
    ap.add_argument(
        "--temp-batch-size",
        type=int,
        default=None,
        help="cap EXL3 tile scratch batches for low-free-VRAM rebuilds",
    )
    ap.add_argument(
        "--replace-unmarked",
        action="store_true",
        help="atomically replace existing layer shards lacking this build marker",
    )
    ap.add_argument(
        "--reuse-allocation",
        action="store_true",
        help="preserve the destination's existing keep/EXL3 partition",
    )
    ap.add_argument(
        "--build-id",
        help="explicit identity for resumable TP-local layer completion markers",
    )
    args = ap.parse_args()
    if args.worker is None:
        parent(
            args.dest,
            args.keep_frac,
            args.tp_size,
            temp_batch_size=args.temp_batch_size,
            replace_unmarked=args.replace_unmarked,
            reuse_allocation=args.reuse_allocation,
            build_id=args.build_id,
            num_workers=args.num_workers,
        )
    else:
        layers = [int(x) for x in args.layers.split(",")] if args.layers else None
        worker(
            args.worker,
            args.dest,
            layers,
            args.tp_size,
            temp_batch_size=args.temp_batch_size,
            replace_unmarked=args.replace_unmarked,
            build_id=args.build_id,
            num_workers=args.num_workers,
        )
