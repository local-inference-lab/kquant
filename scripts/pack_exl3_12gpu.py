"""12-GPU EXL3-3.0 pack driver.

Parent computes the keep-set (top experts by L0 traffic proxy; kept experts
stay MXFP4 and are skipped here), shards layers across one worker per GPU,
and tracks progress via completed per-layer shards. Workers are independent
processes (CUDA_VISIBLE_DEVICES pinned) writing
  <dest>/exl3-layer-XXXXX.safetensors  (+ .errs.json)
Resume-safe: existing shards are skipped.

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


def worker(worker_id: int, dest: Path) -> None:
    import torch

    from kquant.io.hf_cache import resolve
    from kquant.pack.exl3 import make_shared_h, quantize_layer, write_layer_shard

    alloc = json.loads((dest / "allocation-exl3.json").read_text())["layers"]
    layers = [
        int(layer) for i, layer in enumerate(sorted(alloc, key=int))
        if i % NUM_GPUS == worker_id
    ]
    cache = resolve()
    dev = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES pins the physical GPU
    shared_h = {k: make_shared_h(k, dev) for k in (3584, 3072)}
    for layer in layers:
        shard = dest / f"exl3-layer-{layer:05d}.safetensors"
        if shard.exists():
            print(f"[w{worker_id}] layer {layer}: exists, skip", flush=True)
            continue
        demoted = alloc[str(layer)]["exl3"]
        t0 = time.time()
        tensors, errs = quantize_layer(cache, layer, demoted, dev, shared_h)
        write_layer_shard(dest, layer, tensors)
        (dest / f"exl3-layer-{layer:05d}.errs.json").write_text(json.dumps(errs))
        vals = list(errs.values())
        print(
            f"[w{worker_id}] layer {layer}: {len(demoted)} experts in "
            f"{time.time()-t0:.0f}s, proxy {min(vals):.4f}-{max(vals):.4f}",
            flush=True,
        )
    print(f"[w{worker_id}] DONE", flush=True)


def parent(dest: Path, keep_frac: float) -> None:
    from kquant.pack.exl3 import write_manifest

    alloc = compute_allocation(dest, keep_frac)
    n_exl3 = sum(len(v["exl3"]) for v in alloc.values())
    print(f"allocation: {n_exl3} demoted experts to quantize, "
          f"{sum(len(v['keep']) for v in alloc.values())} MXFP4 keeps", flush=True)
    write_manifest(dest, {"keep_frac": keep_frac})
    procs = []
    for i in range(NUM_GPUS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        p = subprocess.Popen(
            [sys.executable, __file__, "--worker", str(i), "--dest", str(dest)],
            env=env,
        )
        procs.append(p)
    rc = 0
    for p in procs:
        rc |= p.wait()
    sys.exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--keep-frac", type=float, default=0.152)
    ap.add_argument("--worker", type=int, default=None)
    args = ap.parse_args()
    if args.worker is None:
        parent(args.dest, args.keep_frac)
    else:
        worker(args.worker, args.dest)
