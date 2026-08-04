"""12-GPU EXL3-3.0 pack driver.

In legacy mode the parent computes the keep-set first and skips kept experts.
In candidate-pool mode every expert is encoded and scored before allocation,
so later keep-set changes only filter/repackage candidates. Workers are
independent processes (CUDA_VISIBLE_DEVICES pinned) writing
  <dest>/exl3-layer-XXXXX.safetensors  (+ .errs.json)
Resume-safe: a layer is skipped only when its complete output set exists.

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


def _layer_outputs_complete(output_dir: Path, layer: int, candidate_pool: bool) -> bool:
    stem = f"exl3-layer-{layer:05d}"
    required = [
        output_dir / f"{stem}.safetensors",
        output_dir / f"{stem}.errs.json",
    ]
    if candidate_pool:
        required.append(output_dir / f"{stem}.metrics.safetensors")
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def compute_allocation(
    dest: Path,
    keep_frac: float,
    *,
    stats_path: Path = Path("out/static.kqstats"),
    traffic_mode: str = "bias",
) -> dict:
    import numpy as np

    from kquant import constants as C
    from kquant.dynstats import ComposedStats, load_bundle
    from kquant.rank.l0 import traffic_proxy

    st = load_bundle(stats_path)
    traffic = traffic_proxy(ComposedStats([st]), traffic_mode)  # [92, 896]
    flat = traffic.flatten()
    n_keep = round(keep_frac * flat.size)
    keep_mask = np.zeros(flat.size, dtype=bool)
    if n_keep:
        keep_idx = np.argpartition(flat, -n_keep)[-n_keep:]
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
        "traffic": traffic_mode,
        "stats": str(stats_path.resolve()),
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
    *,
    candidate_pool: bool = False,
    hessians: Path | None = None,
) -> None:
    import torch

    from kquant import constants as C
    from kquant.capture import load_layer_hessians
    from kquant.io.hf_cache import resolve
    from kquant.pack.candidates import write_candidate_metrics
    from kquant.pack.exl3 import make_shared_h, quantize_layer, write_layer_shard

    alloc = None
    if not candidate_pool:
        alloc = json.loads((dest / "allocation-exl3.json").read_text())["layers"]
    output_dir = dest / "candidates" if candidate_pool else dest
    output_dir.mkdir(parents=True, exist_ok=True)
    if layers is None:
        layers = [
            int(layer)
            for i, layer in enumerate(C.MOE_LAYERS)
            if i % NUM_GPUS == worker_id
        ]
    cache = resolve()
    dev = torch.device("cuda", 0)  # CUDA_VISIBLE_DEVICES pins the physical GPU
    for layer in layers:
        if _layer_outputs_complete(output_dir, layer, candidate_pool):
            print(f"[w{worker_id}] layer {layer}: complete, skip", flush=True)
            continue
        experts = (
            list(range(C.NUM_EXPERTS)) if candidate_pool else alloc[str(layer)]["exl3"]
        )
        if hessians is None:
            h13 = h2 = None
        else:
            h13, h2 = load_layer_hessians(hessians, layer)
        # exllama finalizes/mutates these dictionaries.  They may be shared by
        # every expert and batch in this layer, but never by another layer.
        shared_h = {
            C.LATENT: make_shared_h(C.LATENT, dev, h13),
            C.EXPERT_INTER: make_shared_h(C.EXPERT_INTER, dev, h2),
        }
        t0 = time.time()
        tensors, errs, residuals = quantize_layer(cache, layer, experts, dev, shared_h)
        write_layer_shard(output_dir, layer, tensors)
        _atomic_json(
            output_dir / f"exl3-layer-{layer:05d}.errs.json",
            errs,
        )
        if candidate_pool:
            write_candidate_metrics(
                output_dir / f"exl3-layer-{layer:05d}.metrics.safetensors",
                layer=layer,
                errors=errs,
                residuals=residuals,
                experts=experts,
            )
        vals = [
            float(value["proxy"]) if isinstance(value, dict) else float(value)
            for value in errs.values()
        ]
        print(
            f"[w{worker_id}] layer {layer}: {len(experts)} experts in "
            f"{time.time() - t0:.0f}s, proxy {min(vals):.4f}-{max(vals):.4f}",
            flush=True,
        )
    print(f"[w{worker_id}] DONE", flush=True)


def parent(
    dest: Path,
    keep_frac: float,
    *,
    candidate_pool: bool,
    hessians: Path | None,
    stats_path: Path,
    traffic_mode: str,
) -> None:
    from kquant import constants as C
    from kquant.pack.exl3 import SHARED_SU, write_manifest

    dest.mkdir(parents=True, exist_ok=True)
    if candidate_pool:
        alloc = None
        print(
            f"candidate pool: quantizing all {92 * 896:,} layer-experts before allocation",
            flush=True,
        )
    else:
        alloc = compute_allocation(
            dest,
            keep_frac,
            stats_path=stats_path,
            traffic_mode=traffic_mode,
        )
        n_exl3 = sum(len(v["exl3"]) for v in alloc.values())
        print(
            f"allocation: {n_exl3} demoted experts to quantize, "
            f"{sum(len(v['keep']) for v in alloc.values())} MXFP4 keeps",
            flush=True,
        )
    manifest_meta = {
        "source_model": C.MODEL_ID,
        "source_revision": C.REVISION,
        "keep_frac": float(keep_frac),
        "hessian": "identity" if hessians is None else str(hessians.resolve()),
        "candidate_pool": bool(candidate_pool),
        "stats": str(stats_path.resolve()),
        "shared_su": bool(SHARED_SU),
        "quantizer_contract": "exl3-canonical-residual-v1",
    }
    manifest_path = dest / "kquant_exl3_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        incompatible = {
            key: (existing.get(key), value)
            for key, value in manifest_meta.items()
            if existing.get(key) != value
        }
        if incompatible:
            raise SystemExit(
                f"destination {dest} contains an incompatible EXL3 run: "
                f"{incompatible}; use a fresh destination"
            )
    write_manifest(dest, manifest_meta)
    procs = []
    for i in range(NUM_GPUS):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        command = [
            sys.executable,
            __file__,
            "--worker",
            str(i),
            "--dest",
            str(dest),
            "--stats",
            str(stats_path),
        ]
        if candidate_pool:
            command.append("--candidate-pool")
        if hessians is not None:
            command.extend(("--hessians", str(hessians)))
        p = subprocess.Popen(
            command,
            env=env,
        )
        procs.append(p)
    rc = 0
    for p in procs:
        rc |= p.wait()
    if rc:
        sys.exit(rc)
    if candidate_pool:
        from kquant.pack.candidates import (
            choose_allocation,
            materialize_allocation,
            score_candidates,
            write_candidate_allocation,
        )

        candidate_dir = dest / "candidates"
        scores = score_candidates(candidate_dir, stats_path)
        alloc = choose_allocation(scores, keep_frac)
        write_candidate_allocation(
            dest,
            alloc,
            keep_frac=keep_frac,
            stats_path=stats_path,
            candidate_dir=candidate_dir,
        )
        materialize_allocation(candidate_dir, dest, alloc)
        print(
            "candidate allocation materialized; future allocations can reuse "
            f"{candidate_dir} without quantizing again",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--keep-frac", type=float, default=0.152)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument(
        "--candidate-pool",
        action="store_true",
        help="quantize all experts before allocation and retain reusable candidates",
    )
    ap.add_argument("--hessians", type=Path, default=None, help="input .kqhess")
    ap.add_argument("--stats", type=Path, default=Path("out/static.kqstats"))
    ap.add_argument(
        "--traffic",
        choices=("bias", "measured"),
        default="bias",
        help="legacy pre-allocation traffic mode (candidate pools always use measured)",
    )
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help="comma-separated explicit layer list (overrides modulo)",
    )
    args = ap.parse_args()
    if args.worker is None:
        if args.candidate_pool and args.traffic != "measured":
            args.traffic = "measured"
        parent(
            args.dest,
            args.keep_frac,
            candidate_pool=args.candidate_pool,
            hessians=args.hessians,
            stats_path=args.stats,
            traffic_mode=args.traffic,
        )
    else:
        layers = [int(x) for x in args.layers.split(",")] if args.layers else None
        worker(
            args.worker,
            args.dest,
            layers,
            candidate_pool=args.candidate_pool,
            hessians=args.hessians,
        )
