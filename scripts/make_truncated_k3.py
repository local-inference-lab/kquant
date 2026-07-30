"""Build a truncated Kimi-K3 (first N decoder layers) for TP-geometry A/B.

Two variants:
  --source stock : tensors from the HF cache snapshot (BF16 attention,
                   stock MXFP4 experts), all-kept hybrid map.
  --source ours  : tensors from an EXL3 serve dir (baked MXFP8 attention,
                   keep+EXL3 experts, shared-su), its quant config sliced.

Keeps layers [0, N) plus embed/lm_head/final norms/attn_res/vision, rewrites
config for N layers, emits one merged safetensors index.

Usage:
  make_truncated_k3.py --source ours --src-dir /models/Kimi-K3-EXL3-3p09-serve \
      --layers 4 --dest /models/k3-trunc-ours
  make_truncated_k3.py --source stock --layers 4 --dest /models/k3-trunc-stock
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")
SHARD_BYTES = 12 << 30


def wanted(name: str, n_layers: int) -> bool:
    m = LAYER_RE.search(name)
    if m:
        return int(m.group(1)) < n_layers
    return True  # embed, lm_head, norms, attn_res, vision, mm


def iter_source(src_index: Path):
    base = src_index.parent
    wm = json.loads(src_index.read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, f in wm.items():
        by_file.setdefault(f, []).append(name)
    for f, names in sorted(by_file.items()):
        with safe_open(str(base / f), framework="pt") as sf:
            for name in names:
                yield name, sf.get_tensor(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("stock", "ours"), required=True)
    ap.add_argument("--src-dir", default="/models/Kimi-K3-EXL3-3p09-serve")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    if args.source == "stock":
        import sys

        sys.path.insert(0, "/home/luke/projects/kquant")
        from kquant.io.hf_cache import resolve

        snap = Path(resolve().snapshot_dir)
        src_index = snap / "model.safetensors.index.json"
        cfg_src = snap / "config.json"
        aux_src = snap
    else:
        src_index = Path(args.src_dir) / "model.safetensors.index.json"
        cfg_src = Path(args.src_dir) / "config.json"
        aux_src = Path(args.src_dir)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # --- tensors ---
    weight_map: dict[str, str] = {}
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    idx = 0
    total = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, idx
        if not buf:
            return
        fn = f"model-{idx:05d}.safetensors"
        save_file(buf, str(dest / fn))
        for k in buf:
            weight_map[k] = fn
        idx += 1
        buf, buf_bytes = {}, 0

    for name, t in iter_source(src_index):
        if not wanted(name, args.layers):
            continue
        buf[name] = t
        nb = t.numel() * t.element_size()
        buf_bytes += nb
        total += nb
        if buf_bytes >= SHARD_BYTES:
            flush()
    flush()
    (dest / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    print(f"tensors: {len(weight_map)} kept, {total/2**30:.1f} GiB")

    # --- config ---
    cfg = json.loads(cfg_src.read_text())
    tc = cfg.get("text_config", cfg)
    n = args.layers
    tc["num_hidden_layers"] = n
    la = tc["linear_attn_config"]
    la["kda_layers"] = [x for x in la["kda_layers"] if x <= n]
    la["full_attn_layers"] = [x for x in la["full_attn_layers"] if x <= n]

    moe_layers = [str(i) for i in range(1, n)]
    if args.source == "stock":
        tc["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "hybrid_bit_map": {L: [4] * 896 for L in moe_layers},
            "kept_format": "mxfp4_e8m0k32",
        }
    else:
        q = tc["quantization_config"]
        q["hybrid_bit_map"] = {
            L: q["hybrid_bit_map"][L] for L in moe_layers
        }
    cfg.pop("quantization_config", None) if "text_config" in cfg else None
    (dest / "config.json").write_text(json.dumps(cfg, indent=1))

    # --- tokenizer / aux ---
    for f in aux_src.iterdir():
        if f.suffix == ".safetensors" or f.name in (
            "config.json", "model.safetensors.index.json",
        ) or f.name.startswith(("exl3-layer", "keep-mxfp4", "00-nonexpert")):
            continue
        link = dest / f.name
        if not link.exists():
            if f.is_symlink() or f.is_file():
                shutil.copy(f.resolve(), link)
    print(f"done: {dest}")


if __name__ == "__main__":
    main()
