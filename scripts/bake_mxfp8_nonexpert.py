"""Bake the online-MXFP8 overlay into the checkpoint (offline conversion).

Reads the Phase A non-expert side-shards and re-emits them with the overlay
target tensors quantized to MXFP8 (fp8 e4m3 values + per-32-input-column
e8m0 scales), so serving skips the per-boot online conversion. Non-target
tensors pass through byte-identical.

Semantics mirror vllm mxfp8_e4m3_quantize / dequant_mxfp8_to_bf16:
  scale[b] = e8m0(amax(block_b) / 448)   (power-of-2, ceil)
  q = rne_e4m3(x / 2^(scale-127))
Quantizing per checkpoint tensor commutes with vLLM's output-dim fusion
(q_a+kv_a) because scales are per-row blocks of input columns.

Usage: python scripts/bake_mxfp8_nonexpert.py [--src DIR] [--dest DIR] [--jobs N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/models/Kimi-K3-NF3R-Uniform-3p25-serve"
DEST = "/models/Kimi-K3-mxfp8-nonexpert"
BLOCK = 32
SHARD_BYTES = 8 << 30
EXPERT = re.compile(r"\.experts\.\d+\.(w1|w2|w3)\.")

# Mirrors the serving overlay: linear+shared_experts -> mxfp8, minus ignores.
TARGET = re.compile(
    r"language_model\.model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_a_proj"
    r"|q_a_layernorm)\.weight$"
    r"|(mlp|block_sparse_moe)\.shared_experts\.(gate_proj|up_proj|down_proj)"
    r"\.weight$"
    r"|mlp\.(gate_proj|up_proj|down_proj)\.weight$)"
)
EXCLUDE = re.compile(
    r"kv_b_proj|conv1d|\.b_proj|\.g_proj|f_a_proj|f_b_proj|lm_head"
    r"|attn_res|layernorm|norm\."
)


def is_target(name: str, shape: tuple) -> bool:
    if EXCLUDE.search(name):
        return False
    if not TARGET.search(name):
        return False
    return len(shape) == 2 and shape[-1] % BLOCK == 0


def mxfp8_quantize_cpu(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = w.to(torch.float32)
    out, inp = x.shape
    xb = x.view(out, inp // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=-1)
    # e8m0 scale: smallest power of two with amax/2^e <= 448 (e4m3 max)
    e = torch.ceil(torch.log2(amax.clamp(min=1e-30) / 448.0))
    e = e.clamp(min=-127, max=128)
    scale_u8 = (e + 127.0).to(torch.uint8)
    q = (xb / torch.exp2(e).unsqueeze(-1)).clamp(-448.0, 448.0)
    q8 = q.view(out, inp).to(torch.float8_e4m3fn)
    return q8, scale_u8


def process_shard(args: tuple) -> str:
    src_file, dest_dir = args
    out: dict[str, torch.Tensor] = {}
    n_baked = 0
    with safe_open(src_file, framework="pt") as sf:
        for name in sf.keys():
            t = sf.get_tensor(name)
            if is_target(name, tuple(t.shape)):
                q8, scale = mxfp8_quantize_cpu(t)
                out[name] = q8
                out[name.replace(".weight", ".weight_scale")] = scale
                n_baked += 1
            else:
                out[name] = t
    dest = Path(dest_dir) / Path(src_file).name
    save_file(out, str(dest))
    return f"{Path(src_file).name}: {n_baked} baked"


def process_source_checkpoint(dest_dir: str) -> None:
    """Extract and bake non-expert tensors directly from the HF checkpoint.

    This avoids materializing the roughly 135-GiB BF16 Phase-A side shards
    before converting their eligible 2-D linears to MXFP8.  The emitted file
    names intentionally match the overlay contract consumed by
    package_exl3_serve_dir.py.
    """
    from kquant.io.hf_cache import resolve

    cache = resolve()
    index_path = Path(cache.snapshot_dir) / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    nonexpert = [name for name in weight_map if not EXPERT.search(name)]
    by_shard: dict[str, list[str]] = {}
    for name in nonexpert:
        by_shard.setdefault(weight_map[name], []).append(name)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    out_idx = 0
    total_bytes = 0
    total_baked = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, out_idx
        if not buf:
            return
        out_path = dest / f"00-nonexpert-{out_idx:05d}.safetensors"
        save_file(buf, str(out_path))
        print(
            f"{out_path.name}: {buf_bytes / 2**30:.2f} GiB, "
            f"{len(buf)} tensors",
            flush=True,
        )
        out_idx += 1
        buf = {}
        buf_bytes = 0

    for shard_name, names in sorted(by_shard.items()):
        with safe_open(str(cache.shard_paths[shard_name]), framework="pt") as sf:
            for name in names:
                tensor = sf.get_tensor(name)
                emitted = {name: tensor}
                if is_target(name, tuple(tensor.shape)):
                    quantized, scale = mxfp8_quantize_cpu(tensor)
                    scale_name = name.replace(".weight", ".weight_scale")
                    if scale_name in weight_map:
                        raise ValueError(
                            f"generated MXFP8 scale collides with source tensor "
                            f"{scale_name}"
                        )
                    emitted = {name: quantized, scale_name: scale}
                    total_baked += 1
                for emitted_name, emitted_tensor in emitted.items():
                    if emitted_name in buf:
                        raise ValueError(f"duplicate output tensor {emitted_name}")
                    buf[emitted_name] = emitted_tensor
                    nbytes = emitted_tensor.numel() * emitted_tensor.element_size()
                    buf_bytes += nbytes
                    total_bytes += nbytes
                if buf_bytes >= SHARD_BYTES:
                    flush()
    flush()
    print(
        f"done: {len(nonexpert)} source tensors, {total_baked} MXFP8 linears, "
        f"{total_bytes / 2**30:.2f} GiB in {out_idx} shards",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument(
        "--from-source-checkpoint",
        action="store_true",
        help=(
            "extract non-expert tensors from the resolved official checkpoint "
            "and bake MXFP8 directly, without Phase-A side shards"
        ),
    )
    args = ap.parse_args()
    if args.from_source_checkpoint:
        process_source_checkpoint(args.dest)
        return
    os.makedirs(args.dest, exist_ok=True)
    shards = sorted(glob.glob(f"{args.src}/00-nonexpert-*.safetensors"))
    work = [(s, args.dest) for s in shards]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for msg in ex.map(process_shard, work):
            print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
