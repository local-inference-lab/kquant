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
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant.mxfp8 import MXFP8_BLOCK_SIZE, mxfp8_quantize_cpu

SRC = "/models/Kimi-K3-NF3R-Uniform-3p25-serve"
DEST = "/models/Kimi-K3-mxfp8-nonexpert"
BLOCK = MXFP8_BLOCK_SIZE

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


def process_shard(args: tuple) -> str:
    src_file, dest_dir = args
    out: dict[str, torch.Tensor] = {}
    n_baked = 0
    with safe_open(src_file, framework="pt") as sf:
        for name in sf.keys():  # noqa: SIM118 - safe_open is not iterable
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    shards = sorted(glob.glob(f"{args.src}/00-nonexpert-*.safetensors"))
    work = [(s, args.dest) for s in shards]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for msg in ex.map(process_shard, work):
            print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
