"""Pack the hybrid artifact: keep-tier passthrough + NF3-tier re-rounding.

Artifact layout (expert tensors only; non-expert tensors stay in the source
checkpoint and are referenced from kquant_config.json):

  keep_mxfp4 : original names (...wN.weight_packed / ...wN.weight_scale),
               bytes copied verbatim from the source shards;
  nf3-family : ...wN.nf3_packed  uint8 [out, K*3/8]  (3-bit codes, 8 codes
               per 24-bit little-endian group)
               ...wN.nf3_scale   uint8 [out, K/32]   (E4M3 bits of
               scale * 2^SCALE_BIAS_LOG2)

Shards are written as model-kquant-XXXXX.safetensors with an index file
mirroring the HF convention.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from kquant import constants as C
from kquant.io import mxfp4
from kquant.io.hf_cache import CheckpointCache
from kquant.io.stream import iter_expert_batches
from kquant.quantize import (
    SCALE_BIAS_LOG2,
    CodebookFormat,
    encode_scale_bytes,
    pack_3bit,
    quantize_blocks,
)
from kquant.rank.solver import Allocation

_M_IDX = {m: i for i, m in enumerate(C.EXPERT_MATRICES)}


class ShardWriter:
    def __init__(self, out_dir: Path, target_bytes: int = 4 * C.GIB):
        self.out_dir = out_dir
        self.target = target_bytes
        self.buf: dict[str, torch.Tensor] = {}
        self.buf_bytes = 0
        self.shard_idx = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, tensor: torch.Tensor) -> None:
        self.buf[name] = tensor.cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        self.buf_bytes += nbytes
        self.total_bytes += nbytes
        if self.buf_bytes >= self.target:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        self.shard_idx += 1
        fname = f"model-kquant-{self.shard_idx:05d}.safetensors"
        save_file(self.buf, str(self.out_dir / fname))
        for k in self.buf:
            self.weight_map[k] = fname
        self.buf = {}
        self.buf_bytes = 0

    def finish(self) -> dict:
        self.flush()
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": self.weight_map,
        }
        (self.out_dir / "model-kquant.index.json").write_text(json.dumps(index))
        return index


def pack_artifact(
    cache: CheckpointCache,
    alloc: Allocation,
    formats: dict[str, CodebookFormat],
    out_dir: str | Path,
    device: str = "cpu",
    layers: set[int] | None = None,
    batch_size: int = 64,
) -> dict:
    out_dir = Path(out_dir)
    writer = ShardWriter(out_dir)
    keep_idx = alloc.formats.index("keep_mxfp4")
    packable = set(cache.complete_moe_layers())
    if layers:
        packable &= layers
    t0 = time.time()
    tier_bytes = {f: 0 for f in alloc.formats}
    tier_counts = {f: 0 for f in alloc.formats}

    for batch in tqdm(
        iter_expert_batches(cache, batch_size=batch_size, layers=packable),
        desc="pack", unit="batch",
    ):
        mi = _M_IDX[batch.matrix]
        li = batch.layers.long() - C.MOE_LAYERS[0]
        fidx = alloc.assignment[li.numpy(), batch.experts.numpy()]

        keep_mask = fidx == keep_idx
        for j in range(len(fidx)):
            layer, expert = int(batch.layers[j]), int(batch.experts[j])
            base = (
                f"{C.LM_PREFIX}layers.{layer}.block_sparse_moe."
                f"experts.{expert}.{batch.matrix}"
            )
            fmt_name = alloc.formats[int(fidx[j])]
            if keep_mask[j]:
                writer.add(f"{base}.weight_packed", batch.packed[j])
                writer.add(f"{base}.weight_scale", batch.scale[j])
                nb = batch.packed[j].numel() + batch.scale[j].numel()
            else:
                fmt = formats[fmt_name]
                packed = batch.packed[j : j + 1].to(device)
                scale = batch.scale[j : j + 1].to(device)
                vals, s = mxfp4.dequant_blocked(packed, scale)
                w = vals * s.unsqueeze(-1)
                q = quantize_blocks(w, fmt)
                K = w.shape[-2] * C.MXFP4_BLOCK
                codes = q.codes.reshape(1, w.shape[1], K)
                writer.add(f"{base}.{fmt.name}_packed", pack_3bit(codes)[0])
                writer.add(f"{base}.{fmt.name}_scale",
                           encode_scale_bytes(q.scales[0]))
                nb = w.shape[1] * (K * 3 // 8) + q.scales[0].numel()
            if mi == 0:
                tier_counts[fmt_name] += 1
            tier_bytes[fmt_name] += nb

    index = writer.finish()
    config = {
        "kind": "kquant_hybrid_artifact",
        "model": C.MODEL_ID,
        "revision": C.REVISION,
        "non_expert_source": "original checkpoint (tensors unchanged)",
        "formats": {
            "keep_mxfp4": {"storage": "source compressed-tensors passthrough"},
            **{
                n: {
                    "levels": list(f.levels),
                    "block": C.NF3_BLOCK,
                    "code_bits": f.code_bits,
                    "packing": "3bit_le_8codes_per_24bit",
                    "scale": "uint8=e4m3_bits(scale*2^bias)",
                    "scale_bias_log2": SCALE_BIAS_LOG2,
                    "scale_floor": C.NF3_SCALE_FLOOR,
                }
                for n, f in formats.items()
            },
        },
        "layers_packed": sorted(packable),
        "tier_expert_counts": tier_counts,
        "tier_bytes": tier_bytes,
        "total_bytes": writer.total_bytes,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "kquant_config.json").write_text(json.dumps(config, indent=1))
    return config
