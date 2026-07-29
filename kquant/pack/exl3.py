"""EXL3-3.0 expert packer (Phase B).

Quantizes demoted experts with exllamav3's trellis quantizer using the
shared-Hessian batch path (one ldlq pass over experts concatenated along
out-features — exact, and it keeps quantize_tiles fed with large batches).
Emits per-layer safetensors shards of native EXL3 tensors that b12x's
trellis_moe.prepare_weights wraps zero-copy:

    ...experts.{e}.{w}.exl3_trellis  [K/16, N/16, 16*bits] int16
    ...experts.{e}.{w}.exl3_suh      [in_features]  fp16
    ...experts.{e}.{w}.exl3_svh      [out_features] fp16

Identity Hessian for now; per-layer measured Hessians (L2 dynstats) drop in
by replacing `make_shared_h` — the shared-H batch contract is unchanged as
long as experts of a layer share it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from kquant import constants as C
from kquant.io.hf_cache import CheckpointCache
from kquant.io.mxfp4 import dequant
from kquant.io.stream import load_tensor

MATRICES = ("w1", "w3", "w2")
BITS = 3
BATCH = 32
SIGMA_REG = 0.025
MCG_MULT = 0xCBAC1FED  # exllamav3 codebook_mcg_mult; recorded in the manifest


def expert_seed(layer: int, expert: int, matrix: str) -> int:
    return layer * 1_000_000 + expert * 10 + MATRICES.index(matrix)


def make_shared_h(in_features: int, device: torch.device) -> dict:
    return {
        "H": torch.eye(in_features, dtype=torch.float32, device=device),
        "first_key": f"h{in_features}",
        "count": 1,
        "finalized": False,
        "num_total": 1,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }


def _tensor_name(layer: int, expert: int, matrix: str, part: str) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        f"experts.{expert}.{matrix}.exl3_{part}"
    )


def quantize_layer(
    cache: CheckpointCache,
    layer: int,
    experts: list[int],
    device: torch.device,
    shared_h_by_k: dict[int, dict],
    batch: int = BATCH,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Quantize the demoted experts of one layer; returns (tensors, proxy errs)."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    out: dict[str, torch.Tensor] = {}
    errs: dict[str, float] = {}
    for matrix in MATRICES:
        for start in range(0, len(experts), batch):
            group = experts[start : start + batch]
            weights, qas = [], []
            for e in group:
                base = (
                    f"language_model.model.layers.{layer}.block_sparse_moe."
                    f"experts.{e}.{matrix}"
                )
                w = dequant(
                    load_tensor(cache, base + ".weight_packed").to(device),
                    load_tensor(cache, base + ".weight_scale").to(device),
                ).float()
                weights.append(w.T.contiguous())  # (in, out)
                qas.append(
                    {
                        "K": BITS,
                        "seed": expert_seed(layer, e, matrix),
                        "sigma_reg": SIGMA_REG,
                        "devices": [f"cuda:{device.index or 0}"],
                        "device_ratios": None,
                        "apply_out_scales": False,
                        "mcg": True,
                    }
                )
            kdim = weights[0].shape[0]
            shared_h = shared_h_by_k[kdim]
            results = quantize_exl3_batch(weights, [shared_h] * len(group), qas)
            for e, (proxy_err, tensors) in zip(group, results):
                out[_tensor_name(layer, e, matrix, "trellis")] = (
                    tensors["trellis"].cpu()
                )
                out[_tensor_name(layer, e, matrix, "suh")] = tensors["suh"].cpu()
                out[_tensor_name(layer, e, matrix, "svh")] = tensors["svh"].cpu()
                errs[f"{layer}.{e}.{matrix}"] = float(proxy_err)
    return out, errs


def write_layer_shard(
    dest: Path, layer: int, tensors: dict[str, torch.Tensor]
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"exl3-layer-{layer:05d}.safetensors"
    save_file(tensors, str(path))
    return path


def write_manifest(dest: Path, meta: dict) -> None:
    meta = dict(meta)
    meta.setdefault("kind", "kquant_exl3_artifact")
    meta.setdefault("schema_version", 2)
    meta.setdefault("bits", BITS)
    meta.setdefault("codebook", "mcg")
    meta.setdefault("mcg_mult", MCG_MULT)
    meta.setdefault("hessian", "identity")
    (dest / "kquant_exl3_manifest.json").write_text(json.dumps(meta, indent=1))
