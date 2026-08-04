"""EXL3-3.0 expert packer (Phase B).

Quantizes demoted experts with exllamav3's trellis quantizer using the
shared-Hessian batch path (one ldlq pass over experts concatenated along
out-features — exact, and it keeps quantize_tiles fed with large batches).
Emits per-layer safetensors shards of native EXL3 tensors that b12x's
trellis_moe.prepare_weights wraps zero-copy:

    ...experts.{e}.{w}.exl3_trellis  [K/16, N/16, 16*bits] int16
    ...experts.{e}.{w}.exl3_suh      [in_features]  fp16
    ...experts.{e}.{w}.exl3_svh      [out_features] fp16

The caller supplies one measured H13/H2 pair per decoder layer.  A shared-H
object is intentionally reused across batches *within* a layer (the exllama
encoder finalizes it in place), but never across layers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from kquant.io.hf_cache import CheckpointCache
from kquant.io.mxfp4 import dequant
from kquant.io.stream import load_tensor

MATRICES = ("w1", "w3", "w2")
BITS = 3
BATCH = 32
SIGMA_REG = 0.025
MCG_MULT = 0xCBAC1FED  # exllamav3 codebook_mcg_mult; recorded in the manifest


# Shared-su mode: one seed per (layer, matrix) so every expert draws the same
# su/sv sign vectors. Collapses the TP-replicated H-side vectors from [E, H]
# to [H]. Quality gate: A/B closure test vs per-expert seeds.
SHARED_SU = os.environ.get("KQUANT_EXL3_SHARED_SU") == "1"


def expert_seed(layer: int, expert: int, matrix: str) -> int:
    if SHARED_SU:
        return layer * 1_000_000 + MATRICES.index(matrix)
    return layer * 1_000_000 + expert * 10 + MATRICES.index(matrix)


def make_shared_h(
    in_features: int,
    device: torch.device,
    hessian: torch.Tensor | None = None,
) -> dict:
    if hessian is None:
        hessian = torch.eye(in_features, dtype=torch.float32)
    if tuple(hessian.shape) != (in_features, in_features):
        raise ValueError(
            f"Hessian shape {tuple(hessian.shape)} does not match {in_features}"
        )
    canonical_h = hessian.to(device=device, dtype=torch.float32, copy=True).contiguous()
    return {
        # exllamav3 finalizes and mutates H in place. Keep one untouched copy
        # for canonical-coordinate allocation error metrics.
        "H": canonical_h.clone(),
        "error_H": canonical_h,
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
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, float | dict[str, float]],
    dict[str, torch.Tensor],
]:
    """Quantize the demoted experts of one layer; returns (tensors, proxy errs)."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    out: dict[str, torch.Tensor] = {}
    errs: dict[str, float | dict[str, float]] = {}
    residuals: dict[str, torch.Tensor] = {}
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
                qa = {
                    "K": BITS,
                    "seed": expert_seed(layer, e, matrix),
                    "sigma_reg": SIGMA_REG,
                    "devices": [f"cuda:{device.index or 0}"],
                    "device_ratios": None,
                    "apply_out_scales": False,
                    "mcg": True,
                }
                qa["error_hessian"] = shared_h_by_k[w.shape[1]]["error_H"]
                if SHARED_SU and matrix in ("w1", "w3"):
                    # Collapse the TP-replicated H-side su across experts:
                    # shared channel-scale profile + g_scale folded into the
                    # sharded sv instead. (w2's replicated vector is svh,
                    # already pure shared-seed signs.)
                    qa["shared_input_scales_key"] = f"{layer}:{matrix}"
                    qa["g_scale_into_sv"] = True
                qa["return_error_metrics"] = True
                qas.append(qa)
            kdim = weights[0].shape[0]
            shared_h = shared_h_by_k[kdim]
            results = quantize_exl3_batch(weights, [shared_h] * len(group), qas)
            for e, qa, (proxy_err, tensors) in zip(group, qas, results):
                out[_tensor_name(layer, e, matrix, "trellis")] = tensors[
                    "trellis"
                ].cpu()
                out[_tensor_name(layer, e, matrix, "suh")] = tensors["suh"].cpu()
                out[_tensor_name(layer, e, matrix, "svh")] = tensors["svh"].cpu()
                metrics = qa.get("error_metrics")
                if metrics is None:
                    errs[f"{layer}.{e}.{matrix}"] = float(proxy_err)
                else:
                    errs[f"{layer}.{e}.{matrix}"] = {
                        "proxy": float(proxy_err),
                        "numerator": float(metrics["numerator"]),
                        "denominator": float(metrics["denominator"]),
                        "encoder_numerator": float(metrics["encoder_numerator"]),
                        "encoder_denominator": float(metrics["encoder_denominator"]),
                    }
                    residuals[f"{e}.{matrix}"] = metrics["residual_by_input"].to(
                        dtype=torch.float32, device="cpu"
                    )
    return out, errs, residuals


def write_layer_shard(dest: Path, layer: int, tensors: dict[str, torch.Tensor]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"exl3-layer-{layer:05d}.safetensors"
    tmp = dest / f".exl3-layer-{layer:05d}.{os.getpid()}.safetensors.tmp"
    save_file(tensors, str(tmp))
    tmp.rename(path)
    return path


def write_manifest(dest: Path, meta: dict) -> None:
    meta = dict(meta)
    meta.setdefault("kind", "kquant_exl3_artifact")
    meta.setdefault("schema_version", 2)
    meta.setdefault("bits", BITS)
    meta.setdefault("codebook", "mcg")
    meta.setdefault("mcg_mult", MCG_MULT)
    meta.setdefault("hessian", "identity")
    path = dest / "kquant_exl3_manifest.json"
    tmp = dest / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(path)
