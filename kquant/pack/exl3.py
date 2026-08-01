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


def split_tp_local_weight(
    weight: torch.Tensor, matrix: str, tp_size: int
) -> list[torch.Tensor]:
    """Split a transposed expert matrix before EXL3 quantization.

    Gate/up are column parallel (split output/N); down is row parallel (split
    input/K).  Quantizing these local matrices is essential at TP16: slicing a
    globally H128-transformed tensor at 192 channels cuts every other
    Hadamard block in half.
    """

    if matrix not in MATRICES:
        raise ValueError(f"unsupported expert matrix: {matrix}")
    tp_size = int(tp_size)
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    axis = 0 if matrix == "w2" else 1
    width = int(weight.shape[axis])
    if width % tp_size:
        raise ValueError(f"{matrix} dimension {width} is not divisible by TP{tp_size}")
    local = width // tp_size
    if local % 128 not in (0, 64):
        raise ValueError(
            f"{matrix} TP{tp_size} local dimension {local} cannot be represented "
            "as H128 blocks with an optional H64 tail"
        )
    return [part.contiguous() for part in weight.split(local, dim=axis)]


def merge_tp_local_results(
    matrix: str,
    results: list[tuple[float, dict[str, torch.Tensor]]],
) -> tuple[float, dict[str, torch.Tensor]]:
    """Reassemble rank-local encodings into the existing sharded tensor ABI."""

    if matrix not in MATRICES:
        raise ValueError(f"unsupported expert matrix: {matrix}")
    if not results:
        raise ValueError("cannot merge an empty TP result set")
    tensors = [result[1] for result in results]
    trellis_dim = 0 if matrix == "w2" else 1
    merged = {"trellis": torch.cat([t["trellis"] for t in tensors], trellis_dim)}
    if matrix == "w2":
        merged["suh"] = torch.cat([t["suh"] for t in tensors], dim=0)
        replicated, sharded_name = "svh", "suh"
    else:
        merged["svh"] = torch.cat([t["svh"] for t in tensors], dim=0)
        replicated, sharded_name = "suh", "svh"
    reference = tensors[0][replicated]
    if not all(torch.equal(t[replicated], reference) for t in tensors[1:]):
        raise ValueError(
            f"TP-local {matrix} produced inconsistent replicated {replicated}"
        )
    merged[replicated] = reference
    for optional in ("mcg", "mul1"):
        present = [t.get(optional) for t in tensors]
        if any(value is not None for value in present):
            if any(value is None for value in present) or not all(
                torch.equal(value, present[0]) for value in present[1:]
            ):
                raise ValueError(
                    f"TP-local {matrix} produced inconsistent {optional} metadata"
                )
            merged[optional] = present[0]
    expected_sharded = sum(int(t[sharded_name].numel()) for t in tensors)
    if int(merged[sharded_name].numel()) != expected_sharded:
        raise AssertionError(f"lost TP-local {matrix} {sharded_name} values")
    proxy_err = sum(float(result[0]) for result in results) / len(results)
    return proxy_err, merged


def quantize_layer(
    cache: CheckpointCache,
    layer: int,
    experts: list[int],
    device: torch.device,
    shared_h_by_k: dict[int, dict],
    batch: int = BATCH,
    tp_size: int = 1,
    temp_batch_size: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Quantize the demoted experts of one layer; returns (tensors, proxy errs)."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3_batch

    tp_size = int(tp_size)
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if temp_batch_size is not None and int(temp_batch_size) <= 0:
        raise ValueError(f"temp_batch_size must be positive, got {temp_batch_size}")
    out: dict[str, torch.Tensor] = {}
    errs: dict[str, float] = {}
    experts_per_batch = max(1, batch // int(tp_size))
    for matrix in MATRICES:
        for start in range(0, len(experts), experts_per_batch):
            group = experts[start : start + experts_per_batch]
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
                local_weights = split_tp_local_weight(w.T.contiguous(), matrix, tp_size)
                for local_weight in local_weights:
                    weights.append(local_weight)
                    qa = {
                        "K": BITS,
                        "seed": expert_seed(layer, e, matrix),
                        "sigma_reg": SIGMA_REG,
                        "devices": [f"cuda:{device.index or 0}"],
                        "device_ratios": None,
                        "apply_out_scales": False,
                        "mcg": True,
                    }
                    if temp_batch_size is not None:
                        qa["temp_batch_size"] = int(temp_batch_size)
                    if local_weight.shape[0] % 128 == 64:
                        qa["buf_size_k"] = 64
                    if SHARED_SU and matrix in ("w1", "w3"):
                        # Collapse the TP-replicated H-side su across experts:
                        # shared channel-scale profile + g_scale folded into the
                        # sharded sv instead. (w2's replicated vector is svh,
                        # already pure shared-seed signs.)
                        qa["shared_input_scales_key"] = f"{layer}:{matrix}"
                        qa["g_scale_into_sv"] = True
                    qas.append(qa)
            kdim = weights[0].shape[0]
            shared_h = shared_h_by_k[kdim]
            results = quantize_exl3_batch(weights, [shared_h] * len(weights), qas)
            if tp_size == 1:
                merged_results = results
            else:
                merged_results = [
                    merge_tp_local_results(
                        matrix, results[i * tp_size : (i + 1) * tp_size]
                    )
                    for i in range(len(group))
                ]
            for e, (proxy_err, tensors) in zip(group, merged_results):
                out[_tensor_name(layer, e, matrix, "trellis")] = tensors[
                    "trellis"
                ].cpu()
                out[_tensor_name(layer, e, matrix, "suh")] = tensors["suh"].cpu()
                out[_tensor_name(layer, e, matrix, "svh")] = tensors["svh"].cpu()
                errs[f"{layer}.{e}.{matrix}"] = float(proxy_err)
    return out, errs


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
