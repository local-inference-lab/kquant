"""Static per-expert statistics computed directly from the checkpoint.

One streaming pass over expert tensors accumulates, per (layer, expert,
matrix): the 16-bin E2M1 code histogram, block-scale exponent stats, weight
moments (for kurtosis), and the joint (code, block-max-code) histogram that
feeds codebook refitting. Router bias / row norms are read per layer as the
L0 traffic proxy.

Everything is exact arithmetic on weights; no model execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

from kquant import constants as C
from kquant.io import mxfp4
from kquant.io.hf_cache import CheckpointCache
from kquant.io.stream import iter_expert_batches, load_tensor

_M_IDX = {m: i for i, m in enumerate(C.EXPERT_MATRICES)}
_NL, _NE, _NM = C.NUM_MOE_LAYERS, C.NUM_EXPERTS, len(C.EXPERT_MATRICES)


def _lidx(layers: torch.Tensor) -> torch.Tensor:
    """checkpoint layer number (1..92) -> moe layer index (0..91)."""
    return (layers.long() - C.MOE_LAYERS[0])


@dataclass
class StaticAccumulators:
    code_hist: torch.Tensor  # [L, E, M, 16] float64
    joint_hist: torch.Tensor  # [L, M, 16, 8] float64  (code x blockmax-mag)
    m2_sum: torch.Tensor  # [L, E, M] float64  sum w^2
    m4_sum: torch.Tensor  # [L, E, M] float64  sum w^4
    n_elem: torch.Tensor  # [L, E, M] float64
    scale_exp_sum: torch.Tensor  # [L, E, M] float64
    scale_exp_min: torch.Tensor  # [L, E, M] float64
    scale_exp_max: torch.Tensor  # [L, E, M] float64
    present: torch.Tensor  # [L] float64 (count of expert-matrices seen)

    @classmethod
    def new(cls, device: str) -> "StaticAccumulators":
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        return cls(
            code_hist=z(_NL, _NE, _NM, 16),
            joint_hist=z(_NL, _NM, 16, 8),
            m2_sum=z(_NL, _NE, _NM),
            m4_sum=z(_NL, _NE, _NM),
            n_elem=z(_NL, _NE, _NM),
            scale_exp_sum=z(_NL, _NE, _NM),
            scale_exp_min=z(_NL, _NE, _NM) + 1e9,
            scale_exp_max=z(_NL, _NE, _NM) - 1e9,
            present=z(_NL),
        )


# magnitude index for each code 0..15 (|value| rank among the 8 E2M1 levels)
_MAG_IDX = torch.tensor([c % 8 for c in range(16)], dtype=torch.long)


def accumulate_batch(acc: StaticAccumulators, batch, device: str) -> None:
    packed = batch.packed.to(device, non_blocking=True)
    scale = batch.scale.to(device, non_blocking=True)
    li = _lidx(batch.layers).to(device)
    ei = batch.experts.long().to(device)
    mi = _M_IDX[batch.matrix]
    B = packed.shape[0]

    codes = mxfp4.unpack_codes(packed)  # [B, N, K] u8
    # --- code histogram ---------------------------------------------------
    flat = codes.long().reshape(B, -1)
    offs = torch.arange(B, device=device).unsqueeze(1) * 16
    hist = torch.bincount((flat + offs).reshape(-1), minlength=B * 16).reshape(B, 16)
    acc.code_hist.index_put_((li, ei, torch.full_like(li, mi)),
                             hist.to(torch.float64), accumulate=True)

    # --- joint (code, blockmax magnitude) histogram for refit -------------
    nb = codes.shape[-1] // C.MXFP4_BLOCK
    cb = codes.long().reshape(B, -1, nb, C.MXFP4_BLOCK)
    mag = _MAG_IDX.to(device)[cb]  # [B, N, nb, 32]
    bmax = mag.amax(dim=-1, keepdim=True)  # [B, N, nb, 1]
    joint_key = cb * 8 + bmax  # [B, N, nb, 32] in [0, 128)
    jflat = joint_key.reshape(B, -1)
    jhist = torch.bincount(
        (jflat + torch.arange(B, device=device).unsqueeze(1) * 128).reshape(-1),
        minlength=B * 128,
    ).reshape(B, 16, 8)
    acc.joint_hist.index_put_((li, torch.full_like(li, mi)),
                              jhist.to(torch.float64), accumulate=True)

    # --- weight moments ---------------------------------------------------
    vals, s = mxfp4.dequant_blocked(packed, scale)  # [B,N,nb,32], [B,N,nb]
    w = vals * s.unsqueeze(-1)
    w2 = (w.double() ** 2)
    acc.m2_sum.index_put_((li, ei, torch.full_like(li, mi)),
                          w2.sum(dim=(1, 2, 3)), accumulate=True)
    acc.m4_sum.index_put_((li, ei, torch.full_like(li, mi)),
                          (w2 * w2).sum(dim=(1, 2, 3)), accumulate=True)
    acc.n_elem.index_put_((li, ei, torch.full_like(li, mi)),
                          torch.full((B,), float(w[0].numel()),
                                     dtype=torch.float64, device=device),
                          accumulate=True)

    # --- scale exponent stats --------------------------------------------
    se = scale.to(torch.float64) - C.E8M0_BIAS
    acc.scale_exp_sum.index_put_((li, ei, torch.full_like(li, mi)),
                                 se.sum(dim=(1, 2)), accumulate=True)
    cur_min = acc.scale_exp_min[li, ei, mi]
    cur_max = acc.scale_exp_max[li, ei, mi]
    acc.scale_exp_min[li, ei, mi] = torch.minimum(cur_min, se.amin(dim=(1, 2)))
    acc.scale_exp_max[li, ei, mi] = torch.maximum(cur_max, se.amax(dim=(1, 2)))
    acc.present.index_put_((li,), torch.ones(B, dtype=torch.float64, device=device),
                           accumulate=True)


def finalize(acc: StaticAccumulators, cache: CheckpointCache) -> dict[str, torch.Tensor]:
    hist = acc.code_hist
    total = hist.sum(dim=-1, keepdim=True).clamp(min=1.0)
    p = hist / total
    entropy = -(p * torch.where(p > 0, p, torch.ones_like(p)).log2()).sum(dim=-1)
    zero_frac = (hist[..., 0] + hist[..., 8]) / total.squeeze(-1)
    sat_frac = (hist[..., 7] + hist[..., 15]) / total.squeeze(-1)
    n = acc.n_elem.clamp(min=1.0)
    m2 = acc.m2_sum / n
    kurtosis = (acc.m4_sum / n) / (m2 * m2).clamp(min=1e-30)
    scale_blocks = n / C.MXFP4_BLOCK

    out = {
        "code_hist": hist,
        "joint_hist": acc.joint_hist,  # [L, M, 16, 8] (leading dim L only)
        "entropy": entropy,
        "zero_frac": zero_frac,
        "sat_frac": sat_frac,
        "kurtosis": kurtosis,
        "m2_sum": acc.m2_sum,
        "n_elem": acc.n_elem,
        "scale_exp_mean": acc.scale_exp_sum / scale_blocks.clamp(min=1.0),
        "scale_exp_min": acc.scale_exp_min,
        "scale_exp_max": acc.scale_exp_max,
        "present_count": acc.present,
    }

    # router traffic proxies, per available layer
    bias = torch.full((_NL, _NE), float("nan"), dtype=torch.float64)
    rown = torch.full((_NL, _NE), float("nan"), dtype=torch.float64)
    for layer in C.MOE_LAYERS:
        li = layer - C.MOE_LAYERS[0]
        bname = C.router_bias_tensor(layer)
        wname = C.router_weight_tensor(layer)
        if cache.has_tensor(bname):
            bias[li] = load_tensor(cache, bname).to(torch.float64)
        if cache.has_tensor(wname):
            w = load_tensor(cache, wname).to(torch.float32)
            rown[li] = w.norm(dim=1).to(torch.float64)
    out["router_bias"] = bias
    out["router_row_norm"] = rown
    return out


def run_static_stats(
    cache: CheckpointCache,
    device: str = "cpu",
    batch_size: int = 96,
    layers: set[int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    acc = StaticAccumulators.new(device)
    t0 = time.time()
    n_batches = 0
    for batch in tqdm(iter_expert_batches(cache, batch_size=batch_size, layers=layers),
                      desc="static stats", unit="batch"):
        accumulate_batch(acc, batch, device)
        n_batches += 1
    arrays = finalize(acc, cache)
    meta = {
        "kind": "static",
        "device": device,
        "batches": n_batches,
        "elapsed_s": round(time.time() - t0, 1),
        "layers_restricted": sorted(layers) if layers else None,
    }
    return arrays, meta
