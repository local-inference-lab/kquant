"""Static distortion engine: exact re-rounding of the E2M1 grid into candidate
codebooks, per expert x matrix, with optional U-folding for w2.

Distortion(e, m, fmt) = ||W - Q_fmt(W)||_F^2 computed on the dequantized
source weights. For w2 the additional U-folded value tr(D^T G D), G = U^T U
with U the layer's shared routed_expert_up_proj, weights the error by the
actual write-back path into the residual stream.
"""

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from kquant import constants as C
from kquant.io import mxfp4
from kquant.io.hf_cache import CheckpointCache
from kquant.io.stream import iter_expert_batches, load_tensor
from kquant.quantize import BUILTIN_FORMATS, CodebookFormat, quantize_blocks

_M_IDX = {m: i for i, m in enumerate(C.EXPERT_MATRICES)}
_NL, _NE, _NM = C.NUM_MOE_LAYERS, C.NUM_EXPERTS, len(C.EXPERT_MATRICES)


class _UCache:
    """Per-layer G = U^T U, computed lazily on first use."""

    def __init__(self, cache: CheckpointCache, device: str):
        self.cache = cache
        self.device = device
        self.g: dict[int, torch.Tensor | None] = {}

    def get(self, layer: int) -> torch.Tensor | None:
        if layer not in self.g:
            name = C.latent_up_proj_tensor(layer)
            if not self.cache.has_tensor(name):
                self.g[layer] = None
            else:
                u = load_tensor(self.cache, name).to(self.device, torch.float32)
                self.g[layer] = u.T @ u  # [LATENT, LATENT]
        return self.g[layer]


def run_distortion(
    cache: CheckpointCache,
    formats: dict[str, CodebookFormat] | None = None,
    device: str = "cpu",
    batch_size: int = 64,
    layers: set[int] | None = None,
    fold_u: bool = True,
    scale_search: tuple[float, ...] = (1.0,),
) -> tuple[dict[str, torch.Tensor], dict]:
    formats = formats or dict(BUILTIN_FORMATS)
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
    dist = {name: z(_NL, _NE, _NM) for name in formats}
    fold = {name: z(_NL, _NE) for name in formats} if fold_u else {}
    fold_ref = z(_NL, _NE) if fold_u else None  # tr(W^T G W) baseline
    ucache = _UCache(cache, device)

    t0 = time.time()
    n_batches = 0
    for batch in tqdm(
        iter_expert_batches(cache, batch_size=batch_size, layers=layers),
        desc="distortion", unit="batch",
    ):
        packed = batch.packed.to(device, non_blocking=True)
        scale = batch.scale.to(device, non_blocking=True)
        li = (batch.layers.long() - C.MOE_LAYERS[0]).to(device)
        ei = batch.experts.long().to(device)
        mi = _M_IDX[batch.matrix]
        vals, s = mxfp4.dequant_blocked(packed, scale)
        w = vals * s.unsqueeze(-1)  # [B, N, nb, 32] f32

        do_fold = fold_u and batch.matrix == "w2"
        g_by_layer: dict[int, torch.Tensor | None] = {}
        if do_fold:
            for layer in batch.layers.unique().tolist():
                g_by_layer[int(layer)] = ucache.get(int(layer))

        for name, fmt in formats.items():
            q = quantize_blocks(w, fmt, scale_search=scale_search)
            dist[name].index_put_(
                (li, ei, torch.full_like(li, mi)),
                q.sq_err.sum(dim=(1, 2)), accumulate=True,
            )
            if do_fold:
                delta = (w - q.recon).reshape(w.shape[0], w.shape[1], -1)
                # w2 is [out=LATENT, in]; error column space lives in LATENT
                for j in range(delta.shape[0]):
                    layer = int(batch.layers[j])
                    G = g_by_layer.get(layer)
                    if G is None:
                        continue
                    d = delta[j]  # [LATENT, K]
                    fold[name][li[j], ei[j]] += torch.einsum(
                        "ik,ij,jk->", d, G, d
                    ).double()
        if fold_u and batch.matrix == "w2":
            wf = w.reshape(w.shape[0], w.shape[1], -1)
            for j in range(wf.shape[0]):
                G = g_by_layer.get(int(batch.layers[j]))
                if G is None:
                    continue
                fold_ref[li[j], ei[j]] += torch.einsum(
                    "ik,ij,jk->", wf[j], G, wf[j]
                ).double()
        n_batches += 1

    arrays: dict[str, torch.Tensor] = {}
    for name in formats:
        arrays[f"distortion.{name}"] = dist[name]
        if fold_u:
            arrays[f"distortion_fold.{name}"] = fold[name]
    if fold_u:
        arrays["fold_ref"] = fold_ref
    meta = {
        "kind": "distortion",
        "formats": {n: list(f.levels) for n, f in formats.items()},
        "fold_u": fold_u,
        "scale_search": list(scale_search),
        "device": device,
        "batches": n_batches,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return arrays, meta
