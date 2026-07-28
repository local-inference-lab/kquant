"""Codebook refitting from the joint (code, block-max) histogram.

The static pass records, per matrix type, how often each E2M1 code co-occurs
with each block max-magnitude. Under absmax scaling every weight normalizes to
value/blockmax, a finite set of <=113 distinct ratios, so the exact weighted
distribution of normalized weights is known and Lloyd-Max refitting is a small
deterministic computation - no weight re-reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kquant import constants as C
from kquant.quantize import CodebookFormat

_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def normalized_points(joint_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """joint_hist [..., 16, 8] -> (values, weights) of normalized weights.

    Blocks whose max magnitude is 0 are all-zero and excluded (they quantize
    exactly under any codebook).
    """
    jh = joint_hist.reshape(-1, 16, 8).sum(axis=0)
    vals, wts = [], []
    lut = np.array(C.E2M1_LUT)
    for code in range(16):
        for bm in range(1, 8):
            n = jh[code, bm]
            if n <= 0:
                continue
            vals.append(lut[code] / _MAGS[bm])
            wts.append(n)
    return np.array(vals), np.array(wts)


def weighted_mse(levels: np.ndarray, vals: np.ndarray, wts: np.ndarray) -> float:
    idx = np.abs(vals[:, None] - levels[None, :]).argmin(axis=1)
    err = (vals - levels[idx]) ** 2
    return float((err * wts).sum() / wts.sum())


def lloyd_max(
    vals: np.ndarray,
    wts: np.ndarray,
    n_levels: int,
    init: np.ndarray,
    iters: int = 200,
    symmetric: bool = True,
) -> np.ndarray:
    levels = init.astype(np.float64).copy()
    for _ in range(iters):
        idx = np.abs(vals[:, None] - levels[None, :]).argmin(axis=1)
        new = levels.copy()
        for j in range(n_levels):
            m = idx == j
            if wts[m].sum() > 0:
                new[j] = (vals[m] * wts[m]).sum() / wts[m].sum()
        new.sort()
        if symmetric:
            new = (new - new[::-1]) / 2  # antisymmetrize
        if np.allclose(new, levels, atol=1e-7):
            levels = new
            break
        levels = new
    return levels


def refit_codebooks(joint_hist: np.ndarray) -> dict:
    """Produce refit NF3/NF2 codebooks (global, plus per matrix type)."""
    out: dict = {"per_matrix": {}}
    scopes = {"global": joint_hist}
    for i, m in enumerate(C.EXPERT_MATRICES):
        scopes[m] = joint_hist[:, i]

    for scope, jh in scopes.items():
        vals, wts = normalized_points(np.asarray(jh))
        entry = {}
        for base_name, base_levels, nl in (
            ("nf3", np.array(C.NF3_CODEBOOK), 8),
            ("nf2", np.array(C.NF2_CODEBOOK), 4),
        ):
            stock_mse = weighted_mse(base_levels, vals, wts)
            refit = lloyd_max(vals, wts, nl, init=base_levels)
            refit /= max(abs(refit).max(), 1e-9)  # keep max|level| = 1
            entry[base_name] = {
                "stock_levels": base_levels.tolist(),
                "stock_mse": stock_mse,
                "refit_levels": refit.tolist(),
                "refit_mse": weighted_mse(refit, vals, wts),
            }
        if scope == "global":
            out["global"] = entry
        else:
            out["per_matrix"][scope] = entry
    return out


def load_refit_formats(path: str | Path) -> dict[str, CodebookFormat]:
    """refit.json -> CodebookFormat registry entries (global scope)."""
    data = json.loads(Path(path).read_text())
    out = {}
    for base, bits, code_bits in (("nf3", C.NF3_BITS_EFFECTIVE, 3),
                                  ("nf2", C.NF2_BITS_EFFECTIVE, 2)):
        entry = data["global"][base]
        out[f"{base}_refit"] = CodebookFormat(
            f"{base}_refit", tuple(entry["refit_levels"]), bits, code_bits
        )
    return out
