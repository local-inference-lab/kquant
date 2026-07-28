"""L0 ranker: damage estimate = static distortion x checkpoint-derived traffic.

Traffic proxy: the noaux_tc router bias is trained to boost underused experts
and suppress overused ones, so -bias orders experts by training-time
popularity. We z-score it per layer and map through exp(alpha * -z),
normalized to mean 1 per layer. `--traffic uniform` ablates the proxy.

w2 distortion can optionally be replaced by its U-folded value, rescaled into
raw-Frobenius units via fold_ref so the three matrices stay comparable.
"""

from __future__ import annotations

import numpy as np

from kquant import constants as C
from kquant.dynstats import ComposedStats

_NL, _NE = C.NUM_MOE_LAYERS, C.NUM_EXPERTS
W2 = C.EXPERT_MATRICES.index("w2")


def traffic_proxy(stats: ComposedStats, mode: str = "bias",
                  alpha: float = 1.0) -> np.ndarray:
    if mode == "uniform":
        return np.ones((_NL, _NE))
    if mode == "measured":
        g2 = stats.require("gate_sq_sum")
        t = g2 / np.nanmean(np.where(g2 > 0, g2, np.nan), axis=1, keepdims=True)
        return np.nan_to_num(t, nan=1.0)
    if mode != "bias":
        raise ValueError(f"unknown traffic mode {mode}")
    bias = stats.require("router_bias")
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(bias, axis=1, keepdims=True)
        sd = np.nanstd(bias, axis=1, keepdims=True)
        z = (bias - mu) / np.where(sd > 0, sd, 1.0)
        t = np.exp(-alpha * z)
        t /= np.nanmean(t, axis=1, keepdims=True)
    return np.nan_to_num(t, nan=1.0)


def matrix_distortion(stats: ComposedStats, fmt: str, w2_mode: str) -> np.ndarray:
    """[L, E] total distortion for one demoted format."""
    d = np.array(stats.require(f"distortion.{fmt}"), dtype=np.float64)  # [L,E,3]
    total = d.sum(axis=2)
    if w2_mode == "folded":
        fold = stats.get(f"distortion_fold.{fmt}")
        fold_ref = stats.get("fold_ref")
        m2 = stats.get("m2_sum")
        if fold is None or fold_ref is None or m2 is None:
            raise KeyError("folded w2 requested but fold arrays not in stats")
        w2_raw = d[:, :, W2]
        w2_m2 = np.array(m2, dtype=np.float64)[:, :, W2]
        rescaled = np.divide(
            fold, np.where(fold_ref > 0, fold_ref, 1.0)
        ) * w2_m2
        total = total - w2_raw + np.where(fold_ref > 0, rescaled, w2_raw)
    return total


def damage_table(
    stats: ComposedStats,
    demoted_formats: list[str],
    traffic_mode: str = "bias",
    traffic_alpha: float = 1.0,
    w2_mode: str = "raw",
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (formats, damage [L,E,F], traffic [L,E]).

    Format list always begins with keep_mxfp4 at damage 0.
    """
    traffic = traffic_proxy(stats, traffic_mode, traffic_alpha)
    formats = ["keep_mxfp4"] + list(demoted_formats)
    damage = np.zeros((_NL, _NE, len(formats)))
    for i, fmt in enumerate(demoted_formats, start=1):
        damage[:, :, i] = matrix_distortion(stats, fmt, w2_mode) * traffic
    return formats, damage, traffic
