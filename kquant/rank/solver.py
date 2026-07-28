"""Multiple-choice knapsack over (expert, format) with per-layer keep floors.

Every expert picks exactly one format. Greedy over convex-hull upgrade
segments is optimal up to one fractional item for this problem class, which
at 82k experts x <=4 formats is far below the noise floor of the damage
estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kquant import constants as C
from kquant.quantize import format_bits

_NL, _NE = C.NUM_MOE_LAYERS, C.NUM_EXPERTS
PARAMS_PER_EXPERT = C.TOTAL_EXPERT_PARAMS // (_NL * _NE)


def format_bytes(fmt: str) -> int:
    return int(PARAMS_PER_EXPERT * format_bits(fmt) / 8)


@dataclass
class Allocation:
    formats: list[str]
    assignment: np.ndarray  # int8 [L, E] index into formats
    layers_solved: list[int]  # checkpoint layer numbers actually ranked
    total_bytes: int  # extrapolated to all 92 layers
    achieved_bpw: float
    solved_bytes: int = 0  # bytes over ranked layers only
    solved_bpw: float = 0.0
    total_damage: float = 0.0
    keep_per_layer: dict[int, int] | None = None

    def counts(self) -> dict[str, int]:
        c = {}
        for i, f in enumerate(self.formats):
            c[f] = int((self.assignment == i).sum())
        return c


def solve(
    formats: list[str],
    damage: np.ndarray,  # [L, E, F], damage[..., 0] is keep (0)
    budget_bytes: int,
    solved_mask: np.ndarray,  # bool [L] layers with valid damage
    min_keep_per_layer: int = 0,
    default_fmt: str | None = None,
) -> Allocation:
    F = len(formats)
    fbytes = np.array([format_bytes(f) for f in formats], dtype=np.int64)
    order = np.argsort(fbytes)  # cheapest first
    scale = solved_mask.sum() / _NL  # budget share of solved layers
    layer_budget = int(budget_bytes * scale)

    # start everyone on the cheapest format
    cheapest = int(order[0])
    assignment = np.full((_NL, _NE), cheapest, dtype=np.int8)
    spent = int(solved_mask.sum()) * _NE * int(fbytes[cheapest])

    # per-expert convex hull of (bytes, damage) upgrade steps
    upgrades: list[tuple[float, int, int, int, int]] = []  # (-slope, l, e, from, to)
    li_all, = np.nonzero(solved_mask)
    for l in li_all:
        dmg = damage[l]  # [E, F]
        for e in range(_NE):
            cur = cheapest
            # hull walk: repeatedly find best slope improvement
            while True:
                best_j, best_slope = -1, 0.0
                for j in order:
                    j = int(j)
                    db = fbytes[j] - fbytes[cur]
                    dd = dmg[e, cur] - dmg[e, j]
                    if db <= 0 or dd <= 0:
                        continue
                    slope = dd / db
                    if slope > best_slope:
                        best_slope, best_j = slope, j
                if best_j < 0:
                    break
                upgrades.append((-best_slope, int(l), e, cur, best_j))
                cur = best_j

    # forced keeps (per-layer floor)
    keep_idx = formats.index("keep_mxfp4")
    if min_keep_per_layer > 0:
        for l in li_all:
            worst = np.argsort(-damage[l, :, assignment[l, 0]])[:min_keep_per_layer]
            for e in worst:
                spent += int(fbytes[keep_idx] - fbytes[assignment[l, e]])
                assignment[l, e] = keep_idx
        upgrades = [u for u in upgrades
                    if assignment[u[1], u[2]] != keep_idx]

    upgrades.sort()
    total_damage = float(
        damage[solved_mask][np.arange(solved_mask.sum())[:, None],
                            np.arange(_NE)[None, :],
                            assignment[solved_mask]].sum()
    )
    for negslope, l, e, frm, to in upgrades:
        if assignment[l, e] != frm:
            continue  # superseded by an earlier hull step
        cost = int(fbytes[to] - fbytes[frm])
        if spent + cost > layer_budget:
            continue
        spent += cost
        total_damage -= damage[l, e, frm] - damage[l, e, to]
        assignment[l, e] = to

    # solved-layer metrics before filling unsolved layers
    solved_bytes = int(fbytes[assignment[solved_mask]].sum())
    solved_params = int(solved_mask.sum()) * _NE * PARAMS_PER_EXPERT
    solved_bpw = solved_bytes * 8 / max(solved_params, 1)

    # unsolved layers get the primary demoted format (first after keep)
    if default_fmt is None:
        default_fmt = next(f for f in formats if f != "keep_mxfp4")
    dflt = formats.index(default_fmt)
    for l in range(_NL):
        if not solved_mask[l]:
            assignment[l, :] = dflt

    total_bytes = int(fbytes[assignment].sum())
    keep_per_layer = {
        int(C.MOE_LAYERS[l]): int((assignment[l] == keep_idx).sum())
        for l in range(_NL)
    }
    return Allocation(
        formats=formats,
        assignment=assignment,
        layers_solved=[int(C.MOE_LAYERS[l]) for l in li_all],
        total_bytes=total_bytes,
        achieved_bpw=total_bytes * 8 / C.TOTAL_EXPERT_PARAMS,
        solved_bytes=solved_bytes,
        solved_bpw=solved_bpw,
        total_damage=total_damage,
        keep_per_layer=keep_per_layer,
    )
