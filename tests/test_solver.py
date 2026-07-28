import numpy as np

from kquant import constants as C
from kquant.rank.solver import format_bytes, solve


def _damage(nl=C.NUM_MOE_LAYERS, ne=C.NUM_EXPERTS, seed=0):
    rng = np.random.default_rng(seed)
    formats = ["keep_mxfp4", "nf3"]
    dmg = np.zeros((nl, ne, 2))
    dmg[:, :, 1] = rng.lognormal(0.0, 1.0, size=(nl, ne))
    return formats, dmg


def test_budget_monotone_keeps():
    formats, dmg = _damage()
    mask = np.ones(C.NUM_MOE_LAYERS, dtype=bool)
    total = C.NUM_MOE_LAYERS * C.NUM_EXPERTS
    b_lo = int(total * (format_bytes("nf3") + 0.05 * (format_bytes("keep_mxfp4") - format_bytes("nf3"))))
    b_hi = int(total * (format_bytes("nf3") + 0.20 * (format_bytes("keep_mxfp4") - format_bytes("nf3"))))
    a_lo = solve(formats, dmg, b_lo, mask)
    a_hi = solve(formats, dmg, b_hi, mask)
    assert a_lo.counts()["keep_mxfp4"] < a_hi.counts()["keep_mxfp4"]
    assert a_lo.total_bytes <= b_lo
    assert a_hi.total_bytes <= b_hi


def test_keeps_are_highest_damage():
    formats, dmg = _damage(seed=1)
    mask = np.ones(C.NUM_MOE_LAYERS, dtype=bool)
    total = C.NUM_MOE_LAYERS * C.NUM_EXPERTS
    budget = int(total * (format_bytes("nf3") + 0.10 * (format_bytes("keep_mxfp4") - format_bytes("nf3"))))
    a = solve(formats, dmg, budget, mask)
    keep = a.assignment == 0
    # with uniform byte costs, kept experts must dominate demoted ones globally
    kept_min = dmg[:, :, 1][keep].min()
    demoted_max = dmg[:, :, 1][~keep].max()
    assert kept_min >= demoted_max * 0.999


def test_min_keep_floor():
    formats, dmg = _damage(seed=2)
    mask = np.ones(C.NUM_MOE_LAYERS, dtype=bool)
    budget = int(C.NUM_MOE_LAYERS * C.NUM_EXPERTS * format_bytes("nf3") * 1.02)
    a = solve(formats, dmg, budget, mask, min_keep_per_layer=4)
    assert all(v >= 4 for v in a.keep_per_layer.values())


def test_unsolved_layers_get_default():
    formats, dmg = _damage(seed=3)
    mask = np.ones(C.NUM_MOE_LAYERS, dtype=bool)
    mask[10:] = False
    budget = int(C.NUM_MOE_LAYERS * C.NUM_EXPERTS * format_bytes("keep_mxfp4"))
    a = solve(formats, dmg, budget, mask)
    assert (a.assignment[10:] == formats.index("nf3")).all()
