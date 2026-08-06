from __future__ import annotations

import torch

from scripts.select_qsrt_rotations import _select_one


def test_rotation_selection_keeps_draw_zero_when_it_is_best() -> None:
    selected, evidence = _select_one(
        {
            0: torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
            1: torch.tensor([1.1, 0.9, 1.1], dtype=torch.float64),
        },
        replicates=1_000,
        minimum_relative_improvement=0.0,
        seed=17,
    )

    assert selected == 0
    assert evidence["reason"] == "draw_zero_is_argmin"


def test_rotation_selection_accepts_a_consistent_material_improvement() -> None:
    selected, evidence = _select_one(
        {
            0: torch.ones(32, dtype=torch.float64),
            3: torch.full((32,), 0.98, dtype=torch.float64),
        },
        replicates=2_000,
        minimum_relative_improvement=0.005,
        seed=29,
    )

    assert selected == 3
    assert evidence["reason"] == "familywise_paired_lower_bound_passed"


def test_rotation_selection_rejects_an_improvement_below_the_margin() -> None:
    selected, evidence = _select_one(
        {
            0: torch.ones(32, dtype=torch.float64),
            2: torch.full((32,), 0.999, dtype=torch.float64),
        },
        replicates=2_000,
        minimum_relative_improvement=0.005,
        seed=31,
    )

    assert selected == 0
    assert evidence["proposed_draw"] == 2
    assert evidence["reason"] == "familywise_paired_lower_bound_did_not_pass"
