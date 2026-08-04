from __future__ import annotations

import numpy as np
import pytest

from kquant import constants as C
from kquant.pack.sqrt_c_allocation import (
    choose_sqrt_c_lagrangian,
    choose_sqrt_c_target,
    sqrt_c_total_container_bytes,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    damage = np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)
    costs = np.full(
        (C.NUM_MOE_LAYERS, C.NUM_EXPERTS),
        16_700_416,
        dtype=np.int64,
    )
    return damage, costs


def test_sqrt_c_lagrangian_selects_only_valuable_x4_experts() -> None:
    damage, costs = _inputs()
    damage[0, 3] = 10.0
    damage[1, 4] = 1.0

    allocation = choose_sqrt_c_lagrangian(
        damage,
        costs,
        lagrange_lambda=3e-7,
    )

    assert allocation.x4_mask[0, 3]
    assert not allocation.x4_mask[1, 4]
    assert allocation.x4_experts == 1
    assert allocation.container_bytes == sqrt_c_total_container_bytes(
        allocation.x4_mask, costs
    )


def test_sqrt_c_target_returns_supported_point_under_budget() -> None:
    damage, costs = _inputs()
    damage[0, :4] = [4.0, 3.0, 2.0, 1.0]
    all_compressed = sqrt_c_total_container_bytes(
        np.zeros_like(damage, dtype=np.bool_), costs
    )
    budget = all_compressed + 9_000_000

    allocation = choose_sqrt_c_target(
        damage,
        costs,
        target_container_bytes=budget,
        iterations=40,
    )

    assert allocation.container_bytes <= budget
    assert allocation.budget_slack_bytes == budget - allocation.container_bytes


def test_sqrt_c_target_rejects_below_all_compressed_storage() -> None:
    damage, costs = _inputs()
    minimum = sqrt_c_total_container_bytes(
        np.zeros_like(damage, dtype=np.bool_), costs
    )
    with pytest.raises(ValueError, match="below all-compressed"):
        choose_sqrt_c_target(
            damage,
            costs,
            target_container_bytes=minimum - 1,
        )
