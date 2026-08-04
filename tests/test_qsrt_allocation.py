from __future__ import annotations

import numpy as np
import pytest

from kquant import constants as C
from kquant.pack.qsrt_allocation import (
    choose_qsrt_lagrangian,
    choose_qsrt_target,
    qsrt_total_container_bytes,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    damage = np.zeros((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)
    costs = np.full(
        (C.NUM_MOE_LAYERS, C.NUM_EXPERTS),
        16_700_416,
        dtype=np.int64,
    )
    return damage, costs


def test_qsrt_lagrangian_selects_only_valuable_x4t_experts() -> None:
    damage, costs = _inputs()
    damage[0, 3] = 10.0
    damage[1, 4] = 1.0

    allocation = choose_qsrt_lagrangian(
        damage,
        costs,
        lagrange_lambda=3e-7,
    )

    assert allocation.x4t_mask[0, 3]
    assert not allocation.x4t_mask[1, 4]
    assert allocation.x4t_experts == 1
    assert allocation.container_bytes == qsrt_total_container_bytes(
        allocation.x4t_mask, costs
    )


def test_qsrt_target_returns_supported_point_under_budget() -> None:
    damage, costs = _inputs()
    damage[0, :4] = [4.0, 3.0, 2.0, 1.0]
    all_compressed = qsrt_total_container_bytes(
        np.zeros_like(damage, dtype=np.bool_), costs
    )
    budget = all_compressed + 9_000_000

    allocation = choose_qsrt_target(
        damage,
        costs,
        target_container_bytes=budget,
        iterations=40,
    )

    assert allocation.container_bytes <= budget
    assert allocation.budget_slack_bytes == budget - allocation.container_bytes


def test_qsrt_target_rejects_below_all_compressed_storage() -> None:
    damage, costs = _inputs()
    minimum = qsrt_total_container_bytes(
        np.zeros_like(damage, dtype=np.bool_), costs
    )
    with pytest.raises(ValueError, match="below all-compressed"):
        choose_qsrt_target(
            damage,
            costs,
            target_container_bytes=minimum - 1,
        )
