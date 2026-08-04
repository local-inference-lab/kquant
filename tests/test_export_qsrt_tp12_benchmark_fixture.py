from __future__ import annotations

import pytest
import torch

from kquant import constants as C
from scripts.export_qsrt_tp12_benchmark_fixture import (
    _selected_rows,
    build_fixture_tensors,
)


def test_fixture_preserves_routes_gates_and_cyclic_mode_ownership() -> None:
    modes_r13 = torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8)
    modes_r2 = torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8)
    modes_r13[0] = 1
    modes_r13[1] = 2
    modes_r2[1] = 1
    modes_r2[2] = 3
    routes = torch.arange(3 * C.TOP_K, dtype=torch.int32).reshape(3, C.TOP_K)
    gates = routes.float() / 100
    observations = torch.tensor([10, 20, 30], dtype=torch.int64)

    tensors, summary = build_fixture_tensors(
        layer=1,
        rank=0,
        modes_r13=modes_r13,
        modes_r2=modes_r2,
        routes=routes,
        gates=gates,
        observations=observations,
        row_limit=3,
        seed=7,
    )

    assert torch.equal(tensors["topk_ids"], routes)
    assert torch.equal(tensors["topk_weights"], gates)
    assert torch.equal(tensors["observation_ids"], observations)
    assert not torch.equal(tensors["fc1_pair_modes"], tensors["fc2_pair_modes"])
    assert summary["fc1_p24_experts"] == int(tensors["fc1_pair_modes"].sum())
    assert summary["fc2_p24_experts"] == int(tensors["fc2_pair_modes"].sum())
    assert summary["fc1_p24_route_executions"] == int(
        tensors["fc1_pair_modes"][routes.long()].sum()
    )
    assert summary["fc2_p24_route_executions"] == int(
        tensors["fc2_pair_modes"][routes.long()].sum()
    )


def test_fixture_row_selection_is_seeded_bounded_and_sorted() -> None:
    first = _selected_rows(100, 17, 123)
    second = _selected_rows(100, 17, 123)

    assert torch.equal(first, second)
    assert first.numel() == 17
    assert bool(torch.all(first[1:] > first[:-1]))
    assert _selected_rows(5, 20, 3).tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="positive"):
        _selected_rows(5, 0, 3)
