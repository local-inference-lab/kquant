from __future__ import annotations

import pytest
import torch

from kquant import constants as C
from scripts.summarize_qsrt_tp12_runtime import _layer_runtime_summary


def test_layer_runtime_summary_counts_cyclic_p24_ownership() -> None:
    modes_r13 = torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8)
    modes_r2 = torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8)
    modes_r13[0] = 1
    modes_r2[1] = 2
    routes = torch.arange(C.TOP_K, dtype=torch.int32).reshape(1, C.TOP_K)

    summary, per_rank_routes, per_rank_hits, max_rank_routes = (
        _layer_runtime_summary(
            layer=1,
            modes_r13=modes_r13,
            modes_r2=modes_r2,
            routes=routes,
        )
    )

    assert summary["selected_rate_shifted_experts"] == 2
    assert summary["selected_fc1_rate_shifted_experts"] == 1
    assert summary["selected_fc2_rate_shifted_experts"] == 1
    assert summary["selected_p24_expert_rank_assignments"] == 3
    assert summary["rate_shifted_routed_expert_slots"] == 2
    assert summary["tokens_routing_any_rate_shifted_expert"] == 1
    assert summary["p24_rank_route_executions"] == 3
    assert summary["p24_rank_route_fraction"] == pytest.approx(
        3 / (2 * 12 * C.TOP_K)
    )
    assert int(per_rank_routes.sum()) == 3
    assert int(per_rank_hits.sum()) == 3
    assert max_rank_routes.tolist() == [1]
    assert summary["slowest_rank_p24_routes_per_token_histogram"] == {"1": 1}


def test_layer_runtime_summary_rejects_invalid_route() -> None:
    modes_r13 = torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8)
    modes_r2 = torch.zeros(C.NUM_EXPERTS, dtype=torch.uint8)
    routes = torch.zeros((1, C.TOP_K), dtype=torch.int32)
    routes[0, 0] = C.NUM_EXPERTS

    with pytest.raises(ValueError, match="invalid expert ID"):
        _layer_runtime_summary(
            layer=1,
            modes_r13=modes_r13,
            modes_r2=modes_r2,
            routes=routes,
        )
