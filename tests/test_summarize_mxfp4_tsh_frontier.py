from __future__ import annotations

from scripts.summarize_mxfp4_tsh_frontier import (
    _greedy_binary_selection,
    _k4_on_per_expert_convex_hull,
)


def _record(*, byte_count: int, confirmation: float, validation: float) -> dict:
    return {
        "bytes": byte_count,
        "weights": 800,
        "confirmation_sse": confirmation,
        "validation_sse": validation,
    }


def test_exact_tier_selection_uses_confirmation_gain_per_byte() -> None:
    low = {
        (24, 0): _record(byte_count=300, confirmation=10.0, validation=4.0),
        (24, 1): _record(byte_count=300, confirmation=8.0, validation=9.0),
    }
    high = {
        (24, 0): _record(byte_count=400, confirmation=0.0, validation=0.0),
        (24, 1): _record(byte_count=350, confirmation=0.0, validation=0.0),
    }

    records, selected = _greedy_binary_selection(
        low,
        high,
        target_bpw=3.25,
        selection_metric="confirmation_sse",
    )

    assert selected == [(24, 1)]
    assert records[(24, 0)] is low[(24, 0)]
    assert records[(24, 1)] is high[(24, 1)]


def test_k4_above_chord_is_not_on_convex_hull() -> None:
    k3 = _record(byte_count=300, confirmation=1.0, validation=1.0)
    k4 = _record(byte_count=400, confirmation=0.25, validation=0.25)
    exact = _record(byte_count=404, confirmation=0.0, validation=0.0)

    assert not _k4_on_per_expert_convex_hull(
        k3, k4, exact, metric="confirmation_sse"
    )


def test_exceptionally_good_k4_can_remain_on_convex_hull() -> None:
    k3 = _record(byte_count=300, confirmation=1.0, validation=1.0)
    k4 = _record(byte_count=400, confirmation=0.01, validation=0.01)
    exact = _record(byte_count=404, confirmation=0.0, validation=0.0)

    assert _k4_on_per_expert_convex_hull(
        k3, k4, exact, metric="confirmation_sse"
    )
