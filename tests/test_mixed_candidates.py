from __future__ import annotations

import pytest
import torch

from kquant.capture import LayerSamples
from kquant.mixed_candidates import (
    activation_block_contexts,
    functional_sse_by_request,
    partition_requests,
    request_documents,
    select_expert_rows,
    select_phase1_mode,
    select_phase1_rate_pair,
)
from kquant.mixed_exl3 import INTERMEDIATE_CHANNELS, RECORDS_PER_EXPERT
from kquant.pack.mixed_candidates import (
    _defer_functional_row_sse,
    _finish_deferred_functional_sse,
    _prepare_deferred_functional_rows,
)


def _samples() -> LayerSamples:
    observations = torch.tensor([1 << 32, 2 << 32, 3 << 32], dtype=torch.int64)
    return LayerSamples(
        input_values=torch.arange(12, dtype=torch.float16).reshape(3, 4),
        input_weights=torch.ones(3),
        input_observations=observations,
        input_experts=torch.tensor([[4, 7], [2, 3], [7, 9]], dtype=torch.int32),
        input_gates=torch.tensor(
            [[0.25, 0.75], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32
        ),
        input_split=torch.zeros(3, dtype=torch.int8),
        routed_latent=torch.zeros((3, 4), dtype=torch.float16),
        mid_values=torch.zeros((1, INTERMEDIATE_CHANNELS), dtype=torch.float16),
        mid_weights=torch.ones(1),
        mid_observations=observations[:1],
        mid_experts=torch.tensor([7], dtype=torch.int32),
        mid_split=torch.zeros(1, dtype=torch.int8),
    )


def test_partition_requests_is_disjoint_and_complete() -> None:
    requests = {index: f"document-{index}" for index in range(1, 65)}

    result = partition_requests(requests)

    assert set(result.fit).isdisjoint(result.confirmation)
    assert result.all == requests
    assert set(result.fit.values()).isdisjoint(result.confirmation.values())


def test_request_documents_can_drop_repeated_capture_epochs() -> None:
    report = {
        "kind": "kquant_interim_calibration_corpus_run",
        "finalized": True,
        "planned_requests": 3,
        "completed_requests": 3,
        "documents": [
            {"document_hash": "doc-a"},
            {"document_hash": "doc-a"},
            {"document_hash": "doc-b"},
        ],
    }

    with pytest.raises(ValueError, match="must be unique"):
        request_documents(report)
    assert request_documents(report, deduplicate=True) == {
        1: "doc-a",
        3: "doc-b",
    }


def test_select_expert_rows_preserves_gate_and_request_pairing() -> None:
    rows = select_expert_rows(
        _samples(),
        7,
        {1: "first", 3: "third"},
    )

    torch.testing.assert_close(rows.inputs, _samples().input_values[[0, 2]])
    torch.testing.assert_close(rows.gates, torch.tensor([0.75, 0.2]))
    assert rows.request_steps.tolist() == [1, 3]
    assert rows.rows == 2
    assert rows.documents == 2


def test_activation_contexts_rank_complete_four_channel_groups() -> None:
    groups = INTERMEDIATE_CHANNELS // 4
    group_energy = torch.arange(1, groups + 1, dtype=torch.float32)
    middle = group_energy.sqrt().repeat_interleave(4).reshape(1, -1)

    assignments, scores = activation_block_contexts(
        middle,
        torch.ones(1),
    )

    assert assignments.shape == (groups,)
    assert torch.equal(torch.bincount(assignments), torch.full((24,), 32))
    assert assignments[:32].tolist() == [0] * 32
    assert assignments[-32:].tolist() == [RECORDS_PER_EXPERT - 1] * 32
    torch.testing.assert_close(scores, group_energy)


def test_functional_sse_is_clustered_by_request() -> None:
    reference = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]])
    candidate = torch.tensor([[2.0, 2.0], [0.0, 0.0], [3.0, 3.0]])
    gates = torch.tensor([0.5, 1.0, 0.25])
    steps = torch.tensor([4, 4, 9], dtype=torch.int64)

    sse, energy, counts = functional_sse_by_request(
        reference,
        candidate,
        gates,
        steps,
        {4: "a", 9: "b"},
    )

    torch.testing.assert_close(sse, torch.tensor([4.25, 0.25], dtype=torch.float64))
    torch.testing.assert_close(
        energy, torch.tensor([5.25, 0.625], dtype=torch.float64)
    )
    assert counts.tolist() == [2, 1]


@pytest.mark.parametrize("mask", [torch.tensor([True, False, True]), torch.zeros(3, dtype=torch.bool)])
def test_deferred_functional_sse_matches_reference(mask: torch.Tensor) -> None:
    reference = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 1.0]])
    candidates = (
        torch.tensor([[2.0, 2.0], [0.0, 0.0], [3.0, 3.0]]),
        torch.tensor([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]]),
    )
    gates = torch.tensor([0.5, 1.0, 0.25])
    steps = torch.tensor([4, 4, 9], dtype=torch.int64)
    requests = {4: "a", 9: "b"}
    plan = _prepare_deferred_functional_rows(
        reference,
        gates,
        steps,
        mask,
        requests,
        device=torch.device("cpu"),
    )
    keys = ((0, 0), (1, 2))
    actual = _finish_deferred_functional_sse(
        plan,
        keys,
        [_defer_functional_row_sse(plan, candidate) for candidate in candidates],
    )

    for key, candidate in zip(keys, candidates):
        expected_sse, expected_energy, expected_counts = functional_sse_by_request(
            reference[mask],
            candidate[mask],
            gates[mask],
            steps[mask],
            requests,
        )
        torch.testing.assert_close(actual[key], expected_sse, rtol=0, atol=0)
        torch.testing.assert_close(plan.reference_energy, expected_energy, rtol=0, atol=0)
        assert torch.equal(plan.counts, expected_counts)


def test_phase1_mode_requires_independent_confirmation_evidence() -> None:
    fit_counts = torch.ones(8, dtype=torch.int64)
    confirmation_counts = torch.ones(6, dtype=torch.int64)
    fit = {
        0: torch.full((8,), 10.0),
        1: torch.full((8,), 10.5),
        2: torch.full((8,), 8.0),
    }
    confirmation = {
        0: torch.full((6,), 10.0),
        1: torch.full((6,), 10.5),
        2: torch.full((6,), 8.0),
    }

    accepted = select_phase1_mode(
        fit,
        confirmation,
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        mode_ids=(0, 1, 2),
        bootstrap_replicates=200,
        seed=7,
    )
    assert accepted.proposed_r == 2
    assert accepted.selected_r == 2
    assert accepted.accepted
    assert accepted.confirmation_ci95[0] is not None
    assert accepted.confirmation_ci95[0] > 0

    confirmation[2] = torch.full((6,), 12.0)
    rejected = select_phase1_mode(
        fit,
        confirmation,
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        mode_ids=(0, 1, 2),
        bootstrap_replicates=200,
        seed=7,
    )
    assert rejected.proposed_r == 2
    assert rejected.selected_r == 0
    assert not rejected.accepted


def test_phase1_mode_falls_back_before_search_with_weak_fit_support() -> None:
    fit = {mode: torch.ones(5) for mode in (0, 1)}
    confirmation = {mode: torch.ones(5) for mode in (0, 1)}

    selected = select_phase1_mode(
        fit,
        confirmation,
        fit_counts=torch.ones(5, dtype=torch.int64),
        confirmation_counts=torch.ones(5, dtype=torch.int64),
        mode_ids=(0, 1),
        bootstrap_replicates=200,
    )

    assert selected.selected_r == 0
    assert selected.reason == "insufficient fit-document support"


def test_phase1_rate_pair_is_selected_on_disjoint_confirmation_rows() -> None:
    pairs = tuple((r13, r2) for r13 in (0, 1, 2) for r2 in (0, 1, 2))
    fit_counts = torch.ones(8, dtype=torch.int64)
    confirmation_counts = torch.ones(8, dtype=torch.int64)
    # The fitted encoder population prefers R0/R0.  This must not suppress a
    # rate transfer that consistently wins on rows excluded from construction.
    fit = {
        pair: torch.full((8,), 10.0 + pair[0] + pair[1]) for pair in pairs
    }
    confirmation = {pair: torch.full((8,), 10.0) for pair in pairs}
    confirmation[(2, 1)] = torch.full((8,), 7.0)

    selected = select_phase1_rate_pair(
        fit,
        confirmation,
        fit_counts=fit_counts,
        confirmation_counts=confirmation_counts,
        modes=pairs,
        bootstrap_replicates=200,
        seed=7,
    )

    assert selected.proposed == (2, 1)
    assert selected.selected == (2, 1)
    assert selected.accepted
    assert selected.confirmation_ci95[0] is not None
    assert selected.confirmation_ci95[0] > 0


def test_phase1_rate_pair_requires_both_support_partitions() -> None:
    pairs = ((0, 0), (0, 1))
    values = {pair: torch.ones(6) for pair in pairs}

    selected = select_phase1_rate_pair(
        values,
        values,
        fit_counts=torch.ones(6, dtype=torch.int64),
        confirmation_counts=torch.ones(3, dtype=torch.int64),
        modes=pairs,
        min_fit_documents=4,
        min_confirmation_documents=4,
        bootstrap_replicates=200,
    )

    assert selected.selected == (0, 0)
    assert selected.reason == "insufficient confirmation-document support"
