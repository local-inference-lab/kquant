from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import kquant.pack.qsrt_candidates as packed_candidates

from kquant.capture import LayerSamples
from kquant.qsrt_candidates import (
    activation_block_contexts,
    functional_sse_by_request,
    index_expert_rows,
    layer_fallback_contexts,
    partition_requests,
    permutation_policy_contexts,
    request_documents,
    select_expert_rows,
    select_phase1_mode,
    select_phase1_rate_pair,
)
from kquant.qsrt import INTERMEDIATE_CHANNELS, RECORDS_PER_EXPERT
from kquant.pack.qsrt_candidates import (
    _candidate_middle_by_r13,
    _conditional_h2_by_r13,
    _defer_functional_row_sse,
    _finish_deferred_functional_sse,
    _folded_intermediate_conditioning,
    _prepare_deferred_functional_rows,
    _quantize_conditional_down_grid,
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


def test_input_only_fallback_contexts_cover_all_neuron_groups() -> None:
    samples = _samples()
    samples = LayerSamples(
        **{
            **samples.__dict__,
            "mid_values": torch.empty((0, INTERMEDIATE_CHANNELS)),
            "mid_weights": torch.empty(0),
            "mid_observations": torch.empty(0, dtype=torch.int64),
            "mid_experts": torch.empty(0, dtype=torch.int32),
            "mid_split": torch.empty(0, dtype=torch.int8),
        }
    )

    contexts, scores = layer_fallback_contexts(samples, {1: "document"})

    assert contexts.shape == (INTERMEDIATE_CHANNELS // 4,)
    assert scores.shape == contexts.shape
    assert torch.equal(torch.unique(contexts), torch.arange(RECORDS_PER_EXPERT))
    assert torch.bincount(contexts).unique().item() == 32


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


def test_index_expert_rows_matches_individual_selection() -> None:
    samples = _samples()
    requests = {1: "first", 3: "third"}
    index = index_expert_rows(samples, requests, num_experts=10)

    for expert in range(10):
        expected = select_expert_rows(samples, expert, requests)
        actual = index.select(expert)
        torch.testing.assert_close(actual.inputs, expected.inputs)
        torch.testing.assert_close(actual.gates, expected.gates)
        torch.testing.assert_close(actual.request_steps, expected.request_steps)


def test_index_expert_rows_sums_duplicate_route_gates() -> None:
    samples = _samples()
    samples = LayerSamples(
        **{
            **samples.__dict__,
            "input_experts": torch.tensor(
                [[7, 7], [2, 3], [7, 9]], dtype=torch.int32
            ),
        }
    )
    requests = {1: "first", 3: "third"}

    expected = select_expert_rows(samples, 7, requests)
    actual = index_expert_rows(samples, requests, num_experts=10).select(7)

    torch.testing.assert_close(actual.inputs, expected.inputs)
    torch.testing.assert_close(actual.gates, expected.gates)
    torch.testing.assert_close(actual.request_steps, expected.request_steps)


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


def test_identity_permutation_policy_keeps_logical_records() -> None:
    groups = INTERMEDIATE_CHANNELS // 4
    middle = torch.arange(1, groups + 1, dtype=torch.float32).sqrt().repeat_interleave(4)

    assignments, _scores = permutation_policy_contexts(
        middle.reshape(1, -1), torch.ones(1), policy="identity"
    )

    assert assignments[:32].tolist() == [0] * 32
    assert assignments[-32:].tolist() == [23] * 32
    assert torch.equal(torch.bincount(assignments), torch.full((24,), 32))


@pytest.mark.parametrize(
    "policy", ("energy_balanced", "stratified_energy_balanced")
)
def test_balanced_permutation_policies_are_complete_and_deterministic(policy) -> None:
    groups = INTERMEDIATE_CHANNELS // 4
    energy = torch.linspace(0.25, 9.0, groups)
    middle = energy.sqrt().repeat_interleave(4).reshape(1, -1)

    first, scores = permutation_policy_contexts(
        middle, torch.ones(1), policy=policy
    )
    second, _ = permutation_policy_contexts(middle, torch.ones(1), policy=policy)

    assert torch.equal(first, second)
    assert torch.equal(torch.bincount(first), torch.full((24,), 32))
    record_energy = torch.zeros(24).scatter_add_(0, first, scores)
    if policy == "energy_balanced":
        assert float(record_energy.max() - record_energy.min()) < 9.0
    else:
        assert float(record_energy[-4:].min()) > float(record_energy[:4].max())


def test_folded_conditioning_is_grid_aligned_and_constant_per_context() -> None:
    groups = INTERMEDIATE_CHANNELS // 4
    contexts = torch.div(torch.arange(groups), 32, rounding_mode="floor")
    scores = torch.linspace(0.25, 4.0, groups)

    profile = _folded_intermediate_conditioning(
        scores, contexts, power=0.5
    )

    assert profile is not None
    grouped = profile.reshape(-1, 4)
    assert torch.all(grouped == grouped[:, :1])
    for context in range(24):
        values = grouped[contexts == context]
        assert torch.all(values == values.flatten()[0])
    grid = torch.tensor([0.85, 0.925, 1.0, 1.075, 1.15])
    assert torch.all((profile[:, None] - grid[None, :]).abs().min(dim=1).values < 1e-6)


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


def test_conditional_h2_uses_each_decoded_upstream_candidate() -> None:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
    )
    gates = torch.tensor([1.0, 0.75, 0.5, 0.25])
    identity = torch.eye(2)
    upstream = {
        "w1": {
            0: SimpleNamespace(reconstruction=torch.eye(2)),
            1: SimpleNamespace(reconstruction=2.0 * torch.eye(2)),
        },
        "w3": {
            0: SimpleNamespace(reconstruction=torch.eye(2)),
            1: SimpleNamespace(reconstruction=torch.tensor([[1.0, 1.0], [0.0, 1.0]])),
        },
    }
    middle = _candidate_middle_by_r13(inputs, upstream, (0, 1))
    h2_by_r13, evidence = _conditional_h2_by_r13(
        inputs,
        gates,
        torch.ones(4, dtype=torch.bool),
        middle,
        global_h13=identity,
        global_h2=identity,
        device=torch.device("cpu"),
        hessian_policy="captured_blend",
        have_local_support=True,
    )

    assert set(h2_by_r13) == {0, 1}
    assert not torch.allclose(h2_by_r13[0], h2_by_r13[1])
    assert evidence["R0"]["h2_shrinkage_policy"] == (
        "weighted_oas_scaled_identity"
    )
    assert evidence["R1"]["h2_effective_sample_size"] == pytest.approx(
        evidence["R0"]["h2_effective_sample_size"]
    )


def test_conditional_h2_ignores_pooled_global_h2_values() -> None:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
    )
    gates = torch.tensor([1.0, 0.75, 0.5, 0.25])
    identity = torch.eye(2)
    upstream = {
        "w1": {0: SimpleNamespace(reconstruction=torch.eye(2))},
        "w3": {0: SimpleNamespace(reconstruction=torch.eye(2))},
    }
    middle = _candidate_middle_by_r13(inputs, upstream, (0,))

    baseline, _ = _conditional_h2_by_r13(
        inputs,
        gates,
        torch.ones(4, dtype=torch.bool),
        middle,
        global_h13=identity,
        global_h2=identity,
        device=torch.device("cpu"),
        hessian_policy="captured_blend",
        have_local_support=True,
    )
    pooled = torch.tensor([[100.0, 80.0], [80.0, 100.0]])
    adversarial, _ = _conditional_h2_by_r13(
        inputs,
        gates,
        torch.ones(4, dtype=torch.bool),
        middle,
        global_h13=identity,
        global_h2=pooled,
        device=torch.device("cpu"),
        hessian_policy="captured_blend",
        have_local_support=True,
    )
    torch.testing.assert_close(adversarial[0], baseline[0], rtol=0, atol=0)


@pytest.mark.parametrize("hessian_policy", ["identity", "captured_blend"])
def test_conditional_h2_fallback_is_plain_identity(hessian_policy: str) -> None:
    inputs = torch.tensor([[1.0, 0.0]])
    gates = torch.ones(1)
    upstream = {
        "w1": {0: SimpleNamespace(reconstruction=torch.eye(2))},
        "w3": {0: SimpleNamespace(reconstruction=torch.eye(2))},
    }
    middle = _candidate_middle_by_r13(inputs, upstream, (0,))
    pooled = torch.tensor([[100.0, 80.0], [80.0, 100.0]])

    actual, evidence = _conditional_h2_by_r13(
        inputs,
        gates,
        torch.ones(1, dtype=torch.bool),
        middle,
        global_h13=torch.eye(2),
        global_h2=pooled,
        device=torch.device("cpu"),
        hessian_policy=hessian_policy,
        have_local_support=hessian_policy == "identity",
    )
    torch.testing.assert_close(actual[0], torch.eye(2), rtol=0, atol=0)
    assert evidence["R0"]["h2_shrinkage_policy"] == "identity_fallback"


def test_conditional_down_grid_flattens_r13_times_r2(monkeypatch) -> None:
    calls = []

    def fake_batch(
        sources,
        contexts,
        *,
        modes_by_expert,
        hessians_by_expert,
        **kwargs,
    ):
        calls.append((sources, contexts, modes_by_expert, hessians_by_expert, kwargs))
        return [
            {
                "w2": {
                    mode.mode_id: (index, mode.mode_id)
                    for mode in modes
                }
            }
            for index, modes in enumerate(modes_by_expert)
        ]

    monkeypatch.setattr(
        packed_candidates, "quantize_qsrt_matrix_expert_batch", fake_batch
    )
    modes = (SimpleNamespace(mode_id=0), SimpleNamespace(mode_id=1))
    hessians = [{0: torch.eye(2), 1: 2.0 * torch.eye(2)}]
    result = _quantize_conditional_down_grid(
        [torch.ones(2, 2)],
        [torch.arange(2)],
        [modes],
        hessians,
        layer=1,
        device=torch.device("cpu"),
        shared_scale_scope=object(),
        quantizer_module=object(),
        codebook="test",
        layout="importance_ordered",
        ldlq_tf32=False,
        tailbite_context=8,
    )

    assert len(calls) == 1
    assert len(calls[0][0]) == 2  # one flattened encode group per r13
    assert calls[0][3][0]["w2"] is hessians[0][0]
    assert calls[0][3][1]["w2"] is hessians[0][1]
    assert set(result[0]) == {0, 1}
    assert set(result[0][0]) == {0, 1}
    assert set(result[0][1]) == {0, 1}


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
