import pytest
import torch

from kquant.capture import LayerSamples
from scripts.experiment_coupled_exl3 import (
    _coupled_activation_metrics,
    _expert_output,
    _selected_expert_rows,
)


def _samples():
    request_steps = (0, 1, 1, 2)
    observations = torch.tensor(
        [(step << 32) | index for index, step in enumerate(request_steps)],
        dtype=torch.int64,
    )
    return LayerSamples(
        input_values=torch.tensor(
            [[100.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
        ),
        input_weights=torch.ones(4),
        input_observations=observations,
        input_experts=torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]]),
        input_gates=torch.tensor(
            [[0.9, 0.1], [0.5, 0.2], [0.3, 0.7], [0.4, 0.6]]
        ),
        input_split=torch.zeros(4, dtype=torch.int8),
        routed_latent=torch.ones(4, 2),
        mid_values=torch.empty(0, 2),
        mid_weights=torch.empty(0),
        mid_observations=torch.empty(0, dtype=torch.int64),
        mid_experts=torch.empty(0, dtype=torch.int32),
        mid_split=torch.empty(0, dtype=torch.int8),
    )


def test_selected_expert_rows_exclude_startup_epoch_and_keep_applied_gate():
    inputs, gates, _mixture, request_steps = _selected_expert_rows(
        _samples(), 0, None, {1: "doc-a", 2: "doc-b"}
    )
    assert inputs.shape[0] == 3
    assert gates.tolist() == pytest.approx([0.5, 0.7, 0.6])
    assert request_steps.tolist() == [1, 1, 2]


def test_coupled_metrics_close_exact_document_sums():
    samples = _samples()
    identity = torch.eye(2)
    source = (identity, identity, identity)
    reconstruction = (identity, identity, torch.zeros_like(identity))
    metrics = _coupled_activation_metrics(
        source,
        reconstruction,
        training_samples=samples,
        validation_samples=None,
        expert=0,
        device=torch.device("cpu"),
        training_request_documents={1: "doc-a", 2: "doc-b"},
        validation_request_documents={1: "doc-a", 2: "doc-b"},
    )["train"]
    assert metrics is not None
    selected_inputs = samples.input_values[1:]
    reference = _expert_output(selected_inputs, *source)
    gates = torch.tensor([0.5, 0.7, 0.6])
    expected = float((reference.double().square().sum(dim=1) * gates.square()).sum())
    assert metrics["rows"] == 3
    assert metrics["documents"] == 2
    assert metrics["route_weighted_squared_error"] == pytest.approx(expected)
    assert metrics["route_weighted_reference_energy"] == pytest.approx(expected)
    assert metrics["route_weighted_expert_nmse"] == pytest.approx(1.0)
    assert sum(
        row["route_weighted_squared_error"]
        for row in metrics["request_contributions"]
    ) == pytest.approx(expected)


def test_coupled_metrics_expose_a_disjoint_training_confirmation_fold():
    samples = _samples()
    identity = torch.eye(2)
    metrics = _coupled_activation_metrics(
        (identity, identity, identity),
        (identity, identity, torch.zeros_like(identity)),
        training_samples=samples,
        validation_samples=samples,
        expert=0,
        device=torch.device("cpu"),
        training_request_documents={1: "doc-a"},
        training_confirmation_documents={2: "doc-b"},
        validation_request_documents={1: "doc-a", 2: "doc-b"},
    )

    assert metrics["train"]["documents"] == 1
    assert metrics["train"]["rows"] == 2
    assert metrics["confirmation"]["documents"] == 1
    assert metrics["confirmation"]["rows"] == 1
    assert metrics["validation"]["documents"] == 2
    assert metrics["validation"]["rows"] == 3
