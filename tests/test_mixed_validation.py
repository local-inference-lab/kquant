from __future__ import annotations

import numpy as np
import pytest
import torch

from kquant import constants as C
from kquant.pack.mixed_validation import (
    VALIDATION_DAMAGE_METRIC,
    official_expert_output,
    physical_expert_output,
    validate_validation_layer_metrics,
)
from kquant.tp_simulator import situ


def test_physical_expert_output_preserves_common_neuron_permutation() -> None:
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn((2, C.LATENT), generator=generator)
    w1 = torch.randn((C.EXPERT_INTER, C.LATENT), generator=generator) * 0.01
    w3 = torch.randn((C.EXPERT_INTER, C.LATENT), generator=generator) * 0.01
    w2 = torch.randn((C.LATENT, C.EXPERT_INTER), generator=generator) * 0.01
    permutation = torch.randperm(C.EXPERT_INTER, generator=generator)

    expected = situ(inputs @ w1.T, inputs @ w3.T) @ w2.T
    actual = physical_expert_output(
        inputs,
        w1[permutation].T.contiguous(),
        w3[permutation].T.contiguous(),
        w2[:, permutation].T.contiguous(),
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_official_expert_output_dequantizes_directly_on_requested_device() -> None:
    class RecordingStore:
        def __init__(self) -> None:
            self.devices: list[torch.device | str | None] = []
            self.weights = {
                "w1": torch.tensor(
                    [[1.0, 0.0, -1.0], [0.5, 0.25, 0.0]], dtype=torch.float32
                ),
                "w3": torch.tensor(
                    [[0.0, 1.0, 0.5], [-0.5, 0.0, 1.0]], dtype=torch.float32
                ),
                "w2": torch.tensor([[1.0, -0.5]], dtype=torch.float32),
            }

        def load_matrix(
            self,
            layer: int,
            expert: int,
            matrix: str,
            *,
            device: torch.device | str | None = None,
        ) -> torch.Tensor:
            assert layer == 1
            assert expert == 2
            self.devices.append(device)
            return self.weights[matrix].to(device=device)

    inputs = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)
    store = RecordingStore()
    requested = torch.device("cpu")

    actual = official_expert_output(
        store,
        inputs,
        layer=1,
        expert=2,
        device=requested,
    )
    gate = torch.nn.functional.linear(inputs, store.weights["w1"])
    up = torch.nn.functional.linear(inputs, store.weights["w3"])
    expected = torch.nn.functional.linear(situ(gate, up), store.weights["w2"])

    torch.testing.assert_close(actual, expected)
    assert store.devices == [requested, requested, requested]


def _validation_metrics() -> dict[str, torch.Tensor]:
    sse = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
    reference = torch.tensor([[10.0, 20.0], [0.0, 0.0]], dtype=torch.float64)
    return {
        "expert_ids": torch.tensor([0, 1], dtype=torch.int16),
        "selected_r13": torch.tensor([1, 0], dtype=torch.uint8),
        "selected_r2": torch.tensor([2, 0], dtype=torch.uint8),
        "validation_sse": sse,
        "validation_reference_energy": reference,
        "validation_counts": torch.tensor([[2, 3], [0, 0]], dtype=torch.int32),
        "validation_gate_square_sum": torch.tensor(
            [1.5, 0.0], dtype=torch.float64
        ),
        VALIDATION_DAMAGE_METRIC: sse.sum(dim=1),
        "selection_corpus_damage": torch.tensor(
            [12.0, 4.0], dtype=torch.float64
        ),
    }


def test_validation_layer_metrics_close_to_candidate_and_documents() -> None:
    validated = validate_validation_layer_metrics(
        _validation_metrics(),
        selected_r13=np.array([1, 0], dtype=np.uint8),
        selected_r2=np.array([2, 0], dtype=np.uint8),
        selection_damage=np.array([12.0, 4.0]),
        documents=2,
        expert_ids=(0, 1),
    )

    assert validated["damage"].tolist() == [3.0, 0.0]
    assert validated["per_document_damage"].tolist() == [
        [1.0, 2.0],
        [0.0, 0.0],
    ]
    assert validated["reference_energy"].tolist() == [30.0, 0.0]
    assert validated["routed_rows"].tolist() == [5, 0]
    assert validated["supported_documents"].tolist() == [2, 0]


def test_validation_layer_metrics_reject_damage_drift() -> None:
    metrics = _validation_metrics()
    metrics[VALIDATION_DAMAGE_METRIC][0] += 1
    with pytest.raises(ValueError, match="per-document SSE"):
        validate_validation_layer_metrics(
            metrics,
            selected_r13=np.array([1, 0], dtype=np.uint8),
            selected_r2=np.array([2, 0], dtype=np.uint8),
            selection_damage=np.array([12.0, 4.0]),
            documents=2,
            expert_ids=(0, 1),
        )
