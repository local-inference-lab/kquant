from types import SimpleNamespace

import pytest
import torch

import scripts.validate_qsrt_b12x_tp12 as validator
from scripts.validate_qsrt_b12x_tp12 import (
    COMPRESSED_FIELD_MAP,
    _b12x_codebook_contract,
    _compare_compressed_readers,
    _numerical_metrics,
    _reference_routed_experts,
    _right_hadamard,
    _sample_slots,
    _select_route_slots,
)


@pytest.mark.parametrize(
    ("codebook", "source_format", "expected_marker"),
    (
        ("sqg-normal-e4m3", "exl3_trellis_sqg_e4m3", {}),
        (
            "sqg-cheb-normal-e4m3",
            "exl3_trellis_sqg_cheb_e4m3",
            {},
        ),
    ),
)
def test_b12x_codebook_contract_selects_exact_procedural_marker(
    codebook: str,
    source_format: str,
    expected_marker: dict[str, int],
) -> None:
    actual_source, actual_marker = _b12x_codebook_contract(codebook)

    assert actual_source == source_format
    assert actual_marker == expected_marker


def test_b12x_codebook_contract_rejects_unknown_codebook() -> None:
    with pytest.raises(ValueError, match="unsupported materialized QSRT codebook"):
        _b12x_codebook_contract("learned")


def _reader_pair() -> tuple[SimpleNamespace, SimpleNamespace]:
    reference_values = {
        name: torch.arange(4, dtype=torch.int32) for name in COMPRESSED_FIELD_MAP
    }
    reference = SimpleNamespace(rank=5, **reference_values)
    serving = SimpleNamespace(
        rank=5,
        **{
            serving_name: reference_values[reference_name].clone()
            for reference_name, serving_name in COMPRESSED_FIELD_MAP.items()
        },
    )
    return reference, serving


def test_cross_reader_comparison_requires_every_tensor_to_be_exact() -> None:
    reference, serving = _reader_pair()
    _compare_compressed_readers(reference, serving)

    serving.fc2_pair_modes[2] += 1
    with pytest.raises(ValueError, match="fc2_pair_modes differs"):
        _compare_compressed_readers(reference, serving)


def test_cross_reader_comparison_rejects_contract_shape_drift() -> None:
    reference, serving = _reader_pair()
    serving.w2_trellis = serving.w2_trellis[:3]

    with pytest.raises(ValueError, match="w2_trellis contract mismatch"):
        _compare_compressed_readers(reference, serving)


def test_route_selection_prioritizes_sparse_p24_experts() -> None:
    payload = SimpleNamespace(
        fc1_pair_modes=torch.tensor([0, 1, 0, 0, 1], dtype=torch.int32),
        fc2_pair_modes=torch.tensor([0, 0, 1, 0, 0], dtype=torch.int32),
    )

    assert _select_route_slots(payload, 4).tolist() == [1, 2, 4, 0]


def test_route_selection_rejects_too_few_compressed_experts() -> None:
    payload = SimpleNamespace(
        fc1_pair_modes=torch.zeros(2, dtype=torch.int32),
        fc2_pair_modes=torch.zeros(2, dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="fewer than topk=3"):
        _select_route_slots(payload, 3)


def test_reference_hadamard_preserves_explicit_fp16_boundaries() -> None:
    value = torch.arange(128, dtype=torch.float32).reshape(1, -1) / 17
    hadamard = torch.eye(128, dtype=torch.float32)
    suh = torch.linspace(0.5, 1.5, 128)
    svh = torch.linspace(1.5, 0.5, 128)

    actual = _right_hadamard(
        value,
        hadamard,
        suh=suh,
        svh=svh,
        store_fp16=True,
    )
    expected = ((value * suh).to(torch.float16).float() * svh).to(torch.float16)

    assert torch.equal(actual, expected)


def test_complete_reference_route_runs_without_b12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the full independent oracle before a real slab is available."""

    monkeypatch.setattr(
        validator,
        "normalized_hadamard",
        lambda *, device: torch.eye(128, dtype=torch.float32, device=device),
    )

    def regularized_matrix(
        _reader: object,
        *,
        expert: int,
        matrix: str,
        rank: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert expert == 7
        assert rank == 5
        if matrix in ("w1", "w3"):
            weight = torch.zeros((3584, 256), dtype=torch.float32, device=device)
            weight[0, 0] = 1.0 if matrix == "w1" else 2.0
            return (
                weight,
                torch.ones(3584, dtype=torch.float16, device=device),
                torch.ones(256, dtype=torch.float16, device=device),
            )
        assert matrix == "w2"
        weight = torch.zeros((256, 3584), dtype=torch.float32, device=device)
        weight[0, 0] = 3.0
        return (
            weight,
            torch.ones(256, dtype=torch.float16, device=device),
            torch.ones(3584, dtype=torch.float16, device=device),
        )

    monkeypatch.setattr(validator, "_regularized_matrix", regularized_matrix)
    source = torch.zeros((1, 3584), dtype=torch.bfloat16)
    source[0, 0] = 1.0
    result = _reference_routed_experts(
        object(),
        source=source,
        selected_experts=torch.tensor([7], dtype=torch.int64),
        topk_weights=torch.tensor([[0.25]], dtype=torch.float32),
        rank=5,
    )

    gate = torch.tensor(1.0, dtype=torch.float16).float()
    up = torch.tensor(2.0, dtype=torch.float16).float()
    activated = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.float16)
    expected = (activated.float() * 3.0).to(torch.float16).float() * 0.25

    assert result.shape == (1, 3584)
    assert result.dtype == torch.float32
    assert result[0, 0] == expected
    assert torch.count_nonzero(result[0, 1:]) == 0


def test_numerical_metrics_and_sample_slots() -> None:
    value = torch.tensor([[3.0, 4.0]])
    metrics = _numerical_metrics(value, value)

    assert metrics["relative_error"] == 0.0
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["max_abs_error"] == 0.0
    assert _sample_slots(0) == ()
    assert _sample_slots(1) == (0,)
    assert _sample_slots(8) == (0, 4, 7)
