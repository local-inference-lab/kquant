import pytest
import torch

from kquant.tp_simulator import (
    column_parallel,
    kimi_topk,
    padded_size,
    reduce_partials,
    row_parallel_partials,
    situ,
)


@pytest.mark.parametrize(
    ("size", "tp_size", "expected"),
    [(7, 1, 7), (7, 3, 9), (896, 12, 900), (3584, 16, 3584)],
)
def test_padded_size(size: int, tp_size: int, expected: int) -> None:
    assert padded_size(size, tp_size) == expected


@pytest.mark.parametrize("tp_size", [1, 3, 4, 12, 16])
def test_column_parallel_matches_unsharded_with_padding(tp_size: int) -> None:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(2, 13, generator=generator)
    weight = torch.randn(17, 13, generator=generator)

    expected = torch.nn.functional.linear(inputs, weight)
    actual = column_parallel(inputs, weight, tp_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tp_size", [1, 3, 4, 12, 16])
def test_row_parallel_matches_unsharded_with_padding(tp_size: int) -> None:
    generator = torch.Generator().manual_seed(8)
    inputs = torch.randn(2, 17, generator=generator)
    weight = torch.randn(11, 17, generator=generator)

    expected = torch.nn.functional.linear(inputs, weight)
    actual = reduce_partials(
        row_parallel_partials(inputs, weight, tp_size),
        dtype=inputs.dtype,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_kimi_topk_bias_selects_but_does_not_weight_with_bias() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    correction = torch.tensor([[10.0, 0.0, 0.0, 0.0]]).squeeze(0)

    weights, ids = kimi_topk(
        logits,
        correction,
        2,
        routed_scaling_factor=2.0,
    )

    assert ids.tolist() == [[0, 3]]
    raw = logits.sigmoid()[0, ids[0].long()]
    expected = 2.0 * raw / raw.sum()
    torch.testing.assert_close(weights[0], expected)


def test_situ_matches_definition() -> None:
    gate = torch.tensor([[-5.0, 0.0, 5.0]])
    up = torch.tensor([[30.0, -2.0, 3.0]])
    expected = (
        4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        * (25.0 * torch.tanh(up / 25.0))
    )

    torch.testing.assert_close(situ(gate, up), expected)
