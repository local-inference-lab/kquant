from itertools import product

import pytest
import torch

from kquant.e2m1_trellis import (
    E2M1_VALUES,
    decode_payload,
    encode_tailbiting,
    geometry,
    identity_transition_codes,
    opposite_zero_transition_codes,
    pack_payload,
    tailbiting_cost,
    unpack_payload,
)


def _brute_force_cost(source: torch.Tensor, bits: int, labels: tuple[int, ...]) -> float:
    spec = geometry(bits)
    best = float("inf")
    for branches in product(range(spec.branches), repeat=source.numel()):
        payload = torch.tensor([branches], dtype=torch.uint8)
        decoded = decode_payload(payload, bits, labels)[0]
        error = (E2M1_VALUES[source.long()] - E2M1_VALUES[decoded.long()]).square().sum()
        best = min(best, float(error))
    return best


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_pack_payload_round_trip(bits: int):
    generator = torch.Generator().manual_seed(1234 + bits)
    payload = torch.randint(0, 1 << bits, (7, 256), generator=generator, dtype=torch.uint8)
    packed = pack_payload(payload, bits)
    assert packed.shape == (7, 256 * bits // 8)
    assert torch.equal(unpack_payload(packed, bits, symbols=256), payload)


def test_opposite_zero_seed_uses_opposite_transition_corners():
    labels = opposite_zero_transition_codes()
    assert labels[0] == 0
    assert labels[15] == 8
    assert sorted(labels) == list(range(16))
    assert E2M1_VALUES[labels[0]] == E2M1_VALUES[labels[15]] == 0


@pytest.mark.parametrize("labels", (identity_transition_codes(), opposite_zero_transition_codes()))
def test_k4_has_raw_symbol_and_numeric_closure(labels: tuple[int, ...]):
    source = torch.arange(16, dtype=torch.uint8).repeat(3, 2)
    result = encode_tailbiting(source, 4, labels)
    assert torch.equal(result.reconstructed_codes, source)
    assert torch.count_nonzero(result.weighted_squared_error) == 0
    assert torch.equal(decode_payload(result.payload, 4, labels), source)


@pytest.mark.parametrize("bits", (2, 3))
def test_viterbi_matches_exhaustive_search(bits: int):
    labels = opposite_zero_transition_codes()
    source = torch.tensor([[0, 1, 8]], dtype=torch.uint8)
    cost = tailbiting_cost(source, bits, labels)
    assert float(cost[0]) == pytest.approx(_brute_force_cost(source[0], bits, labels))
    result = encode_tailbiting(source, bits, labels)
    assert float(result.weighted_squared_error[0]) == pytest.approx(float(cost[0]))


def test_duplicate_zero_labels_can_reconstruct_the_same_numeric_value():
    labels = opposite_zero_transition_codes()
    # K3 transition 0000 emits +0 and transition 1111 emits -0.  Repeating
    # branch 0 closes in state 0; repeating branch 7 closes in state 1.
    positive_path = torch.zeros((1, 4), dtype=torch.uint8)
    negative_path = torch.full((1, 4), 7, dtype=torch.uint8)
    positive = decode_payload(positive_path, 3, labels)
    negative = decode_payload(negative_path, 3, labels)
    assert torch.equal(positive, torch.zeros_like(positive))
    assert torch.equal(negative, torch.full_like(negative, 8))
    assert torch.equal(E2M1_VALUES[positive.long()], E2M1_VALUES[negative.long()])


def test_weighted_cost_matches_reconstruction():
    source = torch.tensor([[0, 1, 2, 3], [15, 14, 13, 12]], dtype=torch.uint8)
    weights = torch.tensor([[1.0, 2.0, 4.0, 8.0], [0.5, 1.5, 2.5, 3.5]])
    labels = opposite_zero_transition_codes()
    result = encode_tailbiting(source, 3, labels, weights=weights)
    expected = (
        E2M1_VALUES[source.long()] - E2M1_VALUES[result.reconstructed_codes.long()]
    ).square() * weights
    assert torch.allclose(result.weighted_squared_error, expected.sum(dim=1))


@pytest.mark.parametrize("bad", ((0,) * 16, tuple(range(15)), tuple(range(15)) + (16,)))
def test_rejects_malformed_transition_tables(bad: tuple[int, ...]):
    source = torch.zeros((1, 8), dtype=torch.uint8)
    with pytest.raises(ValueError):
        tailbiting_cost(source, 3, bad)
