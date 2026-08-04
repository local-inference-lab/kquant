from __future__ import annotations

import torch

from kquant.exl3_reference import decode_mul1_e4m3_states
from scripts.experiment_trellis_e4m3_procedural_v2 import (
    _alias_summary,
    _premix,
    _premix_many,
    _states,
    mul1_permuted_codebook,
    mul1_qd_codebook,
    mul1_qd_multixor_codebook,
    weighted_dp4a_codebook,
)


def test_zero_shift_permuted_candidate_is_published_mul1_e4m3() -> None:
    candidate = mul1_permuted_codebook(0)
    expected = decode_mul1_e4m3_states(_states()).float()
    assert torch.equal(candidate.codebook, expected)


def test_left_xorshift_is_a_permutation() -> None:
    states = _states()
    for shift in (1, 7, 8, 15):
        mixed = _premix(states, shift)
        assert torch.unique(mixed).numel() == 1 << 16


def test_multitap_left_xorshift_is_a_permutation() -> None:
    mixed = _premix_many(_states(), (8, 9, 11))
    assert torch.unique(mixed).numel() == 1 << 16
    assert torch.equal(mixed & 0x1F, _states() & 0x1F)


def test_qd_reaches_more_e4m3_values_and_is_finite() -> None:
    baseline = mul1_permuted_codebook(0).codebook
    candidate = mul1_qd_codebook(
        left_shift=8,
        dither_bits=3,
        dither_shift=7,
        phase=1.25,
    ).codebook
    assert bool(torch.isfinite(candidate).all())
    assert torch.unique(candidate).numel() > torch.unique(baseline).numel()
    assert torch.unique(candidate).numel() >= 140


def test_left_xorshift_creates_useful_k3_aliases() -> None:
    baseline = mul1_permuted_codebook(0).codebook
    candidate = mul1_permuted_codebook(8).codebook
    assert _alias_summary(baseline, 3)["states_with_alias_fraction"] == 0.0
    alias_fraction = _alias_summary(candidate, 3)["states_with_alias_fraction"]
    assert 0.15 < alias_fraction < 0.30


def test_multitap_qd_has_dense_alphabet_and_k5_overlap() -> None:
    candidate = mul1_qd_multixor_codebook(
        left_shifts=(8, 9, 11),
        dither_bits=3,
        dither_shift=7,
        phase=1.25,
    ).codebook
    assert torch.unique(candidate).numel() >= 145
    k5 = _alias_summary(candidate, 5)
    assert k5["states_with_alias_fraction"] > 0.99
    assert 25.0 < k5["mean_distinct_labels_per_state"] < 28.0


def test_weighted_dp4a_candidate_is_reproducible_e4m3() -> None:
    first = weighted_dp4a_codebook(
        weights=(1, 1, 1, 2), left_shift=8, phase=1.125
    )
    second = weighted_dp4a_codebook(
        weights=(1, 1, 1, 2), left_shift=8, phase=1.125
    )
    assert torch.equal(first.codebook, second.codebook)
    assert first.codebook.dtype == torch.float32
    assert bool(torch.isfinite(first.codebook).all())
