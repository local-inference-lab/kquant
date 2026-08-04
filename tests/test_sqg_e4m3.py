from __future__ import annotations

import pytest
import torch

from kquant.exl3_reference import (
    CODEBOOK_SQG_NORMAL_E4M3,
    decode_mixed_regularized_weight,
    decode_regularized_weight,
)
from kquant.sqg_e4m3 import (
    sqg_e4m3_bytes,
    sqg_e4m3_bytes_from_rank_lut,
    sqg_e4m3_codebook,
    sqg_e4m3_codebook_from_rank_lut,
    sqg_rank_permutation,
)


@pytest.mark.parametrize("bits", (2, 3, 4, 5))
def test_sqg_preprojection_mapping_is_a_permutation(bits: int) -> None:
    ranks = sqg_rank_permutation(bits)
    assert ranks.dtype == torch.int64
    assert ranks.shape == (1 << 16,)
    assert torch.equal(torch.sort(ranks).values, torch.arange(1 << 16))


@pytest.mark.parametrize("bits", (2, 3, 4, 5))
@pytest.mark.parametrize("mode", ("normal", "tail"))
def test_sqg_codebook_round_trips_exact_e4m3(bits: int, mode: str) -> None:
    raw = sqg_e4m3_bytes(bits, mode)
    codebook = sqg_e4m3_codebook(bits, mode)
    assert raw.dtype == torch.uint8
    assert codebook.dtype == torch.float16
    assert raw.shape == codebook.shape == (1 << 16,)
    assert torch.equal(codebook.to(torch.float8_e4m3fn).view(torch.uint8), raw)
    assert bool(torch.isfinite(codebook).all())


@pytest.mark.parametrize("bits", (2, 3, 4, 5))
def test_sqg_each_state_spans_all_coarse_strata(bits: int) -> None:
    ranks = sqg_rank_permutation(bits)
    width = 16 - bits
    strata = (ranks >> width).reshape(1 << width, 1 << bits)
    expected = torch.arange(1 << bits).expand_as(strata)
    assert torch.equal(torch.sort(strata, dim=1).values, expected)


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_shared_rank_lut_preserves_one_graph_for_every_rate(bits: int) -> None:
    rank_lut = torch.arange(1 << 16, dtype=torch.int64).remainder(126).to(torch.uint8)
    raw = sqg_e4m3_bytes_from_rank_lut(bits, rank_lut)
    expected = rank_lut.index_select(0, sqg_rank_permutation(bits))
    assert torch.equal(raw, expected)
    assert torch.equal(
        sqg_e4m3_codebook_from_rank_lut(bits, rank_lut)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8),
        expected,
    )


def test_shared_rank_lut_rejects_nonfinite_e4m3() -> None:
    rank_lut = torch.zeros(1 << 16, dtype=torch.uint8)
    rank_lut[0] = 0x7F
    with pytest.raises(ValueError, match="finite E4M3"):
        sqg_e4m3_bytes_from_rank_lut(2, rank_lut)


def test_reference_decoder_accepts_custom_codebook() -> None:
    states = torch.arange(256, dtype=torch.int16).reshape(1, 1, 256)
    codebook = sqg_e4m3_codebook(3, "normal")
    decoded = decode_regularized_weight(
        states, codebook_values=codebook
    )
    assert decoded.shape == (16, 16)
    assert torch.equal(
        torch.sort(decoded.flatten()).values,
        torch.sort(codebook[:256].float()).values,
    )


@pytest.mark.parametrize("rate_axis", ("k", "n"))
def test_reference_decoder_applies_rate_specific_sqg_tables(rate_axis: str) -> None:
    states = torch.arange(3 * 3 * 256, dtype=torch.int32).to(torch.int16).reshape(
        3, 3, 256
    )
    tile_bits = (2, 3, 4)
    decoded = decode_mixed_regularized_weight(
        states,
        rate_axis=rate_axis,
        tile_bits=tile_bits,
        codebook=CODEBOOK_SQG_NORMAL_E4M3,
    )
    axis = 0 if rate_axis == "k" else 1
    pieces = []
    for tile, bits in enumerate(tile_bits):
        selected = states.narrow(axis, tile, 1)
        pieces.append(
            decode_regularized_weight(
                selected,
                codebook=CODEBOOK_SQG_NORMAL_E4M3,
                bits=bits,
            )
        )
    expected = torch.cat(pieces, dim=0 if rate_axis == "k" else 1)
    assert torch.equal(decoded, expected)
