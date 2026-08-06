from __future__ import annotations

import pytest
import torch

from kquant.exl3_reference import (
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3,
    CODEBOOK_SQG_NORMAL_E4M3,
    decode_qsrt_regularized_weight,
    decode_regularized_weight,
)
from kquant.sqg_e4m3 import (
    sqg_cheb_normal_e4m3_bytes,
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_codebook_bytes,
    sqg_e4m3_bytes,
    sqg_e4m3_bytes_from_rank_lut,
    sqg_e4m3_codebook,
    sqg_e4m3_codebook_from_rank_lut,
    sqg_k2_eight_stratum_rank_permutation,
    sqg_rank_permutation,
)


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_sqg_preprojection_mapping_is_a_permutation(bits: int) -> None:
    ranks = sqg_rank_permutation(bits)
    assert ranks.dtype == torch.int64
    assert ranks.shape == (1 << 16,)
    assert torch.equal(torch.sort(ranks).values, torch.arange(1 << 16))


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_sqg_codebook_round_trips_exact_e4m3(bits: int) -> None:
    raw = sqg_e4m3_bytes(bits, "normal")
    codebook = sqg_e4m3_codebook(bits, "normal")
    assert raw.dtype == torch.uint8
    assert codebook.dtype == torch.float16
    assert raw.shape == codebook.shape == (1 << 16,)
    assert torch.equal(codebook.to(torch.float8_e4m3fn).view(torch.uint8), raw)
    assert bool(torch.isfinite(codebook).all())


@pytest.mark.parametrize("bits", (2, 3, 4))
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


@pytest.mark.parametrize("bits", (2, 3, 4))
def test_sqg_cheb_normal_named_codebook_uses_exact_shared_rank_law(bits: int) -> None:
    expected = sqg_cheb_normal_rank_e4m3_bytes().index_select(
        0, sqg_rank_permutation(bits)
    )
    assert torch.equal(sqg_cheb_normal_e4m3_bytes(bits), expected)
    assert torch.equal(
        sqg_codebook_bytes(bits, CODEBOOK_SQG_CHEB_NORMAL_E4M3), expected
    )
    assert not torch.equal(
        expected, sqg_codebook_bytes(bits, CODEBOOK_SQG_NORMAL_E4M3)
    )


@pytest.mark.parametrize("bits", (2, 3, 4))
@pytest.mark.parametrize("rate_axis", ("k", "n"))
def test_w2_k2_q8h4_profile_changes_only_k_axis_k2(
    bits: int, rate_axis: str
) -> None:
    actual = sqg_codebook_bytes(
        bits,
        CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3,
        rate_axis=rate_axis,
    )
    rank_lut = sqg_cheb_normal_rank_e4m3_bytes()
    ranks = (
        sqg_k2_eight_stratum_rank_permutation(4)
        if bits == 2 and rate_axis == "k"
        else sqg_rank_permutation(bits)
    )
    assert torch.equal(actual, rank_lut.index_select(0, ranks))
    if bits == 2 and rate_axis == "k":
        assert not torch.equal(
            actual, sqg_codebook_bytes(bits, CODEBOOK_SQG_CHEB_NORMAL_E4M3)
        )


def test_w2_k2_q8h4_profile_requires_rate_axis() -> None:
    with pytest.raises(ValueError, match="requires rate_axis"):
        sqg_codebook_bytes(2, CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3)


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
    decoded = decode_qsrt_regularized_weight(
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


def test_reference_decoder_applies_w2_only_k2_q8h4_profile() -> None:
    states = torch.arange(3 * 3 * 256, dtype=torch.int32).to(torch.int16).reshape(
        3, 3, 256
    )
    tile_bits = (2, 3, 4)
    down = decode_qsrt_regularized_weight(
        states,
        rate_axis="k",
        tile_bits=tile_bits,
        codebook=CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3,
    )
    upstream = decode_qsrt_regularized_weight(
        states,
        rate_axis="n",
        tile_bits=tile_bits,
        codebook=CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3,
    )
    native_down = decode_qsrt_regularized_weight(
        states,
        rate_axis="k",
        tile_bits=tile_bits,
        codebook=CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    )
    native_upstream = decode_qsrt_regularized_weight(
        states,
        rate_axis="n",
        tile_bits=tile_bits,
        codebook=CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    )
    assert not torch.equal(down[:16], native_down[:16])
    assert torch.equal(down[16:], native_down[16:])
    assert torch.equal(upstream, native_upstream)
