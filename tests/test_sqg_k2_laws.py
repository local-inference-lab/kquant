from __future__ import annotations

import hashlib

import pytest
import torch

from kquant.sqg_e4m3 import (
    sqg_cheb_normal_rank_e4m3_bytes,
    sqg_k2_tail_compressed_rank_e4m3_bytes,
)


def _decoded(raw: torch.Tensor) -> torch.Tensor:
    return raw.view(torch.float8_e4m3fn).float()


def test_sqg_cheb_normal_reference_is_exact() -> None:
    raw = sqg_cheb_normal_rank_e4m3_bytes()

    assert hashlib.sha256(bytes(raw.tolist())).hexdigest() == (
        "4ccef33598b9a772a6c48023030d87c46501255858077eab4561657ffe63a4b0"
    )
    assert torch.unique(raw).numel() == 155
    assert torch.unique(raw[_decoded(raw) == 0]).tolist() == [0]


@pytest.mark.parametrize(
    ("law", "parameter"),
    (("asinh", 4.0), ("rational", 0.02)),
)
def test_sqg_k2_tail_laws_are_finite_monotone_and_symmetric(
    law: str, parameter: float
) -> None:
    raw = sqg_k2_tail_compressed_rank_e4m3_bytes(law, parameter)
    values = _decoded(raw)

    assert torch.isfinite(values).all()
    assert torch.all(values[1:] >= values[:-1])
    assert torch.equal(values, -values.flip(0))
    assert torch.unique(raw[values == 0]).tolist() == [0]


def test_sqg_k2_tail_laws_validate_parameters() -> None:
    with pytest.raises(ValueError, match="has no parameter"):
        sqg_k2_tail_compressed_rank_e4m3_bytes("normal", 1.0)
    with pytest.raises(ValueError, match="positive parameter"):
        sqg_k2_tail_compressed_rank_e4m3_bytes("asinh", 0.0)
    with pytest.raises(ValueError, match="unsupported"):
        sqg_k2_tail_compressed_rank_e4m3_bytes("unknown", 1.0)


def test_sqg_k2_q33_changes_only_outer_quartiles() -> None:
    normal = sqg_cheb_normal_rank_e4m3_bytes()
    q33 = sqg_k2_tail_compressed_rank_e4m3_bytes("outer-slope", 33.0 / 32.0)

    assert torch.equal(q33[1 << 14 : 3 << 14], normal[1 << 14 : 3 << 14])
    assert not torch.equal(q33[: 1 << 14], normal[: 1 << 14])
    assert not torch.equal(q33[3 << 14 :], normal[3 << 14 :])
    assert hashlib.sha256(bytes(q33.tolist())).hexdigest() == (
        "b4edcdabc3c57c781024f1a4f52189120f2d23528cf5d9cef9d48a4f7165cd36"
    )


@pytest.mark.parametrize(
    "law", ("normal-clipped", "r44-full", "r44-clipped")
)
def test_sqg_k2_factorial_controls_are_finite_monotone(law: str) -> None:
    raw = sqg_k2_tail_compressed_rank_e4m3_bytes(law)
    values = _decoded(raw)
    assert torch.isfinite(values).all()
    assert torch.all(values[1:] >= values[:-1])
    assert torch.equal(values, -values.flip(0))
