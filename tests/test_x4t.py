from __future__ import annotations

import pytest
import torch

from kquant.x4t import (
    X4T_POSITION_BITS,
    effective_x4t_bpw,
    pack_x4t_scale_components,
    x4t_scale_storage_bytes,
)


def _scale(rows: int = 32, columns: int = 17) -> torch.Tensor:
    result = torch.empty((rows, columns), dtype=torch.uint8)
    for row in range(rows):
        base = 110 + row % 7
        result[row] = torch.tensor(
            [base + (column & 1) for column in range(columns)], dtype=torch.uint8
        )
    result[0, 0] = 140
    result[17, columns - 1] = 3
    return result.contiguous()


def test_x4t_components_are_exact_and_deterministic() -> None:
    scale = _scale()
    fixed, exceptions = pack_x4t_scale_components(scale)
    repeated = pack_x4t_scale_components(scale.clone())

    assert torch.equal(fixed, repeated[0])
    assert torch.equal(exceptions, repeated[1])
    assert x4t_scale_storage_bytes(scale) == fixed.numel() + 4 * exceptions.numel()
    assert effective_x4t_bpw(scale) < 4.25


def test_x4t_components_close_over_fixed_and_exception_streams() -> None:
    scale = _scale(columns=8)
    fixed, exceptions = pack_x4t_scale_components(scale)

    assert fixed.dtype == torch.uint8
    assert exceptions.dtype == torch.uint32
    assert int(exceptions.numel()) == 2
    assert int(exceptions[0]) & ((1 << X4T_POSITION_BITS) - 1) == 0


def test_x4t_requires_complete_16_row_tiles() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        pack_x4t_scale_components(torch.zeros((15, 8), dtype=torch.uint8))
