from __future__ import annotations

import torch

from kquant import constants as C
from kquant.pack.x4t_tp12 import _rank_scale_components
from kquant.x4t import pack_x4t_scale_components


def _scale(rows: int, columns: int) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.int64)[:, None]
    column = torch.arange(columns, dtype=torch.int64)[None, :]
    base = 90 + row % 17
    values = base + ((row + column) & 1)
    values[(row * 13 + column * 7) % 97 == 0] += 9
    return values.to(torch.uint8).contiguous()


def test_rank_scale_components_match_direct_rank_encoding() -> None:
    for matrix in ("w1", "w2"):
        rows, in_features = C.EXPERT_SHAPES[matrix]
        columns = in_features // C.MXFP4_BLOCK
        scale = _scale(rows, columns)
        fixed, exceptions = pack_x4t_scale_components(scale)
        for rank in range(12):
            actual = _rank_scale_components(
                matrix,
                fixed,
                exceptions,
                rank,
            )
            if matrix == "w1":
                rank_rows = rows // 12
                expected_scale = scale[
                    rank * rank_rows : (rank + 1) * rank_rows
                ].contiguous()
            else:
                rank_columns = columns // 12
                expected_scale = scale[
                    :,
                    rank * rank_columns : (rank + 1) * rank_columns,
                ].contiguous()
            expected_fixed, expected_exceptions = pack_x4t_scale_components(
                expected_scale
            )
            assert actual.rows == expected_scale.shape[0]
            assert actual.columns == expected_scale.shape[1]
            assert torch.equal(actual.fixed, expected_fixed)
            assert torch.equal(actual.exceptions, expected_exceptions)
