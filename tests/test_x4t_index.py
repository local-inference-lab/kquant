from __future__ import annotations

import numpy as np
import pytest

from kquant.pack.x4t_index import (
    X4T_COST_INDEX_SCHEMA_VERSION,
    X4T_COST_LAYER_KIND,
    validate_x4t_cost_layer,
)
from kquant.x4t import (
    X4T_EXPERTS_PER_LAYER,
    X4T_LAYER_FIXED_BYTES,
    X4T_MATRIX_ORDER,
    x4t_matrix_storage_bytes_from_exception_count,
)


def _document() -> dict:
    exceptions = [[0, 1, 2] for _ in range(X4T_EXPERTS_PER_LAYER)]
    matrix = [
        [
            x4t_matrix_storage_bytes_from_exception_count(name, count)
            for name, count in zip(X4T_MATRIX_ORDER, row, strict=True)
        ]
        for row in exceptions
    ]
    expert = [sum(row) + 28 for row in matrix]
    return {
        "kind": X4T_COST_LAYER_KIND,
        "schema_version": X4T_COST_INDEX_SCHEMA_VERSION,
        "source_revision": "revision",
        "layer": 24,
        "matrix_order": list(X4T_MATRIX_ORDER),
        "matrix_storage_bytes": matrix,
        "matrix_exception_counts": exceptions,
        "expert_storage_bytes": expert,
        "all_x4t_sidecar_bytes": X4T_LAYER_FIXED_BYTES + sum(expert),
    }


def test_x4t_cost_layer_validation_closes_exact_bytes() -> None:
    matrix, expert, exceptions = validate_x4t_cost_layer(
        _document(), source_revision="revision"
    )

    assert matrix.shape == (896, 3)
    assert expert.shape == (896,)
    assert exceptions.shape == matrix.shape
    assert np.array_equal(expert, matrix.sum(axis=1) + 28)


def test_x4t_cost_layer_rejects_inconsistent_and_mismatched_costs() -> None:
    document = _document()
    document["matrix_storage_bytes"][0][0] += 1
    with pytest.raises(ValueError, match="exception counts"):
        validate_x4t_cost_layer(document, source_revision="revision")

    document = _document()
    document["expert_storage_bytes"][0] += 1
    with pytest.raises(ValueError, match="do not close"):
        validate_x4t_cost_layer(document, source_revision="revision")
