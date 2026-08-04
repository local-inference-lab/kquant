from __future__ import annotations

import numpy as np
import pytest

from kquant.pack.x4t_index import (
    X4T_COST_INDEX_SCHEMA_VERSION,
    X4T_COST_LAYER_KIND,
    validate_x4t_cost_layer,
)
from kquant.x4t import X4T_DATA_OFFSET, X4T_EXPERTS_PER_LAYER, X4T_MATRIX_ORDER


def _document() -> dict:
    matrix = [[4096, 8192, 12288] for _ in range(X4T_EXPERTS_PER_LAYER)]
    expert = [sum(row) for row in matrix]
    return {
        "kind": X4T_COST_LAYER_KIND,
        "schema_version": X4T_COST_INDEX_SCHEMA_VERSION,
        "source_revision": "revision",
        "layer": 24,
        "matrix_order": list(X4T_MATRIX_ORDER),
        "matrix_storage_bytes": matrix,
        "expert_storage_bytes": expert,
        "all_x4t_sidecar_bytes": X4T_DATA_OFFSET + sum(expert),
    }


def test_x4t_cost_layer_validation_closes_exact_bytes() -> None:
    matrix, expert = validate_x4t_cost_layer(
        _document(), source_revision="revision"
    )

    assert matrix.shape == (896, 3)
    assert expert.shape == (896,)
    assert np.array_equal(expert, matrix.sum(axis=1))


def test_x4t_cost_layer_rejects_unaligned_and_mismatched_costs() -> None:
    document = _document()
    document["matrix_storage_bytes"][0][0] += 1
    with pytest.raises(ValueError, match="aligned"):
        validate_x4t_cost_layer(document, source_revision="revision")

    document = _document()
    document["expert_storage_bytes"][0] += 4096
    with pytest.raises(ValueError, match="do not close"):
        validate_x4t_cost_layer(document, source_revision="revision")
