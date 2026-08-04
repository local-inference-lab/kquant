from __future__ import annotations

import pytest
import torch

from scripts.experiment_uniform_k_mxfp4_endpoint import (
    _compact_functional_evidence,
    _load_sqg_rank_lut,
)
from scripts.summarize_uniform_k_mxfp4_endpoint import _select_frontier


def test_compact_functional_evidence_preserves_aggregate_folds() -> None:
    candidates = {
        "mul1-e4m3:K4": {
            "routed_function": {
                "train": {
                    "rows": 3,
                    "route_weighted_squared_error": 1.25,
                    "request_contributions": [{"request_step": 1}],
                },
                "confirmation": None,
            }
        }
    }

    _compact_functional_evidence(candidates)

    assert candidates["mul1-e4m3:K4"]["routed_function"]["train"] == {
        "rows": 3,
        "route_weighted_squared_error": 1.25,
    }
    assert candidates["mul1-e4m3:K4"]["routed_function"]["confirmation"] is None


def test_compact_functional_evidence_rejects_immutable_fold() -> None:
    candidates = {
        "mul1-e4m3:K4": {"routed_function": {"train": ("not", "mutable")}}
    }

    with pytest.raises(TypeError, match="fold evidence must be mutable"):
        _compact_functional_evidence(candidates)


def test_load_sqg_rank_lut_records_reproducible_contract(tmp_path) -> None:
    path = tmp_path / "normal-e4m3.bin"
    path.write_bytes(bytes(index % 126 for index in range(1 << 16)))

    values, metadata = _load_sqg_rank_lut(path)

    assert values.dtype == torch.uint8
    assert values.shape == (1 << 16,)
    assert metadata["bytes"] == 1 << 16
    assert len(metadata["sha256"]) == 64
    assert metadata["graph_contract"].startswith("one shared rank law")


def test_load_sqg_rank_lut_rejects_wrong_size(tmp_path) -> None:
    path = tmp_path / "short.bin"
    path.write_bytes(b"\0" * 8)
    with pytest.raises(ValueError, match="65,536 bytes"):
        _load_sqg_rank_lut(path)


def test_rate_frontier_selects_on_confirmation_not_validation() -> None:
    records = {}
    for expert, confirmation_gain, validation_gain in (
        (0, 4.0, 1.0),
        (1, 3.0, 2.0),
        (2, 2.0, 3.0),
        (3, 1.0, 4.0),
    ):
        records[(24, expert, 3)] = {
            "confirmation_sse": 10.0,
            "validation_sse": 10.0,
        }
        records[(24, expert, 4)] = {
            "confirmation_sse": 10.0 - confirmation_gain,
            "validation_sse": 10.0 - validation_gain,
        }

    selected, oracle = _select_frontier(
        records, fraction=0.5, selection_scope="layer"
    )

    assert selected == [(24, 0), (24, 1)]
    assert oracle == [(24, 2), (24, 3)]
