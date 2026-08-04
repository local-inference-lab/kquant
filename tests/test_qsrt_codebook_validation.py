from __future__ import annotations

import pytest
import torch

from scripts.validate_qsrt_codebooks import (
    _compact_functional_evidence,
    _load_sqg_rank_lut,
)


def test_compact_functional_evidence_preserves_aggregate_folds() -> None:
    candidates = {
        "sqg-normal-e4m3:K4": {
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

    assert candidates["sqg-normal-e4m3:K4"]["routed_function"]["train"] == {
        "rows": 3,
        "route_weighted_squared_error": 1.25,
    }
    assert candidates["sqg-normal-e4m3:K4"]["routed_function"]["confirmation"] is None


def test_compact_functional_evidence_rejects_immutable_fold() -> None:
    candidates = {
        "sqg-normal-e4m3:K4": {
            "routed_function": {"train": ("not", "mutable")}
        }
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
