from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from scripts.validate_qsrt_codebooks import (
    _compact_functional_evidence,
    _load_sqg_rank_lut,
    _validate_corpus_contract,
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


def test_corpus_contract_accepts_validation_fold_nested_in_training_exclusion(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "teacher"
    training_capture = tmp_path / "training.kqcapture"
    validation_capture = tmp_path / "validation.kqcapture"
    args = Namespace(
        checkpoint=checkpoint,
        capture=training_capture,
        validation_capture=validation_capture,
        training_report=tmp_path / "training.json",
        validation_report=tmp_path / "validation.json",
    )
    common = {
        "kind": "kquant_interim_calibration_corpus_run",
        "finalized": True,
        "model_dir": str(checkpoint),
    }
    training = {
        **common,
        "capture_dir": str(training_capture),
        "fold": {
            "modulus": 4,
            "index": 0,
            "mode": "exclude",
            "unit": "document hash",
        },
        "documents": [{"document_hash": "train-document"}],
    }
    validation = {
        **common,
        "capture_dir": str(validation_capture),
        "fold": {
            "modulus": 8,
            "index": 0,
            "mode": "include",
            "unit": "document hash",
        },
        "documents": [{"document_hash": "validation-document"}],
    }

    provenance, training_requests, validation_requests = _validate_corpus_contract(
        args, training, validation
    )

    assert provenance["document_overlap"] == 0
    assert provenance["training_fold"]["modulus"] == 4
    assert provenance["validation_fold"]["modulus"] == 8
    assert training_requests == {1: "train-document"}
    assert validation_requests == {1: "validation-document"}
