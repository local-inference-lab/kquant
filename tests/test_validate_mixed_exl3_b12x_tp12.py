import pytest
import torch

from scripts.validate_mixed_exl3_b12x_tp12 import (
    _metrics,
    _selected_modes,
    _tensor_prefix,
)


def _ledger(selected_r13: int = 3, selected_r2: int = 1) -> dict:
    return {
        "kind": "kquant_tp12_mixed_exl3_candidate_pool",
        "layer": 24,
        "selections": {
            "138": {
                "selection": {
                    "selected_r13": selected_r13,
                    "selected_r2": selected_r2,
                }
            }
        },
    }


def test_selected_modes_read_separate_expert_static_ledger_entries():
    assert _selected_modes(_ledger(), layer=24, expert=138) == (3, 1)


def test_selected_mode_fails_closed_on_wrong_layer_or_unknown_mode():
    with pytest.raises(ValueError, match="does not match"):
        _selected_modes(_ledger(), layer=40, expert=138)
    with pytest.raises(ValueError, match="unknown mixed-EXL mode ID"):
        _selected_modes(_ledger(13, 1), layer=24, expert=138)


def test_selected_modes_read_legacy_shared_mode():
    ledger = _ledger()
    selection = ledger["selections"]["138"]["selection"]
    selection.clear()
    selection["selected_r"] = 2
    assert _selected_modes(ledger, layer=24, expert=138) == (2, 2)


def test_tensor_prefix_uses_packager_tensor_contract():
    assert _tensor_prefix(1, 7, "w2") == (
        "language_model.model.layers.1.block_sparse_moe.experts."
        "7.w2.mixed_exl3_"
    )


def test_metrics_report_identity_and_relative_error():
    expected = torch.tensor([[3.0, 4.0]])
    identical = _metrics(expected, expected)
    assert identical == {
        "relative_error": 0.0,
        "cosine": pytest.approx(1.0),
        "max_abs_error": 0.0,
    }

    actual = torch.tensor([[0.0, 4.0]])
    changed = _metrics(actual, expected)
    assert changed["relative_error"] == pytest.approx(0.6)
    assert changed["max_abs_error"] == 3.0
