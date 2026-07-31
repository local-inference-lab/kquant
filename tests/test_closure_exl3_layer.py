from __future__ import annotations

import inspect

import pytest
import torch

from scripts.closure_exl3_layer import (
    SOURCE_W13_LAYOUT,
    _pad_axis,
    _padded_w4a16_intermediate,
    _run_source_w4a16_rank,
    _scenario_set,
)


def test_original_mxfp4_uses_production_gate_first_layout() -> None:
    assert SOURCE_W13_LAYOUT == "w31"
    source = inspect.getsource(_run_source_w4a16_rank)
    assert "PreparedWeightLayout.MMA_PACKED" in source
    assert "PreparedWeightLayout.SOURCE_NATIVE" not in source


def test_tp16_w4a16_padding_is_zero_and_128_aligned() -> None:
    assert _padded_w4a16_intermediate(192) == 256
    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    padded = _pad_axis(value, axis=0, target=4, fill=0)
    assert padded.tolist() == [[1, 2], [3, 4], [0, 0], [0, 0]]


def test_kept_kernel_scenarios_are_rejected() -> None:
    with pytest.raises(ValueError, match="kept-weight kernel"):
        _scenario_set(list(range(16)), ["all-kept"], 16)
