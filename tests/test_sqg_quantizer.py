from __future__ import annotations

from types import SimpleNamespace

import torch

import kquant.sqg_quantizer as sqg_quantizer
from kquant.sqg_quantizer import install_sqg_quantizer


def test_rate_specific_none_dispatches_to_kquant_mcg(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, dict]] = []
    procedural_calls: list[tuple] = []

    def original(tiles: torch.Tensor, args: dict):
        calls.append((tiles, args))
        return "upstream"

    def procedural(*args):
        procedural_calls.append(args)

    monkeypatch.setattr(
        sqg_quantizer,
        "_sqg_temp_buffers",
        lambda _device, _bits: (torch.empty(0), torch.empty(0)),
    )
    monkeypatch.setattr(
        sqg_quantizer,
        "_extension",
        lambda: SimpleNamespace(quantize_tiles_procedural=procedural),
    )

    module = SimpleNamespace(quantize_tiles=original)
    install_sqg_quantizer(module)
    tiles = torch.zeros((1, 256), dtype=torch.float32)
    result = module.quantize_tiles(
        tiles,
        {
            "K": 2,
            "devices": ["cuda:0"],
            "sqg_e4m3_luts_by_bits": {2: None, 3: None, 4: None},
        },
    )

    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    assert not calls
    assert len(procedural_calls) == 1
    assert procedural_calls[0][5:] == (2, 1, 128)
