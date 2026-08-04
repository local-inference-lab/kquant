"""Research-only bridge from EXL's production encoder to SQG E4M3 tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from kquant.sqg_e4m3 import sqg_e4m3_bytes


@lru_cache(maxsize=1)
def _extension():
    project = Path(__file__).resolve().parents[1]
    exllamav3 = Path("/home/luke/projects/exllamav3").resolve()
    return load(
        name="kquant_sqg_quantize_ext_v2",
        sources=[
            str(project / "kquant/csrc/sqg_quantize.cpp"),
            str(project / "kquant/csrc/sqg_quantize.cu"),
        ],
        extra_include_paths=[str(exllamav3 / "exllamav3/exllamav3_ext")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-Xcudafe",
            "--diag_suppress=177",
            "-Xcudafe",
            "--diag_suppress=20012",
        ],
        verbose=False,
    )


def install_sqg_quantizer(quantizer_module) -> None:
    """Teach a loaded EXL encoder module to consume ``sqg_e4m3_lut``.

    The patch is process-local.  Existing MUL1/MCG calls keep using the
    original extension, while SQG calls use a small dedicated CUDA extension
    with the identical Viterbi state and traceback implementation.
    """

    if getattr(quantizer_module, "_kquant_sqg_installed", False):
        return
    original = quantizer_module.quantize_tiles

    device_luts: dict[tuple[str, int, str], torch.Tensor] = {}

    def quantize_tiles(tiles: torch.Tensor, quant_args: dict):
        codebook = quant_args.get("sqg_e4m3_lut")
        mode = quant_args.get("sqg_e4m3_mode")
        if codebook is None and mode is None:
            return original(tiles, quant_args)
        if len(quant_args["devices"]) != 1:
            raise ValueError("the SQG validation hook currently requires one CUDA device")
        tiles = tiles.contiguous()
        if tiles.dtype != torch.float32 or tiles.ndim != 2 or tiles.shape[1] != 256:
            raise ValueError("SQG tiles must be contiguous FP32 [N, 256]")
        bits = int(quant_args["K"])
        if codebook is None:
            if mode not in ("normal", "tail"):
                raise ValueError("sqg_e4m3_mode must be 'normal' or 'tail'")
            key = (str(tiles.device), bits, mode)
            codebook = device_luts.get(key)
            if codebook is None:
                codebook = sqg_e4m3_bytes(bits, mode, device=tiles.device)
                device_luts[key] = codebook
        output = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        costs, edges = quantizer_module.get_temp_buffers(tiles.device, bits)
        lut = codebook.to(device=tiles.device, dtype=torch.uint8).contiguous()
        _extension().quantize_tiles_sqg(
            tiles,
            output,
            indices,
            costs,
            edges,
            lut,
            bits,
            int(quant_args.get("tailbite_context", 128)),
        )
        return output, indices

    quantizer_module.quantize_tiles = quantize_tiles
    quantizer_module._kquant_sqg_installed = True
