"""Research-only bridge from EXL's production encoder to SQG E4M3 tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

import torch
from torch.utils.cpp_extension import load

from kquant.sqg_e4m3 import sqg_e4m3_bytes


@lru_cache(maxsize=1)
def _extension():
    project = Path(__file__).resolve().parents[1]
    return load(
        name="kquant_sqg_quantize_ext_v5",
        sources=[
            str(project / "kquant/csrc/sqg_quantize.cpp"),
            str(project / "kquant/csrc/sqg_quantize.cu"),
        ],
        extra_include_paths=[str(project / "kquant/csrc")],
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


@lru_cache(maxsize=None)
def _sqg_temp_buffers(device: torch.device, bits: int):
    """Allocate the packed traceback buffers consumed by kquant's kernel."""

    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    edges = 65536 >> bits
    decisions = edges // 2
    free_bytes, _ = torch.cuda.mem_get_info(device)
    decision_bytes_per_tile = 256 * decisions
    affordable = max(256, int(free_bytes * 0.5) // decision_bytes_per_tile)
    max_batch = min(max(256, 3 * multiprocessors), affordable)
    costs = torch.zeros(
        (max_batch, 2, edges), dtype=torch.float16, device=device
    )
    traceback = torch.empty(
        (max_batch, 256, decisions), dtype=torch.uint8, device=device
    )
    return costs, traceback


def install_sqg_quantizer(quantizer_module) -> None:
    """Teach a loaded EXL encoder module to consume ``sqg_e4m3_lut``.

    The patch is process-local. SQG and explicit MCG/MUL1 controls use
    kquant's CUDA extension with one Viterbi/tail-biting implementation. A
    ``None`` entry in a rate-specific mapping explicitly selects MCG,
    permitting controlled hybrid rate-curve studies. Calls without any
    kquant codebook argument retain the unmodified upstream EXL behavior.
    """

    if getattr(quantizer_module, "_kquant_sqg_installed", False):
        return
    original = quantizer_module.quantize_tiles

    device_luts: dict[tuple[str, int, str], torch.Tensor] = {}

    def quantize_tiles(tiles: torch.Tensor, quant_args: dict):
        codebook = quant_args.get("sqg_e4m3_lut")
        rate_codebooks = quant_args.get("sqg_e4m3_luts_by_bits")
        mode = quant_args.get("sqg_e4m3_mode")
        if codebook is None and rate_codebooks is None and mode is None:
            return original(tiles, quant_args)
        if len(quant_args["devices"]) != 1:
            raise ValueError("the SQG validation hook currently requires one CUDA device")
        tiles = tiles.contiguous()
        if tiles.dtype != torch.float32 or tiles.ndim != 2 or tiles.shape[1] != 256:
            raise ValueError("SQG tiles must be contiguous FP32 [N, 256]")
        bits = int(quant_args["K"])
        if rate_codebooks is not None:
            if codebook is not None or mode is not None:
                raise ValueError(
                    "rate-specific SQG LUTs cannot be combined with another SQG law"
                )
            if not isinstance(rate_codebooks, Mapping):
                raise TypeError("sqg_e4m3_luts_by_bits must be a mapping")
            if set(rate_codebooks) - {2, 3, 4}:
                raise ValueError("rate-specific SQG LUT keys must be K2, K3, or K4")
            try:
                codebook = rate_codebooks[bits]
            except KeyError as exc:
                raise ValueError(f"missing rate-specific SQG K{bits} LUT") from exc
            if codebook is None:
                output = torch.empty_like(tiles)
                indices = torch.empty_like(tiles, dtype=torch.int16)
                costs, edges = _sqg_temp_buffers(tiles.device, bits)
                _extension().quantize_tiles_procedural(
                    tiles,
                    output,
                    indices,
                    costs,
                    edges,
                    bits,
                    1,
                    int(quant_args.get("tailbite_context", 128)),
                )
                return output, indices
        elif codebook is None:
            if mode not in ("normal", "tail"):
                raise ValueError("sqg_e4m3_mode must be 'normal' or 'tail'")
            key = (str(tiles.device), bits, mode)
            codebook = device_luts.get(key)
            if codebook is None:
                codebook = sqg_e4m3_bytes(bits, mode, device=tiles.device)
                device_luts[key] = codebook
        output = torch.empty_like(tiles)
        indices = torch.empty_like(tiles, dtype=torch.int16)
        costs, edges = _sqg_temp_buffers(tiles.device, bits)
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
