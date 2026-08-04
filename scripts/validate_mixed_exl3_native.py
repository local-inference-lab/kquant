"""Validate the TP12 reference trellis primitives against exllamav3 CUDA.

This synthetic closure checks K2/K3/K4 native packing, cyclic full-state
unpacking, MCG reconstruction in row-major tile order, and all 65,536 MCG
codebook values.  It loads no checkpoint or model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kquant.exl3_reference import (
    decode_mcg_states,
    decode_regularized_weight,
    reconstruct_trellis_states,
)
from kquant.mixed_exl3 import SCHEMA, pack_trellis_edges
from scripts.experiment_codec_transfers import _load_exl3_quantizer


def run(device: torch.device, exllamav3_root: Path) -> dict:
    if device.type != "cuda":
        raise ValueError("native EXL closure requires a CUDA device")
    _load_exl3_quantizer(exllamav3_root)
    from exllamav3.ext import exllamav3_ext as ext

    rates = []
    for bits in (2, 3, 4):
        generator = torch.Generator(device=device).manual_seed(900 + bits)
        edges = torch.randint(
            0,
            1 << bits,
            (2, 8, 256),
            dtype=torch.int16,
            device=device,
            generator=generator,
        )
        states = reconstruct_trellis_states(edges, bits)
        reference_packed = pack_trellis_edges(states, bits)
        native_packed = torch.empty_like(reference_packed)
        ext.pack_trellis(native_packed, states.contiguous(), bits)
        if not torch.equal(native_packed, reference_packed):
            raise ValueError(f"K{bits} native pack mismatch")

        native_states = torch.empty_like(states)
        ext.unpack_trellis(native_states, native_packed, bits)
        if not torch.equal(native_states, states):
            raise ValueError(f"K{bits} native state reconstruction mismatch")

        native_weight = torch.empty(
            (states.shape[0] * 16, states.shape[1] * 16),
            dtype=torch.float16,
            device=device,
        )
        ext.reconstruct(native_weight, native_packed, bits, True, False)
        reference_weight = decode_regularized_weight(states).half()
        if not torch.equal(native_weight, reference_weight):
            raise ValueError(f"K{bits} native MCG reconstruction mismatch")
        rates.append(
            {
                "bits": bits,
                "packed_words": native_packed.numel(),
                "pack_exact": True,
                "state_exact": True,
                "mcg_weight_exact": True,
            }
        )

    states = (
        torch.arange(65536, dtype=torch.int32, device=device)
        .to(torch.int16)
        .reshape(1, -1)
    )
    native_codebook = torch.empty_like(states, dtype=torch.float16)
    ext.decode(states, native_codebook, True, False)
    reference_codebook = decode_mcg_states(states)
    if not torch.equal(native_codebook, reference_codebook):
        raise ValueError("full native MCG codebook mismatch")

    return {
        "kind": "kquant_mixed_exl3_native_validation",
        "schema_version": 2,
        "codec_schema": SCHEMA,
        "device": str(device),
        "checkpoint_loaded": False,
        "rates": rates,
        "mcg_codebook_values": states.numel(),
        "mcg_codebook_exact": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(torch.device(args.device), args.exllamav3_root)
    rendered = json.dumps(payload, indent=1, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered)


if __name__ == "__main__":
    main()
