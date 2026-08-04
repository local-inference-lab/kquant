#!/usr/bin/env python3
"""Search a table-free K5-specific QD-MUL1 -> E4M3 reconstruction map.

The current quantization-aware MUL1 candidate hard-codes one residual hash,
three residual bits, and one normalization phase.  That candidate is useful at
K3/K4, but its K5 error remains above FP16 MCG.  This script asks the narrower
question: is that gap explained by those arbitrary constants?

Every candidate retains the same inexpensive structure::

    u = t ^ (t << r0) [^ (t << r1)]
    p = uint32(u * 0x83dcd12d)
    s = bytesum(p)
    h = p ^ (p >> a) ^ (p >> b)
    q = (s << d) + ((h >> z) & (2**d - 1)) - center
    y = E4M3_RNE(FP16(q) * FP16(alpha))

The source is synthetic Gaussian and the objective is diagonal SSE.  Proposal
screening, gain fitting, candidate selection, and blind evaluation use four
disjoint sample sets.  Only the selected candidate and predeclared controls
are evaluated on the blind split, avoiding a second round of winner selection.
Official-weight, dense-H, and routed-functional validation remain mandatory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

from safetensors.torch import load_file
import torch

from kquant.exl3_reference import MUL1_MULT, decode_mul1_e4m3_states
from kquant.trellis_window import encode_trellis, trellis_alias_metrics
from scripts.experiment_trellis_e4m3_procedural_gaussian import (
    _parse_floats,
    _sse,
)
from scripts.experiment_trellis_e4m3_procedural_v2 import (
    _e4m3_from_fixedpoint,
    _premix_many,
    _product_and_bytes,
    _states,
)


@dataclass(frozen=True)
class Parameters:
    premix: tuple[int, ...]
    fold: tuple[int, int]
    dither_bits: int
    dither_shift: int
    phase: float

    @property
    def name(self) -> str:
        premix = "id" if not self.premix else "x" + "_".join(map(str, self.premix))
        return (
            f"qd5_{premix}_f{self.fold[0]}_{self.fold[1]}_"
            f"d{self.dither_bits}_s{self.dither_shift}_p{self.phase:g}"
        )


def _parse_phases(value: str) -> tuple[float, ...]:
    return _parse_floats(value)


def _candidate(parameters: Parameters) -> tuple[torch.Tensor, dict[str, object]]:
    states = _states()
    if parameters.premix:
        states = _premix_many(states, parameters.premix)
    product, byte_columns = _product_and_bytes(states)
    byte_sum = byte_columns.sum(dim=1)
    a, b = parameters.fold
    folded = product ^ (product >> a) ^ (product >> b)
    mask = (1 << parameters.dither_bits) - 1
    residual = (folded >> parameters.dither_shift) & mask
    statistic = (byte_sum << parameters.dither_bits) + residual
    center = int(round(float(statistic.double().mean().item())))
    fixedpoint = statistic - center
    codebook, alpha = _e4m3_from_fixedpoint(fixedpoint, parameters.phase)
    return codebook, {
        "multiplier": MUL1_MULT,
        "multiplier_hex": f"0x{MUL1_MULT:08X}",
        "premix_left_xorshifts": list(parameters.premix),
        "fold_right_xorshifts": list(parameters.fold),
        "dither_bits": parameters.dither_bits,
        "dither_shift": parameters.dither_shift,
        "phase": parameters.phase,
        "center": center,
        "fp16_alpha": alpha,
        "numeric_labels": int(torch.unique(codebook).numel()),
        "minimum": float(codebook.min().item()),
        "maximum": float(codebook.max().item()),
        "mean": float(codebook.double().mean().item()),
        "standard_deviation": float(codebook.double().std(unbiased=False).item()),
    }


def _parameter_grid() -> tuple[Parameters, ...]:
    # The predeclared set includes the identity, the best prior plain-MUL1 K5
    # one-tap map, the robust K3 map, and the prior QD K5 proposal.  These are
    # compile-time K-specific mappings and require no record metadata.
    premixes = ((), (6,), (15,), (7, 15))
    # Each fold has exactly the same two-shift/two-XOR cost.  The published QD
    # candidate is (13, 23).
    folds = ((7, 19), (9, 23), (13, 23), (17, 29))
    return tuple(
        Parameters(premix, fold, dither_bits, dither_shift, phase)
        for premix in premixes
        for fold in folds
        for dither_bits in (2, 3, 4, 5)
        for dither_shift in (0, 7, 15, 23, 31)
        for phase in (0.875, 1.0, 1.25, 1.5)
    )


def _fit_gain(
    source: torch.Tensor,
    *,
    codebook: torch.Tensor,
    gains: tuple[float, ...],
    max_trace_mib: int,
) -> tuple[float, list[dict[str, float]]]:
    rows = [
        {
            "gain": gain,
            "sse": _sse(
                source,
                codebook=codebook,
                bits=5,
                gain=gain,
                max_trace_mib=max_trace_mib,
            ),
        }
        for gain in gains
    ]
    selected = min(rows, key=lambda row: row["sse"])
    return float(selected["gain"]), rows


def _errors(
    source: torch.Tensor,
    *,
    codebook: torch.Tensor,
    gain: float,
    max_trace_mib: int,
) -> torch.Tensor:
    encoded = encode_trellis(
        source * gain,
        5,
        16,
        codebook,
        max_trace_bytes=max_trace_mib << 20,
        return_transitions=False,
    )
    return encoded.squared_error.double() / (gain * gain)


def _evaluate(
    source: torch.Tensor,
    *,
    codebook: torch.Tensor,
    gain: float,
    max_trace_mib: int,
) -> tuple[dict[str, object], torch.Tensor]:
    errors = _errors(
        source,
        codebook=codebook,
        gain=gain,
        max_trace_mib=max_trace_mib,
    )
    energy = source.double().square().sum(dim=1)
    return (
        {
            "gain": gain,
            "sse": float(errors.sum().item()),
            "nmse": float(errors.sum().item() / energy.sum().item()),
            "mean_per_sequence_nmse": float((errors / energy).mean().item()),
            "median_per_sequence_nmse": float((errors / energy).median().item()),
        },
        errors,
    )


def _paired(candidate: torch.Tensor, control: torch.Tensor) -> dict[str, float]:
    difference = candidate - control
    standard_error = difference.std(unbiased=True) / difference.numel() ** 0.5
    return {
        "mean_sse_difference_per_sequence": float(difference.mean().item()),
        "standard_error": float(standard_error.item()),
        "mean_difference_over_control": float(
            difference.sum().item() / control.sum().item()
        ),
        "sequence_win_fraction": float((difference < 0).double().mean().item()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    screen = torch.randn(args.screen_sequences, args.length, generator=generator)
    gain_fit = torch.randn(args.gain_fit_sequences, args.length, generator=generator)
    selection = torch.randn(args.selection_sequences, args.length, generator=generator)
    blind = torch.randn(args.blind_sequences, args.length, generator=generator)
    started = time.perf_counter()

    parameter_grid = _parameter_grid()
    proposals: list[dict[str, object]] = []
    parameters_by_name: dict[str, Parameters] = {}
    for index, parameters in enumerate(parameter_grid, 1):
        codebook, metadata = _candidate(parameters)
        proposals.append(
            {
                "name": parameters.name,
                "parameters": metadata,
                "screen_sse": _sse(
                    screen,
                    codebook=codebook,
                    bits=5,
                    gain=args.screen_gain,
                    max_trace_mib=args.max_trace_mib,
                ),
            }
        )
        parameters_by_name[parameters.name] = parameters
        if index % 512 == 0:
            print(f"screened {index}/{len(parameter_grid)}", flush=True)
    proposals.sort(key=lambda row: float(row["screen_sse"]))

    finalist_rows: list[dict[str, object]] = []
    for proposal in proposals[: args.finalists]:
        parameters = parameters_by_name[str(proposal["name"])]
        codebook, metadata = _candidate(parameters)
        gain, gain_grid = _fit_gain(
            gain_fit,
            codebook=codebook,
            gains=args.gains,
            max_trace_mib=args.max_trace_mib,
        )
        selection_sse = _sse(
            selection,
            codebook=codebook,
            bits=5,
            gain=gain,
            max_trace_mib=args.max_trace_mib,
        )
        finalist_rows.append(
            {
                **proposal,
                "parameters": metadata,
                "gain": gain,
                "gain_grid": gain_grid,
                "selection_sse": selection_sse,
            }
        )
    finalist_rows.sort(key=lambda row: float(row["selection_sse"]))
    winner_row = finalist_rows[0]
    winner_parameters = parameters_by_name[str(winner_row["name"])]
    winner_codebook, winner_metadata = _candidate(winner_parameters)

    oracle = load_file(str(args.oracle_codebooks), device="cpu")
    controls: dict[str, torch.Tensor] = {
        "fp16_mcg": oracle["procedural_mcg_fp16"].float().contiguous(),
        "published_mul1_e4m3": decode_mul1_e4m3_states(_states()).float(),
    }
    default_parameters = Parameters((), (13, 23), 3, 7, 1.25)
    default_codebook, _ = _candidate(default_parameters)
    controls["prior_qd_mul1_e4m3"] = default_codebook

    blind_results: dict[str, object] = {}
    blind_errors: dict[str, torch.Tensor] = {}
    fitted_control_gains: dict[str, object] = {}
    for name, codebook in controls.items():
        gain, grid = _fit_gain(
            gain_fit,
            codebook=codebook,
            gains=args.gains,
            max_trace_mib=args.max_trace_mib,
        )
        fitted_control_gains[name] = {"gain": gain, "grid": grid}
        blind_results[name], blind_errors[name] = _evaluate(
            blind,
            codebook=codebook,
            gain=gain,
            max_trace_mib=args.max_trace_mib,
        )
    winner_gain = float(winner_row["gain"])
    blind_results["selected_qd"], blind_errors["selected_qd"] = _evaluate(
        blind,
        codebook=winner_codebook,
        gain=winner_gain,
        max_trace_mib=args.max_trace_mib,
    )
    paired = {
        name: _paired(blind_errors["selected_qd"], errors)
        for name, errors in blind_errors.items()
        if name != "selected_qd"
    }

    alias = trellis_alias_metrics(winner_codebook, 5, 16)
    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_qd_k5_search",
        "schema_version": 1,
        "scope": {
            "warning": (
                "Synthetic iid Gaussian diagonal-SSE search only; official-weight, "
                "dense-H, and routed-functional blind validation remain mandatory."
            ),
            "rate": 5,
            "window_bits": 16,
            "seed": args.seed,
            "sequence_length": args.length,
            "screen_sequences": args.screen_sequences,
            "gain_fit_sequences": args.gain_fit_sequences,
            "selection_sequences": args.selection_sequences,
            "blind_sequences": args.blind_sequences,
            "candidate_count": len(proposals),
            "finalist_count": args.finalists,
            "screen_gain": args.screen_gain,
            "gain_grid": list(args.gains),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "top_screen": proposals[: args.report_screen],
        "finalists": finalist_rows,
        "selected": {
            "name": winner_row["name"],
            "parameters": winner_metadata,
            "gain": winner_gain,
            "alias_geometry": alias,
        },
        "control_gain_fits": fitted_control_gains,
        "blind": blind_results,
        "selected_paired_against_controls": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gains", type=_parse_floats, default=(0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45))
    parser.add_argument("--screen-gain", type=float, default=1.15)
    parser.add_argument("--screen-sequences", type=int, default=1)
    parser.add_argument("--gain-fit-sequences", type=int, default=8)
    parser.add_argument("--selection-sequences", type=int, default=32)
    parser.add_argument("--blind-sequences", type=int, default=128)
    parser.add_argument("--finalists", type=int, default=20)
    parser.add_argument("--report-screen", type=int, default=20)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument(
        "--oracle-codebooks",
        type=Path,
        default=Path(
            "out/trellis-learned-codebooks-l1-l24-l40-shared-l16-v2.safetensors"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-qd-k5-search-v1.json"),
    )
    args = parser.parse_args()
    if min(
        args.screen_sequences,
        args.gain_fit_sequences,
        args.selection_sequences,
        args.blind_sequences,
        args.finalists,
        args.report_screen,
        args.length,
        args.cpu_threads,
        args.max_trace_mib,
    ) < 1:
        parser.error("counts, length, threads, and trace size must be positive")
    if args.length % 2:
        parser.error("sequence length must be even")
    if args.screen_gain <= 0.0:
        parser.error("--screen-gain must be positive")
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": payload["elapsed_seconds"],
                "selected": payload["selected"],
                "blind": payload["blind"],
                "paired": payload["selected_paired_against_controls"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
