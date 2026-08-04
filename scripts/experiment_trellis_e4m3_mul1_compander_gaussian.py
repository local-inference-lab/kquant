#!/usr/bin/env python3
"""Fit a tiny byte-sum compander over the MUL1 trellis graph.

This is a diagnostic between the published affine MUL1 decoder and a full
65,536-entry learned reconstruction table.  MUL1 first reduces a transition
index to the integer statistic

    s = bytesum(uint32(state * 0x83DCD12D)),  0 <= s <= 1020.

The experiment learns one E4M3 output byte for each possible ``s`` while
holding the rolling L16 graph fixed.  The resulting 1,021-byte table is an
oracle for procedural or piecewise companders of the same statistic; it is
not proposed as the production decoder by itself.

All fitting, ridge selection, gain fitting, and evaluation samples are
disjoint.  The source is synthetic iid Gaussian, so official-weight and
dense-H validation remain mandatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch

from kquant.exl3_reference import decode_mul1_e4m3_states, mul1_byte_sums
from kquant.trellis_window import encode_trellis
from scripts.experiment_trellis_e4m3_procedural_gaussian import (
    _parse_floats,
    _parse_ints,
    _sse,
)


STATISTIC_VALUES = 1021


def _initial_table(statistic: torch.Tensor) -> torch.Tensor:
    states = torch.arange(1 << 16, dtype=torch.int64)
    decoded = decode_mul1_e4m3_states(states).float()
    table = torch.zeros(STATISTIC_VALUES, dtype=torch.float32)
    table.scatter_(0, statistic.long(), decoded)
    return table


def _expand(table: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
    return table.index_select(0, statistic.long()).contiguous()


def _project_e4m3(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float8_e4m3fn).float().contiguous()


def _lloyd_update(
    source: torch.Tensor,
    *,
    codebook: torch.Tensor,
    statistic: torch.Tensor,
    initial_table: torch.Tensor,
    bits: int,
    gain: float,
    ridge: float,
    max_trace_mib: int,
) -> tuple[torch.Tensor, float, int]:
    scaled = source * gain
    encoded = encode_trellis(
        scaled,
        bits,
        16,
        codebook,
        max_trace_bytes=max_trace_mib << 20,
        return_transitions=True,
    )
    assert encoded.transition_indices is not None
    indices = statistic.index_select(
        0, encoded.transition_indices.long().flatten()
    ).long()
    targets = scaled.flatten().float()
    counts = torch.bincount(indices, minlength=STATISTIC_VALUES).float()
    sums = torch.zeros(STATISTIC_VALUES, dtype=torch.float32)
    sums.scatter_add_(0, indices, targets)
    denominator = counts + ridge
    centroid = torch.where(
        denominator > 0,
        (sums + ridge * initial_table) / denominator.clamp_min(1.0e-30),
        initial_table,
    )
    updated = _project_e4m3(centroid)
    fixed_path_sse = float(
        (updated.index_select(0, indices).double() - targets.double())
        .square()
        .sum()
        .item()
    )
    return updated, fixed_path_sse, int(torch.count_nonzero(counts).item())


def _fit_one(
    training: torch.Tensor,
    *,
    statistic: torch.Tensor,
    initial_table: torch.Tensor,
    bits: int,
    gain: float,
    ridge: float,
    iterations: int,
    max_trace_mib: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    table = initial_table.clone()
    codebook = _expand(table, statistic)
    current_sse = _sse(
        training,
        codebook=codebook,
        bits=bits,
        gain=gain,
        max_trace_mib=max_trace_mib,
    )
    rows: list[dict[str, float | int]] = []
    for iteration in range(iterations):
        proposal, fixed_path_sse, visited = _lloyd_update(
            training,
            codebook=codebook,
            statistic=statistic,
            initial_table=initial_table,
            bits=bits,
            gain=gain,
            ridge=ridge,
            max_trace_mib=max_trace_mib,
        )
        proposal_codebook = _expand(proposal, statistic)
        proposal_sse = _sse(
            training,
            codebook=proposal_codebook,
            bits=bits,
            gain=gain,
            max_trace_mib=max_trace_mib,
        )
        accepted = proposal_sse <= current_sse
        rows.append(
            {
                "iteration": iteration + 1,
                "visited_statistics": visited,
                "fixed_path_sse_scaled": fixed_path_sse,
                "training_sse": proposal_sse,
                "accepted": int(accepted),
            }
        )
        if not accepted:
            break
        table = proposal
        codebook = proposal_codebook
        if proposal_sse == current_sse:
            break
        current_sse = proposal_sse
    return table, rows


def _fit_gain(
    source: torch.Tensor,
    *,
    codebook: torch.Tensor,
    bits: int,
    gains: tuple[float, ...],
    max_trace_mib: int,
) -> tuple[float, list[dict[str, float]]]:
    rows = [
        {
            "gain": gain,
            "sse": _sse(
                source,
                codebook=codebook,
                bits=bits,
                gain=gain,
                max_trace_mib=max_trace_mib,
            ),
        }
        for gain in gains
    ]
    selected = min(rows, key=lambda row: row["sse"])
    return selected["gain"], rows


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    gain_fit = torch.randn(
        args.gain_fit_sequences, args.length, generator=generator
    )
    training = torch.randn(args.training_sequences, args.length, generator=generator)
    selection = torch.randn(args.selection_sequences, args.length, generator=generator)
    evaluation = torch.randn(args.evaluation_sequences, args.length, generator=generator)
    evaluation_energy = float(evaluation.double().square().sum().item())
    states = torch.arange(1 << 16, dtype=torch.int64)
    statistic = mul1_byte_sums(states).long()
    initial_table = _initial_table(statistic)
    initial_codebook = _expand(initial_table, statistic)
    mcg = load_file(str(args.oracle_codebooks), device="cpu")[
        "procedural_mcg_fp16"
    ].float()
    output: dict[str, object] = {}
    tensors: dict[str, torch.Tensor] = {
        "statistic": statistic.to(torch.int16),
        "initial_table": initial_table.to(torch.float8_e4m3fn).view(torch.uint8),
    }
    started = time.perf_counter()
    for bits in args.rates:
        baseline_gain, baseline_gain_rows = _fit_gain(
            gain_fit,
            codebook=initial_codebook,
            bits=bits,
            gains=args.gains,
            max_trace_mib=args.max_trace_mib,
        )
        candidates: list[dict[str, object]] = []
        candidate_tables: dict[float, torch.Tensor] = {}
        for ridge in args.ridges:
            table, iterations = _fit_one(
                training,
                statistic=statistic,
                initial_table=initial_table,
                bits=bits,
                gain=baseline_gain,
                ridge=ridge,
                iterations=args.iterations,
                max_trace_mib=args.max_trace_mib,
            )
            codebook = _expand(table, statistic)
            own_gain, own_gain_rows = _fit_gain(
                gain_fit,
                codebook=codebook,
                bits=bits,
                gains=args.gains,
                max_trace_mib=args.max_trace_mib,
            )
            selection_sse = _sse(
                selection,
                codebook=codebook,
                bits=bits,
                gain=own_gain,
                max_trace_mib=args.max_trace_mib,
            )
            candidates.append(
                {
                    "ridge": ridge,
                    "gain": own_gain,
                    "gain_grid": own_gain_rows,
                    "iterations": iterations,
                    "selection_sse": selection_sse,
                    "table_numeric_labels": int(torch.unique(table).numel()),
                    "changed_entries": int(torch.count_nonzero(table != initial_table)),
                }
            )
            candidate_tables[ridge] = table
            print(
                f"K{bits} ridge={ridge:g} gain={own_gain:g} "
                f"selection={selection_sse:.8g}",
                flush=True,
            )
        selected = min(candidates, key=lambda row: float(row["selection_sse"]))
        selected_table = candidate_tables[float(selected["ridge"])]
        selected_codebook = _expand(selected_table, statistic)
        selected_sse = _sse(
            evaluation,
            codebook=selected_codebook,
            bits=bits,
            gain=float(selected["gain"]),
            max_trace_mib=args.max_trace_mib,
        )
        initial_sse = _sse(
            evaluation,
            codebook=initial_codebook,
            bits=bits,
            gain=baseline_gain,
            max_trace_mib=args.max_trace_mib,
        )
        mcg_gain, mcg_gain_rows = _fit_gain(
            gain_fit,
            codebook=mcg,
            bits=bits,
            gains=args.gains,
            max_trace_mib=args.max_trace_mib,
        )
        mcg_sse = _sse(
            evaluation,
            codebook=mcg,
            bits=bits,
            gain=mcg_gain,
            max_trace_mib=args.max_trace_mib,
        )
        tensors[f"K{bits}_selected_table"] = selected_table.to(
            torch.float8_e4m3fn
        ).view(torch.uint8)
        output[str(bits)] = {
            "published_mul1": {
                "gain": baseline_gain,
                "gain_grid": baseline_gain_rows,
                "evaluation_sse": initial_sse,
                "evaluation_nmse": initial_sse / evaluation_energy,
            },
            "procedural_mcg_fp16": {
                "gain": mcg_gain,
                "gain_grid": mcg_gain_rows,
                "evaluation_sse": mcg_sse,
                "evaluation_nmse": mcg_sse / evaluation_energy,
            },
            "candidates": candidates,
            "selected": selected,
            "selected_evaluation_sse": selected_sse,
            "selected_evaluation_nmse": selected_sse / evaluation_energy,
            "selected_over_mul1": selected_sse / initial_sse,
            "selected_over_mcg": selected_sse / mcg_sse,
        }
    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_mul1_compander_gaussian",
        "schema_version": 1,
        "scope": {
            "warning": (
                "Synthetic Gaussian diagnostic only; a 1,021-byte table is an "
                "oracle for a later procedural/piecewise compander."
            ),
            "rates": list(args.rates),
            "window_bits": 16,
            "statistic_values": STATISTIC_VALUES,
            "table_bytes": STATISTIC_VALUES,
            "seed": args.seed,
            "gain_fit_sequences": args.gain_fit_sequences,
            "training_sequences": args.training_sequences,
            "selection_sequences": args.selection_sequences,
            "evaluation_sequences": args.evaluation_sequences,
            "sequence_length": args.length,
            "evaluation_energy": evaluation_energy,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    save_file(tensors, str(args.tensors))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=_parse_ints, default=(5,))
    parser.add_argument(
        "--gains",
        type=_parse_floats,
        default=(0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35),
    )
    parser.add_argument("--ridges", type=_parse_floats, default=(1.0, 8.0, 64.0))
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--gain-fit-sequences", type=int, default=8)
    parser.add_argument("--training-sequences", type=int, default=64)
    parser.add_argument("--selection-sequences", type=int, default=16)
    parser.add_argument("--evaluation-sequences", type=int, default=64)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260807)
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
        default=Path("out/trellis-e4m3-mul1-compander-gaussian-v1.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("out/trellis-e4m3-mul1-compander-gaussian-v1.safetensors"),
    )
    args = parser.parse_args()
    if any(bits not in (2, 3, 4, 5) for bits in args.rates):
        parser.error("rates must be drawn from K2--K5")
    if any(value <= 0 for value in (
        args.iterations,
        args.gain_fit_sequences,
        args.training_sequences,
        args.selection_sequences,
        args.evaluation_sequences,
        args.length,
        args.cpu_threads,
        args.max_trace_mib,
    )):
        parser.error("sequence counts, length, iterations, and limits must be positive")
    if args.length % 2:
        parser.error("sequence length must be even")
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tensors": str(args.tensors),
                "elapsed_seconds": payload["elapsed_seconds"],
                "results": payload["results"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
