#!/usr/bin/env python3
"""Search cheap rate-specific MUL1 label pre-mixes on Gaussian sequences.

The searched map is an upper-triangular GF(2) permutation of the 16-bit
transition index:

    u = t ^ (t << a) [^ (t << b)]

It requires no lookup table.  Shifts begin at five so K2--K5 branch bits are
preserved.  The decoder may specialize the constant tap set by K without new
per-record metadata because K is already part of the TrellisShift mode.

Two reconstruction cores are screened independently:

* the published MUL1 -> FP16 -> E4M3 path; and
* quantization-aware MUL1 (QD), which adds deterministic fractional bits
  before E4M3 rounding and reaches a denser useful E4M3 alphabet.

This is a synthetic, CPU-only proposal search.  It must not be used as the
production selection result.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import torch

from kquant.exl3_reference import decode_mul1_e4m3_states
from scripts.experiment_trellis_e4m3_procedural_gaussian import (
    _parse_floats,
    _parse_ints,
    _sse,
)
from scripts.experiment_trellis_e4m3_procedural_v2 import (
    Candidate,
    _premix_many,
    _states,
    mul1_qd_codebook,
    mul1_qd_multixor_codebook,
)


def _tap_sets(min_shift: int, max_shift: int, max_taps: int) -> tuple[tuple[int, ...], ...]:
    shifts = tuple(range(min_shift, max_shift + 1))
    return ((),) + tuple(
        taps
        for count in range(1, max_taps + 1)
        for taps in combinations(shifts, count)
    )


def _candidate(family: str, taps: tuple[int, ...], phase: float) -> Candidate:
    if family == "mul1_e4m3":
        codebook = decode_mul1_e4m3_states(_premix_many(_states(), taps)).float()
        tag = "identity" if not taps else "_".join(str(value) for value in taps)
        return Candidate(
            family=family,
            name=f"mul1_e4m3_xlm{tag}",
            codebook=codebook.contiguous(),
            parameters={"left_xorshifts": list(taps)},
            estimated_scalar_ops=(),
        )
    if family != "mul1_qd":
        raise ValueError(f"unknown family {family}")
    if taps:
        return mul1_qd_multixor_codebook(
            left_shifts=taps,
            dither_bits=3,
            dither_shift=7,
            phase=phase,
        )
    return mul1_qd_codebook(
        left_shift=0,
        dither_bits=3,
        dither_shift=7,
        phase=phase,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    search = torch.randn(
        args.search_sequences, args.length, generator=generator, dtype=torch.float32
    )
    gain_fit = torch.randn(
        args.gain_fit_sequences,
        args.length,
        generator=generator,
        dtype=torch.float32,
    )
    evaluation = torch.randn(
        args.evaluation_sequences,
        args.length,
        generator=generator,
        dtype=torch.float32,
    )
    evaluation_energy = float(evaluation.double().square().sum().item())
    tap_sets = _tap_sets(args.min_shift, args.max_shift, args.max_taps)
    families = ("mul1_e4m3", "mul1_qd")
    results: dict[str, object] = {}
    started = time.perf_counter()
    for bits in args.rates:
        rate_rows: dict[str, object] = {}
        for family in families:
            search_gain = (
                args.mul1_search_gain
                if family == "mul1_e4m3"
                else args.qd_search_gain
            )
            proposals = []
            for taps in tap_sets:
                candidate = _candidate(family, taps, args.qd_phase)
                proposals.append(
                    {
                        "taps": list(taps),
                        "name": candidate.name,
                        "search_sse": _sse(
                            search,
                            codebook=candidate.codebook,
                            bits=bits,
                            gain=search_gain,
                            max_trace_mib=args.max_trace_mib,
                        ),
                    }
                )
            proposals.sort(key=lambda row: float(row["search_sse"]))
            # The one-sequence proposal screen is intentionally cheap and can
            # be noisy.  Always carry the identity mapper into disjoint
            # evaluation so a searched result cannot "win" merely because it
            # displaced the actual baseline before the blind split.
            finalist_proposals = list(proposals[: args.finalists])
            identity = next(row for row in proposals if not row["taps"])
            if all(row["name"] != identity["name"] for row in finalist_proposals):
                finalist_proposals.append(identity)
            finalists = []
            for proposal in finalist_proposals:
                taps = tuple(int(value) for value in proposal["taps"])
                candidate = _candidate(family, taps, args.qd_phase)
                gain_rows = [
                    {
                        "gain": gain,
                        "sse": _sse(
                            gain_fit,
                            codebook=candidate.codebook,
                            bits=bits,
                            gain=gain,
                            max_trace_mib=args.max_trace_mib,
                        ),
                    }
                    for gain in args.gains
                ]
                selected = min(gain_rows, key=lambda row: float(row["sse"]))
                evaluation_sse = _sse(
                    evaluation,
                    codebook=candidate.codebook,
                    bits=bits,
                    gain=float(selected["gain"]),
                    max_trace_mib=args.max_trace_mib,
                )
                finalists.append(
                    {
                        **proposal,
                        "gain_grid": gain_rows,
                        "selected_gain": float(selected["gain"]),
                        "evaluation_sse": evaluation_sse,
                        "evaluation_nmse": evaluation_sse / evaluation_energy,
                    }
                )
            finalists.sort(key=lambda row: float(row["evaluation_sse"]))
            rate_rows[family] = {
                "search_gain": search_gain,
                "proposal_count": len(proposals),
                "top_search": proposals[: args.report_search],
                "finalists": finalists,
                "selected": finalists[0],
            }
            winner = finalists[0]
            print(
                f"K{bits} {family}: taps={winner['taps']} "
                f"gain={winner['selected_gain']:.3g} "
                f"nmse={winner['evaluation_nmse']:.8f}",
                flush=True,
            )
        results[str(bits)] = rate_rows
    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_premix_gaussian_search",
        "schema_version": 1,
        "scope": {
            "warning": (
                "Synthetic iid Gaussian proposal search only. Refit and validate "
                "on disjoint official-weight and dense-H functional corpora."
            ),
            "window_bits": 16,
            "rates": list(args.rates),
            "tap_sets": len(tap_sets),
            "min_shift": args.min_shift,
            "max_shift": args.max_shift,
            "max_taps": args.max_taps,
            "qd_phase": args.qd_phase,
            "seed": args.seed,
            "search_sequences": args.search_sequences,
            "gain_fit_sequences": args.gain_fit_sequences,
            "evaluation_sequences": args.evaluation_sequences,
            "sequence_length": args.length,
            "evaluation_energy": evaluation_energy,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=_parse_ints, default=(3, 4, 5))
    parser.add_argument("--gains", type=_parse_floats, default=(0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35))
    parser.add_argument("--mul1-search-gain", type=float, default=0.95)
    parser.add_argument("--qd-search-gain", type=float, default=1.15)
    parser.add_argument("--qd-phase", type=float, default=1.25)
    parser.add_argument("--min-shift", type=int, default=5)
    parser.add_argument("--max-shift", type=int, default=15)
    parser.add_argument("--max-taps", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=6)
    parser.add_argument("--report-search", type=int, default=12)
    parser.add_argument("--search-sequences", type=int, default=1)
    parser.add_argument("--gain-fit-sequences", type=int, default=4)
    parser.add_argument("--evaluation-sequences", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-premix-gaussian-search-v1.json"),
    )
    args = parser.parse_args()
    if any(bits not in (2, 3, 4, 5) for bits in args.rates):
        parser.error("rates must be drawn from K2--K5")
    if not 5 <= args.min_shift <= args.max_shift <= 15:
        parser.error("require 5 <= min shift <= max shift <= 15")
    if args.max_taps not in (1, 2, 3):
        parser.error("--max-taps must be 1, 2, or 3")
    if min(
        args.finalists,
        args.report_search,
        args.search_sequences,
        args.gain_fit_sequences,
        args.evaluation_sequences,
        args.length,
        args.cpu_threads,
        args.max_trace_mib,
    ) < 1:
        parser.error("counts, length, threads, and trace size must be positive")
    if args.length % 2:
        parser.error("sequence length must be even")
    payload = run(args)
    print(
        json.dumps(
            {"output": str(args.output), "elapsed_seconds": payload["elapsed_seconds"]},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
