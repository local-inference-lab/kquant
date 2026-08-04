#!/usr/bin/env python3
"""CPU-only Viterbi screen for procedural L16 -> E4M3 codebooks.

This deliberately uses disjoint synthetic Gaussian sequences rather than the
official checkpoint.  Hadamard-regularized weight tiles are close enough to
Gaussian for this to reject bad graph labellings without touching a GPU or
the running all-expert candidate pool.  It is not a substitute for the blind
official-weight corpus or production dense-H BlockLDLQ.

The screen includes K3, K4, and K5 so it directly reports the scalar
rate-transfer shape around a uniform-K4 endpoint:

    donor cost    = D3 - D4
    recipient gain = D4 - D5
    equal exchange ratio = (D3 + D5) / (2 * D4)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import load_file
import torch

from kquant.trellis_window import encode_trellis
from scripts.experiment_trellis_e4m3_procedural_v2 import (
    Candidate,
    dual_sum_codebook,
    mul1_permuted_codebook,
    mul1_qd_codebook,
    mul1_qd_multixor_codebook,
    weighted_dp4a_codebook,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item)
    if not result or any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated gains")
    return result


def _fixed_candidate(name: str, family: str, codebook: torch.Tensor) -> Candidate:
    return Candidate(
        family=family,
        name=name,
        codebook=codebook.float().contiguous(),
        parameters={},
        estimated_scalar_ops=(),
    )


def _shortlist(
    oracle: dict[str, torch.Tensor], correction: dict[str, torch.Tensor]
) -> tuple[Candidate, ...]:
    return (
        _fixed_candidate(
            "procedural_mcg_fp16", "reference", oracle["procedural_mcg_fp16"]
        ),
        _fixed_candidate("learned_e4m3", "oracle", oracle["learned_e4m3"]),
        mul1_permuted_codebook(0),
        _fixed_candidate(
            "binary_corrected_mul1_e4m3",
            "table_oracle",
            correction["binary_e4m3_mul1_anchor_plus_alt.expanded"],
        ),
        mul1_permuted_codebook(8),
        mul1_qd_codebook(
            left_shift=8, dither_bits=3, dither_shift=7, phase=1.25
        ),
        mul1_qd_multixor_codebook(
            left_shifts=(8, 10), dither_bits=3, dither_shift=7, phase=1.25
        ),
        mul1_qd_multixor_codebook(
            left_shifts=(8, 9, 11),
            dither_bits=3,
            dither_shift=7,
            phase=1.25,
        ),
        mul1_qd_multixor_codebook(
            left_shifts=(6, 8, 10),
            dither_bits=3,
            dither_shift=7,
            phase=1.25,
        ),
        weighted_dp4a_codebook(
            weights=(1, 3, 5, 7), left_shift=7, phase=1.5
        ),
        dual_sum_codebook(
            second_multiplier=0x020762D1, left_shift=8, phase=1.5
        ),
    )


def _sse(
    source: torch.Tensor,
    *,
    codebook: torch.Tensor,
    bits: int,
    gain: float,
    max_trace_mib: int,
) -> float:
    encoded = encode_trellis(
        source * gain,
        bits,
        16,
        codebook,
        max_trace_bytes=max_trace_mib << 20,
        return_transitions=False,
    )
    return float((encoded.squared_error.double() / (gain * gain)).sum().item())


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    fitting = torch.randn(
        args.fit_sequences, args.length, generator=generator, dtype=torch.float32
    )
    evaluation = torch.randn(
        args.evaluation_sequences,
        args.length,
        generator=generator,
        dtype=torch.float32,
    )
    evaluation_energy = float(evaluation.double().square().sum().item())
    oracle = load_file(str(args.oracle_codebooks), device="cpu")
    correction = load_file(str(args.correction_codebooks), device="cpu")
    rows: dict[str, object] = {}
    started = time.perf_counter()
    for candidate_index, candidate in enumerate(_shortlist(oracle, correction), 1):
        by_rate: dict[str, object] = {}
        for bits in args.rates:
            fit_rows = [
                {
                    "gain": gain,
                    "sse": _sse(
                        fitting,
                        codebook=candidate.codebook,
                        bits=bits,
                        gain=gain,
                        max_trace_mib=args.max_trace_mib,
                    ),
                }
                for gain in args.gains
            ]
            selected = min(fit_rows, key=lambda row: float(row["sse"]))
            evaluation_sse = _sse(
                evaluation,
                codebook=candidate.codebook,
                bits=bits,
                gain=float(selected["gain"]),
                max_trace_mib=args.max_trace_mib,
            )
            by_rate[str(bits)] = {
                "selected_gain": float(selected["gain"]),
                "fitting_grid": fit_rows,
                "evaluation_sse": evaluation_sse,
                "evaluation_nmse": evaluation_sse / evaluation_energy,
            }
        if all(str(bits) in by_rate for bits in (3, 4, 5)):
            d3 = float(by_rate["3"]["evaluation_sse"])
            d4 = float(by_rate["4"]["evaluation_sse"])
            d5 = float(by_rate["5"]["evaluation_sse"])
            donor = d3 - d4
            recipient = d4 - d5
            rate_transfer = {
                "k3_to_k4_donor_cost": donor,
                "k4_to_k5_recipient_gain": recipient,
                "donor_cost_over_recipient_gain": (
                    donor / recipient if recipient > 0.0 else None
                ),
                "equal_k3_k5_exchange_over_uniform_k4": (d3 + d5) / (2.0 * d4),
            }
        else:
            rate_transfer = None
        rows[candidate.name] = {
            "family": candidate.family,
            "parameters": candidate.parameters,
            "by_rate": by_rate,
            "rate_transfer": rate_transfer,
        }
        print(
            f"[{candidate_index}] {candidate.name}: "
            + " ".join(
                f"K{bits}={float(by_rate[str(bits)]['evaluation_nmse']):.7f}"
                for bits in args.rates
            ),
            flush=True,
        )
    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_procedural_gaussian_viterbi_screen",
        "schema_version": 1,
        "scope": {
            "warning": (
                "Synthetic iid Gaussian, diagonal SSE, CPU reference Viterbi only; "
                "official-weight and dense-H validation remain mandatory."
            ),
            "window_bits": 16,
            "rates": list(args.rates),
            "fit_sequences": args.fit_sequences,
            "evaluation_sequences": args.evaluation_sequences,
            "sequence_length": args.length,
            "seed": args.seed,
            "gains": list(args.gains),
            "evaluation_energy": evaluation_energy,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=_parse_ints, default=(3, 4, 5))
    parser.add_argument("--gains", type=_parse_floats, default=(0.65, 0.75, 0.85, 0.95, 1.05))
    parser.add_argument("--fit-sequences", type=int, default=4)
    parser.add_argument("--evaluation-sequences", type=int, default=16)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260803)
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
        "--correction-codebooks",
        type=Path,
        default=Path("out/trellis-e4m3-binary-subcode-predictors-v4.safetensors"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-procedural-gaussian-v1.json"),
    )
    args = parser.parse_args()
    if any(bits not in (2, 3, 4, 5) for bits in args.rates):
        parser.error("rates must be drawn from K2--K5")
    if min(args.fit_sequences, args.evaluation_sequences, args.length) < 1:
        parser.error("sequence counts and length must be positive")
    if args.length % 2:
        parser.error("sequence length must be even")
    if args.cpu_threads < 1 or args.max_trace_mib < 1:
        parser.error("--cpu-threads and --max-trace-mib must be positive")
    payload = run(args)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": payload["elapsed_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
