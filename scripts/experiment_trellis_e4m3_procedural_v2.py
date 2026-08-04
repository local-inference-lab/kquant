#!/usr/bin/env python3
"""Screen table-free L16 -> E4M3 trellis reconstruction functions.

This is deliberately separate from the production encoder.  It compares
cheap procedural labellings against the already-fitted L16 E4M3 oracle using
only whole-codebook and trellis-geometry proxies.  A survivor still has to be
rerun through Viterbi on the disjoint weight corpus and, ultimately, through
the production BlockLDLQ objective.

The main candidate is ``mul1_qd`` (quantization-aware dither):

    u = state ^ ((state << r) & 0xffff)       # optional invertible pre-mix
    p = uint32(u * 0x83dcd12d)
    q = (bytesum(p) << d) + hash_d(p) - center
    y = E4M3(fp16(q) * fp16(alpha))

Unlike post-hoc rounding of MUL1, the fixed-point residual gives the E4M3
converter enough resolution to reach the dense levels near zero.  The
left-xorshift is an invertible permutation of transition indices: it does not
alter the global marginal, but it changes which labels share an origin state
and therefore controls current-output-preserving successor choices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from safetensors.torch import load_file
import torch

from kquant.exl3_reference import MUL1_MULT, decode_mul1_e4m3_states


MASK32 = 0xFFFFFFFF
SECOND_MULTIPLIERS = (
    0xCBAC1FED,  # EXL MCG multiplier
    0x020762D1,  # QTIP 1MAD multiplier
    0x9E3779B1,  # odd Weyl/hash multiplier
)
REPORTED_RATES = (2, 3, 4, 5)
SCREEN_RATES = (3, 4, 5)


@dataclass(frozen=True)
class Candidate:
    family: str
    name: str
    codebook: torch.Tensor
    parameters: dict[str, object]
    estimated_scalar_ops: tuple[str, ...]


def _states() -> torch.Tensor:
    return torch.arange(1 << 16, dtype=torch.int64)


def _premix(states: torch.Tensor, left_shift: int) -> torch.Tensor:
    if left_shift == 0:
        return states
    if not 1 <= left_shift <= 15:
        raise ValueError("left_shift must be zero or in [1, 15]")
    return (states ^ ((states << left_shift) & 0xFFFF)) & 0xFFFF


def _premix_many(states: torch.Tensor, left_shifts: tuple[int, ...]) -> torch.Tensor:
    """Apply a cheap invertible GF(2) transition-label pre-mix.

    Every shifted term is strictly above the diagonal of the 16-bit linear
    map, while the unshifted identity supplies its diagonal.  The resulting
    map is invertible for any unique set of shifts.  Shifts of at least five
    also preserve the low branch-symbol bits for every K2--K5 rate here.
    """

    if len(set(left_shifts)) != len(left_shifts):
        raise ValueError("left_shifts must be unique")
    result = states.clone()
    for left_shift in left_shifts:
        if not 1 <= left_shift <= 15:
            raise ValueError("left shifts must be in [1, 15]")
        result ^= (states << left_shift) & 0xFFFF
    return result & 0xFFFF


def _product_and_bytes(
    states: torch.Tensor, multiplier: int = MUL1_MULT
) -> tuple[torch.Tensor, torch.Tensor]:
    product = (states * multiplier) & MASK32
    byte_columns = torch.stack(
        tuple((product >> shift) & 0xFF for shift in (0, 8, 16, 24)), dim=1
    )
    return product, byte_columns


def _e4m3_from_fixedpoint(q: torch.Tensor, phase: float) -> tuple[torch.Tensor, float]:
    """Normalize integer ``q`` in FP16, then round exactly to finite E4M3."""

    if not math.isfinite(phase) or phase <= 0.0:
        raise ValueError("phase must be positive and finite")
    standard_deviation = float(q.double().std(unbiased=False).item())
    if standard_deviation == 0.0:
        raise ValueError("fixed-point statistic has zero variance")
    # This is the actual compiled constant: rounding it to FP16 here makes the
    # structural screen reproduce the proposed decoder rather than an FP32
    # idealization.
    alpha = float(torch.tensor(phase / standard_deviation).half().item())
    reconstructed = (
        (q.to(torch.float16) * torch.tensor(alpha, dtype=torch.float16))
        .to(torch.float8_e4m3fn)
        .float()
        .contiguous()
    )
    return reconstructed, alpha


def mul1_qd_codebook(
    *,
    left_shift: int,
    dither_bits: int,
    dither_shift: int,
    phase: float,
) -> Candidate:
    """MUL1 byte sum with deterministic sub-lattice resolution."""

    if not 1 <= dither_bits <= 6:
        raise ValueError("dither_bits must be in [1, 6]")
    if not 0 <= dither_shift <= 31:
        raise ValueError("dither_shift must be in [0, 31]")
    state = _premix(_states(), left_shift)
    product, byte_columns = _product_and_bytes(state)
    byte_sum = byte_columns.sum(dim=1)
    # Three separated folds make every output bit depend on more than one
    # product byte while remaining three shifts and XORs in the literal form.
    folded = product ^ (product >> 13) ^ (product >> 23)
    mask = (1 << dither_bits) - 1
    residual = (folded >> dither_shift) & mask
    statistic = (byte_sum << dither_bits) + residual
    center = int(round(float(statistic.double().mean().item())))
    q = statistic - center
    codebook, alpha = _e4m3_from_fixedpoint(q, phase)
    return Candidate(
        family="mul1_qd",
        name=(
            f"mul1_qd_xl{left_shift}_d{dither_bits}_s{dither_shift}_p{phase:g}"
        ),
        codebook=codebook,
        parameters={
            "multiplier": MUL1_MULT,
            "multiplier_hex": f"0x{MUL1_MULT:08X}",
            "left_xorshift": left_shift,
            "dither_bits": dither_bits,
            "dither_shift": dither_shift,
            "center": center,
            "fp16_alpha": alpha,
            "phase": phase,
        },
        estimated_scalar_ops=(
            "optional shl+xor pre-mix",
            "mul.lo.u32",
            "dp4a byte sum",
            "three shifts/two xor plus mask for residual",
            "shift+add+subtract",
            "s32->f16 conversion",
            "f16 multiply",
            "f16->e4m3 conversion",
        ),
    )


def mul1_qd_multixor_codebook(
    *,
    left_shifts: tuple[int, ...],
    dither_bits: int,
    dither_shift: int,
    phase: float,
) -> Candidate:
    """Quantization-aware MUL1 with a multi-tap transition-label pre-mix."""

    if not left_shifts:
        raise ValueError("left_shifts must be nonempty")
    if not 1 <= dither_bits <= 6:
        raise ValueError("dither_bits must be in [1, 6]")
    if not 0 <= dither_shift <= 31:
        raise ValueError("dither_shift must be in [0, 31]")
    state = _premix_many(_states(), left_shifts)
    product, byte_columns = _product_and_bytes(state)
    byte_sum = byte_columns.sum(dim=1)
    folded = product ^ (product >> 13) ^ (product >> 23)
    mask = (1 << dither_bits) - 1
    residual = (folded >> dither_shift) & mask
    statistic = (byte_sum << dither_bits) + residual
    center = int(round(float(statistic.double().mean().item())))
    q = statistic - center
    codebook, alpha = _e4m3_from_fixedpoint(q, phase)
    shift_tag = "_".join(str(value) for value in left_shifts)
    return Candidate(
        family="mul1_qd_multixor",
        name=(
            f"mul1_qd_xlm{shift_tag}_d{dither_bits}_s{dither_shift}_p{phase:g}"
        ),
        codebook=codebook,
        parameters={
            "multiplier": MUL1_MULT,
            "multiplier_hex": f"0x{MUL1_MULT:08X}",
            "left_xorshifts": list(left_shifts),
            "dither_bits": dither_bits,
            "dither_shift": dither_shift,
            "center": center,
            "fp16_alpha": alpha,
            "phase": phase,
        },
        estimated_scalar_ops=(
            f"{len(left_shifts)} shl+xor transition-label pre-mix taps",
            "mul.lo.u32",
            "dp4a byte sum",
            "three shifts/two xor plus mask for residual",
            "shift+add+subtract",
            "s32->f16 conversion",
            "f16 multiply",
            "f16->e4m3 conversion",
        ),
    )


def weighted_dp4a_codebook(
    *,
    weights: tuple[int, int, int, int],
    left_shift: int,
    phase: float,
) -> Candidate:
    """One-product, one-DP4A family with a learned-at-compile-time dot vector."""

    if any(value < 1 or value > 127 for value in weights):
        raise ValueError("DP4A weights must be unsigned bytes in [1, 127]")
    state = _premix(_states(), left_shift)
    _, byte_columns = _product_and_bytes(state)
    weight = torch.tensor(weights, dtype=torch.int64)
    statistic = (byte_columns * weight).sum(dim=1)
    center = int(round(float(statistic.double().mean().item())))
    q = statistic - center
    codebook, alpha = _e4m3_from_fixedpoint(q, phase)
    packed_weights = sum(value << (8 * index) for index, value in enumerate(weights))
    return Candidate(
        family="weighted_dp4a",
        name=f"weighted_{''.join(map(str, weights))}_xl{left_shift}_p{phase:g}",
        codebook=codebook,
        parameters={
            "multiplier": MUL1_MULT,
            "multiplier_hex": f"0x{MUL1_MULT:08X}",
            "packed_dp4a_weights": packed_weights,
            "packed_dp4a_weights_hex": f"0x{packed_weights:08X}",
            "weights_low_to_high_byte": list(weights),
            "left_xorshift": left_shift,
            "center": center,
            "fp16_alpha": alpha,
            "phase": phase,
        },
        estimated_scalar_ops=(
            "optional shl+xor pre-mix",
            "mul.lo.u32",
            "dp4a with packed immediate weights",
            "subtract",
            "s32->f16 conversion",
            "f16 multiply",
            "f16->e4m3 conversion",
        ),
    )


def dual_sum_codebook(
    *,
    second_multiplier: int,
    left_shift: int,
    phase: float,
) -> Candidate:
    """Eight-byte CLT statistic from two independent products."""

    state = _premix(_states(), left_shift)
    _, first_bytes = _product_and_bytes(state, MUL1_MULT)
    _, second_bytes = _product_and_bytes(state, second_multiplier)
    statistic = first_bytes.sum(dim=1) + second_bytes.sum(dim=1)
    center = int(round(float(statistic.double().mean().item())))
    q = statistic - center
    codebook, alpha = _e4m3_from_fixedpoint(q, phase)
    return Candidate(
        family="dual_sum",
        name=f"dual_{second_multiplier:08x}_xl{left_shift}_p{phase:g}",
        codebook=codebook,
        parameters={
            "first_multiplier": MUL1_MULT,
            "first_multiplier_hex": f"0x{MUL1_MULT:08X}",
            "second_multiplier": second_multiplier,
            "second_multiplier_hex": f"0x{second_multiplier:08X}",
            "left_xorshift": left_shift,
            "center": center,
            "fp16_alpha": alpha,
            "phase": phase,
        },
        estimated_scalar_ops=(
            "optional shl+xor pre-mix",
            "two mul.lo.u32",
            "two dp4a byte sums",
            "add+subtract",
            "s32->f16 conversion",
            "f16 multiply",
            "f16->e4m3 conversion",
        ),
    )


def mul1_permuted_codebook(left_shift: int) -> Candidate:
    state = _premix(_states(), left_shift)
    codebook = decode_mul1_e4m3_states(state).float()
    return Candidate(
        family="mul1_permuted",
        name=f"published_mul1_xl{left_shift}",
        codebook=codebook,
        parameters={
            "multiplier": MUL1_MULT,
            "multiplier_hex": f"0x{MUL1_MULT:08X}",
            "left_xorshift": left_shift,
        },
        estimated_scalar_ops=(
            "optional shl+xor pre-mix",
            "published MUL1-E4M3 decoder",
        ),
    )


def _alias_summary(codebook: torch.Tensor, bits: int) -> dict[str, float | int]:
    branches = 1 << bits
    values = codebook.reshape(-1, branches)
    equality = values.unsqueeze(2) == values.unsqueeze(1)
    multiplicity = equality.sum(dim=2)
    first = ~equality.tril(diagonal=-1).any(dim=2)
    distinct = first.sum(dim=1)
    return {
        "states_with_alias_fraction": float((distinct < branches).float().mean().item()),
        "edges_with_alias_fraction": float((multiplicity > 1).float().mean().item()),
        "mean_distinct_labels_per_state": float(distinct.float().mean().item()),
        "min_distinct_labels_per_state": int(distinct.min().item()),
        "max_alias_multiplicity": int(multiplicity.max().item()),
        "conditional_alias_bits_uniform": float(multiplicity.float().log2().mean().item()),
    }


def _fit_positive_gain(candidate: torch.Tensor, target: torch.Tensor) -> float:
    denominator = candidate.double().square().sum()
    if not bool(denominator > 0):
        return 1.0
    return max(0.0, float((candidate.double() * target.double()).sum() / denominator))


def _nmse(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    gain = _fit_positive_gain(candidate, target)
    error = ((candidate.double() * gain - target.double()).square()).sum()
    energy = target.double().square().sum().clamp_min(1.0e-30)
    return {"gain": gain, "nmse": float(error / energy)}


def _distribution_nmse(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return _nmse(candidate.sort().values, target.sort().values)


def _outgoing_set_nmse(
    candidate: torch.Tensor, target: torch.Tensor, bits: int
) -> dict[str, float]:
    branches = 1 << bits
    candidate_sets = candidate.reshape(-1, branches).sort(dim=1).values
    target_sets = target.reshape(-1, branches).sort(dim=1).values
    return _nmse(candidate_sets, target_sets)


def _moments(values: torch.Tensor) -> dict[str, object]:
    x = values.double()
    centered = x - x.mean()
    variance = centered.square().mean()
    quantiles = torch.tensor(
        [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0],
        dtype=torch.float64,
    )
    return {
        "numeric_labels": int(torch.unique(values).numel()),
        "minimum": float(x.min()),
        "maximum": float(x.max()),
        "mean": float(x.mean()),
        "standard_deviation": float(x.std(unbiased=False)),
        "skewness": float(centered.pow(3).mean() / variance.pow(1.5).clamp_min(1.0e-30)),
        "excess_kurtosis": float(
            centered.pow(4).mean() / variance.square().clamp_min(1.0e-30) - 3.0
        ),
        "zero_fraction": float((values == 0).float().mean()),
        "quantiles": {
            f"{float(q):g}": float(value)
            for q, value in zip(quantiles, torch.quantile(x, quantiles))
        },
    }


def _candidate_metrics(
    candidate: Candidate,
    *,
    learned: torch.Tensor,
    corrected_mul1: torch.Tensor,
    target_alias: dict[int, dict[str, float | int]],
) -> dict[str, object]:
    aliases = {
        bits: _alias_summary(candidate.codebook, bits) for bits in REPORTED_RATES
    }
    alias_distance = 0.0
    for bits in SCREEN_RATES:
        for key in ("states_with_alias_fraction", "edges_with_alias_fraction"):
            alias_distance += (
                float(aliases[bits][key]) - float(target_alias[bits][key])
            ) ** 2
    moments = _moments(candidate.codebook)
    # This score is only for ordering the CPU structural screen.  It cannot
    # substitute for a Viterbi encode: aligned table SSE in particular is not
    # invariant to a useful permutation of state labels.
    distribution = _distribution_nmse(candidate.codebook, learned)
    k3_sets = _outgoing_set_nmse(candidate.codebook, learned, 3)
    coverage_shortfall = max(0.0, (140.0 - float(moments["numeric_labels"])) / 140.0)
    screen_score = (
        alias_distance
        + 0.25 * float(distribution["nmse"])
        + 0.10 * float(k3_sets["nmse"])
        + coverage_shortfall**2
    )
    return {
        "family": candidate.family,
        "name": candidate.name,
        "parameters": candidate.parameters,
        "estimated_scalar_ops": list(candidate.estimated_scalar_ops),
        "moments": moments,
        "aliases": {str(key): value for key, value in aliases.items()},
        "proxies": {
            "aligned_to_learned_e4m3": _nmse(candidate.codebook, learned),
            "aligned_to_corrected_mul1": _nmse(candidate.codebook, corrected_mul1),
            "sorted_distribution_to_learned_e4m3": distribution,
            "sorted_k3_outgoing_sets_to_learned_e4m3": k3_sets,
            "alias_distance_to_learned_e4m3": alias_distance,
            "structural_screen_score": screen_score,
        },
    }


def _candidates() -> Iterable[Candidate]:
    for left_shift in (0, 5, 7, 8, 9, 10, 12):
        yield mul1_permuted_codebook(left_shift)

    for left_shift in (0, 5, 7, 8, 9, 10, 12):
        for dither_bits in (2, 3, 4):
            for dither_shift in (0, 7, 15, 23):
                for phase in (0.875, 1.0, 1.125, 1.25, 1.5):
                    yield mul1_qd_codebook(
                        left_shift=left_shift,
                        dither_bits=dither_bits,
                        dither_shift=dither_shift,
                        phase=phase,
                    )

    # Sparse upper-triangular binary mixes cost no table bytes.  These
    # particular controls cover a range of K3/K4/K5 alias geometries.
    for left_shifts in (
        (8, 10),
        (8, 9, 11),
        (6, 8, 10),
        (4, 11, 12),
    ):
        for phase in (1.0, 1.125, 1.25, 1.375, 1.5):
            yield mul1_qd_multixor_codebook(
                left_shifts=left_shifts,
                dither_bits=3,
                dither_shift=7,
                phase=phase,
            )

    weight_vectors = (
        (1, 1, 1, 1),
        (1, 1, 1, 2),
        (1, 1, 2, 2),
        (1, 1, 2, 3),
        (1, 2, 3, 4),
        (1, 3, 5, 7),
    )
    for weights in weight_vectors:
        for left_shift in (0, 7, 8, 10):
            for phase in (0.875, 1.0, 1.125, 1.25, 1.5):
                yield weighted_dp4a_codebook(
                    weights=weights, left_shift=left_shift, phase=phase
                )

    for second_multiplier in SECOND_MULTIPLIERS:
        for left_shift in (0, 7, 8, 10):
            for phase in (0.875, 1.0, 1.125, 1.25, 1.5):
                yield dual_sum_codebook(
                    second_multiplier=second_multiplier,
                    left_shift=left_shift,
                    phase=phase,
                )


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.cpu_threads)
    oracle = load_file(str(args.oracle_codebooks), device="cpu")
    correction = load_file(str(args.correction_codebooks), device="cpu")
    learned = oracle["learned_e4m3"].float().contiguous()
    corrected_mul1 = correction[
        "binary_e4m3_mul1_anchor_plus_alt.expanded"
    ].float().contiguous()
    target_alias = {
        bits: _alias_summary(learned, bits) for bits in REPORTED_RATES
    }

    rows = []
    for index, candidate in enumerate(_candidates(), start=1):
        rows.append(
            _candidate_metrics(
                candidate,
                learned=learned,
                corrected_mul1=corrected_mul1,
                target_alias=target_alias,
            )
        )
        if index % 100 == 0:
            print(f"screened {index} candidates", flush=True)
    rows.sort(key=lambda row: float(row["proxies"]["structural_screen_score"]))

    by_family: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_procedural_structural_screen",
        "schema_version": 1,
        "scope": {
            "window_bits": 16,
            "rates": list(REPORTED_RATES),
            "structural_screen_rates": list(SCREEN_RATES),
            "transition_labels": 65536,
            "oracle_codebooks": str(args.oracle_codebooks),
            "correction_codebooks": str(args.correction_codebooks),
            "warning": (
                "Structural and table-level proxies only. A production decision requires "
                "fresh Viterbi paths, held-out matrix scoring, and dense-H BlockLDLQ."
            ),
        },
        "learned_e4m3_reference": {
            "moments": _moments(learned),
            "aliases": {str(key): value for key, value in target_alias.items()},
        },
        "candidate_count": len(rows),
        "top_overall": rows[: args.top],
        "top_by_family": {
            family: family_rows[: args.top_per_family]
            for family, family_rows in by_family.items()
        },
        "all_candidates": rows if args.include_all else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
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
        default=Path("out/trellis-e4m3-procedural-structural-v1.json"),
    )
    parser.add_argument("--top", type=int, default=16)
    parser.add_argument("--top-per-family", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--include-all", action="store_true")
    args = parser.parse_args()
    if args.top < 1 or args.top_per_family < 1 or args.cpu_threads < 1:
        parser.error("--top, --top-per-family, and --cpu-threads must be positive")
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": payload["candidate_count"],
                "top": [
                    {
                        "name": row["name"],
                        "numeric_labels": row["moments"]["numeric_labels"],
                        "screen_score": row["proxies"]["structural_screen_score"],
                    }
                    for row in payload["top_overall"][:8]
                ],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
