#!/usr/bin/env python3
"""Search lookup-free MUL1-style E4M3 procedural codebooks.

The decoder family is deliberately tiny:

    p = uint32(state * multiplier)
    z = byte_sum(p)
    y = E4M3(slope * z + intercept)

Only the odd uint32 multiplier and two affine constants are learned.  This
keeps the representation compatible with the MUL1 multiply/DP4A structure
and avoids the 8 KiB transition lookup used by the binary-subcode oracle.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from safetensors.torch import load_file
import torch

from kquant import constants as C
from kquant.exl3_reference import MUL1_MULT, decode_mul1_states
from kquant.source_weights import OfficialMXFP4Store
from kquant.trellis_window import fp8_constrained_codebook, geometry
from scripts.experiment_trellis_e4m3_state_palettes import (
    _save_tensors,
    _transition_statistics,
    _validate_groups,
    _validate_oracle_signature,
)
from scripts.experiment_trellis_learned_codebooks import _score, _slice_each_group
from scripts.experiment_trellis_window_fp8 import (
    _atomic_json,
    _comparison,
    _paired_metrics,
    _parse_ints,
    collect_corpus,
    fit_group_gains,
)


def _candidate_multipliers(count: int, seed: int) -> list[int]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    rng = random.Random(seed)
    values = {MUL1_MULT}
    for bit in range(32):
        values.add((MUL1_MULT ^ (1 << bit)) | 1)
    for shift in (0, 8, 16, 24):
        for delta in (-31, -15, -7, -3, -1, 1, 3, 7, 15, 31):
            values.add((MUL1_MULT + (delta << shift)) & 0xFFFFFFFF | 1)
    while len(values) < count:
        values.add(rng.getrandbits(32) | 1)
    return sorted(values)[:count]


def _mutations(parents: list[int], seed: int) -> list[int]:
    rng = random.Random(seed)
    values = set(parents)
    for parent in parents:
        for bit in range(32):
            values.add((parent ^ (1 << bit)) | 1)
        for _ in range(32):
            mask = (1 << rng.randrange(32)) | (1 << rng.randrange(32))
            values.add((parent ^ mask) | 1)
        for delta in (-257, -65, -17, -5, -3, 3, 5, 17, 65, 257):
            values.add((parent + delta) & 0xFFFFFFFF | 1)
    return sorted(values)


def _fit_affine_candidates(
    multipliers: list[int],
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    batch_size: int,
) -> list[dict[str, object]]:
    device = targets.device
    states = torch.arange(targets.numel(), dtype=torch.int64, device=device)
    weight = weights.double()
    target = targets.double()
    sw = weight.sum()
    sy = (weight * target).sum()
    rows: list[dict[str, object]] = []
    for start in range(0, len(multipliers), batch_size):
        local = torch.tensor(
            multipliers[start : start + batch_size], dtype=torch.int64, device=device
        )
        product = (local[:, None] * states[None, :]) & 0xFFFFFFFF
        x = (
            (product & 0xFF)
            + ((product >> 8) & 0xFF)
            + ((product >> 16) & 0xFF)
            + ((product >> 24) & 0xFF)
        ).double()
        sx = (x * weight).sum(dim=1)
        sxx = (x.square() * weight).sum(dim=1)
        sxy = (x * (weight * target)).sum(dim=1)
        denominator = sw * sxx - sx.square()
        slope = (sw * sxy - sx * sy) / denominator.clamp_min(1.0e-12)
        intercept = (sy - slope * sx) / sw
        reconstructed = (
            (x * slope[:, None] + intercept[:, None])
            .float()
            .to(torch.float8_e4m3fn)
            .float()
        )
        objective = (
            weight[None, :] * (reconstructed.double() - target[None, :]).square()
        ).sum(dim=1)
        for index, multiplier in enumerate(local.tolist()):
            rows.append(
                {
                    "multiplier": int(multiplier),
                    "multiplier_hex": f"0x{int(multiplier):08X}",
                    "slope": float(slope[index].item()),
                    "intercept": float(intercept[index].item()),
                    "proxy_objective": float(objective[index].item()),
                }
            )
        del product, x, reconstructed
    rows.sort(key=lambda row: float(row["proxy_objective"]))
    return rows


def _expand(row: dict[str, object], *, device: torch.device) -> torch.Tensor:
    states = torch.arange(1 << 16, dtype=torch.int64, device=device)
    multiplier = int(row["multiplier"])
    product = (states * multiplier) & 0xFFFFFFFF
    byte_sum = (
        (product & 0xFF)
        + ((product >> 8) & 0xFF)
        + ((product >> 16) & 0xFF)
        + ((product >> 24) & 0xFF)
    ).float()
    return (
        (byte_sum * float(row["slope"]) + float(row["intercept"]))
        .to(torch.float8_e4m3fn)
        .float()
        .contiguous()
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the MUL1 search requires CUDA")
    if args.bits != 3 or args.window_bits != 16:
        raise ValueError("this controlled search requires K3/L16")
    graph = geometry(args.bits, args.window_bits)
    oracle = json.loads(args.oracle_json.read_text())
    _validate_oracle_signature(oracle, args)
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    training_count = args.gain_tiles + args.fitting_tiles + args.selection_tiles
    corpus = collect_corpus(
        store,
        layers=args.layers,
        experts=args.experts,
        training_tiles_per_matrix=training_count,
        evaluation_tiles_per_matrix=args.evaluation_tiles,
        seed=args.seed,
        device=device,
    )
    _validate_groups(corpus, oracle["gain_fit"]["selected_gains"])
    gain_source, gain_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=0,
        count=args.gain_tiles,
    )
    fitting, fitting_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=args.gain_tiles,
        count=args.fitting_tiles,
    )
    selection, selection_group = _slice_each_group(
        corpus.training,
        corpus.training_group,
        groups=corpus.groups,
        start=args.gain_tiles + args.fitting_tiles,
        count=args.selection_tiles,
    )
    oracle_tensors = load_file(str(args.oracle_codebooks), device=str(device))
    mcg = oracle_tensors["procedural_mcg_fp16"].float()
    learned_e4m3 = oracle_tensors["learned_e4m3"].float()
    mul1 = decode_mul1_states(torch.arange(graph.transitions, device=device)).float()
    projected_mul1 = fp8_constrained_codebook(mul1)

    baseline_gain_fit = fit_group_gains(
        gain_source,
        gain_group,
        groups=corpus.groups,
        window_bits=args.window_bits,
        gain_fit_rate=3,
        codebook=projected_mul1,
        max_trace_mib=args.max_trace_mib,
    )
    gains = baseline_gain_fit["selected_gains_tensor"]
    fitting_gain = gains.index_select(0, fitting_group.long())
    fitting_targets = fitting * fitting_gain.unsqueeze(1)
    fitting_weights = fitting_gain.reciprocal().square().unsqueeze(1).expand_as(fitting)
    fitting_encoded, _, _ = _score(
        fitting,
        fitting_group,
        gains=gains,
        codebook=projected_mul1,
        bits=args.bits,
        window_bits=args.window_bits,
        max_trace_mib=args.max_trace_mib,
        return_transitions=True,
    )
    assert fitting_encoded.transition_indices is not None
    transition_targets, transition_weights = _transition_statistics(
        fitting_targets,
        fitting_weights,
        fitting_encoded.transition_indices,
        projected_mul1,
        ridge_pseudocount=args.ridge,
    )

    multipliers = _candidate_multipliers(args.candidates, args.seed)
    rounds: list[dict[str, object]] = []
    proxy_rows = _fit_affine_candidates(
        multipliers,
        transition_targets,
        transition_weights,
        batch_size=args.batch_size,
    )
    rounds.append({"round": 0, "candidates": len(proxy_rows), "best": proxy_rows[:16]})
    for round_index in range(1, args.mutation_rounds + 1):
        parents = [int(row["multiplier"]) for row in proxy_rows[: args.parents]]
        mutations = _mutations(parents, args.seed + round_index)
        proxy_rows = _fit_affine_candidates(
            mutations,
            transition_targets,
            transition_weights,
            batch_size=args.batch_size,
        )
        rounds.append(
            {"round": round_index, "candidates": len(proxy_rows), "best": proxy_rows[:16]}
        )

    finalists_by_multiplier: dict[int, dict[str, object]] = {
        int(row["multiplier"]): row for row in proxy_rows[: args.finalists]
    }
    # Always evaluate the published MUL1 multiplier under the learned affine
    # mapping, even when the proxy search displaces it.
    baseline_affine = _fit_affine_candidates(
        [MUL1_MULT],
        transition_targets,
        transition_weights,
        batch_size=1,
    )[0]
    finalists_by_multiplier[MUL1_MULT] = baseline_affine

    finalist_rows = []
    finalist_codebooks: dict[int, torch.Tensor] = {}
    for multiplier, row in finalists_by_multiplier.items():
        codebook = _expand(row, device=device)
        finalist_codebooks[multiplier] = codebook
        _, selection_sse, seconds = _score(
            selection,
            selection_group,
            gains=gains,
            codebook=codebook,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
            return_transitions=False,
        )
        finalist_rows.append(
            {**row, "selection_sse_fixed_gains": selection_sse, "seconds": seconds}
        )
        print(
            f"{row['multiplier_hex']} proxy={row['proxy_objective']:.7g} "
            f"selection={selection_sse:.7g}",
            flush=True,
        )
    finalist_rows.sort(key=lambda row: float(row["selection_sse_fixed_gains"]))

    refit_rows = []
    refit_gains: dict[int, torch.Tensor] = {}
    for row in finalist_rows[: args.gain_refits]:
        multiplier = int(row["multiplier"])
        codebook = finalist_codebooks[multiplier]
        gain_fit = fit_group_gains(
            gain_source,
            gain_group,
            groups=corpus.groups,
            window_bits=args.window_bits,
            gain_fit_rate=3,
            codebook=codebook,
            max_trace_mib=args.max_trace_mib,
        )
        candidate_gains = gain_fit["selected_gains_tensor"]
        _, selection_sse, seconds = _score(
            selection,
            selection_group,
            gains=candidate_gains,
            codebook=codebook,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
            return_transitions=False,
        )
        refit_gains[multiplier] = candidate_gains
        refit_rows.append(
            {
                **row,
                "selection_sse_refit_gains": selection_sse,
                "gain_fit": {
                    key: value
                    for key, value in gain_fit.items()
                    if key != "selected_gains_tensor"
                },
                "refit_seconds": seconds,
            }
        )
    refit_rows.sort(key=lambda row: float(row["selection_sse_refit_gains"]))
    selected = refit_rows[0]
    selected_multiplier = int(selected["multiplier"])
    selected_codebook = finalist_codebooks[selected_multiplier]
    selected_gains = refit_gains[selected_multiplier]

    codebooks = {
        "procedural_mcg_fp16": (mcg, torch.tensor(
            [float(row["gain"]) for row in oracle["gain_fit"]["selected_gains"]],
            dtype=corpus.training.dtype,
            device=device,
        )),
        "projected_mul1_e4m3": (projected_mul1, gains),
        "learned_e4m3": (learned_e4m3, torch.tensor(
            [float(row["gain"]) for row in oracle["gain_fit"]["selected_gains"]],
            dtype=corpus.training.dtype,
            device=device,
        )),
        "selected_mul1_affine_e4m3": (selected_codebook, selected_gains),
    }
    metrics = {}
    for name, (codebook, candidate_gains) in codebooks.items():
        encoded, _, seconds = _score(
            corpus.evaluation,
            corpus.evaluation_group,
            gains=candidate_gains,
            codebook=codebook,
            bits=args.bits,
            window_bits=args.window_bits,
            max_trace_mib=args.max_trace_mib,
            return_transitions=False,
        )
        metrics[name] = {
            **_paired_metrics(
                corpus.evaluation,
                encoded.reconstructed,
                corpus.evaluation_group,
                corpus.groups,
            ),
            "seconds": seconds,
        }

    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_mul1_multiplier_search",
        "schema_version": 1,
        "signature": {
            **oracle["signature"],
            "candidate_count": args.candidates,
            "mutation_rounds": args.mutation_rounds,
            "parents": args.parents,
            "finalists": args.finalists,
            "gain_refits": args.gain_refits,
            "ridge": args.ridge,
            "decoder": "E4M3(slope * bytesum(uint32(state * multiplier)) + intercept)",
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": int(corpus.evaluation.numel()),
        },
        "proxy_rounds": rounds,
        "finalists": finalist_rows,
        "gain_refits": refit_rows,
        "selected": selected,
        "metrics": metrics,
        "comparisons_to_procedural_mcg": {
            name: _comparison(metrics["procedural_mcg_fp16"], result)
            for name, result in metrics.items()
            if name != "procedural_mcg_fp16"
        },
        "comparisons_to_projected_mul1": {
            name: _comparison(metrics["projected_mul1_e4m3"], result)
            for name, result in metrics.items()
            if name != "projected_mul1_e4m3"
        },
        "storage": {
            "global_bytes": 8,
            "fields": "uint32 multiplier plus fp16 slope and fp16 intercept",
            "transition_lookup_bytes": 0,
        },
    }
    _save_tensors(
        args.tensors,
        {
            "projected_mul1_e4m3": projected_mul1,
            "selected_mul1_affine_e4m3": selected_codebook,
            "selected_gains": selected_gains,
        },
    )
    _atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=_parse_ints, default=(1, 24, 40))
    parser.add_argument(
        "--experts", type=_parse_ints, default=(0, 179, 358, 537, 716, 895)
    )
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--window-bits", type=int, default=16)
    parser.add_argument("--gain-tiles", type=int, default=8)
    parser.add_argument("--fitting-tiles", type=int, default=256)
    parser.add_argument("--selection-tiles", type=int, default=64)
    parser.add_argument("--evaluation-tiles", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=8192)
    parser.add_argument("--mutation-rounds", type=int, default=2)
    parser.add_argument("--parents", type=int, default=32)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--gain-refits", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--oracle-json",
        type=Path,
        default=Path("out/trellis-learned-codebooks-l1-l24-l40-shared-l16-v2.json"),
    )
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
        default=Path("out/trellis-e4m3-mul1-search-v1.json"),
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=Path("out/trellis-e4m3-mul1-search-v1.safetensors"),
    )
    args = parser.parse_args()
    if args.ridge <= 0.0:
        parser.error("--ridge must be positive")
    payload = run(args)
    selected = payload["selected"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tensors": str(args.tensors),
                "selected": {
                    key: selected[key]
                    for key in (
                        "multiplier_hex",
                        "slope",
                        "intercept",
                        "proxy_objective",
                        "selection_sse_fixed_gains",
                        "selection_sse_refit_gains",
                    )
                },
                "comparisons_to_procedural_mcg": payload[
                    "comparisons_to_procedural_mcg"
                ],
                "storage": payload["storage"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
