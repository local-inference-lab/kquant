#!/usr/bin/env python3
"""Compare shorter trellis windows and FP8-constrained reconstruction.

This is a tile-quantizer study, not a replacement checkpoint build.  It
streams official Kimi-K3 MXFP4 expert matrices directly to one GPU, applies
the same class of sign/RMS/Hadamard regularization used before EXL3 tile
quantization, and samples disjoint training and evaluation tiles.

For each requested L, a training split at ``--gain-fit-rate`` chooses the
matrix-local gain for the FP16 procedural alphabet and for an
E4M3-constrained version of that alphabet.
Evaluation then compares at configurable K2--K5 path rates:

* FP16 MCG with fresh Viterbi paths;
* the selected FP16 paths cast to E4M3 after the fact;
* E4M3 with fresh paths at the FP16 gain; and
* E4M3 with fresh paths at its own held-out gain.

The post-hoc control isolates the value-rounding penalty.  The difference
between it and fresh E4M3 paths measures how much route freedom recovers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from kquant import constants as C
from kquant.exl3_reference import (
    _blockwise_hadamard_left,
    _blockwise_hadamard_right,
    decode_mul1_states,
    tensor_core_permutation,
)
from kquant.source_weights import OfficialMXFP4Store
from kquant.trellis_window import (
    TrellisEncoding,
    encode_trellis,
    fp8_constrained_codebook,
    geometry,
    mcg_codebook,
)


CODEBOOK_SCALE = 1.24371088
TILE_SIDE = 16
TILE_VALUES = TILE_SIDE * TILE_SIDE


@dataclass(frozen=True)
class TileCorpus:
    training: torch.Tensor
    training_group: torch.Tensor
    evaluation: torch.Tensor
    evaluation_group: torch.Tensor
    groups: tuple[dict[str, object], ...]


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return result


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def _random_signs(length: int, *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.where(
        torch.rand(length, device=device, generator=generator) < 0.5,
        -torch.ones(length, device=device),
        torch.ones(length, device=device),
    )


def regularize_matrix(source: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Apply EXL-style fixed-sign, RMS, and block-Hadamard regularization.

    ``source`` uses checkpoint orientation [out, in].  The returned matrix is
    in EXL/GEMM orientation [K=in, N=out].  Output-channel scaling is disabled,
    matching the current routed-expert encoder.  Global gain is deliberately
    left for the held-out search below.
    """

    if source.ndim != 2 or source.dtype != torch.float32:
        raise ValueError("source must be a float32 matrix")
    weight = source.T.contiguous()
    k, n = weight.shape
    input_signs = _random_signs(k, device=weight.device, seed=seed)
    output_signs = _random_signs(n, device=weight.device, seed=seed + 1)
    weight /= output_signs.unsqueeze(0)
    weight = _blockwise_hadamard_right(weight)
    input_rms = weight.square().mean(dim=1).sqrt().clamp_min_(1e-10)
    input_scales = input_signs * input_rms / (-CODEBOOK_SCALE)
    weight /= input_scales.unsqueeze(1)
    return _blockwise_hadamard_left(weight)


def sample_disjoint_tiles(
    weight: torch.Tensor,
    *,
    training_count: int,
    evaluation_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k, n = weight.shape
    if k % TILE_SIDE or n % TILE_SIDE:
        raise ValueError(f"matrix shape {(k, n)} is not 16x16 tiled")
    k_tiles = k // TILE_SIDE
    n_tiles = n // TILE_SIDE
    total = k_tiles * n_tiles
    requested = training_count + evaluation_count
    if requested > total:
        raise ValueError(f"requested {requested} tiles from only {total}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randperm(total, generator=generator)[:requested].to(weight.device)
    tile_rows = torch.div(ids, n_tiles, rounding_mode="floor")
    tile_columns = ids.remainder(n_tiles)
    blocks = weight.reshape(k_tiles, TILE_SIDE, n_tiles, TILE_SIDE)
    sampled = blocks[tile_rows, :, tile_columns, :].reshape(requested, TILE_VALUES)
    sampled = sampled.index_select(1, tensor_core_permutation(weight.device))
    return sampled[:training_count].contiguous(), sampled[training_count:].contiguous()


def collect_corpus(
    store: OfficialMXFP4Store,
    *,
    layers: tuple[int, ...],
    experts: tuple[int, ...],
    training_tiles_per_matrix: int,
    evaluation_tiles_per_matrix: int,
    seed: int,
    device: torch.device,
) -> TileCorpus:
    training = []
    training_group = []
    evaluation = []
    evaluation_group = []
    groups: list[dict[str, object]] = []
    for layer in layers:
        for expert in experts:
            for matrix_index, matrix in enumerate(C.EXPERT_MATRICES):
                group_id = len(groups)
                local_seed = seed + layer * 1_000_003 + expert * 1009 + matrix_index * 17
                started = time.perf_counter()
                source = store.load_matrix(layer, expert, matrix, device=device)
                transformed = regularize_matrix(source, seed=local_seed)
                train, evaluate = sample_disjoint_tiles(
                    transformed,
                    training_count=training_tiles_per_matrix,
                    evaluation_count=evaluation_tiles_per_matrix,
                    seed=local_seed + 2,
                )
                groups.append(
                    {
                        "group": group_id,
                        "layer": layer,
                        "expert": expert,
                        "matrix": matrix,
                        "training_tiles": int(train.shape[0]),
                        "evaluation_tiles": int(evaluate.shape[0]),
                        "evaluation_energy": float(evaluate.double().square().sum().item()),
                        "load_transform_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    f"sampled L{layer} E{expert} {matrix}: "
                    f"{train.shape[0]}+{evaluate.shape[0]} tiles",
                    flush=True,
                )
                training.append(train)
                training_group.append(
                    torch.full(
                        (train.shape[0],),
                        group_id,
                        dtype=torch.int16,
                        device=device,
                    )
                )
                evaluation.append(evaluate)
                evaluation_group.append(
                    torch.full(
                        (evaluate.shape[0],),
                        group_id,
                        dtype=torch.int16,
                        device=device,
                    )
                )
                del source, transformed
    return TileCorpus(
        training=torch.cat(training),
        training_group=torch.cat(training_group),
        evaluation=torch.cat(evaluation),
        evaluation_group=torch.cat(evaluation_group),
        groups=tuple(groups),
    )


def _encode(
    source: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    codebook: torch.Tensor,
    gain: float | torch.Tensor,
    max_trace_mib: int,
    return_transitions: bool,
) -> tuple[TrellisEncoding, float]:
    torch.cuda.synchronize(source.device)
    started = time.perf_counter()
    gain_tensor = torch.as_tensor(gain, dtype=source.dtype, device=source.device)
    if gain_tensor.ndim == 0:
        multiplier = gain_tensor
        error_divisor = gain_tensor.square()
    elif gain_tensor.ndim == 1 and gain_tensor.shape[0] == source.shape[0]:
        multiplier = gain_tensor.unsqueeze(1)
        error_divisor = gain_tensor.square()
    else:
        raise ValueError("gain must be scalar or have one value per source tile")
    encoded = encode_trellis(
        source * multiplier,
        bits,
        window_bits,
        codebook,
        max_trace_bytes=max_trace_mib << 20,
        return_transitions=return_transitions,
    )
    torch.cuda.synchronize(source.device)
    elapsed = time.perf_counter() - started
    return TrellisEncoding(
        reconstructed=encoded.reconstructed / multiplier,
        squared_error=encoded.squared_error / error_divisor,
        transition_indices=encoded.transition_indices,
    ), elapsed


def _evaluate_group_gain_grid(
    training: torch.Tensor,
    training_group: torch.Tensor,
    *,
    window_bits: int,
    gain_fit_rate: int,
    codebook: torch.Tensor,
    gains_by_group: tuple[tuple[float, ...], ...],
    subsample_stride: int,
    max_trace_mib: int,
) -> tuple[list[list[dict[str, float]]], float]:
    expanded_tiles = []
    expanded_gains = []
    expanded_slots = []
    candidates_per_group = len(gains_by_group[0])
    if any(len(gains) != candidates_per_group for gains in gains_by_group):
        raise ValueError("every group must have the same number of gain candidates")
    for group, gains in enumerate(gains_by_group):
        local = training[training_group == group][::subsample_stride]
        if local.shape[0] == 0:
            raise ValueError(f"training group {group} has no gain-fit tiles")
        for candidate, gain in enumerate(gains):
            expanded_tiles.append(local)
            expanded_gains.append(
                torch.full(
                    (local.shape[0],), gain, dtype=training.dtype, device=training.device
                )
            )
            expanded_slots.append(
                torch.full(
                    (local.shape[0],),
                    group * candidates_per_group + candidate,
                    dtype=torch.long,
                    device=training.device,
                )
            )
    tiles = torch.cat(expanded_tiles)
    gains = torch.cat(expanded_gains)
    slots = torch.cat(expanded_slots)
    encoded, elapsed = _encode(
        tiles,
        bits=gain_fit_rate,
        window_bits=window_bits,
        codebook=codebook,
        gain=gains,
        max_trace_mib=max_trace_mib,
        return_transitions=False,
    )
    totals = torch.zeros(
        len(gains_by_group) * candidates_per_group,
        dtype=torch.float64,
        device=training.device,
    )
    totals.scatter_add_(0, slots, encoded.squared_error.double())
    totals = totals.reshape(len(gains_by_group), candidates_per_group).cpu()
    rows = [
        [
            {"gain": gain, "sse": float(totals[group, candidate].item())}
            for candidate, gain in enumerate(gains_by_group[group])
        ]
        for group in range(len(gains_by_group))
    ]
    return rows, elapsed


def fit_group_gains(
    training: torch.Tensor,
    training_group: torch.Tensor,
    *,
    groups: tuple[dict[str, object], ...],
    window_bits: int,
    gain_fit_rate: int,
    codebook: torch.Tensor,
    max_trace_mib: int,
) -> dict[str, object]:
    coarse_gains = tuple(0.4 + 0.2 * index for index in range(8))
    coarse, coarse_seconds = _evaluate_group_gain_grid(
        training,
        training_group,
        window_bits=window_bits,
        gain_fit_rate=gain_fit_rate,
        codebook=codebook,
        gains_by_group=tuple(coarse_gains for _ in groups),
        subsample_stride=2,
        max_trace_mib=max_trace_mib,
    )
    centers = tuple(
        float(min(rows, key=lambda item: item["sse"])["gain"]) for rows in coarse
    )
    fine_gains = tuple(
        tuple(max(0.05, center + 0.05 * offset) for offset in (-2, -1, 0, 1, 2))
        for center in centers
    )
    fine, fine_seconds = _evaluate_group_gain_grid(
        training,
        training_group,
        window_bits=window_bits,
        gain_fit_rate=gain_fit_rate,
        codebook=codebook,
        gains_by_group=fine_gains,
        subsample_stride=1,
        max_trace_mib=max_trace_mib,
    )
    selected = tuple(min(rows, key=lambda item: item["sse"]) for rows in fine)
    selected_gains = torch.tensor(
        [float(row["gain"]) for row in selected],
        dtype=training.dtype,
        device=training.device,
    )
    return {
        "selected_gains_tensor": selected_gains,
        "selected_gains": [
            {
                "group": int(groups[group]["group"]),
                "layer": int(groups[group]["layer"]),
                "expert": int(groups[group]["expert"]),
                "matrix": str(groups[group]["matrix"]),
                "gain": float(row["gain"]),
                "training_sse": float(row["sse"]),
            }
            for group, row in enumerate(selected)
        ],
        "gain_summary": {
            "minimum": float(selected_gains.min().item()),
            "maximum": float(selected_gains.max().item()),
            "mean": float(selected_gains.mean().item()),
            "std": float(selected_gains.std().item()),
        },
        "coarse_seconds": coarse_seconds,
        "fine_seconds": fine_seconds,
        "coarse": coarse,
        "fine": fine,
    }


def _paired_metrics(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    group_ids: torch.Tensor,
    groups: tuple[dict[str, object], ...],
) -> dict[str, object]:
    tile_sse = (source.double() - reconstruction.double()).square().sum(dim=1)
    tile_energy = source.double().square().sum(dim=1)
    sse = float(tile_sse.sum().item())
    energy = float(tile_energy.sum().item())
    group_rows = []
    for group in groups:
        group_id = int(group["group"])
        mask = group_ids == group_id
        local_sse = float(tile_sse[mask].sum().item())
        local_energy = float(tile_energy[mask].sum().item())
        group_rows.append(
            {
                "group": group_id,
                "layer": group["layer"],
                "expert": group["expert"],
                "matrix": group["matrix"],
                "sse": local_sse,
                "nmse": local_sse / local_energy,
            }
        )
    return {
        "sse": sse,
        "energy": energy,
        "nmse": sse / energy,
        "tile_median_sse": float(tile_sse.median().item()),
        "tile_p90_sse": float(torch.quantile(tile_sse, 0.90).item()),
        "groups": group_rows,
    }


def _comparison(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, float | int]:
    baseline_groups = {int(row["group"]): row for row in baseline["groups"]}
    candidate_groups = {int(row["group"]): row for row in candidate["groups"]}
    wins = sum(
        float(candidate_groups[group]["sse"]) < float(row["sse"])
        for group, row in baseline_groups.items()
    )
    return {
        "aggregate_sse_ratio": float(candidate["sse"]) / float(baseline["sse"]),
        "aggregate_nmse_delta": float(candidate["nmse"]) - float(baseline["nmse"]),
        "matrix_groups_better": wins,
        "matrix_groups_total": len(baseline_groups),
    }


def _codebook_stats(fp16: torch.Tensor, fp8: torch.Tensor) -> dict[str, object]:
    difference = fp16.double() - fp8.double()
    return {
        "entries": fp16.numel(),
        "fp16_unique_values": int(torch.unique(fp16).numel()),
        "fp8_unique_values": int(torch.unique(fp8).numel()),
        "fp16_min": float(fp16.min().item()),
        "fp16_max": float(fp16.max().item()),
        "fp16_std": float(fp16.float().std().item()),
        "alphabet_projection_mse": float(difference.square().mean().item()),
        "alphabet_projection_max_abs": float(difference.abs().max().item()),
        "unchanged_entry_fraction": float((fp16 == fp8).float().mean().item()),
    }


def _native_l16_validation(
    source: torch.Tensor,
    *,
    evaluation_group: torch.Tensor,
    gains: dict[str, torch.Tensor],
    reference: dict[str, dict[str, object]],
    exllamav3_root: Path,
    rates: tuple[int, ...],
) -> dict[str, object]:
    from scripts.experiment_codec_transfers import _load_exl3_quantizer

    quantize_exl3 = _load_exl3_quantizer(exllamav3_root)
    module = __import__(quantize_exl3.__module__, fromlist=["*"])
    sample = source[: min(16, source.shape[0])]
    output: dict[str, object] = {}
    for bits in rates:
        if bits == 5:
            continue
        gain = gains["fp16_mcg"].index_select(
            0, evaluation_group[: sample.shape[0]].long()
        )
        torch.cuda.synchronize(sample.device)
        started = time.perf_counter()
        native, _ = module.quantize_tiles(
            (sample * gain.unsqueeze(1)).contiguous(), {"K": bits, "mcg": True}
        )
        torch.cuda.synchronize(sample.device)
        native_seconds = time.perf_counter() - started
        native_reconstruction = native / gain.unsqueeze(1)
        native_sse = float(
            (native_reconstruction.double() - sample.double()).square().sum().item()
        )
        generic = reference[str(bits)]["fp16_mcg"]
        # Re-evaluate the same 16 tiles rather than compare against the full
        # corpus metric stored in ``reference``.
        generic_encoding, generic_seconds = _encode(
            sample,
            bits=bits,
            window_bits=16,
            codebook=mcg_codebook(16, device=sample.device),
            gain=gain,
            max_trace_mib=256,
            return_transitions=False,
        )
        generic_sse = float(generic_encoding.squared_error.double().sum().item())
        output[str(bits)] = {
            "tiles": sample.shape[0],
            "native_half_cost_sse": native_sse,
            "generic_float_cost_sse": generic_sse,
            "generic_to_native_sse": generic_sse / native_sse,
            "native_seconds": native_seconds,
            "generic_seconds": generic_seconds,
            "full_corpus_generic_nmse": generic["nmse"],
        }
    if 5 in rates:
        output["unsupported_rates"] = {
            "5": "the production EXL quantize_tiles control supports only K2--K4"
        }
    return output


def _procedural_codebooks(
    window_bits: int,
    *,
    codebook: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if codebook == "mcg":
        fp16 = mcg_codebook(window_bits, device=device)
    elif codebook == "mul1":
        states = torch.arange(1 << window_bits, dtype=torch.int64, device=device)
        fp16 = decode_mul1_states(states).float().contiguous()
    else:
        raise ValueError(f"unsupported procedural codebook: {codebook}")
    return fp16, fp8_constrained_codebook(fp16, dtype=torch.float8_e4m3fn)


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the window/FP8 experiment requires a CUDA device")
    if any(layer not in C.MOE_LAYERS for layer in args.layers):
        raise ValueError("all layers must be routed MoE layers")
    if any(expert < 0 or expert >= C.NUM_EXPERTS for expert in args.experts):
        raise ValueError("experts must be in 0..895")
    if any(rate not in (2, 3, 4, 5) for rate in args.rates):
        raise ValueError("rates must be drawn from K2, K3, K4, and K5")
    if args.gain_fit_rate not in (2, 3, 4, 5):
        raise ValueError("gain-fit-rate must be K2, K3, K4, or K5")
    if any(window_bits < 2 * max(args.rates) for window_bits in args.window_bits):
        raise ValueError("every window must contain at least two symbols at the largest K")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    corpus = collect_corpus(
        store,
        layers=args.layers,
        experts=args.experts,
        training_tiles_per_matrix=args.training_tiles_per_matrix,
        evaluation_tiles_per_matrix=args.evaluation_tiles_per_matrix,
        seed=args.seed,
        device=device,
    )
    payload: dict[str, object] = {
        "kind": "kquant_trellis_window_fp8_experiment",
        "schema_version": 1,
        "signature": {
            "source_checkpoint": str(store.root),
            "source_revision": store.revision,
            "layers": list(args.layers),
            "experts": list(args.experts),
            "matrices": list(C.EXPERT_MATRICES),
            "window_bits": list(args.window_bits),
            "rates": list(args.rates),
            "gain_fit_rate": args.gain_fit_rate,
            "procedural_codebook": args.codebook,
            "fp8_dtype": "float8_e4m3fn",
            "training_tiles_per_matrix": args.training_tiles_per_matrix,
            "evaluation_tiles_per_matrix": args.evaluation_tiles_per_matrix,
            "seed": args.seed,
            "regularization": (
                "official source -> [K,N], deterministic signs, no output scaling, "
                "input RMS / -1.24371088, 128x128 block Hadamards; matrix-local "
                f"gain fit on disjoint K{args.gain_fit_rate} training tiles"
            ),
            "tail_biting": "EXL two-pass roll-half boundary approximation",
            "cost_precision": "float32 reference dynamic program",
            "fp8_contract": (
                "MCG values projected to E4M3, then Viterbi rerun; own-gain result "
                "can fold inverse gain into existing outer scales"
            ),
        },
        "corpus": {
            "training_tiles": corpus.training.shape[0],
            "evaluation_tiles": corpus.evaluation.shape[0],
            "coefficients": corpus.evaluation.numel(),
            "evaluation_energy": float(
                corpus.evaluation.double().square().sum().item()
            ),
            "groups": list(corpus.groups),
        },
        "windows": {},
    }

    l16_reference: dict[str, dict[str, object]] | None = None
    l16_gains: dict[str, torch.Tensor] | None = None
    for window_bits in args.window_bits:
        print(f"fitting L{window_bits} alphabets", flush=True)
        fp16_codebook, fp8_codebook = _procedural_codebooks(
            window_bits,
            codebook=args.codebook,
            device=device,
        )
        fp16_fit = fit_group_gains(
            corpus.training,
            corpus.training_group,
            groups=corpus.groups,
            window_bits=window_bits,
            gain_fit_rate=args.gain_fit_rate,
            codebook=fp16_codebook,
            max_trace_mib=args.max_trace_mib,
        )
        fp8_fit = fit_group_gains(
            corpus.training,
            corpus.training_group,
            groups=corpus.groups,
            window_bits=window_bits,
            gain_fit_rate=args.gain_fit_rate,
            codebook=fp8_codebook,
            max_trace_mib=args.max_trace_mib,
        )
        group_gains = {
            "fp16_mcg": fp16_fit["selected_gains_tensor"],
            "fp8_e4m3": fp8_fit["selected_gains_tensor"],
        }
        evaluation_gains = {
            name: values.index_select(0, corpus.evaluation_group.long())
            for name, values in group_gains.items()
        }
        by_rate: dict[str, dict[str, object]] = {}
        for bits in args.rates:
            print(f"evaluating L{window_bits} K{bits}", flush=True)
            fp16_encoded, fp16_seconds = _encode(
                corpus.evaluation,
                bits=bits,
                window_bits=window_bits,
                codebook=fp16_codebook,
                gain=evaluation_gains["fp16_mcg"],
                max_trace_mib=args.max_trace_mib,
                return_transitions=True,
            )
            assert fp16_encoded.transition_indices is not None
            posthoc_reconstruction = (
                fp8_codebook[fp16_encoded.transition_indices.long()]
                / evaluation_gains["fp16_mcg"].unsqueeze(1)
            )
            fp8_same, fp8_same_seconds = _encode(
                corpus.evaluation,
                bits=bits,
                window_bits=window_bits,
                codebook=fp8_codebook,
                gain=evaluation_gains["fp16_mcg"],
                max_trace_mib=args.max_trace_mib,
                return_transitions=True,
            )
            fp8_own, fp8_own_seconds = _encode(
                corpus.evaluation,
                bits=bits,
                window_bits=window_bits,
                codebook=fp8_codebook,
                gain=evaluation_gains["fp8_e4m3"],
                max_trace_mib=args.max_trace_mib,
                return_transitions=True,
            )
            assert fp8_same.transition_indices is not None
            assert fp8_own.transition_indices is not None
            metrics = {
                "fp16_mcg": _paired_metrics(
                    corpus.evaluation,
                    fp16_encoded.reconstructed,
                    corpus.evaluation_group,
                    corpus.groups,
                ),
                "fp8_posthoc_fp16_path": _paired_metrics(
                    corpus.evaluation,
                    posthoc_reconstruction,
                    corpus.evaluation_group,
                    corpus.groups,
                ),
                "fp8_reroute_fp16_gain": _paired_metrics(
                    corpus.evaluation,
                    fp8_same.reconstructed,
                    corpus.evaluation_group,
                    corpus.groups,
                ),
                "fp8_reroute_own_gain": _paired_metrics(
                    corpus.evaluation,
                    fp8_own.reconstructed,
                    corpus.evaluation_group,
                    corpus.groups,
                ),
            }
            baseline_sse = float(metrics["fp16_mcg"]["sse"])
            posthoc_sse = float(metrics["fp8_posthoc_fp16_path"]["sse"])
            reroute_sse = float(metrics["fp8_reroute_fp16_gain"]["sse"])
            added = posthoc_sse - baseline_sse
            by_rate[str(bits)] = {
                **metrics,
                "comparisons_to_fp16": {
                    name: _comparison(metrics["fp16_mcg"], metrics[name])
                    for name in (
                        "fp8_posthoc_fp16_path",
                        "fp8_reroute_fp16_gain",
                        "fp8_reroute_own_gain",
                    )
                },
                "reroute_recovery_fraction_at_fixed_gain": (
                    (posthoc_sse - reroute_sse) / added if added > 0.0 else math.nan
                ),
                "transition_change_fraction": {
                    "fp8_same_gain_vs_fp16": float(
                        (
                            fp8_same.transition_indices
                            != fp16_encoded.transition_indices
                        ).float().mean().item()
                    ),
                    "fp8_own_gain_vs_fp16": float(
                        (
                            fp8_own.transition_indices
                            != fp16_encoded.transition_indices
                        ).float().mean().item()
                    ),
                },
                "reference_seconds": {
                    "fp16_mcg": fp16_seconds,
                    "fp8_reroute_fp16_gain": fp8_same_seconds,
                    "fp8_reroute_own_gain": fp8_own_seconds,
                },
            }
        window_payload = {
            "geometry": {
                str(bits): {
                    "branches": geometry(bits, window_bits).branches,
                    "states": geometry(bits, window_bits).states,
                    "transitions": geometry(bits, window_bits).transitions,
                    "relative_transition_work_vs_l16": 2.0 ** (window_bits - 16),
                }
                for bits in args.rates
            },
            "codebook": _codebook_stats(fp16_codebook, fp8_codebook),
            "gain_fit": {
                "fp16_mcg": {
                    key: value
                    for key, value in fp16_fit.items()
                    if key != "selected_gains_tensor"
                },
                "fp8_e4m3": {
                    key: value
                    for key, value in fp8_fit.items()
                    if key != "selected_gains_tensor"
                },
            },
            "gains": {
                "scope": "matrix-local",
                "fp16_mcg": fp16_fit["gain_summary"],
                "fp8_e4m3": fp8_fit["gain_summary"],
            },
            "by_rate": by_rate,
        }
        payload["windows"][str(window_bits)] = window_payload
        if window_bits == 16:
            l16_reference = by_rate
            l16_gains = group_gains
        _atomic_json(args.output, payload)

    if (
        args.validate_native_l16
        and args.codebook == "mcg"
        and l16_reference is not None
        and l16_gains is not None
    ):
        print("validating generic L16 against EXL's native half-cost kernel", flush=True)
        payload["native_l16_validation"] = _native_l16_validation(
            corpus.evaluation,
            evaluation_group=corpus.evaluation_group,
            gains=l16_gains,
            reference=l16_reference,
            exllamav3_root=args.exllamav3_root,
            rates=args.rates,
        )
    elif args.validate_native_l16 and args.codebook != "mcg":
        payload["native_l16_validation"] = {
            "status": "not-run",
            "reason": "the existing native comparison implements procedural MCG only",
        }
    _atomic_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=_parse_ints, default=(1, 24, 40))
    parser.add_argument("--experts", type=_parse_ints, default=(0, 448, 895))
    parser.add_argument("--window-bits", type=_parse_ints, default=(12, 14, 16))
    parser.add_argument("--rates", type=_parse_ints, default=(2, 3, 4))
    parser.add_argument("--gain-fit-rate", type=int, default=3)
    parser.add_argument("--codebook", choices=("mcg", "mul1"), default="mcg")
    parser.add_argument("--training-tiles-per-matrix", type=int, default=4)
    parser.add_argument("--evaluation-tiles-per-matrix", type=int, default=12)
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    parser.add_argument(
        "--validate-native-l16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-window-fp8-l12-l14-l16-v1.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    print(json.dumps({"output": str(args.output), "windows": list(payload["windows"])}, indent=2))


if __name__ == "__main__":
    main()
