#!/usr/bin/env python3
"""Measure K32 reconstruction-scale freedom on a learned E4M3 trellis table.

The learned table and matrix gains come from the disjoint-fit experiment in
``experiment_trellis_learned_codebooks.py``.  This follow-up holds that table
fixed and gives the encoder successively richer per-32 scale families.  Scale
selection is part of the block encoder, so it is fitted directly on each blind
evaluation block and every metadata bit is charged explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from safetensors.torch import load_file
import torch

from kquant import constants as C
from kquant.source_weights import OfficialMXFP4Store
from kquant.trellis_window import TrellisEncoding, encode_trellis
from scripts.experiment_trellis_window_fp8 import (
    _atomic_json,
    _comparison,
    _paired_metrics,
    _parse_ints,
    collect_corpus,
)


SCALE_BLOCK = 32


def _matrix_gains(
    experiment: dict[str, object], *, device: torch.device
) -> torch.Tensor:
    rows = experiment["gain_fit"]["selected_gains"]
    groups = experiment["corpus"]["groups"]
    if len(rows) != len(groups):
        raise ValueError("gain rows do not match experiment groups")
    for expected, (gain, group) in enumerate(zip(rows, groups, strict=True)):
        if int(gain["group"]) != expected or int(group["group"]) != expected:
            raise ValueError("experiment groups are not in canonical order")
        for key in ("layer", "expert", "matrix"):
            if gain[key] != group[key]:
                raise ValueError(f"gain/group mismatch for {key}")
    return torch.tensor(
        [float(row["gain"]) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _score(
    source: torch.Tensor,
    codebook: torch.Tensor,
    scale: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    max_trace_mib: int,
) -> tuple[TrellisEncoding, float]:
    torch.cuda.synchronize(source.device)
    started = time.perf_counter()
    encoded = encode_trellis(
        source,
        bits,
        window_bits,
        codebook,
        reconstruction_scale=scale,
        max_trace_bytes=max_trace_mib << 20,
        return_transitions=True,
    )
    torch.cuda.synchronize(source.device)
    return encoded, time.perf_counter() - started


def _fit_scales(
    source: torch.Tensor,
    transitions: torch.Tensor,
    codebook: torch.Tensor,
    base_per_tile: torch.Tensor,
    *,
    exponent_min: int | None,
    exponent_max: int | None,
    exponent_step: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.shape[1] % SCALE_BLOCK:
        raise ValueError("source length must be divisible by the scale block")
    blocks = source.shape[1] // SCALE_BLOCK
    target = source.float().reshape(source.shape[0], blocks, SCALE_BLOCK)
    labels = codebook.index_select(0, transitions.long().flatten()).reshape_as(target)
    xq = (target * labels).sum(dim=2)
    qq = labels.square().sum(dim=2)
    base = base_per_tile.float().reshape(-1, 1)
    continuous = torch.where(qq > 0.0, xq / qq, base).clamp_min_(2.0**-24)

    if exponent_min is None or exponent_max is None:
        selected = continuous
        parameter = torch.log2(selected / base)
    else:
        exponent_codes = torch.arange(
            exponent_min,
            exponent_max + 1,
            dtype=torch.float32,
            device=source.device,
        )
        exponents = exponent_codes * exponent_step
        candidates = base.unsqueeze(2) * torch.exp2(exponents).reshape(1, 1, -1)
        xx = target.square().sum(dim=2).unsqueeze(2)
        losses = (
            xx
            - 2.0 * xq.unsqueeze(2) * candidates
            + qq.unsqueeze(2) * candidates.square()
        )
        choice = losses.argmin(dim=2)
        selected = candidates.expand(source.shape[0], blocks, -1).gather(
            2, choice.unsqueeze(2)
        ).squeeze(2)
        parameter = exponents.index_select(0, choice.flatten()).reshape_as(choice)
    return (
        selected.unsqueeze(2).expand(-1, -1, SCALE_BLOCK).reshape_as(source),
        parameter,
    )


def _run_scale_family(
    source: torch.Tensor,
    codebook: torch.Tensor,
    base_per_tile: torch.Tensor,
    *,
    bits: int,
    window_bits: int,
    exponent_min: int | None,
    exponent_max: int | None,
    exponent_step: float,
    iterations: int,
    max_trace_mib: int,
) -> tuple[TrellisEncoding, dict[str, object], torch.Tensor]:
    scale = base_per_tile.unsqueeze(1).expand_as(source).contiguous()
    parameter = torch.zeros(
        (source.shape[0], source.shape[1] // SCALE_BLOCK),
        dtype=torch.float32,
        device=source.device,
    )
    history = []
    previous_sse = float("inf")
    encoded: TrellisEncoding | None = None
    for iteration in range(iterations + 1):
        encoded, seconds = _score(
            source,
            codebook,
            scale,
            bits=bits,
            window_bits=window_bits,
            max_trace_mib=max_trace_mib,
        )
        sse = float(encoded.squared_error.double().sum().item())
        if sse > previous_sse * (1.0 + 1e-7):
            raise RuntimeError("alternating scale update increased the objective")
        history.append({"iteration": iteration, "sse": sse, "seconds": seconds})
        print(f"scale iteration={iteration} sse={sse:.8g}", flush=True)
        previous_sse = sse
        if iteration == iterations:
            break
        assert encoded.transition_indices is not None
        proposed_scale, proposed_parameter = _fit_scales(
            source,
            encoded.transition_indices,
            codebook,
            base_per_tile,
            exponent_min=exponent_min,
            exponent_max=exponent_max,
            exponent_step=exponent_step,
        )
        parameter = proposed_parameter
        if torch.equal(proposed_scale, scale):
            history[-1]["converged"] = True
            break
        scale = proposed_scale
    assert encoded is not None
    return encoded, {"history": history}, parameter


def _parameter_summary(parameter: torch.Tensor) -> dict[str, object]:
    values = parameter.float().flatten()
    quantiles = torch.quantile(
        values, torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], device=values.device)
    )
    return {
        "minimum_p01_p10_p50_p90_p99_maximum": [float(value) for value in quantiles],
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
        "nonzero_fraction": float((values != 0.0).float().mean().item()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the scale experiment requires CUDA")
    experiment = json.loads(args.experiment.read_text())
    if experiment.get("kind") != "kquant_trellis_learned_codebook_experiment":
        raise ValueError("input is not a learned-codebook experiment")
    signature = experiment["signature"]
    layers = tuple(int(value) for value in signature["layers"])
    experts = tuple(int(value) for value in signature["experts"])
    if args.layers is not None and tuple(args.layers) != layers:
        raise ValueError("--layers must exactly match the learned-codebook experiment")
    if args.experts is not None and tuple(args.experts) != experts:
        raise ValueError("--experts must exactly match the learned-codebook experiment")
    bits = int(signature["bits"])
    window_bits = int(signature["window_bits"])
    if bits != 3 or window_bits != 16:
        raise ValueError("the current study is fixed to L16/K3")

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    tables = load_file(str(args.codebooks), device=str(device))
    learned = tables["learned_e4m3"].float()
    procedural = tables["procedural_mcg_fp16"].float()
    gains = _matrix_gains(experiment, device=device)

    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    training_tiles = (
        int(signature["gain_tiles_per_matrix"])
        + int(signature["fitting_tiles_per_matrix"])
        + int(signature["selection_tiles_per_matrix"])
    )
    corpus = collect_corpus(
        store,
        layers=layers,
        experts=experts,
        training_tiles_per_matrix=training_tiles,
        evaluation_tiles_per_matrix=int(signature["evaluation_tiles_per_matrix"]),
        seed=int(experiment["signature"].get("seed", args.seed)),
        device=device,
    )
    expected_groups = experiment["corpus"]["groups"]
    actual_identity = [
        (int(row["layer"]), int(row["expert"]), str(row["matrix"]))
        for row in corpus.groups
    ]
    expected_identity = [
        (int(row["layer"]), int(row["expert"]), str(row["matrix"]))
        for row in expected_groups
    ]
    if actual_identity != expected_identity:
        raise ValueError("reconstructed corpus group order does not match the experiment")
    base_per_tile = gains.index_select(0, corpus.evaluation_group.long()).reciprocal()

    families = {
        "learned_e4m3_factorized": (None, None, 1.0, 0, 0),
        "continuous_k32": (None, None, 1.0, args.iterations, 16),
        "power2_delta8_k32": (-64, 63, 1.0, args.iterations, 8),
        "power2_delta4_k32": (-8, 7, 1.0, args.iterations, 4),
        "power2_delta3_k32": (-4, 3, 1.0, args.iterations, 3),
        "fractional_log3_k32": (-4, 3, 1.0 / 32.0, args.iterations, 3),
        "fractional_log4_wide_k32": (-8, 7, 1.0 / 32.0, args.iterations, 4),
        "fractional_log4_fine_k32": (-8, 7, 1.0 / 64.0, args.iterations, 4),
    }
    metrics: dict[str, object] = {}

    procedural_scale = base_per_tile.unsqueeze(1).expand_as(corpus.evaluation)
    procedural_encoded, procedural_seconds = _score(
        corpus.evaluation,
        procedural,
        procedural_scale,
        bits=bits,
        window_bits=window_bits,
        max_trace_mib=args.max_trace_mib,
    )
    metrics["procedural_mcg_factorized"] = {
        **_paired_metrics(
            corpus.evaluation,
            procedural_encoded.reconstructed,
            corpus.evaluation_group,
            corpus.groups,
        ),
        "seconds": procedural_seconds,
    }

    scale_fits = {}
    parameters = {}
    for name, (minimum, maximum, step, iterations, metadata_bits) in families.items():
        print(f"running {name}", flush=True)
        encoded, fit, parameter = _run_scale_family(
            corpus.evaluation,
            learned,
            base_per_tile,
            bits=bits,
            window_bits=window_bits,
            exponent_min=minimum,
            exponent_max=maximum,
            exponent_step=step,
            iterations=iterations,
            max_trace_mib=args.max_trace_mib,
        )
        metrics[name] = {
            **_paired_metrics(
                corpus.evaluation,
                encoded.reconstructed,
                corpus.evaluation_group,
                corpus.groups,
            ),
            "scale_parameter": _parameter_summary(parameter),
        }
        scale_fits[name] = fit
        parameters[name] = parameter

    baseline = metrics["learned_e4m3_factorized"]
    comparisons = {
        name: _comparison(baseline, metric)
        for name, metric in metrics.items()
        if name != "learned_e4m3_factorized"
    }
    coefficients = int(corpus.evaluation.numel())
    if coefficients % 32 or coefficients * bits % 8:
        raise ValueError("evaluation corpus does not close exact byte accounting")
    scale_blocks = coefficients // 32
    path_bytes = coefficients * bits // 8
    byte_accounting = {}
    for name, (_, _, _, _, metadata_bits) in families.items():
        scale_bits = scale_blocks * metadata_bits
        scale_bytes = math.ceil(scale_bits / 8)
        byte_accounting[name] = {
            "path_bytes": path_bytes,
            "scale_bytes": scale_bytes,
            "payload_bytes": path_bytes + scale_bytes,
            "payload_bpw": 8.0 * (path_bytes + scale_bytes) / coefficients,
            "global_e4m3_table_bytes": int(learned.numel()),
        }

    payload: dict[str, object] = {
        "kind": "kquant_trellis_e4m3_scale_experiment",
        "schema_version": 1,
        "signature": {
            "source_experiment": str(args.experiment.resolve()),
            "source_codebooks": str(args.codebooks.resolve()),
            "source_checkpoint": str(store.root),
            "layers": list(layers),
            "experts": list(experts),
            "bits": bits,
            "window_bits": window_bits,
            "scale_block": SCALE_BLOCK,
            "iterations": args.iterations,
            "note": (
                "Scale blocks are consecutive in the experimental 16x16 trellis order; "
                "a production MMA layout must prove the same grouping or remap it."
            ),
        },
        "corpus": {
            "groups": list(corpus.groups),
            "evaluation_coefficients": coefficients,
            "scale_blocks": scale_blocks,
        },
        "metrics": metrics,
        "scale_fits": scale_fits,
        "comparisons_to_learned_e4m3_factorized": comparisons,
        "byte_accounting": byte_accounting,
        "interpretation": {
            "continuous_k32": "FP16-scale quality oracle; 16 stored bits per 32 weights",
            "power2_delta8_k32": (
                "matrix base folded into existing EXL factors plus one signed power-of-two "
                "delta byte per 32 weights"
            ),
            "power2_delta4_k32": "same factorization with fixed packed four-bit deltas",
            "power2_delta3_k32": "same factorization with fixed packed three-bit deltas",
            "fractional_log3_k32": (
                "three-bit signed log2-scale index at 1/32-octave spacing"
            ),
            "fractional_log4_wide_k32": (
                "four-bit signed log2-scale index at 1/32-octave spacing"
            ),
            "fractional_log4_fine_k32": (
                "four-bit signed log2-scale index at 1/64-octave spacing"
            ),
        },
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("codebooks", type=Path)
    parser.add_argument("--layers", type=_parse_ints)
    parser.add_argument("--experts", type=_parse_ints)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--max-trace-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/trellis-e4m3-scales.json"),
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "comparisons": payload["comparisons_to_learned_e4m3_factorized"],
                "byte_accounting": payload["byte_accounting"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
