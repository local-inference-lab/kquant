#!/usr/bin/env python3
"""Close persisted mixed-EXL candidate bytes through the TP12 B12X decoder.

Run this script with the B12X/vLLM environment, for example::

    /home/luke/projects/b12x/.venv/bin/python \
      scripts/validate_mixed_exl3_b12x_tp12.py \
      --payload /path/to/mixed-exl3-layer-00040.safetensors \
      --metrics /path/to/mixed-exl3-layer-00040.metrics.safetensors \
      --selection /path/to/mixed-exl3-layer-00040.selection.json \
      --layer 40 --expert 285 --output out/persisted-pair-closure.json

The reference side decodes the exact stored payload with kquant's independent
PyTorch implementation.  The device side presents each of the twelve physical
rank pairs to B12X's public dense primitive, including its real input/output
Hadamard path.  At least one pair of every matrix/pair-kind combination is
also replayed under a CUDA graph and required to be bit-exact with eager mode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from kquant.exl3_reference import (
    CODEBOOK_MCG,
    CODEBOOK_MUL1_E4M3,
    EXL3_CODEBOOKS,
    MCG_MULT,
    MUL1_MULT,
)
from kquant.mixed_exl3 import (
    PAIRS_PER_EXPERT,
    PackedTP12Trellis,
    TP12TrellisDescriptor,
    decode_tp12_exl3_weight,
    resolve_mode,
    tp12_pair_kinds,
)
from kquant.pack.mixed_allocation import validate_selection_ledger_evidence


MATRIX_GEOMETRY = {
    "w1": ("n", 224, 192),
    "w3": ("n", 224, 192),
    "w2": ("k", 192, 224),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _selected_modes(
    ledger: dict[str, Any], *, layer: int, expert: int
) -> tuple[int, int]:
    if ledger.get("kind") != "kquant_tp12_mixed_exl3_candidate_pool":
        raise ValueError("selection ledger is not a TP12 mixed-EXL candidate pool")
    if int(ledger.get("layer", -1)) != layer:
        raise ValueError(
            f"selection ledger layer {ledger.get('layer')!r} does not match {layer}"
        )
    selections = ledger.get("selections")
    if not isinstance(selections, dict) or str(expert) not in selections:
        raise ValueError(f"selection ledger does not contain expert {expert}")
    evidence = selections[str(expert)]
    if not isinstance(evidence, dict) or not isinstance(
        evidence.get("selection"), dict
    ):
        raise ValueError(f"selection evidence for expert {expert} is malformed")
    selection = evidence["selection"]
    selected_r13 = selection.get("selected_r13")
    selected_r2 = selection.get("selected_r2")
    # Read-only compatibility with the historical shared-R pilot ledgers.
    if selected_r13 is None and selected_r2 is None:
        selected_r13 = selected_r2 = selection.get("selected_r")
    for name, selected in (("r13", selected_r13), ("r2", selected_r2)):
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise ValueError(
                f"selected {name} mode for expert {expert} is malformed"
            )
    return (
        resolve_mode(selected_r13).mode_id,
        resolve_mode(selected_r2).mode_id,
    )


def _tensor_prefix(layer: int, expert: int, matrix: str) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts."
        f"{expert}.{matrix}.mixed_exl3_"
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    temporary.replace(path)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    relative_error = difference.norm() / expected_f32.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    return {
        "relative_error": float(relative_error),
        "cosine": float(cosine),
        "max_abs_error": float(difference.abs().max()),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sparkinfer.gemm import trellis_linear
    except ImportError as exc:
        raise RuntimeError(
            "sparkinfer is unavailable; run with the B12X/vLLM Python "
            "environment and put /home/luke/projects/b12x on PYTHONPATH"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B12X closure")
    device = torch.device(args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"TP12 B12X closure requires SM120/SM121, got SM{major}{minor}")

    ledger = _read_json(args.selection)
    if ledger.get("metrics") != args.metrics.name:
        raise ValueError("selection ledger does not identify the supplied metrics")
    metrics = load_file(str(args.metrics), device="cpu")
    try:
        mode_ids = tuple(int(mode) for mode in metrics["mode_ids"].tolist())
        expert_ids = tuple(int(expert) for expert in metrics["expert_ids"].tolist())
    except KeyError as exc:
        raise ValueError("candidate metrics omit mode/expert IDs") from exc
    logical_trellis_schema = validate_selection_ledger_evidence(
        ledger,
        metrics,
        mode_ids=mode_ids,
        expert_ids=expert_ids,
    )
    codebook = ledger.get("codebook", CODEBOOK_MCG)
    if codebook not in EXL3_CODEBOOKS:
        raise ValueError(f"selection ledger has unsupported codebook {codebook!r}")
    selected_r13, selected_r2 = _selected_modes(
        ledger, layer=args.layer, expert=args.expert
    )
    required_names = {
        _tensor_prefix(args.layer, args.expert, matrix) + part
        for matrix in MATRIX_GEOMETRY
        for part in ("trellis", "suh", "svh")
    }
    with safe_open(args.payload, framework="pt", device="cpu") as reader:
        missing = sorted(required_names.difference(reader.keys()))
        if missing:
            raise ValueError(f"payload is missing tensors: {missing}")
        tensors = {
            name: reader.get_tensor(name).contiguous()
            for name in required_names
        }
    marker = torch.tensor(
        MCG_MULT if codebook == CODEBOOK_MCG else MUL1_MULT,
        dtype=torch.uint32,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    result: dict[str, Any] = {
        "kind": "kquant_persisted_mixed_exl3_b12x_tp12_closure",
        "payload": str(args.payload.resolve()),
        "metrics": str(args.metrics.resolve()),
        "selection": str(args.selection.resolve()),
        "layer": args.layer,
        "expert": args.expert,
        "selected_r13": selected_r13,
        "selected_r2": selected_r2,
        "logical_trellis_schema": logical_trellis_schema,
        "tp_size": 12,
        "rows": args.rows,
        "device": torch.cuda.get_device_name(device),
        "capability": [major, minor],
        "thresholds": {
            "max_relative_error": args.max_relative_error,
            "min_cosine": args.min_cosine,
        },
        "matrices": {},
    }
    all_relative_errors: list[float] = []
    all_cosines: list[float] = []
    graph_cases: set[tuple[str, str]] = set()

    for matrix, (rate_axis, k_tiles, n_tiles) in MATRIX_GEOMETRY.items():
        mode_id = selected_r2 if matrix == "w2" else selected_r13
        mode = resolve_mode(mode_id)
        pair_kinds = tp12_pair_kinds(mode)
        prefix = _tensor_prefix(args.layer, args.expert, matrix)
        payload = tensors[prefix + "trellis"].contiguous()
        suh = tensors[prefix + "suh"].contiguous()
        svh = tensors[prefix + "svh"].contiguous()
        descriptor = TP12TrellisDescriptor(
            mode_id=mode_id,
            rate_axis=rate_axis,
            k_tiles=k_tiles,
            n_tiles=n_tiles,
            schema=logical_trellis_schema,
        )
        packed = PackedTP12Trellis(descriptor, payload)
        packed.validate()
        reference = decode_tp12_exl3_weight(
            packed, suh, svh, codebook=codebook
        )
        payload_pairs = payload.reshape(
            PAIRS_PER_EXPERT, descriptor.words_per_pair
        )
        rank_results: list[dict[str, Any]] = []

        for rank, pair_kind in enumerate(pair_kinds):
            begin = rank * 256
            end = begin + 256
            if rate_axis == "n":
                reference_pair = reference[:, begin:end].to(device)
                pair_suh = suh.to(device)
                pair_svh = svh[begin:end].to(device)
            else:
                reference_pair = reference[begin:end, :].to(device)
                pair_suh = suh[begin:end].to(device)
                pair_svh = svh.to(device)
            pair_payload = payload_pairs[rank].to(device)
            prepared = trellis_linear.prepare_pair_weight(
                pair_payload,
                pair_suh,
                pair_svh,
                pair_kind=pair_kind,
                rate_axis=rate_axis,
                mcg=marker if codebook == CODEBOOK_MCG else None,
                mul1_e4m3=(
                    marker if codebook == CODEBOOK_MUL1_E4M3 else None
                ),
                codebook=codebook,
                params_dtype=torch.float16,
            )
            if prepared.trellis.numel() * prepared.trellis.element_size() != (
                pair_payload.numel() * pair_payload.element_size()
            ):
                raise ValueError("B12X preparation changed the compact payload byte count")

            source = (
                torch.randn(
                    (args.rows, reference_pair.shape[0]),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                * args.input_scale
            ).to(torch.float16)
            output = torch.empty(
                (args.rows, reference_pair.shape[1]),
                dtype=torch.float16,
                device=device,
            )
            gemm_output = torch.empty_like(output)
            rotated_f16 = torch.empty_like(source)
            c_tmp = torch.empty((1 << 20,), dtype=torch.float32, device=device)
            actual = trellis_linear.run(
                source,
                prepared,
                output=output,
                gemm_output=gemm_output,
                rotated_f16=rotated_f16,
                c_tmp=c_tmp,
            ).clone()
            expected = (source.float() @ reference_pair.float()).to(torch.float16)
            torch.cuda.synchronize(device)
            rank_metrics = _metrics(actual, expected)
            all_relative_errors.append(rank_metrics["relative_error"])
            all_cosines.append(rank_metrics["cosine"])

            graph_exact: bool | None = None
            graph_key = (matrix, pair_kind)
            if graph_key not in graph_cases:
                graph_cases.add(graph_key)
                eager = trellis_linear.run(
                    source,
                    prepared,
                    output=output,
                    gemm_output=gemm_output,
                    rotated_f16=rotated_f16,
                    c_tmp=c_tmp,
                ).clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = trellis_linear.run(
                        source,
                        prepared,
                        output=output,
                        gemm_output=gemm_output,
                        rotated_f16=rotated_f16,
                        c_tmp=c_tmp,
                    )
                graph.replay()
                torch.cuda.synchronize(device)
                graph_exact = bool(torch.equal(captured, eager))
                if not graph_exact:
                    raise ValueError(
                        f"CUDA graph output differs from eager for {matrix} {pair_kind}"
                    )

            rank_results.append(
                {
                    "rank": rank,
                    "pair_kind": pair_kind,
                    "payload_bytes": pair_payload.numel()
                    * pair_payload.element_size(),
                    "graph_exact": graph_exact,
                    **rank_metrics,
                }
            )

        matrix_max_relative = max(
            item["relative_error"] for item in rank_results
        )
        matrix_min_cosine = min(item["cosine"] for item in rank_results)
        result["matrices"][matrix] = {
            "mode_id": mode_id,
            "mode": mode.name,
            "rate_axis": rate_axis,
            "pair_kinds": list(pair_kinds),
            "payload_bytes": payload.numel() * payload.element_size(),
            "max_relative_error": matrix_max_relative,
            "min_cosine": matrix_min_cosine,
            "ranks": rank_results,
        }

    result["max_relative_error"] = max(all_relative_errors)
    result["min_cosine"] = min(all_cosines)
    result["graph_cases"] = [
        {"matrix": matrix, "pair_kind": pair_kind}
        for matrix, pair_kind in sorted(graph_cases)
    ]
    result["passed"] = bool(
        result["max_relative_error"] <= args.max_relative_error
        and result["min_cosine"] >= args.min_cosine
    )
    if not result["passed"]:
        raise ValueError(
            "persisted B12X closure failed: "
            f"max relative error {result['max_relative_error']:.6g}, "
            f"min cosine {result['min_cosine']:.6g}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--input-scale", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--max-relative-error", type=float, default=2.0e-2)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.input_scale <= 0:
        parser.error("--input-scale must be positive")
    if not 1 <= args.layer <= 92:
        parser.error("--layer must lie in 1..92")
    if not 0 <= args.expert < 896:
        parser.error("--expert must lie in 0..895")
    return args


def main() -> None:
    args = parse_args()
    result = validate(args)
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
