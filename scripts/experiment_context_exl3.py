"""Test codec-style context-dependent bit allocation inside EXL3.

The expert's 3072 middle channels are ranked by captured activation energy in
four-channel groups. Equal-size contexts receive different EXL trellis bit
widths while the mean remains three bits per weight. Both independent-context
and full-Hessian mixed-rate encoders are tested. This is an offline simulation
of a format-changing, context-adaptive trellis codec; it has no production
decoder or kernel yet.

Only retained weights from the interim EXL3 hybrid and samples/Hessians built
from that interim teacher are used. The official checkpoint is never resolved
or loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.codec_research import normalized_mse
from kquant.interim_weights import InterimKeepStore
from kquant.mixed_exl3 import (
    RATE_TRANSFER_MODES,
    ModeSpec,
    importance_group_order,
    mode_from_context_bits,
    tp12_storage_group_order,
)
from kquant.pack.exl3 import MCG_MULT, SIGMA_REG, make_shared_h
from kquant.pack.mixed_exl3 import quantize_mixed_matrix
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_context_palettes import (
    VECTOR_DIMENSION,
    _activation_metrics,
    _contexts_for_expert,
    _load_anchor,
)


def _parse_allocations(value: str) -> tuple[tuple[int, ...], ...]:
    configurations = []
    for item in value.split(";"):
        bits = tuple(int(part) for part in item.split(","))
        if not bits or min(bits) < 2 or max(bits) > 8:
            raise argparse.ArgumentTypeError("EXL context widths must be 2..8 bits")
        configurations.append(bits)
    contexts = len(configurations[0])
    if any(len(bits) != contexts for bits in configurations):
        raise argparse.ArgumentTypeError(
            "all allocations must use the same number of contexts"
        )
    if any(sum(bits) != 3 * contexts for bits in configurations):
        raise argparse.ArgumentTypeError(
            "each allocation must average exactly three bits per weight"
        )
    return tuple(configurations)


def _parse_r_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("R values must be comma-separated integers") from exc
    if not values or any(not 0 <= r <= 12 for r in values):
        raise argparse.ArgumentTypeError("R values must be in 0..12")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("R values must not contain duplicates")
    return values


def _parse_layouts(value: str) -> tuple[str, ...]:
    layouts = tuple(part.strip() for part in value.split(",") if part.strip())
    allowed = {"importance_ordered", "tp12_balanced"}
    if not layouts or any(layout not in allowed for layout in layouts):
        raise argparse.ArgumentTypeError(
            "layouts must be importance_ordered and/or tp12_balanced"
        )
    return layouts


def _channel_indices(
    block_contexts: torch.Tensor, context: int, device: torch.device
) -> torch.Tensor:
    blocks = torch.nonzero(block_contexts == context).flatten()
    offsets = torch.arange(VECTOR_DIMENSION, device=blocks.device)
    channels = blocks[:, None] * VECTOR_DIMENSION + offsets[None]
    return channels.flatten().to(device)


def _quantize_partition(
    source: torch.Tensor,
    channels: torch.Tensor,
    *,
    bit_width: int,
    hessian: torch.Tensor | None,
    layer: int,
    context: int,
    device: torch.device,
    quantize_exl3,
) -> tuple[torch.Tensor, dict]:
    # EXL3 accepts [input, output]. Selecting columns from the ordinary w2
    # [output, input] matrix makes an independent, channel-permuted partition.
    weight = source.index_select(1, channels).T.contiguous()
    local_hessian = None
    if hessian is not None:
        cpu_channels = channels.cpu()
        local_hessian = hessian.index_select(0, cpu_channels).index_select(
            1, cpu_channels
        )
    shared_h = make_shared_h(weight.shape[0], device, local_hessian)
    quant_args = {
        "K": bit_width,
        "seed": layer * 1_000_000 + 100 * context + 2,
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "mcg": True,
    }
    quantized, proxy, tensors = quantize_exl3(
        weight,
        shared_h,
        quant_args,
        True,
        progress_str="",
    )
    stored_bits = sum(
        tensor.numel() * tensor.element_size() * 8 for tensor in tensors.values()
    )
    return quantized.T.contiguous(), {
        "context": context,
        "bit_width": bit_width,
        "channels": int(channels.numel()),
        "proxy": float(proxy),
        "stored_bytes": stored_bits // 8,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }


def _run_configuration(
    source: torch.Tensor,
    block_contexts: torch.Tensor,
    bit_widths: tuple[int, ...],
    *,
    hessian: torch.Tensor | None,
    layer: int,
    device: torch.device,
    quantize_exl3,
) -> tuple[torch.Tensor, dict]:
    reconstruction = torch.empty_like(source)
    parts = []
    stored_bits = 0
    for context, bit_width in enumerate(bit_widths):
        channels = _channel_indices(block_contexts, context, device)
        decoded, metadata = _quantize_partition(
            source,
            channels,
            bit_width=bit_width,
            hessian=hessian,
            layer=layer,
            context=context,
            device=device,
            quantize_exl3=quantize_exl3,
        )
        reconstruction[:, channels] = decoded
        stored_bits += metadata["stored_bytes"] * 8
        parts.append(metadata)
        del decoded, channels
    return reconstruction, {"parts": parts, "stored_bits": stored_bits}


def _quantize_full_hessian_mixed(
    source: torch.Tensor,
    block_contexts: torch.Tensor,
    bit_widths: tuple[int, ...],
    *,
    matrix: str,
    hessian: torch.Tensor,
    layer: int,
    layout: str,
    device: torch.device,
    quantizer_module,
) -> tuple[torch.Tensor, dict]:
    """Keep full-H feedback and emit a closed TP12 mixed-rate payload."""

    encoding = quantize_mixed_matrix(
        source,
        block_contexts,
        matrix=matrix,
        mode=_mode_for_allocation(bit_widths),
        hessian=hessian,
        layer=layer,
        layout=layout,
        device=device,
        quantizer_module=quantizer_module,
        # Research scripts generally encode one expert per process.  The
        # source storage identity keeps repeated mode calls for that matrix in
        # one scale-sharing group without coupling unrelated expert studies.
        shared_scale_scope=("experiment", layer, int(source.data_ptr())),
    )
    return encoding.reconstruction, encoding.coding


def _mode_for_allocation(bit_widths: tuple[int, ...]) -> ModeSpec:
    canonical = mode_from_context_bits(bit_widths)
    # Preserve the requested context granularity while using canonical R_r
    # identity in descriptors and experiment results.
    return ModeSpec(canonical.mode_id, canonical.name, bit_widths)


def _context_block_order(block_contexts: torch.Tensor, layout: str) -> torch.Tensor:
    """Compatibility wrapper around the validated TP12 context order."""

    contexts = int(block_contexts.max()) + 1
    mode = ModeSpec(0, f"R0-{contexts}c", (3,) * contexts)
    if layout == "importance_ordered":
        return importance_group_order(block_contexts, mode)
    if layout == "tp12_balanced":
        return tp12_storage_group_order(block_contexts, mode)
    raise ValueError(f"unsupported context layout: {layout}")


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    if args.expert not in store.kept_experts(args.layer):
        raise ValueError(f"expert {args.layer}.{args.expert} is not retained")
    samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = (
        None
        if args.validation_capture is None
        else load_layer_samples(args.validation_capture, args.layer - 1)
    )
    context_count = len(args.allocations[0])
    block_contexts, context_meta = _contexts_for_expert(
        samples,
        args.expert,
        context_count,
        split=None if validation_samples is not None else 0,
    )
    _, h2 = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    source = store.load_matrix(args.layer, args.expert, "w2").float().to(device)
    block_contexts = block_contexts.to(device)
    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])
    map_bits = block_contexts.numel() * math.ceil(math.log2(context_count))
    results = []
    full_hessian_mixed_results = []
    if args.include_independent_contexts:
        for bit_widths in args.allocations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            reconstruction, coding = _run_configuration(
                source,
                block_contexts,
                bit_widths,
                hessian=h2,
                layer=args.layer,
                device=device,
                quantize_exl3=quantize_exl3,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "bit_widths_low_to_high_context": list(bit_widths),
                    "fixed_trellis_rate_bpw": sum(bit_widths) / len(bit_widths),
                    "actual_rate_including_exl_scales_and_context_bpw": (
                        coding["stored_bits"] + map_bits
                    )
                    / source.numel(),
                    "expert_context_map_bytes": map_bits // 8,
                    "weight_nmse": normalized_mse(source, reconstruction),
                    "captured_activation": _activation_metrics(
                        source,
                        reconstruction,
                        samples=samples,
                        validation_samples=validation_samples,
                        expert=args.expert,
                        device=device,
                    ),
                    "parts": coding["parts"],
                    "quantization_seconds": elapsed,
                }
            )
            del reconstruction
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for layout in args.layouts:
        for bit_widths in args.allocations:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            reconstruction, coding = _quantize_full_hessian_mixed(
                source,
                block_contexts,
                bit_widths,
                matrix="w2",
                hessian=h2,
                layer=args.layer,
                layout=layout,
                device=device,
                quantizer_module=quantizer_module,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            mode_bits = math.ceil(math.log2(len(args.allocations)))
            full_hessian_mixed_results.append(
                {
                    "layout": coding["layout"],
                    "mode": coding["mode"],
                    "mode_id": coding["mode_id"],
                    "rate_axis": coding["rate_axis"],
                    "bit_widths_low_to_high_context": list(bit_widths),
                    "fixed_trellis_rate_bpw": coding["index_bits"] / source.numel(),
                    "actual_rate_including_exl_scales_and_context_bpw": (
                        coding["index_bits"] + coding["scale_bits"] + map_bits
                    )
                    / source.numel(),
                    "actual_rate_if_permutation_baked_into_w13_bpw": (
                        coding["index_bits"] + coding["scale_bits"] + mode_bits
                    )
                    / source.numel(),
                    "expert_context_map_bytes": map_bits // 8,
                    "allocation_mode_bits": mode_bits,
                    "weight_nmse": normalized_mse(source, reconstruction),
                    "captured_activation": _activation_metrics(
                        source,
                        reconstruction,
                        samples=samples,
                        validation_samples=validation_samples,
                        expert=args.expert,
                        device=device,
                    ),
                    "proxy": coding["proxy"],
                    "permutation_is_128_context_aligned": coding[
                        "permutation_is_128_context_aligned"
                    ],
                    "tp12_rank_trellis_bpw": coding["tp12_rank_trellis_bpw"],
                    "postpack_tp12_rank_trellis_bpw": coding[
                        "postpack_tp12_rank_trellis_bpw"
                    ],
                    "postpack_moves_complete_128_channel_records": coding[
                        "postpack_moves_complete_128_channel_records"
                    ],
                    "reference_edge_roundtrip": coding[
                        "reference_edge_roundtrip"
                    ],
                    "reference_state_roundtrip": coding[
                        "reference_state_roundtrip"
                    ],
                    "stored_decode_vs_encoder": coding[
                        "stored_decode_vs_encoder"
                    ],
                    "trellis_descriptor": coding["trellis_descriptor"],
                    "scale_tensor_shapes": coding["scale_tensor_shapes"],
                    "quantization_seconds": elapsed,
                }
            )
            del reconstruction
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "kind": "kquant_context_exl3_experiment",
        "schema_version": 2,
        "source": {
            "checkpoint": str(store.root),
            "constraint": "interim EXL3 hybrid only; official model not loaded",
            "capture": str(args.capture.resolve()),
            "validation_capture": (
                None
                if args.validation_capture is None
                else str(args.validation_capture.resolve())
            ),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "expert": args.expert,
            "matrix": "w2",
        },
        "context": {
            "definition": (
                "equal-population rank bins of four-channel mean activation "
                "energy from the experiment's training capture rows"
            ),
            **context_meta,
            "channels_per_context": source.shape[1] // context_count,
            "channel_partition_is_128_aligned": (source.shape[1] // context_count) % 128
            == 0,
        },
        "experiment": {
            "hessian": "captured dense submatrices",
            "mcg_multiplier": MCG_MULT,
            "partitioned_results": results,
            "full_hessian_mixed_results": full_hessian_mixed_results,
        },
        "anchors": {
            "unpartitioned_identity_h_exl3": _load_anchor(args.identity_exl3_anchor),
            "unpartitioned_captured_h_exl3": _load_anchor(args.hessian_exl3_anchor),
        },
        "limitations": [
            "support counts and document-level uncertainty must authorize mode selection",
            "separate contexts discard cross-context Hessian terms",
            "mixed full-H encoding uses K=3 global-scale search for all contexts",
            "baked permutation requires matching w1/w3 rows and w2 columns",
            "reference closure covers stored trellis edge symbols, not procedural state reconstruction",
            "rate excludes final alignment padding and pooled expert-mode metadata",
            "equal average TP rank rate does not prove equal mixed-K decode latency",
            "no production decoder or MMA kernel implements mixed-bit partitions",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("/data/kquant/k3-route-aware-pilot-v2-exl3-r4.kqcapture"),
    )
    parser.add_argument(
        "--hessians",
        type=Path,
        default=Path("out/k3-route-aware-pilot-v2-exl3-r4-subset.kqhess"),
    )
    parser.add_argument(
        "--validation-capture",
        type=Path,
        help=(
            "optional document-disjoint capture used only for held-out metrics; "
            "when supplied, all rows in --capture may inform ranking"
        ),
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--allocations", type=_parse_allocations)
    modes.add_argument("--r-values", type=_parse_r_values)
    parser.add_argument(
        "--layouts", type=_parse_layouts, default=("importance_ordered",)
    )
    parser.add_argument("--include-independent-contexts", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--identity-exl3-anchor", type=Path)
    parser.add_argument("--hessian-exl3-anchor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r_values = tuple(range(13)) if args.r_values is None else args.r_values
    if args.allocations is None:
        args.allocations = tuple(
            RATE_TRANSFER_MODES[r].context_bits for r in r_values
        )
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
