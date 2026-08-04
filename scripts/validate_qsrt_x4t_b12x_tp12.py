#!/usr/bin/env python3
"""Close one persisted QSRT X4T tier through the production TP12 W4A16 path.

The gate reads the artifact with vLLM's production slab/X4T reader, rebuilds
the persistent X4T scale components exactly as serving does, and compares the
routed B12X result with both a dense-scale execution and an independent
checkpoint-order MXFP4 reconstruction.  The latter is essential: dense/X4T
self-parity alone cannot detect a shared gate/up half-order mistake.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from kquant.io import mxfp4
from kquant.pack.qsrt_slab import TP12LayerReader, layer_filename
from kquant.x4t import x4t_layer_path


TP_SIZE = 12
ROUTE_EXPERTS = 896
HIDDEN_SIZE = 3584
LOCAL_INTERMEDIATE = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sample_slots(count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    return tuple(sorted({0, count // 2, count - 1}))


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    return {
        "relative_error": float(
            difference.norm() / expected_f32.norm().clamp_min(1.0e-9)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f32.flatten(), expected_f32.flatten(), dim=0
            )
        ),
        "max_abs_error": float(difference.abs().max()),
    }


def _reference(
    source: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    swapped: bool = False,
) -> torch.Tensor:
    """Independent BF16-boundary SiTU reference in checkpoint row order."""

    output = torch.zeros(
        (int(source.shape[0]), HIDDEN_SIZE),
        dtype=torch.float32,
        device=source.device,
    )
    source_f32 = source.float()
    for slot in range(int(w13.shape[0])):
        first = mxfp4.dequant(
            w13[slot, :LOCAL_INTERMEDIATE],
            w13_scale[slot, :LOCAL_INTERMEDIATE],
        ).to(device=source.device, dtype=torch.bfloat16).float()
        second = mxfp4.dequant(
            w13[slot, LOCAL_INTERMEDIATE:],
            w13_scale[slot, LOCAL_INTERMEDIATE:],
        ).to(device=source.device, dtype=torch.bfloat16).float()
        gate_weight, up_weight = (second, first) if swapped else (first, second)
        gate = source_f32 @ gate_weight.T
        up = source_f32 @ up_weight.T
        middle = (
            4.0
            * torch.tanh(gate / 4.0)
            * torch.sigmoid(gate)
            * 25.0
            * torch.tanh(up / 25.0)
        ).to(torch.bfloat16).float()
        down_weight = mxfp4.dequant(w2[slot], w2_scale[slot]).to(
            device=source.device, dtype=torch.bfloat16
        ).float()
        route = (middle @ down_weight.T).to(torch.bfloat16).float()
        output += topk_weights[:, slot : slot + 1] * route
    return output


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the X4T TP12 closure")
    if not 0 <= args.rank < TP_SIZE:
        raise ValueError(f"rank must lie in 0..{TP_SIZE - 1}")
    if args.input_scale <= 0:
        raise ValueError("input scale must be positive")

    device = torch.device(args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"SM120/SM121 is required, got SM{major}{minor}")

    artifact = args.artifact.resolve()
    slab_path = artifact / layer_filename(args.layer)
    x4t_path = x4t_layer_path(artifact, args.layer)
    if not slab_path.is_file() or not x4t_path.is_file():
        raise FileNotFoundError("the requested QSRT slab/X4T pair is incomplete")

    reader_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.kquant_kimi_k3_qsrt_tp12"
    )
    hybrid_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid"
    )
    x4t_module = importlib.import_module(
        "vllm.model_executor.layers.quantization.kquant_x4t"
    )
    reader_path = Path(reader_module.__file__).resolve()
    with TP12LayerReader(slab_path) as slab:
        if slab.header.layer != args.layer:
            raise ValueError("slab layer does not match --layer")
        expected_bits = [4 if spec.is_mxfp4 else 3 for spec in slab.formats]
        expected_kept = slab.kept_experts
        codebook = slab.header.codebook
    payload = reader_module.read_tp12_rank_payload(
        slab_path,
        layer=args.layer,
        rank=args.rank,
        x4t_path=x4t_path,
        expected_bits=expected_bits,
        expected_codebook=codebook,
    )
    kept_ids = tuple(int(value) for value in payload.kept_expert_ids.tolist())
    if kept_ids != expected_kept:
        raise ValueError("production reader changed the X4T expert ordering")
    slots = _sample_slots(len(kept_ids))
    if not slots:
        raise ValueError("the requested layer contains no X4T experts")
    index = torch.tensor(slots, dtype=torch.long)
    selected_ids = tuple(kept_ids[slot] for slot in slots)
    w13 = payload.w13_mxfp4.index_select(0, index).contiguous()
    w13_scale = payload.w13_mxfp4_scale.index_select(0, index).contiguous()
    w2 = payload.w2_mxfp4.index_select(0, index).contiguous()
    w2_scale = payload.w2_mxfp4_scale.index_select(0, index).contiguous()
    del payload

    pack_components = x4t_module.pack_x4t_scale_components
    w13_components = [pack_components(scale) for scale in w13_scale]
    w2_components = [pack_components(scale) for scale in w2_scale]

    from sparkinfer._lib.quant.x4t_scales import make_x4t_scale_batch
    from sparkinfer.moe._shared.kernels.w4a16.kernel import (
        compile_w4a16_fused_moe,
        run_w4a16_moe,
    )
    from sparkinfer.moe._shared.kernels.w4a16.prepare import (
        make_w4a16_packed_buffers,
        prepare_w4a16_fp4_e8m0_k32_weights,
        prepare_w4a16_x4t_tp12_weights,
    )

    w13_layout = hybrid_module._QSRT_X4T_W13_LAYOUT
    w13_task_rows = hybrid_module._QSRT_X4T_W13_EXCEPTION_TASK_ROWS
    w2_task_rows = hybrid_module._QSRT_X4T_W2_EXCEPTION_TASK_ROWS
    w13_rotation = hybrid_module._QSRT_X4T_W13_EXCEPTION_ROW_ROTATION
    if (w13_layout, w13_task_rows, w2_task_rows, w13_rotation) != (
        "w31",
        128,
        896,
        0,
    ):
        raise ValueError("vLLM's QSRT X4T ABI is not the validated TP12 ABI")

    w13_x4t = make_x4t_scale_batch(
        [fixed for fixed, _ in w13_components],
        [exceptions for _, exceptions in w13_components],
        rows=512,
        columns=112,
        device=device,
        exception_task_rows=w13_task_rows,
        exception_row_rotation=w13_rotation,
    )
    w2_x4t = make_x4t_scale_batch(
        [fixed for fixed, _ in w2_components],
        [exceptions for _, exceptions in w2_components],
        rows=3584,
        columns=8,
        device=device,
        exception_task_rows=w2_task_rows,
    )
    w13_device, w13_scale_device, w2_device, w2_scale_device = (
        value.to(device) for value in (w13, w13_scale, w2, w2_scale)
    )
    global_scale = torch.ones(len(slots), dtype=torch.float32, device=device)
    dense = prepare_w4a16_fp4_e8m0_k32_weights(
        w13_device,
        w13_scale_device,
        global_scale,
        w2_device,
        w2_scale_device,
        global_scale.clone(),
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout=w13_layout,
    )
    sentinel = 0xD6
    w13_scratch = torch.full(
        (len(slots), 112, 512), sentinel, dtype=torch.uint8, device=device
    )
    w2_scratch = torch.full(
        (len(slots), 8, 3584), sentinel, dtype=torch.uint8, device=device
    )
    compressed = prepare_w4a16_x4t_tp12_weights(
        w13_device,
        w13_x4t,
        global_scale,
        w2_device,
        w2_x4t,
        global_scale.clone(),
        w13_scratch,
        w2_scratch,
        activation="situ",
        params_dtype=torch.bfloat16,
        w13_layout=w13_layout,
    )
    if not torch.equal(compressed.w13, dense.w13) or not torch.equal(
        compressed.w2, dense.w2
    ):
        raise RuntimeError("X4T and dense preparation changed the nibble plane")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    source = (
        torch.randn(
            (1, HIDDEN_SIZE),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * args.input_scale
    ).to(torch.bfloat16)
    topk_ids = torch.tensor([selected_ids], dtype=torch.int32, device=device)
    topk_weights = torch.full(
        (1, len(slots)),
        1.0 / len(slots),
        dtype=torch.float32,
        device=device,
    )
    expert_map = torch.full(
        (ROUTE_EXPERTS,), -1, dtype=torch.int32, device=device
    )
    expert_map[list(selected_ids)] = torch.arange(
        len(slots), dtype=torch.int32, device=device
    )

    def buffers(prepared: Any) -> Any:
        return make_w4a16_packed_buffers(
            prepared,
            m=1,
            topk=len(slots),
            dtype=torch.bfloat16,
            device=device,
            route_num_experts=ROUTE_EXPERTS,
        )

    def launch(prepared: Any, scratch: Any) -> torch.Tensor:
        return run_w4a16_moe(
            source,
            prepared,
            topk_weights,
            topk_ids,
            activation="situ",
            expert_map=expert_map,
            intermediate_cache13=scratch.intermediate_cache13,
            intermediate_cache2=scratch.intermediate_cache2,
            output=scratch.output,
            fc1_c_tmp=scratch.fc1_c_tmp,
            fc2_c_tmp=scratch.fc2_c_tmp,
            packed_route_indices=scratch.packed_route_indices,
            block_expert_ids=scratch.block_expert_ids,
            packed_route_count=scratch.packed_route_count,
            expert_offsets=scratch.expert_offsets,
        )

    dense_buffers = buffers(dense)
    compressed_buffers = buffers(compressed)
    dense_output = launch(dense, dense_buffers).clone()
    eager = launch(compressed, compressed_buffers).clone()
    torch.cuda.synchronize(device)
    if not torch.equal(eager, dense_output):
        raise RuntimeError("X4T scale predecode differs from dense scale execution")
    for slot in range(len(slots)):
        if not torch.equal(
            compressed.w13_scale[slot], dense.w13_scale[slot].view(torch.uint8)
        ) or not torch.equal(
            compressed.w2_scale[slot], dense.w2_scale[slot].view(torch.uint8)
        ):
            raise RuntimeError("X4T expanded scale bytes differ from dense scales")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(compressed, compressed_buffers)
    compressed.w13_scale.fill_(sentinel)
    compressed.w2_scale.fill_(sentinel)
    compressed_buffers.output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize(device)
    graph_exact = torch.equal(compressed_buffers.output, eager)
    if not graph_exact:
        raise RuntimeError("X4T graph replay differs from eager execution")

    # vLLM uses a separate direct-top-k launch for single-token decode.  It
    # remaps global expert ids to this tier's local ids before entering b12x,
    # so exercise that exact route ABI as well as the packed prefill path.
    props = torch.cuda.get_device_properties(device)
    direct_launch = compile_w4a16_fused_moe(
        size_m=8,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=256,
        num_experts=len(slots),
        top_k=len(slots),
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=8,
        max_m_blocks=8 * len(slots),
        element_dtype="bf16",
        fast_math=True,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(
            getattr(props, "shared_memory_per_block_optin", 101_376)
        ),
        weight_layout=compressed.weight_layout,
        scale_format=compressed.scale_format,
        force_tile_config=(64, 256, 64, 256),
        direct_topk_routes=True,
        tc_decode_fused_sum=True,
    )
    local_topk_ids = torch.arange(
        len(slots), dtype=torch.int32, device=device
    ).view(1, -1)
    direct_buffers = make_w4a16_packed_buffers(
        compressed,
        m=8,
        topk=len(slots),
        dtype=torch.bfloat16,
        device=device,
        route_num_experts=ROUTE_EXPERTS,
    )
    compressed.w13_scale.fill_(sentinel)
    compressed.w2_scale.fill_(sentinel)
    direct_buffers.output.zero_()
    direct_output = run_w4a16_moe(
        source,
        compressed,
        topk_weights,
        local_topk_ids,
        activation="situ",
        intermediate_cache13=direct_buffers.intermediate_cache13,
        intermediate_cache2=direct_buffers.intermediate_cache2,
        output=direct_buffers.output[:1],
        fc1_c_tmp=direct_buffers.fc1_c_tmp,
        fc2_c_tmp=direct_buffers.fc2_c_tmp,
        packed_route_indices=direct_buffers.packed_route_indices,
        block_expert_ids=direct_buffers.block_expert_ids,
        packed_route_count=direct_buffers.packed_route_count,
        expert_offsets=direct_buffers.expert_offsets,
        fused_launch=direct_launch,
    )[:1].clone()
    torch.cuda.synchronize(device)
    direct_metrics = _metrics(direct_output, dense_output)
    direct_pass = bool(
        direct_metrics["relative_error"] <= args.max_relative_error
        and direct_metrics["cosine"] >= args.min_cosine
    )
    if not direct_pass:
        raise RuntimeError(
            "X4T direct-decode launch failed the dense execution gate: "
            f"relative_error={direct_metrics['relative_error']:.6g}, "
            f"cosine={direct_metrics['cosine']:.6g}"
        )

    expected = _reference(
        source,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
    )
    swapped = _reference(
        source,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_weights,
        swapped=True,
    )
    numerical = _metrics(eager, expected)
    swapped_numerical = _metrics(eager, swapped)
    numerical_pass = bool(
        numerical["relative_error"] <= args.max_relative_error
        and numerical["cosine"] >= args.min_cosine
        and swapped_numerical["relative_error"]
        > 10.0 * numerical["relative_error"]
    )
    if not numerical_pass:
        raise RuntimeError(
            "X4T W4A16 output failed the independent checkpoint-order gate: "
            f"relative_error={numerical['relative_error']:.6g}, "
            f"cosine={numerical['cosine']:.6g}, "
            "swapped_relative_error="
            f"{swapped_numerical['relative_error']:.6g}"
        )

    return {
        "kind": "kquant_kimi_k3_qsrt_x4t_b12x_tp12_closure",
        "artifact": str(artifact),
        "layer_file": str(slab_path),
        "x4t_file": str(x4t_path),
        "x4t_file_sha256": _sha256(x4t_path),
        "layer": args.layer,
        "rank": args.rank,
        "tp_size": TP_SIZE,
        "device": torch.cuda.get_device_name(device),
        "capability": [major, minor],
        "production_reader": str(reader_path),
        "production_reader_sha256": _sha256(reader_path),
        "x4t_experts": len(kept_ids),
        "sampled_experts": list(selected_ids),
        "w13_layout": w13_layout,
        "w13_exception_task_rows": w13_task_rows,
        "w2_exception_task_rows": w2_task_rows,
        "w13_exception_row_rotation": w13_rotation,
        "input_scale": args.input_scale,
        "nibble_plane_exact": True,
        "expanded_scales_exact": True,
        "dense_x4t_output_exact": True,
        "cuda_graph_exact": graph_exact,
        "direct_decode_reference": {
            "local_topk_ids": local_topk_ids.tolist(),
            "max_relative_error": args.max_relative_error,
            "min_cosine": args.min_cosine,
            **direct_metrics,
            "pass": direct_pass,
        },
        "independent_reference": {
            "implementation": "checkpoint_order_mxfp4_bf16_situ",
            "max_relative_error": args.max_relative_error,
            "min_cosine": args.min_cosine,
            **numerical,
            "pass": numerical_pass,
        },
        "swapped_gate_up_control": swapped_numerical,
        "output_norm": float(eager.float().norm()),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--max-relative-error", type=float, default=1.0e-2)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args)
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
