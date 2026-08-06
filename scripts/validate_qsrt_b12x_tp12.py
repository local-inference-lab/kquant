#!/usr/bin/env python3
"""Run one persisted TP12 layer/rank through the production routed decoder.

This gate complements source-to-slab byte closure.  It first requires the
independent production vLLM reader to reproduce kquant's exact compressed
tensors and sampled kept-MXFP4 tensors.  It then feeds those production-reader
bytes to B12X and checks the complete route/FC1/SiTU/FC2/reduction result
against an independent PyTorch reconstruction and under CUDA graph replay.
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

from kquant.exl3_reference import (
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3,
    CODEBOOK_SQG_NORMAL_E4M3,
    decode_qsrt_regularized_weight,
    normalized_hadamard,
)
from kquant.qsrt import (
    KEEP_STORAGE_EXTERNAL_X4T,
    LATENT_CHANNELS,
    SCHEMA,
    TILES_PER_RECORD_AXIS,
    TP_SIZE,
    PackedTP12Trellis,
    TP12TrellisDescriptor,
    matrix_rate_axis,
    tp12_logical_pair_index,
    tp12_record_bits,
    unpack_tp12_trellis_states,
)
from kquant.pack.qsrt_slab import (
    TP12CompressedRankPayload,
    TP12LayerReader,
    layer_filename,
)
from kquant.source_weights import PackedMXFP4Matrix
from kquant.x4t import (
    X4TLayerReader,
    tp12_mxfp4_rank_components,
    x4t_layer_path,
)


ROUTE_EXPERTS = 896
LOCAL_INTERMEDIATE = 256
TILES = (64, 256, 64, 256)
MATRIX_GEOMETRY = {
    "w1": (LATENT_CHANNELS // 16, 3072 // 16),
    "w3": (LATENT_CHANNELS // 16, 3072 // 16),
    "w2": (3072 // 16, LATENT_CHANNELS // 16),
}
COMPRESSED_FIELD_MAP = {
    "expert_ids": "compressed_expert_ids",
    "w13_trellis": "w13_trellis",
    "w2_trellis": "w2_trellis",
    "gate_suh": "gate_suh",
    "up_suh": "up_suh",
    "intermediate_rotations": "intermediate_rotations",
    "down_svh": "down_svh",
    "fc1_pair_modes": "fc1_pair_modes",
    "fc2_pair_modes": "fc2_pair_modes",
}


def _b12x_codebook_contract(codebook: str) -> tuple[str, dict[str, int]]:
    """Return B12X's exact source name and required procedural marker."""

    if codebook == CODEBOOK_SQG_NORMAL_E4M3:
        return "exl3_trellis_sqg_e4m3", {}
    if codebook == CODEBOOK_SQG_CHEB_NORMAL_E4M3:
        return "exl3_trellis_sqg_cheb_e4m3", {}
    if codebook == CODEBOOK_SQG_CHEB_NORMAL_K2_Q8H4_W2_E4M3:
        return "exl3_trellis_sqg_cheb_k2_q8h4_w2_e4m3", {}
    raise ValueError(f"unsupported materialized QSRT codebook: {codebook!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_tensor(
    name: str, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"serving-reader {name} contract mismatch: "
            f"{actual.dtype} {tuple(actual.shape)} != "
            f"{expected.dtype} {tuple(expected.shape)}"
        )
    if not torch.equal(actual, expected):
        mismatched = int(torch.count_nonzero(actual != expected))
        raise ValueError(
            f"serving-reader {name} differs from the materializer reference "
            f"at {mismatched}/{actual.numel()} elements"
        )


def _compare_compressed_readers(
    reference: TP12CompressedRankPayload,
    serving: Any,
) -> None:
    """Require the independent vLLM reader to reproduce every B12X tensor."""

    if serving.rank != reference.rank:
        raise ValueError(
            f"serving-reader rank {serving.rank} != reference rank {reference.rank}"
        )
    for reference_name, serving_name in COMPRESSED_FIELD_MAP.items():
        _require_exact_tensor(
            reference_name,
            getattr(serving, serving_name),
            getattr(reference, reference_name),
        )


def _sample_slots(count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    return tuple(sorted({0, count // 2, count - 1}))


def _compare_kept_reader_samples(
    reader: TP12LayerReader,
    serving: Any,
    *,
    rank: int,
    x4t_reader: X4TLayerReader | None,
) -> int:
    """Check first/middle/last kept slots and all three matrix orientations."""

    serving_ids = tuple(int(value) for value in serving.kept_expert_ids.tolist())
    if serving_ids != reader.kept_experts:
        raise ValueError("serving-reader kept expert IDs disagree with the slab")
    local_width = 3072 // TP_SIZE
    checked = 0
    for slot in _sample_slots(len(serving_ids)):
        expert = serving_ids[slot]
        for matrix in ("w1", "w3", "w2"):
            if x4t_reader is None:
                raise ValueError("QSRT kept experts require an X4T sidecar")
            record = x4t_reader.read(expert, matrix)
            source = PackedMXFP4Matrix(
                packed=record.packed,
                scale=record.scale,
            )
            packed, scale = tp12_mxfp4_rank_components(
                source, matrix, rank
            )
            if matrix == "w1":
                actual_packed = serving.w13_mxfp4[slot, :local_width]
                actual_scale = serving.w13_mxfp4_scale[slot, :local_width]
            elif matrix == "w3":
                actual_packed = serving.w13_mxfp4[slot, local_width:]
                actual_scale = serving.w13_mxfp4_scale[slot, local_width:]
            else:
                actual_packed = serving.w2_mxfp4[slot]
                actual_scale = serving.w2_mxfp4_scale[slot]
            _require_exact_tensor(
                f"kept expert {expert} {matrix} packed", actual_packed, packed
            )
            _require_exact_tensor(
                f"kept expert {expert} {matrix} scale", actual_scale, scale
            )
        checked += 1
    return checked


def _load_vllm_reader() -> tuple[Any, Path]:
    try:
        module = importlib.import_module(
            "vllm.model_executor.layers.quantization.kquant_kimi_k3_qsrt_tp12"
        )
    except ImportError as exc:
        raise RuntimeError(
            "the production vLLM QSRT reader is unavailable; run with "
            "/home/luke/projects/vllm/.venv/bin/python"
        ) from exc
    module_path = Path(module.__file__).resolve()
    return module.read_tp12_rank_payload, module_path


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


def _resolve_layer_path(args: argparse.Namespace) -> Path:
    if args.layer_file is not None:
        return args.layer_file.resolve()
    if args.artifact is None:
        raise ValueError("one of --artifact or --layer-file is required")
    return (args.artifact / layer_filename(args.layer)).resolve()


def _resolve_x4t_path(args: argparse.Namespace) -> Path | None:
    if args.x4t_file is not None:
        return args.x4t_file.resolve()
    if args.artifact is None:
        return None
    candidate = x4t_layer_path(args.artifact, args.layer).resolve()
    return candidate if candidate.is_file() else None


def _right_hadamard(
    value: torch.Tensor,
    hadamard: torch.Tensor,
    *,
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    store_fp16: bool,
) -> torch.Tensor:
    """Independent FP16-boundary model of B12X's block-Hadamard path."""

    work = value.float()
    if suh is not None:
        work = (work * suh.float()).to(torch.float16).float()
    rows, width = work.shape
    if width % 128:
        raise ValueError("Hadamard reference width must be divisible by 128")
    work = (work.reshape(rows, width // 128, 128) @ hadamard).reshape(
        rows, width
    )
    if svh is not None:
        work = work * svh.float()
    return work.to(torch.float16) if store_fp16 else work


def _regularized_matrix(
    reader: TP12LayerReader,
    *,
    expert: int,
    matrix: str,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parts = reader.read_compressed_matrix(expert, matrix)
    format_spec = reader.formats[expert]
    mode_id = format_spec.r2 if matrix == "w2" else format_spec.r13
    k_tiles, n_tiles = MATRIX_GEOMETRY[matrix]
    packed = PackedTP12Trellis(
        TP12TrellisDescriptor(
            mode_id=mode_id,
            rate_axis=matrix_rate_axis(matrix),
            k_tiles=k_tiles,
            n_tiles=n_tiles,
            schema=SCHEMA,
        ),
        parts["trellis"].to(device),
    )
    tile_bits = tuple(
        bits
        for record_bits in tp12_record_bits(mode_id)
        for bits in (record_bits,) * TILES_PER_RECORD_AXIS
    )
    regularized = decode_qsrt_regularized_weight(
        unpack_tp12_trellis_states(packed),
        rate_axis=packed.descriptor.rate_axis,
        tile_bits=tile_bits,
        codebook=reader.header.codebook,
    )
    logical_pair = tp12_logical_pair_index(reader.header.layer, expert, rank)
    begin = logical_pair * 256
    end = begin + 256
    if matrix in ("w1", "w3"):
        regularized = regularized[:, begin:end].contiguous()
        suh = parts["suh"]
        svh = parts["svh"][begin:end]
    else:
        regularized = regularized[begin:end].contiguous()
        suh = parts["suh"][begin:end]
        svh = parts["svh"]
    return regularized, suh.to(device), svh.to(device)


def _reference_routed_experts(
    reader: TP12LayerReader,
    *,
    source: torch.Tensor,
    selected_experts: torch.Tensor,
    topk_weights: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Reconstruct the complete expert path without using B12X decode code."""

    hadamard = normalized_hadamard(device=source.device)
    output = torch.zeros(
        (int(source.shape[0]), LATENT_CHANNELS),
        dtype=torch.float32,
        device=source.device,
    )
    selected = [int(expert) for expert in selected_experts.cpu().tolist()]
    for slot, expert in enumerate(selected):
        gate_weight, gate_suh, gate_svh = _regularized_matrix(
            reader,
            expert=expert,
            matrix="w1",
            rank=rank,
            device=source.device,
        )
        gate_a = _right_hadamard(
            source.to(torch.float16),
            hadamard,
            suh=gate_suh,
            store_fp16=True,
        )
        gate = (gate_a.float() @ gate_weight.float()).to(torch.float16)
        gate = _right_hadamard(
            gate, hadamard, svh=gate_svh, store_fp16=True
        )
        del gate_a, gate_weight, gate_suh, gate_svh

        up_weight, up_suh, up_svh = _regularized_matrix(
            reader,
            expert=expert,
            matrix="w3",
            rank=rank,
            device=source.device,
        )
        up_a = _right_hadamard(
            source.to(torch.float16),
            hadamard,
            suh=up_suh,
            store_fp16=True,
        )
        up = (up_a.float() @ up_weight.float()).to(torch.float16)
        up = _right_hadamard(up, hadamard, svh=up_svh, store_fp16=True)
        del up_a, up_weight, up_suh, up_svh

        activated = (
            4.0
            * torch.tanh(gate.float() / 4.0)
            * torch.sigmoid(gate.float())
            * 25.0
            * torch.tanh(up.float() / 25.0)
        ).to(torch.float16)
        del gate, up

        down_weight, down_suh, down_svh = _regularized_matrix(
            reader,
            expert=expert,
            matrix="w2",
            rank=rank,
            device=source.device,
        )
        down_a = _right_hadamard(
            activated,
            hadamard,
            suh=down_suh,
            store_fp16=True,
        )
        down = (down_a.float() @ down_weight.float()).to(torch.float16)
        route = _right_hadamard(
            down, hadamard, svh=down_svh, store_fp16=False
        )
        output += topk_weights[:, slot : slot + 1] * route
        del activated, down_a, down, route, down_weight, down_suh, down_svh
    return output


def _reference_route_middle(
    reader: TP12LayerReader,
    *,
    source: torch.Tensor,
    selected_experts: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Reconstruct route-major physical pre-w2 rows independently of B12X."""

    hadamard = normalized_hadamard(device=source.device)
    routes: list[torch.Tensor] = []
    selected = [int(expert) for expert in selected_experts.cpu().tolist()]
    for expert in selected:
        gate_weight, gate_suh, gate_svh = _regularized_matrix(
            reader,
            expert=expert,
            matrix="w1",
            rank=rank,
            device=source.device,
        )
        gate_a = _right_hadamard(
            source.to(torch.float16),
            hadamard,
            suh=gate_suh,
            store_fp16=True,
        )
        gate = (gate_a.float() @ gate_weight.float()).to(torch.float16)
        gate = _right_hadamard(
            gate, hadamard, svh=gate_svh, store_fp16=True
        )

        up_weight, up_suh, up_svh = _regularized_matrix(
            reader,
            expert=expert,
            matrix="w3",
            rank=rank,
            device=source.device,
        )
        up_a = _right_hadamard(
            source.to(torch.float16),
            hadamard,
            suh=up_suh,
            store_fp16=True,
        )
        up = (up_a.float() @ up_weight.float()).to(torch.float16)
        up = _right_hadamard(up, hadamard, svh=up_svh, store_fp16=True)
        routes.append(
            (
                4.0
                * torch.tanh(gate.float() / 4.0)
                * torch.sigmoid(gate.float())
                * 25.0
                * torch.tanh(up.float() / 25.0)
            ).to(torch.float16)
        )

    # The kernel cache is token-major, then top-k route-major.
    return torch.stack(routes, dim=1).reshape(-1, LOCAL_INTERMEDIATE)


def _numerical_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    difference = actual_f32 - expected_f32
    relative = difference.norm() / expected_f32.norm().clamp_min(1.0e-9)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    return {
        "relative_error": float(relative),
        "cosine": float(cosine),
        "max_abs_error": float(difference.abs().max()),
    }


def _select_route_slots(payload: TP12CompressedRankPayload, topk: int) -> torch.Tensor:
    """Prefer P24-bearing experts so a sparse layer exercises both arms."""

    rate_shifted = (payload.fc1_pair_modes != 0) | (payload.fc2_pair_modes != 0)
    shifted_slots = torch.nonzero(rate_shifted).flatten()
    uniform_slots = torch.nonzero(~rate_shifted).flatten()
    slots = torch.cat((shifted_slots, uniform_slots))[:topk]
    if int(slots.numel()) != topk:
        raise ValueError(f"rank payload has fewer than topk={topk} experts")
    return slots


def validate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sparkinfer.moe import fused_moe
    except ImportError as exc:
        raise RuntimeError(
            "sparkinfer is unavailable; run with the B12X/vLLM environment"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the B12X routed closure")
    if not 0 <= args.rank < TP_SIZE:
        raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
    if args.tokens <= 0 or args.topk <= 0:
        raise ValueError("tokens and topk must be positive")
    if args.max_relative_error <= 0 or not 0 < args.min_cosine <= 1:
        raise ValueError("invalid numerical closure thresholds")

    device = torch.device(args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise RuntimeError(f"SM120/SM121 is required, got SM{major}{minor}")
    layer_path = _resolve_layer_path(args)
    x4t_path = _resolve_x4t_path(args)
    read_serving_rank, serving_reader_path = _load_vllm_reader()
    vllm_root = args.vllm_root.resolve()
    if not serving_reader_path.is_relative_to(vllm_root):
        raise ValueError(
            f"production reader {serving_reader_path} is outside {vllm_root}"
        )
    with TP12LayerReader(layer_path) as reader:
        if reader.header.layer != args.layer:
            raise ValueError(
                f"slab contains layer {reader.header.layer}, expected {args.layer}"
            )
        reference_payload = reader.read_compressed_rank(args.rank)
        external_x4t = (
            reader.header.layout.keep_storage == KEEP_STORAGE_EXTERNAL_X4T
        )
        if external_x4t != (x4t_path is not None):
            requirement = "requires" if external_x4t else "must not use"
            raise ValueError(
                f"materialized slab {requirement} an external X4T sidecar"
            )
        x4t_reader = X4TLayerReader(x4t_path) if x4t_path is not None else None
        if x4t_reader is not None and x4t_reader.layer != args.layer:
            raise ValueError("X4T sidecar layer disagrees with its trellis slab")
        expected_bits = [
            4 if format_spec.is_mxfp4 else 3 for format_spec in reader.formats
        ]
        codebook = reader.header.codebook
        serving_payload = read_serving_rank(
            layer_path,
            layer=args.layer,
            rank=args.rank,
            x4t_path=x4t_path,
            expected_bits=expected_bits,
            expected_codebook=codebook,
        )
        _compare_compressed_readers(reference_payload, serving_payload)
        kept_samples_checked = _compare_kept_reader_samples(
            reader,
            serving_payload,
            rank=args.rank,
            x4t_reader=x4t_reader,
        )
        kept_experts = len(reader.kept_experts)

    cpu_payload = TP12CompressedRankPayload(
        rank=args.rank,
        **{
            reference_name: getattr(serving_payload, serving_name)
            for reference_name, serving_name in COMPRESSED_FIELD_MAP.items()
        },
    )
    kept_payload_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            serving_payload.w13_mxfp4,
            serving_payload.w13_mxfp4_scale,
            serving_payload.w2_mxfp4,
            serving_payload.w2_mxfp4_scale,
        )
    )
    del reference_payload, serving_payload
    if cpu_payload.num_experts < args.topk:
        raise ValueError(
            f"rank payload has {cpu_payload.num_experts} compressed experts, "
            f"fewer than topk={args.topk}"
        )

    payload_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            cpu_payload.w13_trellis,
            cpu_payload.w2_trellis,
            cpu_payload.gate_suh,
            cpu_payload.up_suh,
            cpu_payload.intermediate_rotations,
            cpu_payload.down_svh,
            cpu_payload.fc1_pair_modes,
            cpu_payload.fc2_pair_modes,
        )
    )
    payload = cpu_payload.to(device)
    source_format, codebook_marker = _b12x_codebook_contract(codebook)
    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format=source_format,
        activation="situ",
        params_dtype=torch.bfloat16,
        num_experts=payload.num_experts,
        hidden_size=LATENT_CHANNELS,
        intermediate_size=LOCAL_INTERMEDIATE,
        w13_layout="w13",
        trellis_bits=3,
        trellis_tile_config=TILES,
        trellis_pair_format="tp12_p24_p33",
    )
    weights = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        **codebook_marker,
        **payload.b12x_prepare_kwargs(),
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    source = (
        torch.randn(
            (args.tokens, LATENT_CHANNELS),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * args.input_scale
    ).to(torch.bfloat16)
    selected_slots = _select_route_slots(cpu_payload, args.topk)
    selected_cpu = cpu_payload.expert_ids.index_select(0, selected_slots)
    selected = selected_cpu.to(device)
    topk_ids = selected.reshape(1, -1).expand(args.tokens, -1).contiguous()
    topk_weights = torch.full(
        (args.tokens, args.topk),
        1.0 / args.topk,
        dtype=torch.float32,
        device=device,
    )
    expert_map = torch.full(
        (ROUTE_EXPERTS,), -1, dtype=torch.int32, device=device
    )
    expert_map[payload.expert_ids.to(torch.long)] = torch.arange(
        payload.num_experts, dtype=torch.int32, device=device
    )
    if codebook == CODEBOOK_SQG_CHEB_NORMAL_E4M3:
        from sparkinfer.moe._shared.kernels.trellis_w4a8 import (
            make_trellis_w4a8_moe_scratch,
            run_trellis_w4a8_moe,
        )

        runtime_decoder = "trellis_w4a8_direct_lut"
        prepared = weights.representation.value
        scratch = make_trellis_w4a8_moe_scratch(
            m=args.tokens,
            topk=args.topk,
            hidden_size=LATENT_CHANNELS,
            intermediate_size=LOCAL_INTERMEDIATE,
            device=device,
        )

        def run_compressed() -> torch.Tensor:
            return run_trellis_w4a8_moe(
                source,
                prepared,
                topk_weights,
                topk_ids,
                scratch,
                expert_map=expert_map,
                fast_math=True,
            )

        restore_middle = None

    else:
        runtime_decoder = "trellis_w4a16"
        execution_plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=args.tokens,
                num_topk=args.topk,
                route_num_experts=ROUTE_EXPERTS,
                device=device,
                weight_plan=weights.plan,
                quant_mode="w4a16",
                w4a16_block_size_m=8,
            )
        )
        scratch_spec = execution_plan.scratch_specs()[0]
        scratch = torch.empty(
            scratch_spec.shape,
            dtype=scratch_spec.dtype,
            device=scratch_spec.device,
        )
        binding = fused_moe.bind(
            execution_plan,
            scratch=scratch,
            a=source,
            experts=weights,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=expert_map,
            output_expert_map=expert_map,
        )

        def run_compressed() -> torch.Tensor:
            return binding.run()

        def restore_middle() -> torch.Tensor:
            from vllm.model_executor.layers.fused_moe.kquant_capture import (
                restore_kquant_exl3_mid,
            )

            prepared = weights.representation.value
            cache = binding.intermediate_cache2
            if cache is None:
                raise RuntimeError("B12X did not expose its pre-w2 cache")
            logical_scratch = torch.empty(
                (args.tokens * args.topk, LOCAL_INTERMEDIATE),
                dtype=cache.dtype,
                device=device,
            )
            return restore_kquant_exl3_mid(
                binding=binding,
                topk_ids=topk_ids,
                expert_map=expert_map,
                intermediate_rotations=prepared.intermediate_rotations,
                logical_scratch=logical_scratch,
            )

    eager = run_compressed().clone()
    restored_middle = None if restore_middle is None else restore_middle().clone()
    torch.cuda.synchronize(device)
    if not bool(torch.all(torch.isfinite(eager))):
        raise RuntimeError("materialized rank produced non-finite eager output")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_compressed()
    graph.replay()
    torch.cuda.synchronize(device)
    graph_exact = torch.equal(captured, eager)
    if not graph_exact:
        raise RuntimeError("materialized rank graph output differs from eager")

    with TP12LayerReader(layer_path) as reader:
        expected = _reference_routed_experts(
            reader,
            source=source,
            selected_experts=selected_cpu,
            topk_weights=topk_weights,
            rank=args.rank,
        )
        expected_middle = (
            None
            if restored_middle is None
            else _reference_route_middle(
                reader,
                source=source,
                selected_experts=selected_cpu,
                rank=args.rank,
            )
        )
    numerical = _numerical_metrics(eager, expected)
    middle_numerical = (
        None
        if restored_middle is None or expected_middle is None
        else _numerical_metrics(restored_middle, expected_middle)
    )
    numerical_pass = bool(
        numerical["relative_error"] <= args.max_relative_error
        and numerical["cosine"] >= args.min_cosine
    )
    if not numerical_pass:
        raise RuntimeError(
            "materialized routed decode disagrees with the independent "
            f"reference: relative_error={numerical['relative_error']:.6g}, "
            f"cosine={numerical['cosine']:.6g}"
        )

    routed_local = selected_slots.to(device)
    return {
        "kind": "kquant_kimi_k3_qsrt_b12x_tp12_closure",
        "layer_file": str(layer_path),
        "layer_file_bytes": layer_path.stat().st_size,
        "x4t_file": None if x4t_path is None else str(x4t_path),
        "x4t_file_bytes": None if x4t_path is None else x4t_path.stat().st_size,
        "x4t_file_sha256": None if x4t_path is None else _sha256(x4t_path),
        "layer": args.layer,
        "rank": args.rank,
        "tp_size": TP_SIZE,
        "serving_reader": str(serving_reader_path),
        "serving_reader_sha256": _sha256(serving_reader_path),
        "serving_reader_bit_exact": True,
        "kept_reader_samples_checked": kept_samples_checked,
        "device": torch.cuda.get_device_name(device),
        "capability": [major, minor],
        "compressed_experts": payload.num_experts,
        "kept_experts": kept_experts,
        "codebook": codebook,
        "b12x_source_format": source_format,
        "runtime_decoder": runtime_decoder,
        "compressed_payload_bytes_loaded": payload_bytes,
        "kept_payload_bytes_loaded": kept_payload_bytes,
        "tokens": args.tokens,
        "topk": args.topk,
        "routed_experts": selected_cpu.tolist(),
        "routed_fc1_p24_fraction": float(
            payload.fc1_pair_modes.index_select(0, routed_local).float().mean()
        ),
        "routed_fc2_p24_fraction": float(
            payload.fc2_pair_modes.index_select(0, routed_local).float().mean()
        ),
        "output_norm": float(eager.float().norm()),
        "output_max_abs": float(eager.float().abs().max()),
        "eager_finite": True,
        "cuda_graph_exact": graph_exact,
        "restored_pre_w2": middle_numerical,
        "independent_reference": {
            "implementation": (
                f"kquant_pytorch_regularized_{codebook.replace('-', '_')}"
                "_hadamard_situ"
            ),
            "max_relative_error": args.max_relative_error,
            "min_cosine": args.min_cosine,
            **numerical,
            "pass": numerical_pass,
        },
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact", type=Path)
    source.add_argument("--layer-file", type=Path)
    parser.add_argument(
        "--x4t-file",
        type=Path,
        help="external X4T peer required with a standalone QSRT --layer-file",
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help=(
            "standard deviation of the synthetic RMS-normalized hidden-state "
            "probe (default: 1.0); very small probes are not a valid numerical "
            "closure because the checkpoint's FP16 transform scales can drive "
            "intermediates into the subnormal regime"
        ),
    )
    parser.add_argument("--max-relative-error", type=float, default=2.0e-2)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument(
        "--vllm-root", type=Path, default=Path("/home/luke/projects/vllm")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args)
    if args.output is not None:
        _atomic_json(args.output, result)
    print(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
