#!/usr/bin/env python
"""Numerical closure for K3's original MXFP4 and EXL3 MoE weights.

The original checkpoint is always exercised through the ordinary MMA-packed
W4A16 MoE kernel.  It never uses the hybrid artifact's source-native
"kept-weight" representation.

The script keeps three questions separate:

1. Does the normal W4A16 kernel match reconstructed original MXFP4 weights?
2. Does the EXL3 kernel match reconstruction of the exact EXL3 artifact?
3. How far is the EXL3 artifact from the original MXFP4 source?

Logical TP ranks run sequentially on one GPU.  A non-128-aligned original
MXFP4 shard (K3 TP16 has 3072 / 16 = 192 channels) is padded with zero
channels to a supported packed-W4A16 geometry.  The padded channels make no
contribution, so this exercises the normal kernel while preserving the exact
logical TP partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kquant.correctness import write_json
from kquant.io.hf_cache import CheckpointCache, resolve
from kquant.io.mxfp4 import dequant
from kquant.io.stream import load_tensor
from kquant.tp_simulator import comparison_metrics, reduce_partials, situ

MATRICES = ("w1", "w3", "w2")
SOURCE_W4A16 = "source-w4a16"
EXL3 = "all-exl3"
SOURCE_W13_LAYOUT = "w31"


class IndexedTensorStore:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )
        self.weight_map: dict[str, str] = index["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(name)
        with safe_open(
            str(self.model_dir / filename), framework="pt"
        ) as handle:
            return handle.get_tensor(name)


@dataclass(frozen=True)
class Scenario:
    name: str
    experts: tuple[int, ...]
    representation: str


def _base(layer: int, expert: int) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        f"experts.{expert}"
    )


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def _label(value: Any) -> str:
    enum_value = getattr(value, "value", None)
    return str(enum_value if enum_value is not None else value)


def _tile_config(hidden_size: int, intermediate_size: int) -> tuple[int, ...]:
    if hidden_size == 3584 and intermediate_size == 192:
        return (128, 64, 64, 128)
    for candidate in ((64, 256, 64, 256), (64, 128, 64, 128)):
        fc1_k, fc1_n, fc2_k, fc2_n = candidate
        if (
            hidden_size % fc1_k == 0
            and (2 * intermediate_size) % fc1_n == 0
            and intermediate_size % fc2_k == 0
            and hidden_size % fc2_n == 0
        ):
            return candidate
    raise ValueError(
        f"no trellis tile config for H={hidden_size}, I={intermediate_size}"
    )


def _load_source_raw(
    cache: CheckpointCache,
    layer: int,
    expert: int,
) -> dict[str, dict[str, torch.Tensor]]:
    base = _base(layer, expert)
    return {
        matrix: {
            "packed": load_tensor(
                cache, f"{base}.{matrix}.weight_packed"
            ),
            "scale": load_tensor(cache, f"{base}.{matrix}.weight_scale"),
        }
        for matrix in MATRICES
    }


def _load_exl3_raw(
    store: IndexedTensorStore,
    layer: int,
    expert: int,
) -> dict[str, dict[str, torch.Tensor]]:
    base = _base(layer, expert)
    return {
        matrix: {
            part: store.get(f"{base}.{matrix}.exl3_{part}")
            for part in ("trellis", "suh", "svh")
        }
        for matrix in MATRICES
    }


def _reconstruct_source(
    raw: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    *,
    scale_byte_max: int | None = None,
) -> dict[str, torch.Tensor]:
    result = {}
    for matrix, value in raw.items():
        scale = value["scale"]
        if scale_byte_max is not None:
            # This is the exact representable-scale contract used by
            # prepare_w4a16_fp4_e8m0_k32_weights before it rearranges the
            # scales into MMA layout.
            scale = scale.view(torch.uint8).clamp(max=scale_byte_max)
        result[matrix] = dequant(value["packed"], scale).to(
            device=device,
            dtype=torch.bfloat16,
        )
    return result


def _reconstruct_exl3(
    raw: dict[str, dict[str, torch.Tensor]],
    *,
    hidden_size: int,
    intermediate_size: int,
    mcg_mult: int,
    device: torch.device,
    exllamav3_dir: Path,
) -> dict[str, torch.Tensor]:
    path = str(exllamav3_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    from exllamav3.modules.quant.exl3 import LinearEXL3

    marker = torch.tensor(
        mcg_mult & 0xFFFFFFFF,
        dtype=torch.uint32,
        device=device,
    ).view(torch.int32)
    result = {}
    for matrix in MATRICES:
        in_features, out_features = (
            (hidden_size, intermediate_size)
            if matrix in {"w1", "w3"}
            else (intermediate_size, hidden_size)
        )
        value = raw[matrix]
        module = LinearEXL3(
            config=None,
            in_features=in_features,
            out_features=out_features,
            suh=value["suh"].to(device).contiguous(),
            svh=value["svh"].to(device).contiguous(),
            trellis=value["trellis"].to(device).contiguous(),
            mcg=marker,
            out_dtype=torch.float16,
        )
        # EXL3 reconstructs [in, out]; torch.linear consumes [out, in].
        result[matrix] = (
            module.get_weight_tensor()
            .T.contiguous()
            .to(dtype=torch.bfloat16)
        )
        del module
    return result


def _routes(
    tokens: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(topk, generator=generator) for _ in range(tokens)]
    ).to(device=device, dtype=torch.int32)
    logits = torch.randn(tokens, topk, generator=generator)
    weights = torch.softmax(logits, dim=-1).to(
        device=device,
        dtype=torch.float32,
    )
    return ids.contiguous(), weights.contiguous()


def _scratch(plan: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
    values = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    )
    return values[0] if len(values) == 1 else values


def _run_prepared(
    x: torch.Tensor,
    prepared: Any,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
) -> tuple[torch.Tensor, bool, dict[str, str]]:
    from sparkinfer.moe import fused_moe

    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=int(x.shape[0]),
            num_topk=int(route_ids.shape[1]),
            device=x.device,
            weight_plan=prepared.plan,
            core_token_counts=(int(x.shape[0]),),
            # IDs are already compact local expert IDs.  Asking b12x for a
            # global routing workspace here silently changes that contract.
            route_num_experts=0,
            quant_mode="w4a16",
            frozen=True,
        )
    )
    scratch = _scratch(plan)
    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x.contiguous(),
        experts=prepared,
        topk_weights=route_weights,
        topk_ids=route_ids,
    )
    output = fused_moe.run(binding=binding)[: x.shape[0]].float().clone()
    zero_binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=x.contiguous(),
        experts=prepared,
        topk_weights=torch.zeros_like(route_weights),
        topk_ids=route_ids,
    )
    zero = fused_moe.run(binding=zero_binding)[: x.shape[0]]
    inactive_zero = bool(torch.count_nonzero(zero).item() == 0)
    core_plan = getattr(plan, "_core_workspace_plan", plan)
    return (
        output,
        inactive_zero,
        {
            "implementation": _label(
                getattr(core_plan, "implementation", "unknown")
            ),
            "execution": _label(getattr(core_plan, "execution", "unknown")),
        },
    )


def _padded_w4a16_intermediate(local_intermediate: int) -> int:
    """Packed W4A16 kernels require a 128-aligned intermediate extent."""
    if local_intermediate <= 0:
        raise ValueError("local intermediate size must be positive")
    return ((local_intermediate + 127) // 128) * 128


def _pad_axis(
    value: torch.Tensor,
    *,
    axis: int,
    target: int,
    fill: int,
) -> torch.Tensor:
    current = int(value.shape[axis])
    if current > target:
        raise ValueError(f"cannot pad axis {axis} from {current} down to {target}")
    if current == target:
        return value.contiguous()
    shape = list(value.shape)
    shape[axis] = target
    output = torch.full(
        shape,
        fill_value=fill,
        dtype=value.dtype,
        device=value.device,
    )
    output.narrow(axis, 0, current).copy_(value)
    return output


def _run_source_w4a16_rank(
    x: torch.Tensor,
    raws: list[dict[str, dict[str, torch.Tensor]]],
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    tp_size: int,
    rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, bool, dict[str, Any]]:
    """Run source MXFP4 through the ordinary MMA-packed W4A16 kernel."""
    from sparkinfer.moe import fused_moe
    from sparkinfer.moe._shared.execution import PreparedWeightLayout

    local = intermediate_size // tp_size
    padded = _padded_w4a16_intermediate(local)
    row_slice = slice(rank * local, (rank + 1) * local)
    byte_slice = slice(rank * local // 2, (rank + 1) * local // 2)
    scale_slice = slice(rank * local // 32, (rank + 1) * local // 32)

    w13 = torch.stack(
        [
            torch.cat(
                (
                    _pad_axis(
                        raw["w1"]["packed"][row_slice],
                        axis=0,
                        target=padded,
                        fill=0,
                    ),
                    _pad_axis(
                        raw["w3"]["packed"][row_slice],
                        axis=0,
                        target=padded,
                        fill=0,
                    ),
                )
            )
            for raw in raws
        ]
    ).to(x.device)
    # A zero FP4 payload is independent of its block scale.  Use the unit
    # E8M0 exponent for padding so the packed representation is canonical.
    w13_scale = torch.stack(
        [
            torch.cat(
                (
                    _pad_axis(
                        raw["w1"]["scale"][row_slice],
                        axis=0,
                        target=padded,
                        fill=127,
                    ),
                    _pad_axis(
                        raw["w3"]["scale"][row_slice],
                        axis=0,
                        target=padded,
                        fill=127,
                    ),
                )
            )
            for raw in raws
        ]
    ).to(x.device)
    w2 = torch.stack(
        [
            _pad_axis(
                raw["w2"]["packed"][:, byte_slice],
                axis=1,
                target=padded // 2,
                fill=0,
            )
            for raw in raws
        ]
    ).to(x.device)
    w2_scale = torch.stack(
        [
            _pad_axis(
                raw["w2"]["scale"][:, scale_slice],
                axis=1,
                target=padded // 32,
                fill=127,
            )
            for raw in raws
        ]
    ).to(x.device)
    ones = torch.ones(len(raws), dtype=torch.float32, device=x.device)
    scalar = torch.ones((), dtype=torch.float32, device=x.device)
    plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="fp4_e8m0_k32",
        activation="situ",
        params_dtype=x.dtype,
        num_experts=len(raws),
        hidden_size=hidden_size,
        intermediate_size=padded,
        # vLLM fuses the checkpoint rows as [w1/gate, w3/up].  b12x names
        # that gate-first source order "w31"; using "w13" rotates the halves
        # and silently swaps SiTU's nonlinear and linear inputs.
        w13_layout=SOURCE_W13_LAYOUT,
        w4a16_layout=PreparedWeightLayout.MMA_PACKED,
    )
    selected_layout = plan.required_weight_layout("w4a16")
    if selected_layout is not PreparedWeightLayout.MMA_PACKED:
        raise RuntimeError(
            "original MXFP4 closure must use normal MMA-packed W4A16; "
            f"planner selected {_label(selected_layout)!r}"
        )
    prepared = fused_moe.prepare_weights(
        plan=plan,
        params_dtype=x.dtype,
        w1_fp4=w13,
        w1_blockscale=w13_scale,
        w1_global_scale=ones,
        a1_gscale=scalar,
        w2_fp4=w2,
        w2_blockscale=w2_scale,
        w2_global_scale=ones.clone(),
        a2_gscale=scalar.clone(),
    )
    output, inactive_zero, execution = _run_prepared(
        x, prepared, route_ids, route_weights
    )
    return (
        output,
        inactive_zero,
        {
            **execution,
            "quant_mode": "w4a16",
            "source_format": "fp4_e8m0_k32",
            "weight_layout": _label(selected_layout),
            "w13_layout": SOURCE_W13_LAYOUT,
            "logical_intermediate": local,
            "kernel_intermediate": padded,
            "zero_padded_channels": padded - local,
        },
    )


def _run_exl3_rank(
    x: torch.Tensor,
    raws: list[dict[str, dict[str, torch.Tensor]]],
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    tp_size: int,
    rank: int,
    hidden_size: int,
    intermediate_size: int,
    mcg_mult: int,
) -> tuple[torch.Tensor, bool, dict[str, Any]]:
    from sparkinfer.moe import fused_moe

    local = intermediate_size // tp_size
    tile_slice = slice(rank * local // 16, (rank + 1) * local // 16)
    value_slice = slice(rank * local, (rank + 1) * local)
    w13 = torch.stack(
        [
            torch.stack(
                (raw["w1"]["trellis"], raw["w3"]["trellis"])
            )
            for raw in raws
        ]
    )[:, :, :, tile_slice].permute(1, 0, 2, 3, 4).contiguous().to(x.device)
    w2 = torch.stack(
        [raw["w2"]["trellis"][tile_slice] for raw in raws]
    ).contiguous().to(x.device)
    gate_suh = torch.stack(
        [raw["w1"]["suh"] for raw in raws]
    ).contiguous().to(x.device)
    up_suh = torch.stack(
        [raw["w3"]["suh"] for raw in raws]
    ).contiguous().to(x.device)
    inter_rot = torch.stack(
        [
            torch.cat(
                (
                    raw["w1"]["svh"][value_slice],
                    raw["w2"]["suh"][value_slice],
                    raw["w3"]["svh"][value_slice],
                )
            )
            for raw in raws
        ]
    ).contiguous().to(x.device)
    down_svh = torch.stack(
        [raw["w2"]["svh"] for raw in raws]
    ).contiguous().to(x.device)
    tile_config = _tile_config(hidden_size, local)
    plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="situ",
        params_dtype=x.dtype,
        num_experts=len(raws),
        hidden_size=hidden_size,
        intermediate_size=local,
        w13_layout="w13",
        trellis_bits=3,
        trellis_tile_config=tile_config,
    )
    prepared = fused_moe.prepare_weights(
        plan=plan,
        params_dtype=x.dtype,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=inter_rot,
        down_svh=down_svh,
        trellis_mcg=mcg_mult,
    )
    output, inactive_zero, execution = _run_prepared(
        x, prepared, route_ids, route_weights
    )
    return (
        output,
        inactive_zero,
        {
            **execution,
            "quant_mode": "w4a16",
            "source_format": "exl3_trellis_mcg",
            "weight_layout": _label(
                plan.required_weight_layout("w4a16")
            ),
            "logical_intermediate": local,
            "kernel_intermediate": local,
            "zero_padded_channels": 0,
            "tile_config": list(tile_config),
        },
    )


def _reference_rank(
    x: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    experts: tuple[int, ...],
    weights: dict[int, dict[str, torch.Tensor]],
    *,
    tp_size: int,
    rank: int,
) -> torch.Tensor:
    intermediate_size = next(iter(weights.values()))["w1"].shape[0]
    local = intermediate_size // tp_size
    shard = slice(rank * local, (rank + 1) * local)
    output = torch.zeros(
        x.shape[0],
        next(iter(weights.values()))["w2"].shape[0],
        dtype=torch.float32,
        device=x.device,
    )
    for position, expert in enumerate(experts):
        selected = (route_ids == position).nonzero(as_tuple=False)
        if selected.numel() == 0:
            continue
        token_ids = selected[:, 0]
        route_slots = selected[:, 1]
        expert_weights = weights[expert]
        gate = torch.nn.functional.linear(
            x[token_ids], expert_weights["w1"][shard]
        )
        up = torch.nn.functional.linear(
            x[token_ids], expert_weights["w3"][shard]
        )
        activated = situ(gate, up)
        partial = torch.nn.functional.linear(
            activated, expert_weights["w2"][:, shard]
        )
        contribution = (
            route_weights[token_ids, route_slots].unsqueeze(1)
            * partial.float()
        )
        output.index_add_(0, token_ids, contribution)
    return output


def _scenario_set(
    exl3_experts: list[int],
    requested: list[str],
    topk: int,
) -> list[Scenario]:
    if len(exl3_experts) < topk:
        raise ValueError(
            f"layer has only {len(exl3_experts)} EXL3 experts; need {topk}"
        )
    # Use the same expert IDs for both representations so their outputs can
    # be compared without changing the route.
    experts = tuple(exl3_experts[:topk])
    scenarios = []
    for name in requested:
        if name == SOURCE_W4A16:
            scenarios.append(Scenario(name, experts, SOURCE_W4A16))
        elif name == EXL3:
            scenarios.append(Scenario(name, experts, EXL3))
        elif name in {"all-kept", "mixed"}:
            raise ValueError(
                f"scenario {name!r} would exercise the kept-weight kernel; "
                "it is intentionally excluded from numerical closure"
            )
        else:
            raise ValueError(f"unknown scenario {name!r}")
    return scenarios


@torch.inference_mode()
def closure(args: argparse.Namespace) -> dict:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
    os.environ.setdefault("SPARKINFER_MOE_FORCE_A16", "1")
    serve_dir = args.serve_dir.resolve()
    artifact_dir = args.artifact.resolve()
    config = json.loads((serve_dir / "config.json").read_text())
    text = config.get("text_config", config)
    hidden_size = int(text["routed_expert_hidden_size"])
    intermediate_size = int(text["moe_intermediate_size"])
    topk = int(text["num_experts_per_token"])
    if topk != 16:
        raise ValueError(f"expected K3 top-16 routing, got {topk}")

    allocation_document = json.loads(
        (artifact_dir / "allocation-exl3.json").read_text()
    )
    allocation = allocation_document["layers"][str(args.layer)]
    exl3_experts = [int(value) for value in allocation["exl3"]]
    manifest = json.loads(
        (artifact_dir / "kquant_exl3_manifest.json").read_text()
    )
    mcg_mult = int(manifest["mcg_mult"])
    scenarios = _scenario_set(
        exl3_experts,
        [value.strip() for value in args.scenarios.split(",") if value.strip()],
        topk,
    )
    if not scenarios:
        raise ValueError("no runnable scenarios")

    tp_sizes = [
        int(value) for value in args.tp_sizes.split(",") if value.strip()
    ]
    token_counts = [
        int(value) for value in args.token_counts.split(",") if value.strip()
    ]
    for tp_size in tp_sizes:
        if intermediate_size % tp_size:
            raise ValueError(
                f"intermediate size {intermediate_size} is not divisible "
                f"by TP{tp_size}"
            )
        if intermediate_size // tp_size % 32:
            raise ValueError(
                f"TP{tp_size} local intermediate must be divisible by 32"
            )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    store = IndexedTensorStore(serve_dir)
    source_cache = resolve(repo_dir=args.cache_repo_dir)
    if source_cache.missing_shards:
        raise FileNotFoundError(
            f"source cache is missing {len(source_cache.missing_shards)} shards"
        )

    selected_experts = sorted(
        {expert for scenario in scenarios for expert in scenario.experts}
    )
    source_raw: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    exl3_raw: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    source_weights: dict[int, dict[str, torch.Tensor]] = {}
    w4a16_weights: dict[int, dict[str, torch.Tensor]] = {}
    artifact_weights: dict[int, dict[str, torch.Tensor]] = {}
    need_exl3 = any(scenario.representation == EXL3 for scenario in scenarios)
    for index, expert in enumerate(selected_experts, start=1):
        print(
            f"loading source expert {index}/{len(selected_experts)}: {expert}",
            flush=True,
        )
        source_raw[expert] = _load_source_raw(
            source_cache, args.layer, expert
        )
        source_weights[expert] = _reconstruct_source(
            source_raw[expert], device
        )
        w4a16_weights[expert] = _reconstruct_source(
            source_raw[expert],
            device,
            scale_byte_max=247,
        )
        if need_exl3:
            exl3_raw[expert] = _load_exl3_raw(
                store, args.layer, expert
            )
            artifact_weights[expert] = _reconstruct_exl3(
                exl3_raw[expert],
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                mcg_mult=mcg_mult,
                device=device,
                exllamav3_dir=args.exllamav3_dir.resolve(),
            )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = {
        tokens: (
            torch.randn(
                tokens,
                hidden_size,
                generator=generator,
                dtype=torch.float32,
            )
            * args.input_scale
        ).to(device=device, dtype=torch.bfloat16)
        for tokens in token_counts
    }
    routes = {
        tokens: _routes(
            tokens,
            topk,
            args.seed + tokens,
            device,
        )
        for tokens in token_counts
    }

    result_arms = []
    outputs: dict[tuple[str, int, int], torch.Tensor] = {}
    output_modes: dict[tuple[str, int, int], str] = {}
    production_kernel_gaps: dict[tuple[str, int], dict[str, Any]] = {}
    passed = True
    started = time.monotonic()
    for scenario in scenarios:
        print(f"scenario {scenario.name}", flush=True)
        scenario_raw = (
            source_raw if scenario.representation == SOURCE_W4A16 else exl3_raw
        )
        exact_weights = (
            w4a16_weights
            if scenario.representation == SOURCE_W4A16
            else artifact_weights
        )
        raws = [scenario_raw[expert] for expert in scenario.experts]
        for tokens in token_counts:
            x = inputs[tokens]
            route_ids, route_weights = routes[tokens]
            exact_reference_by_tp = {}
            source_reference_by_tp = {}
            kernel_by_tp = {}
            kernel_closure = {}
            inactive_zero = {}
            kernel_paths = {}
            for tp_size in tp_sizes:
                print(f"  M={tokens} logical TP{tp_size}", flush=True)
                local_intermediate = intermediate_size // tp_size
                kernel_supported = (
                    scenario.representation == SOURCE_W4A16
                    or (
                        hidden_size % 128 == 0
                        and local_intermediate % 128 == 0
                    )
                )
                unsupported_reason = None
                if not kernel_supported:
                    unsupported_reason = (
                        "full-rotation EXL3 Trellis requires hidden and "
                        "rank-local intermediate dimensions divisible by 128; "
                        f"got H={hidden_size}, I_local={local_intermediate}"
                    )
                    production_kernel_gaps[(scenario.name, tp_size)] = {
                        "scenario": scenario.name,
                        "tp_size": tp_size,
                        "reason": unsupported_reason,
                    }
                    # Logical reference coverage is still useful, but it does
                    # not substitute for production-kernel coverage.
                    passed = False
                kernel_rank_outputs = []
                exact_rank_outputs = []
                source_rank_outputs = []
                rank_metrics = {}
                rank_zero = {}
                rank_paths = {}
                for rank in range(tp_size):
                    exact = _reference_rank(
                        x,
                        route_ids,
                        route_weights,
                        scenario.experts,
                        exact_weights,
                        tp_size=tp_size,
                        rank=rank,
                    )
                    source = _reference_rank(
                        x,
                        route_ids,
                        route_weights,
                        scenario.experts,
                        source_weights,
                        tp_size=tp_size,
                        rank=rank,
                    )
                    exact_rank_outputs.append(exact)
                    source_rank_outputs.append(source)
                    if kernel_supported:
                        if scenario.representation == SOURCE_W4A16:
                            actual, zero, path = _run_source_w4a16_rank(
                                x,
                                raws,
                                route_ids,
                                route_weights,
                                tp_size=tp_size,
                                rank=rank,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                            )
                        else:
                            actual, zero, path = _run_exl3_rank(
                                x,
                                raws,
                                route_ids,
                                route_weights,
                                tp_size=tp_size,
                                rank=rank,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                                mcg_mult=mcg_mult,
                            )
                        kernel_rank_outputs.append(actual)
                        metrics = comparison_metrics(exact, actual)
                        rank_pass = (
                            torch.isfinite(actual).all().item()
                            and metrics["relative_l2"]
                            <= args.kernel_relative_l2_limit
                            and metrics["cosine"] >= args.cosine_limit
                        )
                        rank_metrics[str(rank)] = {
                            **metrics,
                            "pass": bool(rank_pass),
                        }
                        rank_zero[str(rank)] = zero
                        rank_paths[str(rank)] = path
                        passed &= bool(rank_pass and zero)

                exact_output = reduce_partials(
                    exact_rank_outputs,
                    dtype=torch.bfloat16,
                )
                source_output = reduce_partials(
                    source_rank_outputs,
                    dtype=torch.bfloat16,
                )
                exact_reference_by_tp[tp_size] = exact_output
                source_reference_by_tp[tp_size] = source_output
                key = (scenario.name, tokens, tp_size)
                if kernel_supported:
                    kernel_output = reduce_partials(
                        kernel_rank_outputs,
                        dtype=torch.bfloat16,
                    )
                    kernel_by_tp[tp_size] = kernel_output
                    outputs[key] = kernel_output
                    output_modes[key] = "production-kernel"
                    combined_metrics = comparison_metrics(
                        exact_output, kernel_output
                    )
                    combined_pass = (
                        torch.isfinite(kernel_output).all().item()
                        and combined_metrics["relative_l2"]
                        <= args.kernel_relative_l2_limit
                        and combined_metrics["cosine"] >= args.cosine_limit
                    )
                    kernel_closure[str(tp_size)] = {
                        "status": "run",
                        **combined_metrics,
                        "pass": bool(combined_pass),
                        "rank_metrics": rank_metrics,
                    }
                    inactive_zero[str(tp_size)] = {
                        "status": "run",
                        "pass": all(rank_zero.values()),
                        "ranks": rank_zero,
                    }
                    kernel_paths[str(tp_size)] = rank_paths
                    passed &= bool(
                        combined_pass
                        and inactive_zero[str(tp_size)]["pass"]
                    )
                else:
                    outputs[key] = exact_output
                    output_modes[key] = "exact-reconstructed-reference"
                    kernel_closure[str(tp_size)] = {
                        "status": "not-run",
                        "reason": unsupported_reason,
                        "pass": False,
                    }
                    inactive_zero[str(tp_size)] = {
                        "status": "not-run",
                        "reason": unsupported_reason,
                        "pass": False,
                    }
                    kernel_paths[str(tp_size)] = {
                        "status": "not-run",
                        "reason": unsupported_reason,
                    }

            reference_tp = tp_sizes[0]
            reference_tp_comparisons = {}
            for tp_size in tp_sizes[1:]:
                metrics = comparison_metrics(
                    exact_reference_by_tp[reference_tp],
                    exact_reference_by_tp[tp_size],
                )
                comparison_pass = (
                    metrics["relative_l2"] <= args.tp_relative_l2_limit
                    and metrics["cosine"] >= args.cosine_limit
                )
                reference_tp_comparisons[str(tp_size)] = {
                    **metrics,
                    "reference_tp": reference_tp,
                    "pass": bool(comparison_pass),
                }
                passed &= bool(comparison_pass)

            kernel_tp_comparisons = {}
            supported_tps = [
                tp_size for tp_size in tp_sizes if tp_size in kernel_by_tp
            ]
            if supported_tps:
                kernel_reference_tp = supported_tps[0]
                for tp_size in supported_tps[1:]:
                    metrics = comparison_metrics(
                        kernel_by_tp[kernel_reference_tp],
                        kernel_by_tp[tp_size],
                    )
                    comparison_pass = (
                        metrics["relative_l2"]
                        <= args.tp_relative_l2_limit
                        and metrics["cosine"] >= args.cosine_limit
                    )
                    kernel_tp_comparisons[str(tp_size)] = {
                        **metrics,
                        "reference_tp": kernel_reference_tp,
                        "pass": bool(comparison_pass),
                    }
                    passed &= bool(comparison_pass)

            quantization_error = {
                str(tp_size): comparison_metrics(
                    source_reference_by_tp[tp_size],
                    exact_reference_by_tp[tp_size],
                )
                for tp_size in tp_sizes
            }
            result_arms.append(
                {
                    "scenario": scenario.name,
                    "representation": scenario.representation,
                    "experts": list(scenario.experts),
                    "tokens": tokens,
                    "kernel_vs_exact_representation": kernel_closure,
                    "exact_representation_vs_original_source": (
                        quantization_error
                    ),
                    "logical_tp_reference_comparisons": (
                        reference_tp_comparisons
                    ),
                    "logical_tp_kernel_comparisons": kernel_tp_comparisons,
                    "inactive_routes_exact_zero": inactive_zero,
                    "kernel_paths": kernel_paths,
                    "kernel_output_sha256": {
                        str(tp_size): _digest(kernel_by_tp[tp_size])
                        for tp_size in supported_tps
                    },
                    "exact_representation_output_sha256": {
                        str(tp_size): _digest(
                            exact_reference_by_tp[tp_size]
                        )
                        for tp_size in tp_sizes
                    },
                }
            )
            torch.cuda.empty_cache()

    representation_comparisons = []
    scenario_names = {scenario.name for scenario in scenarios}
    if {SOURCE_W4A16, EXL3} <= scenario_names:
        for tokens in token_counts:
            for tp_size in tp_sizes:
                representation_comparisons.append(
                    {
                        "tokens": tokens,
                        "tp_size": tp_size,
                        "original_w4a16_vs_exl3": comparison_metrics(
                            outputs[(SOURCE_W4A16, tokens, tp_size)],
                            outputs[(EXL3, tokens, tp_size)],
                        ),
                        "original_mode": output_modes[
                            (SOURCE_W4A16, tokens, tp_size)
                        ],
                        "exl3_mode": output_modes[
                            (EXL3, tokens, tp_size)
                        ],
                    }
                )

    return {
        "schema_version": 3,
        "artifact": str(artifact_dir),
        "serve_dir": str(serve_dir),
        "layer": args.layer,
        "topk": topk,
        "tp_execution": "sequential logical ranks on one GPU",
        "tp_sizes": tp_sizes,
        "token_counts": token_counts,
        "scenarios": [scenario.name for scenario in scenarios],
        "production_kernel_gaps": list(production_kernel_gaps.values()),
        "production_kernel_coverage_complete": not production_kernel_gaps,
        "kept_weight_kernel_exercised": False,
        "original_mxfp4_kernel_contract": {
            "quant_mode": "w4a16",
            "weight_layout": "mma_packed",
            "non_aligned_shards": (
                "zero-pad local intermediate channels to 128 alignment"
            ),
        },
        "limits": {
            "kernel_relative_l2": args.kernel_relative_l2_limit,
            "tp_relative_l2": args.tp_relative_l2_limit,
            "cosine": args.cosine_limit,
        },
        "arms": result_arms,
        "representation_comparisons": representation_comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "pass": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09"),
    )
    parser.add_argument(
        "--serve-dir",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tp-sizes", default="4,12,16")
    parser.add_argument("--token-counts", default="1,8,9,64")
    parser.add_argument(
        "--scenarios",
        default=f"{SOURCE_W4A16},{EXL3}",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-repo-dir", type=Path)
    parser.add_argument(
        "--exllamav3-dir",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--kernel-relative-l2-limit", type=float, default=0.04)
    parser.add_argument("--tp-relative-l2-limit", type=float, default=0.03)
    parser.add_argument("--cosine-limit", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = closure(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
