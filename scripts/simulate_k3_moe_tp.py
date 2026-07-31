#!/usr/bin/env python
"""Simulate one real Kimi-K3 MoE block at arbitrary logical TP sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kquant.io.mxfp4 import dequant
from kquant.tp_simulator import (
    column_parallel,
    comparison_metrics,
    kimi_topk,
    reduce_partials,
    rms_norm,
    row_parallel_partials,
    situ,
)


class TensorStore:
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


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(value).hexdigest()


def load_input(
    args: argparse.Namespace,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    if args.input is not None:
        with safe_open(str(args.input), framework="pt") as handle:
            inputs = handle.get_tensor(args.input_key)
        inputs = inputs.reshape(-1, inputs.shape[-1])[: args.tokens]
        if inputs.shape[-1] != hidden_size:
            raise ValueError(
                f"input width {inputs.shape[-1]} != hidden size {hidden_size}"
            )
        return inputs.to(device=device, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    return (
        torch.randn(
            args.tokens,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
        )
        .mul_(args.input_scale)
        .to(device=device, dtype=torch.bfloat16)
    )


def expert_weights(
    store: TensorStore,
    layer: int,
    expert: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        f"experts.{expert}"
    )
    weights = []
    for matrix in ("w1", "w3", "w2"):
        packed = store.get(f"{base}.{matrix}.weight_packed")
        scale = store.get(f"{base}.{matrix}.weight_scale")
        weights.append(dequant(packed, scale).to(device, torch.bfloat16))
    return tuple(weights)  # type: ignore[return-value]


def routed_expert_partials(
    inputs: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tp_size: int,
) -> list[torch.Tensor]:
    w1, w3, w2 = weights
    intermediate = w1.shape[0]
    if intermediate % tp_size:
        raise ValueError(
            f"expert intermediate size {intermediate} is not divisible by TP{tp_size}"
        )
    local = intermediate // tp_size
    partials: list[torch.Tensor] = []
    for rank in range(tp_size):
        shard = slice(rank * local, (rank + 1) * local)
        gate = torch.nn.functional.linear(inputs, w1[shard])
        up = torch.nn.functional.linear(inputs, w3[shard])
        activated = situ(gate, up)
        partials.append(
            torch.nn.functional.linear(activated, w2[:, shard])
        )
    return partials


def shared_expert_partials(
    inputs: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    tp_size: int,
) -> list[torch.Tensor]:
    intermediate = gate.shape[0]
    if intermediate % tp_size:
        raise ValueError(
            f"shared intermediate size {intermediate} is not divisible by TP{tp_size}"
        )
    local = intermediate // tp_size
    partials = []
    for rank in range(tp_size):
        shard = slice(rank * local, (rank + 1) * local)
        activated = situ(
            torch.nn.functional.linear(inputs, gate[shard]),
            torch.nn.functional.linear(inputs, up[shard]),
        )
        partials.append(
            torch.nn.functional.linear(activated, down[:, shard])
        )
    return partials


@torch.inference_mode()
def simulate(args: argparse.Namespace) -> dict:
    model_dir = args.model.resolve()
    config = json.loads((model_dir / "config.json").read_text())
    config = config.get("text_config", config)
    hidden_size = int(config["hidden_size"])
    latent_size = int(config["routed_expert_hidden_size"])
    topk = int(config["num_experts_per_token"])
    if topk != 16:
        raise ValueError(f"expected Kimi-K3 top-16 routing, got {topk}")
    tp_sizes = [int(value) for value in args.tp_sizes.split(",")]
    if len(tp_sizes) != len(set(tp_sizes)) or any(value <= 0 for value in tp_sizes):
        raise ValueError("--tp-sizes must contain unique positive integers")
    reference_tp = args.reference_tp
    all_tp_sizes = [reference_tp, *[value for value in tp_sizes if value != reference_tp]]

    device = torch.device(args.device)
    store = TensorStore(model_dir)
    prefix = (
        f"language_model.model.layers.{args.layer}.block_sparse_moe"
    )
    inputs = load_input(args, hidden_size, device)
    gate_weight = store.get(f"{prefix}.gate.weight").to(
        device, torch.bfloat16
    )
    correction = store.get(f"{prefix}.gate.e_score_correction_bias").to(
        device, torch.float32
    )
    latent_down_weight = store.get(
        f"{prefix}.routed_expert_down_proj.weight"
    ).to(device, torch.bfloat16)
    latent_norm_weight = store.get(
        f"{prefix}.routed_expert_norm.weight"
    ).to(device, torch.bfloat16)
    latent_up_weight = store.get(
        f"{prefix}.routed_expert_up_proj.weight"
    ).to(device, torch.bfloat16)
    shared_gate = store.get(f"{prefix}.shared_experts.gate_proj.weight").to(
        device, torch.bfloat16
    )
    shared_up = store.get(f"{prefix}.shared_experts.up_proj.weight").to(
        device, torch.bfloat16
    )
    shared_down = store.get(f"{prefix}.shared_experts.down_proj.weight").to(
        device, torch.bfloat16
    )

    state: dict[int, dict] = {}
    selected_experts: set[int] = set()
    for tp_size in all_tp_sizes:
        router_logits = column_parallel(
            inputs,
            gate_weight,
            tp_size,
            out_dtype=torch.float32,
        )
        route_weights, route_ids = kimi_topk(
            router_logits,
            correction,
            topk,
            routed_scaling_factor=float(config["routed_scaling_factor"]),
        )
        latent_inputs = column_parallel(
            inputs, latent_down_weight, tp_size
        )
        state[tp_size] = {
            "router_logits": router_logits,
            "route_weights": route_weights,
            "route_ids": route_ids,
            "latent_inputs": latent_inputs,
            "routed_partials": [
                torch.zeros(
                    inputs.shape[0],
                    latent_size,
                    dtype=torch.float32,
                    device=device,
                )
                for _ in range(tp_size)
            ],
        }
        selected_experts.update(int(value) for value in route_ids.flatten())

    for index, expert in enumerate(sorted(selected_experts), start=1):
        print(
            f"expert {index}/{len(selected_experts)}: {expert}",
            flush=True,
        )
        weights = expert_weights(store, args.layer, expert, device)
        for tp_size in all_tp_sizes:
            route_ids = state[tp_size]["route_ids"]
            route_weights = state[tp_size]["route_weights"]
            for token in range(inputs.shape[0]):
                positions = (route_ids[token] == expert).nonzero().flatten()
                for position in positions:
                    partials = routed_expert_partials(
                        state[tp_size]["latent_inputs"][token : token + 1],
                        weights,
                        tp_size,
                    )
                    route_weight = route_weights[token, position].float()
                    for rank, partial in enumerate(partials):
                        state[tp_size]["routed_partials"][rank][token] += (
                            route_weight * partial[0].float()
                        )
        del weights
        if device.type == "cuda":
            torch.cuda.empty_cache()

    outputs: dict[int, torch.Tensor] = {}
    for tp_size in all_tp_sizes:
        routed = reduce_partials(
            state[tp_size]["routed_partials"],
            dtype=torch.bfloat16,
        )
        routed = rms_norm(
            routed,
            latent_norm_weight,
            float(config["rms_norm_eps"]),
        )
        routed_partials = row_parallel_partials(
            routed, latent_up_weight, tp_size
        )
        shared_partials = shared_expert_partials(
            inputs,
            shared_gate,
            shared_up,
            shared_down,
            tp_size,
        )
        rank_partials = [
            routed_partial + shared_partial
            for routed_partial, shared_partial in zip(
                routed_partials, shared_partials
            )
        ]
        outputs[tp_size] = reduce_partials(
            rank_partials, dtype=torch.bfloat16
        )

    reference = outputs[reference_tp]
    reference_routes = state[reference_tp]["route_ids"]
    comparisons = {}
    passed = True
    for tp_size in tp_sizes:
        metrics = comparison_metrics(reference, outputs[tp_size])
        route_equal = torch.equal(reference_routes, state[tp_size]["route_ids"])
        arm_pass = (
            route_equal
            and metrics["relative_l2"] <= args.relative_l2_limit
            and metrics["cosine"] >= args.cosine_limit
        )
        comparisons[str(tp_size)] = {
            **metrics,
            "route_ids_equal": route_equal,
            "pass": arm_pass,
            "output_sha256": tensor_digest(outputs[tp_size]),
        }
        passed &= arm_pass

    return {
        "schema_version": 1,
        "model": str(model_dir),
        "layer": args.layer,
        "tokens": args.tokens,
        "seed": args.seed,
        "simulation": "sequential logical rank slices on one device",
        "reference_tp": reference_tp,
        "reference_output_sha256": tensor_digest(reference),
        "reference_route_ids": reference_routes.cpu().tolist(),
        "selected_experts": sorted(selected_experts),
        "limits": {
            "relative_l2": args.relative_l2_limit,
            "cosine": args.cosine_limit,
        },
        "comparisons": comparisons,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/models/k3-trunc-stock"))
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--tp-sizes", default="4,12,16")
    parser.add_argument("--reference-tp", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-key", default="hidden_states")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--relative-l2-limit", type=float, default=0.02)
    parser.add_argument("--cosine-limit", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = simulate(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
