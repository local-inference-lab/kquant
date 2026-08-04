"""Read-only access to retained experts in a packaged interim K3 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.capture import load_layer_samples
from kquant.io.mxfp4 import dequant


def weight_name(layer: int, expert: int, matrix: str, part: str) -> str:
    return (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts."
        f"{expert}.{matrix}.{part}"
    )


class InterimKeepStore:
    """Resolve only the retained MXFP4 tier of an interim EXL3 serve tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        index_path = self.root / C.INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        values = set(self.weight_map.values())
        if not any(name.startswith("exl3-layer-") for name in values):
            raise ValueError(f"{self.root} is not an interim EXL3 checkpoint")
        if not any(name.startswith("keep-mxfp4-") for name in values):
            raise ValueError(f"{self.root} has no retained interim MXFP4 experts")

    def kept_experts(self, layer: int) -> list[int]:
        experts = []
        for expert in range(C.NUM_EXPERTS):
            name = weight_name(layer, expert, "w1", "weight_packed")
            filename = self.weight_map.get(name, "")
            if filename.startswith("keep-mxfp4-"):
                experts.append(expert)
        return experts

    def load_matrix(self, layer: int, expert: int, matrix: str) -> torch.Tensor:
        packed_name = weight_name(layer, expert, matrix, "weight_packed")
        scale_name = weight_name(layer, expert, matrix, "weight_scale")
        try:
            packed_file = self.weight_map[packed_name]
            scale_file = self.weight_map[scale_name]
        except KeyError as exc:
            raise KeyError(f"expert {layer}.{expert}.{matrix} is not retained") from exc
        if packed_file != scale_file or not packed_file.startswith("keep-mxfp4-"):
            raise ValueError(
                f"expert {layer}.{expert}.{matrix} is not in the interim keep tier"
            )
        path = self.root / packed_file
        with safe_open(path, framework="pt", device="cpu") as handle:
            packed = handle.get_tensor(packed_name)
            scale = handle.get_tensor(scale_name)
        return dequant(packed, scale)


def routed_kept_experts(
    store: InterimKeepStore,
    capture: str | Path,
    layer: int,
) -> tuple[list[int], dict[int, float]]:
    """Order retained experts by captured applied-gate-square mass."""

    kept = set(store.kept_experts(layer))
    samples = load_layer_samples(capture, layer - 1)
    traffic: dict[int, float] = {}
    for expert_row, gate_row in zip(samples.input_experts, samples.input_gates):
        for expert, gate in zip(expert_row.tolist(), gate_row.tolist()):
            if expert in kept:
                traffic[expert] = traffic.get(expert, 0.0) + float(gate) ** 2
    ordered = sorted(traffic, key=lambda expert: (-traffic[expert], expert))
    ordered.extend(sorted(kept - set(ordered)))
    return ordered, traffic
