"""Probe texture/video codec transfers on the interim K3 hybrid checkpoint.

The script reads retained MXFP4 experts from an already packaged interim
EXL3 serve directory.  It never resolves or instantiates the official model.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import torch

from kquant import constants as C
from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.codec_research import (
    affine_prediction,
    endpoint_ternary,
    endpoint_two_subset,
    gather_tiles,
    nearest_neuron_prediction,
    normalized_mse,
    quantized_low_rank,
    sample_tile_coordinates,
    scalar_affine_3bit,
)
from kquant.interim_weights import InterimKeepStore, routed_kept_experts


def _load_exl3_quantizer(exllamav3_root: Path):
    """Import the encoder without executing exllamav3's serving package.

    The encoder only needs the modules, extension, and utility namespaces.
    Importing the top-level package also pulls tokenizer/generator dependencies
    that are intentionally absent from kquant's environment.
    """

    root = exllamav3_root.resolve() / "exllamav3"
    packages = (
        ("exllamav3", root),
        ("exllamav3.modules", root / "modules"),
        ("exllamav3.modules.quant", root / "modules" / "quant"),
        ("exllamav3.util", root / "util"),
    )
    for name, path in packages:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = module.__path__
        module.__spec__ = spec
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)

    util = sys.modules["exllamav3.util"]

    def cuda_sync_active() -> None:
        for device_id in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{device_id}")
            if torch.cuda.memory_allocated(device) > 0:
                torch.cuda.synchronize(device)

    util.cuda_sync_active = cuda_sync_active

    memory = types.ModuleType("exllamav3.util.memory")

    def free_mem() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    memory.free_mem = free_mem
    memory.list_gpu_tensors = lambda *args, **kwargs: None
    sys.modules[memory.__name__] = memory
    util.memory = memory

    progress = types.ModuleType("exllamav3.util.progress")

    class ProgressBar:
        def __init__(self, text: str, count: int, transient: bool = True):
            self.text = text
            self.count = count
            self.transient = transient

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def update(self, value: int) -> None:
            return None

        def new_task(self, text: str, count: int) -> None:
            self.text = text
            self.count = count

    progress.ProgressBar = ProgressBar
    sys.modules[progress.__name__] = progress
    util.progress = progress

    module = importlib.import_module("exllamav3.modules.quant.exl3_lib.quantize")
    return module.quantize_exl3


def _select_experts(
    store: InterimKeepStore,
    capture: Path,
    layer: int,
    maximum: int,
    preferred: int | None = None,
) -> tuple[list[int], dict[int, float]]:
    ordered, traffic = routed_kept_experts(store, capture, layer)
    if preferred is not None:
        if preferred not in ordered:
            raise ValueError(f"expert {layer}.{preferred} is not retained")
        ordered = [preferred, *(expert for expert in ordered if expert != preferred)]
    return ordered[:maximum], traffic


def _signature(
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    input_index: torch.Tensor,
    output_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = w1[:, input_index].contiguous()
    signature = torch.cat(
        (gate, w3[:, input_index], w2[output_index, :].T), dim=1
    ).contiguous()
    return gate, signature


def _zero_structure(matrix: torch.Tensor) -> dict[str, float | int]:
    zero = matrix == 0
    return {
        "element_fraction": float(zero.float().mean()),
        "zero_rows": int(zero.all(dim=1).sum()),
        "zero_columns": int(zero.all(dim=0).sum()),
    }


def _candidate_metrics(tiles: torch.Tensor) -> dict[str, dict[str, float]]:
    candidates = {
        "scalar_affine_3bit": scalar_affine_3bit(tiles),
        "endpoint_ternary": endpoint_ternary(tiles),
        "endpoint_two_subset": endpoint_two_subset(tiles),
        "low_rank_r3_i7": quantized_low_rank(tiles, rank=3, factor_bits=7),
        "low_rank_r4_i5": quantized_low_rank(tiles, rank=4, factor_bits=5),
    }
    return {
        name: {
            "bits_per_weight": result.bits_per_weight,
            "nmse": normalized_mse(tiles, result.reconstruction),
        }
        for name, result in candidates.items()
    }


def _exl3_baseline(
    source: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    layer: int,
    expert: int,
    matrix: str,
    device: torch.device,
    exllamav3_root: Path,
    activation_inputs: torch.Tensor | None = None,
    activation_weights: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
) -> dict:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("the EXL3 baseline requires a CUDA device")
    quantize_exl3 = _load_exl3_quantizer(exllamav3_root)

    from kquant.pack.exl3 import BITS, MCG_MULT, SIGMA_REG, make_shared_h

    weight = source.T.contiguous().to(device)
    # The exllamav3 encoder finalizes/scales its input tensor in place. Keep a
    # canonical copy for output-space scoring after quantization.
    reference_weight = weight.clone()
    shared_h = make_shared_h(weight.shape[0], device, hessian)
    quant_args = {
        "K": BITS,
        "seed": layer * 1_000_000 + C.EXPERT_MATRICES.index(matrix),
        "sigma_reg": SIGMA_REG,
        "devices": [str(device)],
        "device_ratios": None,
        "apply_out_scales": False,
        "mcg": True,
    }
    quantized, proxy, tensors = quantize_exl3(
        weight, shared_h, quant_args, True, progress_str="EXL3 baseline <step>"
    )
    activation_metrics = None
    if activation_inputs is not None:
        inputs = activation_inputs.float().to(device)
        row_weights = (
            torch.ones(inputs.shape[0], device=device)
            if activation_weights is None
            else activation_weights.float().to(device)
        )
        reference_output = inputs @ reference_weight
        quantized_output = inputs @ quantized
        output_denominator = (
            reference_output.double().square().sum(dim=1) * row_weights.double()
        ).sum()
        output_error = (
            (reference_output - quantized_output).double().square().sum(dim=1)
            * row_weights.double()
        ).sum()
        activation_metrics = {
            "rows": int(inputs.shape[0]),
            "weighted_output_nmse": float(output_error / output_denominator),
        }
    reconstructed = quantized.T.cpu()
    source_tiles = gather_tiles(source, coordinates, tile_size=16)
    quantized_tiles = gather_tiles(reconstructed, coordinates, tile_size=16)
    full_nmse = normalized_mse(source, reconstructed)
    zero_columns = (source == 0).all(dim=0)
    zero_column_count = int(zero_columns.sum())
    zero_restored_nmse = full_nmse
    zero_column_error_fraction = 0.0
    if zero_column_count:
        restored = reconstructed.clone()
        restored[:, zero_columns] = 0
        zero_restored_nmse = normalized_mse(source, restored)
        zero_column_error_fraction = float(
            reconstructed[:, zero_columns].double().square().sum()
            / source.double().square().sum()
        )
        if activation_metrics is not None:
            restored_quantized = quantized.clone()
            restored_quantized[zero_columns.to(device)] = 0
            restored_output = inputs @ restored_quantized
            restored_error = (
                (reference_output - restored_output).double().square().sum(dim=1)
                * row_weights.double()
            ).sum()
            activation_metrics["weighted_output_nmse_with_zero_columns_restored"] = (
                float(restored_error / output_denominator)
            )
            del restored_quantized, restored_output
        del restored
    del weight, reference_weight, quantized, reconstructed, shared_h, tensors
    torch.cuda.empty_cache()
    result = {
        "bits_per_weight": 3.0,
        "proxy": float(proxy),
        "full_nmse": full_nmse,
        "zero_columns": zero_column_count,
        "zero_column_error_fraction": zero_column_error_fraction,
        "full_nmse_with_zero_columns_restored": zero_restored_nmse,
        "sampled_tile_nmse": normalized_mse(source_tiles, quantized_tiles),
        "mcg_multiplier": MCG_MULT,
        "hessian": "identity" if hessian is None else "captured_dense",
    }
    if activation_metrics is not None:
        result["captured_activation"] = activation_metrics
    return result


def _expert_activation_rows(
    capture: Path,
    layer: int,
    expert: int,
    matrix: str,
    split: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    samples = load_layer_samples(capture, layer - 1)
    split_value = {"all": None, "train": 0, "validation": 1}[split]
    if matrix == "w2":
        selected = samples.mid_experts == expert
        if split_value is not None:
            selected &= samples.mid_split == split_value
        if not torch.count_nonzero(selected):
            return None, None
        return samples.mid_values[selected], samples.mid_weights[selected]
    selected = samples.input_experts == expert
    if split_value is not None:
        selected &= (samples.input_split == split_value)[:, None]
    token_selected = selected.any(dim=1)
    if not torch.count_nonzero(token_selected):
        return None, None
    gate_square = torch.where(
        selected,
        samples.input_gates.square(),
        torch.zeros_like(samples.input_gates),
    ).sum(dim=1)
    return samples.input_values[token_selected], gate_square[token_selected]


def run(args: argparse.Namespace) -> dict:
    store = InterimKeepStore(args.checkpoint)
    experts, traffic = _select_experts(
        store,
        args.capture,
        args.layer,
        args.max_experts,
        args.baseline_expert,
    )
    if len(experts) < 2:
        raise ValueError("at least two retained experts are required")
    shape = C.EXPERT_SHAPES[args.matrix]
    coordinates = sample_tile_coordinates(
        shape, tile_size=16, count=args.tiles, seed=args.seed
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    input_index = torch.randperm(C.LATENT, generator=generator)[: args.signature_dims]
    output_index = torch.randperm(C.LATENT, generator=generator)[: args.signature_dims]

    matrices: dict[int, torch.Tensor] = {}
    tiles: dict[int, torch.Tensor] = {}
    gates: dict[int, torch.Tensor] = {}
    signatures: dict[int, torch.Tensor] = {}
    zero_structure: dict[str, dict[str, dict[str, float | int]]] = {}
    per_expert_candidates = {}
    for expert in experts:
        loaded = {
            matrix: store.load_matrix(args.layer, expert, matrix)
            for matrix in C.EXPERT_MATRICES
        }
        gate, signature = _signature(
            loaded["w1"], loaded["w3"], loaded["w2"], input_index, output_index
        )
        gates[expert] = gate
        signatures[expert] = signature
        matrix = loaded[args.matrix]
        matrices[expert] = matrix
        tiles[expert] = gather_tiles(matrix, coordinates, tile_size=16)
        per_expert_candidates[str(expert)] = _candidate_metrics(tiles[expert])
        zero_structure[str(expert)] = {
            name: _zero_structure(value) for name, value in loaded.items()
        }

    affine_pairs = []
    neuron_pairs = []
    for target in experts:
        for reference in experts:
            if target == reference:
                continue
            prediction = affine_prediction(tiles[target], tiles[reference])
            affine_pairs.append(
                {
                    "target": target,
                    "reference": reference,
                    "residual_ratio": normalized_mse(tiles[target], prediction),
                }
            )
            match = nearest_neuron_prediction(
                gates[target],
                gates[reference],
                signatures[target],
                signatures[reference],
            )
            neuron_pairs.append(
                {
                    "target": target,
                    "reference": reference,
                    **{
                        key: value
                        for key, value in match.items()
                        if key != "best_index"
                    },
                }
            )

    result = {
        "kind": "kquant_codec_transfer_experiment",
        "schema_version": 1,
        "source": {
            "checkpoint": str(store.root),
            "constraint": "interim EXL3 hybrid only; official model not loaded",
            "capture": str(args.capture.resolve()),
            "layer": args.layer,
            "matrix": args.matrix,
            "experts": experts,
            "captured_gate_square_mass": {
                str(expert): traffic.get(expert, 0.0) for expert in experts
            },
        },
        "sampling": {
            "tile_size": 16,
            "tiles": int(coordinates.shape[0]),
            "seed": args.seed,
            "signature_dimensions_per_branch": args.signature_dims,
        },
        "per_expert_candidates": per_expert_candidates,
        "zero_structure": zero_structure,
        "inter_expert_affine": {
            "pairs": affine_pairs,
            "best": min(affine_pairs, key=lambda value: value["residual_ratio"]),
            "median_residual_ratio": float(
                torch.tensor(
                    [value["residual_ratio"] for value in affine_pairs]
                ).median()
            ),
        },
        "permutation_motion": {
            "matching": "non-bijective nearest neighbor by sampled w1 cosine",
            "pairs": neuron_pairs,
            "best": min(neuron_pairs, key=lambda value: value["scaled_residual_ratio"]),
            "median_scaled_residual_ratio": float(
                torch.tensor(
                    [value["scaled_residual_ratio"] for value in neuron_pairs]
                ).median()
            ),
        },
    }
    if args.exl3_baseline:
        expert = experts[0]
        activation_inputs, activation_weights = _expert_activation_rows(
            args.capture,
            args.layer,
            expert,
            args.matrix,
            args.activation_split,
        )
        hessian = None
        if args.hessians is not None:
            h13, h2 = load_layer_hessians(args.hessians, args.layer)
            hessian = h2 if args.matrix == "w2" else h13
        result["exl3_baseline"] = {
            "expert": expert,
            **_exl3_baseline(
                matrices[expert],
                coordinates,
                layer=args.layer,
                expert=expert,
                matrix=args.matrix,
                device=torch.device(args.device),
                exllamav3_root=args.exllamav3_root,
                activation_inputs=activation_inputs,
                activation_weights=activation_weights,
                hessian=hessian,
            ),
        }
    return result


def main() -> None:
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
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--matrix", choices=C.EXPERT_MATRICES, default="w1")
    parser.add_argument("--max-experts", type=int, default=5)
    parser.add_argument("--baseline-expert", type=int)
    parser.add_argument("--tiles", type=int, default=512)
    parser.add_argument("--signature-dims", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exl3-baseline", action="store_true")
    parser.add_argument("--hessians", type=Path)
    parser.add_argument(
        "--activation-split",
        choices=("all", "train", "validation"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root", type=Path, default=Path("/home/luke/projects/exllamav3")
    )
    args = parser.parse_args()
    if not 1 <= args.layer <= C.NUM_MOE_LAYERS:
        parser.error("--layer must be 1..92")
    if args.max_experts < 2 or args.tiles <= 0 or args.signature_dims <= 0:
        parser.error("expert/tile/signature counts must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
