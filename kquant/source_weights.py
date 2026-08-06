"""Read-only, tensor-streamed access to the official K3 MXFP4 weights.

This module never instantiates the model.  Each call opens only the shard that
owns one packed expert matrix and its scale tensor, then either returns those
original bytes or dequantizes that single matrix on the requested device.  In
particular, CUDA callers transfer the compact uint8 representation first and
never materialize the full float32 matrix on CPU.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.io.hf_cache import CheckpointCache, resolve
from kquant.io.mxfp4 import dequant
from kquant.io.stream import load_tensor


def weight_name(layer: int, expert: int, matrix: str, part: str) -> str:
    """Return the canonical tensor name for one routed-expert weight part."""

    return (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts."
        f"{expert}.{matrix}.{part}"
    )


@dataclass(frozen=True)
class PackedMXFP4Matrix:
    """One official matrix in its original code-and-scale representation."""

    packed: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class OfficialMXFP4LayerStore:
    """Layer-scoped source view backed by already-open shard handles.

    Opening a large safetensors shard reparses its header.  All-expert passes
    would otherwise pay that cost for every packed tensor and scale plane.
    This view keeps only the owning shard handles open; ``get_tensor`` still
    materializes one compact source matrix at a time.
    """

    parent: "OfficialMXFP4Store"
    layer: int
    handles: dict[Path, object]
    experts: frozenset[int]
    matrices: frozenset[str]

    def load_packed_matrix(
        self, layer: int, expert: int, matrix: str
    ) -> PackedMXFP4Matrix:
        if layer != self.layer:
            raise ValueError(f"layer view is pinned to decoder layer {self.layer}")
        if expert not in self.experts or matrix not in self.matrices:
            raise ValueError("matrix is outside this layer view")
        packed_name = weight_name(layer, expert, matrix, "weight_packed")
        scale_name = weight_name(layer, expert, matrix, "weight_scale")
        packed_path = self.parent.cache.tensor_path(packed_name)
        scale_path = self.parent.cache.tensor_path(scale_name)
        if packed_path is None or scale_path is None:
            raise FileNotFoundError("official source matrix is incomplete")
        packed = self.handles[Path(packed_path)].get_tensor(packed_name)
        scale = self.handles[Path(scale_path)].get_tensor(scale_name)
        return self.parent._validated_packed_matrix(
            packed, scale, packed_name=packed_name, scale_name=scale_name, matrix=matrix
        )

    def load_matrix(
        self,
        layer: int,
        expert: int,
        matrix: str,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        raw = self.load_packed_matrix(layer, expert, matrix)
        return self.parent._dequant_packed_matrix(raw, device=device)


class OfficialMXFP4Store:
    """Stream individual source matrices from the official packed checkpoint."""

    def __init__(
        self,
        cache: CheckpointCache | None = None,
        *,
        repo_dir: str | Path | None = None,
        revision: str = C.REVISION,
    ):
        if cache is not None and repo_dir is not None:
            raise ValueError("repo_dir cannot be supplied with an explicit cache")
        self.cache = (
            resolve(repo_dir=repo_dir, revision=revision)
            if cache is None
            else cache
        )
        self.revision = revision if cache is None else self.cache.snapshot_dir.name
        self.root = Path(self.cache.snapshot_dir).resolve()
        if self.cache.missing_shards:
            raise FileNotFoundError(
                "official source checkpoint is incomplete: "
                f"{len(self.cache.missing_shards)} missing shards"
            )

    def available_experts(self, layer: int) -> list[int]:
        if layer not in C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer} is not a K3 MoE layer")
        return list(range(C.NUM_EXPERTS))

    def load_matrix(
        self,
        layer: int,
        expert: int,
        matrix: str,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Load and dequantize one matrix on ``device``.

        The source checkpoint is necessarily read through CPU, but only its
        original uint8 codes and scales live there when a CUDA device is
        requested.  ``device=None`` preserves the historical CPU behavior for
        validation and materialization callers.
        """

        raw = self.load_packed_matrix(layer, expert, matrix)
        return self._dequant_packed_matrix(raw, device=device)

    @staticmethod
    def _dequant_packed_matrix(
        raw: PackedMXFP4Matrix,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        target = torch.device("cpu") if device is None else torch.device(device)
        if target.type == "cpu":
            return dequant(raw.packed, raw.scale)
        packed = raw.packed.to(device=target, non_blocking=True)
        scale = raw.scale.to(device=target, non_blocking=True)
        return dequant(packed, scale).contiguous()

    def load_packed_matrix(
        self, layer: int, expert: int, matrix: str
    ) -> PackedMXFP4Matrix:
        """Stream one source matrix without changing its official MXFP4 bytes."""

        if layer not in C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer} is not a K3 MoE layer")
        if not 0 <= expert < C.NUM_EXPERTS:
            raise ValueError(f"expert must be in 0..{C.NUM_EXPERTS - 1}")
        if matrix not in C.EXPERT_MATRICES:
            raise ValueError(f"matrix must be one of {C.EXPERT_MATRICES}")
        packed_name = weight_name(layer, expert, matrix, "weight_packed")
        scale_name = weight_name(layer, expert, matrix, "weight_scale")
        packed = load_tensor(self.cache, packed_name)
        scale = load_tensor(self.cache, scale_name)
        return self._validated_packed_matrix(
            packed,
            scale,
            packed_name=packed_name,
            scale_name=scale_name,
            matrix=matrix,
        )

    @staticmethod
    def _validated_packed_matrix(
        packed: torch.Tensor,
        scale: torch.Tensor,
        *,
        packed_name: str,
        scale_name: str,
        matrix: str,
    ) -> PackedMXFP4Matrix:
        out_features, in_features = C.EXPERT_SHAPES[matrix]
        expected_packed = (out_features, in_features // 2)
        expected_scale = (out_features, in_features // C.MXFP4_BLOCK)
        if packed.dtype != torch.uint8 or tuple(packed.shape) != expected_packed:
            raise ValueError(
                f"{packed_name} has {packed.dtype} {tuple(packed.shape)}, expected "
                f"torch.uint8 {expected_packed}"
            )
        if scale.dtype != torch.uint8 or tuple(scale.shape) != expected_scale:
            raise ValueError(
                f"{scale_name} has {scale.dtype} {tuple(scale.shape)}, expected "
                f"torch.uint8 {expected_scale}"
            )
        if packed.device.type != "cpu" or scale.device.type != "cpu":
            raise ValueError("official packed matrices must be streamed through CPU")
        return PackedMXFP4Matrix(
            packed=packed.contiguous(),
            scale=scale.contiguous(),
        )

    @contextmanager
    def open_layer(
        self,
        layer: int,
        *,
        experts: tuple[int, ...] | None = None,
        matrices: tuple[str, ...] = C.EXPERT_MATRICES,
    ) -> Iterator[OfficialMXFP4LayerStore]:
        """Open every shard needed by a bounded all-expert layer pass once."""

        if layer not in C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer} is not a K3 MoE layer")
        selected_experts = (
            tuple(range(C.NUM_EXPERTS)) if experts is None else tuple(experts)
        )
        if (
            not selected_experts
            or len(set(selected_experts)) != len(selected_experts)
            or any(not 0 <= expert < C.NUM_EXPERTS for expert in selected_experts)
        ):
            raise ValueError("layer-view experts must be unique IDs in 0..895")
        if (
            not matrices
            or len(set(matrices)) != len(matrices)
            or any(matrix not in C.EXPERT_MATRICES for matrix in matrices)
        ):
            raise ValueError(f"matrices must be drawn from {C.EXPERT_MATRICES}")

        paths: set[Path] = set()
        for expert in selected_experts:
            for matrix in matrices:
                for part in ("weight_packed", "weight_scale"):
                    name = weight_name(layer, expert, matrix, part)
                    path = self.cache.tensor_path(name)
                    if path is None:
                        raise FileNotFoundError(f"shard for {name} not downloaded yet")
                    paths.add(Path(path))
        with ExitStack() as stack:
            handles = {
                path: stack.enter_context(safe_open(str(path), framework="pt"))
                for path in paths
            }
            yield OfficialMXFP4LayerStore(
                parent=self,
                layer=layer,
                handles=handles,
                experts=frozenset(selected_experts),
                matrices=frozenset(matrices),
            )

    def load_scale_plane(
        self, layer: int, expert: int, matrix: str
    ) -> torch.Tensor:
        """Stream only the official UE8M0 scale plane.

        Scale-codec studies should not pay the I/O or host-memory cost of the
        much larger packed nibble tensor when they never inspect its values.
        """

        if layer not in C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer} is not a K3 MoE layer")
        if not 0 <= expert < C.NUM_EXPERTS:
            raise ValueError(f"expert must be in 0..{C.NUM_EXPERTS - 1}")
        if matrix not in C.EXPERT_MATRICES:
            raise ValueError(f"matrix must be one of {C.EXPERT_MATRICES}")
        scale_name = weight_name(layer, expert, matrix, "weight_scale")
        scale = load_tensor(self.cache, scale_name)
        out_features, in_features = C.EXPERT_SHAPES[matrix]
        expected = (out_features, in_features // C.MXFP4_BLOCK)
        if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
            raise ValueError(
                f"{scale_name} has {scale.dtype} {tuple(scale.shape)}, expected "
                f"torch.uint8 {expected}"
            )
        if scale.device.type != "cpu":
            raise ValueError("official scale planes must be streamed through CPU")
        return scale.contiguous()

    def iter_layer_scale_planes(
        self,
        layer: int,
        *,
        experts: tuple[int, ...] | None = None,
        matrices: tuple[str, ...] = C.EXPERT_MATRICES,
    ) -> Iterator[tuple[int, str, torch.Tensor]]:
        """Yield one layer's scale planes while opening each shard only once.

        Kimi-K3 currently stores a complete routed-expert layer in one shard,
        but the implementation deliberately supports multiple shards.  This
        avoids reparsing a large safetensors header for every one of the 2,688
        scale tensors in an all-expert X4T cost scan.
        """

        if layer not in C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer} is not a K3 MoE layer")
        if not matrices or len(set(matrices)) != len(matrices):
            raise ValueError("scale-plane matrices must be nonempty and unique")
        if any(matrix not in C.EXPERT_MATRICES for matrix in matrices):
            raise ValueError(f"matrices must be drawn from {C.EXPERT_MATRICES}")
        selected_experts = (
            tuple(range(C.NUM_EXPERTS)) if experts is None else tuple(experts)
        )
        if (
            not selected_experts
            or len(set(selected_experts)) != len(selected_experts)
            or any(
                isinstance(expert, bool)
                or not isinstance(expert, int)
                or not 0 <= expert < C.NUM_EXPERTS
                for expert in selected_experts
            )
        ):
            raise ValueError("scale-plane experts must be unique IDs in 0..895")

        names = [
            (expert, matrix, weight_name(layer, expert, matrix, "weight_scale"))
            for expert in selected_experts
            for matrix in matrices
        ]
        paths: set[Path] = set()
        for _expert, _matrix, name in names:
            path = self.cache.tensor_path(name)
            if path is None:
                raise FileNotFoundError(f"shard for {name} not downloaded yet")
            paths.add(Path(path))

        with ExitStack() as stack:
            handles = {
                path: stack.enter_context(safe_open(str(path), framework="pt"))
                for path in paths
            }
            for expert, matrix, name in names:
                path = self.cache.tensor_path(name)
                if path is None:
                    raise AssertionError("validated scale tensor became unavailable")
                scale = handles[Path(path)].get_tensor(name)
                out_features, in_features = C.EXPERT_SHAPES[matrix]
                expected = (out_features, in_features // C.MXFP4_BLOCK)
                if scale.dtype != torch.uint8 or tuple(scale.shape) != expected:
                    raise ValueError(
                        f"{name} has {scale.dtype} {tuple(scale.shape)}, expected "
                        f"torch.uint8 {expected}"
                    )
                if scale.device.type != "cpu":
                    raise ValueError(
                        "official scale planes must be streamed through CPU"
                    )
                yield expert, matrix, scale.contiguous()
