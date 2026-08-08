"""Authenticated, mmap-backed access to Fruit routed-expert matrices.

Fruit training checkpoints use one of two native expert representations: three
expert-stacked tensors per MoE block, or three ordinary ``nn.Linear`` weights
per expert. This module authenticates and inventories the complete expert
representation before exposing any matrix, while keeping checkpoint storages
memory mapped and materializing only the requested matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open

FruitMatrix = Literal["w1", "w3", "w2"]
FruitRepresentation = Literal["stacked", "per_expert"]
FRUIT_EXPERT_MATRICES: tuple[FruitMatrix, ...] = ("w1", "w3", "w2")
_STACKED_NAMES: dict[FruitMatrix, str] = {
    "w1": "w_gate",
    "w3": "w_up",
    "w2": "w_down",
}
_PROJECTION_NAMES: dict[FruitMatrix, str] = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STACKED_KEY_RE = re.compile(r"^(?:layers\.\d+\.mlp|mtp_block\.mlp)\.w_[^.]+$")
_PER_EXPERT_KEY_RE = re.compile(
    r"^(?:layers\.\d+\.mlp|mtp_block\.mlp)\.experts\.\d+\..+$"
)
_INTEGRAL_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


@dataclass(frozen=True)
class FruitModelSpec:
    """Immutable source identity, geometry, and convention contract."""

    model_id: str
    checkpoint_sha256: str
    layers: tuple[int, ...]
    mtp_layer: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    trained_rope_theta: float
    serve_conv_v: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a nonempty string")
        if (
            not isinstance(self.checkpoint_sha256, str)
            or _SHA256_RE.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hex digits")
        if (
            not isinstance(self.layers, tuple)
            or not self.layers
            or len(set(self.layers)) != len(self.layers)
            or any(type(layer) is not int or layer < 0 for layer in self.layers)
        ):
            raise ValueError("layers must be a tuple of unique nonnegative integers")
        if tuple(sorted(self.layers)) != self.layers:
            raise ValueError("layers must be strictly increasing")
        if type(self.mtp_layer) is not int or self.mtp_layer < 0:
            raise ValueError("mtp_layer must be a nonnegative integer")
        if self.mtp_layer in self.layers:
            raise ValueError("mtp_layer must be distinct from ordinary MoE layers")
        for name, value in (
            ("num_experts", self.num_experts),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.trained_rope_theta, bool) or not isinstance(
            self.trained_rope_theta, (int, float)
        ):
            raise TypeError("trained_rope_theta must be a real number")
        if (
            not math.isfinite(float(self.trained_rope_theta))
            or float(self.trained_rope_theta) <= 0
        ):
            raise ValueError("trained_rope_theta must be a finite positive number")
        if self.serve_conv_v is not None and self.serve_conv_v != 2:
            raise ValueError("serve_conv_v must be None for legacy layout or 2")


FRUIT_ANNEALED_SPEC = FruitModelSpec(
    model_id="malaiwah/GLM-5.2-SIQ-Fruit",
    checkpoint_sha256=(
        "98ac7cb4f7799194424782b505d622069fecf4dbca5f5acb2658f2a66c3631f6"
    ),
    layers=tuple(range(3, 13)),
    mtp_layer=13,
    num_experts=256,
    hidden_size=1024,
    intermediate_size=512,
    trained_rope_theta=500_000.0,
    serve_conv_v=None,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _prefix(spec: FruitModelSpec, layer: int) -> str:
    if type(layer) is not int:
        raise ValueError("layer must be an integer")
    if layer == spec.mtp_layer:
        return "mtp_block.mlp"
    if layer in spec.layers:
        return f"layers.{layer}.mlp"
    supported = (*spec.layers, spec.mtp_layer)
    raise ValueError(f"layer must be one of {supported}")


def _matrix_shape(spec: FruitModelSpec, matrix: FruitMatrix) -> tuple[int, int]:
    if matrix in ("w1", "w3"):
        return (spec.intermediate_size, spec.hidden_size)
    return (spec.hidden_size, spec.intermediate_size)


def _state_dict(
    loaded: Any,
) -> tuple[Mapping[str, torch.Tensor], Literal["state_dict", "model"]]:
    if not isinstance(loaded, Mapping):
        raise TypeError("Fruit checkpoint must contain a state-dict mapping")
    if "model" in loaded:
        if set(loaded) != {"model"}:
            raise ValueError(
                "Fruit checkpoint wrapper must be exactly {'model': state_dict}"
            )
        if not isinstance(loaded["model"], Mapping):
            raise TypeError("Fruit checkpoint model must be a state-dict mapping")
        state = loaded["model"]
        container: Literal["state_dict", "model"] = "model"
    else:
        state = loaded
        container = "state_dict"
    if not state:
        raise ValueError("Fruit checkpoint state dict is empty")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("Fruit state dict must map string keys only to tensors")
    return state, container


def _expected_keys(
    spec: FruitModelSpec, representation: FruitRepresentation
) -> set[str]:
    prefixes = [f"layers.{layer}.mlp" for layer in spec.layers]
    prefixes.append("mtp_block.mlp")
    if representation == "stacked":
        return {
            f"{prefix}.{_STACKED_NAMES[matrix]}"
            for prefix in prefixes
            for matrix in FRUIT_EXPERT_MATRICES
        }
    return {
        f"{prefix}.experts.{expert}.{_PROJECTION_NAMES[matrix]}.weight"
        for prefix in prefixes
        for expert in range(spec.num_experts)
        for matrix in FRUIT_EXPERT_MATRICES
    }


def _format_key_difference(missing: set[str], unexpected: set[str]) -> str:
    parts: list[str] = []
    if missing:
        sample = ", ".join(sorted(missing)[:3])
        parts.append(f"missing {len(missing)} ({sample})")
    if unexpected:
        sample = ", ".join(sorted(unexpected)[:3])
        parts.append(f"unexpected {len(unexpected)} ({sample})")
    return "; ".join(parts)


def _validate_inventory(
    state: Mapping[str, torch.Tensor], spec: FruitModelSpec
) -> tuple[FruitRepresentation, tuple[str, ...]]:
    stacked = {key for key in state if _STACKED_KEY_RE.fullmatch(key)}
    per_expert = {key for key in state if _PER_EXPERT_KEY_RE.fullmatch(key)}
    if stacked and per_expert:
        raise ValueError("Fruit expert inventory mixes stacked and per-expert layouts")
    if not stacked and not per_expert:
        raise ValueError("Fruit checkpoint has no supported expert tensor inventory")
    representation: FruitRepresentation = "stacked" if stacked else "per_expert"
    actual = stacked if stacked else per_expert
    expected = _expected_keys(spec, representation)
    if actual != expected:
        raise ValueError(
            "Fruit expert tensor inventory is not exact: "
            + _format_key_difference(expected - actual, actual - expected)
        )

    dtypes: set[str] = set()
    native_to_matrix = {
        **{native: matrix for matrix, native in _STACKED_NAMES.items()},
        **{native: matrix for matrix, native in _PROJECTION_NAMES.items()},
    }
    for key in sorted(expected):
        tensor = state[key]
        if not tensor.is_floating_point():
            raise ValueError(f"{key} has non-floating dtype {tensor.dtype}")
        if tensor.device.type != "cpu":
            raise ValueError(f"{key} must be mmap-backed on CPU")
        if representation == "stacked":
            native_name = key.rsplit(".", 1)[1]
            matrix = native_to_matrix[native_name]
            expected_shape = (spec.num_experts, *_matrix_shape(spec, matrix))
        else:
            native_name = key.rsplit(".", 2)[1]
            matrix = native_to_matrix[native_name]
            expected_shape = _matrix_shape(spec, matrix)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{key} has shape {tuple(tensor.shape)}, expected {expected_shape}"
            )
        dtypes.add(str(tensor.dtype))
    return representation, tuple(sorted(dtypes))


def _marker_scalar(state: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    marker = state[name]
    if tuple(marker.shape) != (1,) or marker.device.type != "cpu":
        raise ValueError(f"{name} must be a one-element CPU tensor")
    return marker


def _validate_conventions(
    state: Mapping[str, torch.Tensor], spec: FruitModelSpec
) -> dict[str, object]:
    has_version = "serve_conv_v" in state
    has_theta = "rope_theta_trained" in state
    if has_version != has_theta:
        raise ValueError(
            "serve_conv_v and rope_theta_trained must be present or absent together"
        )
    if not has_version:
        if spec.serve_conv_v is not None:
            raise ValueError("native Fruit source is missing convention markers")
        return {
            "kind": "legacy",
            "trained_rope_theta": float(spec.trained_rope_theta),
            "trained_rope_theta_provenance": "model_spec",
            "serve_conv_v": None,
        }

    if spec.serve_conv_v is None:
        raise ValueError(
            "legacy Fruit source unexpectedly has native convention markers"
        )
    version = _marker_scalar(state, "serve_conv_v")
    if version.dtype not in _INTEGRAL_DTYPES:
        raise ValueError("serve_conv_v must have an integral dtype")
    version_value = int(version.item())
    if version_value != 2 or version_value != spec.serve_conv_v:
        raise ValueError("serve_conv_v must equal the spec-pinned supported version 2")

    theta = _marker_scalar(state, "rope_theta_trained")
    if not theta.is_floating_point():
        raise ValueError("rope_theta_trained must have a floating dtype")
    theta_value = float(theta.item())
    if not math.isfinite(theta_value) or theta_value <= 0:
        raise ValueError("rope_theta_trained must be finite and positive")
    if theta_value != float(spec.trained_rope_theta):
        raise ValueError("rope_theta_trained does not match the model spec")
    return {
        "kind": "native",
        "trained_rope_theta": theta_value,
        "trained_rope_theta_provenance": (
            "checkpoint_marker_validated_against_model_spec"
        ),
        "serve_conv_v": version_value,
    }


class FruitCheckpointStore:
    """Authenticated mmap-backed store for individual Fruit expert matrices."""

    def __init__(
        self,
        path: str | Path,
        *,
        spec: FruitModelSpec = FRUIT_ANNEALED_SPEC,
        expected_sha256: str | None = None,
    ) -> None:
        supplied_path = Path(path)
        if supplied_path.is_symlink():
            raise ValueError("Fruit checkpoint path must not be a symbolic link")
        if supplied_path.suffix != ".pt":
            raise ValueError("Fruit checkpoint path must have a .pt suffix")
        try:
            resolved = supplied_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Fruit checkpoint not found: {supplied_path}"
            ) from exc
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Fruit checkpoint path must name a regular file")

        expected = (
            spec.checkpoint_sha256 if expected_sha256 is None else expected_sha256
        )
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            raise ValueError("expected_sha256 must be 64 lowercase hex digits")
        if expected != spec.checkpoint_sha256:
            raise ValueError(
                "expected_sha256 does not identify the supplied model spec"
            )
        actual = _sha256(resolved)
        if actual != expected:
            raise ValueError(
                f"Fruit checkpoint SHA-256 mismatch: got {actual}, expected {expected}"
            )

        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        after = resolved.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(
                "Fruit checkpoint changed while it was being authenticated"
            )

        state, container = _state_dict(loaded)
        conventions = _validate_conventions(state, spec)
        representation, dtypes = _validate_inventory(state, spec)

        self.path = resolved
        self._file_identity = identity_after
        self.spec = spec
        self.representation = representation
        self._state = state
        self.evidence: dict[str, object] = {
            "checkpoint_bytes": before.st_size,
            "checkpoint_filename": resolved.name,
            "checkpoint_sha256": actual,
            "conventions": conventions,
            "expert_inventory": {
                "dtypes": list(dtypes),
                "matrices": list(FRUIT_EXPERT_MATRICES),
                "representation": representation,
                "tensor_count": len(_expected_keys(spec, representation)),
            },
            "geometry": {
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "layers": list(spec.layers),
                "mtp_layer": spec.mtp_layer,
                "num_experts": spec.num_experts,
            },
            "kind": "kquant-fruit-source-preflight",
            "model_id": spec.model_id,
            "path": str(resolved),
            "source_container": container,
            "status": "pass",
            "version": 1,
        }

    def _assert_source_unchanged(self) -> None:
        try:
            current = self.path.stat()
        except OSError as exc:
            raise ValueError(
                "Fruit checkpoint became unavailable after authentication"
            ) from exc
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identity != self._file_identity:
            raise ValueError("Fruit checkpoint changed after authentication")

    def load_matrix(
        self,
        layer: int,
        expert: int,
        matrix: str,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Materialize one logical ``[out, in]`` matrix as contiguous float32."""

        prefix = _prefix(self.spec, layer)
        if type(expert) is not int or not 0 <= expert < self.spec.num_experts:
            raise ValueError(f"expert must be in 0..{self.spec.num_experts - 1}")
        if matrix not in FRUIT_EXPERT_MATRICES:
            raise ValueError(f"matrix must be one of {FRUIT_EXPERT_MATRICES}")
        target = torch.device("cpu") if device is None else torch.device(device)
        self._assert_source_unchanged()
        if self.representation == "stacked":
            source = self._state[f"{prefix}.{_STACKED_NAMES[matrix]}"][expert]
        else:
            source = self._state[
                f"{prefix}.experts.{expert}.{_PROJECTION_NAMES[matrix]}.weight"
            ]
        result = source.to(device=target, dtype=torch.float32, copy=True).contiguous()
        self._assert_source_unchanged()
        expected_shape = _matrix_shape(self.spec, matrix)
        if tuple(result.shape) != expected_shape:
            raise AssertionError("validated Fruit matrix changed shape")
        return result


class FruitSafetensorsStore:
    """Authenticated HF safetensors fallback for Fruit expert matrices."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_manifest_sha256: str,
        spec: FruitModelSpec = FRUIT_ANNEALED_SPEC,
    ) -> None:
        supplied_root = Path(root)
        if supplied_root.is_symlink():
            raise ValueError("Fruit safetensors root must not be a symbolic link")
        try:
            resolved = supplied_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Fruit safetensors root not found: {supplied_root}"
            ) from exc
        if not resolved.is_dir():
            raise ValueError("Fruit safetensors root must name a directory")
        if (
            not isinstance(expected_manifest_sha256, str)
            or _SHA256_RE.fullmatch(expected_manifest_sha256) is None
        ):
            raise ValueError("expected_manifest_sha256 must be 64 lowercase hex digits")

        manifest_path = resolved / "MANIFEST.sha256"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        actual_manifest_sha256 = _sha256(manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "Fruit safetensors manifest SHA-256 mismatch: "
                f"got {actual_manifest_sha256}, "
                f"expected {expected_manifest_sha256}"
            )
        entries: dict[str, str] = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"malformed Fruit safetensors manifest line: {line!r}")
            digest, filename = fields
            if (
                _SHA256_RE.fullmatch(digest) is None
                or Path(filename).name != filename
                or filename in entries
            ):
                raise ValueError(f"invalid Fruit safetensors manifest line: {line!r}")
            entries[filename] = digest
        required = {"config.json", "model.safetensors.index.json"}
        if not required.issubset(entries):
            raise ValueError("Fruit safetensors manifest omits config or tensor index")

        file_identities: dict[str, tuple[int, int, int, int, int]] = {}
        for filename, expected_sha256 in entries.items():
            path = resolved / filename
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
            before = path.stat()
            actual_sha256 = _sha256(path)
            after = path.stat()
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity:
                raise ValueError(
                    "Fruit safetensors file changed while it was authenticated"
                )
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Fruit safetensors SHA-256 mismatch for {filename}: "
                    f"got {actual_sha256}, expected {expected_sha256}"
                )
            file_identities[filename] = after_identity

        config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
        expected_config = {
            "dtype": "bfloat16",
            "hidden_size": spec.hidden_size,
            "moe_intermediate_size": spec.intermediate_size,
            "n_routed_experts": spec.num_experts,
            "num_hidden_layers": spec.mtp_layer,
        }
        if not isinstance(config, dict) or any(
            config.get(name) != value for name, value in expected_config.items()
        ):
            raise ValueError("Fruit safetensors model geometry mismatch")
        rope_theta = config.get("rope_theta")
        if rope_theta is None:
            rope_parameters = config.get("rope_parameters")
            if isinstance(rope_parameters, dict):
                rope_theta = rope_parameters.get("rope_theta")
        if rope_theta != spec.trained_rope_theta:
            raise ValueError("Fruit safetensors trained rope theta mismatch")

        index = json.loads(
            (resolved / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise TypeError("Fruit safetensors index has no weight_map")
        weight_map = index["weight_map"]
        if any(
            not isinstance(name, str)
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or filename not in entries
            for name, filename in weight_map.items()
        ):
            raise ValueError("Fruit safetensors index has invalid entries")

        keys = {
            (layer, expert, matrix): (
                f"model.layers.{layer}.mlp.experts.{expert}."
                f"{_PROJECTION_NAMES[matrix]}.weight"
            )
            for layer in (*spec.layers, spec.mtp_layer)
            for expert in range(spec.num_experts)
            for matrix in FRUIT_EXPERT_MATRICES
        }
        expected_expert_keys = set(keys.values())
        actual_expert_keys = {name for name in weight_map if ".mlp.experts." in name}
        if actual_expert_keys != expected_expert_keys:
            raise ValueError(
                "Fruit safetensors expert inventory is not exact: "
                + _format_key_difference(
                    expected_expert_keys - actual_expert_keys,
                    actual_expert_keys - expected_expert_keys,
                )
            )

        names_by_file: dict[str, list[tuple[str, FruitMatrix]]] = {}
        for (_, _, matrix), name in keys.items():
            names_by_file.setdefault(weight_map[name], []).append((name, matrix))
        for filename, names in names_by_file.items():
            with safe_open(resolved / filename, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name, matrix in names:
                    if name not in available:
                        raise ValueError(
                            f"Fruit safetensors shard {filename} omits {name}"
                        )
                    tensor = handle.get_slice(name)
                    if tensor.get_dtype() != "BF16":
                        raise ValueError(
                            f"{name} has dtype {tensor.get_dtype()}, expected BF16"
                        )
                    if tuple(tensor.get_shape()) != _matrix_shape(spec, matrix):
                        raise ValueError(
                            f"{name} has shape {tuple(tensor.get_shape())}, "
                            f"expected {_matrix_shape(spec, matrix)}"
                        )

        self.path = resolved
        self.spec = spec
        self.representation: FruitRepresentation = "per_expert"
        self._entries = entries
        self._file_identities = file_identities
        self._weight_map: dict[str, str] = weight_map
        self._keys = keys
        self.evidence: dict[str, object] = {
            "checkpoint_sha256": spec.checkpoint_sha256,
            "conventions": {
                "kind": "legacy",
                "trained_rope_theta": float(spec.trained_rope_theta),
                "trained_rope_theta_provenance": "model_spec",
                "serve_conv_v": spec.serve_conv_v,
            },
            "expert_inventory": {
                "dtypes": ["torch.bfloat16"],
                "matrices": list(FRUIT_EXPERT_MATRICES),
                "representation": "per_expert",
                "tensor_count": len(keys),
            },
            "geometry": {
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "layers": list(spec.layers),
                "mtp_layer": spec.mtp_layer,
                "num_experts": spec.num_experts,
            },
            "kind": "kquant-fruit-source-preflight",
            "model_id": spec.model_id,
            "path": str(resolved),
            "safetensors_manifest_sha256": actual_manifest_sha256,
            "source_container": "hf_bf16_safetensors",
            "status": "pass",
            "version": 1,
        }

    def _assert_file_unchanged(self, filename: str) -> None:
        path = self.path / filename
        try:
            current = path.stat()
        except OSError as exc:
            raise ValueError(
                "Fruit safetensors source became unavailable after authentication"
            ) from exc
        identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if identity != self._file_identities[filename]:
            raise ValueError("Fruit safetensors source changed after authentication")

    def load_matrix(
        self,
        layer: int,
        expert: int,
        matrix: str,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Materialize one HF expert matrix as contiguous float32."""

        _prefix(self.spec, layer)
        if type(expert) is not int or not 0 <= expert < self.spec.num_experts:
            raise ValueError(f"expert must be in 0..{self.spec.num_experts - 1}")
        if matrix not in FRUIT_EXPERT_MATRICES:
            raise ValueError(f"matrix must be one of {FRUIT_EXPERT_MATRICES}")
        key = self._keys[(layer, expert, matrix)]
        filename = self._weight_map[key]
        self._assert_file_unchanged(filename)
        with safe_open(self.path / filename, framework="pt", device="cpu") as handle:
            source = handle.get_tensor(key)
        target = torch.device("cpu") if device is None else torch.device(device)
        result = source.to(device=target, dtype=torch.float32, copy=True).contiguous()
        self._assert_file_unchanged(filename)
        if tuple(result.shape) != _matrix_shape(self.spec, matrix):
            raise AssertionError("validated Fruit matrix changed shape")
        return result


def preflight_fruit_checkpoint(
    path: str | Path,
    spec: FruitModelSpec,
    expected_sha256: str,
) -> dict[str, object]:
    """Authenticate and inventory a Fruit checkpoint as JSON-safe evidence."""

    store = FruitCheckpointStore(
        path,
        spec=spec,
        expected_sha256=expected_sha256,
    )
    return store.evidence
