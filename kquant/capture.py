"""Merge graph-safe vLLM/B12X calibration captures.

The runtime artifact is deliberately rank-sharded.  Routing is recorded once
on TP rank zero, input-channel moments are expert-sharded across TP ranks, and
middle-channel moments are channel-sharded exactly like the TP-local w2 input.
Raw input and middle samples remain append-only so a true dense Hessian can be
constructed without an online TP collective.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_torch_file
from safetensors.torch import save_file as save_torch_file

from kquant import constants as C
from kquant.io.stats_bundle import save_stats_bundle

CAPTURE_SCHEMA_VERSION = 2
HESSIAN_SCHEMA_VERSION = 2
SAMPLE_CACHE_SCHEMA_VERSION = 1
_RANK_RE = re.compile(r"rank-(\d+)$")


@dataclass(frozen=True)
class CaptureGeometry:
    num_layers: int = C.NUM_MOE_LAYERS
    num_experts: int = C.NUM_EXPERTS
    input_size: int = C.LATENT
    intermediate_size: int = C.EXPERT_INTER
    top_k: int = C.TOP_K

    @classmethod
    def from_manifest(cls, manifest: dict) -> CaptureGeometry:
        geometry = manifest.get("geometry", {})
        return cls(
            num_layers=int(geometry.get("num_layers", C.NUM_MOE_LAYERS)),
            num_experts=int(geometry.get("num_experts", C.NUM_EXPERTS)),
            input_size=int(geometry.get("input_size", C.LATENT)),
            intermediate_size=int(geometry.get("intermediate_size", C.EXPERT_INTER)),
            top_k=int(geometry.get("top_k", C.TOP_K)),
        )


@dataclass(frozen=True)
class RankCapture:
    rank: int
    path: Path
    manifest: dict
    stats: dict[str, torch.Tensor]

    @property
    def input_expert_range(self) -> tuple[int, int]:
        begin, end = self.manifest["input_expert_range"]
        return int(begin), int(end)

    @property
    def intermediate_channel_range(self) -> tuple[int, int]:
        begin, end = self.manifest["intermediate_channel_range"]
        return int(begin), int(end)

    def sample_parts(self) -> list[Path]:
        return sorted((self.path / "samples").glob("part-*.safetensors"))


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"capture is missing {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def load_capture(
    path: str | Path, *, load_stats: bool = True
) -> tuple[dict, CaptureGeometry, list[RankCapture]]:
    path = Path(path)
    root = _read_json(path / "manifest.json")
    version = int(root.get("schema_version", 0))
    if version != CAPTURE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported kqcapture schema {version}; expected {CAPTURE_SCHEMA_VERSION}"
        )
    if not bool(root.get("complete", False)):
        raise ValueError(
            "capture is not finalized; create its finalize sentinel and send one "
            "last request before merging"
        )
    geometry = CaptureGeometry.from_manifest(root)
    if geometry == CaptureGeometry():
        if root.get("model") != C.MODEL_ID or root.get("revision") != C.REVISION:
            raise ValueError(
                "production K3 capture model/revision does not match kquant constants"
            )
        source = root.get("source")
        if source not in {
            "official_mxfp4_normal_w4a16",
            "interim_exl3_3p09_hybrid",
        }:
            raise ValueError(
                "production K3 calibration has an unsupported teacher source: "
                f"{source!r}"
            )
        if source == "interim_exl3_3p09_hybrid" and not root.get(
            "teacher_checkpoint"
        ):
            raise ValueError("interim EXL3 capture is missing teacher_checkpoint")
    rank_dirs = sorted(
        p for p in path.iterdir() if p.is_dir() and _RANK_RE.match(p.name)
    )
    if not rank_dirs:
        raise ValueError(f"capture {path} contains no rank-* directories")
    captures: list[RankCapture] = []
    for rank_dir in rank_dirs:
        match = _RANK_RE.match(rank_dir.name)
        assert match is not None
        rank = int(match.group(1))
        manifest = _read_json(rank_dir / "manifest.json")
        if int(manifest.get("schema_version", 0)) != version:
            raise ValueError(f"rank {rank} capture schema differs from root")
        if int(manifest.get("rank", -1)) != rank:
            raise ValueError(f"rank manifest mismatch in {rank_dir}")
        if not bool(manifest.get("complete", False)):
            raise ValueError(f"capture rank {rank} is not finalized")
        input_dropped = int(manifest.get("input_samples_dropped", 0))
        mid_dropped = int(manifest.get("mid_samples_dropped", 0))
        if input_dropped or mid_dropped:
            raise ValueError(
                f"capture rank {rank} overflowed its sample ring "
                f"(input={input_dropped}, mid={mid_dropped}); rerun with a larger "
                "VLLM_KQUANT_SAMPLE_CAPACITY"
            )
        stats_path = rank_dir / "stats.safetensors"
        if not stats_path.exists():
            raise ValueError(f"capture rank {rank} is missing {stats_path.name}")
        captures.append(
            RankCapture(
                rank=rank,
                path=rank_dir,
                manifest=manifest,
                stats=(
                    load_torch_file(str(stats_path), device="cpu")
                    if load_stats
                    else {}
                ),
            )
        )

    expected_world = int(root.get("tp_world_size", len(captures)))
    actual_ranks = [capture.rank for capture in captures]
    if actual_ranks != list(range(expected_world)):
        raise ValueError(
            f"capture ranks {actual_ranks} do not cover TP world 0..{expected_world - 1}"
        )
    for capture in captures:
        if int(capture.manifest.get("tp_world_size", expected_world)) != expected_world:
            raise ValueError(f"rank {capture.rank} TP world size differs from root")
        if capture.manifest.get("run_id") != root.get("run_id"):
            raise ValueError(f"rank {capture.rank} run_id differs from root")
        if geometry == CaptureGeometry():
            expected_layers = list(C.MOE_LAYERS)
            if capture.manifest.get("registered_decoder_layers") != expected_layers:
                raise ValueError(
                    f"rank {capture.rank} did not register all K3 MoE layers"
                )
    return root, geometry, captures


def _require_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    *,
    key: str,
    rank: int,
) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"rank {rank} {key} has shape {tuple(tensor.shape)}, expected {expected}"
        )


def _require(stats: dict[str, torch.Tensor], key: str, rank: int) -> torch.Tensor:
    try:
        return stats[key]
    except KeyError as exc:
        raise ValueError(f"rank {rank} is missing stats tensor {key!r}") from exc


def _normalized_moment(
    square_sum: torch.Tensor, weight_sum: torch.Tensor
) -> torch.Tensor:
    denominator = weight_sum.to(torch.float32).unsqueeze(-1)
    return torch.where(
        denominator > 0,
        square_sum.to(torch.float32)
        / denominator.clamp_min_(torch.finfo(torch.float32).tiny),
        torch.zeros_like(square_sum, dtype=torch.float32),
    )


def merge_stats(
    capture_path: str | Path,
    output_path: str | Path,
    *,
    extra_manifest: dict | None = None,
) -> Path:
    """Merge one TP-only ``.kqcapture`` into a schema-v2 ``.kqstats``."""

    root, geometry, ranks = load_capture(capture_path)
    l, e, kin, kmid = (
        geometry.num_layers,
        geometry.num_experts,
        geometry.input_size,
        geometry.intermediate_size,
    )

    route_keys = ("tokens_routed", "gate_sum", "gate_sq_sum")
    route_arrays: dict[str, torch.Tensor] = {}
    for key in route_keys:
        source = _require(ranks[0].stats, key, ranks[0].rank)
        _require_shape(source, (l, e), key=key, rank=ranks[0].rank)
        route_arrays[key] = source.clone()
        for capture in ranks[1:]:
            other = capture.stats.get(key)
            if other is not None and torch.count_nonzero(other).item() != 0:
                raise ValueError(
                    f"rank {capture.rank} contains nonzero {key}; routing must be rank-0-only"
                )

    in_sq = torch.zeros((l, e, kin), dtype=torch.float32)
    in_weight = torch.zeros((l, e), dtype=torch.float64)
    in_count = torch.zeros((l, e), dtype=torch.int64)
    covered_experts = torch.zeros(e, dtype=torch.bool)
    for capture in ranks:
        begin, end = capture.input_expert_range
        if not 0 <= begin <= end <= e:
            raise ValueError(
                f"rank {capture.rank} has invalid input expert range [{begin}, {end})"
            )
        if torch.any(covered_experts[begin:end]):
            raise ValueError("input expert ranges overlap")
        covered_experts[begin:end] = True
        width = end - begin
        square = _require(capture.stats, "act_in_sq_sum", capture.rank)
        weight = _require(capture.stats, "act_in_weight_sum", capture.rank)
        count = _require(capture.stats, "act_in_count", capture.rank)
        _require_shape(square, (l, width, kin), key="act_in_sq_sum", rank=capture.rank)
        _require_shape(weight, (l, width), key="act_in_weight_sum", rank=capture.rank)
        _require_shape(count, (l, width), key="act_in_count", rank=capture.rank)
        in_sq[:, begin:end].copy_(square)
        in_weight[:, begin:end].copy_(weight)
        in_count[:, begin:end].copy_(count)
    if not torch.all(covered_experts):
        missing = torch.nonzero(~covered_experts).flatten().tolist()
        raise ValueError(
            f"input expert ranges do not cover all experts: {missing[:16]}"
        )

    mid_sq = torch.zeros((l, e, kmid), dtype=torch.float32)
    # Counts and weight sums are route statistics and therefore must agree on
    # every channel shard.  Keep one copy after validating the replicas.
    mid_weight: torch.Tensor | None = None
    mid_count: torch.Tensor | None = None
    covered_channels = torch.zeros(kmid, dtype=torch.bool)
    for capture in ranks:
        begin, end = capture.intermediate_channel_range
        if not 0 <= begin <= end <= kmid:
            raise ValueError(
                f"rank {capture.rank} has invalid intermediate range [{begin}, {end})"
            )
        if torch.any(covered_channels[begin:end]):
            raise ValueError("intermediate channel ranges overlap")
        covered_channels[begin:end] = True
        width = end - begin
        square = _require(capture.stats, "act_mid_sq_sum", capture.rank)
        weight = _require(capture.stats, "act_mid_weight_sum", capture.rank)
        count = _require(capture.stats, "act_mid_count", capture.rank)
        _require_shape(square, (l, e, width), key="act_mid_sq_sum", rank=capture.rank)
        _require_shape(weight, (l, e), key="act_mid_weight_sum", rank=capture.rank)
        _require_shape(count, (l, e), key="act_mid_count", rank=capture.rank)
        mid_sq[:, :, begin:end].copy_(square)
        if mid_weight is None:
            mid_weight = weight.to(torch.float64).clone()
            mid_count = count.to(torch.int64).clone()
        else:
            if not torch.allclose(
                mid_weight, weight.to(torch.float64), rtol=0, atol=1e-9
            ):
                raise ValueError(f"rank {capture.rank} act_mid_weight_sum differs")
            assert mid_count is not None
            if not torch.equal(mid_count, count.to(torch.int64)):
                raise ValueError(f"rank {capture.rank} act_mid_count differs")
    if not torch.all(covered_channels):
        missing = torch.nonzero(~covered_channels).flatten().tolist()
        raise ValueError(
            f"intermediate channel ranges do not cover all channels: {missing[:16]}"
        )
    assert mid_weight is not None and mid_count is not None

    arrays = {
        **route_arrays,
        "act_m2_in": _normalized_moment(in_sq, in_weight),
        "act_m2_mid": _normalized_moment(mid_sq, mid_weight),
        "act_weight_sum_in": in_weight,
        "act_weight_sum_mid": mid_weight,
        "act_count_in": in_count,
        "act_count_mid": mid_count,
    }
    manifest = {
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "source_capture": str(Path(capture_path).resolve()),
        "run_id": root.get("run_id"),
        "tp_world_size": len(ranks),
        "weighting": "gate_square",
        "normalization": "sum(gate_weight^2 * activation^2) / sum(gate_weight^2)",
        "sampling": root.get("sampling", {}),
        "corpus": root.get("corpus"),
        "executed_tokens": int(root.get("executed_tokens", 0)),
        "teacher_source": root.get("source"),
        "teacher_checkpoint": root.get("teacher_checkpoint"),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    # StatsBundle validates production K3 dimensions.  Small synthetic
    # captures are supported by the lower-level helpers/tests, but are not a
    # valid production .kqstats artifact.
    if geometry != CaptureGeometry():
        raise ValueError(
            f"cannot emit production kqstats for non-K3 geometry {geometry}"
        )
    return save_stats_bundle(
        output_path, arrays, "vllm-b12x-kquant-capture", manifest
    )


@dataclass(frozen=True)
class _SamplePart:
    values: torch.Tensor
    weights: torch.Tensor
    observations: torch.Tensor
    expert: torch.Tensor | None = None
    route_experts: torch.Tensor | None = None
    route_gates: torch.Tensor | None = None
    split: torch.Tensor | None = None
    routed_latent: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerSamples:
    """One zero-based MoE-layer row of route-aware raw calibration samples."""

    input_values: torch.Tensor
    input_weights: torch.Tensor
    input_observations: torch.Tensor
    input_experts: torch.Tensor
    input_gates: torch.Tensor
    input_split: torch.Tensor
    routed_latent: torch.Tensor
    mid_values: torch.Tensor
    mid_weights: torch.Tensor
    mid_observations: torch.Tensor
    mid_experts: torch.Tensor
    mid_split: torch.Tensor


_SampleIndex = dict[int, list[_SamplePart]]


@dataclass
class LayerSampleIndex:
    """Compact reusable raw-sample index for a selected set of MoE layers."""

    geometry: CaptureGeometry
    ranks: list[RankCapture]
    input_index: _SampleIndex
    mid_indices: list[_SampleIndex]
    remaining: set[int]

    def pop(self, layer: int) -> LayerSamples:
        if layer not in self.remaining:
            raise ValueError(f"MoE layer row {layer} is absent or already consumed")
        input_samples = _load_input_layer_samples(self.input_index, layer)
        mid_samples = _load_mid_layer_samples(
            self.ranks,
            self.mid_indices,
            layer,
            self.geometry.intermediate_size,
        )
        self.remaining.remove(layer)
        assert input_samples.route_experts is not None
        assert input_samples.route_gates is not None
        assert input_samples.split is not None
        assert input_samples.routed_latent is not None
        assert mid_samples.expert is not None
        assert mid_samples.split is not None
        return LayerSamples(
            input_values=input_samples.values,
            input_weights=input_samples.weights,
            input_observations=input_samples.observations,
            input_experts=input_samples.route_experts,
            input_gates=input_samples.route_gates,
            input_split=input_samples.split,
            routed_latent=input_samples.routed_latent,
            mid_values=mid_samples.values,
            mid_weights=mid_samples.weights,
            mid_observations=mid_samples.observations,
            mid_experts=mid_samples.expert,
            mid_split=mid_samples.split,
        )


@dataclass
class LayerSampleCacheIndex:
    """Lazy reader for a layer-indexed, input-only calibration cache."""

    path: Path
    geometry: CaptureGeometry
    entries: dict[str, dict]
    remaining: set[int]

    def pop(self, layer: int) -> LayerSamples:
        if layer not in self.remaining:
            raise ValueError(f"MoE layer row {layer} is absent or already consumed")
        entry = self.entries.get(str(layer + 1))
        if entry is None:
            raise ValueError(f"sample cache has no MoE layer row {layer}")
        tensors = load_torch_file(str(self.path / entry["file"]), device="cpu")
        required = (
            "input.values",
            "input.weight",
            "input.observation",
            "input.experts",
            "input.gates",
            "input.split",
            "input.routed_latent",
        )
        missing = [key for key in required if key not in tensors]
        if missing:
            raise ValueError(f"cached layer {layer} is missing {missing}")
        self.remaining.remove(layer)
        return LayerSamples(
            input_values=tensors["input.values"],
            input_weights=tensors["input.weight"],
            input_observations=tensors["input.observation"],
            input_experts=tensors["input.experts"],
            input_gates=tensors["input.gates"],
            input_split=tensors["input.split"],
            routed_latent=tensors["input.routed_latent"],
            # Teacher middle rows are deliberately excluded.  The production
            # candidate path recomputes official-source post-SiTU rows and
            # uses identity ordering when an expert has no routed support.
            mid_values=torch.empty(
                (0, self.geometry.intermediate_size), dtype=torch.bfloat16
            ),
            mid_weights=torch.empty(0, dtype=torch.float32),
            mid_observations=torch.empty(0, dtype=torch.int64),
            mid_experts=torch.empty(0, dtype=torch.int32),
            mid_split=torch.empty(0, dtype=torch.int8),
        )


def _group_sample_part(
    values: dict[str, torch.Tensor],
    prefix: str,
    selected: set[int],
    *,
    path: Path,
    geometry: CaptureGeometry,
) -> dict[int, _SamplePart]:
    """Split one loaded part by layer without rereading it for every layer."""

    layer_key = f"{prefix}.layer"
    if layer_key not in values:
        return {}
    required = (
        f"{prefix}.values",
        f"{prefix}.weight",
        f"{prefix}.observation",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"sample part {path} is missing {missing}")
    layers = values[layer_key].to(torch.int64)
    rows = values[required[0]]
    weights = values[required[1]].to(torch.float32)
    observations = values[required[2]].to(torch.int64)
    expert = values.get(f"{prefix}.expert")
    route_experts = values.get(f"{prefix}.experts")
    route_gates = values.get(f"{prefix}.gates")
    split = values.get(f"{prefix}.split")
    routed_latent = values.get(f"{prefix}.routed_latent")
    n = int(layers.numel())
    if (
        layers.ndim != 1
        or rows.ndim != 2
        or weights.ndim != 1
        or observations.ndim != 1
        or int(rows.shape[0]) != n
        or int(weights.numel()) != n
        or int(observations.numel()) != n
        or (expert is not None and (expert.ndim != 1 or expert.numel() != n))
    ):
        raise ValueError(f"sample part {path} has inconsistent {prefix} rows")
    if prefix == "input":
        required_input = {
            "input.experts": route_experts,
            "input.gates": route_gates,
            "input.split": split,
            "input.routed_latent": routed_latent,
        }
        missing_input = [key for key, value in required_input.items() if value is None]
        if missing_input:
            raise ValueError(f"sample part {path} is missing {missing_input}")
        assert route_experts is not None
        assert route_gates is not None
        assert split is not None
        assert routed_latent is not None
        expected_routes = (n, geometry.top_k)
        if tuple(route_experts.shape) != expected_routes:
            raise ValueError(
                f"sample part {path} input.experts has shape "
                f"{tuple(route_experts.shape)}, expected {expected_routes}"
            )
        if tuple(route_gates.shape) != expected_routes:
            raise ValueError(
                f"sample part {path} input.gates has shape "
                f"{tuple(route_gates.shape)}, expected {expected_routes}"
            )
        if tuple(routed_latent.shape) != (n, geometry.input_size):
            raise ValueError(
                f"sample part {path} input.routed_latent has shape "
                f"{tuple(routed_latent.shape)}"
            )
        if expert is not None:
            raise ValueError(f"sample part {path} has unexpected input.expert")
        if n and (
            int(route_experts.min()) < 0
            or int(route_experts.max()) >= geometry.num_experts
        ):
            raise ValueError(f"sample part {path} has invalid input expert IDs")
        expected_weight = route_gates.to(torch.float32).square().sum(dim=1)
        if not torch.allclose(weights, expected_weight, rtol=2e-6, atol=2e-7):
            raise ValueError(
                f"sample part {path} input weights do not match applied gates"
            )
    elif prefix == "mid":
        missing_mid = [
            key
            for key, value in (("mid.expert", expert), ("mid.split", split))
            if value is None
        ]
        if missing_mid:
            raise ValueError(f"sample part {path} is missing {missing_mid}")
        assert expert is not None
        if n and (int(expert.min()) < 0 or int(expert.max()) >= geometry.num_experts):
            raise ValueError(f"sample part {path} has invalid mid expert IDs")
        if any(
            value is not None
            for value in (route_experts, route_gates, routed_latent)
        ):
            raise ValueError(f"sample part {path} has unexpected mid route metadata")
    else:
        raise ValueError(f"unsupported sample prefix {prefix!r}")
    assert split is not None
    if split.ndim != 1 or split.numel() != n:
        raise ValueError(f"sample part {path} has inconsistent {prefix}.split")
    if n and not torch.all((split == 0) | (split == 1)):
        raise ValueError(f"sample part {path} has invalid {prefix}.split values")
    if not n:
        return {}

    # Select before sorting.  Indexing the complete part and then retaining a
    # narrow slice leaves that slice backed by the complete (often ~500 MiB)
    # sorted tensor.  A one-layer candidate worker would consequently pin most
    # of the 30-GiB capture.  ``index_select`` gives the requested layers their
    # own compact storage while the complete loaded part can be released.
    selected_tensor = torch.tensor(sorted(selected), dtype=layers.dtype)
    selected_rows = torch.nonzero(
        torch.isin(layers, selected_tensor), as_tuple=False
    ).flatten()
    if not selected_rows.numel():
        return {}
    selected_layers = layers.index_select(0, selected_rows)
    order = torch.argsort(selected_layers, stable=True)
    selected_rows = selected_rows.index_select(0, order)
    layers = selected_layers.index_select(0, order)
    rows = rows.index_select(0, selected_rows)
    weights = weights.index_select(0, selected_rows)
    observations = observations.index_select(0, selected_rows)
    expert = None if expert is None else expert.index_select(0, selected_rows)
    route_experts = (
        None
        if route_experts is None
        else route_experts.index_select(0, selected_rows)
    )
    route_gates = (
        None if route_gates is None else route_gates.index_select(0, selected_rows)
    )
    split = split.index_select(0, selected_rows)
    routed_latent = (
        None
        if routed_latent is None
        else routed_latent.index_select(0, selected_rows)
    )
    unique, counts = torch.unique_consecutive(layers, return_counts=True)
    grouped: dict[int, _SamplePart] = {}
    start = 0
    for raw_layer, raw_count in zip(unique.tolist(), counts.tolist()):
        stop = start + int(raw_count)
        layer = int(raw_layer)
        if layer in selected:
            grouped[layer] = _SamplePart(
                values=rows[start:stop],
                weights=weights[start:stop],
                observations=observations[start:stop],
                expert=None if expert is None else expert[start:stop],
                route_experts=(
                    None if route_experts is None else route_experts[start:stop]
                ),
                route_gates=None if route_gates is None else route_gates[start:stop],
                split=split[start:stop],
                routed_latent=(
                    None if routed_latent is None else routed_latent[start:stop]
                ),
            )
        start = stop
    return grouped


def _append_grouped(index: _SampleIndex, grouped: dict[int, _SamplePart]) -> None:
    for layer, part in grouped.items():
        index.setdefault(layer, []).append(part)


def _index_samples(
    ranks: list[RankCapture], selected: set[int], geometry: CaptureGeometry
) -> tuple[_SampleIndex, list[_SampleIndex]]:
    """Load every capture part once and retain only selected layer rows."""

    input_index: _SampleIndex = {}
    mid_indices: list[_SampleIndex] = [{} for _ in ranks]
    for rank_index, capture in enumerate(ranks):
        for part in capture.sample_parts():
            values = load_torch_file(str(part), device="cpu")
            if capture.rank == 0:
                _append_grouped(
                    input_index,
                    _group_sample_part(
                        values, "input", selected, path=part, geometry=geometry
                    ),
                )
            _append_grouped(
                mid_indices[rank_index],
                _group_sample_part(
                    values, "mid", selected, path=part, geometry=geometry
                ),
            )
    return input_index, mid_indices


def _concatenate_sample_parts(
    index: _SampleIndex,
    layer: int,
    prefix: str,
) -> _SamplePart:
    parts = index.pop(layer, None)
    if not parts:
        raise ValueError(f"layer {layer} has no {prefix} Hessian samples")
    rows = torch.cat([part.values for part in parts])
    weights = torch.cat([part.weights for part in parts])
    observations = torch.cat([part.observations for part in parts])

    def concatenate_optional(name: str) -> torch.Tensor | None:
        tensors = [getattr(part, name) for part in parts]
        if any(value is None for value in tensors):
            if not all(value is None for value in tensors):
                raise ValueError(
                    f"layer {layer} mixes {prefix} parts with/without {name}"
                )
            return None
        return torch.cat([value for value in tensors if value is not None])

    expert = concatenate_optional("expert")
    route_experts = concatenate_optional("route_experts")
    route_gates = concatenate_optional("route_gates")
    split = concatenate_optional("split")
    routed_latent = concatenate_optional("routed_latent")
    order = torch.argsort(observations, stable=True)
    observations = observations[order]
    if observations.numel() > 1 and torch.any(observations[1:] == observations[:-1]):
        raise ValueError(f"layer {layer} has duplicate {prefix} observation IDs")
    return _SamplePart(
        values=rows[order],
        weights=weights[order],
        observations=observations,
        expert=None if expert is None else expert[order],
        route_experts=None if route_experts is None else route_experts[order],
        route_gates=None if route_gates is None else route_gates[order],
        split=None if split is None else split[order],
        routed_latent=None if routed_latent is None else routed_latent[order],
    )


def _accumulate_hessian(
    rows: torch.Tensor,
    weights: torch.Tensor,
    *,
    device: torch.device,
    chunk_rows: int,
) -> tuple[torch.Tensor, float]:
    if rows.ndim != 2 or weights.ndim != 1 or rows.shape[0] != weights.shape[0]:
        raise ValueError("invalid Hessian sample shapes")
    dim = int(rows.shape[1])
    accumulator = torch.zeros((dim, dim), dtype=torch.float32, device=device)
    denominator = 0.0
    for start in range(0, int(rows.shape[0]), max(int(chunk_rows), 1)):
        x = rows[start : start + chunk_rows].to(device=device, dtype=torch.float32)
        w = weights[start : start + chunk_rows].to(device=device, dtype=torch.float32)
        accumulator.addmm_(x.T, x * w.unsqueeze(1))
        denominator += float(w.double().sum().item())
    return accumulator, denominator


def _request_step_mask(
    observations: torch.Tensor,
    *,
    request_step_min: int,
    request_step_max: int | None,
    request_steps: list[int] | None,
) -> torch.Tensor:
    """Return the authenticated request-epoch mask for captured observations."""

    steps = torch.bitwise_right_shift(observations.to(torch.int64), 32)
    selected = steps >= request_step_min
    if request_step_max is not None:
        selected &= steps <= request_step_max
    if request_steps is not None:
        allowed = torch.tensor(request_steps, dtype=steps.dtype)
        selected &= torch.isin(steps, allowed)
    return selected


def _build_h13_identity_h2_streaming(
    capture_path: str | Path,
    output: Path,
    *,
    root: dict,
    geometry: CaptureGeometry,
    ranks: list[RankCapture],
    selected: list[int],
    device: torch.device,
    min_rows: int,
    sample_split: str,
    request_step_min: int,
    request_step_max: int | None,
    request_steps: list[int] | None,
) -> Path:
    """Build all requested H13 matrices in one tensor-selective rank-0 pass.

    The production H2 prior is identity.  Expert-local H2 matrices are rebuilt
    just in time from official-source post-SiTU rows, so reading or pooling the
    TP-sharded teacher ``mid.values`` here would be both wasteful and
    semantically wrong.
    """

    rank_zero = next((capture for capture in ranks if capture.rank == 0), None)
    if rank_zero is None:
        raise ValueError("capture has no rank-zero input samples")
    selected_set = set(selected)
    split_values = {"train": 0, "validation": 1, "all": None}
    split_value = split_values[sample_split]
    accumulators = {
        layer: torch.zeros(
            (geometry.input_size, geometry.input_size),
            dtype=torch.float32,
            device=device,
        )
        for layer in selected
    }
    denominators = {layer: 0.0 for layer in selected}
    rows = {layer: 0 for layer in selected}
    total_rows = {layer: 0 for layer in selected}
    validation_rows = {layer: 0 for layer in selected}
    required = (
        "input.layer",
        "input.values",
        "input.weight",
        "input.observation",
        "input.split",
    )

    for part in rank_zero.sample_parts():
        with safe_open(str(part), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            missing = [key for key in required if key not in keys]
            if missing:
                raise ValueError(f"sample part {part} is missing {missing}")
            part_layers = handle.get_tensor("input.layer").to(torch.int64)
            part_split = handle.get_tensor("input.split").to(torch.int8)
            observations = handle.get_tensor("input.observation").to(torch.int64)
            weights = handle.get_tensor("input.weight").to(torch.float32)
            if not (
                part_layers.ndim == part_split.ndim == observations.ndim == weights.ndim == 1
                and part_layers.numel()
                == part_split.numel()
                == observations.numel()
                == weights.numel()
            ):
                raise ValueError(f"sample part {part} has invalid input metadata")
            values = handle.get_tensor("input.values")
            if tuple(values.shape) != (part_layers.numel(), geometry.input_size):
                raise ValueError(f"sample part {part} has invalid input.values shape")
            if not bool(torch.all((part_split == 0) | (part_split == 1))):
                raise ValueError(f"sample part {part} has invalid input split labels")
            epoch_mask = _request_step_mask(
                observations,
                request_step_min=request_step_min,
                request_step_max=request_step_max,
                request_steps=request_steps,
            )
            for raw_layer in torch.unique(part_layers).tolist():
                layer = int(raw_layer)
                if layer not in selected_set:
                    continue
                layer_mask = part_layers == layer
                total_rows[layer] += int(layer_mask.sum())
                validation_rows[layer] += int((layer_mask & (part_split == 1)).sum())
                mask = layer_mask & epoch_mask
                if split_value is not None:
                    mask &= part_split == split_value
                indices = torch.nonzero(mask, as_tuple=False).flatten()
                if not indices.numel():
                    continue
                x = values.index_select(0, indices).to(
                    device=device, dtype=torch.float32
                )
                w = weights.index_select(0, indices).to(
                    device=device, dtype=torch.float32
                )
                accumulators[layer].addmm_(x.T, x * w[:, None])
                denominators[layer] += float(w.double().sum())
                rows[layer] += int(indices.numel())

    layer_meta: dict[str, dict] = {}
    for layer in selected:
        if rows[layer] < min_rows:
            raise ValueError(
                f"layer {layer} has too few H13 rows: {rows[layer]}, minimum={min_rows}"
            )
        if denominators[layer] <= 0:
            raise ValueError(f"layer {layer} has a non-positive H13 denominator")
        h13 = accumulators.pop(layer).div_(denominators[layer])
        h13 = ((h13 + h13.T) * 0.5).cpu().contiguous()
        layer_path = output / f"layer-{layer + 1:05d}.safetensors"
        tmp = output / f".{layer_path.name}.tmp"
        save_torch_file({"w13": h13}, str(tmp))
        tmp.replace(layer_path)
        layer_meta[str(layer + 1)] = {
            "row": layer,
            "w13_rows": rows[layer],
            "w13_total_rows": total_rows[layer],
            "w13_validation_rows": validation_rows[layer],
            "w13_weight_sum": denominators[layer],
            "w2_prior": "identity",
            "w2_rows": 0,
            "file": layer_path.name,
        }

    manifest = {
        "schema_version": HESSIAN_SCHEMA_VERSION,
        "kind": "kquant_dense_hessians",
        "model": root.get("model", C.MODEL_ID),
        "revision": root.get("revision", C.REVISION),
        "run_id": root.get("run_id"),
        "source_capture": str(Path(capture_path).resolve()),
        "weighting": "gate_square",
        "normalization": "sum(weight * x xT) / sum(weight)",
        "sample_split": sample_split,
        "h13_policy": "captured_layer_global",
        "h2_policy": "identity_prior_expert_local_jit",
        "request_step_range": {
            "minimum_inclusive": request_step_min,
            "maximum_inclusive": request_step_max,
            "encoding": "observation_id >> 32",
        },
        "request_step_filter": request_steps,
        "geometry": {
            "num_layers": geometry.num_layers,
            "input_size": geometry.input_size,
            "intermediate_size": geometry.intermediate_size,
        },
        "layers": layer_meta,
    }
    manifest_path = output / "manifest.json"
    tmp = output / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(manifest_path)
    return output


def _load_input_layer_samples(
    index: _SampleIndex, layer: int
) -> _SamplePart:
    samples = _concatenate_sample_parts(index, layer, "input")
    if any(
        value is None
        for value in (
            samples.route_experts,
            samples.route_gates,
            samples.split,
            samples.routed_latent,
        )
    ):
        raise ValueError(f"layer {layer} input samples lack route-aware metadata")
    return samples


def _load_mid_layer_samples(
    ranks: list[RankCapture],
    indices: list[_SampleIndex],
    layer: int,
    intermediate_size: int,
) -> _SamplePart:
    by_rank: list[_SamplePart] = []
    for capture, index in zip(ranks, indices):
        if layer not in index:
            raise ValueError(f"rank {capture.rank} layer {layer} has no mid samples")
        samples = _concatenate_sample_parts(index, layer, "mid")
        if samples.expert is None or samples.split is None:
            raise ValueError(f"rank {capture.rank} layer {layer} lacks mid metadata")
        by_rank.append(samples)

    reference = by_rank[0]
    reference_obs = reference.observations
    reference_weight = reference.weights
    assert reference.expert is not None and reference.split is not None
    full = torch.empty(
        (reference_obs.numel(), intermediate_size), dtype=reference.values.dtype
    )
    for capture, samples in zip(ranks, by_rank):
        if not torch.equal(samples.observations, reference_obs):
            raise ValueError(
                f"rank {capture.rank} layer {layer} mid observation IDs do not align"
            )
        if not torch.allclose(samples.weights, reference_weight, rtol=0, atol=1e-7):
            raise ValueError(
                f"rank {capture.rank} layer {layer} mid sample weights differ"
            )
        if not torch.equal(samples.expert, reference.expert):
            raise ValueError(
                f"rank {capture.rank} layer {layer} mid expert IDs differ"
            )
        if not torch.equal(samples.split, reference.split):
            raise ValueError(f"rank {capture.rank} layer {layer} mid splits differ")
        begin, end = capture.intermediate_channel_range
        if tuple(samples.values.shape) != (reference_obs.numel(), end - begin):
            raise ValueError(
                f"rank {capture.rank} layer {layer} mid sample width mismatch"
            )
        full[:, begin:end].copy_(samples.values)
    return _SamplePart(
        values=full,
        weights=reference_weight,
        observations=reference_obs,
        expert=reference.expert,
        split=reference.split,
    )


def index_layer_samples(
    capture_path: str | Path, layers: Iterable[int]
) -> LayerSampleIndex:
    """Read the capture once and retain compact rows for selected layers."""

    _root, geometry, ranks = load_capture(capture_path, load_stats=False)
    selected = {int(layer) for layer in layers}
    if not selected:
        raise ValueError("at least one MoE layer row must be selected")
    invalid = sorted(
        layer for layer in selected if not 0 <= layer < geometry.num_layers
    )
    if invalid:
        raise ValueError(f"MoE layer rows are out of range: {invalid}")
    input_index, mid_indices = _index_samples(ranks, selected, geometry)
    return LayerSampleIndex(
        geometry=geometry,
        ranks=ranks,
        input_index=input_index,
        mid_indices=mid_indices,
        remaining=set(selected),
    )


def build_layer_sample_cache(
    capture_path: str | Path,
    output_path: str | Path,
    *,
    layers: Iterable[int] | None = None,
) -> Path:
    """Repack rank-0 route/input rows into one directly addressable file per layer."""

    root, geometry, ranks = load_capture(capture_path, load_stats=False)
    rank_zero = next((capture for capture in ranks if capture.rank == 0), None)
    if rank_zero is None:
        raise ValueError("capture has no rank-zero input samples")
    selected = (
        list(range(geometry.num_layers))
        if layers is None
        else sorted({int(layer) for layer in layers})
    )
    if not selected or any(not 0 <= layer < geometry.num_layers for layer in selected):
        raise ValueError("sample-cache layers are empty or out of range")
    output = Path(output_path)
    if not output.name.endswith(".kqsamples"):
        output = output.with_name(output.name + ".kqsamples")
    if output.exists():
        raise FileExistsError(f"sample cache destination already exists: {output}")
    output.mkdir(parents=True)

    required = (
        "input.values",
        "input.weight",
        "input.observation",
        "input.experts",
        "input.gates",
        "input.split",
        "input.routed_latent",
        "input.layer",
    )
    selected_set = set(selected)
    grouped: _SampleIndex = {}
    try:
        for part in rank_zero.sample_parts():
            with safe_open(str(part), framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                missing = [key for key in required if key not in keys]
                if missing:
                    raise ValueError(f"sample part {part} is missing {missing}")
                values = {key: handle.get_tensor(key) for key in required}
            _append_grouped(
                grouped,
                _group_sample_part(
                    values,
                    "input",
                    selected_set,
                    path=part,
                    geometry=geometry,
                ),
            )

        entries: dict[str, dict] = {}
        for layer in selected:
            samples = _load_input_layer_samples(grouped, layer)
            assert samples.route_experts is not None
            assert samples.route_gates is not None
            assert samples.split is not None
            assert samples.routed_latent is not None
            layer_path = output / f"layer-{layer + 1:05d}.safetensors"
            tmp = output / f".{layer_path.name}.tmp"
            save_torch_file(
                {
                    "input.values": samples.values.contiguous(),
                    "input.weight": samples.weights.contiguous(),
                    "input.observation": samples.observations.contiguous(),
                    "input.experts": samples.route_experts.contiguous(),
                    "input.gates": samples.route_gates.contiguous(),
                    "input.split": samples.split.contiguous(),
                    "input.routed_latent": samples.routed_latent.contiguous(),
                },
                str(tmp),
            )
            tmp.replace(layer_path)
            entries[str(layer + 1)] = {
                "row": layer,
                "rows": int(samples.values.shape[0]),
                "validation_rows": int((samples.split == 1).sum()),
                "file": layer_path.name,
            }

        manifest = {
            "schema_version": SAMPLE_CACHE_SCHEMA_VERSION,
            "kind": "kquant_layer_sample_cache",
            "model": root.get("model", C.MODEL_ID),
            "revision": root.get("revision", C.REVISION),
            "run_id": root.get("run_id"),
            "source_capture": str(Path(capture_path).resolve()),
            "contents": "rank0_route_input_rows_no_teacher_middle",
            "geometry": {
                "num_layers": geometry.num_layers,
                "num_experts": geometry.num_experts,
                "input_size": geometry.input_size,
                "intermediate_size": geometry.intermediate_size,
                "top_k": geometry.top_k,
            },
            "layers": entries,
        }
        tmp = output / ".manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(output / "manifest.json")
    except Exception:
        # Keep a missing manifest as an unmistakable incomplete-cache marker.
        raise
    return output


def index_cached_layer_samples(
    cache_path: str | Path, layers: Iterable[int]
) -> LayerSampleCacheIndex:
    """Open selected layer files lazily from a repacked input sample cache."""

    path = Path(cache_path)
    manifest = _read_json(path / "manifest.json")
    if manifest.get("kind") != "kquant_layer_sample_cache":
        raise ValueError(f"{path} is not a layer sample cache")
    if int(manifest.get("schema_version", 0)) != SAMPLE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported layer sample cache schema in {path}")
    geometry = CaptureGeometry.from_manifest(manifest)
    selected = {int(layer) for layer in layers}
    if not selected or any(not 0 <= layer < geometry.num_layers for layer in selected):
        raise ValueError("cached sample layers are empty or out of range")
    entries = manifest.get("layers")
    if not isinstance(entries, dict):
        raise ValueError("sample cache manifest has no layer index")
    missing = [layer for layer in selected if str(layer + 1) not in entries]
    if missing:
        raise ValueError(f"sample cache lacks layer rows {missing}")
    return LayerSampleCacheIndex(
        path=path,
        geometry=geometry,
        entries=entries,
        remaining=selected,
    )


def load_layer_samples(capture_path: str | Path, layer: int) -> LayerSamples:
    """Load one zero-based MoE layer with all route-aware sample metadata."""

    return index_layer_samples(capture_path, [layer]).pop(layer)


def build_hessians(
    capture_path: str | Path,
    output_path: str | Path,
    *,
    layers: Iterable[int] | None = None,
    device: str | torch.device = "cpu",
    chunk_rows: int = 1024,
    min_rows: int = 1,
    sample_split: str = "train",
    request_step_min: int = 0,
    request_step_max: int | None = None,
    request_steps: Iterable[int] | None = None,
    h2_policy: str = "identity_prior",
) -> Path:
    """Build production H13 bundles with a symbolic identity H2 prior.

    ``diagnostic_captured_global`` is retained only to test TP joins and to
    quantify why pooled post-SiTU coordinates are invalid. Candidate encoding
    never uses its H2 values as a prior or fallback.
    """

    if request_step_min < 0:
        raise ValueError("request_step_min must be non-negative")
    if request_step_max is not None and request_step_max < request_step_min:
        raise ValueError("request_step_max must be at least request_step_min")
    selected_request_steps = (
        None if request_steps is None else sorted({int(step) for step in request_steps})
    )
    if selected_request_steps is not None and (
        not selected_request_steps or selected_request_steps[0] < 0
    ):
        raise ValueError("request_steps must be a nonempty set of nonnegative epochs")
    if h2_policy not in {"diagnostic_captured_global", "identity_prior"}:
        raise ValueError(
            "h2_policy must be 'identity_prior' or "
            "'diagnostic_captured_global'"
        )

    root, geometry, ranks = load_capture(capture_path, load_stats=False)
    output = Path(output_path)
    if not output.name.endswith(".kqhess"):
        output = output.with_name(output.name + ".kqhess")
    output.mkdir(parents=True, exist_ok=True)
    selected = (
        list(range(geometry.num_layers)) if layers is None else sorted(set(layers))
    )
    split_values = {"train": 0, "validation": 1, "all": None}
    if sample_split not in split_values:
        raise ValueError("sample_split must be 'train', 'validation', or 'all'")
    split_value = split_values[sample_split]
    target_device = torch.device(device)
    if h2_policy == "identity_prior":
        return _build_h13_identity_h2_streaming(
            capture_path,
            output,
            root=root,
            geometry=geometry,
            ranks=ranks,
            selected=selected,
            device=target_device,
            min_rows=min_rows,
            sample_split=sample_split,
            request_step_min=request_step_min,
            request_step_max=request_step_max,
            request_steps=selected_request_steps,
        )
    input_index, mid_indices = _index_samples(ranks, set(selected), geometry)
    layer_meta: dict[str, dict] = {}
    for layer in selected:
        if not 0 <= layer < geometry.num_layers:
            raise ValueError(f"MoE layer row {layer} is out of range")
        input_samples = _load_input_layer_samples(input_index, layer)
        mid_samples = _load_mid_layer_samples(
            ranks,
            mid_indices,
            layer,
            geometry.intermediate_size,
        )
        assert input_samples.split is not None and mid_samples.split is not None
        input_mask = (
            torch.ones_like(input_samples.split, dtype=torch.bool)
            if split_value is None
            else input_samples.split == split_value
        )
        mid_mask = (
            torch.ones_like(mid_samples.split, dtype=torch.bool)
            if split_value is None
            else mid_samples.split == split_value
        )
        # Capture observation IDs encode the model-execution/request epoch in
        # their high 32 bits.  This lets a corpus run exclude pre-corpus probe
        # requests and any post-corpus traffic without relying on the random
        # train/validation sample split.
        input_steps = torch.bitwise_right_shift(input_samples.observations, 32)
        mid_steps = torch.bitwise_right_shift(mid_samples.observations, 32)
        input_mask &= input_steps >= request_step_min
        mid_mask &= mid_steps >= request_step_min
        if request_step_max is not None:
            input_mask &= input_steps <= request_step_max
            mid_mask &= mid_steps <= request_step_max
        if selected_request_steps is not None:
            allowed_input_steps = torch.tensor(
                selected_request_steps, dtype=input_steps.dtype
            )
            allowed_mid_steps = torch.tensor(
                selected_request_steps, dtype=mid_steps.dtype
            )
            input_mask &= torch.isin(input_steps, allowed_input_steps)
            mid_mask &= torch.isin(mid_steps, allowed_mid_steps)
        x13 = input_samples.values[input_mask]
        w13 = input_samples.weights[input_mask]
        x2 = mid_samples.values[mid_mask]
        w2 = mid_samples.weights[mid_mask]
        if x13.shape[0] < min_rows or x2.shape[0] < min_rows:
            raise ValueError(
                f"layer {layer} has too few Hessian rows: "
                f"w13={x13.shape[0]}, w2={x2.shape[0]}, minimum={min_rows}"
            )
        h13_num, h13_den = _accumulate_hessian(
            x13, w13, device=target_device, chunk_rows=chunk_rows
        )
        h2_num, h2_den = _accumulate_hessian(
            x2, w2, device=target_device, chunk_rows=chunk_rows
        )
        if h13_den <= 0 or h2_den <= 0:
            raise ValueError(f"layer {layer} has a non-positive Hessian denominator")
        h13 = h13_num.div_(h13_den)
        h2 = h2_num.div_(h2_den)
        # Roundoff and atomics can introduce tiny asymmetry.  Store an exactly
        # symmetric FP32 matrix so downstream Cholesky behavior is reproducible.
        h13 = ((h13 + h13.T) * 0.5).cpu().contiguous()
        h2 = ((h2 + h2.T) * 0.5).cpu().contiguous()
        layer_path = output / f"layer-{layer + 1:05d}.safetensors"
        tmp = output / f".{layer_path.name}.tmp"
        save_torch_file({"w13": h13, "w2": h2}, str(tmp))
        tmp.replace(layer_path)
        layer_meta[str(layer + 1)] = {
            "row": layer,
            "w13_rows": int(x13.shape[0]),
            "w13_total_rows": int(input_samples.values.shape[0]),
            "w13_validation_rows": int((input_samples.split == 1).sum()),
            "w13_weight_sum": h13_den,
            "w2_rows": int(x2.shape[0]),
            "w2_total_rows": int(mid_samples.values.shape[0]),
            "w2_validation_rows": int((mid_samples.split == 1).sum()),
            "w2_weight_sum": h2_den,
            "file": layer_path.name,
        }

    manifest = {
        "schema_version": HESSIAN_SCHEMA_VERSION,
        "kind": "kquant_dense_hessians",
        "model": root.get("model", C.MODEL_ID),
        "revision": root.get("revision", C.REVISION),
        "run_id": root.get("run_id"),
        "source_capture": str(Path(capture_path).resolve()),
        "weighting": "gate_square",
        "normalization": "sum(weight * x xT) / sum(weight)",
        "sample_split": sample_split,
        "h13_policy": "captured_layer_global",
        "h2_policy": "diagnostic_invalid_layer_global",
        "request_step_range": {
            "minimum_inclusive": request_step_min,
            "maximum_inclusive": request_step_max,
            "encoding": "observation_id >> 32",
        },
        "request_step_filter": selected_request_steps,
        "geometry": {
            "num_layers": geometry.num_layers,
            "input_size": geometry.input_size,
            "intermediate_size": geometry.intermediate_size,
        },
        "layers": layer_meta,
    }
    manifest_path = output / "manifest.json"
    tmp = output / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(manifest_path)
    return output


def load_layer_hessians(
    path: str | Path, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load decoder ``layer`` (1..92) H13 and H2 from a ``.kqhess``."""

    path = Path(path)
    manifest = _read_json(path / "manifest.json")
    if int(manifest.get("schema_version", 0)) > HESSIAN_SCHEMA_VERSION:
        raise ValueError(f"Hessian bundle {path} is newer than this kquant")
    entry = manifest.get("layers", {}).get(str(int(layer)))
    if entry is None:
        raise KeyError(f"Hessian bundle {path} has no decoder layer {layer}")
    tensors = load_torch_file(str(path / entry["file"]), device="cpu")
    if entry.get("w2_prior") == "identity":
        intermediate_size = int(manifest["geometry"]["intermediate_size"])
        return tensors["w13"], torch.eye(intermediate_size, dtype=torch.float32)
    return tensors["w13"], tensors["w2"]
