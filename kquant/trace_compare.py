"""Validation and comparison for opt-in Kimi correctness traces."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

TRACE_SCHEMA_VERSION = 1
PARTIAL_STAGES = frozenset(
    {
        "routed_latent_partial",
        "routed_output_partial",
        "shared_output_partial",
    }
)
ROUTE_ID_STAGES = frozenset({"canonical_topk_ids"})
_TRACE_FILE_RE = re.compile(
    r"^layer-(?P<layer>\d+)\.call-(?P<call>\d+)\."
    r"(?P<stage>[A-Za-z0-9_]+)\.pt$"
)


class TraceFormatError(ValueError):
    """Raised when a trace is incomplete, inconsistent, or corrupted."""


@dataclass(frozen=True, order=True)
class TraceKey:
    layer: int
    call: int
    stage: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "call": self.call,
            "stage": self.stage,
        }


@dataclass
class TraceRun:
    root: Path
    world_size: int
    captured_ranks: tuple[int, ...]
    manifest: dict[str, Any]
    tensors: dict[TraceKey, dict[int, torch.Tensor]]


def select_trace_call(
    run: TraceRun,
    call: int,
    *,
    normalized_call: int = 0,
) -> TraceRun:
    """Select one model invocation and normalize its call index for comparison."""
    if call < 0 or normalized_call < 0:
        raise ValueError("trace call indices must be non-negative")
    tensors = {
        TraceKey(key.layer, normalized_call, key.stage): values
        for key, values in run.tensors.items()
        if key.call == call
    }
    if not tensors:
        available = sorted({key.call for key in run.tensors})
        raise TraceFormatError(
            f"{run.root}: call {call} was not captured; available={available}"
        )
    manifest = dict(run.manifest)
    manifest["selected_call"] = call
    manifest["normalized_call"] = normalized_call
    return TraceRun(
        root=run.root,
        world_size=run.world_size,
        captured_ranks=run.captured_ranks,
        manifest=manifest,
        tensors=tensors,
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceFormatError(f"{path}: cannot read manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceFormatError(f"{path}: manifest must be a JSON object")
    if value.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceFormatError(
            f"{path}: unsupported schema_version {value.get('schema_version')!r}"
        )
    return value


def _load_tensor(path: Path, key: TraceKey, rank: int, world_size: int) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TraceFormatError(f"{path}: cannot load trace tensor: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceFormatError(f"{path}: trace payload must be a dictionary")
    metadata = payload.get("metadata")
    tensor = payload.get("tensor")
    if not isinstance(metadata, dict) or not isinstance(tensor, torch.Tensor):
        raise TraceFormatError(f"{path}: payload requires metadata and tensor")

    expected = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "layer": key.layer,
        "call": key.call,
        "stage": key.stage,
        "tp_rank": rank,
        "tp_world_size": world_size,
        "saved_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }
    differences = {
        field: {"expected": value, "actual": metadata.get(field)}
        for field, value in expected.items()
        if metadata.get(field) != value
    }
    if differences:
        raise TraceFormatError(f"{path}: metadata mismatch: {differences}")

    digest = _tensor_digest(tensor)
    if metadata.get("sha256") != digest:
        raise TraceFormatError(
            f"{path}: tensor digest mismatch "
            f"(metadata={metadata.get('sha256')!r}, actual={digest})"
        )
    return tensor


def load_trace(root: Path) -> TraceRun:
    """Load a trace directory and verify rank, metadata, and tensor integrity."""
    root = root.resolve()
    rank_dirs = sorted(path for path in root.glob("tp-rank-*") if path.is_dir())
    if not rank_dirs:
        raise TraceFormatError(f"{root}: no tp-rank-* directories")

    manifests: dict[int, dict[str, Any]] = {}
    for rank_dir in rank_dirs:
        try:
            directory_rank = int(rank_dir.name.removeprefix("tp-rank-"))
        except ValueError as exc:
            raise TraceFormatError(f"{rank_dir}: malformed rank directory") from exc
        manifest = _read_manifest(rank_dir / "manifest.json")
        if manifest.get("tp_rank") != directory_rank:
            raise TraceFormatError(
                f"{rank_dir}: manifest rank {manifest.get('tp_rank')!r} "
                f"does not match directory rank {directory_rank}"
            )
        manifests[directory_rank] = manifest

    first_rank = min(manifests)
    first = manifests[first_rank]
    world_size = first.get("tp_world_size")
    configured_ranks = first.get("ranks")
    if not isinstance(world_size, int) or world_size <= 0:
        raise TraceFormatError(f"{root}: invalid tp_world_size {world_size!r}")
    if (
        not isinstance(configured_ranks, list)
        or not configured_ranks
        or not all(isinstance(rank, int) for rank in configured_ranks)
    ):
        raise TraceFormatError(f"{root}: manifest ranks must be a non-empty list")
    captured_ranks = tuple(sorted(set(configured_ranks)))
    if any(rank < 0 or rank >= world_size for rank in captured_ranks):
        raise TraceFormatError(
            f"{root}: captured ranks {captured_ranks} exceed TP{world_size}"
        )
    if tuple(sorted(manifests)) != captured_ranks:
        raise TraceFormatError(
            f"{root}: rank directories {tuple(sorted(manifests))} do not match "
            f"configured ranks {captured_ranks}"
        )

    stable_manifest_fields = (
        "schema_version",
        "tp_world_size",
        "layers",
        "ranks",
        "start_call",
        "max_calls",
        "max_tokens",
    )
    for rank, manifest in manifests.items():
        for field in stable_manifest_fields:
            if manifest.get(field) != first.get(field):
                raise TraceFormatError(
                    f"{root}: rank {rank} manifest field {field!r} differs "
                    f"from rank {first_rank}"
                )

    tensors: dict[TraceKey, dict[int, torch.Tensor]] = {}
    keys_by_rank: dict[int, set[TraceKey]] = {}
    for rank, manifest in manifests.items():
        rank_dir = root / f"tp-rank-{rank:03d}"
        keys: set[TraceKey] = set()
        for path in sorted(rank_dir.glob("*.pt")):
            match = _TRACE_FILE_RE.fullmatch(path.name)
            if match is None:
                raise TraceFormatError(f"{path}: malformed trace filename")
            key = TraceKey(
                layer=int(match.group("layer")),
                call=int(match.group("call")),
                stage=match.group("stage"),
            )
            if key in keys:
                raise TraceFormatError(f"{path}: duplicate trace key {key}")
            keys.add(key)
            tensors.setdefault(key, {})[rank] = _load_tensor(
                path, key, rank, world_size
            )
        if not keys:
            raise TraceFormatError(f"{rank_dir}: no trace tensors")
        keys_by_rank[rank] = keys

    reference_keys = keys_by_rank[first_rank]
    for rank, keys in keys_by_rank.items():
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise TraceFormatError(
                f"{root}: rank {rank} tensor coverage differs; "
                f"missing={missing}, extra={extra}"
            )

    return TraceRun(
        root=root,
        world_size=world_size,
        captured_ranks=captured_ranks,
        manifest=first,
        tensors=tensors,
    )


def _float_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_float = reference.float()
    actual_float = actual.float()
    delta = actual_float - reference_float
    reference_norm = torch.linalg.vector_norm(reference_float)
    actual_norm = torch.linalg.vector_norm(actual_float)
    delta_norm = torch.linalg.vector_norm(delta)
    if reference_norm == 0:
        relative_l2 = 0.0 if actual_norm == 0 else math.inf
    else:
        relative_l2 = float(delta_norm / reference_norm)
    if reference_norm == 0 and actual_norm == 0:
        cosine = 1.0
    elif reference_norm == 0 or actual_norm == 0:
        cosine = 0.0
    else:
        cosine = float(
            torch.dot(reference_float.flatten(), actual_float.flatten())
            / (reference_norm * actual_norm)
        )
    return {
        "relative_l2": relative_l2,
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "cosine": cosine,
        "reference_norm": float(reference_norm),
        "actual_norm": float(actual_norm),
        "exact": torch.equal(reference, actual),
    }


def _comparison_tensor(run: TraceRun, key: TraceKey) -> tuple[torch.Tensor, str]:
    rank_tensors = run.tensors[key]
    if key.stage not in PARTIAL_STAGES:
        return rank_tensors[run.captured_ranks[0]], "replicated"
    expected_ranks = tuple(range(run.world_size))
    if run.captured_ranks != expected_ranks:
        raise TraceFormatError(
            f"{run.root}: {key.stage} requires all TP ranks for aggregation; "
            f"captured={run.captured_ranks}, expected={expected_ranks}"
        )
    return (
        torch.stack(
            [rank_tensors[rank].float() for rank in run.captured_ranks]
        ).sum(dim=0),
        "fp32_rank_sum",
    )


def compare_traces(
    left: TraceRun,
    right: TraceRun,
    *,
    max_relative_l2: float = 0.02,
    min_cosine: float = 0.999,
    max_route_mismatch_fraction: float = 0.0,
    allow_key_coverage_difference: bool = False,
) -> dict[str, Any]:
    """Compare two complete traces with TP-aware partial aggregation."""
    if max_relative_l2 < 0:
        raise ValueError("max_relative_l2 must be non-negative")
    if not -1 <= min_cosine <= 1:
        raise ValueError("min_cosine must be between -1 and 1")
    if not 0 <= max_route_mismatch_fraction <= 1:
        raise ValueError("max_route_mismatch_fraction must be between 0 and 1")

    issues: list[str] = []
    left_keys = set(left.tensors)
    right_keys = set(right.tensors)
    if left_keys != right_keys and not allow_key_coverage_difference:
        issues.append(
            "trace key coverage differs: "
            f"left_only={len(left_keys - right_keys)}, "
            f"right_only={len(right_keys - left_keys)}"
        )

    replicated_rank_mismatches: list[dict[str, Any]] = []
    for side, run in (("left", left), ("right", right)):
        for key in sorted(run.tensors):
            if key.stage in PARTIAL_STAGES:
                continue
            values = run.tensors[key]
            reference_rank = run.captured_ranks[0]
            mismatched = [
                rank
                for rank in run.captured_ranks[1:]
                if not torch.equal(values[reference_rank], values[rank])
            ]
            if mismatched:
                replicated_rank_mismatches.append(
                    {
                        "side": side,
                        **key.as_dict(),
                        "reference_rank": reference_rank,
                        "mismatched_ranks": mismatched,
                    }
                )

    comparisons: list[dict[str, Any]] = []
    for key in sorted(left_keys & right_keys):
        left_tensor, reduction = _comparison_tensor(left, key)
        right_tensor, right_reduction = _comparison_tensor(right, key)
        if key.stage == "canonical_topk_weights":
            ids_key = TraceKey(key.layer, key.call, "canonical_topk_ids")
            if ids_key not in left.tensors or ids_key not in right.tensors:
                raise TraceFormatError(
                    f"{key}: canonical_topk_weights requires canonical_topk_ids"
                )
            left_ids, _ = _comparison_tensor(left, ids_key)
            right_ids, _ = _comparison_tensor(right, ids_key)
            if left_ids.shape != left_tensor.shape:
                raise TraceFormatError(
                    f"{left.root}: route ID/weight shape mismatch for {key}"
                )
            if right_ids.shape != right_tensor.shape:
                raise TraceFormatError(
                    f"{right.root}: route ID/weight shape mismatch for {key}"
                )
            left_tensor = left_tensor.gather(
                -1, left_ids.argsort(dim=-1).long()
            )
            right_tensor = right_tensor.gather(
                -1, right_ids.argsort(dim=-1).long()
            )
            reduction = "replicated_aligned_by_expert_id"
            right_reduction = reduction
        item: dict[str, Any] = {
            **key.as_dict(),
            "reduction": reduction,
            "left_dtype": str(left_tensor.dtype),
            "right_dtype": str(right_tensor.dtype),
            "left_shape": list(left_tensor.shape),
            "right_shape": list(right_tensor.shape),
        }
        if reduction != right_reduction:
            item["pass"] = False
            item["issue"] = f"reduction differs: {reduction} vs {right_reduction}"
        elif left_tensor.shape != right_tensor.shape:
            item["pass"] = False
            item["issue"] = "shape differs"
        elif key.stage in ROUTE_ID_STAGES:
            position_mismatch_count = int((left_tensor != right_tensor).sum())
            element_count = left_tensor.numel()
            width = left_tensor.shape[-1] if left_tensor.ndim else 1
            left_rows = left_tensor.reshape(-1, width)
            right_rows = right_tensor.reshape(-1, width)
            membership_mismatch_count = 0
            membership_changed_rows = 0
            for left_row, right_row in zip(left_rows, right_rows):
                left_members = set(left_row.tolist())
                right_members = set(right_row.tolist())
                removed = len(left_members - right_members)
                membership_mismatch_count += removed
                membership_changed_rows += int(removed > 0)
            mismatch_fraction = (
                membership_mismatch_count / element_count
                if element_count
                else 0.0
            )
            item.update(
                {
                    "position_mismatch_count": position_mismatch_count,
                    "membership_mismatch_count": membership_mismatch_count,
                    "membership_changed_rows": membership_changed_rows,
                    "element_count": element_count,
                    "membership_mismatch_fraction": mismatch_fraction,
                    "pass": mismatch_fraction <= max_route_mismatch_fraction,
                }
            )
        elif left_tensor.is_floating_point() and right_tensor.is_floating_point():
            metrics = _float_metrics(left_tensor, right_tensor)
            finite = all(
                math.isfinite(metrics[name])
                for name in ("relative_l2", "max_abs", "mean_abs", "cosine")
            )
            item.update(metrics)
            item["pass"] = (
                finite
                and metrics["relative_l2"] <= max_relative_l2
                and metrics["cosine"] >= min_cosine
            )
        else:
            mismatch_count = int((left_tensor != right_tensor).sum())
            item.update(
                {
                    "mismatch_count": mismatch_count,
                    "element_count": left_tensor.numel(),
                    "pass": mismatch_count == 0,
                }
            )
        comparisons.append(item)

    float_items = [item for item in comparisons if "relative_l2" in item]
    route_items = [item for item in comparisons if item["stage"] in ROUTE_ID_STAGES]
    failed_comparisons = [item for item in comparisons if not item["pass"]]
    passed = (
        not issues
        and not replicated_rank_mismatches
        and not failed_comparisons
    )
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "pass": passed,
        "left": {
            "root": str(left.root),
            "tp_world_size": left.world_size,
            "captured_ranks": list(left.captured_ranks),
        },
        "right": {
            "root": str(right.root),
            "tp_world_size": right.world_size,
            "captured_ranks": list(right.captured_ranks),
        },
        "thresholds": {
            "max_relative_l2": max_relative_l2,
            "min_cosine": min_cosine,
            "max_route_mismatch_fraction": max_route_mismatch_fraction,
        },
        "issues": issues,
        "replicated_rank_mismatches": replicated_rank_mismatches,
        "comparisons": comparisons,
        "summary": {
            "comparison_count": len(comparisons),
            "failed_comparison_count": len(failed_comparisons),
            "worst_relative_l2": max(
                (item["relative_l2"] for item in float_items),
                default=0.0,
            ),
            "minimum_cosine": min(
                (item["cosine"] for item in float_items),
                default=1.0,
            ),
            "route_mismatch_count": sum(
                item.get("membership_mismatch_count", 0) for item in route_items
            ),
            "route_position_mismatch_count": sum(
                item.get("position_mismatch_count", 0) for item in route_items
            ),
            "route_element_count": sum(
                item.get("element_count", 0) for item in route_items
            ),
        },
    }
