"""Compute the exact provisional TP12 expert-container byte ledger.

The input is an allocation JSON with one complete keep/compressed partition
for each of Kimi-K3's 92 MoE layers.  This tool is read-only: it does not load
either checkpoint and does not materialize a model.  Every ``R_r`` schedule
has the same physical rate, so neither ``r13`` nor ``r2`` changes this ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kquant.mixed_exl3 import (
    EXPERTS_PER_LAYER,
    EXPERT_RANK_MXFP4_BYTES,
    EXPERT_RANK_SCALE_BYTES,
    EXPERT_RANK_TRELLIS_BYTES,
    EXPERT_TRELLIS_BYTES,
    FORMAT_TABLE_BYTES,
    LAYER_HEADER_BYTES,
    SCHEMA,
    SHARED_SCALE_BYTES,
    TP_SIZE,
    TP12LayerHeader,
    TP12LayerLayout,
)

MOE_LAYERS = tuple(range(1, 93))
PARAMETERS_PER_EXPERT = 3 * 3072 * 3584


def _expert_ids(value: object, *, layer: int, tier: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"layer {layer} tier {tier} must be a list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"layer {layer} tier {tier} contains a non-integer ID")
    if any(item < 0 or item >= EXPERTS_PER_LAYER for item in value):
        raise ValueError(f"layer {layer} tier {tier} contains an out-of-range ID")
    if len(set(value)) != len(value):
        raise ValueError(f"layer {layer} tier {tier} contains duplicate IDs")
    return value


def _layer_partition(allocation: dict, layer: int) -> tuple[list[int], list[int]]:
    try:
        entry = allocation["layers"][str(layer)]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"allocation is missing MoE layer {layer}") from exc
    if not isinstance(entry, dict):
        raise ValueError(f"allocation layer {layer} must be an object")
    keep = _expert_ids(entry.get("keep"), layer=layer, tier="keep")
    compressed_value = entry.get("compressed", entry.get("exl3"))
    compressed = _expert_ids(
        compressed_value, layer=layer, tier="compressed"
    )
    keep_set = set(keep)
    compressed_set = set(compressed)
    if keep_set & compressed_set:
        raise ValueError(f"allocation layer {layer} has overlapping tiers")
    if keep_set | compressed_set != set(range(EXPERTS_PER_LAYER)):
        raise ValueError(
            f"allocation layer {layer} does not cover all {EXPERTS_PER_LAYER} experts"
        )
    return keep, compressed


def build_ledger(allocation: dict, source: str) -> dict:
    layers = []
    total_kept = 0
    total_compressed = 0
    disk_bytes = 0
    disk_padding_bytes = 0
    resident_bytes_per_rank = 0
    for layer in MOE_LAYERS:
        keep, compressed = _layer_partition(allocation, layer)
        layout = TP12LayerLayout(len(compressed), len(keep))
        header = TP12LayerHeader(layer, layout)
        layers.append(header.to_manifest())
        total_kept += len(keep)
        total_compressed += len(compressed)
        disk_bytes += layout.disk_bytes
        disk_padding_bytes += layout.disk_padding_bytes
        resident_bytes_per_rank += layout.resident_bytes_per_rank

    expert_assignments = len(MOE_LAYERS) * EXPERTS_PER_LAYER
    parameters = expert_assignments * PARAMETERS_PER_EXPERT
    raw_trellis_bytes = total_compressed * EXPERT_TRELLIS_BYTES
    raw_local_scale_bytes = (
        total_compressed * TP_SIZE * EXPERT_RANK_SCALE_BYTES
    )
    raw_shared_scale_bytes = len(MOE_LAYERS) * SHARED_SCALE_BYTES
    raw_keep_bytes = total_kept * TP_SIZE * EXPERT_RANK_MXFP4_BYTES
    raw_format_table_bytes = len(MOE_LAYERS) * FORMAT_TABLE_BYTES
    resident_bytes_fleet = TP_SIZE * resident_bytes_per_rank
    accounted_disk_bytes = (
        raw_trellis_bytes
        + raw_local_scale_bytes
        + raw_shared_scale_bytes
        + raw_keep_bytes
        + raw_format_table_bytes
        + disk_padding_bytes
        + len(MOE_LAYERS) * LAYER_HEADER_BYTES
    )
    if accounted_disk_bytes != disk_bytes:
        raise AssertionError("TP12 disk component ledger does not close")
    return {
        "kind": "kquant_mixed_exl3_tp12_byte_ledger",
        "schema_version": 2,
        "source": source,
        "scope": (
            "expert-format-owned bytes only; excludes non-expert weights, "
            "filesystem metadata, framework allocator overhead, and KV/workspace memory"
        ),
        "contract": {
            "codec_schema": SCHEMA,
            "tp_size": TP_SIZE,
            "layers": len(MOE_LAYERS),
            "experts_per_layer": EXPERTS_PER_LAYER,
            "expert_assignments": expert_assignments,
            "parameters_per_expert": PARAMETERS_PER_EXPERT,
            "compressed_experts": total_compressed,
            "kept_mxfp4_experts": total_kept,
        },
        "disk": {
            "trellis_payload_bytes": raw_trellis_bytes,
            "expert_local_scale_payload_bytes": raw_local_scale_bytes,
            "layer_shared_scale_payload_bytes": raw_shared_scale_bytes,
            "kept_mxfp4_payload_bytes": raw_keep_bytes,
            "packed_format_table_payload_bytes": raw_format_table_bytes,
            "alignment_padding_bytes": disk_padding_bytes,
            "fixed_layer_header_bytes": len(MOE_LAYERS) * LAYER_HEADER_BYTES,
            "container_bytes": disk_bytes,
            "effective_bits_per_original_expert_weight": disk_bytes * 8 / parameters,
        },
        "resident": {
            "format_owned_bytes_per_rank": resident_bytes_per_rank,
            "format_owned_bytes_fleet": resident_bytes_fleet,
            "note": (
                "includes aligned hot metadata, replicated layer-shared scales, "
                "and the rank-local compressed/MXFP4 payload; excludes allocator overhead"
            ),
        },
        "layers": layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("allocation", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allocation = json.loads(args.allocation.read_text())
    payload = build_ledger(allocation, str(args.allocation.resolve()))
    rendered = json.dumps(payload, indent=1, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered)


if __name__ == "__main__":
    main()
