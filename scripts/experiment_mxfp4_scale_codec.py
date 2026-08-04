"""Measure an exact row-palette codec for official MXFP4 scale planes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from kquant import constants as C
from kquant.mxfp4_scale_codec import (
    compact_scale_plane_stats,
    effective_mxfp4_bpw,
    pack_scale_plane,
    pack_scale_plane_compact,
    unpack_scale_plane,
    unpack_scale_plane_compact,
)
from kquant.source_weights import OfficialMXFP4Store


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _parse_ints(value: str, *, minimum: int, maximum: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(minimum, maximum + 1))
    try:
        result = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    if any(not minimum <= item <= maximum for item in result):
        raise argparse.ArgumentTypeError(
            f"values must lie in {minimum}..{maximum}"
        )
    return result


def _matrix_evidence(
    scale: torch.Tensor, *, tile_rows: int, format_version: int
) -> dict[str, object]:
    if format_version == 1:
        payload = pack_scale_plane(scale, tile_rows=tile_rows)
        reconstruction = unpack_scale_plane(payload)
    elif format_version == 2:
        payload = pack_scale_plane_compact(scale, tile_rows=tile_rows)
        reconstruction = unpack_scale_plane_compact(payload)
    else:
        raise ValueError(f"unsupported scale codec version {format_version}")
    if not torch.equal(reconstruction, scale):
        raise RuntimeError("scale codec failed exact reconstruction")
    histogram = torch.bincount(scale.flatten().to(torch.int64), minlength=256)
    occupied = torch.nonzero(histogram).flatten()
    probabilities = histogram[occupied].double() / scale.numel()
    entropy = float(-(probabilities * probabilities.log2()).sum())
    rows, columns = map(int, scale.shape)
    tile_count = math.ceil(rows / tile_rows)
    selector_bytes = rows * math.ceil(columns / 8)
    if format_version == 1:
        fixed_bytes = 16 + 4 * (tile_count + 1) + rows * 3 + selector_bytes
        exception_bytes = len(payload) - fixed_bytes
        if exception_bytes < 0 or exception_bytes % 2:
            raise RuntimeError("scale payload byte accounting did not close")
        exceptions = exception_bytes // 2
    else:
        compact_stats = compact_scale_plane_stats(payload)
        exceptions = compact_stats["exceptions"]
    weights = scale.numel() * C.MXFP4_BLOCK
    original_bytes = weights // 2 + scale.numel()
    coded_bytes = weights // 2 + len(payload)
    return {
        "rows": rows,
        "scale_columns": columns,
        "weights": weights,
        "original_scale_bytes": scale.numel(),
        "coded_scale_bytes": len(payload),
        "original_total_bytes": original_bytes,
        "coded_total_bytes": coded_bytes,
        "original_bpw": original_bytes * 8 / weights,
        "coded_bpw": effective_mxfp4_bpw(scale, payload),
        "exact_reconstruction": True,
        "scale_min": int(occupied.min()),
        "scale_max": int(occupied.max()),
        "scale_unique": int(occupied.numel()),
        "scale_entropy_bits": entropy,
        "row_palette_exceptions": exceptions,
        "row_palette_exception_rate": exceptions / scale.numel(),
        "tile_rows": tile_rows,
        "tile_count": tile_count,
        "format_version": format_version,
        **(
            {
                "adjacent_palette_escape_rows": compact_stats["palette_escapes"],
                "extended_exception_count_rows": compact_stats["count_escapes"],
            }
            if format_version == 2
            else {}
        ),
    }


def _aggregate(values: list[dict[str, object]]) -> dict[str, float | int | bool]:
    weights = sum(int(value["weights"]) for value in values)
    original_scale_bytes = sum(int(value["original_scale_bytes"]) for value in values)
    coded_scale_bytes = sum(int(value["coded_scale_bytes"]) for value in values)
    original_total_bytes = sum(int(value["original_total_bytes"]) for value in values)
    coded_total_bytes = sum(int(value["coded_total_bytes"]) for value in values)
    exceptions = sum(int(value["row_palette_exceptions"]) for value in values)
    result: dict[str, float | int | bool] = {
        "matrices": len(values),
        "weights": weights,
        "original_scale_bytes": original_scale_bytes,
        "coded_scale_bytes": coded_scale_bytes,
        "scale_byte_ratio": coded_scale_bytes / original_scale_bytes,
        "original_total_bytes": original_total_bytes,
        "coded_total_bytes": coded_total_bytes,
        "original_bpw": original_total_bytes * 8 / weights,
        "coded_bpw": coded_total_bytes * 8 / weights,
        "total_byte_savings": 1.0 - coded_total_bytes / original_total_bytes,
        "row_palette_exceptions": exceptions,
        "row_palette_exception_rate": exceptions / original_scale_bytes,
        "exact_reconstruction": all(bool(value["exact_reconstruction"]) for value in values),
    }
    if values and all("adjacent_palette_escape_rows" in value for value in values):
        rows = sum(int(value["rows"]) for value in values)
        palette_escapes = sum(
            int(value["adjacent_palette_escape_rows"]) for value in values
        )
        count_escapes = sum(
            int(value["extended_exception_count_rows"]) for value in values
        )
        result.update(
            {
                "rows": rows,
                "adjacent_palette_escape_rows": palette_escapes,
                "adjacent_palette_escape_rate": palette_escapes / rows,
                "extended_exception_count_rows": count_escapes,
                "extended_exception_count_rate": count_escapes / rows,
            }
        )
    return result


def run(args: argparse.Namespace) -> dict:
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    signature = {
        "official_source": str(store.root),
        "official_revision": args.official_revision,
        "layers": list(args.layers),
        "experts": list(args.experts),
        "matrices": list(C.EXPERT_MATRICES),
        "tile_rows": args.tile_rows,
        "format_version": args.format_version,
    }
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        payload = json.loads(args.output.read_text())
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match this experiment")
        if payload.get("complete"):
            return payload
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "kind": "kquant_exact_mxfp4_scale_codec_experiment",
            "schema_version": 1,
            "signature": signature,
            "format": (
                "row palette plus one selector bit per scale, sparse exact "
                "exceptions, and uint32 independently-decodable tile offsets"
            ),
            "results": {},
            "complete": False,
        }
        _atomic_write(args.output, payload)

    for layer in args.layers:
        layer_results = payload["results"].setdefault(str(layer), {})
        for expert in args.experts:
            key = str(expert)
            if key in layer_results:
                continue
            matrices = {}
            for matrix in C.EXPERT_MATRICES:
                scale = store.load_scale_plane(layer, expert, matrix)
                matrices[matrix] = _matrix_evidence(
                    scale,
                    tile_rows=args.tile_rows,
                    format_version=args.format_version,
                )
            layer_results[key] = {
                "matrices": matrices,
                "aggregate": _aggregate(list(matrices.values())),
            }
            _atomic_write(args.output, payload)

    all_matrices = [
        matrix
        for layer in payload["results"].values()
        for expert in layer.values()
        for matrix in expert["matrices"].values()
    ]
    payload["aggregate"] = _aggregate(all_matrices)
    payload["complete"] = True
    _atomic_write(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layers",
        type=lambda value: _parse_ints(value, minimum=1, maximum=92),
        required=True,
    )
    parser.add_argument(
        "--experts",
        type=lambda value: _parse_ints(value, minimum=0, maximum=895),
        required=True,
    )
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--format-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument("--official-revision", default=C.REVISION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.tile_rows <= 255:
        parser.error("--tile-rows must lie in 1..255")
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps({
        "aggregate": payload["aggregate"],
        "output": str(args.output.resolve()),
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
