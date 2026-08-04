"""Scan exact zero columns in every retained interim K3 expert.

The scan reads packed MXFP4 codes only from ``keep-mxfp4-*`` shards in a
packaged interim EXL3 checkpoint. It never resolves or reads the official
checkpoint, and it does not decode the interim EXL3 tier.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.codec_research import packed_mxfp4_zero_statistics
from kquant.interim_weights import InterimKeepStore

_W2_PACKED = re.compile(
    r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\."
    r"(\d+)\.w2\.weight_packed$"
)


def _summarize(values: list[int]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
            dtype=tensor.dtype,
        ),
    )
    with_any = sum(value > 0 for value in values)
    return {
        "assignments": len(values),
        "experts_with_any_zero_column": with_any,
        "fraction_with_any_zero_column": with_any / len(values),
        "mean_zero_columns": float(tensor.mean()),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
                quantiles,
            )
        },
        "mean_prunable_intermediate_fraction": float(tensor.mean() / C.EXPERT_INTER),
        "histogram": {
            str(key): count for key, count in sorted(Counter(values).items())
        },
    }


def run(checkpoint: Path) -> dict:
    store = InterimKeepStore(checkpoint)
    files_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, filename in store.weight_map.items():
        if filename.startswith("keep-mxfp4-") and _W2_PACKED.fullmatch(key):
            files_to_keys[filename].append(key)

    records = []
    for file_index, filename in enumerate(sorted(files_to_keys), start=1):
        path = store.root / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in files_to_keys[filename]:
                match = _W2_PACKED.fullmatch(key)
                assert match is not None
                layer, expert = map(int, match.groups())
                statistics = packed_mxfp4_zero_statistics(handle.get_tensor(key))
                records.append(
                    {
                        "layer": layer,
                        "expert": expert,
                        **statistics,
                    }
                )
        print(f"scanned {file_index}/{len(files_to_keys)} retained shards", flush=True)

    expected = sum(len(store.kept_experts(layer)) for layer in C.MOE_LAYERS)
    if len(records) != expected:
        raise ValueError(
            f"found {len(records)} retained w2 tensors, expected {expected}"
        )
    by_layer: dict[int, list[int]] = defaultdict(list)
    for record in records:
        by_layer[record["layer"]].append(record["zero_columns"])
    values = [record["zero_columns"] for record in records]
    top = sorted(
        records,
        key=lambda value: (
            -value["zero_columns"],
            value["layer"],
            value["expert"],
        ),
    )
    return {
        "kind": "kquant_interim_zero_column_scan",
        "schema_version": 1,
        "source": {
            "checkpoint": str(store.root),
            "constraint": (
                "retained MXFP4 tier of interim EXL3 hybrid only; official model "
                "not loaded"
            ),
            "coverage": "all retained layer/expert assignments",
        },
        "global": _summarize(values),
        "by_layer": {
            str(layer): _summarize(by_layer[layer]) for layer in sorted(by_layer)
        },
        "largest": top[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True))
    temporary.replace(args.output)
    print(json.dumps(result["global"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
