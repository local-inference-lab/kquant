#!/usr/bin/env python3
"""Measure the authenticated Fruit expert curve at uniform K2/K3/K4 rates."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from kquant.exl3_loader import load_qsrt_encoder
from kquant.fruit_calibration import FruitCalibrationStore
from kquant.fruit_qsrt import measure_fruit_uniform_rates
from kquant.fruit_source import (
    FRUIT_ANNEALED_SPEC,
    FruitCheckpointStore,
    FruitSafetensorsStore,
)
from kquant.sqg_quantizer import install_sqg_quantizer
from scripts.build_fruit_qsrt_model import (
    BASE_MANIFEST_SHA256,
    current_encoder_provenance,
)
from scripts.encode_fruit_qsrt import SAMPLED_ASSIGNMENTS, _assignment

_RATES = (2, 3, 4)
_SCHEMA = "kquant_fruit_uniform_rate_sweep_v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(_canonical_json(value), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_rates(value: str) -> tuple[int, ...]:
    try:
        rates = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rates must be comma-separated integers"
        ) from exc
    if (
        not rates
        or len(set(rates)) != len(rates)
        or any(rate not in _RATES for rate in rates)
    ):
        raise argparse.ArgumentTypeError("rates must be a unique subset of 2,3,4")
    return rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--assignment", action="append", type=_assignment)
    selection.add_argument("--sample", action="store_true")
    parser.add_argument("--rates", type=_parse_rates, default=_RATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit uniform-rate sweep requires a CUDA device")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    assignments = (
        SAMPLED_ASSIGNMENTS
        if args.sample
        else tuple(sorted(set(args.assignment or ())))
    )
    if not assignments:
        raise ValueError("no Fruit assignments selected")

    if args.checkpoint.is_dir():
        store = FruitSafetensorsStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_manifest_sha256=BASE_MANIFEST_SHA256,
        )
    else:
        store = FruitCheckpointStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_sha256=FRUIT_ANNEALED_SPEC.checkpoint_sha256,
        )
    calibration_store = FruitCalibrationStore(args.calibration)
    quantizer_module = load_qsrt_encoder(args.exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    signature = {
        "schema": _SCHEMA,
        "source": store.evidence,
        "calibration": {
            "capture_id": calibration_store.capture_id,
            "fingerprint": calibration_store.fingerprint,
            "manifest_sha256": calibration_store.manifest_sha256,
        },
        "encoder": current_encoder_provenance(
            exllamav3_root=args.exllamav3_root,
            calibration=calibration_store,
        ),
        "rates": list(args.rates),
        "assignments": [list(value) for value in assignments],
    }
    if args.resume:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match this Fruit rate sweep")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "schema": _SCHEMA,
            "signature": signature,
            "metric_target": (
                "local routed expert output on sealed document-disjoint Fruit "
                "fit/confirmation/validation rows"
            ),
            "storage_scope": (
                "uniform path-rate endpoint with per-candidate FP16 scales before "
                "layer deduplication; not a serving package"
            ),
            "results": {},
            "complete": False,
        }
        _atomic_json(args.output, payload)

    results = payload.get("results")
    if not isinstance(results, dict):
        raise TypeError("Fruit rate-sweep results must be a JSON object")
    started = time.perf_counter()
    calibration_layer = None
    calibration_layer_id = None
    with torch.inference_mode():
        for index, (layer, expert) in enumerate(assignments, start=1):
            key = f"{layer}:{expert}"
            if key in results:
                print(
                    f"[{index}/{len(assignments)}] layer {layer} expert {expert}: resume"
                )
                continue
            if calibration_layer_id != layer:
                calibration_layer = calibration_store.load_layer(layer)
                calibration_layer_id = layer
            assert calibration_layer is not None
            result = measure_fruit_uniform_rates(
                store,
                calibration_layer.expert_rows(expert),
                layer=layer,
                expert=expert,
                rates=args.rates,
                device=device,
                quantizer_module=quantizer_module,
            )
            results[key] = result
            payload["elapsed_seconds"] = time.perf_counter() - started
            _atomic_json(args.output, payload)
            print(
                f"[{index}/{len(assignments)}] layer {layer} expert {expert}: "
                f"{result['status']}"
            )
            torch.cuda.empty_cache()
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["complete"] = True
    _atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
