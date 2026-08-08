#!/usr/bin/env python3
"""Encode and assemble a complete exact-rate Fruit QSRT Hugging Face model."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from kquant.exl3_loader import load_qsrt_encoder
from kquant.fruit_calibration import FruitCalibrationStore
from kquant.fruit_qsrt import (
    FRUIT_QSRT_ARTIFACT_TENSORS,
    FRUIT_QSRT_ATOM_BUNDLE_BYTES,
    FRUIT_QSRT_ATOM_CHANNELS,
    FRUIT_QSRT_ATOM_SCHEMA,
    FRUIT_QSRT_ATOM_STORAGE,
    FRUIT_QSRT_ATOM_TENSOR,
    FRUIT_QSRT_ATOM_TENSORS,
    FRUIT_QSRT_ATOMS_PER_EXPERT,
    FRUIT_QSRT_CODEBOOK,
    FRUIT_QSRT_FORMAT_SECTION_BYTES,
    FRUIT_QSRT_FORMAT_TENSOR,
    FRUIT_QSRT_PAIR_COUNT,
    FRUIT_QSRT_PAIR_WORDS,
    FRUIT_QSRT_PROFILE_ID,
    FRUIT_QSRT_SCHEMA,
    FRUIT_QSRT_SHARED_SCALE_TENSOR,
    FRUIT_QSRT_STORAGE_ALIGNMENT,
    FruitMatrixStore,
    encode_fruit_expert,
    pack_fruit_atom_layer,
)
from kquant.fruit_source import (
    FRUIT_ANNEALED_SPEC,
    FruitCheckpointStore,
    FruitSafetensorsStore,
)
from kquant.sqg_quantizer import install_sqg_quantizer

LAYERS = (*FRUIT_ANNEALED_SPEC.layers, FRUIT_ANNEALED_SPEC.mtp_layer)
EXPERTS = FRUIT_ANNEALED_SPEC.num_experts
HIDDEN_SIZE = FRUIT_ANNEALED_SPEC.hidden_size
INTERMEDIATE_SIZE = FRUIT_ANNEALED_SPEC.intermediate_size
PAIR_COUNT = FRUIT_QSRT_PAIR_COUNT
PAIR_WORDS = FRUIT_QSRT_PAIR_WORDS
BASE_MANIFEST_SHA256 = (
    "8a7e30f3a948bbac203013160b2e6bb8d0ed50c36cf2ca1c3978701124cc7671"
)
EXLLAMAV3_REVISION = "791c83073f7f90c44f765a0ceeab7a05fa15b96b"
_COMPLETE_MARKER_NAME = "QSRT_COMPLETE.json"
MODEL_CARD_TEMPLATE = r"""---
license: mit
library_name: vllm
pipeline_tag: text-generation
tags:
- glm
- mixture-of-experts
- kquant
- qsrt
- vllm
- b12x
- experimental
---

# GLM-5.2 QSRT Fruit

This is a **5.04B-parameter GLM-5.2 serving proxy**, not the 754B GLM-5.2 model.
It is the first complete Fruit checkpoint encoded in KQuant's canonical QSRT
atom format and served without reconstructing dense expert weights.

The artifact is a codec/storage/runtime integration release. Packaging,
provenance, exact state decoding, and the canonical single-GPU runtime path are
implemented. The checkpoint itself has no downstream chat-quality
qualification; see [Known limitations](#known-limitations).

## What is included

- 13 transformer layers: 3 dense and 10 MoE layers, plus the packaged MTP
  expert layer.
- Hidden size 1,024; MoE intermediate size 512; 256 routed experts per MoE/MTP
  layer.
- 2,816 QSRT experts in 11 canonical atom files.
- SQG-XOR-Cheb-T12 E4M3 codebook, three-bit trellis payload, fixed P24/P33 pair
  records, and physical atom rotation.
- Canonical `qsrt_atoms_v1` storage with complete per-file SHA-256 manifests
  and a fail-closed `QSRT_COMPLETE.json` marker.
- W4A16 prefill/reference execution and W4A8 decode execution through B12X.

The expert allocation selected by the frozen calibration evidence is recorded
in each `qsrt-layer-*.json` sidecar. Aggregate allocation counts are:

| Allocation code | Experts |
|---|---:|
__ALLOCATION_ROWS__

## Size and memory

The apples-to-apples baseline is the complete BF16 tensor set. All three rows
below cover the same 5,040,368,896 logical parameters and count only
Safetensors files; `effective bpw` is stored bytes times eight divided by that
parameter count, so it includes container and quantization metadata.

| Tensor payload | Bytes | GiB | Effective bpw | Relative to BF16 |
|---|---:|---:|---:|---:|
| BF16 source | 10,081,800,232 | 9.3894 | 16.0017 | baseline |
| Prior SIQ mixed | 3,102,116,152 | 2.8891 | 4.9236 | 69.23% smaller |
| This QSRT model | 2,909,352,104 | 2.7095 | 4.6177 | 71.14% smaller |

The whole-model rates include 611,183,872 non-routed parameters retained in
BF16. Isolating the 4,429,185,024 routed-expert weights gives:

| Routed-expert format | Stored bytes | Nominal path bpw | Effective stored bpw |
|---|---:|---:|---:|
| BF16 | 8,858,370,048 | 16.0000 | 16.0000 |
| Prior SIQ mixed (1,856 K3 / 960 K4 experts) | 1,879,717,272 | 3.3409 | 3.3951 |
| QSRT P24/P33 atoms | 1,686,953,224 | 3.0000 | 3.0470 |

QSRT is therefore 10.25% smaller than SIQ on the routed-expert component and
6.21% smaller on the compared tensor files. Package-level totals are not used
for the comparison because tokenizer, card, and optional evaluation files are
not model weights. The previous card's 7,593,020,594-byte BF16 row was not the
complete BF16 tensor set and has been removed.

W4A8 and W4A16 use the same stored weights, so their loader weight storage is
identical; W4A8 changes the decode execution path, not the checkpoint size.

### Hugging Face repository-size audit

The Hugging Face model API with `blobs=true` reported the following immutable
snapshot on 2026-08-08. `Repository bytes` sums every sibling's reported size;
`Safetensors bytes` sums only `*.safetensors`. These are observed repository
payloads, not parameter-count estimates.

| Artifact | Revision | Repository bytes | Safetensors bytes |
|---|---|---:|---:|
| [Fruit QSRT (pre-adjacent-rate-evidence publication)](https://huggingface.co/malaiwah/GLM-5.2-QSRT-Fruit/tree/c1a0c62d220602fdd8b7940dcba716671fb0033c) | `c1a0c62d` | 2,963,027,998 | 2,909,352,104 |
| [Fruit BF16](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit-bf16/tree/ff1178d233fddc644dc053c723d58839eb921334) | `ff1178d2` | 10,102,776,679 | 10,081,800,232 |
| [Fruit prior mixed SIQ](https://huggingface.co/malaiwah/GLM-5.2-SIQ-Fruit/tree/c1798e3676fa16b4a874381171adab1e3033fbd5) | `c1798e36` | 3,125,527,019 | 3,102,116,152 |
| [Full GLM-5.2 BF16](https://huggingface.co/zai-org/GLM-5.2/tree/b4734de4facf877f85769a911abafc5283eab3d9) | `b4734de4` | 1,506,693,036,946 | 1,506,667,387,408 |
| [Full GLM-5.2 FP8](https://huggingface.co/zai-org/GLM-5.2-FP8/tree/ba978f7d347eaf65d22f1a86833408afdb953541) | `ba978f7d` | 755,663,676,164 | 755,632,050,320 |
| [Full GLM-5.2 NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4/tree/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa) | `aec724e8` | 464,874,323,992 | 464,823,042,096 |

The three full-model rows ground real download/storage scale only. They are not
used for Fruit percentage claims because Fruit has 5.04B parameters while the
production model has roughly 754B. The apples-to-apples Fruit tensor
comparison above remains the codec-size result.

__RATE_SWEEP_SECTION__

## Evidence boundary

The completion seal covers every top-level package file and every regular file
under `evaluation/`. The sealed adjacent-rate report measures local routed
expert reconstruction on authenticated, document-disjoint calibration rows.
It does not establish chat quality, broad downstream task quality, or general
serving throughput.

## Reproducible runtime

The model requires the matching experimental branches until the pull requests
merge:

- KQuant encoder: [`local-inference-lab/kquant#4`](https://github.com/local-inference-lab/kquant/pull/4),
  encoded with KQuant revision `__KQUANT_REVISION__`.
- B12X kernels: [`local-inference-lab/b12x#129`](https://github.com/local-inference-lab/b12x/pull/129),
  tested revision `__B12X_REVISION__`.
- vLLM loader: [`local-inference-lab/vllm#269`](https://github.com/local-inference-lab/vllm/pull/269),
  tested revision `__VLLM_REVISION__`.

```bash
git clone --branch feat/fruit-qsrt-runtime https://github.com/malaiwah/sparkinfer.git b12x-fruit
git -C b12x-fruit checkout __B12X_REVISION__
git clone --branch feat/fruit-qsrt-runtime https://github.com/malaiwah/vllm-voipmonitor.git vllm-fruit
git -C vllm-fruit checkout __VLLM_REVISION__

hf download malaiwah/GLM-5.2-QSRT-Fruit --local-dir GLM-5.2-QSRT-Fruit

B12X_ROOT="$PWD/b12x-fruit" \
MODEL="$PWD/GLM-5.2-QSRT-Fruit" \
PYTHON_BIN="$PWD/vllm-fruit/.venv/bin/python" \
CUDA_VISIBLE_DEVICES=0 \
MAX_NUM_SEQS=1 \
./vllm-fruit/serve-glm52-fruit-qsrt.sh
```

The tested environment uses SM120, CUDA 13.2-era wheels,
`nvidia-cutlass-dsl >= 4.6`, and the r31 vLLM/B12X image stack. The launcher
defaults to one sequence because the current B12X sparse-prefill backend
requires single-request prefill chunks. Only TP1 has been validated for this
Fruit package.

W4A16 is used for prefill and any row count above the W4A8 decode ceiling. W4A8
is selected for decode-sized batches of at most 16 rows. Unsupported shapes,
activation modes, metadata, or incomplete manifests fail closed.

## Provenance and integrity

- Authenticated encoder source (`__SOURCE_KIND__`) SHA-256:
  `__SOURCE_SHA256__`.
- Calibration capture ID:
  `__CALIBRATION_CAPTURE_ID__`.
- Calibration manifest SHA-256:
  `__CALIBRATION_MANIFEST_SHA256__`.
- The encoder authenticated __CALIBRATION_DOCUMENTS__ documents /
  __CALIBRATION_TOKENS__ tokens from disjoint fit, confirmation, and validation
  splits.
- Full encoding: __ENCODED_EXPERTS__ experts, __ENCODE_SECONDS__ GPU-seconds,
  __PEAK_GIB__ GiB peak CUDA allocation.
- `MANIFEST.sha256`, `qsrt-manifest.json`, `.qsrt-source-evidence.json`,
  `qsrt-calibration-evidence.json`, and `QSRT_COMPLETE.json` bind the published
  package to the source and encoder fingerprints.

## Known limitations

- **Not chat-quality.** An informal four-prompt instruction battery produced
  incoherent answers. No representative downstream instruction/chat
  evaluation is sealed by the builder, so no task-quality claim is made and
  this checkpoint must not be deployed as a user-facing assistant.
- TP2 atom ownership is unit-tested, but only TP1 physical serving has been
  qualified for this Fruit package.
- The current sparse-attention prefill backend requires `max_num_seqs=1`.
- This release qualifies the QSRT codec, storage, loader, and kernels. It does
  not establish broad downstream task quality.

## License

MIT, matching the packaged Fruit source license. KQuant, B12X, and vLLM retain
their respective repository licenses.
"""
_SOURCE_EVIDENCE_NAME = ".qsrt-source-evidence.json"
_SOURCE_EVIDENCE_SHA_NAME = ".qsrt-source-evidence.sha256"
_CALIBRATION_EVIDENCE_NAME = "qsrt-calibration-evidence.json"

_RATE_SWEEP_SCHEMA = "kquant_fruit_uniform_rate_sweep_v1"
_RATE_SWEEP_NAME = "evaluation/fruit-uniform-rate-sweep.json"
_ENCODER_FINGERPRINT_SCHEMA = "kquant_fruit_qsrt_encoder_source_v2"
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve source revision: {root}") from exc
    revision = result.stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"source revision is not a commit digest: {root}")
    return revision


def _require_clean_source(root: Path, pathspec: str) -> None:
    _git_revision(root)
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                pathspec,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect source tree: {root}") from exc
    if result.stdout:
        raise ValueError(f"source tree has uncommitted files under {root / pathspec}")


def _source_tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = tuple(path for path in root.rglob("*") if "__pycache__" not in path.parts)
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"source tree must not contain symbolic links: {path}")
    files = sorted(
        (path for path in entries if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"source tree has no fingerprinted files: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest()


def _encoder_fingerprint_payload(encoder: dict[str, object]) -> dict[str, object]:
    return {
        "schema": _ENCODER_FINGERPRINT_SCHEMA,
        "kquant_source_sha256": encoder["kquant_source_sha256"],
        "exllamav3_source_sha256": encoder["exllamav3_source_sha256"],
        "calibration_fingerprint": encoder["calibration_fingerprint"],
        "calibration_capture_id": encoder["calibration_capture_id"],
        "calibration_manifest_sha256": encoder["calibration_manifest_sha256"],
    }


def current_encoder_provenance(
    *,
    exllamav3_root: Path,
    calibration: FruitCalibrationStore,
) -> dict[str, object]:
    kquant_checkout = Path(__file__).resolve().parents[1]
    kquant_root = kquant_checkout / "kquant"
    encoder: dict[str, object] = {
        "kquant_revision": _git_revision(kquant_checkout),
        "kquant_source_sha256": _source_tree_sha256(kquant_root),
        "exllamav3_revision": EXLLAMAV3_REVISION,
        "exllamav3_source_sha256": _source_tree_sha256(exllamav3_root / "exllamav3"),
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_capture_id": calibration.capture_id,
        "calibration_manifest_sha256": calibration.manifest_sha256,
        "fingerprint_schema": _ENCODER_FINGERPRINT_SCHEMA,
    }
    encoder["fingerprint"] = hashlib.sha256(
        _canonical_json(_encoder_fingerprint_payload(encoder)).encode("utf-8")
    ).hexdigest()
    return encoder


def _producer_provenance(
    *,
    exllamav3_root: Path,
    b12x_root: Path,
    vllm_root: Path,
    calibration: FruitCalibrationStore,
) -> dict[str, object]:
    encoder = current_encoder_provenance(
        exllamav3_root=exllamav3_root,
        calibration=calibration,
    )
    runtime = {
        "b12x_revision": _git_revision(b12x_root),
        "b12x_source_sha256": _source_tree_sha256(b12x_root / "b12x"),
        "vllm_revision": _git_revision(vllm_root),
        "vllm_source_sha256": _source_tree_sha256(vllm_root / "vllm"),
    }
    provenance: dict[str, object] = {
        "schema": "kquant_fruit_qsrt_producer_v1",
        "encoder": encoder,
        "runtime": runtime,
    }
    provenance["fingerprint"] = hashlib.sha256(
        _canonical_json(provenance).encode("utf-8")
    ).hexdigest()
    return provenance


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _rate_number(value: object, *, name: str, positive: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
    ):
        raise ValueError(f"Fruit rate-sweep {name} is invalid")
    return float(value)


def _validate_rate_sweep(
    path: Path,
    *,
    source_evidence: dict[str, object],
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Fruit rate sweep: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _RATE_SWEEP_SCHEMA
        or payload.get("complete") is not True
    ):
        raise ValueError("Fruit rate sweep is incomplete or has the wrong schema")
    signature = payload.get("signature")
    results = payload.get("results")
    if not isinstance(signature, dict) or not isinstance(results, dict):
        raise TypeError("Fruit rate sweep omits its signature or results")
    expected_calibration = {
        "capture_id": calibration.capture_id,
        "fingerprint": calibration.fingerprint,
        "manifest_sha256": calibration.manifest_sha256,
    }
    measured_encoder = signature.get("encoder")
    expected_encoder = producer.get("encoder")
    if not isinstance(measured_encoder, dict) or not isinstance(expected_encoder, dict):
        raise TypeError("Fruit rate sweep omits its encoder identity")
    if (
        signature.get("schema") != _RATE_SWEEP_SCHEMA
        or signature.get("source") != source_evidence
        or signature.get("calibration") != expected_calibration
        or measured_encoder.get("fingerprint_schema") != _ENCODER_FINGERPRINT_SCHEMA
        or measured_encoder.get("fingerprint") != expected_encoder.get("fingerprint")
        or signature.get("rates") != [2, 3, 4]
    ):
        raise ValueError("Fruit rate-sweep provenance does not match this build")
    assignments = signature.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("Fruit rate sweep has no assignments")
    assignment_keys: list[str] = []
    for assignment in assignments:
        if (
            not isinstance(assignment, list)
            or len(assignment) != 2
            or any(type(value) is not int for value in assignment)
        ):
            raise TypeError("Fruit rate-sweep assignment is malformed")
        layer, expert = assignment
        if layer not in LAYERS or not 0 <= expert < EXPERTS:
            raise ValueError("Fruit rate-sweep assignment is outside model geometry")
        assignment_keys.append(f"{layer}:{expert}")
    if len(set(assignment_keys)) != len(assignment_keys):
        raise ValueError("Fruit rate-sweep assignments are not unique")
    if set(results) != set(assignment_keys):
        raise ValueError("Fruit rate-sweep results do not cover every assignment")
    measured = 0
    for key in assignment_keys:
        result = results[key]
        if not isinstance(result, dict):
            raise TypeError(f"Fruit rate-sweep result {key} is malformed")
        status = result.get("status")
        if status == "skipped":
            if not isinstance(result.get("reason"), str) or not isinstance(
                result.get("support"), dict
            ):
                raise ValueError(f"Fruit rate-sweep skip {key} lacks evidence")
            continue
        if status != "measured":
            raise ValueError(f"Fruit rate-sweep result {key} has invalid status")
        measured += 1
        rates = result.get("rates")
        if not isinstance(rates, dict) or set(rates) != {"K2", "K3", "K4"}:
            raise ValueError(f"Fruit rate-sweep result {key} omits rate endpoints")
    if measured == 0:
        raise ValueError("Fruit rate sweep contains no measured assignments")
    return payload


def _rate_sweep_section(payload: dict[str, object]) -> str:
    results = payload.get("results")
    signature = payload.get("signature")
    if not isinstance(results, dict) or not isinstance(signature, dict):
        raise TypeError("Fruit rate-sweep summary input is malformed")
    measured = [
        value
        for value in results.values()
        if isinstance(value, dict) and value.get("status") == "measured"
    ]
    assignments = signature.get("assignments")
    if not isinstance(assignments, list) or not measured:
        raise ValueError("Fruit rate-sweep summary has no measured assignments")

    summaries: dict[str, dict[str, float]] = {}
    for rate_name, bits in (("K2", 2), ("K3", 3), ("K4", 4)):
        weight_error = 0.0
        weight_reference = 0.0
        h_error = 0.0
        h_reference = 0.0
        validation_error = 0.0
        validation_reference = 0.0
        endpoint_bytes = 0.0
        endpoint_bpw = 0.0
        for result in measured:
            rates = result.get("rates")
            if not isinstance(rates, dict):
                raise TypeError("Fruit measured rate result is malformed")
            endpoint = rates.get(rate_name)
            if not isinstance(endpoint, dict) or endpoint.get("bits") != bits:
                raise ValueError(f"Fruit {rate_name} endpoint is malformed")
            aggregate = endpoint.get("aggregate")
            matrices = endpoint.get("matrices")
            routed = endpoint.get("routed_function")
            if (
                not isinstance(aggregate, dict)
                or not isinstance(matrices, list)
                or len(matrices) != 3
                or not isinstance(routed, dict)
            ):
                raise TypeError(f"Fruit {rate_name} metric evidence is malformed")
            matrix_names = {
                matrix.get("matrix") for matrix in matrices if isinstance(matrix, dict)
            }
            if matrix_names != {"w1", "w3", "w2"}:
                raise ValueError(f"Fruit {rate_name} matrix evidence is incomplete")
            for matrix in matrices:
                assert isinstance(matrix, dict)
                weight_error += _rate_number(
                    matrix.get("weight_squared_error"),
                    name=f"{rate_name}.weight_squared_error",
                )
                weight_reference += _rate_number(
                    matrix.get("weight_reference_energy"),
                    name=f"{rate_name}.weight_reference_energy",
                    positive=True,
                )
                h_error += _rate_number(
                    matrix.get("captured_dense_h_numerator"),
                    name=f"{rate_name}.captured_dense_h_numerator",
                )
                h_reference += _rate_number(
                    matrix.get("captured_dense_h_denominator"),
                    name=f"{rate_name}.captured_dense_h_denominator",
                    positive=True,
                )
            validation = routed.get("validation")
            if not isinstance(validation, dict):
                raise TypeError(f"Fruit {rate_name} validation evidence is malformed")
            validation_error += _rate_number(
                validation.get("routed_sse"),
                name=f"{rate_name}.validation.routed_sse",
            )
            validation_reference += _rate_number(
                validation.get("reference_energy"),
                name=f"{rate_name}.validation.reference_energy",
                positive=True,
            )
            endpoint_bytes += _rate_number(
                aggregate.get("bytes_before_layer_deduplication"),
                name=f"{rate_name}.bytes",
                positive=True,
            )
            endpoint_bpw += _rate_number(
                aggregate.get("bpw_before_layer_deduplication"),
                name=f"{rate_name}.bpw",
                positive=True,
            )
        summaries[rate_name] = {
            "weight_nmse": weight_error / weight_reference,
            "captured_h_nmse": h_error / h_reference,
            "validation_nmse": validation_error / validation_reference,
            "mean_bytes": endpoint_bytes / len(measured),
            "mean_bpw": endpoint_bpw / len(measured),
        }

    k2 = summaries["K2"]
    k3 = summaries["K3"]
    k4 = summaries["K4"]
    rows = "\n".join(
        "| {rate} | {mean_bpw:.4f} | {mean_bytes:,.0f} | {weight_nmse:.6f} | "
        "{captured_h_nmse:.6f} | {validation_nmse:.6f} |".format(
            rate=rate,
            **summaries[rate],
        )
        for rate in ("K2", "K3", "K4")
    )
    skipped = len(assignments) - len(measured)
    return f"""## Adjacent-rate evidence

`{_RATE_SWEEP_NAME}` re-encodes the same authenticated expert sample at uniform
K2, K3, and K4, with fresh per-endpoint FP16 scales. It measured
{len(measured)} of {len(assignments)} predeclared assignments; {skipped} lacked
the minimum routed calibration support and were skipped rather than imputed.
These are pre-layer-deduplication expert-local endpoints, not package sizes.

| Endpoint | Mean bpw | Mean bytes/expert | Weight NMSE | Captured-H NMSE | Validation routed NMSE |
|---|---:|---:|---:|---:|---:|
{rows}

Relative to K3, K2 is {k2["weight_nmse"] / k3["weight_nmse"]:.3f}x /
{k2["captured_h_nmse"] / k3["captured_h_nmse"]:.3f}x /
{k2["validation_nmse"] / k3["validation_nmse"]:.3f}x on weight,
captured-H, and validation-routed NMSE. K4 is
{k4["weight_nmse"] / k3["weight_nmse"]:.3f}x /
{k4["captured_h_nmse"] / k3["captured_h_nmse"]:.3f}x /
{k4["validation_nmse"] / k3["validation_nmse"]:.3f}x on the same metrics.
"""


def _render_model_card(
    *,
    source_evidence: dict[str, object],
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
    rate_sweep: dict[str, object],
    layers: dict[str, dict[str, object]],
) -> str:
    format_counts: Counter[str] = Counter()
    elapsed_seconds = 0.0
    peak_cuda_bytes = 0
    encoded_experts = 0
    for layer in layers.values():
        counts = layer.get("format_counts")
        experts = layer.get("experts")
        if not isinstance(counts, dict) or not isinstance(experts, list):
            raise TypeError("Fruit layer evidence cannot render the model card")
        for name, count in counts.items():
            if not isinstance(name, str) or type(count) is not int or count < 0:
                raise TypeError("Fruit allocation evidence is malformed")
            format_counts[name] += count
        for expert in experts:
            if not isinstance(expert, dict):
                raise TypeError("Fruit expert evidence is malformed")
            elapsed = expert.get("encode_elapsed_seconds")
            peak = expert.get("peak_cuda_bytes")
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
                or type(peak) is not int
                or peak < 0
            ):
                raise ValueError("Fruit encoding telemetry is malformed")
            elapsed_seconds += float(elapsed)
            peak_cuda_bytes = max(peak_cuda_bytes, peak)
            encoded_experts += 1
    expected_experts = len(LAYERS) * EXPERTS
    if (
        encoded_experts != expected_experts
        or sum(format_counts.values()) != expected_experts
    ):
        raise ValueError("Fruit model-card expert evidence is incomplete")

    documents = calibration.manifest.get("documents")
    if not isinstance(documents, list) or not all(
        isinstance(document, dict)
        and type(document.get("tokens")) is int
        and int(document["tokens"]) > 0
        for document in documents
    ):
        raise TypeError("Fruit calibration document evidence is malformed")
    calibration_tokens = sum(int(document["tokens"]) for document in documents)
    encoder = producer.get("encoder")
    runtime = producer.get("runtime")
    if not isinstance(encoder, dict) or not isinstance(runtime, dict):
        raise TypeError("Fruit producer evidence is malformed")
    replacements = {
        "__ALLOCATION_ROWS__": "\n".join(
            f"| `{name}` | {count:,} |" for name, count in sorted(format_counts.items())
        ),
        "__RATE_SWEEP_SECTION__": _rate_sweep_section(rate_sweep),
        "__KQUANT_REVISION__": str(encoder["kquant_revision"]),
        "__B12X_REVISION__": str(runtime["b12x_revision"]),
        "__VLLM_REVISION__": str(runtime["vllm_revision"]),
        "__SOURCE_KIND__": str(source_evidence["source_kind"]),
        "__SOURCE_SHA256__": str(source_evidence["source_sha256"]),
        "__CALIBRATION_CAPTURE_ID__": calibration.capture_id,
        "__CALIBRATION_MANIFEST_SHA256__": calibration.manifest_sha256,
        "__CALIBRATION_DOCUMENTS__": f"{len(documents):,}",
        "__CALIBRATION_TOKENS__": f"{calibration_tokens:,}",
        "__ENCODED_EXPERTS__": f"{encoded_experts:,}",
        "__ENCODE_SECONDS__": f"{elapsed_seconds:,.2f}",
        "__PEAK_GIB__": f"{peak_cuda_bytes / (1 << 30):.3f}",
    }
    card = MODEL_CARD_TEMPLATE
    for marker, value in replacements.items():
        if marker not in card:
            raise ValueError(f"Fruit model-card marker is missing: {marker}")
        card = card.replace(marker, value)
    return card


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(tensors, temporary, metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _part_paths(output: Path, layer: int, expert: int) -> tuple[Path, Path]:
    root = output / ".qsrt-parts" / f"layer-{layer:03d}"
    return root / f"expert-{expert:03d}.safetensors", root / f"expert-{expert:03d}.json"


def _validate_qsrt_tensor_contract(
    tensor_path: Path,
    *,
    expected_expert_ids: torch.Tensor,
) -> dict[str, str]:
    expected_count = int(expected_expert_ids.numel())
    expected_shapes = {
        "expert_ids": (expected_count,),
        "formats": (expected_count, 2),
        "permutations": (expected_count, INTERMEDIATE_SIZE),
        "w13_trellis": (2, expected_count, PAIR_COUNT, PAIR_WORDS),
        "w2_trellis": (expected_count, PAIR_COUNT, PAIR_WORDS),
        "fc1_pair_modes": (expected_count, PAIR_COUNT),
        "fc2_pair_modes": (expected_count, PAIR_COUNT),
        "gate_suh": (expected_count, HIDDEN_SIZE),
        "up_suh": (expected_count, HIDDEN_SIZE),
        "intermediate_rotations": (expected_count, 3 * INTERMEDIATE_SIZE),
        "down_svh": (expected_count, HIDDEN_SIZE),
    }
    expected_dtypes = {
        "expert_ids": torch.int32,
        "formats": torch.int8,
        "permutations": torch.int16,
        "w13_trellis": torch.int16,
        "w2_trellis": torch.int16,
        "fc1_pair_modes": torch.int32,
        "fc2_pair_modes": torch.int32,
        "gate_suh": torch.float16,
        "up_suh": torch.float16,
        "intermediate_rotations": torch.float16,
        "down_svh": torch.float16,
    }
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        if keys != set(FRUIT_QSRT_ARTIFACT_TENSORS):
            raise ValueError(f"Fruit QSRT tensor inventory mismatch: {tensor_path}")
        tensors = {
            name: handle.get_tensor(name) for name in FRUIT_QSRT_ARTIFACT_TENSORS
        }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"Fruit QSRT {name} shape mismatch in {tensor_path}: "
                f"{tuple(tensor.shape)} != {expected_shapes[name]}"
            )
        if tensor.dtype != expected_dtypes[name]:
            raise TypeError(
                f"Fruit QSRT {name} dtype mismatch in {tensor_path}: "
                f"{tensor.dtype} != {expected_dtypes[name]}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"Fruit QSRT {name} is noncontiguous in {tensor_path}")
    if not torch.equal(tensors["expert_ids"], expected_expert_ids):
        raise ValueError(f"Fruit QSRT expert IDs mismatch in {tensor_path}")
    formats = tensors["formats"]
    if bool(((formats < 0) | (formats > 2)).any().item()):
        raise ValueError(f"Fruit QSRT formats are outside R0/R1/R2 in {tensor_path}")
    mode_table = torch.tensor(((0, 0), (1, 0), (1, 1)), dtype=torch.int32)
    if not torch.equal(
        tensors["fc1_pair_modes"], mode_table.index_select(0, formats[:, 0].long())
    ) or not torch.equal(
        tensors["fc2_pair_modes"], mode_table.index_select(0, formats[:, 1].long())
    ):
        raise ValueError(
            f"Fruit QSRT pair modes disagree with formats in {tensor_path}"
        )
    expected_permutation = torch.arange(INTERMEDIATE_SIZE, dtype=torch.int16)
    if not bool(
        torch.all(
            torch.sort(tensors["permutations"], dim=1).values == expected_permutation
        ).item()
    ):
        raise ValueError(f"Fruit QSRT permutations are not bijections in {tensor_path}")
    return metadata


def _align_storage(value: int) -> int:
    return (
        (value + FRUIT_QSRT_STORAGE_ALIGNMENT - 1)
        // FRUIT_QSRT_STORAGE_ALIGNMENT
        * FRUIT_QSRT_STORAGE_ALIGNMENT
    )


def _fruit_atom_metadata(
    *, layer: int, source_sha256: str, encoder_fingerprint: str
) -> dict[str, str]:
    atom_payload_bytes = EXPERTS * FRUIT_QSRT_ATOM_BUNDLE_BYTES
    shared_scale_bytes = 3 * EXPERTS * HIDDEN_SIZE * torch.float16.itemsize
    return {
        "schema": FRUIT_QSRT_ATOM_SCHEMA,
        "version": "1",
        "encoding": "qsrt_sqg_e4m3",
        "profile_id": str(FRUIT_QSRT_PROFILE_ID),
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": str(layer),
        "experts": str(EXPERTS),
        "compressed_experts": str(EXPERTS),
        "x4t_experts": "0",
        "intermediate_channels": str(INTERMEDIATE_SIZE),
        "latent_channels": str(HIDDEN_SIZE),
        "record_channels": "128",
        "pair_count": str(PAIR_COUNT),
        "atom_channels": str(FRUIT_QSRT_ATOM_CHANNELS),
        "atom_slots": str(FRUIT_QSRT_ATOMS_PER_EXPERT),
        "atom_bundle_bytes": str(FRUIT_QSRT_ATOM_BUNDLE_BYTES),
        "atom_slot_payload_bytes": str(atom_payload_bytes),
        "atom_slot_stride_bytes": str(_align_storage(atom_payload_bytes)),
        "format_section_bytes": str(FRUIT_QSRT_FORMAT_SECTION_BYTES),
        "shared_scale_rows": str(EXPERTS),
        "shared_scale_section_bytes": str(_align_storage(shared_scale_bytes)),
        "alignment_bytes": str(FRUIT_QSRT_STORAGE_ALIGNMENT),
        "rotation_multiplier": "5",
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }


def _validate_fruit_atom_contract(
    tensor_path: Path,
    *,
    layer: int,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, str]:
    expected_metadata = {
        "format": "pt",
        **_fruit_atom_metadata(
            layer=layer,
            source_sha256=source_sha256,
            encoder_fingerprint=encoder_fingerprint,
        ),
    }
    shared_scale_bytes = 3 * EXPERTS * HIDDEN_SIZE * torch.float16.itemsize
    atom_payload_bytes = EXPERTS * FRUIT_QSRT_ATOM_BUNDLE_BYTES
    atom_stride_bytes = _align_storage(atom_payload_bytes)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(handle.keys()) != set(FRUIT_QSRT_ATOM_TENSORS):
            raise ValueError(
                f"Fruit QSRT atom tensor inventory mismatch: {tensor_path}"
            )
        for name, expected_value in expected_metadata.items():
            if metadata.get(name) != expected_value:
                raise ValueError(
                    f"Fruit QSRT atom metadata {name} mismatch in {tensor_path}"
                )
        format_section = handle.get_tensor(FRUIT_QSRT_FORMAT_TENSOR)
        shared_section = handle.get_tensor(FRUIT_QSRT_SHARED_SCALE_TENSOR)
        atom_shape = handle.get_slice(FRUIT_QSRT_ATOM_TENSOR).get_shape()
    if (
        format_section.dtype != torch.uint8
        or tuple(format_section.shape) != (FRUIT_QSRT_FORMAT_SECTION_BYTES,)
        or bool(torch.any(format_section[EXPERTS:] != 0))
    ):
        raise ValueError(f"Fruit QSRT format section is malformed: {tensor_path}")
    codes = format_section[:EXPERTS]
    r13 = codes >> 4
    r2 = codes & 0xF
    if bool(torch.any((r13 > 2) | (r2 > 2))):
        raise ValueError(f"Fruit QSRT format codes are invalid: {tensor_path}")
    expected_shared_section = _align_storage(shared_scale_bytes)
    if (
        shared_section.dtype != torch.uint8
        or tuple(shared_section.shape) != (expected_shared_section,)
        or bool(torch.any(shared_section[shared_scale_bytes:] != 0))
    ):
        raise ValueError(f"Fruit QSRT shared-scale section is malformed: {tensor_path}")
    shared = (
        shared_section[:shared_scale_bytes]
        .view(torch.float16)
        .reshape(3, EXPERTS, HIDDEN_SIZE)
    )
    if not bool(torch.all(torch.isfinite(shared))):
        raise ValueError(f"Fruit QSRT shared scales are non-finite: {tensor_path}")
    expected_atom_shape = [
        FRUIT_QSRT_ATOMS_PER_EXPERT,
        atom_stride_bytes,
    ]
    if atom_shape != expected_atom_shape:
        raise ValueError(
            f"Fruit QSRT atom slab shape {atom_shape} != {expected_atom_shape}"
        )
    return metadata


def _validate_part(
    tensor_path: Path,
    manifest_path: Path,
    *,
    layer: int,
    expert: int,
    source_sha256: str,
    encoder_fingerprint: str,
    repair: bool = True,
) -> dict[str, object] | None:
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
        if repair:
            for path in (tensor_path, manifest_path):
                if path.is_file():
                    path.unlink()
                elif path.exists():
                    raise ValueError(f"unexpected Fruit QSRT part path: {path}")
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"malformed Fruit QSRT part manifest: {manifest_path}"
        ) from exc
    expected = {
        "schema": FRUIT_QSRT_SCHEMA,
        "version": 1,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": layer,
        "expert": expert,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    mismatches = {
        name
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if mismatches:
        if mismatches == {"encoder_fingerprint"} and repair:
            tensor_path.unlink()
            manifest_path.unlink()
            return None
        if not repair:
            return None
        raise ValueError(f"Fruit QSRT part identity mismatch: {manifest_path}")
    if value.get("safetensors_bytes") != tensor_path.stat().st_size:
        raise ValueError(f"Fruit QSRT part size mismatch: {tensor_path}")
    if value.get("safetensors_sha256") != _sha256(tensor_path):
        raise ValueError(f"Fruit QSRT part hash mismatch: {tensor_path}")
    metadata = _validate_qsrt_tensor_contract(
        tensor_path,
        expected_expert_ids=torch.tensor([expert], dtype=torch.int32),
    )
    for name, expected_value in (
        ("schema", FRUIT_QSRT_SCHEMA),
        ("version", "1"),
        ("profile_id", str(FRUIT_QSRT_PROFILE_ID)),
        ("codebook", FRUIT_QSRT_CODEBOOK),
        ("layer", str(layer)),
        ("expert", str(expert)),
        ("source_sha256", source_sha256),
    ):
        if metadata.get(name) != expected_value:
            raise ValueError(f"Fruit QSRT part metadata mismatch: {tensor_path}")
    if metadata.get("encoder_fingerprint") != encoder_fingerprint:
        if not repair:
            return None
        raise ValueError(f"Fruit QSRT part producer mismatch: {tensor_path}")
    return value


def _write_part(
    output: Path,
    encoding,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
    elapsed_seconds: float,
    peak_cuda_bytes: int,
) -> dict[str, object]:
    tensor_path, manifest_path = _part_paths(output, encoding.layer, encoding.expert)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_safetensors(
        tensor_path,
        encoding.artifact_tensors(),
        {
            "schema": FRUIT_QSRT_SCHEMA,
            "version": "1",
            "profile_id": str(FRUIT_QSRT_PROFILE_ID),
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": str(encoding.layer),
            "expert": str(encoding.expert),
            "source_sha256": source_sha256,
            "encoder_fingerprint": encoder_fingerprint,
        },
    )
    manifest = {
        **encoding.manifest(),
        "version": 1,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
        "safetensors_file": tensor_path.name,
        "safetensors_bytes": tensor_path.stat().st_size,
        "safetensors_sha256": _sha256(tensor_path),
        "encode_elapsed_seconds": elapsed_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
    }
    _atomic_text(manifest_path, _canonical_json(manifest))
    return manifest


def _seed_parts(
    output: Path,
    seed: Path | None,
    source_sha256: str,
    encoder_fingerprint: str,
) -> int:
    if seed is None:
        return 0
    copied = 0
    for layer in LAYERS:
        for expert in range(EXPERTS):
            source_tensor = (
                seed / f"layer-{layer:03d}" / f"expert-{expert:03d}.safetensors"
            )
            source_manifest = seed / f"layer-{layer:03d}" / f"expert-{expert:03d}.json"
            if not source_tensor.is_file() and not source_manifest.is_file():
                continue
            target_tensor, target_manifest = _part_paths(output, layer, expert)
            if target_tensor.exists() or target_manifest.exists():
                continue
            if not source_tensor.is_file() or not source_manifest.is_file():
                continue
            try:
                valid = _validate_part(
                    source_tensor,
                    source_manifest,
                    layer=layer,
                    expert=expert,
                    source_sha256=source_sha256,
                    encoder_fingerprint=encoder_fingerprint,
                    repair=False,
                )
            except (OSError, TypeError, ValueError):
                valid = None
            if valid is None:
                continue
            target_tensor.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source_tensor, target_tensor)
                shutil.copy2(source_manifest, target_manifest)
            except OSError as exc:
                for target in (target_tensor, target_manifest):
                    if target.is_file():
                        target.unlink()
                    elif target.exists():
                        raise ValueError(
                            f"unexpected Fruit QSRT part path: {target}"
                        ) from exc
                continue
            copied += 1
    return copied


def _parts_need_encoder(
    output: Path,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
) -> bool:
    expected = {
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    for layer in LAYERS:
        for expert in range(EXPERTS):
            tensor_path, manifest_path = _part_paths(output, layer, expert)
            if not tensor_path.is_file() or not manifest_path.is_file():
                return True
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return True
            if not isinstance(value, dict) or any(
                value.get(name) != expected_value
                for name, expected_value in expected.items()
            ):
                return True
    return False


def _validate_source_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("sealed Fruit source evidence must be a JSON object")
    expected: dict[str, object] = {
        "kind": "kquant-fruit-source-preflight",
        "version": 1,
        "status": "pass",
        "model_id": FRUIT_ANNEALED_SPEC.model_id,
    }
    if value.get("source_container") == "hf_bf16_safetensors":
        expected.update(
            {
                "checkpoint_sha256": BASE_MANIFEST_SHA256,
                "checkpoint_sha256_provenance": ("safetensors_manifest_authenticated"),
                "source_sha256": BASE_MANIFEST_SHA256,
                "source_kind": "safetensors_manifest",
                "expected_checkpoint_sha256": (FRUIT_ANNEALED_SPEC.checkpoint_sha256),
                "safetensors_manifest_sha256": BASE_MANIFEST_SHA256,
            }
        )
    elif value.get("source_container") in ("model", "state_dict"):
        expected.update(
            {
                "checkpoint_sha256": FRUIT_ANNEALED_SPEC.checkpoint_sha256,
                "checkpoint_sha256_provenance": "checkpoint_file_authenticated",
                "source_sha256": FRUIT_ANNEALED_SPEC.checkpoint_sha256,
                "source_kind": "torch_checkpoint",
            }
        )
    else:
        raise ValueError("sealed Fruit source evidence has an unsupported container")
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(
                f"sealed Fruit source evidence {name!r} mismatch: "
                f"{value.get(name)!r} != {expected_value!r}"
            )
    return value


def _write_source_evidence_seal(
    root: Path,
    source_evidence: dict[str, object],
    producer: dict[str, object],
) -> None:
    source_evidence = _validate_source_evidence(source_evidence)
    encoder = producer.get("encoder")
    if not isinstance(encoder, dict) or not isinstance(encoder.get("fingerprint"), str):
        raise TypeError("Fruit QSRT producer has no encoder fingerprint")
    if not isinstance(producer.get("fingerprint"), str):
        raise TypeError("Fruit QSRT producer has no package fingerprint")
    envelope = {
        "schema": "kquant_fruit_source_seal_v1",
        "producer_fingerprint": producer["fingerprint"],
        "encoder_fingerprint": encoder["fingerprint"],
        "source": source_evidence,
    }
    content = _canonical_json(envelope)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    _atomic_text(root / _SOURCE_EVIDENCE_NAME, content)
    _atomic_text(
        root / _SOURCE_EVIDENCE_SHA_NAME,
        f"{digest}  {_SOURCE_EVIDENCE_NAME}\n",
    )


def _read_source_evidence_seal(
    root: Path | None,
    producer: dict[str, object],
) -> dict[str, object] | None:
    if root is None:
        return None
    evidence_path = root / _SOURCE_EVIDENCE_NAME
    seal_path = root / _SOURCE_EVIDENCE_SHA_NAME
    if not evidence_path.exists() and not seal_path.exists():
        return None
    if not evidence_path.is_file() or not seal_path.is_file():
        raise ValueError(f"incomplete Fruit source evidence seal in {root}")
    fields = seal_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != _SOURCE_EVIDENCE_NAME:
        raise ValueError(f"malformed Fruit source evidence seal {seal_path}")
    if fields[0] != _sha256(evidence_path):
        raise ValueError(f"Fruit source evidence seal mismatch for {evidence_path}")
    envelope = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise TypeError("Fruit source evidence envelope must be a JSON object")
    encoder = producer["encoder"]
    if (
        envelope.get("schema") != "kquant_fruit_source_seal_v1"
        or envelope.get("encoder_fingerprint") != encoder["fingerprint"]
        or not isinstance(envelope.get("producer_fingerprint"), str)
    ):
        raise ValueError(f"Fruit source evidence producer mismatch in {evidence_path}")
    return _validate_source_evidence(envelope.get("source"))


def _write_calibration_evidence(
    output: Path,
    calibration: FruitCalibrationStore,
    producer: dict[str, object],
) -> None:
    encoder = producer.get("encoder")
    if not isinstance(encoder, dict):
        raise TypeError("Fruit QSRT producer encoder evidence is invalid")
    expected = {
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_capture_id": calibration.capture_id,
        "calibration_manifest_sha256": calibration.manifest_sha256,
    }
    if any(encoder.get(name) != value for name, value in expected.items()):
        raise ValueError("Fruit calibration evidence disagrees with the producer")
    _atomic_text(
        output / _CALIBRATION_EVIDENCE_NAME,
        _canonical_json(
            {
                "schema": "kquant_fruit_qsrt_calibration_reference_v1",
                "producer_fingerprint": producer["fingerprint"],
                "encoder_fingerprint": encoder["fingerprint"],
                **expected,
                "raw_activation_layers_included": False,
                "capture_manifest": calibration.manifest,
            }
        ),
    )


def _encode_parts(
    output: Path,
    store: FruitMatrixStore | None,
    quantizer_module,
    calibration_store: FruitCalibrationStore,
    *,
    device: torch.device,
    source_sha256: str,
    encoder_fingerprint: str,
) -> None:
    total = len(LAYERS) * EXPERTS
    ordinal = 0
    started = time.perf_counter()
    for layer in LAYERS:
        calibration_layer = calibration_store.load_layer(layer)
        for expert in range(EXPERTS):
            ordinal += 1
            tensor_path, manifest_path = _part_paths(output, layer, expert)
            previous = _validate_part(
                tensor_path,
                manifest_path,
                layer=layer,
                expert=expert,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
            )
            if previous is not None:
                print(
                    f"[{ordinal}/{total}] layer {layer} expert {expert}: resume",
                    flush=True,
                )
                continue
            if quantizer_module is None or store is None:
                raise RuntimeError(
                    "authenticated Fruit source was not loaded for a missing part"
                )
            torch.cuda.reset_peak_memory_stats(device)
            item_started = time.perf_counter()
            encoding = encode_fruit_expert(
                store,
                layer=layer,
                expert=expert,
                device=device,
                calibration=calibration_layer.expert_rows(expert),
                quantizer_module=quantizer_module,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - item_started
            peak = torch.cuda.max_memory_allocated(device)
            _write_part(
                output,
                encoding,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
                elapsed_seconds=elapsed,
                peak_cuda_bytes=peak,
            )
            print(
                f"[{ordinal}/{total}] layer {layer} expert {expert}: "
                f"R13={encoding.r13} R2={encoding.r2}, {elapsed:.3f}s, "
                f"peak={peak / (1 << 20):.1f} MiB, "
                f"run={time.perf_counter() - started:.1f}s",
                flush=True,
            )
            del encoding
        del calibration_layer


def _layer_paths(output: Path, layer: int) -> tuple[Path, Path]:
    return (
        output / f"qsrt-layer-{layer:03d}.safetensors",
        output / f"qsrt-layer-{layer:03d}.json",
    )


def _validate_layer(
    tensor_path: Path,
    manifest_path: Path,
    *,
    layer: int,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, object] | None:
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
        for path in (tensor_path, manifest_path):
            if path.is_file():
                path.unlink()
            elif path.exists():
                raise ValueError(f"unexpected assembled Fruit QSRT path: {path}")
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"malformed assembled Fruit QSRT manifest: {manifest_path}"
        ) from exc
    expected = {
        "schema": FRUIT_QSRT_ATOM_SCHEMA,
        "version": 1,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "codebook": FRUIT_QSRT_CODEBOOK,
        "layer": layer,
        "expert_ids": list(range(EXPERTS)),
        "expert_count": EXPERTS,
        "source_sha256": source_sha256,
        "encoder_fingerprint": encoder_fingerprint,
    }
    mismatches = {
        name
        for name, expected_value in expected.items()
        if value.get(name) != expected_value
    }
    if mismatches:
        if mismatches == {"encoder_fingerprint"} or value.get("schema") == (
            FRUIT_QSRT_SCHEMA
        ):
            tensor_path.unlink()
            manifest_path.unlink()
            return None
        raise ValueError(
            f"assembled Fruit QSRT layer identity mismatch: {manifest_path}"
        )
    if value.get("safetensors_bytes") != tensor_path.stat().st_size:
        raise ValueError(f"assembled Fruit QSRT layer size mismatch: {tensor_path}")
    if value.get("safetensors_sha256") != _sha256(tensor_path):
        raise ValueError(f"assembled Fruit QSRT layer hash mismatch: {tensor_path}")
    _validate_fruit_atom_contract(
        tensor_path,
        layer=layer,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    experts = value.get("experts")
    if not isinstance(experts, list) or len(experts) != EXPERTS:
        raise ValueError(
            f"assembled Fruit QSRT expert evidence mismatch: {manifest_path}"
        )
    for expert, evidence in enumerate(experts):
        if not isinstance(evidence, dict) or any(
            evidence.get(name) != expected_value
            for name, expected_value in (
                ("layer", layer),
                ("expert", expert),
                ("source_sha256", source_sha256),
                ("encoder_fingerprint", encoder_fingerprint),
            )
        ):
            raise ValueError(
                f"assembled Fruit QSRT expert {expert} identity mismatch: "
                f"{manifest_path}"
            )
    return value


def _assemble_layers(
    output: Path,
    *,
    source_sha256: str,
    encoder_fingerprint: str,
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for layer in LAYERS:
        tensor_path, manifest_path = _layer_paths(output, layer)
        previous = _validate_layer(
            tensor_path,
            manifest_path,
            layer=layer,
            source_sha256=source_sha256,
            encoder_fingerprint=encoder_fingerprint,
        )
        if previous is not None:
            results[str(layer)] = previous
            print(f"layer {layer}: assembled artifact resume", flush=True)
            continue
        parts: dict[str, list[torch.Tensor]] = defaultdict(list)
        experts: list[dict[str, object]] = []
        for expert in range(EXPERTS):
            part_tensor, part_manifest = _part_paths(output, layer, expert)
            evidence = _validate_part(
                part_tensor,
                part_manifest,
                layer=layer,
                expert=expert,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
            )
            if evidence is None:
                raise ValueError(f"Fruit QSRT layer {layer} is missing expert {expert}")
            experts.append(evidence)
            with safe_open(part_tensor, framework="pt", device="cpu") as handle:
                for name in FRUIT_QSRT_ARTIFACT_TENSORS:
                    parts[name].append(handle.get_tensor(name))
        pair_tensors = {
            name: torch.cat(values, dim=1 if name == "w13_trellis" else 0).contiguous()
            for name, values in parts.items()
        }
        tensors = pack_fruit_atom_layer(pair_tensors, layer=layer)
        _atomic_safetensors(
            tensor_path,
            tensors,
            {
                "format": "pt",
                **_fruit_atom_metadata(
                    layer=layer,
                    source_sha256=source_sha256,
                    encoder_fingerprint=encoder_fingerprint,
                ),
            },
        )
        format_counts = Counter(
            (int(expert["format"]["r13"]), int(expert["format"]["r2"]))
            for expert in experts
        )
        manifest: dict[str, object] = {
            "schema": FRUIT_QSRT_ATOM_SCHEMA,
            "version": 1,
            "profile_id": FRUIT_QSRT_PROFILE_ID,
            "codebook": FRUIT_QSRT_CODEBOOK,
            "layer": layer,
            "expert_ids": list(range(EXPERTS)),
            "expert_count": EXPERTS,
            "source_sha256": source_sha256,
            "encoder_fingerprint": encoder_fingerprint,
            "tensor_shapes": {
                name: list(tensor.shape) for name, tensor in tensors.items()
            },
            "format_counts": {
                f"R13={r13},R2={r2}": count
                for (r13, r2), count in sorted(format_counts.items())
            },
            "safetensors_file": tensor_path.name,
            "safetensors_bytes": tensor_path.stat().st_size,
            "safetensors_sha256": _sha256(tensor_path),
            "experts": experts,
        }
        _atomic_text(manifest_path, _canonical_json(manifest))
        results[str(layer)] = manifest
        print(
            f"layer {layer}: assembled {tensor_path.stat().st_size / (1 << 20):.2f} MiB",
            flush=True,
        )
    return results


def _authenticate_base_model(base_model: Path) -> dict[str, object]:
    manifest_path = base_model / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != BASE_MANIFEST_SHA256:
        raise ValueError(
            "Fruit BF16 base manifest identity mismatch: "
            f"{manifest_sha256} != {BASE_MANIFEST_SHA256}"
        )
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"malformed Fruit BF16 base manifest line: {line!r}")
        digest, filename = fields
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or Path(filename).name != filename
            or filename in entries
        ):
            raise ValueError(f"invalid Fruit BF16 base manifest line: {line!r}")
        entries[filename] = digest
    required_static = {
        "LICENSE",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not required_static.issubset(entries):
        raise ValueError("Fruit BF16 base manifest omits required static files")
    for filename, expected_sha256 in entries.items():
        path = base_model / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Fruit BF16 base hash mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    index = json.loads(
        (base_model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("Fruit BF16 base index has no weight_map")
    shard_names = set(weight_map.values())
    if not shard_names.issubset(entries):
        raise ValueError("Fruit BF16 base manifest omits indexed weight shards")
    config = json.loads((base_model / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "dtype": "bfloat16",
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": INTERMEDIATE_SIZE,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 13,
        "vocab_size": 154_880,
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("Fruit BF16 base model geometry mismatch")
    return {
        "schema": "kquant_fruit_bf16_base_v1",
        "manifest_file": "MANIFEST.sha256",
        "manifest_sha256": manifest_sha256,
        "file_count": len(entries),
        "files": entries,
        "config_identity": expected_config,
    }


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        if target.samefile(source):
            return
        if target.stat().st_size != source.stat().st_size or _sha256(target) != _sha256(
            source
        ):
            raise ValueError(
                f"existing static model file differs from source: {target}"
            )
        return
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def _tensor_nbytes(handle, name: str) -> int:
    tensor_slice = handle.get_slice(name)
    dtype = tensor_slice.get_dtype()
    try:
        itemsize = _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}") from exc
    elements = 1
    for extent in tensor_slice.get_shape():
        elements *= int(extent)
    return elements * itemsize


def _strip_routed_experts(source: Path, target: Path) -> tuple[set[str], int]:
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        names = handle.keys()
        expert_names = {name for name in names if ".mlp.experts." in name}
        if len(expert_names) != 3 * EXPERTS:
            raise ValueError(
                f"expected {3 * EXPERTS} routed-expert tensors in {source}, "
                f"got {len(expert_names)}"
            )
        removed_bytes = sum(_tensor_nbytes(handle, name) for name in expert_names)
        kept = {
            name: handle.get_tensor(name) for name in names if name not in expert_names
        }
    _atomic_safetensors(target, kept, metadata)
    return expert_names, removed_bytes


def _materialize_base_model(base_model: Path, output: Path) -> None:
    index_path = base_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("base model safetensors index has no weight_map")
    removed_names: set[str] = set()
    removed_bytes = 0
    source_files = sorted(set(weight_map.values()))
    for filename in source_files:
        source = base_model / filename
        target = output / filename
        if filename.startswith("model-layer-"):
            layer_text = filename.removeprefix("model-layer-").removesuffix(
                ".safetensors"
            )
            try:
                layer = int(layer_text)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Fruit BF16 model layer shard filename: {filename}"
                ) from exc
        else:
            layer = -1
        if layer in LAYERS:
            names, byte_count = _strip_routed_experts(source, target)
            removed_names.update(names)
            removed_bytes += byte_count
        else:
            _link_or_copy(source, target)
    expected_removed = 3 * EXPERTS * len(LAYERS)
    if len(removed_names) != expected_removed:
        raise ValueError(
            f"expected {expected_removed} routed-expert weights, got {len(removed_names)}"
        )
    filtered_map = {
        name: filename
        for name, filename in weight_map.items()
        if name not in removed_names
    }
    if len(weight_map) - len(filtered_map) != expected_removed:
        raise ValueError("base index routed-expert inventory disagrees with shards")
    metadata = dict(index.get("metadata") or {})
    total_size = int(metadata.get("total_size", 0))
    if total_size <= removed_bytes:
        raise ValueError("base index total_size is invalid")
    metadata["total_size"] = total_size - removed_bytes
    _atomic_text(
        output / "model.safetensors.index.json",
        _canonical_json({"metadata": metadata, "weight_map": filtered_map}),
    )
    for filename in (
        "LICENSE",
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        _link_or_copy(base_model / filename, output / filename)


def _write_config(
    base_model: Path,
    output: Path,
    producer: dict[str, object],
    source_evidence: dict[str, object],
) -> None:
    config = json.loads((base_model / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "hybrid_bit_map": {str(layer): [3] * EXPERTS for layer in LAYERS},
        "kept_format": "mxfp4_e8m0k32",
        "demoted_format": "qsrt_sqg_e4m3",
        "qsrt": {
            "schema": FRUIT_QSRT_ATOM_SCHEMA,
            "storage_format": FRUIT_QSRT_ATOM_STORAGE,
            "encoding": "qsrt_sqg_e4m3",
            "codebook": FRUIT_QSRT_CODEBOOK,
            "profile_id": FRUIT_QSRT_PROFILE_ID,
            "artifact_manifest": "qsrt-manifest.json",
            "producer_fingerprint": producer["fingerprint"],
            "encoder_fingerprint": producer["encoder"]["fingerprint"],
            "source_kind": source_evidence["source_kind"],
            "source_sha256": source_evidence["source_sha256"],
            "runtime": "w4a8",
        },
    }
    _atomic_text(output / "config.json", _canonical_json(config))


def _remove_part_cache(output: Path) -> None:
    parts = output / ".qsrt-parts"
    if parts.is_symlink():
        raise ValueError(f"Fruit QSRT part cache must not be a symbolic link: {parts}")
    if not parts.exists():
        return
    if not parts.is_dir():
        raise ValueError(f"Fruit QSRT part cache is not a directory: {parts}")
    shutil.rmtree(parts)


def _package_files(output: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in output.iterdir():
        if path.is_symlink():
            raise ValueError(f"Fruit package path must not be a symbolic link: {path}")
        if path.is_file() and path.name not in {
            "MANIFEST.sha256",
            _COMPLETE_MARKER_NAME,
        }:
            files[path.name] = path
    evaluation = output / "evaluation"
    if evaluation.is_dir():
        for path in evaluation.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    f"Fruit evaluation path must not be a symbolic link: {path}"
                )
            if path.is_file():
                relative = path.relative_to(output).as_posix()
                files[relative] = path
    return dict(sorted(files.items()))


def _write_package_manifests(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    calibration: FruitCalibrationStore,
    rate_sweep: dict[str, object],
    layers: dict[str, dict[str, object]],
) -> None:
    evaluation = output / "evaluation"
    if evaluation.exists():
        shutil.rmtree(evaluation)
    evaluation.mkdir()
    rate_sweep_path = output / _RATE_SWEEP_NAME
    _atomic_text(rate_sweep_path, _canonical_json(rate_sweep))
    manifest_layers = {
        layer: {
            "qsrt_atoms": value["safetensors_file"],
            "bytes": value["safetensors_bytes"],
            "sha256": value["safetensors_sha256"],
            "expert_count": value["expert_count"],
            "evidence": f"qsrt-layer-{int(layer):03d}.json",
        }
        for layer, value in sorted(layers.items(), key=lambda item: int(item[0]))
    }
    manifest = {
        "schema": "kquant_qsrt_model_manifest_v1",
        "version": 1,
        "codec": "QSRT",
        "storage_schema": FRUIT_QSRT_ATOM_SCHEMA,
        "encoding": "qsrt_sqg_e4m3",
        "codebook": FRUIT_QSRT_CODEBOOK,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "geometry": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "topk": 8,
        },
        "runtime": {
            "tensor_parallel": "whole_atom_partition",
            "validated_tensor_parallel_sizes": [1],
            "decode": "trellis_w4a8",
            "decode_max_tokens": 16,
            "fallback": "trellis_w4a16",
            "prefill": "trellis_w4a16",
        },
        "base_model": base_provenance,
        "producer": producer,
        "source": source_evidence,
        "evaluation": {
            "uniform_rate_sweep": {
                "file": _RATE_SWEEP_NAME,
                "sha256": _sha256(rate_sweep_path),
            }
        },
        "layers": manifest_layers,
        "complete": True,
    }
    _atomic_text(output / "qsrt-manifest.json", _canonical_json(manifest))
    _atomic_text(
        output / "README.md",
        _render_model_card(
            source_evidence=source_evidence,
            calibration=calibration,
            producer=producer,
            rate_sweep=rate_sweep,
            layers=layers,
        ),
    )
    entries = [
        f"{_sha256(path)}  {relative}"
        for relative, path in _package_files(output).items()
    ]
    _atomic_text(output / "MANIFEST.sha256", "\n".join(entries) + "\n")


def _completion_record(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "kquant_qsrt_complete_v2",
        "package_manifest_sha256": _sha256(output / "qsrt-manifest.json"),
        "checksum_manifest_sha256": _sha256(output / "MANIFEST.sha256"),
        "model_index_sha256": _sha256(output / "model.safetensors.index.json"),
        "source": {
            "kind": source_evidence["source_kind"],
            "sha256": source_evidence["source_sha256"],
        },
        "base_manifest_sha256": base_provenance["manifest_sha256"],
        "producer_fingerprint": producer["fingerprint"],
        "encoder_fingerprint": producer["encoder"]["fingerprint"],
    }


def _validate_checksum_manifest(output: Path) -> None:
    manifest_path = output / "MANIFEST.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        relative = Path(filename)
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not filename
            or relative.is_absolute()
            or relative.as_posix() != filename
            or ".." in relative.parts
            or filename in entries
        ):
            raise ValueError(f"invalid package checksum line: {line!r}")
        entries[filename] = digest
    expected_files = set(_package_files(output))
    if set(entries) != expected_files:
        raise ValueError("package checksum inventory mismatch")
    for filename, expected_sha256 in entries.items():
        actual_sha256 = _sha256(output / filename)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"package hash mismatch for {filename}: "
                f"{actual_sha256} != {expected_sha256}"
            )


def _validate_output_package(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    rate_sweep: dict[str, object],
    require_complete: bool,
) -> None:
    manifest_path = output / "qsrt-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Fruit QSRT package manifest must be a JSON object")
    expected_identity = {
        "schema": "kquant_qsrt_model_manifest_v1",
        "version": 1,
        "codec": "QSRT",
        "storage_schema": FRUIT_QSRT_ATOM_SCHEMA,
        "encoding": "qsrt_sqg_e4m3",
        "codebook": FRUIT_QSRT_CODEBOOK,
        "profile_id": FRUIT_QSRT_PROFILE_ID,
        "geometry": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "topk": 8,
        },
        "runtime": {
            "tensor_parallel": "whole_atom_partition",
            "validated_tensor_parallel_sizes": [1],
            "decode": "trellis_w4a8",
            "decode_max_tokens": 16,
            "fallback": "trellis_w4a16",
            "prefill": "trellis_w4a16",
        },
        "base_model": base_provenance,
        "producer": producer,
        "source": source_evidence,
        "evaluation": {
            "uniform_rate_sweep": {
                "file": _RATE_SWEEP_NAME,
                "sha256": _sha256(output / _RATE_SWEEP_NAME),
            }
        },
        "complete": True,
    }
    if any(manifest.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("Fruit QSRT package identity mismatch")
    sealed_source = _read_source_evidence_seal(output, producer)
    if sealed_source != source_evidence:
        raise ValueError("Fruit QSRT package and sealed source evidence disagree")
    sealed_rate_sweep = json.loads(
        (output / _RATE_SWEEP_NAME).read_text(encoding="utf-8")
    )
    if sealed_rate_sweep != rate_sweep:
        raise ValueError("Fruit QSRT package and sealed rate sweep disagree")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise TypeError("Fruit QSRT output config has no quantization_config")
    qsrt = quantization.get("qsrt")
    if not isinstance(qsrt, dict) or any(
        qsrt.get(name) != value
        for name, value in (
            ("schema", FRUIT_QSRT_ATOM_SCHEMA),
            ("storage_format", FRUIT_QSRT_ATOM_STORAGE),
            ("encoding", "qsrt_sqg_e4m3"),
            ("codebook", FRUIT_QSRT_CODEBOOK),
            ("profile_id", FRUIT_QSRT_PROFILE_ID),
            ("runtime", "w4a8"),
            ("producer_fingerprint", producer["fingerprint"]),
            ("encoder_fingerprint", producer["encoder"]["fingerprint"]),
            ("source_kind", source_evidence["source_kind"]),
            ("source_sha256", source_evidence["source_sha256"]),
        )
    ):
        raise ValueError("Fruit QSRT output config identity mismatch")
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("Fruit QSRT output index has no weight_map")
    if any(".mlp.experts." in name for name in weight_map):
        raise ValueError("Fruit QSRT output index still references routed experts")
    for filename in set(weight_map.values()):
        path = output / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if filename.startswith("model-layer-"):
            with safe_open(path, framework="pt", device="cpu") as handle:
                names = handle.keys()
                if any(".mlp.experts." in name for name in names):
                    raise ValueError(
                        f"Fruit QSRT output shard still contains routed experts: {path}"
                    )
    encoder_fingerprint = producer["encoder"]["fingerprint"]
    validated_layers: dict[str, dict[str, object]] = {}
    for layer in LAYERS:
        tensor_path, layer_manifest_path = _layer_paths(output, layer)
        value = _validate_layer(
            tensor_path,
            layer_manifest_path,
            layer=layer,
            source_sha256=str(source_evidence["source_sha256"]),
            encoder_fingerprint=encoder_fingerprint,
        )
        if value is None:
            raise ValueError(f"Fruit QSRT output is missing assembled layer {layer}")
        validated_layers[str(layer)] = value
    expected_layers = {
        layer: {
            "qsrt_atoms": value["safetensors_file"],
            "bytes": value["safetensors_bytes"],
            "sha256": value["safetensors_sha256"],
            "expert_count": value["expert_count"],
            "evidence": f"qsrt-layer-{int(layer):03d}.json",
        }
        for layer, value in validated_layers.items()
    }
    if manifest.get("layers") != expected_layers:
        raise ValueError("Fruit QSRT package layer ledger mismatch")
    _validate_checksum_manifest(output)
    marker_path = output / _COMPLETE_MARKER_NAME
    if require_complete:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != _completion_record(
            output,
            source_evidence=source_evidence,
            base_provenance=base_provenance,
            producer=producer,
        ):
            raise ValueError("Fruit QSRT completion marker mismatch")
    elif marker_path.exists():
        raise ValueError("Fruit QSRT completion marker exists before final validation")


def _write_complete_marker(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
) -> None:
    _atomic_text(
        output / _COMPLETE_MARKER_NAME,
        _canonical_json(
            _completion_record(
                output,
                source_evidence=source_evidence,
                base_provenance=base_provenance,
                producer=producer,
            )
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("base_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exllamav3-root", required=True, type=Path)
    parser.add_argument("--b12x-root", required=True, type=Path)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--rate-sweep", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-cache", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit QSRT encoding requires a CUDA device")
    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if args.output.resolve() == args.base_model.resolve():
        raise ValueError("output must not alias base_model")
    source_roots = (
        (Path(__file__).resolve().parents[1], "kquant"),
        (args.exllamav3_root, "exllamav3"),
        (args.b12x_root, "b12x"),
        (args.vllm_root, "vllm"),
    )
    for source_root, pathspec in source_roots:
        _require_clean_source(source_root, pathspec)
    calibration = FruitCalibrationStore(args.calibration)
    if _git_revision(args.exllamav3_root) != EXLLAMAV3_REVISION:
        raise ValueError("ExLlamaV3 source revision does not match the pinned encoder")
    producer = _producer_provenance(
        exllamav3_root=args.exllamav3_root,
        b12x_root=args.b12x_root,
        vllm_root=args.vllm_root,
        calibration=calibration,
    )
    encoder_fingerprint = producer["encoder"]["fingerprint"]
    if not isinstance(encoder_fingerprint, str):
        raise TypeError("Fruit QSRT encoder fingerprint must be a string")
    print("authenticating pinned Fruit BF16 base model", flush=True)
    base_provenance = _authenticate_base_model(args.base_model)
    try:
        store = FruitCheckpointStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_sha256=FRUIT_ANNEALED_SPEC.checkpoint_sha256,
        )
    except (FileNotFoundError, ValueError):
        print(
            "checkpoint unavailable or changed; authenticating pinned BF16 "
            "expert source",
            flush=True,
        )
        store = FruitSafetensorsStore(
            args.base_model,
            spec=FRUIT_ANNEALED_SPEC,
            expected_manifest_sha256=BASE_MANIFEST_SHA256,
        )
    source_evidence = _validate_source_evidence(store.evidence)
    source_sha256 = source_evidence.get("source_sha256")
    if not isinstance(source_sha256, str):
        raise TypeError("Fruit source evidence has no authenticated source digest")
    rate_sweep = _validate_rate_sweep(
        args.rate_sweep,
        source_evidence=source_evidence,
        calibration=calibration,
        producer=producer,
    )
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / _COMPLETE_MARKER_NAME).unlink(missing_ok=True)
    _write_source_evidence_seal(args.output, source_evidence, producer)
    seeded = _seed_parts(
        args.output,
        args.seed_cache,
        source_sha256,
        encoder_fingerprint,
    )
    if seeded:
        print(f"seeded {seeded} authenticated expert parts", flush=True)
    needs_encoder = _parts_need_encoder(
        args.output,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    quantizer_module = None
    if needs_encoder:
        quantizer_module = load_qsrt_encoder(args.exllamav3_root)
        install_sqg_quantizer(quantizer_module)
    _encode_parts(
        args.output,
        store,
        quantizer_module,
        calibration,
        device=device,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    layers = _assemble_layers(
        args.output,
        source_sha256=source_sha256,
        encoder_fingerprint=encoder_fingerprint,
    )
    _remove_part_cache(args.output)
    _materialize_base_model(args.base_model, args.output)
    _write_config(args.base_model, args.output, producer, source_evidence)
    _write_source_evidence_seal(args.output, source_evidence, producer)
    _write_calibration_evidence(args.output, calibration, producer)
    _write_package_manifests(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        calibration=calibration,
        rate_sweep=rate_sweep,
        layers=layers,
    )
    _validate_output_package(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        rate_sweep=rate_sweep,
        require_complete=False,
    )
    _write_complete_marker(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
    )
    marker = json.loads(
        (args.output / _COMPLETE_MARKER_NAME).read_text(encoding="utf-8")
    )
    if marker != _completion_record(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
    ):
        raise ValueError("Fruit QSRT completion marker failed post-write validation")
    print(f"complete Fruit QSRT model: {args.output}", flush=True)


if __name__ == "__main__":
    main()
