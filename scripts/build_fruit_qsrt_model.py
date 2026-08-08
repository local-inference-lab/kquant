#!/usr/bin/env python3
"""Encode and assemble a complete exact-rate Fruit QSRT Hugging Face model."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
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
KQUANT_REVISION = "a9c94ebc1039c77525c7129fcae9a32f4feb4ebc"
EXLLAMAV3_REVISION = "791c83073f7f90c44f765a0ceeab7a05fa15b96b"
B12X_REVISION = "9bbae67841e4818e7472e1edcdca8ebcbda68611"
VLLM_REVISION = "ad1d3d1cf7123864bdd5e2bf1ed52c3437035828"
_SOURCE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".toml"}
)
_COMPLETE_MARKER_NAME = "QSRT_COMPLETE.json"
MODEL_CARD = """---
license: apache-2.0
library_name: transformers
tags:
- kquant
- qsrt
- mixture-of-experts
---

# GLM-5.2-SIQ-Fruit-QSRT

Exact fixed-payload QSRT conversion of the Fruit-annealed GLM-5.2-SIQ model.
Every routed expert in layers 3 through 13 uses the public SQG-XOR-Cheb-T12
codebook, two fixed P24/P33 pairs per matrix, and expert-specific EXL rotations.
Formats are chosen from the complete coupled `(R13, R2)` grid using a
document-disjoint activation capture: fit documents construct global H13 and
candidate-conditional expert H2, confirmation documents bootstrap-gate the
winner against R0/R0, and a third fold remains untouched for validation.
The TP-independent atom container stores every expert at exactly three trellis
bits per coefficient. Serving shards its 16 whole 32-channel atoms at load
time; no dense routed-expert fallback or serialized TP rank exists.

Serving requires the exact kquant, ExLlamaV3, b12x, and vLLM source identities
recorded in `qsrt-manifest.json`. `qsrt-calibration-evidence.json` authenticates
the selection capture without inflating the model with raw activation rows. The
single-GPU runtime uses W4A8 for batches up to 16 tokens and W4A16 above that.
`MANIFEST.sha256` authenticates the published files; `QSRT_COMPLETE.json` is
written only after full structural validation.
"""
_SOURCE_EVIDENCE_NAME = ".qsrt-source-evidence.json"
_SOURCE_EVIDENCE_SHA_NAME = ".qsrt-source-evidence.sha256"
_CALIBRATION_EVIDENCE_NAME = "qsrt-calibration-evidence.json"

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


def _source_tree_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in _SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        ),
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


def _producer_provenance(
    *,
    exllamav3_root: Path,
    b12x_root: Path,
    vllm_root: Path,
    calibration: FruitCalibrationStore,
    output: Path,
) -> dict[str, object]:
    kquant_root = Path(__file__).resolve().parents[1] / "kquant"
    current_encoder = {
        "kquant_revision": KQUANT_REVISION,
        "kquant_source_sha256": _source_tree_sha256(kquant_root),
        "exllamav3_revision": EXLLAMAV3_REVISION,
        "exllamav3_source_sha256": _source_tree_sha256(exllamav3_root / "exllamav3"),
        "calibration_fingerprint": calibration.fingerprint,
        "calibration_capture_id": calibration.capture_id,
        "calibration_manifest_sha256": calibration.manifest_sha256,
    }
    current_encoder["fingerprint"] = hashlib.sha256(
        _canonical_json(current_encoder).encode("utf-8")
    ).hexdigest()
    encoder = _historical_encoder_provenance(output, current_encoder)
    runtime = {
        "b12x_revision": B12X_REVISION,
        "b12x_source_sha256": _source_tree_sha256(b12x_root / "b12x"),
        "vllm_revision": VLLM_REVISION,
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


def _historical_encoder_provenance(
    output: Path,
    current: dict[str, object],
) -> dict[str, object]:
    """Reuse authenticated encoded parts across storage/runtime-only rebuilds."""

    manifest_path = output / "qsrt-manifest.json"
    if not manifest_path.is_file():
        return current
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate = manifest["producer"]["encoder"]
    except (KeyError, OSError, json.JSONDecodeError, TypeError):
        return current
    if not isinstance(candidate, dict) or set(candidate) != set(current):
        return current
    fingerprint = candidate.get("fingerprint")
    unsigned = {name: value for name, value in candidate.items() if name != "fingerprint"}
    if (
        not isinstance(fingerprint, str)
        or hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
        != fingerprint
    ):
        return current
    stable_fields = set(current) - {"fingerprint", "kquant_source_sha256"}
    if any(candidate.get(name) != current.get(name) for name in stable_fields):
        return current
    for layer in LAYERS:
        for expert in range(EXPERTS):
            tensor_path, part_manifest_path = _part_paths(output, layer, expert)
            if not tensor_path.is_file() or not part_manifest_path.is_file():
                return current
            try:
                part = json.loads(part_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return current
            if (
                not isinstance(part, dict)
                or part.get("encoder_fingerprint") != fingerprint
            ):
                return current
    return candidate


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
) -> dict[str, object] | None:
    if not tensor_path.exists() and not manifest_path.exists():
        return None
    if not tensor_path.is_file() or not manifest_path.is_file():
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
        if mismatches == {"encoder_fingerprint"}:
            tensor_path.unlink()
            manifest_path.unlink()
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
    metadata_fingerprint = metadata.get("encoder_fingerprint")
    if metadata_fingerprint not in (None, encoder_fingerprint):
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
            _validate_part(
                source_tensor,
                source_manifest,
                layer=layer,
                expert=expert,
                source_sha256=source_sha256,
                encoder_fingerprint=encoder_fingerprint,
            )
            target_tensor, target_manifest = _part_paths(output, layer, expert)
            if target_tensor.exists() or target_manifest.exists():
                continue
            target_tensor.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_tensor, target_tensor)
            shutil.copy2(source_manifest, target_manifest)
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
    expected = {
        "kind": "kquant-fruit-source-preflight",
        "version": 1,
        "status": "pass",
        "model_id": FRUIT_ANNEALED_SPEC.model_id,
        "checkpoint_sha256": FRUIT_ANNEALED_SPEC.checkpoint_sha256,
    }
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
            layer = int(
                filename.removeprefix("model-layer-").removesuffix(".safetensors")
            )
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
            "artifact_manifest": "qsrt-manifest.json",
            "producer_fingerprint": producer["fingerprint"],
            "encoder_fingerprint": producer["encoder"]["fingerprint"],
            "runtime": "w4a8",
        },
    }
    _atomic_text(output / "config.json", _canonical_json(config))


def _write_package_manifests(
    output: Path,
    *,
    source_evidence: dict[str, object],
    base_provenance: dict[str, object],
    producer: dict[str, object],
    layers: dict[str, dict[str, object]],
) -> None:
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
        "layers": manifest_layers,
        "complete": True,
    }
    _atomic_text(output / "qsrt-manifest.json", _canonical_json(manifest))
    _atomic_text(output / "README.md", MODEL_CARD)
    entries: list[str] = []
    for path in sorted(output.iterdir(), key=lambda value: value.name):
        if (
            path.name in {"MANIFEST.sha256", _COMPLETE_MARKER_NAME}
            or not path.is_file()
        ):
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    _atomic_text(output / "MANIFEST.sha256", "\n".join(entries) + "\n")


def _completion_record(
    output: Path,
    *,
    base_provenance: dict[str, object],
    producer: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "kquant_qsrt_complete_v1",
        "package_manifest_sha256": _sha256(output / "qsrt-manifest.json"),
        "checksum_manifest_sha256": _sha256(output / "MANIFEST.sha256"),
        "model_index_sha256": _sha256(output / "model.safetensors.index.json"),
        "source_checkpoint_sha256": FRUIT_ANNEALED_SPEC.checkpoint_sha256,
        "base_manifest_sha256": base_provenance["manifest_sha256"],
        "producer_fingerprint": producer["fingerprint"],
        "encoder_fingerprint": producer["encoder"]["fingerprint"],
    }


def _validate_checksum_manifest(output: Path) -> None:
    manifest_path = output / "MANIFEST.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"malformed package checksum line: {line!r}")
        digest, filename = fields
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or Path(filename).name != filename
            or filename in entries
        ):
            raise ValueError(f"invalid package checksum line: {line!r}")
        entries[filename] = digest
    expected_files = {
        path.name
        for path in output.iterdir()
        if path.is_file()
        and path.name not in {"MANIFEST.sha256", _COMPLETE_MARKER_NAME}
    }
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
        "complete": True,
    }
    if any(manifest.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("Fruit QSRT package identity mismatch")
    sealed_source = _read_source_evidence_seal(output, producer)
    if sealed_source != source_evidence:
        raise ValueError("Fruit QSRT package and sealed source evidence disagree")
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
            ("runtime", "w4a8"),
            ("producer_fingerprint", producer["fingerprint"]),
            ("encoder_fingerprint", producer["encoder"]["fingerprint"]),
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
            source_sha256=FRUIT_ANNEALED_SPEC.checkpoint_sha256,
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
            base_provenance=base_provenance,
            producer=producer,
        ):
            raise ValueError("Fruit QSRT completion marker mismatch")
    elif marker_path.exists():
        raise ValueError("Fruit QSRT completion marker exists before final validation")


def _write_complete_marker(
    output: Path,
    *,
    base_provenance: dict[str, object],
    producer: dict[str, object],
) -> None:
    _atomic_text(
        output / _COMPLETE_MARKER_NAME,
        _canonical_json(
            _completion_record(
                output,
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
    calibration = FruitCalibrationStore(args.calibration)
    producer = _producer_provenance(
        exllamav3_root=args.exllamav3_root,
        b12x_root=args.b12x_root,
        vllm_root=args.vllm_root,
        output=args.output,
        calibration=calibration,
    )
    encoder_fingerprint = producer["encoder"]["fingerprint"]
    if not isinstance(encoder_fingerprint, str):
        raise TypeError("Fruit QSRT encoder fingerprint must be a string")
    print("authenticating pinned Fruit BF16 base model", flush=True)
    base_provenance = _authenticate_base_model(args.base_model)
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / _COMPLETE_MARKER_NAME).unlink(missing_ok=True)
    source_sha256 = FRUIT_ANNEALED_SPEC.checkpoint_sha256
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
    store = None
    try:
        store = FruitCheckpointStore(
            args.checkpoint,
            spec=FRUIT_ANNEALED_SPEC,
            expected_sha256=source_sha256,
        )
    except (FileNotFoundError, ValueError):
        source_evidence = None
        if not needs_encoder:
            for evidence_root in (args.output, args.seed_cache):
                try:
                    source_evidence = _read_source_evidence_seal(
                        evidence_root, producer
                    )
                except ValueError:
                    source_evidence = None
                if source_evidence is not None:
                    break
        if source_evidence is not None:
            print(
                "checkpoint unavailable or changed; using sealed source evidence",
                flush=True,
            )
        else:
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
            source_evidence = store.evidence
            _write_source_evidence_seal(args.output, source_evidence, producer)
    else:
        source_evidence = store.evidence
        _write_source_evidence_seal(args.output, source_evidence, producer)
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
    _materialize_base_model(args.base_model, args.output)
    _write_config(args.base_model, args.output, producer)
    _write_source_evidence_seal(args.output, source_evidence, producer)
    _write_calibration_evidence(args.output, calibration, producer)
    _write_package_manifests(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        layers=layers,
    )
    _validate_output_package(
        args.output,
        source_evidence=source_evidence,
        base_provenance=base_provenance,
        producer=producer,
        require_complete=False,
    )
    _write_complete_marker(
        args.output,
        base_provenance=base_provenance,
        producer=producer,
    )
    marker = json.loads(
        (args.output / _COMPLETE_MARKER_NAME).read_text(encoding="utf-8")
    )
    if marker != _completion_record(
        args.output,
        base_provenance=base_provenance,
        producer=producer,
    ):
        raise ValueError("Fruit QSRT completion marker failed post-write validation")
    print(f"complete Fruit QSRT model: {args.output}", flush=True)


if __name__ == "__main__":
    main()
