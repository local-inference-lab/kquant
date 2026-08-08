"""Untouched-corpus scoring for selected QSRT candidates.

The candidate pool stores one already selected equal-byte mode pair per expert:
one rate-transfer mode shared by ``w1``/``w3`` and one for ``w2``.
This module decodes that physical payload and scores it against the official
MXFP4 source expert on a corpus that was not used for ranking, Hessian fitting,
or mode selection.  The official checkpoint remains a one-matrix-at-a-time
offline source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3
from kquant.qsrt_candidates import RoutedRows, functional_sse_by_request
from kquant.qsrt import (
    SCHEMA,
    MATRIX_TRELLIS_BYTES,
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
    decode_qsrt_exl3_weight,
    resolve_mode,
)
from kquant.pack.qsrt_candidates import MatrixStore, candidate_tensor_name
from kquant.tp_simulator import situ


VALIDATION_SCORE_KIND = "kquant_kimi_k3_qsrt_validation_scores"
VALIDATION_SCORE_SCHEMA_VERSION = 4
VALIDATION_DAMAGE_METRIC = "validation_official_source_excess_sse"
VALIDATION_DAMAGE_WEIGHTING = (
    "natural-routing token-uniform document-disjoint validation rows with "
    "applied gate squared; no second traffic multiplier"
)

MATRIX_GEOMETRY = {
    "w1": ("n", 224, 192),
    "w3": ("n", 224, 192),
    "w2": ("k", 192, 224),
}


class CandidateTensorReader(Protocol):
    """Minimal interface provided by a safetensors ``safe_open`` handle."""

    def get_tensor(self, name: str) -> torch.Tensor: ...


@dataclass(frozen=True)
class QSRTValidationScores:
    """Complete document-disjoint damage scores for all assignments."""

    root: Path
    manifest: dict
    selected_r13: np.ndarray
    selected_r2: np.ndarray
    damage: np.ndarray
    per_document_damage: np.ndarray
    reference_energy: np.ndarray
    routed_rows: np.ndarray
    supported_documents: np.ndarray
    content_sha256: str


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _update_content_digest(
    digest: Any,
    root: Path,
    path: Path,
) -> None:
    """Bind one canonical sidecar path and its bytes into a set digest."""

    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(len(relative).to_bytes(4, "little"))
    digest.update(relative)
    digest.update(bytes.fromhex(_sha256(path)))


def _require_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        value = tensors[name]
    except KeyError as exc:
        raise ValueError(f"validation metrics are missing {name}") from exc
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(
            f"validation metric {name} has {value.dtype} {tuple(value.shape)}, "
            f"expected {dtype} {shape}"
        )
    return value


def validate_validation_layer_metrics(
    tensors: Mapping[str, torch.Tensor],
    *,
    selected_r13: np.ndarray,
    selected_r2: np.ndarray,
    selection_damage: np.ndarray,
    documents: int,
    expert_ids: tuple[int, ...] = tuple(range(C.NUM_EXPERTS)),
) -> dict[str, np.ndarray]:
    """Validate one held-out score ledger and close all derived totals."""

    experts = len(expert_ids)
    expected_vector = (experts,)
    ids = _require_tensor(
        tensors, "expert_ids", shape=expected_vector, dtype=torch.int16
    )
    if ids.tolist() != list(expert_ids):
        raise ValueError("validation expert IDs are not canonical")
    stored_r13 = _require_tensor(
        tensors, "selected_r13", shape=expected_vector, dtype=torch.uint8
    )
    stored_r2 = _require_tensor(
        tensors, "selected_r2", shape=expected_vector, dtype=torch.uint8
    )
    if not np.array_equal(stored_r13.numpy(), np.asarray(selected_r13)):
        raise ValueError("validation r13 modes differ from the candidate pool")
    if not np.array_equal(stored_r2.numpy(), np.asarray(selected_r2)):
        raise ValueError("validation r2 modes differ from the candidate pool")
    sse = _require_tensor(
        tensors,
        "validation_sse",
        shape=(experts, documents),
        dtype=torch.float64,
    )
    reference = _require_tensor(
        tensors,
        "validation_reference_energy",
        shape=(experts, documents),
        dtype=torch.float64,
    )
    counts = _require_tensor(
        tensors,
        "validation_counts",
        shape=(experts, documents),
        dtype=torch.int32,
    )
    gate_square = _require_tensor(
        tensors,
        "validation_gate_square_sum",
        shape=expected_vector,
        dtype=torch.float64,
    )
    damage = _require_tensor(
        tensors,
        VALIDATION_DAMAGE_METRIC,
        shape=expected_vector,
        dtype=torch.float64,
    )
    stored_selection_damage = _require_tensor(
        tensors,
        "selection_corpus_damage",
        shape=expected_vector,
        dtype=torch.float64,
    )
    if not np.array_equal(
        stored_selection_damage.numpy(), np.asarray(selection_damage)
    ):
        raise ValueError("validation sidecar selection damage drifted")
    for name, value in (
        ("validation_sse", sse),
        ("validation_reference_energy", reference),
        ("validation_gate_square_sum", gate_square),
        (VALIDATION_DAMAGE_METRIC, damage),
    ):
        if not bool(torch.all(torch.isfinite(value))) or bool(torch.any(value < 0)):
            raise ValueError(f"{name} must be finite and non-negative")
    if bool(torch.any(counts < 0)):
        raise ValueError("validation counts must be non-negative")
    if not torch.equal(sse.sum(dim=1), damage):
        raise ValueError("validation damage does not equal per-document SSE")
    rows = counts.sum(dim=1)
    supported = (counts > 0).sum(dim=1)
    if bool(torch.any((rows == 0) & ((damage != 0) | (reference.sum(dim=1) != 0)))):
        raise ValueError("unrouted validation experts must have zero metrics")
    return {
        "damage": damage.numpy(),
        "per_document_damage": sse.numpy(),
        "reference_energy": reference.sum(dim=1).numpy(),
        "routed_rows": rows.numpy(),
        "supported_documents": supported.numpy(),
    }


def physical_expert_output(
    inputs: torch.Tensor,
    w1_input_output: torch.Tensor,
    w3_input_output: torch.Tensor,
    w2_input_output: torch.Tensor,
) -> torch.Tensor:
    """Execute one candidate directly in its shared physical neuron order."""

    if inputs.ndim != 2:
        raise ValueError("expert inputs must be two-dimensional")
    if tuple(w1_input_output.shape) != (C.LATENT, C.EXPERT_INTER):
        raise ValueError("physical w1 has the wrong shape")
    if tuple(w3_input_output.shape) != (C.LATENT, C.EXPERT_INTER):
        raise ValueError("physical w3 has the wrong shape")
    if tuple(w2_input_output.shape) != (C.EXPERT_INTER, C.LATENT):
        raise ValueError("physical w2 has the wrong shape")
    if inputs.shape[1] != C.LATENT:
        raise ValueError("expert inputs have the wrong width")
    gate = inputs @ w1_input_output
    up = inputs @ w3_input_output
    return situ(gate, up) @ w2_input_output


def official_expert_output(
    store: MatrixStore,
    inputs: torch.Tensor,
    *,
    layer: int,
    expert: int,
    device: torch.device,
) -> torch.Tensor:
    """Stream and execute one official expert with one source matrix resident."""

    source_w1 = store.load_matrix(layer, expert, "w1", device=device)
    gate = F.linear(inputs, source_w1)
    del source_w1
    source_w3 = store.load_matrix(layer, expert, "w3", device=device)
    up = F.linear(inputs, source_w3)
    del source_w3
    middle = situ(gate, up)
    del gate, up
    source_w2 = store.load_matrix(layer, expert, "w2", device=device)
    output = F.linear(middle, source_w2)
    del middle, source_w2
    return output


def decode_candidate_matrix(
    reader: CandidateTensorReader,
    *,
    layer: int,
    expert: int,
    matrix: str,
    mode_id: int,
    device: torch.device,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_SQG_NORMAL_E4M3,
) -> torch.Tensor:
    """Decode one stored matrix to EXL's physical ``[input, output]`` order."""

    if matrix not in MATRIX_GEOMETRY:
        raise ValueError(f"unsupported expert matrix {matrix!r}")
    mode = resolve_mode(mode_id)
    rate_axis, k_tiles, n_tiles = MATRIX_GEOMETRY[matrix]
    descriptor = QSRTTrellisDescriptor(
        mode_id=mode.mode_id,
        rate_axis=rate_axis,
        k_tiles=k_tiles,
        n_tiles=n_tiles,
        schema=logical_trellis_schema,
    )
    names = {
        part: candidate_tensor_name(layer, expert, matrix, part)
        for part in ("trellis", "suh", "svh")
    }
    payload = reader.get_tensor(names["trellis"])
    suh = reader.get_tensor(names["suh"])
    svh = reader.get_tensor(names["svh"])
    if payload.numel() * payload.element_size() != MATRIX_TRELLIS_BYTES:
        raise ValueError(f"{matrix} candidate trellis has the wrong byte count")
    packed = PackedQSRTTrellis(
        descriptor,
        payload.to(device=device, dtype=torch.int16).contiguous(),
    )
    return decode_qsrt_exl3_weight(
        packed,
        suh.to(device=device, dtype=torch.float16).contiguous(),
        svh.to(device=device, dtype=torch.float16).contiguous(),
        codebook=codebook,
    )


def selected_candidate_output(
    reader: CandidateTensorReader,
    inputs: torch.Tensor,
    *,
    layer: int,
    expert: int,
    r13_mode_id: int,
    r2_mode_id: int,
    device: torch.device,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_SQG_NORMAL_E4M3,
) -> torch.Tensor:
    """Decode and execute one selected candidate with bounded matrix residency."""

    w1 = decode_candidate_matrix(
        reader,
        layer=layer,
        expert=expert,
        matrix="w1",
        mode_id=r13_mode_id,
        device=device,
        logical_trellis_schema=logical_trellis_schema,
        codebook=codebook,
    )
    gate = inputs @ w1.float()
    del w1
    w3 = decode_candidate_matrix(
        reader,
        layer=layer,
        expert=expert,
        matrix="w3",
        mode_id=r13_mode_id,
        device=device,
        logical_trellis_schema=logical_trellis_schema,
        codebook=codebook,
    )
    up = inputs @ w3.float()
    del w3
    middle = situ(gate, up)
    del gate, up
    w2 = decode_candidate_matrix(
        reader,
        layer=layer,
        expert=expert,
        matrix="w2",
        mode_id=r2_mode_id,
        device=device,
        logical_trellis_schema=logical_trellis_schema,
        codebook=codebook,
    )
    output = middle @ w2.float()
    del middle, w2
    return output


def score_selected_expert(
    store: MatrixStore,
    reader: CandidateTensorReader,
    rows: RoutedRows,
    requests: Mapping[int, str],
    *,
    layer: int,
    expert: int,
    r13_mode_id: int,
    r2_mode_id: int,
    device: torch.device,
    logical_trellis_schema: str = SCHEMA,
    codebook: str = CODEBOOK_SQG_NORMAL_E4M3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return held-out routed SSE, reference energy, and counts by document."""

    if rows.rows == 0:
        count = len(requests)
        return (
            torch.zeros(count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.int64),
        )
    inputs = rows.inputs.float().to(device=device)
    gates = rows.gates.float().to(device=device)
    reference = official_expert_output(
        store,
        inputs,
        layer=layer,
        expert=expert,
        device=device,
    )
    candidate = selected_candidate_output(
        reader,
        inputs,
        layer=layer,
        expert=expert,
        r13_mode_id=r13_mode_id,
        r2_mode_id=r2_mode_id,
        device=device,
        logical_trellis_schema=logical_trellis_schema,
        codebook=codebook,
    )
    result = functional_sse_by_request(
        reference,
        candidate,
        gates,
        rows.request_steps,
        requests,
    )
    del inputs, gates, reference, candidate
    return result


def load_qsrt_validation_scores(
    root: str | Path,
    candidate_pool: object,
) -> QSRTValidationScores:
    """Load a complete 92-layer sidecar and bind it to its candidate pool."""

    # Local attribute validation avoids an import cycle with raw_keep_allocation.
    try:
        candidate_root = Path(candidate_pool.root).resolve()
        candidate_manifest = candidate_pool.manifest
        candidate_r13 = candidate_pool.selected_r13
        candidate_r2 = candidate_pool.selected_r2
        candidate_damage = candidate_pool.damage
        candidate_content_sha256 = candidate_pool.content_sha256
        candidate_trellis_schema = candidate_pool.trellis_schema
    except AttributeError as exc:
        raise TypeError("candidate_pool is not a QSRTCandidatePool") from exc
    root = Path(root).resolve()
    manifest_path = root / "qsrt-validation-manifest.json"
    manifest = _read_json(manifest_path)
    expected = {
        "kind": VALIDATION_SCORE_KIND,
        "schema_version": VALIDATION_SCORE_SCHEMA_VERSION,
        "candidate_pool": str(candidate_root),
        "candidate_schema_version": candidate_manifest.get("schema_version"),
        "candidate_pool_content_sha256": candidate_content_sha256,
        "candidate_logical_trellis_schema": candidate_trellis_schema,
        "candidate_codebook": candidate_pool.codebook,
        "candidate_mode_ids": list(candidate_pool.mode_ids),
        "source_model": C.MODEL_ID,
        "source_revision": candidate_manifest.get("source_revision"),
        "metric": VALIDATION_DAMAGE_METRIC,
        "selection_data_used": False,
    }
    for name, value in expected.items():
        actual = manifest.get(name)
        if name == "candidate_codebook" and actual is None:
            actual = "mcg"
        if name == "candidate_mode_ids" and actual is None:
            actual = candidate_manifest.get("mode_ids")
        if actual != value:
            raise ValueError(
                f"validation manifest {name} mismatch: "
                f"{actual!r} != {value!r}"
            )
    candidate_manifest_path = candidate_root / "qsrt-candidate-manifest.json"
    if manifest.get("candidate_manifest_sha256") != _sha256(
        candidate_manifest_path
    ):
        raise ValueError("validation sidecar candidate manifest hash drifted")
    documents = int(manifest.get("validation_documents", -1))
    if documents < 1:
        raise ValueError("validation sidecar has no documents")
    completion = _read_json(root / "qsrt-validation-completion.json")
    if (
        completion.get("kind") != VALIDATION_SCORE_KIND
        or completion.get("schema_version") != VALIDATION_SCORE_SCHEMA_VERSION
        or not completion.get("complete")
        or completion.get("layers") != list(C.MOE_LAYERS)
        or completion.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("validation completion marker does not close")

    content_digest = hashlib.sha256()
    content_digest.update(b"kquant-qsrt-validation-score-set-v3\0")
    _update_content_digest(content_digest, root, manifest_path)
    _update_content_digest(
        content_digest,
        root,
        root / "qsrt-validation-completion.json",
    )

    damage = np.empty((C.NUM_MOE_LAYERS, C.NUM_EXPERTS), dtype=np.float64)
    per_document_damage = np.empty(
        (C.NUM_MOE_LAYERS, C.NUM_EXPERTS, documents),
        dtype=np.float64,
    )
    reference = np.empty_like(damage)
    routed_rows = np.empty_like(damage, dtype=np.int64)
    supported_documents = np.empty_like(damage, dtype=np.int32)
    for row, layer in enumerate(C.MOE_LAYERS):
        stem = f"qsrt-layer-{layer:05d}.validation"
        metrics_path = root / "scores" / f"{stem}.safetensors"
        ledger_path = root / "scores" / f"{stem}.json"
        if not metrics_path.is_file() or not ledger_path.is_file():
            raise ValueError(f"validation sidecar is missing layer {layer}")
        _update_content_digest(content_digest, root, metrics_path)
        _update_content_digest(content_digest, root, ledger_path)
        ledger = _read_json(ledger_path)
        expected_ledger = {
            "kind": VALIDATION_SCORE_KIND,
            "schema_version": VALIDATION_SCORE_SCHEMA_VERSION,
            "complete": True,
            "layer": layer,
            "metrics": metrics_path.name,
            "experts": C.NUM_EXPERTS,
            "validation_documents": documents,
            "external_validation_used_for_mode_selection": False,
        }
        for name, value in expected_ledger.items():
            if ledger.get(name) != value:
                raise ValueError(f"layer {layer} validation ledger {name} drifted")
        tensors = load_file(str(metrics_path), device="cpu")
        validated = validate_validation_layer_metrics(
            tensors,
            selected_r13=candidate_r13[row],
            selected_r2=candidate_r2[row],
            selection_damage=candidate_damage[row],
            documents=documents,
        )
        damage[row] = validated["damage"]
        per_document_damage[row] = validated["per_document_damage"]
        reference[row] = validated["reference_energy"]
        routed_rows[row] = validated["routed_rows"]
        supported_documents[row] = validated["supported_documents"]
        if int(routed_rows[row].sum()) != int(ledger["validation_rows"]):
            raise ValueError(f"layer {layer} validation row total drifted")
        if not np.isclose(float(damage[row].sum()), ledger["validation_damage"]):
            raise ValueError(f"layer {layer} validation damage total drifted")
        if not np.isclose(
            float(reference[row].sum()), ledger["validation_reference_energy"]
        ):
            raise ValueError(f"layer {layer} validation reference total drifted")
    return QSRTValidationScores(
        root=root,
        manifest=manifest,
        selected_r13=np.asarray(candidate_r13, dtype=np.uint8).copy(),
        selected_r2=np.asarray(candidate_r2, dtype=np.uint8).copy(),
        damage=damage,
        per_document_damage=per_document_damage,
        reference_energy=reference,
        routed_rows=routed_rows,
        supported_documents=supported_documents,
        content_sha256=content_digest.hexdigest(),
    )
