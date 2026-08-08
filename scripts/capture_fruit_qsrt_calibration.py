#!/usr/bin/env python3
"""Capture document-disjoint routed activations for production Fruit QSRT."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.torch_version import TorchVersion
from transformers import AutoTokenizer

from kquant.candidate_hessian import weighted_covariance
from kquant.fruit_calibration import (
    FRUIT_CALIBRATION_AXES,
    FRUIT_CALIBRATION_CLOSURE_LIMITS,
    FRUIT_CALIBRATION_CORPUS,
    FRUIT_CALIBRATION_LAYERS,
    FRUIT_CALIBRATION_PROTOCOL,
    FRUIT_CALIBRATION_REFERENCE_SHA256,
    FRUIT_CALIBRATION_SCHEMA,
    FRUIT_CALIBRATION_SOURCE,
    FRUIT_CALIBRATION_TOKEN_BOUNDS,
    FRUIT_CALIBRATION_TOKENIZER_FILES,
    FRUIT_CALIBRATION_TOPK,
    FRUIT_CALIBRATION_TRAINER_FILES,
    FRUIT_CALIBRATION_VERSION,
    calibration_fingerprint,
)
from kquant.fruit_source import FRUIT_ANNEALED_SPEC

_SMALL_TOKENIZER_FILES = tuple(FRUIT_CALIBRATION_TOKENIZER_FILES)

_BF16_BASE_MANIFEST_SHA256 = (
    "8a7e30f3a948bbac203013160b2e6bb8d0ed50c36cf2ca1c3978701124cc7671"
)
_BF16_BASE_INDEX = "model.safetensors.index.json"
_MODEL_MARKER_KEYS = frozenset({"rope_theta_trained", "serve_conv_v"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_safetensors(
    path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    save_file(tensors, temporary, metadata=metadata)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class _CorpusCandidate:
    axis: str
    source: str
    text: str
    line: int
    digest: str
    family_digest: str
    token_ids: tuple[int, ...]
    token_digest: str


@dataclass(frozen=True)
class _CorpusDocument(_CorpusCandidate):
    split: str


@dataclass
class _LayerChunks:
    inputs: list[torch.Tensor] = field(default_factory=list)
    route_ids: list[torch.Tensor] = field(default_factory=list)
    route_weights: list[torch.Tensor] = field(default_factory=list)
    document_ids: list[torch.Tensor] = field(default_factory=list)

    def append(
        self,
        inputs: torch.Tensor,
        route_ids: torch.Tensor,
        route_weights: torch.Tensor,
        document_id: int,
    ) -> None:
        rows = int(inputs.shape[0])
        self.inputs.append(inputs.to(device="cpu", dtype=torch.bfloat16).contiguous())
        self.route_ids.append(
            route_ids.to(device="cpu", dtype=torch.int16).contiguous()
        )
        self.route_weights.append(
            route_weights.to(device="cpu", dtype=torch.float32).contiguous()
        )
        self.document_ids.append(
            torch.full((rows,), int(document_id), dtype=torch.int32)
        )

    def finish(self) -> dict[str, torch.Tensor]:
        if not self.inputs:
            raise ValueError("Fruit calibration layer captured no inputs")
        return {
            "inputs": torch.cat(self.inputs).contiguous(),
            "route_ids": torch.cat(self.route_ids).contiguous(),
            "route_weights": torch.cat(self.route_weights).contiguous(),
            "document_ids": torch.cat(self.document_ids).contiguous(),
        }


def _normalized_corpus_text(raw: str, source: str) -> tuple[str, str]:
    """Mirror Fruit training normalization and identify a prompt family."""

    try:
        envelope = json.loads(raw)
        messages = envelope.get("messages")
        if not isinstance(messages, list) or not messages:
            return raw, _sha256_bytes(f"{source}\0{raw}".encode())
        contents = [message.get("content", "") for message in messages]
        if any(not isinstance(content, str) for content in contents):
            return raw, _sha256_bytes(f"{source}\0{raw}".encode())
        normalized = "\n".join(contents)
        prompt = [
            {
                "role": message.get("role", ""),
                "content": content,
            }
            for message, content in zip(messages, contents, strict=True)
            if message.get("role") != "assistant"
        ]
        if not prompt:
            prompt = [
                {
                    "role": message.get("role", ""),
                    "content": content,
                }
                for message, content in zip(messages, contents, strict=True)
            ]
        family = json.dumps(
            {"source": source, "prompt": prompt},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return normalized, _sha256_bytes(family.encode("utf-8"))
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        return raw, _sha256_bytes(f"{source}\0{raw}".encode())


def _token_digest(token_ids: tuple[int, ...]) -> str:
    values = torch.tensor(token_ids, dtype=torch.int32)
    return _sha256_bytes(values.numpy().tobytes())


def _sample_documents(
    corpus: Path,
    tokenizer: Any,
    *,
    documents_per_axis: int,
    fit_per_axis: int,
    confirmation_per_axis: int,
    min_tokens: int,
    max_tokens: int,
) -> tuple[list[_CorpusDocument], str]:
    if documents_per_axis <= 0:
        raise ValueError("documents_per_axis must be positive")
    if fit_per_axis < 1 or confirmation_per_axis < 1:
        raise ValueError("fit and confirmation quotas must be positive")
    if fit_per_axis + confirmation_per_axis >= documents_per_axis:
        raise ValueError("document quotas leave no untouched validation fold")
    if not 2 <= min_tokens <= max_tokens:
        raise ValueError("invalid calibration token bounds")

    corpus_sha256 = _sha256(corpus)
    by_axis: dict[str, list[_CorpusCandidate]] = defaultdict(list)
    document_axes: dict[str, str] = {}
    with corpus.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed calibration JSONL line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"calibration line {line_number} is not an object")
            axis = value.get("axis")
            source = value.get("source")
            raw = value.get("text")
            if not all(isinstance(item, str) and item for item in (axis, source, raw)):
                raise ValueError(
                    f"calibration line {line_number} lacks axis/source/text"
                )
            text, family_digest = _normalized_corpus_text(raw, source)
            if not text:
                continue
            digest = _sha256_bytes(text.encode("utf-8"))
            previous_axis = document_axes.get(digest)
            if previous_axis is not None:
                if previous_axis != axis:
                    raise ValueError("normalized calibration text spans multiple axes")
                continue
            document_axes[digest] = axis
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_tokens,
                return_attention_mask=False,
            )["input_ids"]
            token_ids = tuple(map(int, encoded))
            if len(token_ids) < min_tokens:
                continue
            by_axis[axis].append(
                _CorpusCandidate(
                    axis=axis,
                    source=source,
                    text=text,
                    line=line_number,
                    digest=digest,
                    family_digest=family_digest,
                    token_ids=token_ids,
                    token_digest=_token_digest(token_ids),
                )
            )
    if len(by_axis) < 4:
        raise ValueError("Fruit calibration corpus must cover at least four axes")

    selected: list[_CorpusDocument] = []
    selected_token_digests: set[str] = set()
    for axis, candidates in sorted(by_axis.items()):
        ordered = sorted(
            candidates,
            key=lambda item: _sha256_bytes(
                f"sample:{item.family_digest}:{item.digest}".encode("ascii")
            ),
        )
        eligible: list[_CorpusCandidate] = []
        selected_families: set[str] = set()
        for candidate in ordered:
            if (
                candidate.family_digest in selected_families
                or candidate.token_digest in selected_token_digests
            ):
                continue
            eligible.append(candidate)
            selected_families.add(candidate.family_digest)
            selected_token_digests.add(candidate.token_digest)
            if len(eligible) == documents_per_axis:
                break
        if len(eligible) != documents_per_axis:
            raise ValueError(
                f"axis {axis!r} has {len(eligible)} disjoint eligible documents, "
                f"expected {documents_per_axis}"
            )
        split_order = sorted(
            eligible,
            key=lambda item: _sha256_bytes(
                f"split:{item.family_digest}:{item.token_digest}".encode("ascii")
            ),
        )
        split_by_digest = {
            item.digest: (
                "fit"
                if index < fit_per_axis
                else "confirmation"
                if index < fit_per_axis + confirmation_per_axis
                else "validation"
            )
            for index, item in enumerate(split_order)
        }
        selected.extend(
            _CorpusDocument(
                axis=item.axis,
                source=item.source,
                text=item.text,
                line=item.line,
                digest=item.digest,
                family_digest=item.family_digest,
                token_ids=item.token_ids,
                token_digest=item.token_digest,
                split=split_by_digest[item.digest],
            )
            for item in sorted(eligible, key=lambda value: value.digest)
        )
    return selected, corpus_sha256


def _configure_trainer_environment(*, serve_native: bool) -> None:
    values = {
        "GEO_H": str(FRUIT_ANNEALED_SPEC.hidden_size),
        "GEO_NL": "13",
        "GEO_HEADS": "16",
        "GEO_QLORA": "1024",
        "GEO_DENSE_INTER": "2048",
        "GEO_MOE_INTER": str(FRUIT_ANNEALED_SPEC.intermediate_size),
        "MOE_IMPL": "grouped",
        "ROPE_THETA": str(int(FRUIT_ANNEALED_SPEC.trained_rope_theta)),
        "FRUIT_ROPE_THETA": str(int(FRUIT_ANNEALED_SPEC.trained_rope_theta)),
        "SERVE_CONV": "1" if serve_native else "0",
        "GRAD_CKPT": "0",
        "FP8_LINEAR": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for name, value in values.items():
        previous = os.environ.get(name)
        if previous is not None and previous != value:
            raise ValueError(
                f"environment {name}={previous!r} conflicts with {value!r}"
            )
        os.environ[name] = value


def _load_trainer(trainer: Path, *, serve_native: bool):
    _configure_trainer_environment(serve_native=serve_native)
    trainer_root = str(trainer.parent.resolve())
    if trainer_root not in sys.path:
        sys.path.insert(0, trainer_root)
    sys.modules.pop("train_fruit", None)
    module = importlib.import_module("train_fruit")
    if Path(module.__file__).resolve() != trainer.resolve():
        raise RuntimeError("imported the wrong Fruit trainer module")
    expected = {
        "H": FRUIT_ANNEALED_SPEC.hidden_size,
        "NL": 13,
        "N_EXP": FRUIT_ANNEALED_SPEC.num_experts,
        "TOPK": FRUIT_CALIBRATION_TOPK,
        "MOE_INTER": FRUIT_ANNEALED_SPEC.intermediate_size,
        "THETA": FRUIT_ANNEALED_SPEC.trained_rope_theta,
    }
    for name, value in expected.items():
        if getattr(module, name) != value:
            raise ValueError(f"Fruit trainer geometry {name} mismatch")
    if bool(module.CONV["serve"]) != serve_native:
        raise ValueError("Fruit trainer serving convention mismatch")
    return module


@dataclass(frozen=True)
class _HFBinding:
    destination: str
    expert: int | None = None


def _hf_binding(name: str) -> _HFBinding:
    roots = {
        "lm_head.weight": _HFBinding("lm_head.weight"),
        "model.embed_tokens.weight": _HFBinding("embed_tokens.weight"),
        "model.norm.weight": _HFBinding("norm.weight"),
    }
    if name in roots:
        return roots[name]
    pieces = name.split(".")
    if len(pieces) < 5 or pieces[:2] != ["model", "layers"]:
        raise ValueError(f"unsupported Fruit BF16 tensor name: {name}")
    try:
        layer = int(pieces[2])
    except ValueError as exc:
        raise ValueError(f"invalid Fruit BF16 layer name: {name}") from exc
    if not 0 <= layer <= FRUIT_ANNEALED_SPEC.mtp_layer:
        raise ValueError(f"Fruit BF16 layer is out of range: {name}")
    suffix = ".".join(pieces[3:])
    if layer == FRUIT_ANNEALED_SPEC.mtp_layer:
        mtp_roots = {
            "eh_proj.weight": "mtp_eh_proj.weight",
            "enorm.weight": "mtp_enorm.weight",
            "hnorm.weight": "mtp_hnorm.weight",
            "shared_head.norm.weight": "norm.weight",
        }
        if suffix in mtp_roots:
            return _HFBinding(mtp_roots[suffix])
        prefix = "mtp_block."
    else:
        prefix = f"layers.{layer}."
    expert_prefix = "mlp.experts."
    if suffix.startswith(expert_prefix):
        expert_parts = suffix[len(expert_prefix) :].split(".")
        if len(expert_parts) != 3 or expert_parts[2] != "weight":
            raise ValueError(f"invalid Fruit BF16 expert tensor name: {name}")
        try:
            expert = int(expert_parts[0])
        except ValueError as exc:
            raise ValueError(f"invalid Fruit BF16 expert index: {name}") from exc
        if not 0 <= expert < FRUIT_ANNEALED_SPEC.num_experts:
            raise ValueError(f"Fruit BF16 expert is out of range: {name}")
        projection = {
            "gate_proj": "w_gate",
            "up_proj": "w_up",
            "down_proj": "w_down",
        }.get(expert_parts[1])
        if projection is None:
            raise ValueError(f"invalid Fruit BF16 expert projection: {name}")
        return _HFBinding(f"{prefix}mlp.{projection}", expert)
    if suffix == "mlp.gate.e_score_correction_bias":
        suffix = "mlp.e_score_correction_bias"
    return _HFBinding(prefix + suffix)


def _authenticate_bf16_base(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Fruit BF16 base must be a real directory")
    manifest_path = root / "MANIFEST.sha256"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != _BF16_BASE_MANIFEST_SHA256:
        raise ValueError("Fruit BF16 base MANIFEST.sha256 identity mismatch")
    entries: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if (
            len(fields) != 2
            or len(fields[0]) != 64
            or any(character not in "0123456789abcdef" for character in fields[0])
            or Path(fields[1]).name != fields[1]
            or fields[1] in entries
        ):
            raise ValueError("Fruit BF16 checksum manifest is malformed")
        entries[fields[1]] = fields[0]
    actual_names = {
        path.name
        for path in root.iterdir()
        if path.name != manifest_path.name and path.is_file() and not path.is_symlink()
    }
    if not set(entries) <= actual_names or actual_names - set(entries) - {"README.md"}:
        raise ValueError("Fruit BF16 base file inventory differs from its manifest")
    for name, expected_sha256 in sorted(entries.items()):
        if _sha256(root / name) != expected_sha256:
            raise ValueError(f"Fruit BF16 base checksum mismatch: {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "dtype": "bfloat16",
        "hidden_size": FRUIT_ANNEALED_SPEC.hidden_size,
        "moe_intermediate_size": FRUIT_ANNEALED_SPEC.intermediate_size,
        "n_routed_experts": FRUIT_ANNEALED_SPEC.num_experts,
        "num_experts_per_tok": FRUIT_CALIBRATION_TOPK,
        "num_hidden_layers": FRUIT_ANNEALED_SPEC.mtp_layer,
        "num_nextn_predict_layers": 1,
        "rope_interleave": True,
        "rope_theta": FRUIT_ANNEALED_SPEC.trained_rope_theta,
        "tie_word_embeddings": False,
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("Fruit BF16 base config differs from the calibration contract")
    if "quantization_config" in config:
        raise ValueError("Fruit calibration source must be an unquantized BF16 export")
    index_path = root / _BF16_BASE_INDEX
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(not isinstance(name, str) for name in weight_map)
        or any(not isinstance(filename, str) for filename in weight_map.values())
    ):
        raise ValueError("Fruit BF16 weight index is malformed")
    shard_names = {name for name in entries if name.endswith(".safetensors")}
    if set(weight_map.values()) != shard_names:
        raise ValueError("Fruit BF16 weight index shard inventory mismatch")
    return (
        {
            "kind": "authenticated_bf16_export",
            "directory": root.name,
            "manifest_sha256": manifest_sha256,
            "index_sha256": entries[_BF16_BASE_INDEX],
            "file_count": len(entries),
        },
        weight_map,
    )


def _load_bf16_model(
    root: Path,
    weight_map: dict[str, str],
    trainer: Any,
    device: torch.device,
) -> torch.nn.Module:
    model = trainer.Fruit()
    state = model.state_dict(keep_vars=True)
    bindings = {name: _hf_binding(name) for name in weight_map}
    destination_keys = {binding.destination for binding in bindings.values()}
    expected_keys = set(state) - _MODEL_MARKER_KEYS
    if destination_keys != expected_keys:
        missing = sorted(expected_keys - destination_keys)
        extra = sorted(destination_keys - expected_keys)
        raise ValueError(
            f"Fruit BF16/trainer tensor inventory mismatch: missing={missing}, extra={extra}"
        )
    indexed_destinations: dict[str, set[int]] = defaultdict(set)
    for binding in bindings.values():
        if binding.expert is not None:
            indexed_destinations[binding.destination].add(binding.expert)
    expected_experts = set(range(FRUIT_ANNEALED_SPEC.num_experts))
    if any(indices != expected_experts for indices in indexed_destinations.values()):
        raise ValueError("Fruit BF16 expert stack is incomplete")

    by_shard: dict[str, set[str]] = defaultdict(set)
    for name, filename in weight_map.items():
        by_shard[filename].add(name)
    loaded: set[tuple[str, int | None]] = set()
    with torch.no_grad():
        for filename, expected_names in sorted(by_shard.items()):
            path = root / filename
            with safe_open(path, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != expected_names:
                    raise ValueError(
                        f"Fruit BF16 shard tensor inventory mismatch: {filename}"
                    )
                for name in sorted(expected_names):
                    source = handle.get_tensor(name)
                    binding = bindings[name]
                    destination = state[binding.destination]
                    if binding.expert is not None:
                        destination = destination[binding.expert]
                    if tuple(source.shape) != tuple(destination.shape):
                        raise ValueError(f"Fruit BF16 tensor shape mismatch: {name}")
                    target_id = (binding.destination, binding.expert)
                    if target_id in loaded:
                        if not torch.equal(source, destination.to(dtype=source.dtype)):
                            raise ValueError(
                                f"Fruit BF16 duplicate tensor disagrees: {name}"
                            )
                    else:
                        destination.copy_(source)
                        loaded.add(target_id)
    expected_loaded = {
        (binding.destination, binding.expert) for binding in bindings.values()
    }
    if loaded != expected_loaded:
        raise ValueError("Fruit BF16 model load did not cover every trainer tensor")
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _verify_source_closure(
    model: torch.nn.Module,
    reference_path: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    reference_sha256 = _sha256(reference_path)
    if reference_sha256 != FRUIT_CALIBRATION_REFERENCE_SHA256:
        raise ValueError("Fruit calibration closure reference identity mismatch")
    with torch.serialization.safe_globals([TorchVersion]):
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(reference, dict)
        or reference.get("kind") != "fruit_full_vocabulary_kld_reference"
        or reference.get("schema_version") != 1
        or not isinstance(reference.get("metadata"), dict)
        or reference["metadata"].get("checkpoint_sha256")
        != FRUIT_ANNEALED_SPEC.checkpoint_sha256
    ):
        raise ValueError("Fruit calibration closure reference contract mismatch")
    token_ids = reference.get("token_ids")
    expected = reference.get("log_probs")
    positions = reference.get("positions")
    vocab_size = reference.get("vocab_size")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(token_id) is not int for token_id in token_ids)
        or type(positions) is not int
        or type(vocab_size) is not int
        or not isinstance(expected, torch.Tensor)
        or expected.dtype != torch.float32
        or tuple(expected.shape) != (positions, vocab_size)
        or positions != len(token_ids) - 1
    ):
        raise ValueError("Fruit calibration closure reference payload mismatch")
    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    torch.use_deterministic_algorithms(True)
    with (
        torch.inference_mode(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        hidden, mtp_hidden = model(inputs)
        logits = model.lm_head(hidden[0, :-1]).float()
        actual = torch.log_softmax(logits, dim=-1).cpu()
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError("Fruit calibration source closure output shape mismatch")
    teacher = expected.double()
    teacher -= torch.logsumexp(teacher, dim=1, keepdim=True)
    candidate = actual.double()
    candidate -= torch.logsumexp(candidate, dim=1, keepdim=True)
    delta = candidate - teacher
    forward_kl = (teacher.exp() * -delta).sum(dim=1)
    if not bool(torch.all(torch.isfinite(delta))) or bool(
        torch.any(forward_kl < -1e-8)
    ):
        raise ValueError("Fruit calibration source closure produced invalid metrics")
    metrics: dict[str, object] = {
        "reference_sha256": reference_sha256,
        "positions": positions,
        "vocab_size": vocab_size,
        "max_abs_logprob": float(delta.abs().max()),
        "rms_logprob": float(delta.square().mean().sqrt()),
        "mean_forward_kl": float(forward_kl.clamp_min(0).mean()),
        "max_forward_kl": float(forward_kl.clamp_min(0).max()),
        "top1_matches": int(
            torch.count_nonzero(teacher.argmax(dim=1) == candidate.argmax(dim=1))
        ),
        "mtp_positions": int(mtp_hidden.shape[1]),
        "mtp_finite": bool(torch.all(torch.isfinite(mtp_hidden))),
    }
    if (
        metrics["top1_matches"] != positions
        or any(
            float(metrics[name]) > limit
            for name, limit in FRUIT_CALIBRATION_CLOSURE_LIMITS.items()
        )
        or metrics["mtp_positions"] != positions
        or not metrics["mtp_finite"]
    ):
        raise ValueError(f"Fruit calibration source closure failed: {metrics}")
    return metrics


def _capture(
    model: torch.nn.Module,
    trainer: Any,
    documents: list[_CorpusDocument],
    *,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    chunks = {layer: _LayerChunks() for layer in FRUIT_CALIBRATION_LAYERS}
    current_document = {"id": -1}
    handles: list[Any] = []

    def install(layer: int, module: torch.nn.Module) -> None:
        def pre_hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if current_document["id"] < 0 or len(args) != 1:
                raise RuntimeError("Fruit calibration hook has no active document")
            inputs = args[0].detach().reshape(-1, trainer.H)
            logits = module.gate(inputs).float()
            scores = torch.sigmoid(logits)
            route_ids = (
                (scores + module.e_score_correction_bias)
                .topk(trainer.TOPK, dim=-1)
                .indices
            )
            route_weights = scores.gather(-1, route_ids)
            route_weights = (
                route_weights / route_weights.sum(dim=-1, keepdim=True)
            ) * trainer.ROUTED_SCALE
            chunks[layer].append(
                inputs,
                route_ids,
                route_weights,
                current_document["id"],
            )

        handles.append(module.register_forward_pre_hook(pre_hook))

    for layer in FRUIT_ANNEALED_SPEC.layers:
        install(layer, model.layers[layer].mlp)
    install(FRUIT_ANNEALED_SPEC.mtp_layer, model.mtp_block.mlp)

    torch.use_deterministic_algorithms(True)
    try:
        with (
            torch.inference_mode(),
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
        ):
            for document_id, document in enumerate(documents):
                current_document["id"] = document_id
                inputs = torch.tensor(
                    [document.token_ids], dtype=torch.long, device=device
                )
                output = model(inputs)
                del inputs, output
                print(
                    f"[{document_id + 1}/{len(documents)}] "
                    f"{document.axis} {document.split} {len(document.token_ids)} tokens",
                    flush=True,
                )
    finally:
        current_document["id"] = -1
        for handle in handles:
            handle.remove()
    return {layer: value.finish() for layer, value in chunks.items()}


def _capture_identity(
    *,
    checkpoint_sha256: str,
    source: dict[str, object],
    source_closure: dict[str, object],
    documents_per_axis: int,
    fit_per_axis: int,
    confirmation_per_axis: int,
    corpus: Path,
    corpus_sha256: str,
    tokenizer: Path,
    tokenizer_hashes: dict[str, str],
    trainer: Path,
    trainer_hashes: dict[str, str],
    documents: list[dict[str, object]],
    routed_scale: float,
    max_tokens: int,
    min_tokens: int,
) -> dict[str, object]:
    return {
        "schema": FRUIT_CALIBRATION_SCHEMA,
        "version": FRUIT_CALIBRATION_VERSION,
        "kind": "fruit_qsrt_activation_calibration",
        "checkpoint_sha256": checkpoint_sha256,
        "source": source,
        "source_closure": source_closure,
        "corpus": {
            "filename": corpus.name,
            "sha256": corpus_sha256,
        },
        "tokenizer": {
            "directory": tokenizer.name,
            "files": tokenizer_hashes,
        },
        "trainer": {
            "filename": trainer.name,
            "files": trainer_hashes,
        },
        "geometry": {
            "hidden_size": FRUIT_ANNEALED_SPEC.hidden_size,
            "intermediate_size": FRUIT_ANNEALED_SPEC.intermediate_size,
            "experts": FRUIT_ANNEALED_SPEC.num_experts,
            "topk": FRUIT_CALIBRATION_TOPK,
            "layers": list(FRUIT_CALIBRATION_LAYERS),
        },
        "routed_scale": routed_scale,
        "conventions": {
            "serve_conv_v": 2,
            "trained_rope_theta": FRUIT_ANNEALED_SPEC.trained_rope_theta,
            "moe_impl": "grouped",
            "sdpa_backend": "math",
            "deterministic_algorithms": True,
        },
        "token_bounds": {"minimum": min_tokens, "maximum": max_tokens},
        "protocol": {
            "normalization": "fruit_data_prep_calib_jsonl_v1",
            "partition": "prompt_family_and_truncated_token_sha256_v1",
            "documents_per_axis": documents_per_axis,
            "fit_per_axis": fit_per_axis,
            "confirmation_per_axis": confirmation_per_axis,
            "validation_per_axis": (
                documents_per_axis - fit_per_axis - confirmation_per_axis
            ),
        },
        "documents": documents,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("trainer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--documents-per-axis", type=int, default=64)
    parser.add_argument("--fit-per-axis", type=int, default=40)
    parser.add_argument("--confirmation-per-axis", type=int, default=16)
    parser.add_argument("--min-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.corpus, args.reference, args.trainer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.source.is_dir() or args.source.is_symlink():
        raise ValueError(
            "Fruit calibration source must be the authenticated BF16 directory"
        )
    if not args.tokenizer.is_dir():
        raise FileNotFoundError(args.tokenizer)
    requested_protocol = {
        "normalization": "fruit_data_prep_calib_jsonl_v1",
        "partition": "prompt_family_and_truncated_token_sha256_v1",
        "documents_per_axis": args.documents_per_axis,
        "fit_per_axis": args.fit_per_axis,
        "confirmation_per_axis": args.confirmation_per_axis,
        "validation_per_axis": (
            args.documents_per_axis - args.fit_per_axis - args.confirmation_per_axis
        ),
    }
    if (
        requested_protocol != FRUIT_CALIBRATION_PROTOCOL
        or {
            "minimum": args.min_tokens,
            "maximum": args.max_tokens,
        }
        != FRUIT_CALIBRATION_TOKEN_BOUNDS
    ):
        raise ValueError("capture arguments differ from the production protocol")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("calibration output directory must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Fruit calibration capture requires CUDA")
    torch.cuda.set_device(device)

    source_evidence, weight_map = _authenticate_bf16_base(args.source)
    if any(
        source_evidence.get(name) != value
        for name, value in FRUIT_CALIBRATION_SOURCE.items()
    ):
        raise ValueError("Fruit calibration source identity mismatch")
    checkpoint_sha256 = FRUIT_ANNEALED_SPEC.checkpoint_sha256
    tokenizer_hashes = {
        name: _sha256(args.tokenizer / name)
        for name in _SMALL_TOKENIZER_FILES
        if (args.tokenizer / name).is_file()
    }
    if tokenizer_hashes != FRUIT_CALIBRATION_TOKENIZER_FILES:
        raise ValueError("Fruit calibration tokenizer identity mismatch")
    trainer_hashes = {args.trainer.name: _sha256(args.trainer)}
    if trainer_hashes != FRUIT_CALIBRATION_TRAINER_FILES:
        raise ValueError("Fruit calibration trainer identity mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=False,
        local_files_only=True,
    )
    documents, corpus_sha256 = _sample_documents(
        args.corpus,
        tokenizer,
        documents_per_axis=args.documents_per_axis,
        fit_per_axis=args.fit_per_axis,
        confirmation_per_axis=args.confirmation_per_axis,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    if (
        corpus_sha256 != FRUIT_CALIBRATION_CORPUS["sha256"]
        or tuple(sorted({document.axis for document in documents}))
        != FRUIT_CALIBRATION_AXES
    ):
        raise ValueError("Fruit calibration corpus identity mismatch")
    document_ledger = [
        {
            "id": document_id,
            "axis": document.axis,
            "source": document.source,
            "line": document.line,
            "sha256": document.digest,
            "family_sha256": document.family_digest,
            "split": document.split,
            "tokens": len(document.token_ids),
            "token_ids_sha256": document.token_digest,
        }
        for document_id, document in enumerate(documents)
    ]
    split_counts = Counter(document.split for document in documents)
    axis_counts = Counter(document.axis for document in documents)
    print(
        "Fruit calibration plan "
        + json.dumps(
            {
                "documents": len(documents),
                "splits": dict(sorted(split_counts.items())),
                "axes": dict(sorted(axis_counts.items())),
                "tokens": sum(len(document.token_ids) for document in documents),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    trainer = _load_trainer(args.trainer, serve_native=True)
    model = _load_bf16_model(args.source, weight_map, trainer, device)
    source_closure = _verify_source_closure(
        model,
        args.reference,
        device=device,
    )
    captured = _capture(model, trainer, documents, device=device)
    routed_scale = float(trainer.ROUTED_SCALE)
    del model
    torch.cuda.empty_cache()

    identity = _capture_identity(
        checkpoint_sha256=checkpoint_sha256,
        source=source_evidence,
        source_closure=source_closure,
        documents_per_axis=args.documents_per_axis,
        fit_per_axis=args.fit_per_axis,
        confirmation_per_axis=args.confirmation_per_axis,
        corpus=args.corpus,
        corpus_sha256=corpus_sha256,
        tokenizer=args.tokenizer,
        tokenizer_hashes=tokenizer_hashes,
        trainer=args.trainer,
        trainer_hashes=trainer_hashes,
        documents=document_ledger,
        routed_scale=routed_scale,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
    )
    capture_id = _sha256_bytes(_canonical_json(identity).encode("utf-8"))
    fit_ids = torch.tensor(
        [
            document_id
            for document_id, document in enumerate(documents)
            if document.split == "fit"
        ],
        dtype=torch.int32,
    )
    layers: dict[str, dict[str, object]] = {}
    for layer in FRUIT_CALIBRATION_LAYERS:
        tensors = captured.pop(layer)
        fit_mask = torch.isin(tensors["document_ids"], fit_ids)
        if not bool(torch.any(fit_mask)):
            raise ValueError(f"Fruit calibration layer {layer} has no fit rows")
        fit_inputs = tensors["inputs"][fit_mask]
        fit_route_weights = tensors["route_weights"][fit_mask].float()
        h13_weights = fit_route_weights.square().sum(dim=1)
        global_h13, denominator = weighted_covariance(
            fit_inputs,
            h13_weights,
            device=device,
            chunk_rows=512,
            return_cpu=True,
        )
        tensors["global_h13"] = global_h13.contiguous()
        filename = f"layer-{layer:03d}.safetensors"
        path = args.output / filename
        _atomic_safetensors(
            path,
            tensors,
            {
                "schema": FRUIT_CALIBRATION_SCHEMA,
                "version": str(FRUIT_CALIBRATION_VERSION),
                "capture_id": capture_id,
                "layer": str(layer),
            },
        )
        layers[str(layer)] = {
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "rows": int(tensors["inputs"].shape[0]),
            "fit_h13_gate_square_sum": denominator,
        }
        print(
            f"layer {layer}: {tensors['inputs'].shape[0]} rows, "
            f"H13 weight={denominator:.6g}, {path.stat().st_size / (1 << 20):.2f} MiB",
            flush=True,
        )

    manifest: dict[str, object] = {
        **identity,
        "capture_id": capture_id,
        "layers": layers,
        "complete": True,
    }
    manifest["fingerprint"] = calibration_fingerprint(manifest)
    _atomic_text(
        args.output / "calibration-manifest.json",
        _canonical_json(manifest),
    )
    print(
        f"complete Fruit calibration capture: {args.output} "
        f"fingerprint={manifest['fingerprint']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
