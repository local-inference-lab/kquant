#!/usr/bin/env python
"""Stream a fixed Kimi-K3 prefill through the official PyTorch decoder.

Only one decoder layer is materialized at a time. Routed experts are
reconstructed in bulk after routing and released after the layer. The expert
source may be the official MXFP4 checkpoint, a hybrid EXL3/MXFP4 artifact, or
a completed TP-independent QSRT artifact with balanced atoms and exact X4T.
Serialized MXFP8 non-expert projections can be dequantized into the same
official PyTorch layer implementation.

This is a correctness reference, not a serving implementation.  It deliberately
uses eager PyTorch MLA attention and the official FLA KDA implementation; it
does not call vLLM or any serving GEMM/MoE kernel.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kquant import constants as KQ_C
from kquant.correctness import DEFAULT_PROMPT, write_json
from kquant.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12, QSRT_CODEBOOKS
from kquant.kimi_stream import (
    MODEL_TENSOR_PREFIX,
    IndexedSafetensors,
    StreamState,
    assign_parameter,
    atomic_torch_save,
    fit_checkpoint_parameter,
    load_stream_state,
    save_stream_state,
    write_trace_manifest,
    write_trace_tensor,
)
from kquant.qsrt import (
    SCHEMA as QSRT_SCHEMA,
    PackedQSRTTrellis,
    QSRTTrellisDescriptor,
    decode_qsrt_exl3_weight,
    matrix_rate_axis,
)
from kquant.pack.qsrt_materialize import (
    QSRT_ARTIFACT_KIND,
    QSRT_ARTIFACT_SCHEMA_VERSION,
    QSRT_MANIFEST_FILENAME,
)
from kquant.pack.qsrt_atoms import (
    FORMAT_SECTION_BYTES,
    LAYER_HEADER_BYTES,
    LAYER_PREFIX,
    SHARED_SCALE_SECTION_BYTES,
    QSRTAtomLayerReader,
    layer_filename as qsrt_layer_filename,
)
from kquant.teacher_proxy import align_routed_post_situ
from kquant.teacher_proxy_suite import (
    LoadedTeacherProxySuite,
    load_teacher_proxy_suite,
    parse_layer_list,
)
from kquant.x4t import X4T_LAYER_FIXED_BYTES, X4TLayerReader, x4t_layer_path

DEFAULT_CHECKPOINT = Path(
    "/home/luke/.cache/huggingface/hub/"
    "models--moonshotai--Kimi-K3/snapshots/"
    "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
)
DEFAULT_EXLLAMAV3_DIR = Path("/home/luke/projects/exllamav3")
EXPERT_MATRICES = ("w1", "w2", "w3")
EXL3_PARTS = ("trellis", "suh", "svh")


@dataclass(frozen=True)
class QSRTArtifact:
    """Strict, read-only inventory for one completed QSRT artifact."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "QSRTArtifact":
        root = Path(root).expanduser().resolve()
        manifest_path = root / QSRT_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise TypeError(f"{manifest_path}: expected a JSON object")
        expected_scalars = {
            "kind": QSRT_ARTIFACT_KIND,
            "schema_version": QSRT_ARTIFACT_SCHEMA_VERSION,
            "complete": True,
            "codec": "QSRT",
        }
        for name, expected in expected_scalars.items():
            if manifest.get(name) != expected:
                raise ValueError(
                    f"{manifest_path}: {name}={manifest.get(name)!r}, "
                    f"expected {expected!r}"
                )
        codebook = manifest.get("trellis_codebook")
        if codebook not in QSRT_CODEBOOKS:
            raise ValueError(
                f"{manifest_path}: unsupported trellis_codebook {codebook!r}"
            )
        layers = manifest.get("layers")
        expected_layers = {str(layer) for layer in KQ_C.MOE_LAYERS}
        if not isinstance(layers, dict) or set(layers) != expected_layers:
            raise ValueError(
                f"{manifest_path}: expected exactly {len(expected_layers)} "
                "MoE layer entries"
            )
        expected_atom_files = {
            str(layers[str(layer)].get("qsrt_atoms"))
            for layer in KQ_C.MOE_LAYERS
        }
        canonical_atom_files = {
            qsrt_layer_filename(layer) for layer in KQ_C.MOE_LAYERS
        }
        if expected_atom_files != canonical_atom_files:
            raise ValueError(f"{manifest_path}: noncanonical atom inventory")
        actual_atom_files = {
            path.name
            for path in root.glob(f"{LAYER_PREFIX}*.safetensors")
            if path.is_file()
        }
        if actual_atom_files != expected_atom_files:
            raise ValueError(
                "QSRT atom layer inventory does not close; "
                f"missing={sorted(expected_atom_files - actual_atom_files)[:3]}, "
                f"extra={sorted(actual_atom_files - expected_atom_files)[:3]}"
            )
        expected_x4t_files = {
            str(layers[str(layer)].get("x4t"))
            for layer in KQ_C.MOE_LAYERS
        }
        canonical_x4t_files = {
            x4t_layer_path(Path("."), layer).name for layer in KQ_C.MOE_LAYERS
        }
        if expected_x4t_files != canonical_x4t_files:
            raise ValueError(f"{manifest_path}: noncanonical X4T inventory")
        actual_x4t_files = {
            path.name
            for path in root.glob("x4t-layer-*.safetensors")
            if path.is_file()
        }
        if actual_x4t_files != expected_x4t_files:
            raise ValueError(
                "QSRT X4T layer inventory does not close; "
                f"missing={sorted(expected_x4t_files - actual_x4t_files)[:3]}, "
                f"extra={sorted(actual_x4t_files - expected_x4t_files)[:3]}"
            )
        partials = sorted(root.glob(".*.partial"))
        if partials:
            raise ValueError(
                f"QSRT artifact contains partial files: {partials[:3]}"
            )
        return cls(root=root, manifest_path=manifest_path, manifest=manifest)

    def layer_path(self, layer_idx: int) -> Path:
        if layer_idx not in KQ_C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer_idx} is not a K3 MoE layer")
        path = self.root / qsrt_layer_filename(layer_idx)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def x4t_layer_path(self, layer_idx: int) -> Path:
        if layer_idx not in KQ_C.MOE_LAYERS:
            raise ValueError(f"decoder layer {layer_idx} is not a K3 MoE layer")
        path = x4t_layer_path(self.root, layer_idx)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @property
    def codebook(self) -> str:
        return str(self.manifest["trellis_codebook"])


def _require_reference_dependencies() -> dict[str, Any]:
    """Import the official-model dependencies with a Transformers 5 shim."""
    try:
        import transformers
        import transformers.utils.generic as generic
        from compressed_tensors import (
            MXFP4PackedCompressor,
            QuantizationConfig,
        )
        from transformers import AutoConfig, AutoTokenizer
        from transformers.dynamic_module_utils import (
            get_class_from_dynamic_module,
        )
        from transformers.masking_utils import create_causal_mask
        from transformers.utils.output_capturing import OutputRecorder
    except ImportError as exc:
        raise RuntimeError(
            "the streaming reference needs the vLLM environment's "
            "transformers, compressed_tensors, and fla-core packages"
        ) from exc

    # Kimi-K3's remote code imports OutputRecorder from its Transformers 4.56
    # location. Transformers 5 moved it to output_capturing.
    compatibility_shim = not hasattr(generic, "OutputRecorder")
    if compatibility_shim:
        generic.OutputRecorder = OutputRecorder

    return {
        "transformers": transformers,
        "AutoConfig": AutoConfig,
        "AutoTokenizer": AutoTokenizer,
        "get_class_from_dynamic_module": get_class_from_dynamic_module,
        "create_causal_mask": create_causal_mask,
        "MXFP4PackedCompressor": MXFP4PackedCompressor,
        "QuantizationConfig": QuantizationConfig,
        "compatibility_shim": compatibility_shim,
    }


def _load_official_code(
    checkpoint: Path, dependencies: dict[str, Any]
) -> tuple[Any, Any, Any]:
    auto_config = dependencies["AutoConfig"]
    get_class = dependencies["get_class_from_dynamic_module"]
    config = auto_config.from_pretrained(
        str(checkpoint), trust_remote_code=True
    )
    text_config = config.text_config
    model_class = get_class(
        "modeling_kimi_k3.KimiK3ForConditionalGeneration",
        str(checkpoint),
    )
    package = model_class.__module__.rsplit(".", 1)[0]
    linear_module = importlib.import_module(f"{package}.modeling_kimi_linear")

    # KimiLinearModel forcibly selects FlashAttention 2, but constructing the
    # individual official decoder layers lets this numerical reference use the
    # ordinary eager attention equation. KDA still runs through official FLA.
    text_config._attn_implementation = "eager"
    text_config.use_cache = False
    return config, text_config, linear_module


def _new_meta_layer(linear_module: Any, config: Any, layer_idx: int):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            layer = linear_module.KimiDecoderLayer(config, layer_idx)
    finally:
        torch.set_default_dtype(previous_dtype)
    layer.eval()
    return layer


@dataclass
class NonexpertLoadStats:
    serialized_bytes: int = 0
    dense_bytes: int = 0
    mxfp8_parameters: list[str] = field(default_factory=list)
    load_seconds: float = 0.0


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _dequantize_mxfp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    device: torch.device,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct ModelOpt row-wise E4M3/E8M0-K32 weights."""
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"MXFP8 weight must be E4M3, got {weight.dtype}")
    if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError(f"MXFP8 scale must be uint8/E8M0, got {scale.dtype}")
    if weight.ndim != 2 or weight.shape[1] % 32:
        raise ValueError(
            f"MXFP8 weight requires [N,K] with K divisible by 32, got "
            f"{tuple(weight.shape)}"
        )
    expected_scale = (int(weight.shape[0]), int(weight.shape[1]) // 32)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"MXFP8 scale shape {tuple(scale.shape)} != {expected_scale}"
        )

    values = weight.to(device=device)
    scales = scale.to(device=device)
    if scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e8m0fnu)
    dense = (
        values.float()
        .view(weight.shape[0], weight.shape[1] // 32, 32)
        .mul_(scales.float().unsqueeze(-1))
        .view(weight.shape)
        .to(dtype=output_dtype)
    )
    return dense


def _load_nonexpert_parameters(
    layer: torch.nn.Module,
    *,
    layer_idx: int,
    store: IndexedSafetensors,
    device: torch.device,
) -> tuple[list[str], NonexpertLoadStats]:
    started = time.monotonic()
    prefix = f"{MODEL_TENSOR_PREFIX}.layers.{layer_idx}."
    compatibility_fixes: list[str] = []
    stats = NonexpertLoadStats()
    parameters = [
        (name, parameter)
        for name, parameter in list(layer.named_parameters())
        if not name.startswith("block_sparse_moe.experts.")
    ]
    requested: list[str] = []
    for name, unused_parameter in parameters:
        if name.startswith("block_sparse_moe.experts."):
            continue
        checkpoint_name = prefix + name
        requested.append(checkpoint_name)
        scale_name = f"{checkpoint_name}_scale"
        if scale_name in store.weight_map:
            requested.append(scale_name)
    loaded = store.get_many(requested)

    for name, parameter in parameters:
        checkpoint_name = prefix + name
        value = loaded[checkpoint_name]
        stats.serialized_bytes += _tensor_bytes(value)
        value, fix = fit_checkpoint_parameter(
            checkpoint_name, value, tuple(parameter.shape)
        )
        if fix is not None:
            compatibility_fixes.append(f"{checkpoint_name}: {fix}")
        scale_name = f"{checkpoint_name}_scale"
        if value.dtype == torch.float8_e4m3fn:
            if scale_name not in loaded:
                raise KeyError(
                    f"{checkpoint_name}: E4M3 weight has no weight_scale"
                )
            scale = loaded[scale_name]
            stats.serialized_bytes += _tensor_bytes(scale)
            value = _dequantize_mxfp8_weight(
                value,
                scale,
                device=device,
                output_dtype=parameter.dtype,
            )
            stats.mxfp8_parameters.append(checkpoint_name)
        else:
            value = value.to(device=device)
        stats.dense_bytes += _tensor_bytes(value)
        assign_parameter(layer, name, value)

    unresolved = [
        name
        for name, parameter in layer.named_parameters()
        if not name.startswith("block_sparse_moe.experts.")
        and parameter.device.type == "meta"
    ]
    if unresolved:
        raise RuntimeError(
            f"layer {layer_idx}: non-expert parameters remain on meta: "
            f"{unresolved[:8]}"
        )
    meta_buffers = [
        name
        for name, buffer in layer.named_buffers()
        if buffer.device.type == "meta"
    ]
    if meta_buffers:
        raise RuntimeError(
            f"layer {layer_idx}: buffers remain on meta: {meta_buffers[:8]}"
        )
    stats.load_seconds = time.monotonic() - started
    return compatibility_fixes, stats


@dataclass
class ExpertLoadStats:
    expert_ids: list[int] = field(default_factory=list)
    exl3_expert_ids: list[int] = field(default_factory=list)
    kept_expert_ids: list[int] = field(default_factory=list)
    dense_bytes: int = 0
    packed_bytes: int = 0
    load_seconds: float = 0.0
    prefetched_expert_ids: list[int] = field(default_factory=list)
    prefetch_hit_expert_ids: list[int] = field(default_factory=list)
    prefetch_miss_expert_ids: list[int] = field(default_factory=list)


def _expert_base(layer_idx: int, expert_id: int) -> str:
    return (
        f"{MODEL_TENSOR_PREFIX}.layers.{layer_idx}."
        f"block_sparse_moe.experts.{expert_id}"
    )


def _expert_representation(
    store: IndexedSafetensors,
    *,
    layer_idx: int,
    expert_id: int,
) -> str:
    base = _expert_base(layer_idx, expert_id)
    has_exl3 = f"{base}.w1.exl3_trellis" in store.weight_map
    has_mxfp4 = f"{base}.w1.weight_packed" in store.weight_map
    if has_exl3 == has_mxfp4:
        raise ValueError(
            f"layer {layer_idx} expert {expert_id}: expected exactly one "
            f"of EXL3 or MXFP4 storage"
        )
    return "exl3" if has_exl3 else "mxfp4"


def _expert_tensor_names(
    store: IndexedSafetensors,
    *,
    layer_idx: int,
    expert_ids: list[int],
) -> list[str]:
    names: list[str] = []
    for expert_id in expert_ids:
        base = _expert_base(layer_idx, expert_id)
        representation = _expert_representation(
            store, layer_idx=layer_idx, expert_id=expert_id
        )
        for matrix in EXPERT_MATRICES:
            if representation == "exl3":
                names.extend(
                    f"{base}.{matrix}.exl3_{part}" for part in EXL3_PARTS
                )
            else:
                names.extend(
                    (
                        f"{base}.{matrix}.weight_packed",
                        f"{base}.{matrix}.weight_scale",
                    )
                )
    return names


def _trace_expert_ids(trace_root: Path, layer_idx: int) -> list[int]:
    path = (
        trace_root
        / "tp-rank-000"
        / f"layer-{layer_idx:03d}.call-000000.canonical_topk_ids.pt"
    )
    if not path.is_file():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=True)
    value = payload.get("tensor") if isinstance(payload, dict) else None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path}: missing trace tensor")
    return sorted({int(expert_id) for expert_id in value.view(-1).tolist()})


class ExpertRawPrefetcher:
    """Fault expected active-expert tensors into RAM one layer ahead."""

    def __init__(
        self,
        *,
        store: IndexedSafetensors,
        trace_root: Path,
        end_layer: int,
    ):
        self.store = store
        self.trace_root = trace_root.expanduser().resolve()
        self.end_layer = end_layer
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="k3-expert-prefetch"
        )
        self._futures: dict[
            int, tuple[list[int], Future[dict[str, torch.Tensor]]]
        ] = {}

    def schedule(self, layer_idx: int) -> bool:
        if layer_idx >= self.end_layer or layer_idx in self._futures:
            return False
        expert_ids = _trace_expert_ids(self.trace_root, layer_idx)
        if not expert_ids:
            return False
        names = _expert_tensor_names(
            self.store, layer_idx=layer_idx, expert_ids=expert_ids
        )
        future = self._executor.submit(
            self.store.get_many, names, clone=True
        )
        self._futures[layer_idx] = (expert_ids, future)
        return True

    def schedule_first(self, start_layer: int) -> None:
        for layer_idx in range(start_layer, self.end_layer):
            if self.schedule(layer_idx):
                return

    def take(
        self, layer_idx: int
    ) -> tuple[list[int], dict[str, torch.Tensor]]:
        item = self._futures.pop(layer_idx, None)
        if item is None:
            return [], {}
        expert_ids, future = item
        return expert_ids, future.result()

    def schedule_after(self, layer_idx: int) -> None:
        for candidate in range(layer_idx + 1, self.end_layer):
            if candidate in self._futures:
                return
            if self.schedule(candidate):
                return

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


class StagedExperts:
    """Bulk-materialize active experts before official PyTorch MoE inference."""

    def __init__(
        self,
        *,
        store: IndexedSafetensors,
        layer_idx: int,
        scheme: Any,
        compressor: Any,
        device: torch.device,
        mcg_mult: int | None,
        exllamav3_dir: Path,
        prefetcher: ExpertRawPrefetcher | None,
    ):
        self.store = store
        self.layer_idx = layer_idx
        self.scheme = scheme
        self.compressor = compressor
        self.device = device
        self.mcg_mult = mcg_mult
        self.exllamav3_dir = exllamav3_dir
        self.prefetcher = prefetcher
        self.stats = ExpertLoadStats()
        self._handles: list[Any] = []
        self._loaded_experts: list[torch.nn.Module] = []
        self._linear_exl3: Any | None = None

    def install(
        self, gate: torch.nn.Module, experts: torch.nn.ModuleList
    ) -> None:
        self._handles.append(
            gate.register_forward_hook(
                lambda unused_module, unused_args, output: self._stage(
                    experts, output[0]
                )
            )
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for expert in self._loaded_experts:
            self._release(expert)
        self._loaded_experts.clear()

    def _import_linear_exl3(self) -> Any:
        if self._linear_exl3 is None:
            path = str(self.exllamav3_dir.expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
            from exllamav3.modules.quant.exl3 import LinearEXL3

            self._linear_exl3 = LinearEXL3
        return self._linear_exl3

    def _reconstruct_exl3(
        self,
        raw: dict[str, torch.Tensor],
        *,
        base: str,
        matrix: str,
        expected: tuple[int, ...],
    ) -> torch.Tensor:
        if self.mcg_mult is None:
            raise ValueError("EXL3 expert encountered without mcg_mult")
        if len(expected) != 2:
            raise ValueError(f"{base}.{matrix}: expected a matrix, got {expected}")
        linear_exl3 = self._import_linear_exl3()
        marker = torch.tensor(
            self.mcg_mult & 0xFFFFFFFF,
            dtype=torch.uint32,
            device=self.device,
        ).view(torch.int32)
        values = {
            part: raw[f"{base}.{matrix}.exl3_{part}"]
            .to(self.device)
            .contiguous()
            for part in EXL3_PARTS
        }
        module = linear_exl3(
            config=None,
            in_features=expected[1],
            out_features=expected[0],
            suh=values["suh"],
            svh=values["svh"],
            trellis=values["trellis"],
            mcg=marker,
            out_dtype=torch.float16,
        )
        dense = (
            module.get_weight_tensor()
            .T.contiguous()
            .to(dtype=torch.bfloat16)
        )
        del module, values, marker
        return dense

    def _stage(
        self,
        experts: torch.nn.ModuleList,
        topk_ids: torch.Tensor,
    ) -> None:
        if self.stats.expert_ids:
            raise RuntimeError(
                f"layer {self.layer_idx}: experts were staged more than once"
            )
        started = time.monotonic()
        expert_ids = sorted(
            {int(expert_id) for expert_id in topk_ids.view(-1).tolist()}
        )
        self.stats.expert_ids = expert_ids

        prefetched_ids: list[int] = []
        raw: dict[str, torch.Tensor] = {}
        if self.prefetcher is not None:
            prefetched_ids, raw = self.prefetcher.take(self.layer_idx)
            self.prefetcher.schedule_after(self.layer_idx)
        self.stats.prefetched_expert_ids = prefetched_ids
        prefetched_set = set(prefetched_ids)
        actual_set = set(expert_ids)
        self.stats.prefetch_hit_expert_ids = sorted(
            actual_set & prefetched_set
        )
        self.stats.prefetch_miss_expert_ids = sorted(
            actual_set - prefetched_set
        )

        required_names = _expert_tensor_names(
            self.store, layer_idx=self.layer_idx, expert_ids=expert_ids
        )
        missing_names = [name for name in required_names if name not in raw]
        if missing_names:
            raw.update(self.store.get_many(missing_names))
        self.stats.packed_bytes = sum(_tensor_bytes(value) for value in raw.values())

        for expert_id in expert_ids:
            expert = experts[expert_id]
            base = _expert_base(self.layer_idx, expert_id)
            representation = _expert_representation(
                self.store,
                layer_idx=self.layer_idx,
                expert_id=expert_id,
            )
            if representation == "exl3":
                self.stats.exl3_expert_ids.append(expert_id)
            else:
                self.stats.kept_expert_ids.append(expert_id)
            for matrix in EXPERT_MATRICES:
                expected = tuple(getattr(expert, matrix).weight.shape)
                if representation == "exl3":
                    dense = self._reconstruct_exl3(
                        raw,
                        base=base,
                        matrix=matrix,
                        expected=expected,
                    )
                else:
                    packed = raw[f"{base}.{matrix}.weight_packed"]
                    scale = raw[f"{base}.{matrix}.weight_scale"]
                    dense = self.compressor.decompress(
                        {
                            "weight_packed": packed,
                            "weight_scale": scale,
                        },
                        self.scheme,
                    )["weight"]
                    dense = dense.to(device=self.device)
                if tuple(dense.shape) != expected:
                    raise ValueError(
                        f"{base}.{matrix}: reconstructed shape "
                        f"{tuple(dense.shape)} != {expected}"
                    )
                self.stats.dense_bytes += _tensor_bytes(dense)
                assign_parameter(expert, f"{matrix}.weight", dense)
            self._loaded_experts.append(expert)
        self.stats.load_seconds += time.monotonic() - started

    @staticmethod
    def _release(expert: torch.nn.Module) -> None:
        for matrix in EXPERT_MATRICES:
            parameter = getattr(expert, matrix).weight
            replacement = torch.empty(
                tuple(parameter.shape),
                dtype=parameter.dtype,
                device="meta",
            )
            assign_parameter(expert, f"{matrix}.weight", replacement)


def _decode_qsrt_trellis_matrix(
    reader: QSRTAtomLayerReader,
    *,
    expert_id: int,
    matrix: str,
    expected: tuple[int, ...],
    device: torch.device,
    codebook: str = CODEBOOK_SQG_XOR_CHEB_T12,
) -> tuple[torch.Tensor, int]:
    """Reconstruct one QSRT trellis matrix as an official ``[out, in]`` weight."""

    if len(expected) != 2:
        raise ValueError(
            f"layer {reader.header.layer} expert {expert_id} {matrix}: "
            f"expected a matrix, got {expected}"
        )
    if expected[0] % 16 or expected[1] % 16:
        raise ValueError(
            f"layer {reader.header.layer} expert {expert_id} {matrix}: "
            f"shape {expected} is not 16x16 tiled"
        )
    format_spec = reader.format_specs[expert_id]
    if format_spec.is_x4t:
        raise ValueError(f"expert {expert_id} is not in the compressed tier")
    parts = reader.read_compressed_matrix(expert_id, matrix)
    mode_id = format_spec.r2 if matrix == "w2" else format_spec.r13
    descriptor = QSRTTrellisDescriptor(
        mode_id=mode_id,
        rate_axis=matrix_rate_axis(matrix),
        k_tiles=expected[1] // 16,
        n_tiles=expected[0] // 16,
        schema=QSRT_SCHEMA,
    )
    packed = PackedQSRTTrellis(
        descriptor,
        parts["trellis"].to(device=device).contiguous(),
    )
    packed.validate()
    suh = parts["suh"].to(device=device).contiguous()
    svh = parts["svh"].to(device=device).contiguous()
    dense = (
        decode_qsrt_exl3_weight(packed, suh, svh, codebook=codebook)
        .T.contiguous()
        .to(dtype=torch.bfloat16)
    )
    if tuple(dense.shape) != expected:
        raise ValueError(
            f"layer {reader.header.layer} expert {expert_id} {matrix}: "
            f"reconstructed shape {tuple(dense.shape)} != {expected}"
        )
    local_scale = parts["suh"] if matrix == "w2" else parts["svh"]
    bytes_read = _tensor_bytes(parts["trellis"]) + _tensor_bytes(local_scale)
    return dense, bytes_read


class QSRTStagedExperts:
    """Stage active experts from one QSRT atom/X4T layer pair."""

    def __init__(
        self,
        *,
        reader: QSRTAtomLayerReader,
        x4t_reader: X4TLayerReader,
        layer_idx: int,
        scheme: Any,
        compressor: Any,
        device: torch.device,
        codebook: str = CODEBOOK_SQG_XOR_CHEB_T12,
    ):
        if reader.header.layer != layer_idx:
            raise ValueError(
                f"QSRT atom layer {reader.header.layer} != decoder layer {layer_idx}"
            )
        if x4t_reader.layer != layer_idx:
            raise ValueError(
                f"X4T layer {x4t_reader.layer} != decoder layer {layer_idx}"
            )
        self.reader = reader
        self.x4t_reader = x4t_reader
        self.layer_idx = layer_idx
        self.scheme = scheme
        self.compressor = compressor
        self.device = device
        self.codebook = codebook
        self.stats = ExpertLoadStats(
            packed_bytes=(
                LAYER_HEADER_BYTES
                + FORMAT_SECTION_BYTES
                + SHARED_SCALE_SECTION_BYTES
                + X4T_LAYER_FIXED_BYTES
            )
        )
        self._handles: list[Any] = []
        self._loaded_experts: list[torch.nn.Module] = []

    def install(
        self, gate: torch.nn.Module, experts: torch.nn.ModuleList
    ) -> None:
        self._handles.append(
            gate.register_forward_hook(
                lambda unused_module, unused_args, output: self._stage(
                    experts, output[0]
                )
            )
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for expert in self._loaded_experts:
            StagedExperts._release(expert)
        self._loaded_experts.clear()
        self.reader.close()

    def _reconstruct_compressed(
        self,
        *,
        expert_id: int,
        matrix: str,
        expected: tuple[int, ...],
    ) -> tuple[torch.Tensor, int]:
        return _decode_qsrt_trellis_matrix(
            self.reader,
            expert_id=expert_id,
            matrix=matrix,
            expected=expected,
            device=self.device,
            codebook=self.codebook,
        )

    def _stage(
        self,
        experts: torch.nn.ModuleList,
        topk_ids: torch.Tensor,
    ) -> None:
        if self.stats.expert_ids:
            raise RuntimeError(
                f"layer {self.layer_idx}: experts were staged more than once"
            )
        started = time.monotonic()
        expert_ids = sorted(
            {int(expert_id) for expert_id in topk_ids.view(-1).tolist()}
        )
        self.stats.expert_ids = expert_ids

        for expert_id in expert_ids:
            expert = experts[expert_id]
            format_spec = self.reader.format_specs[expert_id]
            if format_spec.is_x4t:
                self.stats.kept_expert_ids.append(expert_id)
                self.stats.packed_bytes += 28
            else:
                self.stats.exl3_expert_ids.append(expert_id)
            for matrix in EXPERT_MATRICES:
                expected = tuple(getattr(expert, matrix).weight.shape)
                if format_spec.is_x4t:
                    packed = self.x4t_reader.read(expert_id, matrix)
                    self.stats.packed_bytes += self.x4t_reader.matrix_payload_bytes(
                        expert_id, matrix
                    )
                    dense = self.compressor.decompress(
                        {
                            "weight_packed": packed.packed,
                            "weight_scale": packed.scale,
                        },
                        self.scheme,
                    )["weight"].to(device=self.device)
                else:
                    dense, bytes_read = self._reconstruct_compressed(
                        expert_id=expert_id,
                        matrix=matrix,
                        expected=expected,
                    )
                    self.stats.packed_bytes += bytes_read
                if tuple(dense.shape) != expected:
                    raise ValueError(
                        f"layer {self.layer_idx} expert {expert_id} {matrix}: "
                        f"reconstructed shape {tuple(dense.shape)} != {expected}"
                    )
                self.stats.dense_bytes += _tensor_bytes(dense)
                assign_parameter(expert, f"{matrix}.weight", dense)
            self._loaded_experts.append(expert)
        self.stats.load_seconds += time.monotonic() - started


def _flatten_trace_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim >= 2 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


class LayerCaptures:
    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        capture_routed_post_situ: bool = False,
    ):
        self.values: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._capture_routed_post_situ = (
            capture_routed_post_situ and hasattr(layer, "block_sparse_moe")
        )
        self._post_situ_by_expert: dict[int, torch.Tensor] = {}
        self._capture_output(layer.self_attn, "attention_output")
        if hasattr(layer, "block_sparse_moe"):
            moe = layer.block_sparse_moe
            self._handles.append(
                moe.register_forward_pre_hook(self._capture_moe_input)
            )
            self._capture_output(moe, "moe_output")
            self._handles.append(
                moe.gate.register_forward_pre_hook(self._capture_router_logits)
            )
            self._handles.append(
                moe.gate.register_forward_hook(self._capture_routes)
            )
            self._capture_output(
                moe.routed_expert_down_proj, "routed_latent_input"
            )
            if hasattr(moe, "routed_expert_norm"):
                self._capture_output(
                    moe.routed_expert_norm, "routed_latent_normalized"
                )
            self._capture_output(
                moe.routed_expert_up_proj, "routed_output_partial"
            )
            if hasattr(moe, "shared_experts"):
                self._capture_output(
                    moe.shared_experts, "shared_output_partial"
                )
            if self._capture_routed_post_situ:
                for expert_id, expert in enumerate(moe.experts):
                    if not hasattr(expert, "act_fn"):
                        raise AttributeError(
                            f"expert {expert_id} has no post-SiTU act_fn module"
                        )
                    self._handles.append(
                        expert.act_fn.register_forward_hook(
                            lambda unused_module,
                            unused_args,
                            output,
                            expert_id=expert_id: self._capture_post_situ(
                                expert_id, output
                            )
                        )
                    )
        else:
            self._capture_output(layer.mlp, "dense_mlp_output")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._post_situ_by_expert.clear()

    def finalize(self) -> None:
        if not self._capture_routed_post_situ:
            return
        if "routed_post_situ" in self.values:
            return
        topk_ids = self.values.get("canonical_topk_ids")
        if topk_ids is None:
            raise RuntimeError("post-SiTU capture requires canonical route IDs")
        self.values["routed_post_situ"] = align_routed_post_situ(
            topk_ids,
            self._post_situ_by_expert,
        ).detach()

    def _capture_post_situ(
        self, expert_id: int, output: torch.Tensor
    ) -> None:
        if expert_id in self._post_situ_by_expert:
            raise RuntimeError(
                f"expert {expert_id} produced post-SiTU output more than once"
            )
        self._post_situ_by_expert[expert_id] = output.detach()

    def _capture_output(self, module: torch.nn.Module, stage: str) -> None:
        self._handles.append(
            module.register_forward_hook(
                lambda unused_module, unused_args, output, stage=stage: (
                    self.values.__setitem__(stage, output.detach())
                )
            )
        )

    def _capture_moe_input(
        self, unused_module: torch.nn.Module, args: tuple[Any, ...]
    ) -> None:
        self.values["mlp_input"] = args[0].detach()
        self.values["moe_input"] = args[0].detach()

    def _capture_router_logits(
        self, module: torch.nn.Module, args: tuple[Any, ...]
    ) -> None:
        hidden_states = args[0]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        self.values["router_logits"] = F.linear(
            hidden_states.float(), module.weight.float(), None
        ).detach()

    def _capture_routes(
        self,
        unused_module: torch.nn.Module,
        unused_args: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        topk_ids, topk_weights = output
        self.values["canonical_topk_ids"] = topk_ids.detach().to(torch.int32)
        self.values["canonical_topk_weights"] = (
            topk_weights.detach().float()
        )


def _write_layer_trace(
    trace_root: Path,
    *,
    layer_idx: int,
    layer_input: torch.Tensor,
    block_input: torch.Tensor,
    layer_output: torch.Tensor,
    block_output: torch.Tensor,
    captures: LayerCaptures,
) -> None:
    captures.finalize()
    values = {
        "layer_input": layer_input,
        "block_residual_input": block_input,
        **captures.values,
        "layer_boundary": layer_output,
        "block_residual": block_output,
    }
    for stage, value in values.items():
        write_trace_tensor(
            trace_root,
            layer=layer_idx,
            stage=stage,
            tensor=_flatten_trace_tensor(value),
        )


def _tokenize(
    args: argparse.Namespace, dependencies: dict[str, Any], checkpoint: Path
) -> tuple[Any, torch.Tensor, LoadedTeacherProxySuite | None]:
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        str(checkpoint), trust_remote_code=True
    )
    suite = None
    if args.input_ids_file is not None:
        suite = load_teacher_proxy_suite(args.input_ids_file)
        input_ids = suite.input_ids
    elif args.input_ids:
        ids = [int(value) for value in args.input_ids.split(",") if value]
        input_ids = torch.tensor([ids], dtype=torch.long)
    else:
        ids = tokenizer(
            args.prompt,
            add_special_tokens=args.add_special_tokens,
        )["input_ids"]
        input_ids = torch.tensor([ids], dtype=torch.long)
    if not input_ids.numel():
        raise ValueError("the fixed prefill must contain at least one token")
    if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
        raise ValueError("input IDs must form a nonempty rectangular batch")
    return tokenizer, input_ids, suite


def _initial_state(
    *,
    input_ids: torch.Tensor,
    store: IndexedSafetensors,
    hidden_size: int,
    device: torch.device,
) -> StreamState:
    embeddings = store.get_rows(
        f"{MODEL_TENSOR_PREFIX}.embed_tokens.weight", input_ids
    )
    embeddings = embeddings.view(
        input_ids.shape[0], input_ids.shape[1], hidden_size
    ).to(device=device)
    block_residual = embeddings.new_zeros(
        embeddings.shape[0] * embeddings.shape[1],
        0,
        embeddings.shape[2],
    )
    return StreamState(
        next_layer=0,
        input_ids=input_ids.to(device=device),
        hidden_states=embeddings,
        block_residual=block_residual,
    )


def _finalize(
    *,
    state: StreamState,
    store: IndexedSafetensors,
    config: Any,
    linear_module: Any,
    tokenizer: Any,
    device: torch.device,
    output_dir: Path,
    topk: int,
    lm_head_chunk_rows: int,
) -> dict[str, Any]:
    hidden_states = state.hidden_states
    batch, sequence, hidden_size = hidden_states.shape

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            output_norm = linear_module.KimiRMSNorm(
                hidden_size, eps=config.rms_norm_eps
            )
            output_proj = torch.nn.Linear(hidden_size, 1, bias=False)
            final_norm = linear_module.KimiRMSNorm(
                hidden_size, eps=config.rms_norm_eps
            )
    finally:
        torch.set_default_dtype(previous_dtype)

    assign_parameter(
        output_norm,
        "weight",
        store.get(
            f"{MODEL_TENSOR_PREFIX}.output_attn_res_norm.weight"
        ).to(device),
    )
    assign_parameter(
        output_proj,
        "weight",
        store.get(
            f"{MODEL_TENSOR_PREFIX}.output_attn_res_proj.weight"
        ).to(device),
    )
    assign_parameter(
        final_norm,
        "weight",
        store.get(f"{MODEL_TENSOR_PREFIX}.norm.weight").to(device),
    )
    hidden_states = linear_module._apply_attn_res(
        hidden_states.view(-1, hidden_size),
        state.block_residual,
        output_proj,
        output_norm,
    ).view(batch, sequence, hidden_size)
    hidden_states = final_norm(hidden_states)

    lm_head_name = "language_model.lm_head.weight"
    vocab_size = int(config.vocab_size)
    last_hidden = hidden_states[:, -1, :]
    pieces: list[torch.Tensor] = []
    for start in range(0, vocab_size, lm_head_chunk_rows):
        end = min(vocab_size, start + lm_head_chunk_rows)
        weight = store.get_range(lm_head_name, start, end).to(device)
        pieces.append(F.linear(last_hidden, weight).float().cpu())
        del weight
    logits = torch.cat(pieces, dim=-1)
    logprobs = torch.log_softmax(logits, dim=-1)
    values, ids = torch.topk(logprobs[0], k=topk)
    candidates = [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
            "decoded": tokenizer.decode([int(token_id)]),
            "logprob": float(logprob),
        }
        for rank, (token_id, logprob) in enumerate(
            zip(ids.tolist(), values.tolist()), start=1
        )
    ]
    atomic_torch_save(
        {
            "input_ids": state.input_ids.detach().cpu(),
            "final_hidden_states": hidden_states.detach().cpu(),
            "last_token_logits": logits,
            "last_token_logprobs": logprobs,
        },
        output_dir / "final.pt",
    )
    result = {
        "next_token_id": candidates[0]["token_id"],
        "next_token": candidates[0]["decoded"],
        "topk": candidates,
        "final_hidden_norm": float(hidden_states.float().norm()),
    }
    write_json(output_dir / "final.json", result)
    return result


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()


def _store_for(
    checkpoint: Path,
    *,
    existing_path: Path,
    existing_store: IndexedSafetensors,
) -> IndexedSafetensors:
    if _same_path(checkpoint, existing_path):
        return existing_store
    return IndexedSafetensors(checkpoint)


def _load_exl3_mcg_mult(
    expert_checkpoint: Path,
    manifest_path: Path | None,
) -> tuple[int | None, Path | None]:
    has_exl3 = any(
        name.endswith(".exl3_trellis")
        for name in IndexedSafetensors(expert_checkpoint).weight_map
    )
    if not has_exl3:
        return None, None

    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.append(manifest_path.expanduser())
    candidates.append(expert_checkpoint / "kquant_exl3_manifest.json")
    if expert_checkpoint.name.endswith("-serve"):
        candidates.append(
            expert_checkpoint.with_name(
                expert_checkpoint.name.removesuffix("-serve")
            )
            / "kquant_exl3_manifest.json"
        )
    resolved = next((path.resolve() for path in candidates if path.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(
            "EXL3 tensors require kquant_exl3_manifest.json; checked "
            + ", ".join(str(path) for path in candidates)
        )
    document = json.loads(resolved.read_text())
    if document.get("kind") != "kquant_exl3_artifact":
        raise ValueError(f"{resolved}: not a kquant EXL3 artifact manifest")
    mcg_mult = int(document["mcg_mult"])
    return mcg_mult, resolved


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    qsrt_artifact = (
        None
        if args.qsrt_artifact is None
        else QSRTArtifact.load(args.qsrt_artifact)
    )
    expert_checkpoint = (
        checkpoint
        if args.expert_checkpoint is None
        else args.expert_checkpoint.expanduser().resolve()
    )
    nonexpert_checkpoint = (
        checkpoint
        if args.nonexpert_checkpoint is None
        else args.nonexpert_checkpoint.expanduser().resolve()
    )
    expert_source_path = (
        expert_checkpoint
        if qsrt_artifact is None
        else qsrt_artifact.root
    )
    output_dir = args.output_dir.expanduser().resolve()
    indexed_sources = [checkpoint, nonexpert_checkpoint]
    if qsrt_artifact is None:
        indexed_sources.append(expert_checkpoint)
    for source in indexed_sources:
        if not (source / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(source / "model.safetensors.index.json")
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(checkpoint / "config.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run.json"
    if run_manifest_path.exists() and args.resume is None:
        raise FileExistsError(
            f"{output_dir} already contains run.json; use a fresh output "
            "directory or --resume"
        )

    dependencies = _require_reference_dependencies()
    outer_config, config, linear_module = _load_official_code(
        checkpoint, dependencies
    )
    checkpoint_store = IndexedSafetensors(checkpoint)
    if qsrt_artifact is None:
        expert_store: IndexedSafetensors | None = _store_for(
            expert_checkpoint,
            existing_path=checkpoint,
            existing_store=checkpoint_store,
        )
        nonexpert_store = _store_for(
            nonexpert_checkpoint,
            existing_path=(
                expert_checkpoint
                if expert_store is not checkpoint_store
                else checkpoint
            ),
            existing_store=expert_store,
        )
        mcg_mult, exl3_manifest = _load_exl3_mcg_mult(
            expert_checkpoint, args.exl3_manifest
        )
    else:
        expert_store = None
        nonexpert_store = _store_for(
            nonexpert_checkpoint,
            existing_path=checkpoint,
            existing_store=checkpoint_store,
        )
        mcg_mult, exl3_manifest = None, None
    tokenizer, input_ids, input_suite = _tokenize(
        args, dependencies, checkpoint
    )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("official KDA prefill requires a CUDA device")
    torch.cuda.set_device(device)

    if args.resume is not None:
        state, state_metadata = load_stream_state(args.resume)
        recorded_checkpoint = state_metadata.get("checkpoint")
        if (
            recorded_checkpoint is not None
            and Path(recorded_checkpoint).resolve() != checkpoint
        ):
            raise ValueError(
                f"resume checkpoint {recorded_checkpoint} != {checkpoint}"
            )
        if not torch.equal(state.input_ids.cpu(), input_ids):
            raise ValueError("resume input_ids differ from the requested prompt")
        state = StreamState(
            next_layer=state.next_layer,
            input_ids=state.input_ids.to(device),
            hidden_states=state.hidden_states.to(device),
            block_residual=state.block_residual.to(device),
        )
    else:
        state = _initial_state(
            input_ids=input_ids,
            store=nonexpert_store,
            hidden_size=int(config.hidden_size),
            device=device,
        )

    start_layer = state.next_layer
    if args.start_layer is not None and args.start_layer != start_layer:
        raise ValueError(
            f"--start-layer {args.start_layer} != state next_layer {start_layer}"
        )
    if input_suite is not None:
        suite_end_layer = int(input_suite.manifest["end_layer"])
        if args.end_layer is not None and args.end_layer != suite_end_layer:
            raise ValueError(
                f"--end-layer {args.end_layer} differs from suite "
                f"end_layer {suite_end_layer}"
            )
        end_layer = suite_end_layer
    else:
        end_layer = (
            int(config.num_hidden_layers)
            if args.end_layer is None
            else args.end_layer
        )
    if not 0 <= start_layer <= end_layer <= int(config.num_hidden_layers):
        raise ValueError(
            f"invalid layer range [{start_layer}, {end_layer}) for "
            f"{config.num_hidden_layers} layers"
        )
    if args.no_trace:
        trace_layers: tuple[int, ...] = ()
    elif args.trace_layers is not None:
        trace_layers = args.trace_layers
    elif input_suite is not None:
        trace_layers = tuple(input_suite.manifest["trace_layers"])
    else:
        trace_layers = tuple(range(start_layer, end_layer))
    if any(layer >= end_layer for layer in trace_layers) or (
        args.resume is None and any(layer < start_layer for layer in trace_layers)
    ):
        raise ValueError(
            f"trace layers {trace_layers} fall outside [{start_layer}, {end_layer})"
        )
    if input_suite is not None:
        expected_trace_layers = tuple(input_suite.manifest["trace_layers"])
        if trace_layers != expected_trace_layers:
            raise ValueError(
                f"trace layers {trace_layers} differ from suite "
                f"{expected_trace_layers}"
            )
        if args.no_trace:
            raise ValueError("teacher-proxy input suites require trace output")

    raw_quantization = config.quantization_config
    quantization_config = dependencies["QuantizationConfig"].model_validate(
        raw_quantization
    )
    schemes = list(quantization_config.config_groups.values())
    if len(schemes) != 1:
        raise ValueError(
            f"expected one MXFP4 quantization scheme, found {len(schemes)}"
        )
    scheme = schemes[0]
    compressor = dependencies["MXFP4PackedCompressor"]

    prompt_text = (
        None if args.input_ids or input_suite is not None else args.prompt
    )
    if qsrt_artifact is not None:
        expert_kind = "qsrt"
    else:
        expert_kind = "hybrid_exl3_mxfp4" if mcg_mult is not None else "mxfp4"
    nonexpert_kind = (
        "serialized_mxfp8_or_native"
        if not _same_path(nonexpert_checkpoint, checkpoint)
        else "official_native"
    )
    run_document: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "checkpoint": str(checkpoint),
        "expert_checkpoint": str(expert_source_path),
        "nonexpert_checkpoint": str(nonexpert_checkpoint),
        "qsrt_artifact": None if qsrt_artifact is None else str(qsrt_artifact.root),
        "qsrt_manifest": (
            None if qsrt_artifact is None else str(qsrt_artifact.manifest_path)
        ),
        "qsrt_codebook": None if qsrt_artifact is None else qsrt_artifact.codebook,
        "exl3_manifest": (
            None if exl3_manifest is None else str(exl3_manifest)
        ),
        "architecture": type(outer_config).__name__,
        "text_architecture": type(config).__name__,
        "reference": {
            "decoder": "official remote-code KimiDecoderLayer",
            "kda": "official fla-core chunk_kda",
            "mla": "official eager PyTorch attention",
            "expert_source": expert_kind,
            "nonexpert_source": nonexpert_kind,
            "mxfp4_decompression": (
                "compressed_tensors MXFP4PackedCompressor"
            ),
            "exl3_reconstruction": (
                "kquant QSRT balanced-atom unpack plus independent PyTorch "
                f"{qsrt_artifact.codebook}/Hadamard decode"
                if qsrt_artifact is not None
                else (
                    None
                    if mcg_mult is None
                    else "exllamav3 LinearEXL3.get_weight_tensor"
                )
            ),
            "mxfp8_reconstruction": (
                None
                if _same_path(nonexpert_checkpoint, checkpoint)
                else "FP32 E4M3 * E8M0-K32, rounded to module dtype"
            ),
            "excluded": ["vLLM", "serving GEMM/MoE kernels"],
            "transformers_output_recorder_compatibility_shim": dependencies[
                "compatibility_shim"
            ],
        },
        "prompt": prompt_text,
        "input_ids": input_ids.view(-1).tolist(),
        "input_ids_shape": list(input_ids.shape),
        "decoded_input": (
            tokenizer.decode(input_ids[0].tolist())
            if input_ids.shape[0] == 1
            else None
        ),
        "decoded_inputs": (
            None
            if input_suite is not None
            else [tokenizer.decode(row.tolist()) for row in input_ids]
        ),
        "input_suite": (
            None
            if input_suite is None
            else {
                "path": str(input_suite.path),
                "file_sha256": input_suite.file_sha256,
                "kind": input_suite.manifest["kind"],
                "schema_version": input_suite.manifest["schema_version"],
                "suite_token_hash_sha256": input_suite.manifest[
                    "suite_token_hash_sha256"
                ],
                "batch_shape": input_suite.manifest["batch_shape"],
                "document_hashes": [
                    document["document_hash"]
                    for document in input_suite.manifest["documents"]
                ],
            }
        ),
        "start_layer": start_layer,
        "end_layer": end_layer,
        "trace_layers": list(trace_layers),
        "chunk_size": args.chunk_size,
        "capture_routed_post_situ": args.capture_routed_post_situ,
        "device": str(device),
        "layers": [],
        "compatibility_fixes": [],
    }
    write_json(run_manifest_path, run_document)

    trace_root = output_dir / "trace"
    if not args.no_trace:
        write_trace_manifest(
            trace_root,
            layers=list(trace_layers),
            max_tokens=int(input_ids.numel()),
            checkpoint=str(expert_source_path),
        )

    cache_position = torch.arange(
        state.hidden_states.shape[1], dtype=torch.long, device=device
    )
    positions = cache_position.unsqueeze(0).expand(
        state.hidden_states.shape[0], -1
    )
    causal_mask = dependencies["create_causal_mask"](
        config=config,
        inputs_embeds=state.hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_ids=positions,
    )

    prefetcher = None
    if args.expert_prefetch_trace is not None:
        if expert_store is None:
            raise ValueError(
                "--expert-prefetch-trace is unavailable for QSRT artifacts"
            )
        prefetcher = ExpertRawPrefetcher(
            store=expert_store,
            trace_root=args.expert_prefetch_trace,
            end_layer=end_layer,
        )
        prefetcher.schedule_first(start_layer)

    started = time.monotonic()
    for layer_idx in range(start_layer, end_layer):
        layer_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        layer = _new_meta_layer(linear_module, config, layer_idx)
        fixes, nonexpert_stats = _load_nonexpert_parameters(
            layer,
            layer_idx=layer_idx,
            store=nonexpert_store,
            device=device,
        )
        run_document["compatibility_fixes"].extend(fixes)

        staged_experts = None
        if hasattr(layer, "block_sparse_moe"):
            if qsrt_artifact is None:
                assert expert_store is not None
                staged_experts = StagedExperts(
                    store=expert_store,
                    layer_idx=layer_idx,
                    scheme=scheme,
                    compressor=compressor,
                    device=device,
                    mcg_mult=mcg_mult,
                    exllamav3_dir=args.exllamav3_dir,
                    prefetcher=prefetcher,
                )
            else:
                reader = QSRTAtomLayerReader(
                    qsrt_artifact.layer_path(layer_idx)
                )
                x4t_reader = X4TLayerReader(
                    qsrt_artifact.x4t_layer_path(layer_idx)
                )
                try:
                    staged_experts = QSRTStagedExperts(
                        reader=reader,
                        x4t_reader=x4t_reader,
                        layer_idx=layer_idx,
                        scheme=scheme,
                        compressor=compressor,
                        device=device,
                        codebook=qsrt_artifact.codebook,
                    )
                except BaseException:
                    reader.close()
                    raise
            staged_experts.install(
                layer.block_sparse_moe.gate,
                layer.block_sparse_moe.experts,
            )

        should_trace = layer_idx in trace_layers
        captures = (
            LayerCaptures(
                layer,
                capture_routed_post_situ=args.capture_routed_post_situ,
            )
            if should_trace
            else None
        )
        layer_input = state.hidden_states
        block_input = state.block_residual
        layer_mask = None if layer.is_linear_attn else causal_mask
        layer_output, block_output = layer(
            layer_input,
            attention_mask=layer_mask,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            block_residual=block_input,
            cache_position=cache_position,
        )
        if not torch.isfinite(layer_output).all():
            raise FloatingPointError(
                f"layer {layer_idx} produced non-finite hidden states"
            )

        if should_trace:
            assert captures is not None
            _write_layer_trace(
                trace_root,
                layer_idx=layer_idx,
                layer_input=layer_input,
                block_input=block_input,
                layer_output=layer_output,
                block_output=block_output,
                captures=captures,
            )

        expert_stats = (
            staged_experts.stats
            if staged_experts is not None
            else ExpertLoadStats()
        )
        if captures is not None:
            captures.close()
        if staged_experts is not None:
            staged_experts.close()
        state = StreamState(
            next_layer=layer_idx + 1,
            input_ids=state.input_ids,
            hidden_states=layer_output,
            block_residual=block_output,
        )
        torch.cuda.synchronize(device)
        layer_record = {
            "layer": layer_idx,
            "attention": "kda" if layer.is_linear_attn else "mla",
            "moe": hasattr(layer, "block_sparse_moe"),
            "traced": should_trace,
            "seconds": time.monotonic() - layer_started,
            "nonexpert_serialized_bytes_read": (
                nonexpert_stats.serialized_bytes
            ),
            "nonexpert_dense_bytes_materialized": (
                nonexpert_stats.dense_bytes
            ),
            "nonexpert_load_seconds": nonexpert_stats.load_seconds,
            "nonexpert_mxfp8_parameters": (
                nonexpert_stats.mxfp8_parameters
            ),
            "routed_experts": expert_stats.expert_ids,
            "num_routed_experts": len(expert_stats.expert_ids),
            "exl3_experts": expert_stats.exl3_expert_ids,
            "num_exl3_experts": len(expert_stats.exl3_expert_ids),
            "kept_mxfp4_experts": expert_stats.kept_expert_ids,
            "num_kept_mxfp4_experts": len(expert_stats.kept_expert_ids),
            "expert_packed_bytes_read": expert_stats.packed_bytes,
            "expert_dense_bytes_materialized": expert_stats.dense_bytes,
            "expert_load_seconds": expert_stats.load_seconds,
            "expert_prefetched": expert_stats.prefetched_expert_ids,
            "expert_prefetch_hits": expert_stats.prefetch_hit_expert_ids,
            "expert_prefetch_misses": expert_stats.prefetch_miss_expert_ids,
            "hidden_norm": float(layer_output.float().norm()),
            "block_residual_shape": list(block_output.shape),
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        run_document["layers"].append(layer_record)
        print(
            f"layer {layer_idx:02d} "
            f"{layer_record['attention']} "
            f"{'moe' if layer_record['moe'] else 'dense'}: "
            f"{layer_record['seconds']:.2f}s, "
            f"{layer_record['num_routed_experts']} experts, "
            f"{layer_record['num_exl3_experts']} EXL3/"
            f"{layer_record['num_kept_mxfp4_experts']} kept, "
            f"norm={layer_record['hidden_norm']:.6g}, "
            f"peak={layer_record['peak_gpu_bytes'] / 2**30:.2f} GiB",
            flush=True,
        )

        del layer, layer_input, block_input, layer_output, block_output
        torch.cuda.empty_cache()

        if (
            state.next_layer % args.chunk_size == 0
            or state.next_layer == end_layer
        ):
            state_path = output_dir / f"state-layer-{state.next_layer:03d}.pt"
            save_stream_state(
                state_path,
                state,
                checkpoint=str(checkpoint),
                prompt=prompt_text,
            )
            run_document["latest_state"] = str(state_path)
            write_json(run_manifest_path, run_document)

    final_result = None
    if args.finalize:
        if state.next_layer != int(config.num_hidden_layers):
            raise ValueError(
                "--finalize requires the state after every decoder layer"
            )
        final_result = _finalize(
            state=state,
            store=nonexpert_store,
            config=config,
            linear_module=linear_module,
            tokenizer=tokenizer,
            device=device,
            output_dir=output_dir,
            topk=args.topk,
            lm_head_chunk_rows=args.lm_head_chunk_rows,
        )

    if prefetcher is not None:
        prefetcher.close()

    run_document.update(
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "next_layer": state.next_layer,
            "final": final_result,
        }
    )
    write_json(run_manifest_path, run_document)
    return run_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    expert_group = parser.add_mutually_exclusive_group()
    expert_group.add_argument(
        "--expert-checkpoint",
        type=Path,
        help=(
            "Optional indexed MXFP4 or hybrid EXL3/MXFP4 expert source; "
            "the official --checkpoint remains the model-code source"
        ),
    )
    expert_group.add_argument(
        "--qsrt-artifact",
        type=Path,
        help=(
            "Completed QSRT atom/X4T artifact used as the expert "
            "source; the official --checkpoint remains the model-code source"
        ),
    )
    parser.add_argument(
        "--nonexpert-checkpoint",
        type=Path,
        help=(
            "Optional indexed non-expert source. E4M3 weights with adjacent "
            "E8M0-K32 weight_scale tensors are reconstructed to BF16."
        ),
    )
    parser.add_argument(
        "--exl3-manifest",
        type=Path,
        help=(
            "kquant_exl3_manifest.json; inferred from an artifact or its "
            "*-serve sibling when omitted"
        ),
    )
    parser.add_argument(
        "--exllamav3-dir",
        type=Path,
        default=DEFAULT_EXLLAMAV3_DIR,
    )
    parser.add_argument(
        "--expert-prefetch-trace",
        type=Path,
        help=(
            "Optional reference trace whose routed expert IDs are used only "
            "to prefetch serialized tensors; actual routing remains authoritative"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-ids",
        help="comma-separated IDs; overrides --prompt for model input",
    )
    input_group.add_argument(
        "--input-ids-file",
        type=Path,
        help=(
            "validated kquant teacher-proxy suite JSON; provides a rectangular "
            "batch and overrides --prompt"
        ),
    )
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-layer", type=int)
    parser.add_argument("--end-layer", type=int)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument(
        "--trace-layers",
        type=parse_layer_list,
        help="comma-separated decoder layers to trace while executing the full range",
    )
    parser.add_argument(
        "--capture-routed-post-situ",
        action="store_true",
        help=(
            "also trace route-aligned [token, top_k, 3072] expert SiTU "
            "outputs; opt-in because storage grows with tokens * top_k"
        ),
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.lm_head_chunk_rows <= 0:
        parser.error("--lm-head-chunk-rows must be positive")
    if args.no_trace and args.capture_routed_post_situ:
        parser.error("--capture-routed-post-situ requires trace output")
    if args.qsrt_artifact is not None and args.exl3_manifest is not None:
        parser.error("--exl3-manifest does not apply to --qsrt-artifact")
    if (
        args.qsrt_artifact is not None
        and args.expert_prefetch_trace is not None
    ):
        parser.error(
            "--expert-prefetch-trace is unavailable for QSRT artifacts"
        )
    return args


if __name__ == "__main__":
    run(parse_args())
