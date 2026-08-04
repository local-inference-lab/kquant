"""Materialize selected TP12 QSRT candidates into fixed layer slabs.

The candidate pool is deliberately allocation-independent.  This module joins
one validated allocation with those candidate bytes and the exact X4T tier. It
never instantiates or dequantizes the official model: candidate tensors and
official matrices are consumed one at a time and placed directly into the
trellis slab or X4T sidecar.
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.exl3_reference import CODEBOOK_SQG_NORMAL_E4M3, QSRT_CODEBOOKS
from kquant.qsrt import (
    EXPERTS_PER_LAYER,
    EXPERT_RANK_MXFP4_BYTES,
    FORMAT_SECTION_BYTES,
    FORMAT_TABLE_BYTES,
    INTERMEDIATE_CHANNELS,
    KEEP_STORAGE_EXTERNAL_X4T,
    LATENT_CHANNELS,
    EXPERT_RANK_TRELLIS_BYTES,
    EXPERT_TRELLIS_BYTES,
    LAYER_HEADER_BYTES,
    LOCAL_SCALE_VECTOR_BYTES,
    MATRIX_TRELLIS_BYTES,
    PAIR_BYTES,
    SCALE_BYTES,
    SHARED_SCALE_SECTION_BYTES,
    TP_SIZE,
    ExpertFormatSpec,
    TP12LayerHeader,
    TP12LayerLayout,
    pack_tp12_format_section,
    pack_tp12_shared_scale_section,
    tp12_logical_pair_index,
    unpack_tp12_format_section,
    unpack_tp12_rank_scale_section,
    unpack_tp12_shared_scale_section,
)
from kquant.pack.qsrt_candidates import candidate_tensor_name
from kquant.source_weights import OfficialMXFP4Store, PackedMXFP4Matrix
from kquant.x4t import X4TLayerReader, X4TLayerWriter


QSRT_LAYER_PREFIX = "qsrt-tp12-layer-"
MXFP4_MATRIX_COMPONENT_ORDER = ("weight_packed", "weight_scale")


@dataclass(frozen=True)
class LayerMaterializationSpec:
    """Validated, canonical materialization decision for one decoder layer."""

    layer: int
    formats: tuple[ExpertFormatSpec, ...]
    compressed: tuple[int, ...]
    kept: tuple[int, ...]
    layout: TP12LayerLayout
    codebook: str = CODEBOOK_SQG_NORMAL_E4M3

    def __post_init__(self) -> None:
        if self.layer not in C.MOE_LAYERS:
            raise ValueError("materialized layer must be a K3 MoE layer")
        if self.codebook not in QSRT_CODEBOOKS:
            raise ValueError("materialized layer codebook is unsupported")
        if len(self.formats) != EXPERTS_PER_LAYER:
            raise ValueError(
                f"materialized layer must describe {EXPERTS_PER_LAYER} experts"
            )
        universe = tuple(range(EXPERTS_PER_LAYER))
        if (
            self.compressed != tuple(sorted(self.compressed))
            or self.kept != tuple(sorted(self.kept))
            or len(set(self.compressed)) != len(self.compressed)
            or len(set(self.kept)) != len(self.kept)
            or set(self.compressed).intersection(self.kept)
            or tuple(sorted(self.compressed + self.kept)) != universe
        ):
            raise ValueError("materialized layer tiers must be a canonical partition")
        expected_compressed = tuple(
            expert
            for expert, format_spec in enumerate(self.formats)
            if not format_spec.is_mxfp4
        )
        expected_kept = tuple(
            expert
            for expert, format_spec in enumerate(self.formats)
            if format_spec.is_mxfp4
        )
        if self.compressed != expected_compressed or self.kept != expected_kept:
            raise ValueError("materialized layer tiers disagree with its format table")
        if self.layout != TP12LayerLayout.from_formats(self.formats):
            raise ValueError("materialized layer layout disagrees with its format table")

    @property
    def format_codes(self) -> tuple[int, ...]:
        return tuple(value.code for value in self.formats)


@dataclass(frozen=True)
class TP12CompressedRankPayload:
    """One rank's compressed tier in B12X's public pair-decoder contract.

    The disk slab is expert-major with ``w1, w3, w2`` pairs.  B12X consumes
    projection-major FC1 pairs, a separate FC2 plane, and local rotations in
    ``gate_svh, up_svh, down_suh`` order.  Constructing this value is the only
    format-to-runtime transpose required by the reference loader.
    """

    rank: int
    expert_ids: torch.Tensor
    w13_trellis: torch.Tensor
    w2_trellis: torch.Tensor
    gate_suh: torch.Tensor
    up_suh: torch.Tensor
    intermediate_rotations: torch.Tensor
    down_svh: torch.Tensor
    fc1_pair_modes: torch.Tensor
    fc2_pair_modes: torch.Tensor

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= self.rank < TP_SIZE:
            raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
        count = int(self.expert_ids.numel())
        words = PAIR_BYTES // torch.int16.itemsize
        expected = (
            ("expert_ids", self.expert_ids, torch.int32, (count,)),
            ("w13_trellis", self.w13_trellis, torch.int16, (2, count, words)),
            ("w2_trellis", self.w2_trellis, torch.int16, (count, words)),
            ("gate_suh", self.gate_suh, torch.float16, (1, LATENT_CHANNELS)),
            ("up_suh", self.up_suh, torch.float16, (1, LATENT_CHANNELS)),
            (
                "intermediate_rotations",
                self.intermediate_rotations,
                torch.float16,
                (count, 3 * (INTERMEDIATE_CHANNELS // TP_SIZE)),
            ),
            ("down_svh", self.down_svh, torch.float16, (1, LATENT_CHANNELS)),
            ("fc1_pair_modes", self.fc1_pair_modes, torch.int32, (count,)),
            ("fc2_pair_modes", self.fc2_pair_modes, torch.int32, (count,)),
        )
        device = self.expert_ids.device
        for name, value, dtype, shape in expected:
            if value.dtype != dtype or tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must be contiguous {dtype} {shape}, got "
                    f"{value.dtype} {tuple(value.shape)}"
                )
            if value.device != device:
                raise ValueError(f"{name} must be on {device}, got {value.device}")
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if count:
            ids = self.expert_ids.to(device="cpu", dtype=torch.int64).tolist()
            if ids != sorted(ids) or len(set(ids)) != count:
                raise ValueError("expert_ids must be strictly increasing")
            if ids[0] < 0 or ids[-1] >= EXPERTS_PER_LAYER:
                raise ValueError("expert_ids contain an out-of-range expert")
        for name, modes in (
            ("fc1_pair_modes", self.fc1_pair_modes),
            ("fc2_pair_modes", self.fc2_pair_modes),
        ):
            if not bool(torch.all((modes == 0) | (modes == 1))):
                raise ValueError(f"{name} values must be 0=P33 or 1=P24")
        for name, scale in (
            ("gate_suh", self.gate_suh),
            ("up_suh", self.up_suh),
            ("intermediate_rotations", self.intermediate_rotations),
            ("down_svh", self.down_svh),
        ):
            if not bool(torch.all(torch.isfinite(scale))):
                raise ValueError(f"{name} contains non-finite values")

    @property
    def num_experts(self) -> int:
        return int(self.expert_ids.numel())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "TP12CompressedRankPayload":
        """Move the complete runtime payload to one device."""

        target = torch.device(device)
        values = {
            name: getattr(self, name).to(target, non_blocking=non_blocking)
            for name in (
                "expert_ids",
                "w13_trellis",
                "w2_trellis",
                "gate_suh",
                "up_suh",
                "intermediate_rotations",
                "down_svh",
                "fc1_pair_modes",
                "fc2_pair_modes",
            )
        }
        return TP12CompressedRankPayload(rank=self.rank, **values)

    def b12x_prepare_kwargs(self) -> dict[str, torch.Tensor]:
        """Arguments consumed by ``sparkinfer.moe.fused_moe.prepare_weights``."""

        return {
            "w1_fp4": self.w13_trellis,
            "w2_fp4": self.w2_trellis,
            "gate_suh": self.gate_suh,
            "up_suh": self.up_suh,
            "intermediate_rotations": self.intermediate_rotations,
            "down_svh": self.down_svh,
            "trellis_fc1_pair_modes": self.fc1_pair_modes,
            "trellis_fc2_pair_modes": self.fc2_pair_modes,
        }


def _validate_candidate_tensor(
    tensor: torch.Tensor,
    *,
    matrix: str,
    part: str,
) -> torch.Tensor:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        tensor = tensor.detach().cpu().contiguous()
    if part == "trellis":
        expected_dtype = torch.int16
        expected_values = MATRIX_TRELLIS_BYTES // torch.int16.itemsize
    else:
        expected_dtype = torch.float16
        if (matrix in ("w1", "w3") and part == "suh") or (
            matrix == "w2" and part == "svh"
        ):
            expected_values = LATENT_CHANNELS
        else:
            expected_values = INTERMEDIATE_CHANNELS
    if tensor.dtype != expected_dtype or tensor.ndim != 1 or tensor.numel() != expected_values:
        raise ValueError(
            f"{matrix}.{part} has {tensor.dtype} {tuple(tensor.shape)}, expected "
            f"{expected_dtype} ({expected_values},)"
        )
    if part != "trellis" and not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{matrix}.{part} contains non-finite scales")
    return tensor


def candidate_trellis_pair(
    trellis: torch.Tensor,
    logical_pair: int,
) -> torch.Tensor:
    """Return one logical P24/P33 pair from a candidate matrix payload."""

    trellis = _validate_candidate_tensor(trellis, matrix="w1", part="trellis")
    if (
        isinstance(logical_pair, bool)
        or not isinstance(logical_pair, int)
        or not 0 <= logical_pair < TP_SIZE
    ):
        raise ValueError(f"logical_pair must be in 0..{TP_SIZE - 1}")
    words = PAIR_BYTES // torch.int16.itemsize
    result = trellis.narrow(0, logical_pair * words, words)
    if not result.is_contiguous():
        result = result.contiguous()
    return result


def candidate_local_scale(
    scale: torch.Tensor,
    logical_pair: int,
) -> torch.Tensor:
    """Return one logical 256-neuron slice of a candidate scale vector."""

    if scale.dtype != torch.float16 or scale.ndim != 1 or scale.numel() != INTERMEDIATE_CHANNELS:
        raise ValueError(
            f"local scale must be torch.float16 ({INTERMEDIATE_CHANNELS},)"
        )
    if not bool(torch.all(torch.isfinite(scale))):
        raise ValueError("local scale contains non-finite values")
    if (
        isinstance(logical_pair, bool)
        or not isinstance(logical_pair, int)
        or not 0 <= logical_pair < TP_SIZE
    ):
        raise ValueError(f"logical_pair must be in 0..{TP_SIZE - 1}")
    width = INTERMEDIATE_CHANNELS // TP_SIZE
    result = scale.narrow(0, logical_pair * width, width)
    if not result.is_contiguous():
        result = result.contiguous()
    if result.numel() * result.element_size() != LOCAL_SCALE_VECTOR_BYTES:
        raise AssertionError("local scale byte accounting drifted")
    return result


def _byte_view(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu":
        raise ValueError("only CPU tensors can be written to an artifact")
    if not tensor.is_contiguous():
        raise ValueError("artifact tensors must be contiguous")
    return memoryview(tensor.numpy()).cast("B")


def pwrite_exact(descriptor: int, payload: bytes | bytearray | memoryview, offset: int) -> None:
    """Write a complete byte buffer at an explicit offset, handling short writes."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    view = memoryview(payload).cast("B")
    cursor = 0
    while cursor < len(view):
        written = os.pwrite(descriptor, view[cursor:], offset + cursor)
        if written <= 0:
            raise OSError("pwrite made no forward progress")
        cursor += written


def _pread_exact(descriptor: int, count: int, offset: int) -> bytes:
    result = bytearray()
    while len(result) < count:
        piece = os.pread(descriptor, count - len(result), offset + len(result))
        if not piece:
            raise ValueError("artifact ended before the expected offset")
        result.extend(piece)
    return bytes(result)


def _discard_stale_partial(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"materialization partial is still owned by a live writer: {path}"
            ) from exc
        path.unlink()
    finally:
        os.close(descriptor)


def _tensor_from_bytes(
    payload: bytes | bytearray,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape).clone()
    if not value.is_contiguous():
        value = value.contiguous()
    return value


class TP12LayerReader:
    """Bounded-memory reference reader for one materialized TP12 layer slab."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            self.header = TP12LayerHeader.from_bytes(
                _pread_exact(descriptor, LAYER_HEADER_BYTES, 0)
            )
            if self.path.stat().st_size != self.header.layout.disk_bytes:
                raise ValueError("materialized layer size disagrees with its header")
            format_section = torch.frombuffer(
                bytearray(
                    _pread_exact(
                        descriptor,
                        FORMAT_SECTION_BYTES,
                        self.header.format_offset,
                    )
                ),
                dtype=torch.uint8,
            )
            unpack_tp12_format_section(format_section)
            self.formats = tuple(
                ExpertFormatSpec.from_code(code)
                for code in format_section[:FORMAT_TABLE_BYTES].tolist()
            )
            if TP12LayerLayout.from_formats(self.formats) != self.header.layout:
                raise ValueError("format table and layer-header tier counts disagree")
            slots: list[int] = []
            compressed_slot = 0
            kept_slot = 0
            for format_spec in self.formats:
                if format_spec.is_mxfp4:
                    slots.append(kept_slot)
                    kept_slot += 1
                else:
                    slots.append(compressed_slot)
                    compressed_slot += 1
            self.slots = tuple(slots)
            self.compressed_experts = tuple(
                expert
                for expert, format_spec in enumerate(self.formats)
                if not format_spec.is_mxfp4
            )
            self.kept_experts = tuple(
                expert
                for expert, format_spec in enumerate(self.formats)
                if format_spec.is_mxfp4
            )
            shared_section = torch.frombuffer(
                bytearray(
                    _pread_exact(
                        descriptor,
                        SHARED_SCALE_SECTION_BYTES,
                        self.header.shared_scale_offset,
                    )
                ),
                dtype=torch.uint8,
            )
            self.shared_scales = unpack_tp12_shared_scale_section(shared_section)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor: int | None = descriptor

    @property
    def layout(self) -> TP12LayerLayout:
        return self.header.layout

    def close(self) -> None:
        if self._descriptor is not None:
            descriptor = self._descriptor
            self._descriptor = None
            os.close(descriptor)

    def __enter__(self) -> "TP12LayerReader":
        if self._descriptor is None:
            raise RuntimeError("materialized TP12 layer reader is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def _pread(self, count: int, offset: int) -> bytes:
        if self._descriptor is None:
            raise RuntimeError("materialized TP12 layer reader is closed")
        return _pread_exact(self._descriptor, count, offset)

    def _slot(self, expert: int, *, kept: bool) -> int:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError("expert must be an integer")
        if not 0 <= expert < EXPERTS_PER_LAYER:
            raise ValueError(f"expert must be in 0..{EXPERTS_PER_LAYER - 1}")
        actual_kept = self.formats[expert].is_mxfp4
        if actual_kept != kept:
            expected = "MXFP4" if kept else "compressed"
            raise ValueError(f"expert {expert} is not in the {expected} tier")
        return self.slots[expert]

    def read_compressed_rank(self, rank: int) -> TP12CompressedRankPayload:
        """Read one rank directly into the bounded B12X pair-decoder layout."""

        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 0 <= rank < TP_SIZE:
            raise ValueError(f"rank must be in 0..{TP_SIZE - 1}")
        count = self.layout.compressed_experts
        words = PAIR_BYTES // torch.int16.itemsize
        w13 = torch.empty((2, count, words), dtype=torch.int16)
        w2 = torch.empty((count, words), dtype=torch.int16)
        rank_offset = self.layout.rank_offset(rank)
        for slot in range(count):
            raw = _tensor_from_bytes(
                self._pread(
                    EXPERT_RANK_TRELLIS_BYTES,
                    rank_offset + slot * EXPERT_RANK_TRELLIS_BYTES,
                ),
                dtype=torch.int16,
                shape=(3, words),
            )
            w13[:, slot].copy_(raw[:2])
            w2[slot].copy_(raw[2])

        scale_section = torch.frombuffer(
            bytearray(
                self._pread(
                    self.layout.rank_scale_section_bytes,
                    self.layout.rank_scale_offset(rank),
                )
            ),
            dtype=torch.uint8,
        )
        local_scales = unpack_tp12_rank_scale_section(scale_section, self.layout)
        intermediate_rotations = local_scales.reshape(count, -1).contiguous()
        expert_ids = torch.tensor(self.compressed_experts, dtype=torch.int32)
        r13 = torch.tensor(
            [self.formats[expert].r13 for expert in self.compressed_experts],
            dtype=torch.int32,
        )
        r2 = torch.tensor(
            [self.formats[expert].r2 for expert in self.compressed_experts],
            dtype=torch.int32,
        )
        logical_pairs = torch.tensor(
            [
                tp12_logical_pair_index(self.header.layer, expert, rank)
                for expert in self.compressed_experts
            ],
            dtype=torch.int32,
        )
        return TP12CompressedRankPayload(
            rank=rank,
            expert_ids=expert_ids,
            w13_trellis=w13,
            w2_trellis=w2,
            gate_suh=self.shared_scales[0].reshape(1, -1).clone(),
            up_suh=self.shared_scales[1].reshape(1, -1).clone(),
            intermediate_rotations=intermediate_rotations,
            down_svh=self.shared_scales[2].reshape(1, -1).clone(),
            fc1_pair_modes=(r13 > logical_pairs).to(torch.int32).contiguous(),
            fc2_pair_modes=(r2 > logical_pairs).to(torch.int32).contiguous(),
        )

    def read_compressed_matrix(
        self, expert: int, matrix: str
    ) -> dict[str, torch.Tensor]:
        """Reassemble one logical candidate payload from its twelve rank slices."""

        if matrix not in C.EXPERT_MATRICES:
            raise ValueError(f"unknown expert matrix: {matrix}")
        slot = self._slot(expert, kept=False)
        trellis_raw = bytearray(MATRIX_TRELLIS_BYTES)
        local_raw = bytearray(INTERMEDIATE_CHANNELS * torch.float16.itemsize)
        for rank in range(TP_SIZE):
            logical_pair = tp12_logical_pair_index(
                self.header.layer,
                expert,
                rank,
            )
            pair_begin = logical_pair * PAIR_BYTES
            trellis_raw[pair_begin : pair_begin + PAIR_BYTES] = self._pread(
                PAIR_BYTES,
                self.layout.trellis_pair_offset(rank, slot, matrix),
            )
            local_begin = logical_pair * LOCAL_SCALE_VECTOR_BYTES
            local_raw[local_begin : local_begin + LOCAL_SCALE_VECTOR_BYTES] = self._pread(
                LOCAL_SCALE_VECTOR_BYTES,
                self.layout.local_scale_offset(rank, slot, matrix),
            )
        trellis = _tensor_from_bytes(
            trellis_raw,
            dtype=torch.int16,
            shape=(MATRIX_TRELLIS_BYTES // torch.int16.itemsize,),
        )
        local = _tensor_from_bytes(
            local_raw,
            dtype=torch.float16,
            shape=(INTERMEDIATE_CHANNELS,),
        )
        shared_index = {"w1": 0, "w3": 1, "w2": 2}[matrix]
        shared = self.shared_scales[shared_index].clone()
        if matrix in ("w1", "w3"):
            parts = {"trellis": trellis, "suh": shared, "svh": local}
        else:
            parts = {"trellis": trellis, "suh": local, "svh": shared}
        for part, value in parts.items():
            _validate_candidate_tensor(value, matrix=matrix, part=part)
        return parts

def layer_filename(layer: int) -> str:
    if layer not in C.MOE_LAYERS:
        raise ValueError("K3 MoE layer must be in 1..92")
    return f"{QSRT_LAYER_PREFIX}{layer:05d}.bin"


def expected_logical_source_bytes(spec: LayerMaterializationSpec) -> int:
    """Bytes compared while proving one slab against both source tiers."""

    compressed_matrix_bytes = (
        EXPERT_TRELLIS_BYTES
        + 3 * (LATENT_CHANNELS + INTERMEDIATE_CHANNELS) * SCALE_BYTES
    )
    kept_expert_bytes = TP_SIZE * EXPERT_RANK_MXFP4_BYTES
    return (
        len(spec.compressed) * compressed_matrix_bytes
        + len(spec.kept) * kept_expert_bytes
    )


def candidate_layer_path(root: Path, layer: int) -> Path:
    return root / "candidates" / f"qsrt-layer-{layer:05d}.safetensors"


def validate_materialized_layer(
    path: str | Path,
    spec: LayerMaterializationSpec,
) -> dict[str, int | str]:
    """Validate fixed sections, exact size, and every alignment padding range."""

    path = Path(path)
    stat = path.stat()
    if stat.st_size != spec.layout.disk_bytes:
        raise ValueError(
            f"{path} has {stat.st_size} bytes, expected {spec.layout.disk_bytes}"
        )
    descriptor = os.open(path, os.O_RDONLY)
    try:
        header = TP12LayerHeader.from_bytes(
            _pread_exact(descriptor, LAYER_HEADER_BYTES, 0)
        )
        if (
            header.layer != spec.layer
            or header.layout != spec.layout
            or header.codebook != spec.codebook
        ):
            raise ValueError("materialized layer header disagrees with its allocation")
        format_section = torch.frombuffer(
            bytearray(
                _pread_exact(
                    descriptor,
                    FORMAT_SECTION_BYTES,
                    header.format_offset,
                )
            ),
            dtype=torch.uint8,
        )
        if unpack_tp12_format_section(format_section) != tuple(
            value.name for value in spec.formats
        ):
            raise ValueError("materialized format table disagrees with its allocation")
        shared_section = torch.frombuffer(
            bytearray(
                _pread_exact(
                    descriptor,
                    SHARED_SCALE_SECTION_BYTES,
                    header.shared_scale_offset,
                )
            ),
            dtype=torch.uint8,
        )
        unpack_tp12_shared_scale_section(shared_section)
        for rank in range(TP_SIZE):
            padding = spec.layout.rank_scale_padding_bytes
            if not padding:
                continue
            padding_offset = (
                spec.layout.rank_scale_offset(rank)
                + spec.layout.rank_scale_payload_bytes
            )
            if any(_pread_exact(descriptor, padding, padding_offset)):
                raise ValueError(f"rank {rank} has nonzero local-scale padding")
    finally:
        os.close(descriptor)
    return {
        "file": path.name,
        "disk_bytes": spec.layout.disk_bytes,
        "compressed_experts": len(spec.compressed),
        "kept_experts": len(spec.kept),
        "codebook": spec.codebook,
    }


def _requested_tier(
    requested: Sequence[int] | None,
    available: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    if requested is None:
        return available
    values = tuple(requested)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError(f"{name} expert IDs must be integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} expert IDs must be unique")
    if not set(values).issubset(available):
        raise ValueError(f"{name} payload closure requested an expert outside its tier")
    return values


def _require_bit_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    if torch.equal(actual, expected):
        return
    if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
        raise ValueError(
            f"{name} geometry changed in the materialized artifact: "
            f"{actual.dtype} {tuple(actual.shape)} != "
            f"{expected.dtype} {tuple(expected.shape)}"
        )
    mismatched = int(torch.count_nonzero(actual != expected))
    raise ValueError(
        f"{name} is not bit-exact in the materialized artifact "
        f"({mismatched}/{actual.numel()} values differ)"
    )


def validate_materialized_layer_payloads(
    path: str | Path,
    spec: LayerMaterializationSpec,
    candidate_root: str | Path,
    store: OfficialMXFP4Store,
    *,
    compressed_experts: Sequence[int] | None = None,
    kept_experts: Sequence[int] | None = None,
    x4t_path: str | Path | None = None,
) -> dict[str, int | str]:
    """Prove candidate and official-source bytes survived slab placement exactly."""

    metadata = validate_materialized_layer(path, spec)
    compressed = _requested_tier(
        compressed_experts, spec.compressed, name="compressed"
    )
    kept = _requested_tier(kept_experts, spec.kept, name="kept")
    logical_bytes = 0
    x4t_reader: X4TLayerReader | None = None
    if x4t_path is None:
        raise ValueError("QSRT closure requires its X4T layer sidecar")
    x4t_reader = X4TLayerReader(x4t_path)
    if x4t_reader.layer != spec.layer:
        raise ValueError("X4T sidecar layer disagrees with its trellis slab")
    if x4t_reader.record_count != len(spec.kept) * len(C.EXPERT_MATRICES):
        raise ValueError("X4T sidecar record count disagrees with its allocation")
    with TP12LayerReader(path) as reader:
        if reader.header.layer != spec.layer or reader.formats != spec.formats:
            raise ValueError("materialized reader disagrees with the allocation spec")
        if compressed:
            candidate_path = candidate_layer_path(Path(candidate_root), spec.layer)
            with safe_open(candidate_path, framework="pt", device="cpu") as handle:
                for expert in compressed:
                    for matrix in C.EXPERT_MATRICES:
                        actual = reader.read_compressed_matrix(expert, matrix)
                        for part in ("trellis", "suh", "svh"):
                            name = candidate_tensor_name(
                                spec.layer, expert, matrix, part
                            )
                            try:
                                expected = _validate_candidate_tensor(
                                    handle.get_tensor(name),
                                    matrix=matrix,
                                    part=part,
                                )
                            except KeyError as exc:
                                raise ValueError(
                                    f"candidate layer {spec.layer} expert {expert} "
                                    f"is missing {matrix}.{part}"
                                ) from exc
                            _require_bit_exact(
                                actual[part],
                                expected,
                                name=(
                                    f"layer {spec.layer} expert {expert} "
                                    f"{matrix}.{part}"
                                ),
                            )
                            logical_bytes += expected.numel() * expected.element_size()
        for expert in kept:
            for matrix in C.EXPERT_MATRICES:
                x4t_record = x4t_reader.read(expert, matrix)
                actual = PackedMXFP4Matrix(
                    packed=x4t_record.packed,
                    scale=x4t_record.scale,
                )
                expected = store.load_packed_matrix(spec.layer, expert, matrix)
                for part in MXFP4_MATRIX_COMPONENT_ORDER:
                    actual_part = (
                        actual.packed if part == "weight_packed" else actual.scale
                    )
                    expected_part = (
                        expected.packed if part == "weight_packed" else expected.scale
                    )
                    _require_bit_exact(
                        actual_part,
                        expected_part,
                        name=f"layer {spec.layer} expert {expert} {matrix}.{part}",
                    )
                    logical_bytes += expected_part.numel() * expected_part.element_size()
    return {
        **metadata,
        "payload_closure": "bit_exact",
        "compressed_experts_verified": len(compressed),
        "kept_experts_verified": len(kept),
        "logical_source_bytes_verified": logical_bytes,
    }


def _write_candidate_expert(
    descriptor: int,
    handle,
    spec: LayerMaterializationSpec,
    *,
    expert: int,
    compressed_slot: int,
    shared_reference: dict[str, torch.Tensor],
) -> None:
    shared_part = {"w1": "suh", "w3": "suh", "w2": "svh"}
    local_part = {"w1": "svh", "w3": "svh", "w2": "suh"}
    for matrix in C.EXPERT_MATRICES:
        names = {
            part: candidate_tensor_name(spec.layer, expert, matrix, part)
            for part in ("trellis", "suh", "svh")
        }
        try:
            tensors = {
                part: _validate_candidate_tensor(
                    handle.get_tensor(name), matrix=matrix, part=part
                )
                for part, name in names.items()
            }
        except KeyError as exc:
            raise ValueError(
                f"candidate layer {spec.layer} expert {expert} is incomplete"
            ) from exc

        shared_key = f"{matrix}.{shared_part[matrix]}"
        shared = tensors[shared_part[matrix]]
        reference = shared_reference.get(shared_key)
        if reference is None:
            shared_reference[shared_key] = shared.clone()
        elif not torch.equal(shared, reference):
            difference = (shared.float() - reference.float()).abs()
            raise ValueError(
                f"layer {spec.layer} expert {expert} {shared_key} is not bit-exact "
                f"with the layer-shared transform (max_abs={float(difference.max())})"
            )

        for rank in range(TP_SIZE):
            logical_pair = tp12_logical_pair_index(
                spec.layer,
                expert,
                rank,
            )
            pair = candidate_trellis_pair(tensors["trellis"], logical_pair)
            pwrite_exact(
                descriptor,
                _byte_view(pair),
                spec.layout.trellis_pair_offset(rank, compressed_slot, matrix),
            )
            local = candidate_local_scale(
                tensors[local_part[matrix]],
                logical_pair,
            )
            pwrite_exact(
                descriptor,
                _byte_view(local),
                spec.layout.local_scale_offset(rank, compressed_slot, matrix),
            )


def materialize_layer(
    candidate_root: str | Path,
    store: OfficialMXFP4Store,
    destination: str | Path,
    spec: LayerMaterializationSpec,
    *,
    discard_partial: bool = False,
    preallocate: bool = True,
    x4t_destination: str | Path | None = None,
) -> dict[str, int | str]:
    """Write one layer atomically with bounded source and candidate residency."""

    candidate_root = Path(candidate_root).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if spec.layout.keep_storage != KEEP_STORAGE_EXTERNAL_X4T:
        raise ValueError("QSRT materialization requires external X4T storage")
    if x4t_destination is None:
        raise ValueError("QSRT materialization requires an X4T sidecar destination")
    x4t_path = Path(x4t_destination)
    if x4t_path.exists():
        raise FileExistsError(x4t_path)
    candidate_path = candidate_layer_path(candidate_root, spec.layer)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        if not discard_partial:
            raise FileExistsError(
                f"incomplete materialization exists: {partial}; resume with discard_partial"
            )
        _discard_stale_partial(partial)

    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    x4t_writer: X4TLayerWriter | None = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if preallocate and hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, spec.layout.disk_bytes)
        else:
            os.ftruncate(descriptor, spec.layout.disk_bytes)
        header = TP12LayerHeader(spec.layer, spec.layout, codebook=spec.codebook)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        format_section = pack_tp12_format_section(spec.formats)
        pwrite_exact(descriptor, _byte_view(format_section), header.format_offset)

        shared_reference: dict[str, torch.Tensor] = {}
        if spec.compressed:
            with safe_open(candidate_path, framework="pt", device="cpu") as handle:
                for slot, expert in enumerate(spec.compressed):
                    _write_candidate_expert(
                        descriptor,
                        handle,
                        spec,
                        expert=expert,
                        compressed_slot=slot,
                        shared_reference=shared_reference,
                    )
            expected_shared = {"w1.suh", "w3.suh", "w2.svh"}
            if set(shared_reference) != expected_shared:
                raise AssertionError("layer-shared scale collection did not close")
            shared_section = pack_tp12_shared_scale_section(
                shared_reference["w1.suh"],
                shared_reference["w3.suh"],
                shared_reference["w2.svh"],
            )
        else:
            shared_section = torch.zeros(
                SHARED_SCALE_SECTION_BYTES, dtype=torch.uint8
            )
        pwrite_exact(
            descriptor,
            _byte_view(shared_section),
            header.shared_scale_offset,
        )

        x4t_writer = X4TLayerWriter(x4t_path, layer=spec.layer)
        for expert in spec.kept:
            for matrix in C.EXPERT_MATRICES:
                raw = store.load_packed_matrix(spec.layer, expert, matrix)
                x4t_writer.add(expert, matrix, raw.packed, raw.scale)
        x4t_writer.close()
        os.fsync(descriptor)
        metadata = validate_materialized_layer_payloads(
            partial,
            spec,
            candidate_root,
            store,
            x4t_path=x4t_path,
        )
    except BaseException:
        if x4t_writer is not None:
            x4t_writer.abort()
        if x4t_path is not None:
            x4t_path.unlink(missing_ok=True)
        os.close(descriptor)
        partial.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    partial.replace(destination)
    _fsync_directory(destination.parent)
    # Payload closure is performed against the crash-safe partial file.  Once
    # it is atomically promoted, its contents are unchanged but its basename
    # is the public slab name used by structural validation and receipts.
    return {**metadata, "file": destination.name}
