"""Materialize and read canonical TP-independent QSRT atom layers."""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
import torch
from safetensors import safe_open

from kquant import constants as C
from kquant.pack.qsrt_candidates import candidate_tensor_name
from kquant.qsrt import (
    EXPERTS_PER_LAYER,
    FORMAT_SECTION_BYTES,
    INTERMEDIATE_CHANNELS,
    LAYER_HEADER_BYTES,
    LATENT_CHANNELS,
    MATRIX_RATE_AXIS,
    MATRIX_TRELLIS_BYTES,
    SHARED_SCALE_SECTION_BYTES,
    ExpertFormatSpec,
    QSRTTrellisDescriptor,
    PackedQSRTTrellis,
    pack_qsrt_format_section,
    pack_qsrt_shared_scale_section,
    unpack_qsrt_format_section,
    unpack_qsrt_shared_scale_section,
    resolve_expert_format,
)
from kquant.qsrt_storage import (
    ATOM_BUNDLE_BYTES,
    ATOM_TRELLIS_BYTES,
    ATOMS_PER_EXPERT,
    MATRIX_ATOM_SCALE_BYTES,
    MATRIX_ATOM_SCALE_OFFSETS,
    MATRIX_ATOM_TRELLIS_BYTES,
    MATRIX_ATOM_TRELLIS_OFFSETS,
    SCHEMA,
    QSRTLayerHeader,
    QSRTLayerLayout,
    pack_local_scale_atoms,
    pack_matrix_atoms,
    rotate_expert_atoms,
    unrotate_expert_atoms,
    unpack_local_scale_atoms,
    unpack_matrix_atoms,
)


LAYER_PREFIX = "qsrt-layer-"
LAYER_SUFFIX = ".atoms"


def layer_filename(layer: int) -> str:
    if isinstance(layer, bool) or not isinstance(layer, int) or not 1 <= layer <= 92:
        raise ValueError("Kimi-K3 MoE layer must lie in 1..92")
    return f"{LAYER_PREFIX}{layer:05d}{LAYER_SUFFIX}"


def candidate_layer_path(root: str | Path, layer: int) -> Path:
    return Path(root) / "candidates" / f"qsrt-layer-{layer:05d}.safetensors"


def _flat_candidate_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    values: int,
    name: str,
) -> torch.Tensor:
    result = tensor.detach().cpu().contiguous().reshape(-1)
    if result.dtype != dtype or result.numel() != values:
        raise ValueError(
            f"{name} must contain {values} contiguous {dtype} values, got "
            f"{result.numel()} {result.dtype} values"
        )
    if dtype != torch.int16 and not bool(torch.all(torch.isfinite(result))):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _matrix_descriptor(matrix: str, mode_id: int) -> QSRTTrellisDescriptor:
    rate_axis = MATRIX_RATE_AXIS[matrix]
    return QSRTTrellisDescriptor(
        mode_id=mode_id,
        rate_axis=rate_axis,
        k_tiles=(INTERMEDIATE_CHANNELS if rate_axis == "k" else LATENT_CHANNELS)
        // 16,
        n_tiles=(LATENT_CHANNELS if rate_axis == "k" else INTERMEDIATE_CHANNELS)
        // 16,
    )


def assemble_candidate_atoms(
    *,
    layer: int,
    expert: int,
    format_spec: ExpertFormatSpec,
    tensors: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return one expert as physical ``[96, 129216]`` byte atoms.

    The returned shared-scale dictionary is checked by the layer writer across
    every compressed expert. No trellis state or reconstruction value is
    decoded while changing the storage order.
    """

    if format_spec.is_x4t:
        raise ValueError("X4T experts do not have QSRT atoms")
    assert format_spec.r13 is not None and format_spec.r2 is not None
    if set(tensors) != set(C.EXPERT_MATRICES):
        raise ValueError("candidate must contain w1, w3, and w2")
    logical = torch.empty(
        (ATOMS_PER_EXPERT, ATOM_BUNDLE_BYTES), dtype=torch.uint8
    )
    shared: dict[str, torch.Tensor] = {}
    for matrix in C.EXPERT_MATRICES:
        parts = tensors[matrix]
        if set(parts) != {"trellis", "suh", "svh"}:
            raise ValueError(f"{matrix} candidate components are incomplete")
        mode_id = format_spec.r2 if matrix == "w2" else format_spec.r13
        descriptor = _matrix_descriptor(matrix, mode_id)
        trellis = _flat_candidate_tensor(
            parts["trellis"],
            dtype=torch.int16,
            values=MATRIX_TRELLIS_BYTES // torch.int16.itemsize,
            name=f"{matrix}.trellis",
        )
        packed = PackedQSRTTrellis(descriptor, trellis)
        trellis_atoms = pack_matrix_atoms(packed).view(torch.uint8).reshape(
            ATOMS_PER_EXPERT, MATRIX_ATOM_TRELLIS_BYTES
        )
        trellis_begin = MATRIX_ATOM_TRELLIS_OFFSETS[matrix]
        logical[:, trellis_begin : trellis_begin + MATRIX_ATOM_TRELLIS_BYTES].copy_(
            trellis_atoms
        )

        shared_part = "svh" if matrix == "w2" else "suh"
        local_part = "suh" if matrix == "w2" else "svh"
        shared[f"{matrix}.{shared_part}"] = _flat_candidate_tensor(
            parts[shared_part],
            dtype=torch.float16,
            values=LATENT_CHANNELS,
            name=f"{matrix}.{shared_part}",
        )
        local = _flat_candidate_tensor(
            parts[local_part],
            dtype=torch.float16,
            values=INTERMEDIATE_CHANNELS,
            name=f"{matrix}.{local_part}",
        )
        scale_atoms = pack_local_scale_atoms(local).view(torch.uint8).reshape(
            ATOMS_PER_EXPERT, MATRIX_ATOM_SCALE_BYTES
        )
        scale_begin = MATRIX_ATOM_SCALE_OFFSETS[matrix]
        logical[:, scale_begin : scale_begin + MATRIX_ATOM_SCALE_BYTES].copy_(
            scale_atoms
        )
    if ATOM_TRELLIS_BYTES + 3 * MATRIX_ATOM_SCALE_BYTES != ATOM_BUNDLE_BYTES:
        raise AssertionError("atom bundle accounting drifted")
    return rotate_expert_atoms(logical, layer, expert), shared


@dataclass(frozen=True)
class QSRTAtomLayerSpec:
    layer: int
    formats: tuple[ExpertFormatSpec, ...]
    compressed: tuple[int, ...]
    x4t: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer not in C.MOE_LAYERS:
            raise ValueError("atom layer must be a Kimi-K3 MoE layer")
        if len(self.formats) != EXPERTS_PER_LAYER:
            raise ValueError(f"atom layer must describe {EXPERTS_PER_LAYER} experts")
        expected_compressed = tuple(
            expert for expert, value in enumerate(self.formats) if not value.is_x4t
        )
        expected_x4t = tuple(
            expert for expert, value in enumerate(self.formats) if value.is_x4t
        )
        if self.compressed != expected_compressed or self.x4t != expected_x4t:
            raise ValueError("atom layer tiers disagree with its format table")

    @property
    def layout(self) -> QSRTLayerLayout:
        return QSRTLayerLayout(len(self.compressed), len(self.x4t))


def _pwrite_exact(descriptor: int, payload: torch.Tensor | bytes, offset: int) -> None:
    data = payload if isinstance(payload, bytes) else payload.numpy().tobytes()
    cursor = 0
    while cursor < len(data):
        written = os.pwrite(descriptor, data[cursor:], offset + cursor)
        if written <= 0:
            raise OSError("short QSRT atom write")
        cursor += written


def materialize_atom_layer(
    candidate_root: str | Path,
    destination: str | Path,
    spec: QSRTAtomLayerSpec,
    *,
    discard_partial: bool = False,
) -> dict[str, int | str]:
    """Write one canonical atom layer atomically with bounded residency."""

    candidate_path = candidate_layer_path(candidate_root, spec.layer)
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        if not discard_partial:
            raise FileExistsError(partial)
        partial.unlink()
    layout = spec.layout
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(descriptor, 0, layout.disk_bytes)
        else:
            os.ftruncate(descriptor, layout.disk_bytes)
        _pwrite_exact(descriptor, QSRTLayerHeader(spec.layer, layout).to_bytes(), 0)
        _pwrite_exact(
            descriptor,
            pack_qsrt_format_section(spec.formats),
            LAYER_HEADER_BYTES,
        )
        shared_reference: dict[str, torch.Tensor] = {}
        with safe_open(candidate_path, framework="pt", device="cpu") as handle:
            for compressed_slot, expert in enumerate(spec.compressed):
                tensors = {
                    matrix: {
                        part: handle.get_tensor(
                            candidate_tensor_name(spec.layer, expert, matrix, part)
                        )
                        for part in ("trellis", "suh", "svh")
                    }
                    for matrix in C.EXPERT_MATRICES
                }
                atoms, shared = assemble_candidate_atoms(
                    layer=spec.layer,
                    expert=expert,
                    format_spec=spec.formats[expert],
                    tensors=tensors,
                )
                for name, value in shared.items():
                    reference = shared_reference.setdefault(name, value.clone())
                    if not torch.equal(reference, value):
                        raise ValueError(f"layer-shared transform {name} drifted")
                for physical_slot in range(ATOMS_PER_EXPERT):
                    _pwrite_exact(
                        descriptor,
                        atoms[physical_slot],
                        layout.atom_bundle_offset(physical_slot, compressed_slot),
                    )
        if spec.compressed:
            expected = {"w1.suh", "w3.suh", "w2.svh"}
            if set(shared_reference) != expected:
                raise AssertionError("shared transform inventory did not close")
            shared_section = pack_qsrt_shared_scale_section(
                shared_reference["w1.suh"],
                shared_reference["w3.suh"],
                shared_reference["w2.svh"],
            )
        else:
            shared_section = torch.zeros(SHARED_SCALE_SECTION_BYTES, dtype=torch.uint8)
        _pwrite_exact(
            descriptor,
            shared_section,
            LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES,
        )
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        partial.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    partial.replace(destination)
    return {
        "schema": SCHEMA,
        "layer": spec.layer,
        "compressed_experts": len(spec.compressed),
        "x4t_experts": len(spec.x4t),
        "disk_bytes": layout.disk_bytes,
    }


class QSRTAtomLayerReader:
    """Bounded reader whose serving operation is one contiguous shard extent."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor = os.open(self.path, os.O_RDONLY)
        try:
            self.header = QSRTLayerHeader.from_bytes(
                os.pread(self._descriptor, LAYER_HEADER_BYTES, 0)
            )
            if self.path.stat().st_size != self.header.layout.disk_bytes:
                raise ValueError("QSRT atom layer size disagrees with its header")
            format_raw = os.pread(
                self._descriptor,
                FORMAT_SECTION_BYTES,
                LAYER_HEADER_BYTES,
            )
            self.formats = unpack_qsrt_format_section(
                torch.frombuffer(bytearray(format_raw), dtype=torch.uint8)
            )
            self.format_specs = tuple(
                resolve_expert_format(value) for value in self.formats
            )
            self.compressed = tuple(
                expert
                for expert, format_spec in enumerate(self.format_specs)
                if not format_spec.is_x4t
            )
            self.x4t = tuple(
                expert
                for expert, format_spec in enumerate(self.format_specs)
                if format_spec.is_x4t
            )
            self._compressed_slots = {
                expert: slot for slot, expert in enumerate(self.compressed)
            }
            shared_offset = LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES
            shared_raw = os.pread(
                self._descriptor, SHARED_SCALE_SECTION_BYTES, shared_offset
            )
            self.shared_scales = unpack_qsrt_shared_scale_section(
                torch.frombuffer(bytearray(shared_raw), dtype=torch.uint8)
            )
        except BaseException:
            os.close(self._descriptor)
            raise

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> "QSRTAtomLayerReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read_shard(
        self,
        shard_count: int,
        shard_index: int,
        *,
        require_equal: bool = False,
    ) -> torch.Tensor:
        """Read and de-pad one contiguous whole-atom shard."""

        if self._descriptor < 0:
            raise RuntimeError("QSRT atom reader is closed")
        extent = self.header.layout.shard_extent(
            shard_count,
            shard_index,
            require_equal=require_equal,
        )
        raw = os.pread(self._descriptor, extent.extent_bytes, extent.offset_bytes)
        if len(raw) != extent.extent_bytes:
            raise ValueError("short QSRT atom extent")
        source = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            extent.atom_slots, self.header.layout.atom_slot_stride_bytes
        )
        return source[:, : self.header.layout.atom_slot_payload_bytes].reshape(
            extent.atom_slots,
            self.header.layout.compressed_experts,
            ATOM_BUNDLE_BYTES,
        ).contiguous()

    def read_expert_atoms(self, compressed_slot: int) -> torch.Tensor:
        """Read one expert for bit-exact validation, not serving."""

        if self._descriptor < 0:
            raise RuntimeError("QSRT atom reader is closed")
        physical = torch.empty(
            (ATOMS_PER_EXPERT, ATOM_BUNDLE_BYTES), dtype=torch.uint8
        )
        for slot in range(ATOMS_PER_EXPERT):
            raw = os.pread(
                self._descriptor,
                ATOM_BUNDLE_BYTES,
                self.header.layout.atom_bundle_offset(slot, compressed_slot),
            )
            if len(raw) != ATOM_BUNDLE_BYTES:
                raise ValueError("short QSRT expert atom")
            physical[slot].copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
        return physical

    def read_compressed_matrix(
        self, expert: int, matrix: str
    ) -> dict[str, torch.Tensor]:
        """Reconstruct one full candidate payload for CPU/reference decode.

        Serving paths should use :meth:`read_shard`; this inverse atomization
        exists for bit-exact validation and streamed PyTorch comparisons.
        """

        try:
            compressed_slot = self._compressed_slots[expert]
        except KeyError as exc:
            raise ValueError(f"expert {expert} is not in the QSRT tier") from exc
        if matrix not in C.EXPERT_MATRICES:
            raise ValueError(f"unknown expert matrix: {matrix}")
        format_spec = self.format_specs[expert]
        assert format_spec.r13 is not None and format_spec.r2 is not None
        mode_id = format_spec.r2 if matrix == "w2" else format_spec.r13
        descriptor = _matrix_descriptor(matrix, mode_id)
        physical = self.read_expert_atoms(compressed_slot)
        logical = unrotate_expert_atoms(physical, self.header.layer, expert)

        trellis_begin = MATRIX_ATOM_TRELLIS_OFFSETS[matrix]
        trellis_raw = logical[
            :, trellis_begin : trellis_begin + MATRIX_ATOM_TRELLIS_BYTES
        ].contiguous()
        trellis_atoms = trellis_raw.reshape(-1).view(torch.int16).reshape(
            ATOMS_PER_EXPERT,
            MATRIX_ATOM_TRELLIS_BYTES // torch.int16.itemsize,
        )
        trellis = unpack_matrix_atoms(trellis_atoms.contiguous(), descriptor).payload

        scale_begin = MATRIX_ATOM_SCALE_OFFSETS[matrix]
        scale_raw = logical[
            :, scale_begin : scale_begin + MATRIX_ATOM_SCALE_BYTES
        ].contiguous()
        scale_atoms = scale_raw.reshape(-1).view(torch.float16).reshape(
            ATOMS_PER_EXPERT, -1
        )
        local = unpack_local_scale_atoms(scale_atoms.contiguous())
        w1_suh, w3_suh, w2_svh = self.shared_scales
        if matrix == "w1":
            suh, svh = w1_suh, local
        elif matrix == "w3":
            suh, svh = w3_suh, local
        else:
            suh, svh = local, w2_svh
        return {
            "trellis": trellis,
            "suh": suh.clone().contiguous(),
            "svh": svh.clone().contiguous(),
        }


__all__ = [
    "QSRTAtomLayerReader",
    "QSRTAtomLayerSpec",
    "assemble_candidate_atoms",
    "candidate_layer_path",
    "layer_filename",
    "materialize_atom_layer",
]
