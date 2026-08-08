"""Canonical TP-independent storage geometry for Kimi-K3 QSRT.

The on-disk sharding unit is a balanced 32-channel atom.  One atom contains
matching 16-channel stripes from the low and high records of a QSRT rate pair.
It therefore always carries six trellis bits per coefficient pair: K2+K4 for
a P24 atom or K3+K3 for a P33 atom.  The atom also owns the corresponding
local scale fragments for w1, w3, and w2.

Atoms are the format boundary.  Tensor parallel ranks own complete atoms and
never split, decode, or repack a trellis stream while loading a checkpoint.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

from kquant.qsrt import (
    EXPERTS_PER_LAYER,
    EXPERT_TRELLIS_BYTES,
    FORMAT_SECTION_BYTES,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LAYER_HEADER_BYTES,
    LOCAL_SCALE_VECTORS,
    PAIRS_PER_EXPERT,
    RECORDS_PER_EXPERT,
    SCALE_BYTES,
    SHARED_SCALE_SECTION_BYTES,
    STORAGE_ALIGNMENT,
    TILE_CHANNELS,
    ModeSpec,
    PackedQSRTTrellis,
    align_up,
    pair_rotation,
    record_bits,
    resolve_mode,
)


SCHEMA = "kquant_kimi_k3_qsrt_atoms_v1"
MAGIC = b"KQSRTA1\0"
VERSION = 1
PROFILE_ID_QSRT_E4M3 = 1
_HEADER = struct.Struct("<8s14I4Q")

ATOM_SIDE_CHANNELS = TILE_CHANNELS
ATOM_CHANNELS = 2 * ATOM_SIDE_CHANNELS
ATOMS_PER_RECORD_PAIR = 128 // ATOM_SIDE_CHANNELS
ATOMS_PER_EXPERT = PAIRS_PER_EXPERT * ATOMS_PER_RECORD_PAIR

K2_STRIPE_BYTES = ATOM_SIDE_CHANNELS * LATENT_CHANNELS * 2 // 8
K3_STRIPE_BYTES = ATOM_SIDE_CHANNELS * LATENT_CHANNELS * 3 // 8
K4_STRIPE_BYTES = ATOM_SIDE_CHANNELS * LATENT_CHANNELS * 4 // 8
MATRIX_ATOM_TRELLIS_BYTES = ATOM_CHANNELS * LATENT_CHANNELS * 3 // 8
MATRIX_ATOM_SCALE_BYTES = ATOM_CHANNELS * SCALE_BYTES
ATOM_TRELLIS_BYTES = 3 * MATRIX_ATOM_TRELLIS_BYTES
ATOM_SCALE_BYTES = LOCAL_SCALE_VECTORS * MATRIX_ATOM_SCALE_BYTES
ATOM_BUNDLE_BYTES = ATOM_TRELLIS_BYTES + ATOM_SCALE_BYTES
MATRIX_ATOM_TRELLIS_OFFSETS = {
    "w1": 0,
    "w3": MATRIX_ATOM_TRELLIS_BYTES,
    "w2": 2 * MATRIX_ATOM_TRELLIS_BYTES,
}
MATRIX_ATOM_SCALE_OFFSETS = {
    "w1": ATOM_TRELLIS_BYTES,
    "w3": ATOM_TRELLIS_BYTES + MATRIX_ATOM_SCALE_BYTES,
    "w2": ATOM_TRELLIS_BYTES + 2 * MATRIX_ATOM_SCALE_BYTES,
}

ATOM_SLAB_OFFSET = (
    LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES + SHARED_SCALE_SECTION_BYTES
)

# Equal-width direct-load partitions. These are exactly the divisors of
# Kimi-K3's 96-atom intermediate axis. Other shard counts still receive one
# contiguous whole-atom extent per shard, but their local widths differ by at
# most one atom and therefore require a padded runtime preparation cache.
EQUAL_SHARD_COUNTS = tuple(
    shard_count
    for shard_count in range(1, ATOMS_PER_EXPERT + 1)
    if ATOMS_PER_EXPERT % shard_count == 0
)


def balanced_atom_partition(shard_count: int, shard_index: int) -> tuple[int, int]:
    """Return ``(first_atom, atom_count)`` for any whole-atom partition.

    The quotient/remainder rule covers every physical atom exactly once and
    gives earlier shards one additional atom when 96 is not divisible by the
    requested shard count. No checkpoint bytes or metadata depend on this
    choice.
    """

    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    if not 1 <= shard_count <= ATOMS_PER_EXPERT:
        raise ValueError(f"shard_count must be in 1..{ATOMS_PER_EXPERT}")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise TypeError("shard_index must be an integer")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must be in 0..{shard_count - 1}")
    quotient, remainder = divmod(ATOMS_PER_EXPERT, shard_count)
    atom_count = quotient + int(shard_index < remainder)
    first_atom = shard_index * quotient + min(shard_index, remainder)
    return first_atom, atom_count


def logical_atom_index(record_pair: int, stripe: int) -> int:
    """Return the logical atom for one mirrored record-pair stripe."""

    if isinstance(record_pair, bool) or not isinstance(record_pair, int):
        raise TypeError("record_pair must be an integer")
    if not 0 <= record_pair < PAIRS_PER_EXPERT:
        raise ValueError(f"record_pair must be in 0..{PAIRS_PER_EXPERT - 1}")
    if isinstance(stripe, bool) or not isinstance(stripe, int):
        raise TypeError("stripe must be an integer")
    if not 0 <= stripe < ATOMS_PER_RECORD_PAIR:
        raise ValueError(f"stripe must be in 0..{ATOMS_PER_RECORD_PAIR - 1}")
    return record_pair * ATOMS_PER_RECORD_PAIR + stripe


def atom_components(logical_atom: int) -> tuple[int, int]:
    """Return ``(record_pair, stripe)`` for a logical atom."""

    if isinstance(logical_atom, bool) or not isinstance(logical_atom, int):
        raise TypeError("logical_atom must be an integer")
    if not 0 <= logical_atom < ATOMS_PER_EXPERT:
        raise ValueError(f"logical_atom must be in 0..{ATOMS_PER_EXPERT - 1}")
    return divmod(logical_atom, ATOMS_PER_RECORD_PAIR)


def atom_record_indices(logical_atom: int) -> tuple[int, int]:
    """Return the interleaved low/high physical record IDs in an atom."""

    record_pair, _ = atom_components(logical_atom)
    return 2 * record_pair, 2 * record_pair + 1


def atom_rates(mode: ModeSpec | str | int, logical_atom: int) -> tuple[int, int]:
    """Return the low/high rates stored in an atom for one matrix mode."""

    record_pair, _ = atom_components(logical_atom)
    spec = resolve_mode(mode)
    return (2, 4) if record_pair < spec.mode_id else (3, 3)


def physical_atom_rotation(layer: int, expert: int) -> int:
    """Whole-pair rotation that balances P24 work without breaking atoms."""

    return ATOMS_PER_RECORD_PAIR * pair_rotation(layer, expert)


def physical_atom_slot(layer: int, expert: int, logical_atom: int) -> int:
    """Return the global physical atom slot for one expert-local atom."""

    atom_components(logical_atom)
    return (logical_atom + physical_atom_rotation(layer, expert)) % ATOMS_PER_EXPERT


def logical_atom_for_slot(layer: int, expert: int, physical_slot: int) -> int:
    """Invert :func:`physical_atom_slot`."""

    if isinstance(physical_slot, bool) or not isinstance(physical_slot, int):
        raise TypeError("physical_slot must be an integer")
    if not 0 <= physical_slot < ATOMS_PER_EXPERT:
        raise ValueError(f"physical_slot must be in 0..{ATOMS_PER_EXPERT - 1}")
    return (physical_slot - physical_atom_rotation(layer, expert)) % ATOMS_PER_EXPERT


def _record_word_ranges(
    packed: PackedQSRTTrellis,
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(offset, words, K)`` for the 24 stored records."""

    packed.validate()
    result = []
    cursor = 0
    for bits in record_bits(packed.descriptor.mode):
        words = packed.descriptor.tiles_per_record * TILE_CHANNELS * bits
        result.append((cursor, words, bits))
        cursor += words
    if cursor != packed.payload.numel():
        raise AssertionError("candidate record accounting drifted")
    return tuple(result)


def pack_matrix_atoms(packed: PackedQSRTTrellis) -> torch.Tensor:
    """Transpose packed tile words into 96 fixed-size logical atoms.

    This operation never decodes edge symbols or reconstructs trellis states.
    It only changes the order of complete packed 16x16 tile paths.
    """

    ranges = _record_word_ranges(packed)
    descriptor = packed.descriptor
    words_per_atom = MATRIX_ATOM_TRELLIS_BYTES // torch.int16.itemsize
    output = torch.empty(
        (ATOMS_PER_EXPERT, words_per_atom),
        dtype=torch.int16,
        device=packed.payload.device,
    )
    orthogonal = descriptor.orthogonal_tiles
    for record_pair in range(PAIRS_PER_EXPERT):
        for stripe in range(ATOMS_PER_RECORD_PAIR):
            atom = logical_atom_index(record_pair, stripe)
            pieces = []
            for record in (2 * record_pair, 2 * record_pair + 1):
                offset, words, bits = ranges[record]
                words_per_tile = TILE_CHANNELS * bits
                source = packed.payload.narrow(0, offset, words)
                if descriptor.rate_axis == "k":
                    source = source.reshape(
                        ATOMS_PER_RECORD_PAIR, orthogonal, words_per_tile
                    )[stripe]
                else:
                    source = source.reshape(
                        orthogonal, ATOMS_PER_RECORD_PAIR, words_per_tile
                    )[:, stripe]
                pieces.append(source.contiguous().reshape(-1))
            joined = torch.cat(pieces)
            if joined.numel() != words_per_atom:
                raise AssertionError("balanced atom did not contain six rate bits")
            output[atom].copy_(joined)
    return output.contiguous()


def unpack_matrix_atoms(
    atoms: torch.Tensor,
    descriptor,
) -> PackedQSRTTrellis:
    """Invert :func:`pack_matrix_atoms` without decoding trellis bits."""

    words_per_atom = MATRIX_ATOM_TRELLIS_BYTES // torch.int16.itemsize
    if (
        atoms.dtype != torch.int16
        or tuple(atoms.shape) != (ATOMS_PER_EXPERT, words_per_atom)
        or not atoms.is_contiguous()
    ):
        raise ValueError(
            "matrix atoms must be contiguous int16 "
            f"[{ATOMS_PER_EXPERT}, {words_per_atom}]"
        )
    packed = PackedQSRTTrellis(
        descriptor=descriptor,
        payload=torch.empty(
            descriptor.payload_words,
            dtype=torch.int16,
            device=atoms.device,
        ),
    )
    ranges = _record_word_ranges(packed)
    orthogonal = descriptor.orthogonal_tiles
    for record_pair in range(PAIRS_PER_EXPERT):
        pair_rates = (
            record_bits(descriptor.mode)[2 * record_pair : 2 * record_pair + 2]
        )
        if len(pair_rates) != 2 or sum(pair_rates) != 6:
            raise AssertionError("descriptor contains a non-balanced record pair")
        for stripe in range(ATOMS_PER_RECORD_PAIR):
            atom = atoms[logical_atom_index(record_pair, stripe)]
            atom_cursor = 0
            for side, record in enumerate((2 * record_pair, 2 * record_pair + 1)):
                offset, words, bits = ranges[record]
                if bits != pair_rates[side]:
                    raise AssertionError("record rate accounting drifted")
                words_per_tile = TILE_CHANNELS * bits
                side_words = orthogonal * words_per_tile
                source = atom.narrow(0, atom_cursor, side_words)
                target = packed.payload.narrow(0, offset, words)
                if descriptor.rate_axis == "k":
                    target.reshape(
                        ATOMS_PER_RECORD_PAIR, orthogonal, words_per_tile
                    )[stripe].copy_(source.reshape(orthogonal, words_per_tile))
                else:
                    target.reshape(
                        orthogonal, ATOMS_PER_RECORD_PAIR, words_per_tile
                    )[:, stripe].copy_(source.reshape(orthogonal, words_per_tile))
                atom_cursor += side_words
            if atom_cursor != words_per_atom:
                raise AssertionError("atom inverse did not consume its payload")
    packed.validate()
    return packed


def pack_local_scale_atoms(scale: torch.Tensor) -> torch.Tensor:
    """Transpose one 3,072-value local FP16 scale into logical atoms."""

    if (
        scale.dtype != torch.float16
        or tuple(scale.shape) != (INTERMEDIATE_CHANNELS,)
        or not scale.is_contiguous()
    ):
        raise ValueError(
            f"local scale must be contiguous float16 [{INTERMEDIATE_CHANNELS}]"
        )
    output = torch.empty(
        (ATOMS_PER_EXPERT, ATOM_CHANNELS),
        dtype=torch.float16,
        device=scale.device,
    )
    for record_pair in range(PAIRS_PER_EXPERT):
        for stripe in range(ATOMS_PER_RECORD_PAIR):
            pieces = []
            for record in (2 * record_pair, 2 * record_pair + 1):
                begin = record * 128 + stripe * ATOM_SIDE_CHANNELS
                pieces.append(scale.narrow(0, begin, ATOM_SIDE_CHANNELS))
            output[logical_atom_index(record_pair, stripe)].copy_(torch.cat(pieces))
    return output.contiguous()


def unpack_local_scale_atoms(atoms: torch.Tensor) -> torch.Tensor:
    """Invert :func:`pack_local_scale_atoms`."""

    if (
        atoms.dtype != torch.float16
        or tuple(atoms.shape) != (ATOMS_PER_EXPERT, ATOM_CHANNELS)
        or not atoms.is_contiguous()
    ):
        raise ValueError(
            f"scale atoms must be contiguous float16 [{ATOMS_PER_EXPERT}, {ATOM_CHANNELS}]"
        )
    output = torch.empty(
        INTERMEDIATE_CHANNELS, dtype=torch.float16, device=atoms.device
    )
    for record_pair in range(PAIRS_PER_EXPERT):
        for stripe in range(ATOMS_PER_RECORD_PAIR):
            atom = atoms[logical_atom_index(record_pair, stripe)]
            for side, record in enumerate((2 * record_pair, 2 * record_pair + 1)):
                begin = record * 128 + stripe * ATOM_SIDE_CHANNELS
                output.narrow(0, begin, ATOM_SIDE_CHANNELS).copy_(
                    atom.narrow(
                        0,
                        side * ATOM_SIDE_CHANNELS,
                        ATOM_SIDE_CHANNELS,
                    )
                )
    return output.contiguous()


def rotate_expert_atoms(
    logical_atoms: torch.Tensor, layer: int, expert: int
) -> torch.Tensor:
    """Place one expert's logical atoms in global physical-slot order."""

    if logical_atoms.ndim < 1 or logical_atoms.shape[0] != ATOMS_PER_EXPERT:
        raise ValueError(f"logical_atoms must have leading dimension {ATOMS_PER_EXPERT}")
    order = torch.tensor(
        [
            logical_atom_for_slot(layer, expert, slot)
            for slot in range(ATOMS_PER_EXPERT)
        ],
        dtype=torch.long,
        device=logical_atoms.device,
    )
    return logical_atoms.index_select(0, order).contiguous()


def unrotate_expert_atoms(
    physical_atoms: torch.Tensor, layer: int, expert: int
) -> torch.Tensor:
    """Return physical-slot atoms to logical record-pair order."""

    if physical_atoms.ndim < 1 or physical_atoms.shape[0] != ATOMS_PER_EXPERT:
        raise ValueError(f"physical_atoms must have leading dimension {ATOMS_PER_EXPERT}")
    order = torch.tensor(
        [
            physical_atom_slot(layer, expert, atom)
            for atom in range(ATOMS_PER_EXPERT)
        ],
        dtype=torch.long,
        device=physical_atoms.device,
    )
    return physical_atoms.index_select(0, order).contiguous()


@dataclass(frozen=True)
class QSRTShardExtent:
    """One shard's single contiguous atom-slab read."""

    shard_count: int
    shard_index: int
    first_atom_slot: int
    atom_slots: int
    intermediate_channels: int
    offset_bytes: int
    extent_bytes: int
    payload_bytes: int
    padding_bytes: int


@dataclass(frozen=True)
class QSRTLayerLayout:
    """TP-independent atom-major compressed layer layout.

    The atom tensor has logical shape ``[96, compressed_experts, 129216]``.
    Each first-axis row is padded to 4 KiB.  A serving shard reads a contiguous
    range of first-axis rows, then a GPU preparation kernel drops row padding
    and transposes the data into the fused-MoE operand layout.
    """

    compressed_experts: int
    x4t_experts: int

    def __post_init__(self) -> None:
        for name, value in (
            ("compressed_experts", self.compressed_experts),
            ("x4t_experts", self.x4t_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.compressed_experts + self.x4t_experts != EXPERTS_PER_LAYER:
            raise ValueError(
                f"a layer must account for exactly {EXPERTS_PER_LAYER} experts"
            )

    @property
    def atom_slot_payload_bytes(self) -> int:
        return self.compressed_experts * ATOM_BUNDLE_BYTES

    @property
    def atom_slot_stride_bytes(self) -> int:
        return align_up(self.atom_slot_payload_bytes, STORAGE_ALIGNMENT)

    @property
    def atom_slot_padding_bytes(self) -> int:
        return self.atom_slot_stride_bytes - self.atom_slot_payload_bytes

    @property
    def compressed_payload_bytes(self) -> int:
        return self.compressed_experts * (
            EXPERT_TRELLIS_BYTES
            + LOCAL_SCALE_VECTORS * INTERMEDIATE_CHANNELS * SCALE_BYTES
        )

    @property
    def disk_bytes(self) -> int:
        return ATOM_SLAB_OFFSET + ATOMS_PER_EXPERT * self.atom_slot_stride_bytes

    def atom_bundle_offset(self, physical_slot: int, compressed_slot: int) -> int:
        """Return one expert atom's absolute file offset."""

        if (
            isinstance(physical_slot, bool)
            or not isinstance(physical_slot, int)
            or not 0 <= physical_slot < ATOMS_PER_EXPERT
        ):
            raise ValueError(f"physical_slot must be in 0..{ATOMS_PER_EXPERT - 1}")
        if (
            isinstance(compressed_slot, bool)
            or not isinstance(compressed_slot, int)
            or not 0 <= compressed_slot < self.compressed_experts
        ):
            raise ValueError(
                f"compressed_slot must be in 0..{self.compressed_experts - 1}"
            )
        return (
            ATOM_SLAB_OFFSET
            + physical_slot * self.atom_slot_stride_bytes
            + compressed_slot * ATOM_BUNDLE_BYTES
        )

    def shard_extent(
        self,
        shard_count: int,
        shard_index: int,
        *,
        require_equal: bool = False,
    ) -> QSRTShardExtent:
        """Return one shard's aligned, contiguous whole-atom extent.

        Set ``require_equal`` for serving kernels that require identical local
        dimensions. Uneven partitions remain useful for resharding and for
        building a padded deployment cache without touching canonical bytes.
        """

        first, atom_slots = balanced_atom_partition(shard_count, shard_index)
        if require_equal and shard_count not in EQUAL_SHARD_COUNTS:
            raise ValueError(
                f"{shard_count} shards do not divide the QSRT atom axis; "
                f"equal partitions are {EQUAL_SHARD_COUNTS}"
            )
        extent = atom_slots * self.atom_slot_stride_bytes
        payload = atom_slots * self.atom_slot_payload_bytes
        return QSRTShardExtent(
            shard_count=shard_count,
            shard_index=shard_index,
            first_atom_slot=first,
            atom_slots=atom_slots,
            intermediate_channels=atom_slots * ATOM_CHANNELS,
            offset_bytes=ATOM_SLAB_OFFSET + first * self.atom_slot_stride_bytes,
            extent_bytes=extent,
            payload_bytes=payload,
            padding_bytes=extent - payload,
        )

    def to_manifest(self) -> dict[str, int | str | tuple[int, ...]]:
        return {
            "schema": SCHEMA,
            "alignment_bytes": STORAGE_ALIGNMENT,
            "compressed_experts": self.compressed_experts,
            "x4t_experts": self.x4t_experts,
            "atom_channels": ATOM_CHANNELS,
            "atom_slots": ATOMS_PER_EXPERT,
            "atom_bundle_bytes": ATOM_BUNDLE_BYTES,
            "atom_slot_payload_bytes": self.atom_slot_payload_bytes,
            "atom_slot_stride_bytes": self.atom_slot_stride_bytes,
            "atom_slot_padding_bytes": self.atom_slot_padding_bytes,
            "compressed_payload_bytes": self.compressed_payload_bytes,
            "atom_slab_offset": ATOM_SLAB_OFFSET,
            "disk_bytes": self.disk_bytes,
            "equal_shard_counts": EQUAL_SHARD_COUNTS,
        }


@dataclass(frozen=True)
class QSRTLayerHeader:
    """Canonical TP-free header for one atom-major compressed layer."""

    layer: int
    layout: QSRTLayerLayout
    profile_id: int = PROFILE_ID_QSRT_E4M3

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise TypeError("layer must be an integer")
        if not 1 <= self.layer <= 92:
            raise ValueError("K3 MoE layer must be in 1..92")
        if self.profile_id != PROFILE_ID_QSRT_E4M3:
            raise ValueError("unsupported QSRT reconstruction profile")

    def to_bytes(self) -> bytes:
        header = _HEADER.pack(
            MAGIC,
            VERSION,
            self.layer,
            self.profile_id,
            EXPERTS_PER_LAYER,
            self.layout.compressed_experts,
            self.layout.x4t_experts,
            INTERMEDIATE_CHANNELS,
            LATENT_CHANNELS,
            ATOM_CHANNELS,
            ATOMS_PER_EXPERT,
            ATOM_BUNDLE_BYTES,
            STORAGE_ALIGNMENT,
            FORMAT_SECTION_BYTES,
            SHARED_SCALE_SECTION_BYTES,
            ATOM_SLAB_OFFSET,
            self.layout.atom_slot_payload_bytes,
            self.layout.atom_slot_stride_bytes,
            self.layout.disk_bytes,
        )
        return header + bytes(LAYER_HEADER_BYTES - len(header))

    @classmethod
    def from_bytes(cls, payload: bytes) -> "QSRTLayerHeader":
        if len(payload) != LAYER_HEADER_BYTES:
            raise ValueError(f"QSRT layer header must contain {LAYER_HEADER_BYTES} bytes")
        fields = _HEADER.unpack_from(payload)
        if any(payload[_HEADER.size:]):
            raise ValueError("QSRT layer header padding must be zero")
        (
            magic,
            version,
            layer,
            profile_id,
            experts,
            compressed,
            x4t,
            intermediate,
            latent,
            atom_channels,
            atom_slots,
            atom_bundle_bytes,
            alignment,
            format_section_bytes,
            shared_scale_section_bytes,
            atom_slab_offset,
            slot_payload_bytes,
            slot_stride_bytes,
            disk_bytes,
        ) = fields
        expected_scalars = {
            "magic": (magic, MAGIC),
            "version": (version, VERSION),
            "experts": (experts, EXPERTS_PER_LAYER),
            "intermediate": (intermediate, INTERMEDIATE_CHANNELS),
            "latent": (latent, LATENT_CHANNELS),
            "atom_channels": (atom_channels, ATOM_CHANNELS),
            "atom_slots": (atom_slots, ATOMS_PER_EXPERT),
            "atom_bundle_bytes": (atom_bundle_bytes, ATOM_BUNDLE_BYTES),
            "alignment": (alignment, STORAGE_ALIGNMENT),
            "format_section_bytes": (format_section_bytes, FORMAT_SECTION_BYTES),
            "shared_scale_section_bytes": (
                shared_scale_section_bytes,
                SHARED_SCALE_SECTION_BYTES,
            ),
            "atom_slab_offset": (atom_slab_offset, ATOM_SLAB_OFFSET),
        }
        for name, (actual, expected) in expected_scalars.items():
            if actual != expected:
                raise ValueError(f"QSRT layer header {name} is {actual!r}, expected {expected!r}")
        result = cls(
            layer=layer,
            layout=QSRTLayerLayout(compressed, x4t),
            profile_id=profile_id,
        )
        expected_layout = {
            "atom_slot_payload_bytes": (
                slot_payload_bytes,
                result.layout.atom_slot_payload_bytes,
            ),
            "atom_slot_stride_bytes": (
                slot_stride_bytes,
                result.layout.atom_slot_stride_bytes,
            ),
            "disk_bytes": (disk_bytes, result.layout.disk_bytes),
        }
        for name, (actual, expected) in expected_layout.items():
            if actual != expected:
                raise ValueError(f"QSRT layer header {name} is {actual}, expected {expected}")
        return result


if ATOMS_PER_EXPERT != 96:
    raise AssertionError("Kimi-K3 QSRT must contain 96 balanced atoms")
if K2_STRIPE_BYTES + K4_STRIPE_BYTES != 2 * K3_STRIPE_BYTES:
    raise AssertionError("P24 and P33 atom payloads must have equal size")
if MATRIX_ATOM_TRELLIS_BYTES != K2_STRIPE_BYTES + K4_STRIPE_BYTES:
    raise AssertionError("matrix atom trellis accounting drifted")
if ATOMS_PER_EXPERT * ATOM_BUNDLE_BYTES != (
    EXPERT_TRELLIS_BYTES
    + LOCAL_SCALE_VECTORS * INTERMEDIATE_CHANNELS * SCALE_BYTES
):
    raise AssertionError("atomization changed the compressed expert payload")
