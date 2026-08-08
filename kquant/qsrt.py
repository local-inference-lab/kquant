"""Reference codec geometry for Kimi-K3 QSRT candidates.

Candidate payloads describe complete experts and contain no tensor-parallel
ownership.  TP sharding is a property of the atom-major storage/runtime layer.
The ordinary expert weights use ``[out_features, in_features]`` orientation,
while the EXL encoder consumes their transpose, ``[K, N]``.  Consequently the
shared 3072-channel intermediate axis is EXL ``N`` for ``w1``/``w3`` and EXL
``K`` for ``w2``.

The reference packer preserves the native EXL bit ordering for each 16x16
trellis tile.  Its inverse reconstructs both the stored K-bit edge symbols and
the full 16-bit cyclic trellis states consumed by the SQG decoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from kquant.exl3_reference import (
    CODEBOOK_SQG_XOR_CHEB_T12,
    CODEBOOK_SQG_CHEB_NORMAL_E4M3,
    CODEBOOK_SQG_CHEB,
    CODEBOOK_SQG_NORMAL_E4M3,
    QSRT_CODEBOOKS,
    decode_qsrt_weight,
    reconstruct_trellis_states,
)


EXPERTS_PER_LAYER = 896
INTERMEDIATE_CHANNELS = 3072
LATENT_CHANNELS = 3584
CONTEXT_GROUP_CHANNELS = 4
TILE_CHANNELS = 16
TILE_VALUES = TILE_CHANNELS * TILE_CHANNELS
RECORD_CHANNELS = 128
PAIR_CHANNELS = 2 * RECORD_CHANNELS
RECORDS_PER_EXPERT = INTERMEDIATE_CHANNELS // RECORD_CHANNELS
PAIRS_PER_EXPERT = RECORDS_PER_EXPERT // 2
TILES_PER_RECORD_AXIS = RECORD_CHANNELS // TILE_CHANNELS
SCHEMA = "kquant_kimi_k3_qsrt_candidate_v1"
LOGICAL_CANDIDATE_SCHEMAS = (SCHEMA,)
STORAGE_ALIGNMENT = 4096
PAIR_ROTATION_EXPERT_MULTIPLIER = 5

# The phase-1 table uses one byte per expert.  Compressed entries contain
# two four-bit transfer counts: high nibble r13 for fused w1/w3, low nibble r2
# for w2.  Each value is in 0..2.  0xff denotes the exact X4T tier.  The nibbles
# remain deliberately roomy so the fixed table is cheap to inspect and decode.
FORMAT_BITS = 8
FORMAT_X4T = 0xFF
FORMAT_NIBBLE_MASK = 0x0F

LAYER_HEADER_BYTES = STORAGE_ALIGNMENT
FORMAT_TABLE_BYTES = (EXPERTS_PER_LAYER * FORMAT_BITS + 7) // 8
FORMAT_SECTION_BYTES = STORAGE_ALIGNMENT
HOT_FORMAT_BYTES = EXPERTS_PER_LAYER
HOT_SLOT_BYTES = EXPERTS_PER_LAYER * torch.int16.itemsize
HOT_METADATA_BYTES = HOT_FORMAT_BYTES + HOT_SLOT_BYTES

# Shared hidden-side vectors are w1.suh, w3.suh, and w2.svh.  The three
# corresponding intermediate-side vectors remain expert-local.  Their
# atom-major storage is defined independently from a serving TP size.
SHARED_SCALE_VECTORS = 3
LOCAL_SCALE_VECTORS = 3
SHARED_SCALE_ORDER = ("w1.suh", "w3.suh", "w2.svh")
LOCAL_SCALE_ORDER = ("w1.svh", "w3.svh", "w2.suh")
SCALE_BYTES = torch.float16.itemsize
SHARED_SCALE_BYTES = SHARED_SCALE_VECTORS * LATENT_CHANNELS * SCALE_BYTES
SHARED_SCALE_SECTION_BYTES = 6 * STORAGE_ALIGNMENT

# P24 and P33 both carry a six-bit sum over 256 intermediate channels.
PAIR_BYTES = PAIR_CHANNELS * LATENT_CHANNELS * 3 // 8
MATRIX_TRELLIS_BYTES = PAIRS_PER_EXPERT * PAIR_BYTES
EXPERT_TRELLIS_BYTES = 3 * MATRIX_TRELLIS_BYTES
RateAxis = Literal["k", "n"]
PairKind = Literal["P24", "P33"]

MATRIX_RATE_AXIS: dict[str, RateAxis] = {
    "w1": "n",
    "w3": "n",
    "w2": "k",
}


def pair_rotation(layer: int, expert: int) -> int:
    """Cyclic logical-pair placement used to balance physical atom slots."""

    if (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or layer not in range(1, 93)
    ):
        raise ValueError("K3 MoE layer must be in 1..92")
    if (
        isinstance(expert, bool)
        or not isinstance(expert, int)
        or not 0 <= expert < EXPERTS_PER_LAYER
    ):
        raise ValueError(f"expert must be in 0..{EXPERTS_PER_LAYER - 1}")
    return (PAIR_ROTATION_EXPERT_MULTIPLIER * expert + layer) % PAIRS_PER_EXPERT


def logical_pair_for_slot(layer: int, expert: int, pair_slot: int) -> int:
    """Logical candidate pair stored in one physical pair slot."""

    if (
        isinstance(pair_slot, bool)
        or not isinstance(pair_slot, int)
        or not 0 <= pair_slot < PAIRS_PER_EXPERT
    ):
        raise ValueError(f"pair_slot must be in 0..{PAIRS_PER_EXPERT - 1}")
    return (pair_slot - pair_rotation(layer, expert)) % PAIRS_PER_EXPERT


def physical_pair_slot(layer: int, expert: int, logical_pair: int) -> int:
    """Physical slot containing one logical candidate pair."""

    if (
        isinstance(logical_pair, bool)
        or not isinstance(logical_pair, int)
        or not 0 <= logical_pair < PAIRS_PER_EXPERT
    ):
        raise ValueError(f"logical_pair must be in 0..{PAIRS_PER_EXPERT - 1}")
    return (logical_pair + pair_rotation(layer, expert)) % PAIRS_PER_EXPERT


@dataclass(frozen=True)
class ModeSpec:
    """One rate-transfer schedule, ordered from low to high importance.

    ``context_bits`` contains one entry per final 128-channel record.
    ``mode_id`` is the number of K2/K4 record exchanges.
    """

    mode_id: int
    name: str
    context_bits: tuple[int, ...]
    def __post_init__(self) -> None:
        if isinstance(self.mode_id, bool) or not isinstance(self.mode_id, int):
            raise TypeError("mode_id must be an integer")
        if not 0 <= self.mode_id <= 2:
            raise ValueError("QSRT mode_id must be R0, R1, or R2")
        if not self.name:
            raise ValueError("mode name must not be empty")
        if not self.context_bits or len(self.context_bits) % 2:
            raise ValueError("a mode must contain an even, nonzero context count")
        if any(bits not in (2, 3, 4) for bits in self.context_bits):
            raise ValueError("Kimi-K3 QSRT supports only K2, K3, and K4")
        if sum(self.context_bits) != 3 * len(self.context_bits):
            raise ValueError("a mode must average exactly three trellis bits")
        if INTERMEDIATE_CHANNELS % len(self.context_bits):
            raise ValueError("contexts must divide the 3072-channel axis")
        context_channels = INTERMEDIATE_CHANNELS // len(self.context_bits)
        if context_channels % RECORD_CHANNELS:
            raise ValueError("each context must contain whole 128-channel records")
        if tuple(sorted(self.context_bits)) != self.context_bits:
            raise ValueError("context rates must be monotone from K2 through K4")
        records_per_context = RECORDS_PER_EXPERT // len(self.context_bits)
        k2_records = self.context_bits.count(2) * records_per_context
        k4_records = self.context_bits.count(4) * records_per_context
        if k2_records != k4_records:
            raise ValueError("K2 and K4 record counts must match")
        if self.mode_id != k2_records:
            raise ValueError(
                f"mode_id must equal the rate-transfer count {k2_records}"
            )

    @property
    def context_count(self) -> int:
        return len(self.context_bits)

    @property
    def records_per_context(self) -> int:
        return RECORDS_PER_EXPERT // self.context_count


def _rate_transfer_mode(r: int) -> ModeSpec:
    return ModeSpec(
        r,
        f"R{r}",
        (2,) * r + (3,) * (RECORDS_PER_EXPERT - 2 * r) + (4,) * r,
    )


RATE_TRANSFER_MODES = tuple(_rate_transfer_mode(r) for r in range(3))
R0 = RATE_TRANSFER_MODES[0]
R1 = RATE_TRANSFER_MODES[1]
R2 = RATE_TRANSFER_MODES[2]
REPRESENTATIVE_MODE_CANDIDATES = RATE_TRANSFER_MODES
EXPERIMENTAL_MODE_CANDIDATES = RATE_TRANSFER_MODES

# Frozen phase-1 encoder policy.  The payload schema still parses the complete
# ladder for controlled studies, but the initial all-expert run searches R0,
# R1, and R2 independently for the fused w1/w3 family and for w2.  The
# covariance alphas are offline encoder controls and consume no format bits.
PHASE1_MODE_IDS = (0, 1, 2)
PHASE1_MODE_CANDIDATES = tuple(RATE_TRANSFER_MODES[r] for r in PHASE1_MODE_IDS)
PHASE1_SHARED_RATE_MODE = False
PHASE1_H13_EXPERT_LOCAL_ALPHA = 0.0
# Maximum local weight for adaptive, scaled-identity OAS H2 shrinkage.  The
# actual alpha is support-dependent and can be substantially smaller.
PHASE1_H2_EXPERT_LOCAL_ALPHA = 0.75
PHASE1_H2_LOCAL_BASIS = "decoded_candidate_post_situ"
PHASE1_H2_SHRINKAGE_POLICY = "weighted_oas_scaled_identity"
_MODES_BY_NAME = {mode.name: mode for mode in RATE_TRANSFER_MODES}
_MODES_BY_ID = {mode.mode_id: mode for mode in RATE_TRANSFER_MODES}


def mode_from_context_bits(context_bits: Sequence[int]) -> ModeSpec:
    """Resolve any legal equal-context ladder allocation to canonical ``R_r``."""

    bits = tuple(context_bits)
    if not bits or RECORDS_PER_EXPERT % len(bits):
        raise ValueError("allocation contexts must divide the 24 records")
    records_per_context = RECORDS_PER_EXPERT // len(bits)
    r = bits.count(2) * records_per_context
    # ModeSpec performs monotonicity, rate, and equal-donor validation.
    ModeSpec(r, "candidate", bits)
    return RATE_TRANSFER_MODES[r]


@dataclass(frozen=True)
class ExpertFormatSpec:
    """Provisional per-expert tier and matrix-family rate schedules."""

    r13: int | None
    r2: int | None

    def __post_init__(self) -> None:
        if (self.r13 is None) != (self.r2 is None):
            raise ValueError("r13 and r2 must both be absent or both be present")
        for name, value in (("r13", self.r13), ("r2", self.r2)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 2:
                raise ValueError(f"{name} must be R0, R1, or R2")

    @classmethod
    def x4t(cls) -> "ExpertFormatSpec":
        return cls(None, None)

    @classmethod
    def compressed(cls, r13: int, r2: int | None = None) -> "ExpertFormatSpec":
        return cls(r13, r13 if r2 is None else r2)

    @property
    def is_x4t(self) -> bool:
        return self.r13 is None

    @property
    def name(self) -> str:
        if self.is_x4t:
            return "X4T"
        assert self.r13 is not None and self.r2 is not None
        return f"R{self.r13}" if self.r13 == self.r2 else f"R{self.r13}/R{self.r2}"

    @property
    def code(self) -> int:
        if self.is_x4t:
            return FORMAT_X4T
        assert self.r13 is not None and self.r2 is not None
        return (self.r13 << 4) | self.r2

    @classmethod
    def from_code(cls, code: int) -> "ExpertFormatSpec":
        if code == FORMAT_X4T:
            return cls.x4t()
        r13 = code >> 4
        r2 = code & FORMAT_NIBBLE_MASK
        if r13 > 2 or r2 > 2:
            raise ValueError(f"format table contains invalid rate code 0x{code:02x}")
        return cls.compressed(r13, r2)


ExpertFormatLike = str | ModeSpec | ExpertFormatSpec | tuple[int, int]


def align_up(value: int, alignment: int = STORAGE_ALIGNMENT) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("value must be a non-negative integer")
    if (
        isinstance(alignment, bool)
        or not isinstance(alignment, int)
        or alignment <= 0
        or alignment & (alignment - 1)
    ):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def resolve_expert_format(value: ExpertFormatLike) -> ExpertFormatSpec:
    if isinstance(value, ExpertFormatSpec):
        return value
    if isinstance(value, ModeSpec):
        return ExpertFormatSpec.compressed(value.mode_id)
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("expert format tuple must be (r13, r2)")
        return ExpertFormatSpec.compressed(value[0], value[1])
    if not isinstance(value, str):
        raise TypeError(
            "expert format must be a name, ModeSpec, ExpertFormatSpec, or pair"
        )
    normalized = value.upper()
    if normalized == "X4T":
        return ExpertFormatSpec.x4t()
    if "/" in normalized:
        left, right = normalized.split("/", 1)
        return ExpertFormatSpec.compressed(
            resolve_mode(left).mode_id, resolve_mode(right).mode_id
        )
    return ExpertFormatSpec.compressed(resolve_mode(normalized).mode_id)


def pack_qsrt_format_table(formats: Sequence[ExpertFormatLike]) -> torch.Tensor:
    """Pack 896 one-byte expert format codes."""

    if len(formats) != EXPERTS_PER_LAYER:
        raise ValueError(f"format table must contain {EXPERTS_PER_LAYER} experts")
    return torch.tensor(
        [resolve_expert_format(value).code for value in formats],
        dtype=torch.uint8,
    ).contiguous()


def unpack_qsrt_format_table(payload: torch.Tensor) -> tuple[str, ...]:
    """Decode the format table and reject invalid rate nibbles."""

    if payload.dtype != torch.uint8:
        raise TypeError("format table must use torch.uint8")
    if payload.ndim != 1 or payload.numel() != FORMAT_TABLE_BYTES:
        raise ValueError(
            f"format table must contain exactly {FORMAT_TABLE_BYTES} bytes"
        )
    return tuple(
        ExpertFormatSpec.from_code(code).name for code in payload.cpu().tolist()
    )


def pack_qsrt_format_section(formats: Sequence[ExpertFormatLike]) -> torch.Tensor:
    """Build the canonical 4-KiB on-disk format-table section."""

    table = pack_qsrt_format_table(formats)
    section = torch.zeros(FORMAT_SECTION_BYTES, dtype=torch.uint8)
    section[:FORMAT_TABLE_BYTES].copy_(table)
    return section


def unpack_qsrt_format_section(payload: torch.Tensor) -> tuple[str, ...]:
    if payload.dtype != torch.uint8:
        raise TypeError("format section must use torch.uint8")
    if payload.ndim != 1 or payload.numel() != FORMAT_SECTION_BYTES:
        raise ValueError(
            f"format section must contain exactly {FORMAT_SECTION_BYTES} bytes"
        )
    if bool(torch.any(payload[FORMAT_TABLE_BYTES:] != 0)):
        raise ValueError("format section has nonzero alignment padding")
    return unpack_qsrt_format_table(payload[:FORMAT_TABLE_BYTES])


def expand_qsrt_hot_metadata(
    formats: Sequence[ExpertFormatLike],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand disk codes to uint8 schedules and uint16 tier-local slots."""

    codes = pack_qsrt_format_table(formats)
    # Validate before deriving slots, including codes supplied through strings.
    unpack_qsrt_format_table(codes)
    slots = torch.empty(EXPERTS_PER_LAYER, dtype=torch.int16)
    compressed_slot = 0
    kept_slot = 0
    for expert, code in enumerate(codes.tolist()):
        if code == FORMAT_X4T:
            slots[expert] = kept_slot
            kept_slot += 1
        else:
            slots[expert] = compressed_slot
            compressed_slot += 1
    return codes, slots


def _validate_scale_vector(
    scale: torch.Tensor, length: int, name: str, device: torch.device | None = None
) -> None:
    if scale.dtype != torch.float16:
        raise TypeError(f"{name} must use torch.float16")
    if scale.ndim != 1 or scale.numel() != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and scale.device != device:
        raise ValueError(f"{name} must be on {device}")
    if not bool(torch.all(torch.isfinite(scale))):
        raise ValueError(f"{name} contains non-finite values")


def pack_qsrt_shared_scale_section(
    w1_suh: torch.Tensor, w3_suh: torch.Tensor, w2_svh: torch.Tensor
) -> torch.Tensor:
    """Pack the three layer-shared hidden-side FP16 vectors and zero padding."""

    scales = (w1_suh, w3_suh, w2_svh)
    device = scales[0].device
    for name, scale in zip(SHARED_SCALE_ORDER, scales, strict=True):
        _validate_scale_vector(scale, LATENT_CHANNELS, name, device)
    section = torch.zeros(
        SHARED_SCALE_SECTION_BYTES, dtype=torch.uint8, device=device
    )
    cursor = 0
    for scale in scales:
        raw = scale.view(torch.uint8)
        section[cursor : cursor + raw.numel()].copy_(raw)
        cursor += raw.numel()
    if cursor != SHARED_SCALE_BYTES:
        raise AssertionError("shared scale byte accounting drifted")
    return section


def unpack_qsrt_shared_scale_section(
    payload: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if payload.dtype != torch.uint8:
        raise TypeError("shared scale section must use torch.uint8")
    if payload.ndim != 1 or payload.numel() != SHARED_SCALE_SECTION_BYTES:
        raise ValueError(
            "shared scale section must contain exactly "
            f"{SHARED_SCALE_SECTION_BYTES} bytes"
        )
    if bool(torch.any(payload[SHARED_SCALE_BYTES:] != 0)):
        raise ValueError("shared scale section has nonzero alignment padding")
    values = []
    cursor = 0
    vector_bytes = LATENT_CHANNELS * SCALE_BYTES
    for name in SHARED_SCALE_ORDER:
        vector = (
            payload[cursor : cursor + vector_bytes]
            .clone()
            .view(torch.float16)
            .contiguous()
        )
        _validate_scale_vector(vector, LATENT_CHANNELS, name)
        values.append(vector)
        cursor += vector_bytes
    return tuple(values)  # type: ignore[return-value]


def resolve_mode(mode: ModeSpec | str | int) -> ModeSpec:
    if isinstance(mode, ModeSpec):
        return mode
    if isinstance(mode, str):
        try:
            return _MODES_BY_NAME[mode.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown QSRT mode: {mode}") from exc
    if isinstance(mode, int) and not isinstance(mode, bool):
        try:
            return _MODES_BY_ID[mode]
        except KeyError as exc:
            raise ValueError(f"unknown QSRT mode ID: {mode}") from exc
    raise TypeError("mode must be a ModeSpec, name, or integer ID")


def matrix_rate_axis(matrix: str) -> RateAxis:
    try:
        return MATRIX_RATE_AXIS[matrix]
    except KeyError as exc:
        raise ValueError(f"unknown expert matrix: {matrix}") from exc


def record_contexts(mode: ModeSpec | str | int) -> tuple[int, ...]:
    """Return physical record contexts, paired low/high then moving inward."""

    spec = resolve_mode(mode)
    contexts: list[int] = []
    for low_context in range(spec.context_count // 2):
        high_context = spec.context_count - 1 - low_context
        for _ in range(spec.records_per_context):
            contexts.extend((low_context, high_context))
    if len(contexts) != RECORDS_PER_EXPERT:
        raise AssertionError("mode construction did not cover 24 records")
    return tuple(contexts)


def record_bits(mode: ModeSpec | str | int) -> tuple[int, ...]:
    spec = resolve_mode(mode)
    return tuple(spec.context_bits[context] for context in record_contexts(spec))


def pair_bits(mode: ModeSpec | str | int) -> tuple[tuple[int, int], ...]:
    records = record_bits(mode)
    pairs = tuple(zip(records[0::2], records[1::2], strict=True))
    if len(pairs) != PAIRS_PER_EXPERT or any(sum(pair) != 6 for pair in pairs):
        raise ValueError("every QSRT record pair must have a six-bit rate sum")
    return pairs


def pair_kinds(mode: ModeSpec | str | int) -> tuple[PairKind, ...]:
    kinds: list[PairKind] = []
    for pair in pair_bits(mode):
        if pair == (2, 4):
            kinds.append("P24")
        elif pair == (3, 3):
            kinds.append("P33")
        else:
            raise ValueError(f"unsupported QSRT pair schedule: {pair}")
    return tuple(kinds)


def _validate_block_contexts(
    block_contexts: torch.Tensor, mode: ModeSpec
) -> None:
    expected_groups = INTERMEDIATE_CHANNELS // CONTEXT_GROUP_CHANNELS
    if block_contexts.ndim != 1 or block_contexts.numel() != expected_groups:
        raise ValueError(
            f"block_contexts must contain {expected_groups} four-channel groups"
        )
    if block_contexts.dtype == torch.bool or block_contexts.is_floating_point():
        raise TypeError("block_contexts must use an integer dtype")
    if block_contexts.numel() == 0:
        raise ValueError("block_contexts must not be empty")
    if int(block_contexts.min()) < 0 or int(block_contexts.max()) >= mode.context_count:
        raise ValueError("block_contexts contains an out-of-range context ID")
    counts = torch.bincount(
        block_contexts.to(dtype=torch.long), minlength=mode.context_count
    )
    expected = expected_groups // mode.context_count
    if counts.tolist() != [expected] * mode.context_count:
        raise ValueError("every context must have equal population")


def importance_group_order(
    block_contexts: torch.Tensor, mode: ModeSpec | str | int
) -> torch.Tensor:
    """Return stable low-to-high context order for the LDLQ recursion."""

    spec = resolve_mode(mode)
    _validate_block_contexts(block_contexts, spec)
    return torch.argsort(block_contexts, stable=True)


def storage_group_order(
    block_contexts: torch.Tensor, mode: ModeSpec | str | int
) -> torch.Tensor:
    """Return the physical group order forming canonical P24/P33 pairs."""

    spec = resolve_mode(mode)
    _validate_block_contexts(block_contexts, spec)
    groups_per_record = RECORD_CHANNELS // CONTEXT_GROUP_CHANNELS
    by_context = []
    for context in range(spec.context_count):
        indices = torch.nonzero(block_contexts == context).flatten()
        by_context.append(indices.reshape(-1, groups_per_record))

    chunks = []
    for low_context in range(spec.context_count // 2):
        high_context = spec.context_count - 1 - low_context
        low_records = by_context[low_context]
        high_records = by_context[high_context]
        if low_records.shape != high_records.shape:
            raise ValueError("paired contexts have different record populations")
        for record in range(low_records.shape[0]):
            chunks.extend((low_records[record], high_records[record]))
    return torch.cat(chunks)


def rate_transfer_record_order(
    donor_records: Sequence[int],
    recipient_records: Sequence[int],
    *,
    middle_records: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Build a low-to-high record order for an arbitrary equal-rate exchange.

    Record IDs refer to an existing 24-record partition, usually the
    activation-ranked partition produced by calibration.  Donors occupy the
    K2 end of an ``R_r`` schedule and recipients occupy the K4 end.  The
    recipient suffix is reversed so ``donor_records[i]`` and
    ``recipient_records[i]`` become one physical P24 pair after the canonical
    balanced-atom repack.

    This order can be baked into the common w1/w3-row and w2-column
    permutation.  Therefore an arbitrary *shared* donor/recipient map does
    not require a serving-time per-record map or a different payload format;
    only the offline selector changes.
    """

    donors = tuple(donor_records)
    recipients = tuple(recipient_records)
    if len(donors) != len(recipients):
        raise ValueError("donor and recipient record counts must match")
    if len(donors) > PAIRS_PER_EXPERT:
        raise ValueError(f"at most {PAIRS_PER_EXPERT} rate exchanges are allowed")

    selected = donors + recipients
    if any(
        isinstance(record, bool)
        or not isinstance(record, int)
        or not 0 <= record < RECORDS_PER_EXPERT
        for record in selected
    ):
        raise ValueError(
            f"record IDs must be integers in 0..{RECORDS_PER_EXPERT - 1}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("donor and recipient records must be unique and disjoint")

    remaining = tuple(
        record for record in range(RECORDS_PER_EXPERT) if record not in set(selected)
    )
    if middle_records is None:
        middle = remaining
    else:
        middle = tuple(middle_records)
        if len(middle) != len(remaining) or set(middle) != set(remaining):
            raise ValueError(
                "middle_records must be a permutation of all unselected records"
            )
    return donors + middle + tuple(reversed(recipients))


def relabel_block_contexts_for_record_order(
    block_contexts: torch.Tensor,
    record_order: Sequence[int],
) -> torch.Tensor:
    """Relabel a 24-record partition to follow ``record_order``.

    ``record_order[new_record]`` is the old record ID.  Sorting the returned
    labels with :func:`importance_group_order` yields the requested record
    order while preserving every complete 128-channel record.
    """

    _validate_block_contexts(block_contexts, R0)
    order = tuple(record_order)
    if len(order) != RECORDS_PER_EXPERT:
        raise ValueError(f"record_order must contain {RECORDS_PER_EXPERT} records")
    if any(
        isinstance(record, bool)
        or not isinstance(record, int)
        or not 0 <= record < RECORDS_PER_EXPERT
        for record in order
    ) or len(set(order)) != RECORDS_PER_EXPERT:
        raise ValueError("record_order must be a permutation of all record IDs")
    inverse = torch.empty(
        RECORDS_PER_EXPERT, dtype=torch.long, device=block_contexts.device
    )
    inverse[torch.tensor(order, dtype=torch.long, device=block_contexts.device)] = (
        torch.arange(RECORDS_PER_EXPERT, device=block_contexts.device)
    )
    return inverse[block_contexts.to(dtype=torch.long)]


def expand_group_order(group_order: torch.Tensor) -> torch.Tensor:
    """Expand four-channel group IDs into a new-index -> old-index permutation."""

    expected_groups = INTERMEDIATE_CHANNELS // CONTEXT_GROUP_CHANNELS
    if group_order.ndim != 1 or group_order.numel() != expected_groups:
        raise ValueError(f"group_order must contain {expected_groups} groups")
    _validate_permutation(group_order, expected_groups, "group_order")
    offsets = torch.arange(CONTEXT_GROUP_CHANNELS, device=group_order.device)
    return (
        group_order.to(dtype=torch.long)[:, None] * CONTEXT_GROUP_CHANNELS
        + offsets[None]
    ).flatten()


def _validate_permutation(
    permutation: torch.Tensor, length: int, name: str = "permutation"
) -> None:
    if permutation.ndim != 1 or permutation.numel() != length:
        raise ValueError(f"{name} must contain exactly {length} entries")
    if permutation.dtype == torch.bool or permutation.is_floating_point():
        raise TypeError(f"{name} must use an integer dtype")
    values = permutation.to(dtype=torch.long)
    expected = torch.arange(length, device=values.device)
    if not torch.equal(torch.sort(values).values, expected):
        raise ValueError(f"{name} must be a bijection over 0..{length - 1}")


def invert_permutation(permutation: torch.Tensor) -> torch.Tensor:
    _validate_permutation(permutation, permutation.numel())
    return torch.argsort(permutation.to(dtype=torch.long))


def coupled_intermediate_permutation(
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    permutation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one exact SiTU-neuron permutation to ordinary weight matrices.

    ``permutation[new_channel]`` is the corresponding old channel.  The
    returned matrices satisfy ``w1' = w1[P]``, ``w3' = w3[P]``, and
    ``w2' = w2[:, P]``.
    """

    if w1.ndim != 2 or w3.ndim != 2 or w2.ndim != 2:
        raise ValueError("expert weights must all be two-dimensional")
    if w1.shape != w3.shape:
        raise ValueError("w1 and w3 must have identical shapes")
    intermediate, hidden = w1.shape
    if tuple(w2.shape) != (hidden, intermediate):
        raise ValueError("w2 must have shape [hidden, intermediate]")
    _validate_permutation(permutation, intermediate)
    row_index = permutation.to(device=w1.device, dtype=torch.long)
    w3_index = permutation.to(device=w3.device, dtype=torch.long)
    column_index = permutation.to(device=w2.device, dtype=torch.long)
    return (
        w1.index_select(0, row_index).contiguous(),
        w3.index_select(0, w3_index).contiguous(),
        w2.index_select(1, column_index).contiguous(),
    )


def record_repack_order(
    source_group_order: torch.Tensor, destination_group_order: torch.Tensor
) -> torch.Tensor:
    """Map destination records to source records without splitting a record."""

    expected_groups = INTERMEDIATE_CHANNELS // CONTEXT_GROUP_CHANNELS
    _validate_permutation(source_group_order, expected_groups, "source_group_order")
    _validate_permutation(
        destination_group_order, expected_groups, "destination_group_order"
    )
    groups_per_record = RECORD_CHANNELS // CONTEXT_GROUP_CHANNELS
    source_records = source_group_order.cpu().reshape(-1, groups_per_record)
    destination_records = destination_group_order.cpu().reshape(-1, groups_per_record)
    source_lookup = {
        tuple(record.tolist()): index for index, record in enumerate(source_records)
    }
    if len(source_lookup) != RECORDS_PER_EXPERT:
        raise ValueError("source order contains duplicate or malformed records")
    try:
        order = [
            source_lookup[tuple(record.tolist())]
            for record in destination_records
        ]
    except KeyError as exc:
        raise ValueError("destination order splits or changes a source record") from exc
    return torch.tensor(order, dtype=torch.long, device=source_group_order.device)


def expand_record_order(record_order: torch.Tensor) -> torch.Tensor:
    """Expand destination-record -> source-record mapping to channel indices."""

    _validate_permutation(record_order, RECORDS_PER_EXPERT, "record_order")
    offsets = torch.arange(RECORD_CHANNELS, device=record_order.device)
    return (
        record_order.to(dtype=torch.long)[:, None] * RECORD_CHANNELS + offsets[None]
    ).flatten()


def repack_rate_axis(
    tensor: torch.Tensor, record_order: torch.Tensor, axis: int
) -> torch.Tensor:
    """Move complete 128-channel blocks on a tensor's logical rate axis."""

    if tensor.ndim == 0:
        raise ValueError("cannot repack a scalar tensor")
    normalized_axis = axis % tensor.ndim
    if tensor.shape[normalized_axis] != INTERMEDIATE_CHANNELS:
        raise ValueError("rate axis must contain exactly 3072 channels")
    channels = expand_record_order(record_order).to(device=tensor.device)
    return tensor.index_select(normalized_axis, channels).contiguous()


def repack_encoded_records(
    encoded: torch.Tensor, record_order: torch.Tensor, rate_axis: RateAxis
) -> torch.Tensor:
    """Reorder complete 128-channel records in ``[K/16, N/16, 256]`` states."""

    _validate_encoded_shape(encoded, rate_axis)
    _validate_permutation(record_order, RECORDS_PER_EXPERT, "record_order")
    order = record_order.to(device=encoded.device, dtype=torch.long)
    if rate_axis == "k":
        records = encoded.reshape(
            RECORDS_PER_EXPERT,
            TILES_PER_RECORD_AXIS,
            encoded.shape[1],
            TILE_VALUES,
        )
        return records.index_select(0, order).reshape_as(encoded).contiguous()
    records = (
        encoded.reshape(
            encoded.shape[0],
            RECORDS_PER_EXPERT,
            TILES_PER_RECORD_AXIS,
            TILE_VALUES,
        )
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    reordered = records.index_select(0, order)
    return (
        reordered.permute(1, 0, 2, 3).contiguous().reshape_as(encoded).contiguous()
    )


def _validate_k(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4):
        raise ValueError("QSRT supports K=2, K=3, or K=4")


def pack_trellis_edges(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack low K bits with the native EXL ``pack_trellis`` word ordering."""

    _validate_k(bits)
    if indices.ndim < 1 or indices.shape[-1] != TILE_VALUES:
        raise ValueError("indices must have a final dimension of 256")
    if indices.dtype == torch.bool or indices.is_floating_point():
        raise TypeError("indices must use an integer dtype")
    values = indices.to(dtype=torch.int64) & ((1 << bits) - 1)
    spans = values.reshape(*values.shape[:-1], TILE_CHANNELS, TILE_CHANNELS)
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=values.device)
    bitstream = ((spans[..., None] >> symbol_shifts) & 1).reshape(
        *values.shape[:-1], TILE_CHANNELS, bits * TILE_CHANNELS
    )
    word_shifts = torch.arange(15, -1, -1, device=values.device)
    words = (
        bitstream.reshape(*values.shape[:-1], TILE_CHANNELS, bits, 16)
        << word_shifts
    ).sum(dim=-1)
    flat = words.reshape(*values.shape[:-1], TILE_CHANNELS * bits)
    # Native pack.cu applies SWAP16: adjacent uint16 words exchange places.
    swapped = flat.reshape(*flat.shape[:-1], -1, 2).flip(-1).reshape(flat.shape)
    return swapped.to(dtype=torch.int16).contiguous()


def unpack_trellis_edges(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Invert :func:`pack_trellis_edges` to the stored K-bit edge symbols."""

    _validate_k(bits)
    expected_words = TILE_CHANNELS * bits
    if packed.ndim < 1 or packed.shape[-1] != expected_words:
        raise ValueError(f"packed tiles must end in {expected_words} int16 words")
    if packed.dtype != torch.int16:
        raise TypeError("packed trellis words must use torch.int16")
    words = packed.to(dtype=torch.int64) & 0xFFFF
    words = words.reshape(*words.shape[:-1], -1, 2).flip(-1).reshape(words.shape)
    words = words.reshape(*words.shape[:-1], TILE_CHANNELS, bits)
    word_shifts = torch.arange(15, -1, -1, device=words.device)
    bitstream = ((words[..., None] >> word_shifts) & 1).reshape(
        *words.shape[:-2], TILE_CHANNELS, bits * TILE_CHANNELS
    )
    symbol_bits = bitstream.reshape(
        *words.shape[:-2], TILE_CHANNELS, TILE_CHANNELS, bits
    )
    symbol_shifts = torch.arange(bits - 1, -1, -1, device=words.device)
    values = (symbol_bits << symbol_shifts).sum(dim=-1)
    return values.reshape(*words.shape[:-2], TILE_VALUES).to(torch.int16).contiguous()


def unpack_trellis_states(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack native edge words and reconstruct every cyclic 16-bit state."""

    return reconstruct_trellis_states(unpack_trellis_edges(packed, bits), bits)


@dataclass(frozen=True)
class QSRTTrellisDescriptor:
    """TP-independent description of one complete packed expert matrix."""

    mode_id: int
    rate_axis: RateAxis
    k_tiles: int
    n_tiles: int
    schema: str = SCHEMA
    intermediate_channels: int = INTERMEDIATE_CHANNELS
    record_channels: int = RECORD_CHANNELS

    def __post_init__(self) -> None:
        # Candidate payloads are allocation-independent and stay in logical
        # pair order. Physical ownership is applied only by the atom sharder.
        if self.schema not in LOGICAL_CANDIDATE_SCHEMAS:
            raise ValueError(f"unsupported QSRT schema: {self.schema}")
        if self.intermediate_channels != INTERMEDIATE_CHANNELS:
            raise ValueError("Kimi-K3 QSRT requires a 3072-channel intermediate axis")
        if self.record_channels != RECORD_CHANNELS:
            raise ValueError("Kimi-K3 QSRT requires 128-channel records")
        resolve_mode(self.mode_id)
        if self.rate_axis not in ("k", "n"):
            raise ValueError("rate_axis must be 'k' or 'n'")
        if self.k_tiles <= 0 or self.n_tiles <= 0:
            raise ValueError("tile dimensions must be positive")
        rate_tiles = self.k_tiles if self.rate_axis == "k" else self.n_tiles
        if rate_tiles * TILE_CHANNELS != INTERMEDIATE_CHANNELS:
            raise ValueError("rate axis must contain exactly 3072 channels")
        orthogonal_tiles = self.n_tiles if self.rate_axis == "k" else self.k_tiles
        if orthogonal_tiles * TILE_CHANNELS != LATENT_CHANNELS:
            raise ValueError("orthogonal axis must contain exactly 3584 channels")

    @property
    def mode(self) -> ModeSpec:
        return resolve_mode(self.mode_id)

    @property
    def orthogonal_tiles(self) -> int:
        return self.n_tiles if self.rate_axis == "k" else self.k_tiles

    @property
    def tiles_per_record(self) -> int:
        return TILES_PER_RECORD_AXIS * self.orthogonal_tiles

    @property
    def words_per_pair(self) -> int:
        return self.tiles_per_record * TILE_CHANNELS * 6

    @property
    def payload_words(self) -> int:
        return PAIRS_PER_EXPERT * self.words_per_pair

    @property
    def payload_bytes(self) -> int:
        return self.payload_words * torch.int16.itemsize

    @property
    def weight_count(self) -> int:
        return self.k_tiles * self.n_tiles * TILE_VALUES

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "mode_id": self.mode_id,
            "mode": self.mode.name,
            "rate_axis": self.rate_axis,
            "k_tiles": self.k_tiles,
            "n_tiles": self.n_tiles,
            "intermediate_channels": self.intermediate_channels,
            "record_channels": self.record_channels,
            "pair_channels": PAIR_CHANNELS,
            "record_bits": list(record_bits(self.mode)),
            "pair_kinds": list(pair_kinds(self.mode)),
            "words_per_pair": self.words_per_pair,
            "payload_words": self.payload_words,
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True)
class PackedQSRTTrellis:
    descriptor: QSRTTrellisDescriptor
    payload: torch.Tensor

    def validate(self) -> None:
        if self.payload.dtype != torch.int16:
            raise TypeError("QSRT trellis payload must use torch.int16")
        if self.payload.ndim != 1:
            raise ValueError("QSRT trellis payload must be one-dimensional")
        if self.payload.numel() != self.descriptor.payload_words:
            raise ValueError(
                "QSRT trellis payload length does not match its descriptor"
            )


def _validate_encoded_shape(encoded: torch.Tensor, rate_axis: RateAxis) -> None:
    if encoded.ndim != 3 or encoded.shape[-1] != TILE_VALUES:
        raise ValueError("encoded states must have shape [K/16, N/16, 256]")
    if encoded.dtype == torch.bool or encoded.is_floating_point():
        raise TypeError("encoded states must use an integer dtype")
    if rate_axis not in ("k", "n"):
        raise ValueError("rate_axis must be 'k' or 'n'")
    rate_tiles = encoded.shape[0] if rate_axis == "k" else encoded.shape[1]
    if rate_tiles * TILE_CHANNELS != INTERMEDIATE_CHANNELS:
        raise ValueError("encoded rate axis must contain exactly 3072 channels")


def _record_tiles(
    encoded: torch.Tensor, record: int, rate_axis: RateAxis
) -> torch.Tensor:
    begin = record * TILES_PER_RECORD_AXIS
    if rate_axis == "k":
        return encoded.narrow(0, begin, TILES_PER_RECORD_AXIS).reshape(-1, TILE_VALUES)
    return encoded.narrow(1, begin, TILES_PER_RECORD_AXIS).reshape(-1, TILE_VALUES)


def pack_qsrt_trellis(
    encoded: torch.Tensor,
    mode: ModeSpec | str | int,
    *,
    rate_axis: RateAxis,
    schema: str = SCHEMA,
    validate_states: bool = True,
) -> PackedQSRTTrellis:
    """Pack a complete matrix into fixed-size balanced record pairs."""

    _validate_encoded_shape(encoded, rate_axis)
    spec = resolve_mode(mode)
    descriptor = QSRTTrellisDescriptor(
        mode_id=spec.mode_id,
        rate_axis=rate_axis,
        k_tiles=encoded.shape[0],
        n_tiles=encoded.shape[1],
        schema=schema,
    )
    pieces = []
    for record, bits in enumerate(record_bits(spec)):
        record_states = _record_tiles(encoded, record, rate_axis)
        record_payload = pack_trellis_edges(record_states, bits)
        if validate_states and not torch.equal(
            unpack_trellis_states(record_payload, bits), record_states.to(torch.int16)
        ):
            raise ValueError("encoded input is not a closed cyclic trellis path")
        pieces.append(record_payload)
    payload = torch.cat([piece.reshape(-1) for piece in pieces]).contiguous()
    packed = PackedQSRTTrellis(descriptor, payload)
    packed.validate()
    return packed


def unpack_qsrt_trellis_edges(packed: PackedQSRTTrellis) -> torch.Tensor:
    """Unpack a complete QSRT payload to physical-order K-bit edge symbols."""

    packed.validate()
    descriptor = packed.descriptor
    output = torch.empty(
        (descriptor.k_tiles, descriptor.n_tiles, TILE_VALUES),
        dtype=torch.int16,
        device=packed.payload.device,
    )
    cursor = 0
    for record, bits in enumerate(record_bits(descriptor.mode)):
        word_count = descriptor.tiles_per_record * TILE_CHANNELS * bits
        words = packed.payload.narrow(0, cursor, word_count).reshape(
            descriptor.tiles_per_record, TILE_CHANNELS * bits
        )
        tiles = unpack_trellis_edges(words, bits)
        begin = record * TILES_PER_RECORD_AXIS
        if descriptor.rate_axis == "k":
            output.narrow(0, begin, TILES_PER_RECORD_AXIS).copy_(
                tiles.reshape(
                    TILES_PER_RECORD_AXIS, descriptor.n_tiles, TILE_VALUES
                )
            )
        else:
            output.narrow(1, begin, TILES_PER_RECORD_AXIS).copy_(
                tiles.reshape(
                    descriptor.k_tiles, TILES_PER_RECORD_AXIS, TILE_VALUES
                )
            )
        cursor += word_count
    if cursor != packed.payload.numel():
        raise ValueError("QSRT trellis payload has trailing data")
    return output


def unpack_qsrt_trellis_states(packed: PackedQSRTTrellis) -> torch.Tensor:
    """Unpack a complete QSRT payload to its physical-order 16-bit states."""

    edges = unpack_qsrt_trellis_edges(packed)
    descriptor = packed.descriptor
    output = torch.empty_like(edges)
    rates = record_bits(descriptor.mode)
    # The recurrence is elementwise over every dimension except the trailing
    # 256-symbol path.  Group equal-K records so each matrix performs at most
    # three Python-level recurrences instead of one per 128-channel record.
    for bits in sorted(set(rates)):
        records = [
            record for record, record_bits_value in enumerate(rates)
            if record_bits_value == bits
        ]
        if descriptor.rate_axis == "k":
            grouped_edges = torch.stack(
                [
                    edges.narrow(
                        0,
                        record * TILES_PER_RECORD_AXIS,
                        TILES_PER_RECORD_AXIS,
                    )
                    for record in records
                ]
            )
            grouped_states = reconstruct_trellis_states(grouped_edges, bits)
            for index, record in enumerate(records):
                output.narrow(
                    0,
                    record * TILES_PER_RECORD_AXIS,
                    TILES_PER_RECORD_AXIS,
                ).copy_(grouped_states[index])
        else:
            grouped_edges = torch.stack(
                [
                    edges.narrow(
                        1,
                        record * TILES_PER_RECORD_AXIS,
                        TILES_PER_RECORD_AXIS,
                    )
                    for record in records
                ]
            )
            grouped_states = reconstruct_trellis_states(grouped_edges, bits)
            for index, record in enumerate(records):
                output.narrow(
                    1,
                    record * TILES_PER_RECORD_AXIS,
                    TILES_PER_RECORD_AXIS,
                ).copy_(grouped_states[index])
    return output


def decode_qsrt_exl3_weight(
    packed: PackedQSRTTrellis,
    suh: torch.Tensor,
    svh: torch.Tensor,
    *,
    codebook: str = CODEBOOK_SQG_XOR_CHEB_T12,
) -> torch.Tensor:
    """Decode a complete QSRT payload to its physical EXL-oriented weight."""

    packed.validate()
    _validate_scale_vector(
        suh,
        packed.descriptor.k_tiles * TILE_CHANNELS,
        "suh",
        packed.payload.device,
    )
    _validate_scale_vector(
        svh,
        packed.descriptor.n_tiles * TILE_CHANNELS,
        "svh",
        packed.payload.device,
    )
    tile_bits = tuple(
        bits
        for rate in record_bits(packed.descriptor.mode)
        for bits in (rate,) * TILES_PER_RECORD_AXIS
    )
    return decode_qsrt_weight(
        unpack_qsrt_trellis_states(packed),
        suh,
        svh,
        rate_axis=packed.descriptor.rate_axis,
        tile_bits=tile_bits,
        codebook=codebook,
    )
