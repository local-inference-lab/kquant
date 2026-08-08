"""Geometry-neutral logical record planning for offline QSRT payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

MatrixName = Literal["w1", "w3", "w2"]
RateAxis = Literal["rows", "columns"]


@dataclass(frozen=True)
class LogicalQSRTGeometry:
    """Logical expert dimensions and the width of one rate record."""

    hidden_channels: int
    intermediate_channels: int
    record_channels: int

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_channels", self.hidden_channels),
            ("intermediate_channels", self.intermediate_channels),
            ("record_channels", self.record_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.intermediate_channels % self.record_channels:
            raise ValueError(
                "record_channels must exactly divide intermediate_channels"
            )

    @property
    def record_count(self) -> int:
        return self.intermediate_channels // self.record_channels


@dataclass(frozen=True)
class LogicalRateMode:
    """Canonical K2/K3/K4 rate transfer over a logical record axis."""

    record_count: int
    transfer: int

    def __post_init__(self) -> None:
        if isinstance(self.record_count, bool) or not isinstance(
            self.record_count, int
        ):
            raise TypeError("record_count must be an integer")
        if self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if isinstance(self.transfer, bool) or not isinstance(self.transfer, int):
            raise TypeError("transfer must be an integer")
        if not 0 <= self.transfer <= self.record_count // 2:
            raise ValueError("transfer must be between zero and half the record count")

    @property
    def name(self) -> str:
        return f"R{self.transfer}"

    @property
    def record_counts(self) -> tuple[int, int, int]:
        """Numbers of K2, K3, and K4 records, in that order."""

        return (
            self.transfer,
            self.record_count - 2 * self.transfer,
            self.transfer,
        )

    @property
    def record_bits(self) -> tuple[int, ...]:
        """Canonical low-to-high record rates: K2, then K3, then K4."""

        k2, k3, k4 = self.record_counts
        return (2,) * k2 + (3,) * k3 + (4,) * k4


def rate_transfer_mode(geometry: LogicalQSRTGeometry, transfer: int) -> LogicalRateMode:
    """Derive the canonical rate mode for ``transfer`` K2/K4 exchanges."""

    if not isinstance(geometry, LogicalQSRTGeometry):
        raise TypeError("geometry must be a LogicalQSRTGeometry")
    return LogicalRateMode(geometry.record_count, transfer)


def paired_record_order(mode: LogicalRateMode) -> tuple[int, ...]:
    """Interleave low/high records into fixed six-bit physical pairs."""

    if not isinstance(mode, LogicalRateMode):
        raise TypeError("mode must be a LogicalRateMode")
    if mode.record_count % 2:
        raise ValueError("paired storage requires an even record count")
    return tuple(
        record
        for low in range(mode.record_count // 2)
        for record in (low, mode.record_count - 1 - low)
    )


def paired_record_bits(mode: LogicalRateMode) -> tuple[int, ...]:
    """Return physical record rates after low/high pair interleaving."""

    return tuple(mode.record_bits[record] for record in paired_record_order(mode))


def pair_kinds(mode: LogicalRateMode) -> tuple[str, ...]:
    """Return one ``P33`` or ``P24`` descriptor per fixed-size pair."""

    bits = paired_record_bits(mode)
    pairs = tuple(zip(bits[0::2], bits[1::2], strict=True))
    if any(pair not in ((3, 3), (2, 4)) for pair in pairs):
        raise ValueError("paired storage supports only K3/K3 and K2/K4")
    return tuple("P33" if pair == (3, 3) else "P24" for pair in pairs)


FRUIT_QSRT_GEOMETRY = LogicalQSRTGeometry(
    hidden_channels=1024,
    intermediate_channels=512,
    record_channels=128,
)
R0 = rate_transfer_mode(FRUIT_QSRT_GEOMETRY, 0)
R1 = rate_transfer_mode(FRUIT_QSRT_GEOMETRY, 1)
R2 = rate_transfer_mode(FRUIT_QSRT_GEOMETRY, 2)


def fruit_rate_modes() -> tuple[LogicalRateMode, LogicalRateMode, LogicalRateMode]:
    """Return the complete supported Fruit four-record logical mode set."""

    return R0, R1, R2


def matrix_rate_axis(matrix: MatrixName) -> RateAxis:
    """Return the logical intermediate-channel axis for an expert matrix."""

    if matrix in ("w1", "w3"):
        return "rows"
    if matrix == "w2":
        return "columns"
    raise ValueError(f"unknown expert matrix: {matrix}")


def matrix_shape(geometry: LogicalQSRTGeometry, matrix: MatrixName) -> tuple[int, int]:
    """Return the source-oriented ``[out, in]`` shape of an expert matrix."""

    if not isinstance(geometry, LogicalQSRTGeometry):
        raise TypeError("geometry must be a LogicalQSRTGeometry")
    axis = matrix_rate_axis(matrix)
    if axis == "rows":
        return geometry.intermediate_channels, geometry.hidden_channels
    return geometry.hidden_channels, geometry.intermediate_channels


def matrix_payload_bits(
    geometry: LogicalQSRTGeometry,
    matrix: MatrixName,
    mode: LogicalRateMode,
) -> int:
    """Count the exact trellis path bits for one logical expert matrix."""

    shape = matrix_shape(geometry, matrix)
    if not isinstance(mode, LogicalRateMode):
        raise TypeError("mode must be a LogicalRateMode")
    if mode.record_count != geometry.record_count:
        raise ValueError("mode record count does not match geometry")
    axis = 0 if matrix_rate_axis(matrix) == "rows" else 1
    values_per_record = geometry.record_channels * shape[1 - axis]
    return values_per_record * sum(mode.record_bits)


def matrix_payload_bytes(
    geometry: LogicalQSRTGeometry,
    matrix: MatrixName,
    mode: LogicalRateMode,
) -> int:
    """Count exact whole bytes, rejecting a geometry with fractional bytes."""

    bits = matrix_payload_bits(geometry, matrix, mode)
    if bits % 8:
        raise ValueError("matrix payload is not an exact whole number of bytes")
    return bits // 8


def _record_order_tensor(
    record_order: Sequence[int] | torch.Tensor,
    *,
    record_count: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(record_order, torch.Tensor):
        if record_order.ndim != 1:
            raise ValueError("record_order must be one-dimensional")
        if (
            record_order.dtype == torch.bool
            or record_order.is_floating_point()
            or record_order.is_complex()
        ):
            raise TypeError("record_order must use an integer dtype")
        order = record_order.to(device=device, dtype=torch.long)
    else:
        values = tuple(record_order)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("record_order entries must be integers")
        order = torch.tensor(values, dtype=torch.long, device=device)

    if order.numel() != record_count:
        raise ValueError(f"record_order must contain exactly {record_count} entries")
    expected = torch.arange(record_count, dtype=torch.long, device=device)
    if not torch.equal(torch.sort(order).values, expected):
        raise ValueError("record_order must be a permutation of all record IDs")
    return order


def _repack_record_axis(
    tensor: torch.Tensor,
    record_order: Sequence[int] | torch.Tensor,
    *,
    axis: int,
    record_channels: int,
    inverse: bool,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.ndim == 0:
        raise ValueError("cannot repack a scalar tensor")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    if not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError("axis is out of range for tensor")
    if isinstance(record_channels, bool) or not isinstance(record_channels, int):
        raise TypeError("record_channels must be an integer")
    if record_channels <= 0:
        raise ValueError("record_channels must be positive")

    normalized_axis = axis % tensor.ndim
    axis_channels = tensor.shape[normalized_axis]
    if axis_channels == 0 or axis_channels % record_channels:
        raise ValueError("record_channels must exactly divide the nonempty record axis")
    record_count = axis_channels // record_channels
    order = _record_order_tensor(
        record_order, record_count=record_count, device=tensor.device
    )
    if inverse:
        order = torch.argsort(order)
    offsets = torch.arange(record_channels, dtype=torch.long, device=tensor.device)
    channels = (order[:, None] * record_channels + offsets[None]).flatten()
    return tensor.index_select(normalized_axis, channels).contiguous()


def repack_record_axis(
    tensor: torch.Tensor,
    record_order: Sequence[int] | torch.Tensor,
    *,
    axis: int,
    record_channels: int,
) -> torch.Tensor:
    """Reorder complete records; order maps each destination record to source."""

    return _repack_record_axis(
        tensor,
        record_order,
        axis=axis,
        record_channels=record_channels,
        inverse=False,
    )


def inverse_repack_record_axis(
    tensor: torch.Tensor,
    record_order: Sequence[int] | torch.Tensor,
    *,
    axis: int,
    record_channels: int,
) -> torch.Tensor:
    """Invert :func:`repack_record_axis` for the same record order."""

    return _repack_record_axis(
        tensor,
        record_order,
        axis=axis,
        record_channels=record_channels,
        inverse=True,
    )
