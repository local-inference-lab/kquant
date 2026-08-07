from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from kquant.logical_qsrt import (
    FRUIT_QSRT_GEOMETRY,
    R0,
    R1,
    R2,
    LogicalQSRTGeometry,
    LogicalRateMode,
    fruit_rate_modes,
    inverse_repack_record_axis,
    matrix_payload_bits,
    matrix_payload_bytes,
    matrix_rate_axis,
    matrix_shape,
    rate_transfer_mode,
    repack_record_axis,
)


def test_fruit_geometry_and_supported_modes_are_frozen() -> None:
    assert FRUIT_QSRT_GEOMETRY == LogicalQSRTGeometry(1024, 512, 128)
    assert FRUIT_QSRT_GEOMETRY.record_count == 4
    assert fruit_rate_modes() == (R0, R1, R2)
    assert tuple(mode.name for mode in fruit_rate_modes()) == ("R0", "R1", "R2")
    assert tuple(mode.transfer for mode in fruit_rate_modes()) == (0, 1, 2)
    assert tuple(mode.record_counts for mode in fruit_rate_modes()) == (
        (0, 4, 0),
        (1, 2, 1),
        (2, 0, 2),
    )

    with pytest.raises(FrozenInstanceError):
        R1.transfer = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        FRUIT_QSRT_GEOMETRY.record_channels = 64  # type: ignore[misc]


def test_rate_modes_derive_canonical_record_bit_sequences() -> None:
    assert R0.record_bits == (3, 3, 3, 3)
    assert R1.record_bits == (2, 3, 3, 4)
    assert R2.record_bits == (2, 2, 4, 4)

    geometry = LogicalQSRTGeometry(
        hidden_channels=17,
        intermediate_channels=30,
        record_channels=5,
    )
    assert rate_transfer_mode(geometry, 2).record_bits == (2, 2, 3, 3, 4, 4)
    for mode in fruit_rate_modes():
        assert sum(mode.record_bits) == 3 * FRUIT_QSRT_GEOMETRY.record_count


def test_fruit_matrix_orientations_have_exact_constant_payload_bytes() -> None:
    expected_bits = 512 * 1024 * 3
    expected_bytes = expected_bits // 8
    assert expected_bits == 1_572_864
    assert expected_bytes == 196_608

    assert matrix_rate_axis("w1") == "rows"
    assert matrix_rate_axis("w3") == "rows"
    assert matrix_rate_axis("w2") == "columns"
    assert matrix_shape(FRUIT_QSRT_GEOMETRY, "w1") == (512, 1024)
    assert matrix_shape(FRUIT_QSRT_GEOMETRY, "w3") == (512, 1024)
    assert matrix_shape(FRUIT_QSRT_GEOMETRY, "w2") == (1024, 512)

    for mode in fruit_rate_modes():
        for matrix in ("w1", "w3", "w2"):
            assert (
                matrix_payload_bits(FRUIT_QSRT_GEOMETRY, matrix, mode) == expected_bits
            )
            assert (
                matrix_payload_bytes(FRUIT_QSRT_GEOMETRY, matrix, mode)
                == expected_bytes
            )


def test_logical_geometry_and_rate_modes_reject_malformed_values() -> None:
    with pytest.raises(TypeError, match="hidden_channels must be an integer"):
        LogicalQSRTGeometry(True, 512, 128)
    with pytest.raises(ValueError, match="hidden_channels must be positive"):
        LogicalQSRTGeometry(0, 512, 128)
    with pytest.raises(ValueError, match="exactly divide"):
        LogicalQSRTGeometry(1024, 513, 128)

    with pytest.raises(TypeError, match="record_count must be an integer"):
        LogicalRateMode(True, 0)
    with pytest.raises(ValueError, match="record_count must be positive"):
        LogicalRateMode(0, 0)
    with pytest.raises(TypeError, match="transfer must be an integer"):
        LogicalRateMode(4, True)
    with pytest.raises(ValueError, match="half the record count"):
        LogicalRateMode(4, -1)
    with pytest.raises(ValueError, match="half the record count"):
        LogicalRateMode(4, 3)
    with pytest.raises(TypeError, match="LogicalQSRTGeometry"):
        rate_transfer_mode(object(), 0)  # type: ignore[arg-type]


def test_payload_accounting_rejects_wrong_mode_matrix_and_partial_byte() -> None:
    wrong_geometry_mode = rate_transfer_mode(LogicalQSRTGeometry(8, 6, 2), 1)
    with pytest.raises(ValueError, match="record count does not match geometry"):
        matrix_payload_bits(FRUIT_QSRT_GEOMETRY, "w1", wrong_geometry_mode)
    with pytest.raises(ValueError, match="unknown expert matrix"):
        matrix_payload_bits(FRUIT_QSRT_GEOMETRY, "gate", R0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LogicalRateMode"):
        matrix_payload_bits(FRUIT_QSRT_GEOMETRY, "w1", object())  # type: ignore[arg-type]

    partial_byte_geometry = LogicalQSRTGeometry(1, 1, 1)
    partial_byte_mode = rate_transfer_mode(partial_byte_geometry, 0)
    assert matrix_payload_bits(partial_byte_geometry, "w1", partial_byte_mode) == 3
    with pytest.raises(ValueError, match="whole number of bytes"):
        matrix_payload_bytes(partial_byte_geometry, "w1", partial_byte_mode)


@pytest.mark.parametrize(
    ("matrix", "axis"),
    (("w1", 0), ("w3", 0), ("w2", 1)),
)
def test_record_axis_repack_round_trips_exact_matrix_orientations(
    matrix: str, axis: int
) -> None:
    shape = matrix_shape(FRUIT_QSRT_GEOMETRY, matrix)  # type: ignore[arg-type]
    source = torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape)
    order = torch.tensor((2, 0, 3, 1), dtype=torch.int16)

    repacked = repack_record_axis(
        source,
        order,
        axis=axis,
        record_channels=FRUIT_QSRT_GEOMETRY.record_channels,
    )
    for destination, original in enumerate(order.tolist()):
        actual = repacked.narrow(
            axis,
            destination * FRUIT_QSRT_GEOMETRY.record_channels,
            FRUIT_QSRT_GEOMETRY.record_channels,
        )
        expected = source.narrow(
            axis,
            original * FRUIT_QSRT_GEOMETRY.record_channels,
            FRUIT_QSRT_GEOMETRY.record_channels,
        )
        assert torch.equal(actual, expected)

    restored = inverse_repack_record_axis(
        repacked,
        order,
        axis=axis,
        record_channels=FRUIT_QSRT_GEOMETRY.record_channels,
    )
    assert restored.is_contiguous()
    assert torch.equal(restored, source)


def test_record_axis_repack_strictly_rejects_invalid_inputs() -> None:
    source = torch.zeros((4, 6))
    with pytest.raises(ValueError, match="permutation"):
        repack_record_axis(source, (0, 0), axis=0, record_channels=2)
    with pytest.raises(ValueError, match="exactly 2 entries"):
        repack_record_axis(source, (0,), axis=0, record_channels=2)
    with pytest.raises(TypeError, match="integer dtype"):
        repack_record_axis(source, torch.tensor((0.0, 1.0)), axis=0, record_channels=2)
    with pytest.raises(ValueError, match="one-dimensional"):
        repack_record_axis(source, torch.tensor(((0, 1),)), axis=0, record_channels=2)
    with pytest.raises(ValueError, match="axis is out of range"):
        repack_record_axis(source, (0, 1), axis=2, record_channels=2)
    with pytest.raises(ValueError, match="exactly divide"):
        repack_record_axis(source, (0, 1), axis=1, record_channels=4)
