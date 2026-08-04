from __future__ import annotations

import os

import pytest
import torch

from kquant import constants as C
from kquant.qsrt import (
    FORMAT_SECTION_BYTES,
    INTERMEDIATE_CHANNELS,
    LATENT_CHANNELS,
    LAYER_HEADER_BYTES,
    MATRIX_TRELLIS_BYTES,
    PAIR_BYTES,
    ExpertFormatSpec,
    TP12LayerHeader,
    TP12LayerLayout,
    pack_tp12_format_section,
    pack_tp12_shared_scale_section,
    tp12_logical_pair_index,
)
from kquant.pack.qsrt_slab import (
    LayerMaterializationSpec,
    TP12LayerReader,
    candidate_local_scale,
    candidate_trellis_pair,
    pwrite_exact,
    validate_materialized_layer,
)


def test_candidate_pair_and_local_scale_split_exactly_by_rank() -> None:
    words_per_pair = PAIR_BYTES // torch.int16.itemsize
    trellis = torch.zeros(
        MATRIX_TRELLIS_BYTES // torch.int16.itemsize, dtype=torch.int16
    )
    for rank in range(12):
        trellis[rank * words_per_pair] = rank + 1
    scale = torch.arange(INTERMEDIATE_CHANNELS, dtype=torch.float16)

    for rank in (0, 5, 11):
        pair = candidate_trellis_pair(trellis, rank)
        local = candidate_local_scale(scale, rank)
        assert pair.numel() == words_per_pair
        assert int(pair[0]) == rank + 1
        begin = rank * 256
        assert torch.equal(local, scale[begin : begin + 256])


def test_pwrite_exact_preserves_explicit_offsets(tmp_path) -> None:
    path = tmp_path / "offsets.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, 12)
        pwrite_exact(descriptor, b"abc", 2)
        tensor = torch.tensor([0x6465, 0x6667], dtype=torch.int16)
        pwrite_exact(descriptor, memoryview(tensor.numpy()).cast("B"), 6)
    finally:
        os.close(descriptor)

    assert path.read_bytes() == b"\x00\x00abc\x00edgf\x00\x00"


def test_sparse_qsrt_layer_reader_reassembles_compressed_payload(tmp_path) -> None:
    expert = 17
    formats = tuple(
        ExpertFormatSpec.compressed(1, 2) if item == expert else ExpertFormatSpec.mxfp4()
        for item in range(C.NUM_EXPERTS)
    )
    layout = TP12LayerLayout.from_formats(formats)
    spec = LayerMaterializationSpec(
        layer=1,
        formats=formats,
        compressed=(expert,),
        kept=tuple(item for item in range(C.NUM_EXPERTS) if item != expert),
        layout=layout,
    )
    path = tmp_path / "layer.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        header = TP12LayerHeader(1, layout)
        pwrite_exact(descriptor, header.to_bytes(), 0)
        pwrite_exact(
            descriptor,
            memoryview(pack_tp12_format_section(formats).numpy()),
            LAYER_HEADER_BYTES,
        )
        shared = pack_tp12_shared_scale_section(
            *[
                torch.full((LATENT_CHANNELS,), value, dtype=torch.float16)
                for value in (1.0, 2.0, 3.0)
            ]
        )
        pwrite_exact(
            descriptor,
            memoryview(shared.numpy()),
            LAYER_HEADER_BYTES + FORMAT_SECTION_BYTES,
        )
        words = PAIR_BYTES // torch.int16.itemsize
        for matrix_index, matrix in enumerate(C.EXPERT_MATRICES):
            for rank in range(12):
                logical_pair = tp12_logical_pair_index(1, expert, rank)
                pair = torch.full(
                    (words,), 100 * matrix_index + logical_pair + 1, dtype=torch.int16
                )
                local = torch.full(
                    (256,), 10.0 * (matrix_index + 1) + logical_pair, dtype=torch.float16
                )
                pwrite_exact(
                    descriptor,
                    memoryview(pair.numpy()),
                    layout.trellis_pair_offset(rank, 0, matrix),
                )
                pwrite_exact(
                    descriptor,
                    memoryview(local.numpy()),
                    layout.local_scale_offset(rank, 0, matrix),
                )
    finally:
        os.close(descriptor)

    result = validate_materialized_layer(path, spec)
    assert result["compressed_experts"] == 1
    with TP12LayerReader(path) as reader:
        restored = reader.read_compressed_matrix(expert, "w2")
        assert restored["trellis"].numel() == MATRIX_TRELLIS_BYTES // 2
        assert restored["suh"].shape == (INTERMEDIATE_CHANNELS,)
        assert restored["svh"].shape == (LATENT_CHANNELS,)
        with pytest.raises(ValueError, match="compressed"):
            reader.read_compressed_matrix(0, "w2")
