"""Search shared arbitrary TP12 record maps for official K3 ``w2`` weights.

The cheap proposal independently measures each activation-ranked 128-channel
record at K2, K3, and K4 on the training corpus.  An exact dynamic program
then proposes equal-count donor/recipient sets under those separable slopes.
Every proposed set is authorized only by a complete heterogeneous dense-H
LDLQ encode and document-disjoint validation metrics.

For each proposed map the study also encodes R0 using the identical common
permutation.  This matched-order control separates rate-allocation benefit
from changes caused solely by EXL's backward LDL traversal order.  Official
weights are streamed one tensor at a time on CPU; the resident calibration
teacher is always the interim EXL3 artifact and the official model is never
instantiated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from kquant.capture import LayerSamples, load_layer_hessians, load_layer_samples
from kquant.codec_research import normalized_mse
from kquant.mixed_exl3 import (
    RATE_TRANSFER_MODES,
    RECORDS_PER_EXPERT,
    rate_transfer_record_order,
    relabel_block_contexts_for_record_order,
)
from kquant.rate_transfer_search import (
    RecordRateAssignment,
    order_assignment_for_ldlq,
    top_rate_transfer_assignments,
)
from kquant.source_weights import OfficialMXFP4Store
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_context_exl3 import (
    _channel_indices,
    _parse_r_values,
    _quantize_full_hessian_mixed,
    _quantize_partition,
)
from scripts.experiment_context_palettes import (
    _activation_metrics,
    _contexts_for_expert,
)
from scripts.experiment_retained_w2_rate_transfer import (
    _read_json,
    _request_documents,
    _validate_corpus_reports,
)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _selected_mid_rows(
    samples: LayerSamples,
    expert: int,
    request_documents: Mapping[int, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = samples.mid_experts == expert
    request_steps = torch.bitwise_right_shift(samples.mid_observations, 32)
    allowed = torch.tensor(sorted(request_documents), dtype=request_steps.dtype)
    selected &= torch.isin(request_steps, allowed)
    if not torch.count_nonzero(selected):
        raise ValueError(f"expert {expert} has no selected middle samples")
    return samples.mid_values[selected].float(), samples.mid_weights[selected].double()


def _weighted_output_sse(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    reference = inputs @ source.T
    candidate = inputs @ reconstruction.T
    return float(
        ((reference - candidate).double().square().sum(dim=1) * weights).sum()
    )


def _proposal_slopes(
    source: torch.Tensor,
    base_contexts: torch.Tensor,
    *,
    hessian: torch.Tensor,
    layer: int,
    training_inputs: torch.Tensor,
    training_weights: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_weights: torch.Tensor,
    device: torch.device,
    quantize_exl3,
) -> tuple[list[dict], tuple[float, ...], tuple[float, ...]]:
    records = []
    donor_deltas = []
    recipient_deltas = []
    train_weights_device = training_weights.to(device)
    validation_weights_device = validation_weights.to(device)
    for record in range(RECORDS_PER_EXPERT):
        channels = _channel_indices(base_contexts, record, device)
        source_part = source.index_select(1, channels)
        train_part = training_inputs.index_select(1, channels.cpu()).to(device)
        validation_part = validation_inputs.index_select(1, channels.cpu()).to(device)
        by_k: dict[int, dict] = {}
        for bit_width in (2, 3, 4):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            reconstruction, coding = _quantize_partition(
                source,
                channels,
                bit_width=bit_width,
                hessian=hessian,
                layer=layer,
                context=record,
                device=device,
                quantize_exl3=quantize_exl3,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            by_k[bit_width] = {
                "weight_nmse": normalized_mse(source_part, reconstruction),
                "training_weighted_squared_error": _weighted_output_sse(
                    source_part,
                    reconstruction,
                    train_part,
                    train_weights_device,
                ),
                "validation_weighted_squared_error": _weighted_output_sse(
                    source_part,
                    reconstruction,
                    validation_part,
                    validation_weights_device,
                ),
                "proxy": coding["proxy"],
                "quantization_seconds": time.perf_counter() - started,
            }
            del reconstruction
        donor_delta = (
            by_k[2]["training_weighted_squared_error"]
            - by_k[3]["training_weighted_squared_error"]
        )
        recipient_delta = (
            by_k[4]["training_weighted_squared_error"]
            - by_k[3]["training_weighted_squared_error"]
        )
        donor_deltas.append(donor_delta)
        recipient_deltas.append(recipient_delta)
        records.append(
            {
                "record": record,
                "rates": {f"K{width}": by_k[width] for width in (2, 3, 4)},
                "training_donor_delta_k2_minus_k3": donor_delta,
                "training_recipient_delta_k4_minus_k3": recipient_delta,
            }
        )
        del source_part, train_part, validation_part, channels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records, tuple(donor_deltas), tuple(recipient_deltas)


def _coding_evidence(coding: dict, *, weight_count: int) -> dict:
    return {
        "proxy": coding["proxy"],
        "index_bits": coding["index_bits"],
        "scale_bits": coding["scale_bits"],
        "trellis_and_scale_bpw": (
            coding["index_bits"] + coding["scale_bits"]
        )
        / weight_count,
        "trellis_descriptor": coding["trellis_descriptor"],
        "tp12_rank_trellis_bpw_before_postpack": coding[
            "tp12_rank_trellis_bpw"
        ],
        "tp12_rank_trellis_bpw": coding[
            "postpack_tp12_rank_trellis_bpw"
        ],
        "reference_edge_roundtrip": coding["reference_edge_roundtrip"],
        "reference_state_roundtrip": coding["reference_state_roundtrip"],
        "stored_decode_vs_encoder": coding["stored_decode_vs_encoder"],
        "permutation_is_128_context_aligned": coding[
            "permutation_is_128_context_aligned"
        ],
        "scale_tensor_shapes": coding["scale_tensor_shapes"],
    }


def _encode(
    source: torch.Tensor,
    contexts: torch.Tensor,
    *,
    r: int,
    hessian: torch.Tensor,
    layer: int,
    expert: int,
    samples: LayerSamples,
    validation_samples: LayerSamples,
    training_request_documents: Mapping[int, str],
    validation_request_documents: Mapping[int, str],
    device: torch.device,
    quantizer_module,
) -> dict:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    reconstruction, coding = _quantize_full_hessian_mixed(
        source,
        contexts,
        RATE_TRANSFER_MODES[r].context_bits,
        matrix="w2",
        hessian=hessian,
        layer=layer,
        layout="importance_ordered",
        device=device,
        quantizer_module=quantizer_module,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "mode": f"R{r}",
        "r": r,
        "weight_nmse": normalized_mse(source, reconstruction),
        "captured_activation": _activation_metrics(
            source,
            reconstruction,
            samples=samples,
            validation_samples=validation_samples,
            expert=expert,
            device=device,
            training_request_documents=training_request_documents,
            validation_request_documents=validation_request_documents,
        ),
        "coding": _coding_evidence(coding, weight_count=source.numel()),
        "quantization_seconds": time.perf_counter() - started,
    }
    del reconstruction
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _record_order_for_assignment(
    assignment: RecordRateAssignment,
    donor_deltas: tuple[float, ...],
    recipient_deltas: tuple[float, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    donors, recipients = order_assignment_for_ldlq(
        assignment, donor_deltas, recipient_deltas
    )
    order = rate_transfer_record_order(donors, recipients)
    return donors, recipients, order


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = _validate_corpus_reports(
        args.training_report,
        args.validation_report,
        training_capture=args.capture,
        validation_capture=args.validation_capture,
    )
    training_request_documents = _request_documents(_read_json(args.training_report))
    validation_request_documents = _request_documents(
        _read_json(args.validation_report)
    )
    resident_checkpoint = args.checkpoint.resolve()
    for name, report_path in (
        ("training", args.training_report),
        ("validation", args.validation_report),
    ):
        report_model = Path(str(_read_json(report_path).get("model_dir", ""))).resolve()
        if report_model != resident_checkpoint:
            raise ValueError(
                f"{name} corpus teacher {report_model} != {resident_checkpoint}"
            )
    store = OfficialMXFP4Store(
        repo_dir=args.official_repo_dir,
        revision=args.official_revision,
    )
    if args.expert not in store.available_experts(args.layer):
        raise ValueError(f"expert {args.layer}.{args.expert} is unavailable")

    samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = load_layer_samples(args.validation_capture, args.layer - 1)
    base_contexts, context_meta = _contexts_for_expert(
        samples,
        args.expert,
        RECORDS_PER_EXPERT,
        split=None,
        request_documents=training_request_documents,
    )
    training_inputs, training_weights = _selected_mid_rows(
        samples, args.expert, training_request_documents
    )
    validation_inputs, validation_weights = _selected_mid_rows(
        validation_samples, args.expert, validation_request_documents
    )
    _h13, h2 = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    source = store.load_matrix(args.layer, args.expert, "w2").float().to(device)
    base_contexts = base_contexts.to(device)
    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])

    proposal_records, donor_deltas, recipient_deltas = _proposal_slopes(
        source,
        base_contexts,
        hessian=h2,
        layer=args.layer,
        training_inputs=training_inputs,
        training_weights=training_weights,
        validation_inputs=validation_inputs,
        validation_weights=validation_weights,
        device=device,
        quantize_exl3=quantize_exl3,
    )

    encode_kwargs = {
        "hessian": h2,
        "layer": args.layer,
        "expert": args.expert,
        "samples": samples,
        "validation_samples": validation_samples,
        "training_request_documents": training_request_documents,
        "validation_request_documents": validation_request_documents,
        "device": device,
        "quantizer_module": quantizer_module,
    }
    canonical_r0 = _encode(source, base_contexts, r=0, **encode_kwargs)
    exact_bytes = (
        canonical_r0["coding"]["index_bits"],
        canonical_r0["coding"]["scale_bits"],
    )

    monotone = []
    maps = []
    for r in args.r_values:
        fixed = _encode(source, base_contexts, r=r, **encode_kwargs)
        if (fixed["coding"]["index_bits"], fixed["coding"]["scale_bits"]) != exact_bytes:
            raise ValueError("monotone candidate changed the exact physical byte count")
        monotone.append(fixed)

        assignments = top_rate_transfer_assignments(
            donor_deltas,
            recipient_deltas,
            r=r,
            limit=args.proposals_per_r,
        )
        seen_orders: set[tuple[int, ...]] = set()
        for proposal_rank, assignment in enumerate(assignments, start=1):
            donors, recipients, record_order = _record_order_for_assignment(
                assignment, donor_deltas, recipient_deltas
            )
            if record_order in seen_orders:
                continue
            seen_orders.add(record_order)
            contexts = relabel_block_contexts_for_record_order(
                base_contexts, record_order
            )
            matched_r0 = _encode(source, contexts, r=0, **encode_kwargs)
            mixed = _encode(source, contexts, r=r, **encode_kwargs)
            for name, candidate in (("matched R0", matched_r0), ("mixed", mixed)):
                if (
                    candidate["coding"]["index_bits"],
                    candidate["coding"]["scale_bits"],
                ) != exact_bytes:
                    raise ValueError(f"{name} candidate changed the exact byte count")
                if not candidate["coding"]["reference_edge_roundtrip"] or not candidate[
                    "coding"
                ]["reference_state_roundtrip"]:
                    raise ValueError(f"{name} candidate failed reference closure")
            maps.append(
                {
                    "r": r,
                    "proposal_rank": proposal_rank,
                    "additive_training_delta": assignment.additive_delta,
                    "donor_records_low_to_high": list(donors),
                    "recipient_records_pair_order_high_to_low": list(recipients),
                    "logical_record_order_low_to_high": list(record_order),
                    "matched_order_r0": matched_r0,
                    "mixed": mixed,
                }
            )

    payload = {
        "kind": "kquant_w2_record_map_search",
        "schema_version": 1,
        "constraint": (
            "resident teacher and captures are interim EXL3; one official MXFP4 "
            "w2 tensor is streamed offline and the official model is never instantiated"
        ),
        "signature": {
            "checkpoint": str(args.checkpoint.resolve()),
            "resident_teacher_checkpoint": str(args.checkpoint.resolve()),
            "source_checkpoint": str(store.root),
            "source_revision": store.revision,
            "capture": str(args.capture.resolve()),
            "validation_capture": str(args.validation_capture.resolve()),
            "hessians": str(args.hessians.resolve()),
            "layer": args.layer,
            "expert": args.expert,
            "r_values": list(args.r_values),
            "proposals_per_r": args.proposals_per_r,
            "tp_size": 12,
            "provenance": provenance,
        },
        "selection_contract": {
            "proposal_data": "training documents only",
            "proposal_model": (
                "independent 128-channel K2/K3/K4 local-H encodes followed by "
                "exact top-k additive equal-count assignment DP"
            ),
            "authorization": "complete heterogeneous dense-H LDLQ encode",
            "validation_usage": "reporting only; never used to propose a map",
            "map_storage": (
                "none: the shared map folds into the common w1/w3-row and "
                "w2-column permutation"
            ),
        },
        "context": context_meta,
        "proposal_records": proposal_records,
        "canonical_r0": canonical_r0,
        "monotone": monotone,
        "maps": maps,
        "exact_byte_contract": {
            "index_bits": exact_bytes[0],
            "scale_bits": exact_bytes[1],
            "all_candidates_equal": True,
        },
        "complete": True,
    }
    _atomic_write(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/Kimi-K3-EXL3-3p09-serve"),
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument(
        "--r-values", type=_parse_r_values, default=(1, 2, 3, 5)
    )
    parser.add_argument("--proposals-per-r", type=int, default=4)
    parser.add_argument("--official-repo-dir", type=Path)
    parser.add_argument(
        "--official-revision",
        default="c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must be in 1..92")
    if not 0 <= args.expert < 896:
        parser.error("--expert must be in 0..895")
    if 0 in args.r_values:
        parser.error("--r-values must contain only nonzero modes")
    if args.proposals_per_r <= 0:
        parser.error("--proposals-per-r must be positive")
    return args


def main() -> None:
    payload = run(parse_args())
    print(
        json.dumps(
            {
                "layer": payload["signature"]["layer"],
                "expert": payload["signature"]["expert"],
                "maps": len(payload["maps"]),
                "complete": payload["complete"],
            },
            indent=1,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
