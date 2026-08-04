"""Run a resumable TP12 ``w2`` rate-transfer study.

One invocation handles one K3 MoE layer.  It loads the training capture,
document-disjoint validation capture, and dense layer Hessian once, then
evaluates the requested ``R_r`` candidates for every retained MXFP4 expert.
The resident calibration teacher is always the interim checkpoint.  Weights
may come either from its retained tier or from individual official MXFP4
tensors streamed offline; the script never instantiates the official model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kquant.capture import load_layer_hessians, load_layer_samples
from kquant.codec_research import normalized_mse
from kquant.interim_weights import InterimKeepStore
from kquant.mixed_exl3 import RATE_TRANSFER_MODES
from kquant.source_weights import OfficialMXFP4Store
from scripts.experiment_codec_transfers import _load_exl3_quantizer
from scripts.experiment_context_exl3 import (
    _parse_r_values,
    _quantize_full_hessian_mixed,
)
from scripts.experiment_context_palettes import (
    _activation_metrics,
    _contexts_for_expert,
)


def _parse_experts(value: str) -> tuple[int, ...]:
    try:
        experts = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "experts must be comma-separated integers"
        ) from exc
    if not experts or len(set(experts)) != len(experts):
        raise argparse.ArgumentTypeError("experts must be nonempty and unique")
    if any(not 0 <= expert < 896 for expert in experts):
        raise argparse.ArgumentTypeError("expert IDs must be in 0..895")
    return experts


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _request_documents(report: dict) -> dict[int, str]:
    documents = report.get("documents", [])
    if not isinstance(documents, list) or not documents:
        raise ValueError("corpus report has no documents")
    if report.get("planned_requests", len(documents)) != len(documents):
        raise ValueError("corpus report planned-request count disagrees with documents")
    if report.get("completed_requests", len(documents)) != len(documents):
        raise ValueError("corpus report completed-request count disagrees with documents")
    # Epoch zero is the known pre-corpus server probe.  The sequential corpus
    # driver then submits one request per document at epochs 1..N.
    return {
        request_step: str(document["document_hash"])
        for request_step, document in enumerate(documents, start=1)
    }


def _validate_corpus_reports(
    training_report_path: Path,
    validation_report_path: Path,
    *,
    training_capture: Path,
    validation_capture: Path,
) -> dict:
    training = _read_json(training_report_path)
    validation = _read_json(validation_report_path)
    for name, report, capture in (
        ("training", training, training_capture),
        ("validation", validation, validation_capture),
    ):
        if report.get("kind") != "kquant_interim_calibration_corpus_run":
            raise ValueError(f"{name} report has the wrong kind")
        if not bool(report.get("finalized")):
            raise ValueError(f"{name} corpus report is not finalized")
        reported_capture = Path(str(report.get("capture_dir", ""))).resolve()
        if reported_capture != capture.resolve():
            raise ValueError(
                f"{name} report capture {reported_capture} != {capture.resolve()}"
            )

    training_fold = training.get("fold", {})
    validation_fold = validation.get("fold", {})
    common_fold = ("modulus", "index", "unit")
    if any(training_fold.get(key) != validation_fold.get(key) for key in common_fold):
        raise ValueError("training and validation reports use different fold rules")
    if training_fold.get("mode") != "exclude":
        raise ValueError("training report must exclude the held-out document fold")
    if validation_fold.get("mode") != "include":
        raise ValueError("validation report must include only the held-out document fold")

    def document_hashes(report: dict) -> set[str]:
        request_documents = _request_documents(report)
        hashes = set(request_documents.values())
        if len(hashes) != len(request_documents):
            raise ValueError("corpus report has missing or duplicate document hashes")
        return hashes

    training_hashes = document_hashes(training)
    validation_hashes = document_hashes(validation)
    overlap = training_hashes & validation_hashes
    if overlap:
        raise ValueError(f"training/validation document overlap: {sorted(overlap)[:8]}")
    return {
        "training_report": str(training_report_path.resolve()),
        "validation_report": str(validation_report_path.resolve()),
        "training_documents": len(training_hashes),
        "validation_documents": len(validation_hashes),
        "document_overlap": 0,
        "fold": {
            key: training_fold.get(key) for key in common_fold
        },
        "request_epoch_contract": {
            "startup_epoch": 0,
            "first_document_epoch": 1,
            "encoding": "observation_id >> 32",
        },
    }


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _signature(
    args: argparse.Namespace,
    experts: list[int],
    provenance: dict,
    *,
    source_checkpoint: Path,
) -> dict:
    return {
        "checkpoint": str(args.checkpoint.resolve()),
        "resident_teacher_checkpoint": str(args.checkpoint.resolve()),
        "weight_source": args.weight_source,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "capture": str(args.capture.resolve()),
        "validation_capture": str(args.validation_capture.resolve()),
        "hessians": str(args.hessians.resolve()),
        "layer": args.layer,
        "experts": experts,
        "r_values": list(args.r_values),
        "layout": args.layout,
        "tp_size": 12,
        "metrics_schema": 2,
        "provenance": provenance,
    }


def run(args: argparse.Namespace) -> dict:
    interim_store = InterimKeepStore(args.checkpoint)
    if args.weight_source == "interim-retained":
        store = interim_store
        available = interim_store.kept_experts(args.layer)
    else:
        if args.experts is None:
            raise ValueError(
                "--experts is required with --weight-source official-streamed"
            )
        store = OfficialMXFP4Store(
            repo_dir=args.official_repo_dir,
            revision=args.official_revision,
        )
        available = store.available_experts(args.layer)
    experts = sorted(available if args.experts is None else args.experts)
    missing = sorted(set(experts) - set(available))
    if missing:
        raise ValueError(
            f"layer {args.layer} experts are unavailable from "
            f"{args.weight_source}: {missing[:16]}"
        )

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
    signature = _signature(
        args,
        experts,
        provenance,
        source_checkpoint=store.root,
    )
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        payload = _read_json(args.output)
        if payload.get("signature") != signature:
            raise ValueError("resume output does not match the requested study")
        if payload.get("complete"):
            return payload
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        payload = {
            "kind": "kquant_retained_w2_rate_transfer_study",
            "schema_version": 2,
            "constraint": (
                "resident teacher and captures are interim EXL3; official "
                "source weights, when requested, are streamed one tensor at "
                "a time and the official model is never instantiated"
            ),
            "signature": signature,
            "results": {},
            "skipped": {},
            "complete": False,
        }
        _atomic_write(args.output, payload)

    training_samples = load_layer_samples(args.capture, args.layer - 1)
    validation_samples = load_layer_samples(
        args.validation_capture, args.layer - 1
    )
    _h13, h2 = load_layer_hessians(args.hessians, args.layer)
    device = torch.device(args.device)
    quantize_exl3 = _load_exl3_quantizer(args.exllamav3_root)
    quantizer_module = __import__(quantize_exl3.__module__, fromlist=["*"])
    completed = set(payload["results"]) | set(payload["skipped"])

    for offset, expert in enumerate(experts, start=1):
        key = str(expert)
        if key in completed:
            continue
        try:
            block_contexts, context_meta = _contexts_for_expert(
                training_samples,
                expert,
                RATE_TRANSFER_MODES[0].context_count,
                split=None,
                request_documents=training_request_documents,
            )
        except ValueError as error:
            if "has no selected middle samples" not in str(error):
                raise
            payload["skipped"][key] = {"reason": str(error)}
            _atomic_write(args.output, payload)
            continue

        source = store.load_matrix(args.layer, expert, "w2").float().to(device)
        block_contexts = block_contexts.to(device)
        candidates = []
        for r in args.r_values:
            mode = RATE_TRANSFER_MODES[r]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            reconstruction, coding = _quantize_full_hessian_mixed(
                source,
                block_contexts,
                mode.context_bits,
                matrix="w2",
                hessian=h2,
                layer=args.layer,
                layout=args.layout,
                device=device,
                quantizer_module=quantizer_module,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            candidates.append(
                {
                    "mode": mode.name,
                    "r": r,
                    "bit_widths_low_to_high_record": list(mode.context_bits),
                    "weight_nmse": normalized_mse(source, reconstruction),
                    "captured_activation": _activation_metrics(
                        source,
                        reconstruction,
                        samples=training_samples,
                        validation_samples=validation_samples,
                        expert=expert,
                        device=device,
                        training_request_documents=training_request_documents,
                        validation_request_documents=validation_request_documents,
                    ),
                    "proxy": coding["proxy"],
                    "index_bits": coding["index_bits"],
                    "scale_bits": coding["scale_bits"],
                    "trellis_and_scale_bpw": (
                        coding["index_bits"] + coding["scale_bits"]
                    )
                    / source.numel(),
                    "provisional_expert_format_bits": 8,
                    "quantization_seconds": time.perf_counter() - started,
                    "trellis_descriptor": coding["trellis_descriptor"],
                    "tp12_rank_trellis_bpw": coding["tp12_rank_trellis_bpw"],
                    "postpack_tp12_rank_trellis_bpw": coding[
                        "postpack_tp12_rank_trellis_bpw"
                    ],
                    "reference_edge_roundtrip": coding["reference_edge_roundtrip"],
                    "reference_state_roundtrip": coding[
                        "reference_state_roundtrip"
                    ],
                    "stored_decode_vs_encoder": coding["stored_decode_vs_encoder"],
                }
            )
            del reconstruction
            if device.type == "cuda":
                torch.cuda.empty_cache()

        payload["results"][key] = {
            "context": context_meta,
            "candidates": candidates,
        }
        _atomic_write(args.output, payload)
        print(
            f"layer {args.layer} expert {expert}: "
            f"{offset}/{len(experts)} complete",
            flush=True,
        )
        del source, block_contexts

    payload["complete"] = True
    payload["summary"] = {
        "retained_experts_requested": len(experts),
        "experts_encoded": len(payload["results"]),
        "experts_skipped_without_training_rows": len(payload["skipped"]),
        "candidate_encodes": sum(
            len(result["candidates"]) for result in payload["results"].values()
        ),
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
    parser.add_argument(
        "--weight-source",
        choices=("interim-retained", "official-streamed"),
        default="interim-retained",
        help=(
            "matrix source; official-streamed opens individual official "
            "MXFP4 tensors offline and never instantiates that model"
        ),
    )
    parser.add_argument(
        "--official-repo-dir",
        type=Path,
        help="optional Hugging Face K3 repository cache root",
    )
    parser.add_argument(
        "--official-revision",
        default="c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721",
        help="explicit official snapshot label for streamed source tensors",
    )
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--validation-capture", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--hessians", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=_parse_experts)
    parser.add_argument("--r-values", type=_parse_r_values, default=(0, 6))
    parser.add_argument(
        "--layout",
        choices=("importance_ordered", "tp12_balanced"),
        default="importance_ordered",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--exllamav3-root",
        type=Path,
        default=Path("/home/luke/projects/exllamav3"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.layer <= 92:
        parser.error("--layer must be in 1..92")
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps(payload.get("summary", {}), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
