#!/usr/bin/env python3
"""Prepare an exact, document-diverse batched teacher-proxy input suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kquant.teacher_proxy_suite import (
    build_teacher_proxy_suite,
    parse_layer_list,
)


DEFAULT_CORPUS_REPORT = Path("out/k3-codec-diverse-validation-v2-corpus.json")
DEFAULT_TOKENIZER = Path("/models/Kimi-K3-EXL3-3p09-serve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-report", type=Path, default=DEFAULT_CORPUS_REPORT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--documents", type=int, default=12)
    parser.add_argument("--tokens-per-document", type=int, default=128)
    parser.add_argument("--trace-layers", type=parse_layer_list, default=(1, 24, 40))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.documents < 3:
        parser.error("--documents must be at least three")
    if args.tokens_per_document <= 0:
        parser.error("--tokens-per-document must be positive")
    return args


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer_root = args.tokenizer.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        trust_remote_code=True,
        local_files_only=True,
    )
    payload = build_teacher_proxy_suite(
        corpus_report_path=args.corpus_report,
        tokenizer_root=tokenizer_root,
        tokenizer=tokenizer,
        document_count=args.documents,
        tokens_per_document=args.tokens_per_document,
        trace_layers=args.trace_layers,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    displayed = {
        key: payload[key]
        for key in (
            "kind",
            "schema_version",
            "corpus_report",
            "selection",
            "batch_shape",
            "trace_layers",
            "end_layer",
            "suite_token_hash_sha256",
        )
    }
    displayed["output"] = str(output)
    print(json.dumps(displayed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
