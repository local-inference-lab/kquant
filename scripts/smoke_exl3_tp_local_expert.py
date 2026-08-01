"""Quantize one Kimi-K3 expert as a TP-local EXL3 integration smoke test.

This deliberately opens only the six MXFP4 tensors belonging to one routed
expert (packed weight + scale for w1/w3/w2).  It does not construct vLLM or
load the checkpoint into GPU memory, so it is suitable for validating a new
quantizer/runtime image before the expensive end-to-end model load.
"""

from __future__ import annotations

import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=7)
    parser.add_argument(
        "--experts",
        help="comma-separated expert IDs; overrides --expert for batching tests",
    )
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--temp-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # kquant.pack.exl3 reads this contract at import time.
    os.environ["KQUANT_EXL3_SHARED_SU"] = "1"

    import torch

    from kquant.io.hf_cache import resolve
    from kquant.pack.exl3 import make_shared_h, quantize_layer
    from scripts.pack_exl3_12gpu import validate_tp_size

    device = torch.device(args.device)
    experts = (
        [int(value) for value in args.experts.split(",")]
        if args.experts
        else [args.expert]
    )
    local_intermediate = validate_tp_size(args.tp_size)
    cache = resolve()
    shared_h = {k: make_shared_h(k, device) for k in {3584, local_intermediate}}
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    tensors, errors = quantize_layer(
        cache,
        args.layer,
        experts,
        device,
        shared_h,
        batch=args.batch,
        tp_size=args.tp_size,
        temp_batch_size=args.temp_batch_size,
    )
    elapsed = time.perf_counter() - start
    expected = {
        "w1.exl3_trellis": [3584 // 16, 3072 // 16, 48],
        "w1.exl3_suh": [3584],
        "w1.exl3_svh": [3072],
        "w3.exl3_trellis": [3584 // 16, 3072 // 16, 48],
        "w3.exl3_suh": [3584],
        "w3.exl3_svh": [3072],
        "w2.exl3_trellis": [3072 // 16, 3584 // 16, 48],
        "w2.exl3_suh": [3072],
        "w2.exl3_svh": [3584],
    }
    actual = {
        ".".join(name.split(".")[-2:]): list(tensor.shape)
        for name, tensor in tensors.items()
    }
    if actual != expected:
        raise AssertionError(f"TP-local merged ABI mismatch: {actual}")
    print(
        {
            "pass": True,
            "layer": args.layer,
            "experts": experts,
            "tp_size": args.tp_size,
            "local_intermediate": local_intermediate,
            "checkpoint_tensors_opened": 6 * len(experts),
            "output_tensors": len(tensors),
            "shapes": actual,
            "proxy_errors": errors,
            "elapsed_s": round(elapsed, 3),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / 2**20, 1
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
