# kquant agent guide

This repository builds and validates hybrid Kimi-K3 checkpoints: selected
routed experts remain in the official MXFP4 format, the remaining routed
experts are encoded as EXL3-3 trellis weights, and non-expert linears are
served from an offline MXFP8 overlay. Read this file before relying on the
older `docs/handoff-brief.md`; that brief captures useful history, but its
headline correctness blocker has been resolved.

## Current status (2026-07-31)

- Kimi-K3 has 93 decoder layers. Layer 0 is dense; layers 1-92 contain 896
  routed experts each, for 82,432 layer/expert assignments.
- The current production artifact is `/models/Kimi-K3-EXL3-3p09` with serve
  directory `/models/Kimi-K3-EXL3-3p09-serve`. These resolve into
  `/data/models` on this host.
- Its allocation keeps 7,007 experts in original MXFP4 and stores 75,425 as
  EXL3-3 (`keep_frac=0.085`). It uses the shared-su EXL3 representation.
- `scripts/validate_k3_artifact.py` passes on 3p09: all 720,867 expected expert
  tensors were found, the allocation/config/index agree, and no structural
  issues were reported.
- Full-model streamed PyTorch runs of the official checkpoint and packaged
  3p09 both select token 28202 (` Berlin`) for the fixed prompt
  `The capital of France is Paris. The capital of Germany is`. The packaged
  result is close at the final logit decision, but layer traces show expected
  accumulated quantization and routing drift. This is a correctness anchor,
  not a quality benchmark.
- After the serialized-MXFP8 BF16-ignore loading fix, the full TP12 vLLM path
  also selects ` Berlin` for that prompt, and interactive Kimi reasoning is
  coherent. Earlier `pytorch-vs-vllm-full-tp12.json` captures predate that fix
  and are not the current serving verdict.
- Original MXFP4 numerical closure uses the ordinary production W4A16 MoE
  kernel, not the hybrid kept-weight path. Logical TP ranks are simulated
  sequentially, so TP16 geometry does not require 16 physical GPUs.
- EXL3 production-kernel closure currently runs at TP4 and TP12. Its logical
  TP16 reference is simulated, but the physical EXL3 trellis arm is reported
  as unsupported because K3's TP16 local intermediate width is 192 rather
  than a multiple of 128. Do not interpret that explicit `not-run` as a
  numerical mismatch.
- Correctness infrastructure now includes artifact validation, owned vLLM
  probes, kernel-path log audits, deterministic truncated checkpoints,
  sequential TP simulation, streamed official-PyTorch execution, and
  layer/stage trace comparison.

The current production artifact still uses the L0 router-bias traffic proxy
and identity Hessian. The next quality pipeline is now implemented: vLLM and
B12X can capture exact all-expert routing, conditional activation moments, and
raw Hessian samples from the resident interim EXL3 model while the official
checkpoint remains the offline encoder weight source; kquant can merge TP
shards, build dense per-layer Hessians, quantize all 82,432 experts before
allocation, and reuse that candidate pool for later keep-set changes. It has
unit and CUDA graph-replay coverage, but a full 10M-token production capture
and regenerated quality artifact have not yet been run. A tiny all-rates-one
TP12 preflight against the interim EXL3 teacher passed all 92 MoE layers with
exact TP joins, route/mid pairing, and zero dropped rows. Follow
`docs/exl3-calibration.md`; do not regenerate the production artifact merely
to make per-layer keep counts look uniform.

## Repository boundaries and hygiene

- Quantization and correctness tooling lives here. Production serving changes
  live in `/home/luke/projects/vllm`; kernels live in
  `/home/luke/projects/b12x`; the encoder implementation lives in
  `/home/luke/projects/exllamav3`. Do not stage changes from sibling repos in a
  kquant commit.
- Use `.venv/bin/python`, `.venv/bin/pytest`, or `.venv/bin/kquant` so runs use
  the project environment.
- Never commit checkpoint payloads, generated traces, `out/correctness`,
  `__pycache__`, or `*.pyc`. Correctness traces can consume several GiB.
- Treat `/models` artifacts as valuable. Packaging creates symlinks rather
  than copying payloads, so a serve directory depends on its artifact and
  offline non-expert source remaining present.
- Use a fresh artifact and serve destination for every changed keep fraction,
  seed/scaling mode, Hessian, or encoder revision. The pack driver skips
  existing layer shards; mixing settings in one destination silently creates
  an invalid experiment.
- The 12-GPU parent currently pins workers to numeric CUDA indices 0-11.
  Verify GPU enumeration and that every device is idle before launching.
- A full pack uses roughly 20 GiB of host RAM per worker and has shown heap
  growth. Keep ample RAM/swap available and monitor worker exits and the OOM
  log. Per-layer output is atomic and a same-configuration run is resumable.

## Produce an EXL-3 checkpoint

The production EXL3 path is `scripts/pack_exl3_12gpu.py`, not the generic
`kquant pack` command used by the earlier NF experiments.

### 1. Prepare the environment and source checkpoint

Install the project environment if needed:

```bash
cd /home/luke/projects/kquant
uv sync --dev
```

The packer imports the compatible local exllamav3 encoder and reads the
official Kimi-K3 checkpoint through `kquant.io.hf_cache.resolve()`. Verify all
96 indexed shards and all 92 MoE layers resolve before a long run:

```bash
.venv/bin/python - <<'PY'
from kquant.io.hf_cache import resolve

checkpoint = resolve()
assert not checkpoint.missing_shards, checkpoint.missing_shards
assert len(checkpoint.shard_paths) == 96, len(checkpoint.shard_paths)
assert len(checkpoint.complete_moe_layers()) == 92
print(checkpoint.snapshot_dir)
print("source checkpoint complete")
PY
```

`out/static.kqstats` supplies the current L0 allocation proxy. Preserve it to
reproduce 3p09. If it is genuinely absent, regenerate it from the same source
revision; this is a substantial checkpoint scan:

```bash
.venv/bin/kquant --out out stats --device cuda --batch-size 96
```

Before packing, check free GPUs, host RAM, swap, destination disk, and stale
pack workers. Do not kill processes by an unscoped name match; identify the
actual PIDs first.

### 2. Pack demoted experts

For a reproduction of the current 3p09 allocation, choose a new tag and keep
fraction 0.085. `KQUANT_EXL3_SHARED_SU=1` is required and must be present in
the parent environment so every worker imports the encoder in the same mode.

```bash
KQ_TAG=3p09-repack
KQ_ARTIFACT=/models/Kimi-K3-EXL3-${KQ_TAG}
KQUANT_EXL3_SHARED_SU=1 \
  .venv/bin/python scripts/pack_exl3_12gpu.py \
  --dest "${KQ_ARTIFACT}" \
  --keep-frac 0.085
```

The parent writes:

- `allocation-exl3.json` with the complete keep/EXL3 partition;
- `kquant_exl3_manifest.json` with bits, codebook, multiplier, and Hessian;
- one `exl3-layer-XXXXX.safetensors` plus `.errs.json` for each MoE layer.

There must be 92 EXL3 layer shards at completion. Existing layer shards are
skipped on a resumed run. Resume only with identical settings. An explicit
repair/rebalance worker can be pinned to one physical GPU and given a layer
list:

```bash
CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_UUID \
KQUANT_EXL3_SHARED_SU=1 \
  .venv/bin/python scripts/pack_exl3_12gpu.py \
  --worker 0 \
  --layers 17,29 \
  --dest "${KQ_ARTIFACT}"
```

With `--layers`, the worker number is only a log label. Without it, the worker
processes that worker's modulo-12 layer assignment.

### 3. Extract the original MXFP4 keep tier

The EXL3 pack intentionally omits kept experts. Copy their original packed
weights into dedicated shards so vLLM never has to reference a partial source
checkpoint shard:

```bash
.venv/bin/python scripts/extract_keep_tier.py "${KQ_ARTIFACT}"
```

This must use the same complete official source checkpoint checked in step 1.
The current 3p09 artifact produces 15 `keep-mxfp4-*.safetensors` shards; a
different keep fraction can change that count.

### 4. Ensure the offline MXFP8 non-expert overlay exists

The reusable production overlay is `/models/Kimi-K3-mxfp8-nonexpert` (14
shards, about 71 GiB). Do not rebake it for every expert allocation. If it is
missing or the dense quantization contract changes, recreate it from the Phase
A non-expert shards:

```bash
.venv/bin/python scripts/bake_mxfp8_nonexpert.py \
  --src /models/Kimi-K3-NF3R-Uniform-3p25-serve \
  --dest /models/Kimi-K3-mxfp8-nonexpert \
  --jobs 12
```

The bake converts only its explicit 2-D target linears. BF16 exclusions such
as `kv_b_proj`, `g_proj`, `f_a_proj`, `f_b_proj`, and `b_proj` must stay BF16
and must remain listed in the packaged quantization config's
`ignored_layers`.

### 5. Build the vLLM serve directory

Use a fresh destination. The package script currently takes positional
artifact/destination arguments and uses the fixed non-expert overlay path
above:

```bash
KQ_SERVE=${KQ_ARTIFACT}-serve
.venv/bin/python scripts/package_exl3_serve_dir.py \
  "${KQ_ARTIFACT}" "${KQ_SERVE}"
```

The packager symlinks EXL3, kept-MXFP4, MXFP8, tokenizer, and auxiliary files;
builds a merged safetensors index; emits the per-expert hybrid bit map; and
detects shared-su from the stored tensors rather than trusting a flag.

### 6. Run structural and numerical gates

Structural validation is mandatory and reads safetensors headers without
materializing the model:

```bash
.venv/bin/python scripts/validate_k3_artifact.py \
  --artifact "${KQ_ARTIFACT}" \
  --serve-dir "${KQ_SERVE}" \
  --output "out/validate-${KQ_TAG}.json"
```

Exercise the original checkpoint through the normal W4A16 kernel across the
important logical TP geometries:

```bash
.venv/bin/python scripts/closure_exl3_layer.py \
  --artifact "${KQ_ARTIFACT}" \
  --serve-dir "${KQ_SERVE}" \
  --layer 1 \
  --scenarios source-w4a16 \
  --tp-sizes 4,12,16 \
  --output "out/closure-${KQ_TAG}-source-w4a16.json"
```

Then exercise EXL3 through the production trellis MoE kernel at supported
geometries:

```bash
.venv/bin/python scripts/closure_exl3_layer.py \
  --artifact "${KQ_ARTIFACT}" \
  --serve-dir "${KQ_SERVE}" \
  --layer 1 \
  --scenarios all-exl3 \
  --tp-sizes 4,12 \
  --output "out/closure-${KQ_TAG}-exl3.json"
```

Repeat closures on representative layers if the encoder, allocation logic,
or kernel contract changed. Layer 1 alone is a smoke test, not a model-quality
claim.

### 7. Full streamed reference check

Before calling a new artifact usable, stream the fixed prompt through the
official PyTorch implementation one layer at a time. This path keeps only a
layer and the carried activation state resident, and records layer/stage
traces for localization.

```bash
KQ_SOURCE=/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721

.venv/bin/python scripts/stream_k3_pytorch.py \
  --checkpoint "${KQ_SOURCE}" \
  --output-dir "out/correctness/${KQ_TAG}-official" \
  --chunk-size 4 \
  --finalize

.venv/bin/python scripts/stream_k3_pytorch.py \
  --checkpoint "${KQ_SOURCE}" \
  --expert-checkpoint "${KQ_SERVE}" \
  --nonexpert-checkpoint "${KQ_SERVE}" \
  --exl3-manifest "${KQ_ARTIFACT}/kquant_exl3_manifest.json" \
  --output-dir "out/correctness/${KQ_TAG}-packaged" \
  --chunk-size 4 \
  --finalize

.venv/bin/python scripts/compare_kimi_traces.py \
  "out/correctness/${KQ_TAG}-official/trace" \
  "out/correctness/${KQ_TAG}-packaged/trace" \
  --output "out/correctness/${KQ_TAG}-official-vs-packaged.json"
```

Compare final token IDs and logits directly in both `run.json` files. Trace
thresholds appropriate for TP invariance are too strict for a 4-bit-to-3-bit
quality comparison; inspect the per-layer trajectory rather than weakening a
gate until it says `pass`.

Finally serve through the matching vLLM/b12x revisions and run an owned probe
with `scripts/run_k3_correctness.py`. The current production launcher is
`/home/luke/projects/vllm/serve-kimi-k3-exl3-3p09-tp12.sh`; point
`K3_MODEL_DIR` at the new serve directory. Preserve FP8 KV cache, Kimi tool
and reasoning parsers, graph capture, normal W4A16 MoE, and serialized MXFP8
loading when comparing against the validated production path.

## Tests

Run the CPU/unit suite for every code change:

```bash
.venv/bin/pytest -q
```

Also run the smallest relevant real-data or CUDA closure for changes to
artifact schemas, source loading, sharding, MXFP8, EXL3, or TP behavior. Unit
tests cannot establish model or production-kernel correctness.
