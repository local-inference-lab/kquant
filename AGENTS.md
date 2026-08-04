# kquant agent guide

This repository builds and validates `Kimi-K3-QSRT`, a TP12 hybrid checkpoint.
QSRT experts use SQG-E4M3 trellis reconstruction with expert-static K2/K4 rate
shifts around K3. The high-quality tier uses X4T, which reproduces the official
MXFP4 tensors exactly while compressing their scale plane. Non-expert linears
come from the reusable offline MXFP8 overlay.

The canonical name is **QSRT**: Quantile-Stratified Rate-shifted Trellis codec.
Do not introduce TrellisShift, TSH, SQRT-C, `mixed_exl3`, or generic X4 names in
new code, schemas, artifacts, or documentation. `X4T` is the only current X4
format. MCG and MUL1 may remain only where they are active numerical controls
or external EXL encoder APIs.

## Current state (2026-08-04)

- Kimi-K3 has 93 decoder layers. Layers 1-92 contain 896 routed experts each,
  for 82,432 layer/expert assignments.
- `/models/Kimi-K3-EXL3-3p09` and its serve directory remain the validated
  interim teacher and comparison checkpoint. They are not the QSRT format.
- QSRT v1 is TP12 only. Do not add TP4 or TP16 requirements to its storage,
  allocation, capture, or kernel gates.
- The lossy search is `R0/R1/R2`, with one `r13` decision shared by `w1` and
  `w3` and an independent `r2` decision for `w2`.
- `sqg-normal-e4m3` is the frozen R44 baseline. `sqg-cheb-normal-e4m3` and K2
  law/stratification variants remain controlled research candidates until the
  production donor/recipient study selects one.
- X4T is the exact endpoint. There is no raw-MXFP4 or old variable-rate X4 tier
  in a QSRT artifact.
- The earlier layer-global dense `H2` study was invalid as a production metric:
  post-SiTU coordinate indices are expert-local. Layer-global `H13` is valid,
  but `H2` must be built from expert-stratified routed rows and shrunk toward
  identity. Unsupported experts fall back to identity.
- The immediate calibration target is a source-controlled 1,000,000-token
  training capture. It is the first scale-up from the 65,536-token pilot, not
  the final corpus. Mode-selection and final-validation captures must remain
  document- and prompt-disjoint.
- The next checkpoint must be built into a fresh artifact after the new capture
  and Hessian path are validated. Do not resume the stopped R44/X4T build.

The active design is documented in [docs/qsrt-technical-brief.md](docs/qsrt-technical-brief.md).
Capture and covariance requirements are in
[docs/qsrt-calibration.md](docs/qsrt-calibration.md) and
[docs/dense-h-corpus-plan.md](docs/dense-h-corpus-plan.md).

## Porting QSRT to another gated MoE model

Treat a model such as GLM 5.2 as a new codec port, not as a Kimi checkpoint
with renamed dimensions. Do not write payloads until the following contracts
are explicit and tested.

### 1. Freeze source and architecture identity

Pin the repository/model ID, immutable revision, tokenizer/chat template,
configuration, source tensor index, and every source shard hash. Inventory:

- decoder and MoE layer counts;
- routed and shared expert counts, top-k routing, gate normalization, and any
  expert grouping;
- exact gate/up/down tensor names, stored orientations, dtypes, block scales,
  and logical matrix shapes;
- activation equation and the location of multiplicative gates;
- tensor-parallel sharding axes and local dimensions at the intended TP;
- non-expert linears and their serving format; and
- the source checkpoint's exact high-quality representation.

Add a model-specific constants/adapter module and header-only inventory tests.
Never infer matrix roles merely from a familiar tensor suffix.

### 2. Prove the hidden-coordinate symmetry

QSRT's shared neuron permutation is valid only when the expert's nonlinear
middle operation is coordinatewise and the same permutation is applied to all
coupled branches. Write the model's expert function and prove the exact update,
for example

```text
W_gate' = P W_gate
W_up'   = P W_up
W_down' = W_down P^T.
```

Test full-precision closure on real source experts before quantization. A
Hadamard, sign transform, affine scale, cross-channel normalization, or
model-specific interaction does not inherit this proof. Document where every
encoder transform is cancelled relative to the activation.

### 3. Derive a model-native fixed payload

Choose logical record width, coding-tile width, pair/super-record structure,
and TP ownership from the new intermediate dimension and target GPU MMA atoms.
The general fixed-rate identity is

```text
K_donor + K_recipient = 2 * K_baseline.
```

For Kimi v1 this is P24 versus P33 over paired 128-channel records. Another
model may need a different record width or rank grouping. Prove divisibility,
constant stride, exact bytes, rank balance, bounded random access, and malformed
input rejection before freezing a format. Do not copy Kimi's 24-record mode
table when the new intermediate axis has different geometry.

### 4. Establish the reconstruction law

Select `L`, supported rates, reference distribution, SQG rank permutation, and
finite reconstruction type. Verify graph bijection, one outgoing edge per
stratum, all-rank coverage, finite labels, tail-biting closure, and exact
pack/unpack. Train or synthesize the scalar law on the production transformed
and BlockLDLQ-feedback domain. A source checkpoint's scalar alphabet is not
automatically the optimal reconstruction alphabet for the second codec.

### 5. Build a representative resident teacher

The resident checkpoint must fit at the target TP and expose exact routes,
applied gates, expert inputs, and canonical post-activation rows. The immutable
official checkpoint remains the offline weight source. Assemble source-pinned,
document-disjoint training, selection, and final-validation plans with prose,
dialogue, code, math, multilingual, tools, and realistic long-context traffic.

Run a route census first. Allocate expert-aware row reservoirs, preserve sample
probabilities, and record distinct-document support. `H13` may be shared only
when its input basis is actually shared; `H2` is expert-local. Define shrinkage
and identity fallback before encoding all experts.

### 6. Port the source decoder and candidate encoder

Implement bit-exact source tensor decoding and round-trip tests. Stream one
expert or bounded expert batch at a time; never materialize the full official
model on CPU or GPU. For every expert, emit uniform baseline and allowed
rate-shift candidates, preserve dense-H BlockLDLQ feedback, reconstruct the
complete expert function, and select modes with paired document evidence.
Candidate generation must be allocation-independent and resumable by atomic
layer.

### 7. Define the high-quality endpoint

X4T is specific to an MXFP4 nibble plus UE8M0-scale source. Reuse it only if
the new model has the same exact source contract. Otherwise design a separate
endpoint that either reproduces the source tensor exactly or has an explicitly
validated near-lossless error target. Build an all-expert exact-byte index;
never allocate from nominal bpw alone.

### 8. Implement the target runtime

Add reference CPU decode first, then the model's target-TP B12X and vLLM paths.
Prove layout, TP joins, transform order, A16 numerical closure, graph replay,
register/shared-memory bounds, rank-tail latency, and routed fused-MoE
performance. A decoder microbenchmark is insufficient if activation conversion,
scale preparation, or expert scheduling erases the gain.

### 9. Allocate and release

Score candidates on untouched natural-routing data, solve the global
quality-versus-exact-byte allocation, and materialize only into fresh artifact
and serve directories. Require structural and bit-exact validation, streamed
source comparison, live-routing drift, teacher-logit/KLD, task quality, and
production latency before calling `<Model>-QSRT` usable.

Keep model-specific dimensions, source formats, schema IDs, and kernel support
out of the shared QSRT theory. A successful new-model port should add a small
adapter and explicit format version, not weaken Kimi-K3's frozen TP12 contract.

## Repository boundaries

- Quantization, capture analysis, allocation, packaging, and correctness tools
  live here.
- Production serving changes live in `/home/luke/projects/vllm`.
- B12X kernels live in `/home/luke/projects/b12x`.
- The QSRT offline encoder lives in `kquant/exl3_encoder_backend.py`; its
  tail-biting SQG CUDA code lives under `kquant/csrc`. The checkout at
  `/home/luke/projects/exllamav3` must remain an unmodified upstream dependency
  and supplies only its extension plus Hadamard/tensor utilities. Put every
  QSRT-specific encoder change in this repository.
- Never stage sibling-repository changes in a kquant commit.
- Use `.venv/bin/python` and `.venv/bin/pytest`.
- Never commit checkpoint payloads, captures, generated traces, `out/`,
  `__pycache__`, or `*.pyc`.
- Treat `/models` and `/data/kquant` artifacts as valuable. Use a fresh path for
  every changed corpus, Hessian policy, codebook, allocation, or encoder build.
- Candidate layers are atomic and an identical all-layer build is resumable.
  Never mix settings in an existing destination.
- Before a 12-GPU run, verify GPU enumeration, free memory, host RAM/swap,
  destination capacity, and exact worker PIDs. Do not kill by broad name match.

## Current QSRT pipeline

### 1. Validate the official source

The official checkpoint is an offline weight source; it is never loaded as the
resident calibration model.

```bash
cd /home/luke/projects/kquant
uv sync --dev

.venv/bin/python - <<'PY'
from kquant.io.hf_cache import resolve

checkpoint = resolve()
assert not checkpoint.missing_shards, checkpoint.missing_shards
assert len(checkpoint.shard_paths) == 96
assert len(checkpoint.complete_moe_layers()) == 92
print(checkpoint.snapshot_dir)
PY
```

### 2. Plan and capture the corpus

Build source-controlled JSONL plans, validate raw-record and post-tokenization
separation, then drive the resident interim checkpoint with
`scripts/run_interim_calibration_corpus.py`. Use fresh capture/report paths.
The immediate training target is 1,000,000 prompt tokens.

```bash
.venv/bin/python scripts/validate_calibration_corpus_plans.py \
  <training-report> <selection-report> <final-validation-report> \
  --output out/qsrt-corpus-integrity.json
```

The vLLM launcher must set `K3_KQUANT_CAPTURE_DIR` and use the interim EXL3
teacher. Finalize the capture exactly once and reject any dropped sample rows,
TP join mismatch, epoch-zero probe contamination, or document overlap.

### 3. Build and seal all-expert candidates

Candidate generation streams official MXFP4 matrices one expert batch at a
time. It must use the new capture, the validated Hessian bundle, separate
`r13/r2` selection, and only R0/R1/R2.

```bash
.venv/bin/python scripts/pack_qsrt_candidates_tp12.py \
  --dest <fresh-candidate-pool> \
  --capture <training-capture> \
  --training-report <training-report> \
  --hessians <validated-kqhess> \
  --hessian-policy captured_blend \
  --mode-ids 0,1,2 \
  --codebook sqg-normal-e4m3 \
  --layout qsrt_guarded_reuse \
  --ldlq-tf32

.venv/bin/python scripts/finalize_qsrt_candidate_pool.py \
  <fresh-candidate-pool> --jobs 12
```

Do not use `captured_blend` until the bundle and candidate path prove that
`H2` is expert-stratified. Identity H is the scientific control, not a silent
fallback for a malformed dense-H bundle.

### 4. Score untouched validation data

```bash
.venv/bin/python scripts/score_qsrt_validation.py \
  --candidate-pool <candidate-pool> \
  --validation-capture <validation-capture> \
  --validation-report <validation-report> \
  --dest <validation-scores>

.venv/bin/python scripts/score_qsrt_mode_validation.py \
  --candidate-pool <candidate-pool> \
  --validation-scores <validation-scores> \
  --dest <mode-validation-scores>
```

The candidate pool's fit/confirmation folds select expert-static rate shifts.
The untouched capture estimates keep-tier damage and verifies each accepted
shift against its R0/R0 counterfactual. Final validation cannot tune modes,
codebooks, margins, or the X4T allocation.

### 5. Index X4T and allocate exact bytes

```bash
.venv/bin/python scripts/index_x4t_costs.py \
  --dest <x4t-cost-index> --jobs 12 --resume

.venv/bin/python scripts/allocate_qsrt_tp12.py \
  --candidate-pool <candidate-pool> \
  --validation-scores <validation-scores> \
  --x4t-cost-index <x4t-cost-index> \
  --target-container-bytes 1058586247168 \
  --output <allocation.json>
```

The allocator chooses between each expert's selected fixed-rate QSRT candidate
and its exact X4T byte cost. R0/R1/R2 are rate-shift decisions at constant
three-bit trellis payload; they are not the high-tier allocation.

### 6. Materialize, package, and validate

```bash
.venv/bin/python scripts/materialize_qsrt_tp12.py \
  --candidate-pool <candidate-pool> \
  --x4t-cost-index <x4t-cost-index> \
  --allocation <allocation.json> \
  --dest <fresh-artifact> --resume

.venv/bin/python scripts/package_qsrt_tp12_serve_dir.py \
  <fresh-artifact> <fresh-serve-dir> \
  --x4t-tp12-source <x4t-runtime-source>

.venv/bin/python scripts/validate_qsrt_tp12_artifact.py \
  --artifact <fresh-artifact> \
  --verify-payloads \
  --output out/validate-qsrt.json
```

Every materialized layer is a fixed QSRT trellis slab plus an exact X4T
sidecar. Validation must close schemas, hashes, byte accounting, zero padding,
format tables, candidate bytes, and official MXFP4 reconstruction.

### 7. Quality and runtime gates

Run the TP12 B12X numerical closures, X4T W4A16 closure, performance gate,
streamed PyTorch comparison, vLLM A16 compatibility path, KLD suite, live
routing drift checks, and task evaluations. A8 is a separate speed/quality
tradeoff and must not be used to judge the codec's weight distortion.

The fixed ` Berlin` prompt is a wiring smoke test only. A usable checkpoint
requires document-disjoint KLD and task quality plus production TP12 latency.

## Tests

Run the full CPU/unit suite for every code change:

```bash
.venv/bin/pytest -q
```

For artifact, source-loading, sharding, X4T, QSRT layout, or kernel changes,
also run the smallest relevant real-data or CUDA closure. Unit tests do not
establish full-model or production-kernel correctness.
