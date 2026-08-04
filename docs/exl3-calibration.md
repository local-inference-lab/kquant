# Kimi-K3 EXL3 calibration and all-expert allocation

This is the quality-iteration path for replacing the static router-bias proxy
and identity Hessian. It records the activation distribution of the resident
interim EXL3-3p09 model under TP, builds true per-layer Hessians, encodes every
official expert as an EXL3 candidate, and only then chooses which experts
remain MXFP4. The interim checkpoint is the online teacher; the official
checkpoint remains the offline source of canonical MXFP4 encoder weights.

## 1. Capture the interim model

Use `/home/luke/projects/vllm/serve-kimi-k3-exl3-3p09-tp12.sh`. Setting
`K3_KQUANT_CAPTURE_DIR` enables the collectors, disables prefix caching and
drafting, and preserves CUDA graph replay. Use a new output directory for
every corpus/run. Do not attempt to load the official MXFP4 checkpoint in this
TP12 process: the checkpoint plus resident capture buffers does not fit.

The hybrid collector combines two exact taps. Kept MXFP4 routes use the normal
W4A16 route-major post-SiTU cache. EXL3 routes expose
`H128(h * down_suh)`; a capture-only inverse H128 and expert-local unscale
restores canonical `h` before accumulating moments or saving rows. Thus the
stored `w2` samples are not tied to EXL3's internal rotation gauge.

The B12X full-rotation ABI orders its intermediate scale bundle as
`[gate_svh, up_svh, down_suh]`. Preserve that order in serving, closure, and
capture code. Swapping the final two blocks can look harmless under SiLU and
small-activation tests because their multiplicative factors commute, but it is
not equivalent under K3's SiTU up-branch nonlinearity.

```bash
cd /home/luke/projects/vllm

K3_KQUANT_CAPTURE_DIR=/data/kquant/k3-quality-v1 \
K3_KQUANT_CORPUS=quality-v1 \
./serve-kimi-k3-exl3-3p09-tp12.sh
```

Do not enable a draft model or expert parallelism. The capture contract is TP
only. The collector partitions input-channel moments across expert IDs and
w2-input moments across the normal TP channel shards; it never adds an online
TP collective.

The launcher's internal profile/warmup forward is intentionally excluded. In
the current launcher, one post-arm infrastructure probe executes at capture
epoch zero before the corpus driver starts. Corpus documents submitted by the
sequential driver occupy epochs `1..N`, where the epoch is the high 32 bits of
every raw observation ID. Never silently mix epoch zero into a document-level
study: constrain Hessian construction and ranking/scoring to the request range
recorded by the corpus report. The intended scale and default sampling are:

- approximately 10 million executed tokens;
- exact route count, applied-gate sum, and applied-gate-square sum;
- per-expert diagonal moments sampled at 1/16 tokens;
- raw w13-input rows sampled at 1/512 tokens, paired with all 16 selected
  expert IDs, their applied FP32 combine weights, and the same token's full
  routed-latent mixture after the TP reduction but before latent RMSNorm;
- raw w2-input routes sampled at 1/8192 routes, targeting about 16,384 rows
  per layer, with the selected expert ID retained.

Every raw row also carries a deterministic split. By default 1/16 of token
observations are validation rows; all routes from the same token receive the
same split. The split hash is independent of the sampling hash, and the
offline Hessian builder uses only training rows unless `--hessian-split all`
is explicitly requested. Treat this token split as an integrity check and a
small within-capture holdout, not as the final mode-selection set. The
representative study uses a second capture whose complete JSONL documents are
selected by a disjoint content-hash fold. No row from those documents may be
used to build a Hessian, rank neurons, choose candidate modes, or tune a
selection margin.

Keep two statistical roles distinct even when they originate in the same
corpus. The natural-routing view supplies production route mass, applied-gate
mass, routed-mixture damage, and final keep value. An optional expert-support
view may oversample tail experts to stabilize covariance and ranking
estimates, but its rows must be reweighted back to the natural distribution
before expected model damage is reported. Expert-balanced support is never a
replacement for natural-routing validation.

The device sample ring is drained after every model step. Host samples are
batched into a part every 32 steps (or 256 MiB) to avoid creating millions of
tiny files. If a rank reports any ring overflow, the offline merger rejects
the capture; rerun with a larger `VLLM_KQUANT_SAMPLE_CAPACITY`.

At the end, create the finalize sentinel printed by the launcher and send one
last small request:

```bash
touch /data/kquant/k3-quality-v1.finalize
# Send one request and wait for its response.
```

Every worker then writes a final atomic `stats.safetensors`, disables further
collection, and marks its rank manifest complete. Do not merge a live or
unfinalized capture.

Before committing to the full corpus, run a tiny capture with all sample rates
set to 1 and verify a joined layer. It must have exactly 16 middle rows per
sampled input row, matching observation/expert IDs and `gate**2` weights across
all TP ranks. Check kept-MXFP4 and EXL3 row RMS separately: both should occupy
the same broad numerical range. A large tier-dependent scale gap indicates a
rotation-scale bundle error and invalidates the capture even when every shape
and manifest check passes.

The corrected TP12 all-rates-one preflight was run against
`/models/Kimi-K3-EXL3-3p09-serve`: all 92 layers joined exactly across 12 TP
ranks, all input/route/mid pairings and gate-square weights matched, and no
sample ring dropped rows. This is a pipeline gate, not a model-quality corpus;
the full representative capture is still required.

For the intermediate algorithm study, use document-disjoint train and
validation captures before committing to the approximately 10-million-token
production capture. The current train target is 65,536 tokens from the
weighted diverse/deep/agentic corpus with fold 0 of 8 excluded; the validation
target is 16,384 tokens with only fold 0 included. Use fresh capture and report
paths on every restart. Capture accumulators are process-local and are not
restored from an incomplete directory, so pointing a restarted server at an
old directory would overwrite statistics and can collide with existing sample
part numbers.

The intermediate v2 captures completed on 2026-08-01:

- training: `/data/kquant/k3-codec-diverse-train-v2.kqcapture`, 65,536 corpus
  prompt tokens in 115 documents, plus the excluded epoch-zero probe;
- validation: `/data/kquant/k3-codec-diverse-validation-v2.kqcapture`, 16,384
  corpus prompt tokens in 27 documents, plus the excluded epoch-zero probe;
- all 12 TP manifests are complete, all 92 MoE layers are registered, and all
  input and middle sample drop counters are zero;
- the reports are `out/k3-codec-diverse-train-v2-corpus.json` and
  `out/k3-codec-diverse-validation-v2-corpus.json`, with zero document-hash
  overlap by construction.

This is an intermediate algorithm/pipeline study, not the planned 10M-token
production capture. In particular, 1/64 middle-route sampling leaves sparse
per-expert held-out support; aggregate paired-document estimates are more
credible than individual expert decisions from this capture.

The 16,384-token validation capture is a pilot gate only. It can validate the
all-layer scorer, measure runtime, and reject an obviously bad codec, but it
must not determine the production keep set. Before materialization, create a
fresh validation capture with at least 131,072 prompt tokens and 200 whole
documents, retain the 2:1:1 diverse/deep/agentic token weighting, record the
realized code/prose/multilingual/tool-use mix, and exclude every document hash
present in either v2 capture. Re-run both selected-candidate scoring and the
matched-R0 audit against that larger corpus; only its authenticated sidecar
may drive the final top-7,007 allocation.

The deterministic expanded-corpus plan is prepared at
`out/k3-codec-diverse-validation-v3-128k-corpus.json`. It contains exactly
131,072 tokens in 226 whole documents: 65,536 diverse, 32,768 deep, and
32,768 agentic-coding tokens. Its planner authenticates and excludes all 142
document hashes from the v2 training and pilot-validation reports; measured
overlap with both is zero. The future live capture destination is
`/data/kquant/k3-codec-diverse-validation-v3-128k.kqcapture`.

Run this validation-only capture with the same dense routed-input sampling as
the pilot: moments at 1/1, routed inputs at 1/4 tokens, middle rows at 1/64
routes, a 256-row device ring, and an eight-step drain interval. At 131,072
tokens this targets about 32,768 routed-input token rows per layer, or 524,288
expert-route observations before traffic skew (about 585 per assignment on
average). These settings intentionally spend storage on the input rows used
by complete expert replay; the capture is not a replacement for the later
10M-token Hessian corpus.

```bash
cd /home/luke/projects/vllm

K3_KQUANT_CAPTURE_DIR=/data/kquant/k3-codec-diverse-validation-v3-128k.kqcapture \
K3_KQUANT_CORPUS=k3-codec-diverse-validation-v3-128k \
VLLM_KQUANT_MOMENT_SAMPLE_RATE=1 \
VLLM_KQUANT_INPUT_HESSIAN_SAMPLE_RATE=4 \
VLLM_KQUANT_MID_HESSIAN_SAMPLE_RATE=64 \
VLLM_KQUANT_SAMPLE_CAPACITY=256 \
VLLM_KQUANT_SAMPLE_SAVE_EVERY=8 \
./serve-kimi-k3-exl3-3p09-tp12.sh
```

Then drive and finalize the already prepared deterministic plan:

```bash
cd /home/luke/projects/kquant

PYTHONPATH=/home/luke/projects/kquant \
/home/luke/projects/vllm/.venv/bin/python \
  scripts/run_interim_calibration_corpus.py \
  --source /data/datasets/text/diverse_calib.jsonl=2 \
  --source /data/datasets/text/deep_calib.jsonl=1 \
  --source /data/datasets/text/agentic_coding_calib_generic.jsonl=1 \
  --target-tokens 131072 \
  --fold-modulus 8 --fold-index 0 --fold-mode include \
  --exclude-report out/k3-codec-diverse-train-v2-corpus.json \
  --exclude-report out/k3-codec-diverse-validation-v2-corpus.json \
  --seed 20260802 \
  --capture-dir /data/kquant/k3-codec-diverse-validation-v3-128k.kqcapture \
  --report out/k3-codec-diverse-validation-v3-128k-corpus.json \
  --finalize-file /data/kquant/k3-codec-diverse-validation-v3-128k.kqcapture.finalize \
  --resume
```

The paired routed-latent target adds about 81 MiB to TP12 rank zero with the
default 64-row sample rings. Budget roughly 450 MiB of persistent and shared
capture storage there at a 1,024-token serving capacity, including the EXL3
inverse-rotation scratch; the other ranks need roughly 285 MiB. The scratch
scales linearly with `max_num_batched_tokens`, while the per-layer rings do not.

## 2. Merge TP shards and build Hessians

```bash
cd /home/luke/projects/kquant

.venv/bin/kquant merge-dynstats \
  --capture /data/kquant/k3-quality-v1.kqcapture \
  --stats-out out/k3-quality-v1 \
  --hess-out out/k3-quality-v1 \
  --device cuda:0 \
  --min-rows 16384
```

This produces:

- `out/k3-quality-v1.kqstats`, containing exact routing and normalized
  gate-square-weighted per-expert diagonal moments;
- `out/k3-quality-v1.kqhess/`, containing FP32 `w13` and `w2` dense Hessians
  for decoder layers 1 through 92.

These Hessians are per layer, not per expert. One FP32 `H13` plus `H2` is
about 85 MiB, and all 92 layers occupy about 7.64 GiB before filesystem
overhead. Do not multiply that storage by 896 experts; a per-expert dense
Hessian design would be a different and impractical pipeline.

For each layer the definitions are:

```text
H13 = sum_token((sum_routed_experts g^2) * x xT) / sum_token(sum g^2)
H2  = sum_route(g^2 * h hT) / sum_route(g^2)
```

Here `h` is the canonical post-SiTU activation before any EXL3 down scale or
Hadamard rotation. TP ranks store aligned channel shards of each sampled `h`;
the merger joins them by observation ID and verifies that expert IDs and split
assignments agree before forming `H2`. Both Hessians use the training fold by
default.

When using a separately captured document fold, use all raw random-split rows
from the training corpus but restrict them to the corpus request epochs. For
the v2 intermediate study that command is:

```bash
.venv/bin/kquant merge-dynstats \
  --capture /data/kquant/k3-codec-diverse-train-v2.kqcapture \
  --skip-stats \
  --hess-out /data/kquant/k3-codec-diverse-train-v2-docs \
  --layers 1,24,40 \
  --hessian-split all \
  --request-step-min 1 \
  --request-step-max 115 \
  --device cuda:0 \
  --min-rows 15000
```

The resulting fresh bundle contains 15,180 `H13` rows per selected layer and
16,256, 16,101, and 16,271 `H2` rows for layers 1, 24, and 40 respectively.
The older unfiltered study bundle remains evidence only and must not be used
for the document-disjoint result.

`kquant.capture.load_layer_samples()` exposes the route-aware raw inputs,
applied gates, paired pre-RMSNorm routed-latent teacher targets, joined w2
inputs, expert IDs, and split IDs. This is the input contract for exact-gauge
importance swizzling, held-out mixed-K mode selection, and closed-loop w2
re-encoding.

The mixed-rate study searches the complete monotone `R_r` ladder for
`0 <= r <= 12`: the first `r` ranked 128-neuron records use K2, the final `r`
use K4, and the remainder use K3. It must compare a single shared `r` with
separate `(r13, r2)` values under one common neuron permutation. For `w2`, a
candidate is authorized only by a complete heterogeneous dense-H encode; an
additive per-record exchange score is merely a proposal heuristic. The
document-disjoint validation capture remains untouched until candidate
generation and training-fold selection are complete.

The merger indexes each sample part once, rather than rereading the whole
capture for every layer. At the 10-million-token target, budget roughly 40 GiB
of host RAM for the indexed BF16 sample rows and paired latent targets in
addition to the Hessian work buffers.

### 2.1 Validate the interim teacher proxy

The interim EXL3 artifact remains the only resident calibration model. The
validation gate uses twelve independent 128-token documents from the untouched
validation fold, stratified 6/3/3 across the three source corpora. They execute
as a rectangular batch, so attention never crosses document boundaries and
the official checkpoint is read only once per layer. Only layers 1, 24, and 40
are traced; execution proceeds through layer 40 so every traced activation has
the correct upstream state. The official source is still materialized only one
layer and its active experts at a time.

Prepare and validate the exact token suite without embedding source text. The
tokenizer dependency lives in the vLLM environment; no model weights are
loaded by this step:

```bash
PYTHONPATH=/home/luke/projects/kquant \
  /home/luke/projects/vllm/.venv/bin/python \
  scripts/prepare_k3_teacher_proxy_suite.py \
  --corpus-report out/k3-codec-diverse-validation-v2-corpus.json \
  --documents 12 \
  --tokens-per-document 128 \
  --trace-layers 1,24,40 \
  --output out/correctness/teacher-proxy/document-suite-v1.json
```

Then stream the exact same batch through the official source and interim 3p09
artifact. The suite supplies and authenticates the input IDs, trace layers, and
end layer:

```bash
KQ_STREAM_PY=/home/luke/projects/vllm/.venv/bin/python
KQ_SOURCE=/data/cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721
KQ_SUITE=out/correctness/teacher-proxy/document-suite-v1.json

PYTHONPATH=/home/luke/projects/kquant "${KQ_STREAM_PY}" \
  scripts/stream_k3_pytorch.py \
  --checkpoint "${KQ_SOURCE}" \
  --input-ids-file "${KQ_SUITE}" \
  --capture-routed-post-situ \
  --output-dir out/correctness/teacher-proxy/document-suite-v1-official

KQUANT_EXL3_SHARED_SU=1 \
PYTHONPATH=/home/luke/projects/kquant "${KQ_STREAM_PY}" \
  scripts/stream_k3_pytorch.py \
  --checkpoint "${KQ_SOURCE}" \
  --expert-checkpoint /models/Kimi-K3-EXL3-3p09-serve \
  --nonexpert-checkpoint /models/Kimi-K3-EXL3-3p09-serve \
  --exl3-manifest /models/Kimi-K3-EXL3-3p09/kquant_exl3_manifest.json \
  --input-ids-file "${KQ_SUITE}" \
  --capture-routed-post-situ \
  --output-dir out/correctness/teacher-proxy/document-suite-v1-interim

.venv/bin/python scripts/summarize_k3_teacher_proxy_suite.py \
  --suite "${KQ_SUITE}" \
  --official out/correctness/teacher-proxy/document-suite-v1-official \
  --interim out/correctness/teacher-proxy/document-suite-v1-interim \
  --output out/correctness/teacher-proxy/document-suite-v1-report.json
```

The suite loader verifies the corpus-report hash, source-document and prompt
hashes, tokenizer files, every token vector, batch shape, and trace-layer
contract. The summarizer verifies both run manifests and every trace tensor
digest before measuring route-set overlap, sparse applied-gate agreement,
hidden-state drift, common-route post-SiTU drift, 128-neuron record-rank
Spearman correlation, and `R3` donor/recipient overlap.

The decision is pre-registered rather than chosen after inspecting the result.
It passes only when the whole-document bootstrap 95% lower bound is at least
0.85 for route retention, 0.90 for applied-gate cosine, 0.75 for record-rank
Spearman, and one third for both `R3` donor and recipient overlap. A point
estimate below a threshold fails; a passing point estimate with insufficient
bootstrap support requires more documents. It also requires support for at
least 75% of the possible layer/expert cases, pooled per-expert record-rank
Spearman median at least 0.65 and P10 at least 0.20, and mean per-expert `R3`
donor and recipient overlap at least 0.40. Per-expert ranking statistics require
at least four common routed rows by default. This gate establishes whether
interim activations preserve the rate-record decisions on the scoped layers;
it does not turn the official checkpoint into a resident teacher.

The existing 12-token correctness traces give only a descriptive anchor:
92.01% route-assignment retention and 0.859 mean route-set Jaccard over all 92
MoE layers. They predate post-SiTU tracing and cannot validate ranking or mode
selection. Use document-diverse inputs before accepting the proxy or deciding
whether progressive recapture is needed.

## 3. Build the reusable all-expert candidate pool

This is a full-model operation. It encodes all 82,432 layer/expert assignments
before allocation, so expect approximately a full 3-bit expert model of
candidate storage (around one TiB) plus metrics. The current phase is TP12
only, uses the resident interim artifact as calibration teacher, and streams
official source weights one matrix at a time.

```bash
cd /home/luke/projects/kquant

KQUANT_EXL3_SHARED_SU=1 \
  .venv/bin/python scripts/pack_mixed_exl3_tp12.py \
  --dest /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --capture /data/kquant/k3-codec-diverse-train-v2.kqcapture \
  --training-report out/k3-codec-diverse-train-v2-corpus.json \
  --hessians /data/kquant/k3-codec-diverse-train-v2-fit3of4-all.kqhess \
  --teacher-checkpoint /models/Kimi-K3-EXL3-3p09-serve \
  --exllamav3-root /home/luke/projects/exllamav3 \
  --mode-ids 0,1,2,3,4,5 \
  --codebook mul1-e4m3 \
  --layout r05_guarded_reuse \
  --ldlq-tf32 \
  --tailbite-context 128 \
  --expert-batch-size 20 \
  --cpu-threads 2 \
  --finalize-jobs 12
```

The current writer searches the complete `{R0,R1,R2,R3,R4,R5}` ladder with
separate `r13` and `r2` decisions, dense-H LDLQ, one common
intermediate-neuron permutation, and disjoint fit/confirmation documents. Its
procedural reconstruction is exact L16 MUL1 followed by E4M3 round-to-nearest;
there is no learned lookup table in the production payload. Active layers are
sparse `.partial` files; only the atomic payload, metrics, and selection
triplet counts as complete. Inspect completed layers without touching payload
contents:

The currently running workers record the v2 logical trellis descriptor. Their
candidate bytes are the same allocation-independent logical pair stream used
by v3; v3 adds the offline logical-pair-to-physical-rank rotation. Finalization
binds the v2 source label into the pool digest, held-out scoring preserves it,
and the materializer explicitly converts it to v3 rank slabs. Do not edit or
relabel the candidate ledgers. If this run must be resumed, the packer treats
the historical manifest's absent field as an unambiguous v2 contract and
passes that schema explicitly to every replacement worker; a new destination
records its logical schema directly in the manifest.

```bash
.venv/bin/python scripts/summarize_mixed_exl3_pool.py \
  /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --output out/mixed-exl3-tp12-e4m3-r05-candidates-v1-summary.json
```

Weight the P24/P33 kernel cases by natural routes from the untouched capture.
This is a runtime diagnostic only: it reads selected modes but cannot change
them or the keep allocation. It reports static and routed P24 fractions,
per-rank hit rates, and the distribution of P24 routes on the slowest TP12
rank. It may be run on atomic partial progress, but the kernel acceptance
record must rerun it after all 92 layers are sealed:

```bash
.venv/bin/python scripts/summarize_mixed_exl3_tp12_runtime_mix.py \
  /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  /data/kquant/k3-codec-diverse-validation-v2.kqcapture \
  --output out/mixed-exl3-tp12-phase1-runtime-mix.json
```

Export the exact selected pair table and natural route rows for each logical
rank before the routed-kernel acceptance run. The fixture contains no weights
or activations and is performance-only: it cannot alter selection or
allocation. A completed layer may be used for an interim diagnostic, but the
final record must be regenerated from the sealed pool and should include
representative early, middle, and late layers. Layer 6 is useful as the
current sparse-hit hotspot:

```bash
KQ_BENCH_LAYERS=6,24,48,72,92
IFS=, read -ra KQ_BENCH_LAYER_LIST <<< "${KQ_BENCH_LAYERS}"
for layer in "${KQ_BENCH_LAYER_LIST[@]}"; do
  for rank in $(seq 0 11); do
    .venv/bin/python scripts/export_mixed_exl3_tp12_benchmark_fixture.py \
      --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
      --capture /data/kquant/k3-codec-diverse-validation-v2.kqcapture \
      --layer "${layer}" \
      --rank "${rank}" \
      --rows 2048 \
      --output \
        "out/bench-fixtures/mixed-exl3-tp12-l${layer}-r${rank}.safetensors"
  done
done
```

After the encoder releases the GPUs, replay each fixture against the
same-format all-P33 control. Omit `--experts`: the authenticated fixture fixes
the complete 896-expert table. Route-window copies occur before the CUDA timing
events, while the captured graph sees a different natural window on each
replay:

```bash
KQ_BENCH_GPU=GPU-REPLACE-WITH-IDLE-UUID
for layer in "${KQ_BENCH_LAYER_LIST[@]}"; do
  for rank in $(seq 0 11); do
    CUDA_VISIBLE_DEVICES="${KQ_BENCH_GPU}" \
    SPARKINFER_COMPILE_DISK_CACHE=0 \
      /home/luke/projects/b12x/.venv/bin/python \
        /home/luke/projects/b12x/benchmarks/benchmark_trellis_pair_moe_tp12.py \
        --route-fixture \
          "out/bench-fixtures/mixed-exl3-tp12-l${layer}-r${rank}.safetensors" \
        --scenarios P33,sparse \
        --tokens 1,4,16 \
        --warmup 20 \
        --replays 200 \
        --cold-replays 50 \
        --bootstrap-replicates 10000 \
        --require-known-resources \
        --require-no-local-memory \
        --output \
          "out/bench-fixtures/mixed-exl3-tp12-l${layer}-r${rank}-bench.json"
  done
done

# Isolate both projection orientations at the exact TP12 dimensions. This is
# a decoder gate, not a substitute for the natural-route runs above.
CUDA_VISIBLE_DEVICES="${KQ_BENCH_GPU}" \
SPARKINFER_COMPILE_DISK_CACHE=0 \
  /home/luke/projects/b12x/.venv/bin/python \
    /home/luke/projects/b12x/benchmarks/benchmark_trellis_pair_tp12.py \
    --rows 1,2,4,8 \
    --warmup 20 \
    --replays 200 \
    --cold-replays 50 \
    --bootstrap-replicates 10000 \
    --output out/bench-fixtures/mixed-exl3-tp12-isolated-bench.json

KQ_ROUTED_BENCHMARK_ARGS=()
for layer in "${KQ_BENCH_LAYER_LIST[@]}"; do
  for rank in $(seq 0 11); do
    KQ_ROUTED_BENCHMARK_ARGS+=(
      --benchmark
      "out/bench-fixtures/mixed-exl3-tp12-l${layer}-r${rank}-bench.json"
    )
  done
done
.venv/bin/python scripts/summarize_mixed_exl3_tp12_performance.py \
  --isolated-benchmark \
    out/bench-fixtures/mixed-exl3-tp12-isolated-bench.json \
  "${KQ_ROUTED_BENCHMARK_ARGS[@]}" \
  --layers "${KQ_BENCH_LAYERS}" \
  --tokens 1,4,16 \
  --output out/mixed-exl3-tp12-phase1-performance-summary-v1.json
```

The combined command independently recomputes every paired bootstrap from raw
timings and authenticates the B12X source, candidate manifest, fixture tensors,
captured routes, and complete layer/rank/token matrix. Isolated P24 must have a
95% upper bound no greater than 3% over P33. For every routed token count and
rank, require bit-exact eager/graph closure, stable Torch graph-time allocation,
known resources, zero local-memory spill, sparse median slowdown at most 1%,
and a paired 95% upper bound at most 2% relative to P33. The synthetic
alternating case remains a stress diagnostic, not the production-weighted
gate.

After all 92 atomic triplets exist, seal the pool before any held-out scoring
or allocation. This validates every selector decision and payload header, then
hashes the roughly terabyte-scale pool in parallel and writes one immutable
content index:

```bash
.venv/bin/python scripts/finalize_mixed_exl3_pool.py \
  /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --jobs 12
```

Subsequent tools require this completion index and quickly verify the stored
file identities. Use `--verify-existing-hashes` when a full source rehash is
warranted.

## 4. Score selected candidates on the document-disjoint capture

Do this after all 92 candidate layers close. The command below is the pilot
run: it decodes only the persisted selected format and compares it with the
official MXFP4 source on the separate 16,384-token validation capture. It does
not change modes and does not load the official model resident. Repeat the
same step on the required larger validation capture before allocation.

```bash
.venv/bin/python scripts/score_mixed_exl3_validation.py \
  --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --validation-capture /data/kquant/k3-codec-diverse-validation-v2.kqcapture \
  --validation-report out/k3-codec-diverse-validation-v2-corpus.json \
  --dest /data/kquant/k3-mixed-exl3-tp12-phase1-validation-v1
```

Compare selection-corpus and held-out damage rankings before choosing the keep
score. The held-out sidecar is the preferred allocator input when support and
ranking stability close. Final model KLD/task evaluation remains independent.

```bash
.venv/bin/python scripts/summarize_mixed_exl3_validation.py \
  --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --validation-scores /data/kquant/k3-mixed-exl3-tp12-phase1-validation-v1 \
  --target-allocation /models/Kimi-K3-EXL3-3p09/allocation-exl3.json \
  --output out/mixed-exl3-tp12-phase1-validation-summary.json
```

The report gives global and per-layer rank agreement, exact-budget keep-set
Jaccard, validation support, and the held-out damage regret of using the
selection-corpus keep ranking. It also resamples the 27 whole documents 200
times and reports keep-set Jaccard, assignment churn, inclusion probabilities,
and full-validation regret around the exact-budget promotion boundary. Increase
`--bootstrap-replicates` for the final decision if that boundary is unstable.

The validation sidecar is an authenticated allocation input. Its loader hashes
the manifest, completion marker, and all 92 metric/ledger pairs. The allocation
stores that score-set digest; materialization and later artifact validation
reopen the sidecar, rederive its scores and exact keep optimum, and reject any
provenance or damage-ledger drift.

### 4.1 Audit every accepted mixed mode against matched R0

The selected-candidate sidecar is sufficient for MXFP4 keep allocation, but a
different control is required for the rate-transfer claim. Re-encode every
assignment whose selected mode is `R>0` as uniform `R0` under the exact same
dense-H, transform, and shared-scale contract, then score both payloads on the
untouched documents:

```bash
KQUANT_EXL3_SHARED_SU=1 \
  .venv/bin/python scripts/score_mixed_exl3_mode_validation.py \
  --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --validation-scores /data/kquant/k3-mixed-exl3-tp12-phase1-validation-v1 \
  --dest /data/kquant/k3-mixed-exl3-tp12-phase1-mode-validation-v1 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11 \
  --resume

.venv/bin/python scripts/summarize_mixed_exl3_mode_validation.py \
  --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --validation-scores /data/kquant/k3-mixed-exl3-tp12-phase1-validation-v1 \
  --mode-validation-scores \
    /data/kquant/k3-mixed-exl3-tp12-phase1-mode-validation-v1 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260801 \
  --output out/mixed-exl3-tp12-phase1-mode-validation-summary-v1.json
```

The final status must be `pass`. A positive point estimate with a non-positive
95% lower bound is `repeat_required`; a non-positive point estimate is
`fail`. Do not use the external result to pick modes expert by expert. Change
or reject the global policy, then regenerate candidates if the gate fails.

## 5. Allocate, materialize, and package a fresh TP12 artifact

The initial artifact uses exactly the existing `3p09` expert-container byte
budget, not merely the same nominal keep fraction. The allocator also requires
a global-optimality certificate. At this target the certificate is immediate
when the global top-7,007 validation-damage assignments fit the layer-alignment
allowance; if they do not, the command stops instead of silently freezing its
greedy repair.

```bash
.venv/bin/python scripts/allocate_mixed_exl3_tp12.py \
  --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --validation-scores /data/kquant/k3-mixed-exl3-tp12-phase1-validation-v1 \
  --target-allocation /models/Kimi-K3-EXL3-3p09/allocation-exl3.json \
  --output out/mixed-exl3-tp12-phase1-v3-allocation.json

.venv/bin/python scripts/materialize_mixed_exl3_tp12.py \
  --candidate-pool /data/kquant/k3-mixed-exl3-tp12-e4m3-r05-candidates-v1 \
  --allocation out/mixed-exl3-tp12-phase1-v3-allocation.json \
  --mode-validation-summary \
    out/mixed-exl3-tp12-phase1-mode-validation-summary-v1.json \
  --teacher-proxy-summary \
    out/correctness/teacher-proxy/document-suite-v1-report.json \
  --performance-summary \
    out/mixed-exl3-tp12-phase1-performance-summary-v1.json \
  --dest /models/Kimi-K3-Mixed-EXL3-TP12-phase1-v3

.venv/bin/python scripts/validate_mixed_exl3_tp12_artifact.py \
  --artifact /models/Kimi-K3-Mixed-EXL3-TP12-phase1-v3 \
  --output out/validate-mixed-exl3-tp12-phase1-v3.json

# Exercise every logical TP12 rank from one representative materialized layer.
# This must run in the serving environment so it imports the production vLLM
# reader and the B12X decoder, while retaining kquant on PYTHONPATH for the
# independent slab reader and numerical oracle.
for rank in $(seq 0 11); do
  PYTHONPATH=/home/luke/projects/kquant \
    /home/luke/projects/vllm/.venv/bin/python \
      scripts/validate_materialized_mixed_exl3_b12x_tp12.py \
      --artifact /models/Kimi-K3-Mixed-EXL3-TP12-phase1-v3 \
      --layer 1 \
      --rank "${rank}" \
      --device cuda:0 \
      --tokens 1 \
      --topk 16 \
      --output "out/closure-mixed-exl3-tp12-phase1-v3-l1-r${rank}.json"
done

.venv/bin/python scripts/package_mixed_exl3_tp12_serve_dir.py \
  /models/Kimi-K3-Mixed-EXL3-TP12-phase1-v3 \
  /models/Kimi-K3-Mixed-EXL3-TP12-phase1-v3-serve

.venv/bin/python scripts/validate_mixed_exl3_tp12_serve_dir.py \
  --serve-dir /models/Kimi-K3-Mixed-EXL3-TP12-phase1-v3-serve
```

Run full payload verification at least once before the production quality
claim by adding `--verify-payloads` to the artifact validator. The materialized
B12X gate above is intentionally stronger than graph-replay consistency alone:
it requires the production vLLM reader and kquant reference reader to agree
bit-for-bit, samples retained MXFP4 tensors in all three matrix orientations,
and compares the complete compressed FC1/SiTU/FC2 routed result with an
independent PyTorch regularized-weight/Hadamard reconstruction. It also requires
eager and CUDA-graph output to be bit-exact. Then run the streamed PyTorch
anchor, owned TP12 vLLM probe, KLD dataset, task suite, and routed performance
gates in the main plan and `AGENTS.md`.

The TP reduction contract is also fail-closed in the serving implementation.
Each compressed or kept expert kernel returns a rank-local 256-neuron latent
partial. Kimi first all-reduces that latent vector before its RMSNorm, the
row-parallel routed up projection emits a hidden-width partial, and FusedMoE
performs the final TP all-reduce exactly once. The compact kept-MXFP4 kernel is
built with a no-parallel configuration and model loading rejects it if it ever
advertises an already-reduced output.
