# QSRT calibration and dense-H workflow

This document defines the calibration contract for the next TP12
`Kimi-K3-QSRT` checkpoint. The resident interim EXL3 checkpoint supplies
routing and activation observations. The official Kimi-K3 checkpoint remains
the offline source of canonical MXFP4 expert weights.

The immediate job is a 1,000,000-token training capture. It is a large pilot
and an implementation gate, not the final production corpus. Mode selection
and final validation use separate whole-document folds and fresh captures.

## Why the capture exists

QSRT needs four kinds of evidence:

1. Natural route count, applied-gate mass, and gate-square mass for every
   layer/expert assignment.
2. Routed expert inputs for the `w1`/`w3` covariance and for replaying official
   source experts.
3. Expert-local post-SiTU rows for the `w2` covariance and intermediate-neuron
   importance order.
4. Document identities so mode selection and validation can be paired,
   clustered, and isolated from encoder training.

The capture is not a copy of model weights. The official model must not be
loaded alongside the resident teacher.

## Covariance semantics

`H13 = E[x x^T]` lives in the shared input coordinate system of a layer. A
layer-global estimate is meaningful because every routed expert receives the
same latent coordinates.

`H2[e] = E[h_e h_e^T | e routed]` lives in expert `e`'s post-SiTU
intermediate coordinate system. Coordinate `j` in one expert has no semantic
relationship to coordinate `j` in another expert. Pooling `h` rows across all
896 experts therefore creates an invalid metric even when the accumulation is
numerically correct.

The production rule is:

- build or retain one layer-global `H13`;
- build `H2[e]` from rows routed to expert `e`;
- shrink each expert estimate explicitly toward identity according to support;
- materialize/factor one expert covariance at a time and discard it after
  encoding; and
- use identity H for an unsupported expert, never another expert's `H2` basis.

The candidate encoder must report row count, distinct documents, gate-square
effective sample size, shrinkage, and fallback basis for every assignment.

## Corpus contract

The 1M training plan should preserve independently controlled lanes for:

- explanatory prose and dialogue;
- mathematics and scientific notation;
- multilingual text, including Chinese and multiple scripts;
- code and repository-agent trajectories;
- tool/API calls and structured extraction;
- short general tasks and reasoning; and
- long technical or multi-turn contexts.

Use the source-controlled material described in
`docs/dense-h-corpus-plan.md`. Every source shard must retain its upstream
identity or local manifest, revision, normalization version, byte hash, and
source lane.

Split complete source records by content hash before token truncation. Also
hash the final token sequence after chat templating and truncation. Training,
mode selection, and final validation must have zero raw-record and zero prompt
overlap. Final validation is never used to fit scales, Hessians, permutations,
codebooks, mode margins, or X4T allocation.

## Plan validation

Generate plans with `scripts/run_interim_calibration_corpus.py --dry-run`, then
validate all reports together:

```bash
cd /home/luke/projects/kquant

.venv/bin/python scripts/validate_calibration_corpus_plans.py \
  <training-report> <selection-report> <final-validation-report> \
  --output out/qsrt-corpus-integrity.json
```

The immediate training command follows this shape:

```bash
.venv/bin/python scripts/run_interim_calibration_corpus.py \
  --source <prose.jsonl=weight@cap> \
  --source <math.jsonl=weight@cap> \
  --source <multilingual.jsonl=weight@cap> \
  --source <agent.jsonl=weight@cap> \
  --target-tokens 1000000 \
  --fold-modulus <N> --fold-index <I> --fold-mode exclude \
  --model-dir /models/Kimi-K3-EXL3-3p09-serve \
  --capture-dir <fresh-training-capture> \
  --report <fresh-training-report> \
  --dry-run
```

Inspect realized lane tokens and document counts before removing `--dry-run`.
Do not infer semantic breadth from filenames alone.

## Resident teacher capture

Launch the matching TP12 vLLM tree with a fresh `K3_KQUANT_CAPTURE_DIR` and the
interim EXL3 model. Capture mode must disable prefix-cache reuse and speculative
drafting while preserving the ordinary TP12 execution path and CUDA graph
replay. Do not enable expert parallelism.

The collector has two numerical paths:

- official MXFP4 routes use the normal W4A16 post-SiTU tap;
- trellis routes invert the capture-only intermediate rotation/scale so the
  saved row is in canonical expert coordinates.

The B12X intermediate transform bundle is ordered
`[gate_svh, up_svh, down_suh]`. A different order invalidates SiTU capture even
if simple magnitude checks look plausible.

The launcher may execute an infrastructure probe at capture epoch zero. Corpus
documents occupy the request epochs authenticated by the corpus report. Hessian
construction and replay must exclude every unreported epoch.

All routes from one token share the same observation ID and split. The TP
merger must prove:

- exact input/route/middle observation joins;
- identical expert IDs, gates, and split values across ranks;
- exactly 16 routed expert rows per sampled input row;
- no device-ring or host-part drops; and
- comparable canonical row magnitudes for MXFP4 and trellis routes.

Finalize the capture once, using the sentinel printed by the vLLM launcher and
one final request. Never merge a live capture or restart a server into an old
capture directory.

## Sampling and support

A global route-sampling denominator wastes most storage on hot experts and
leaves the tail unsupported. The 1M run should use a cheap natural-route census
to set document-aware per-expert reservoir targets. Preserve inclusion
probabilities so expected deployment damage can be reweighted correctly.

For each layer/expert assignment record at least:

- natural route count and gate/gate-square mass;
- sampled row count and gate-square effective sample size;
- number of distinct documents and source lanes;
- covariance conditioning/effective rank after shrinkage;
- identity-versus-dense held-out prediction error; and
- neuron-order and `(r13,r2)` stability across document folds.

The first 65,536-token study averaged only a small number of routed rows per
expert. Raising the corpus to 1M is a major support improvement, but it does
not guarantee that every tail expert earns dense H. Identity fallback is an
expected outcome, not an error.

## Candidate and validation separation

The training capture supplies Hessians, importance order, rate-shift proposals,
and the fit/confirmation folds used by the conservative selector. The separate
validation capture serves two purposes:

1. estimate each selected lossy candidate's natural-route damage for the X4T
   allocator; and
2. compare every accepted `(r13,r2)` choice with a reconstructed R0/R0 control
   on the same held-out documents.

Selection must use paired per-document evidence. Coefficient count is not a
statistical sample size. Report traffic-weighted aggregate error, median,
upper-tail regressions, high-traffic worst cases, mode frequencies, and
support-conditioned confidence intervals.

## Acceptance gates

Do not build the next full checkpoint until all of these hold:

- corpus and prompt hashes are disjoint across roles;
- every TP capture manifest is complete and has zero dropped rows;
- official-source replay reconstructs canonical expert coordinates;
- `H2` is expert-stratified and identity fallback is explicit;
- the selected QSRT candidates beat matched R0/R0 on untouched routed replay;
- the X4T exact-byte index closes against official MXFP4 tensors; and
- a fresh materialized subset passes structural, bit-exact, B12X, A16, and KLD
  smoke gates.

Only then launch the all-layer materialization described in `AGENTS.md`.
