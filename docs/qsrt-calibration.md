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
- encode each supported `r13` candidate, replay its decoded `w1/w3`, and build
  `H2[e,r13]` from rows routed to expert `e`;
- shrink each expert estimate toward its own trace-scaled identity according
  to support,

  \[
  \widehat H_{2,e,r13}
  =\alpha_e H^{sample}_{2,e,r13}
  +(1-\alpha_e)\frac{\operatorname{tr}(H^{sample}_{2,e,r13})}{d}I;
  \]

- materialize/factor one expert covariance at a time and discard it after
  encoding; and
- use identity H for an unsupported expert, never another expert's `H2` basis.

`alpha_e` is estimated from gate-square Kish effective sample size through a
weighted OAS-style reliability estimate and capped conservatively. A
layer-global post-SiTU matrix is neither a shrinkage target nor a fallback;
changing or corrupting one must not change any QSRT candidate.

The reusable offline reduction is deliberately asymmetric. Build all 92
layer-global `H13` matrices in one tensor-selective pass over rank-zero input
rows. Store the `H2` identity fallback symbolically, and repack the rank-zero
inputs/routes/gates into one file per layer. Candidate workers encode and
decode each `r13` candidate, recompute its post-SiTU rows, and construct only
the current `(expert,r13)` `H2` on the GPU. They must not rescan or pool the
TP-sharded teacher-middle capture.

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

## Interim production-pool confirmation frequencies

The profile-ID-5 pool provides the first broad production-shaped check that
the conservative selector still finds useful rate shifts after replacing R44
with SQG-Cheb, adding the `w2` K2 virtual-octile reachability, and making `H2`
conditional on decoded `r13` candidates.  At the 2026-08-05 04:59 PDT
snapshot, 44 complete atomic sidecars covered 27,176 unique experts in 33
partly or fully represented layers:

- 6,691 experts (24.621%) selected a confirmed nonzero `(r13,r2)` mode;
- 6,557 (24.128%) selected R1+ on `w2`;
- 5,531 (20.353%) selected R1+ on coupled `w13`; and
- 5,326 (19.598%) selected R2 on at least one axis.

The exact mode histogram was `R0/R0` 20,485; `R0/R1` 266; `R0/R2` 894;
`R1/R0` 125; `R1/R1` 974; `R1/R2` 2,763; `R2/R0` 9; `R2/R1` 18; and
`R2/R2` 1,642.  Every selected nonzero mode in this snapshot cleared the
paired document-bootstrap confirmation lower-bound gate; rejected proposals
fell back to `R0/R0`.

Treat this as an interim selector diagnostic, not an estimate with a random
layer-sampling design.  The pack schedule is work-balanced, split-layer
sidecars finish at different times, and the represented 33 layers are not a
uniform sample of the 92 MoE layers.  Any progress report must therefore:

1. count only sidecars marked `complete`;
2. deduplicate `(layer,expert)` keys across split subsets;
3. report `w13` and `w2` rates separately as well as the joint histogram; and
4. avoid extrapolating the 24.621% incidence to the final pool.

Untouched validation remains a verification and damage-estimation stage.  It
must not be used to retune the modes represented by this confirmation
snapshot, and the later X4T endpoint allocation must not be conflated with an
R0/R1/R2 rate-shift decision.

## Permutation and folded-scale screens

The five-policy proposal from the GLM work was reduced to the four distinct
policies implemented by the Kimi encoder and tested on the 24-expert
production panel with the frozen `w2` K2-octile profile.  `h2_reverse` had the
lowest external routed SSE for every fixed `(r13,r2)` mode and for the
conservatively selected result.  Identity was 0.40% worse at `R0/R0` and
10.60% worse at `R0/R2`; energy-balanced was 0.92% and 13.10% worse;
stratified-energy-balanced was 0.15% and 0.52% worse.  The next pool therefore
keeps `h2_reverse`.

The metadata-free per-128 conditioning grid was then tested at folded-scale
powers `0`, `0.25`, `0.5`, and `1.0` with all other choices fixed.  Every
nonzero strength regressed every complete mode.  At `R0/R2`, powers `0.25`,
`0.5`, and `1.0` increased routed SSE by 0.71%, 1.19%, and 1.77%; their
conservatively selected results were 0.88%, 1.47%, and 2.02% worse.  The next
pool uses folded-scale power zero.  This rejects the tested conditioning
family; it does not prohibit a future differently parameterized scale study.

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
