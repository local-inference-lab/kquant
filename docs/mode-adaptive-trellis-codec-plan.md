# Expert-static rate-transfer trellis codec for Kimi-K3 experts

Status: working design and implementation plan, 2026-08-01. This document is
self-contained, but the experiment history and broader literature notes remain
in `docs/codec-transfer-research.md`.

This is not yet a production format specification or a model-quality claim.
The current quality evidence includes document-disjoint all-retained and
keep-tier-blind streamed-official `w2` studies in layers 1, 24, and 40, plus a
smaller full-expert coupled-mode study. The TP12 reference packer,
stored-stream decoder, coupled `w1`/`w3`/`w2` permutation closure, exact
prototype byte ledger, full `R0..R12` ladder, conservative training-only
selectors, and offline single-matrix official-weight streamer now exist.
The arbitrary record-map gate and document-split expert-local Hessian studies
are complete, and the initial encoder policy is now frozen. Dense fit-fold
Hessians now cover all 92 MoE layers; the offline-streamed all-expert encoder,
strict candidate-pool allocator, and TP12 B12X P24/P33 decoder are implemented.
The bounded-memory candidate writer and allocation-to-TP12 layer-slab
materializer are also implemented, including a reference slab reader,
bit-exact source-to-slab verification, a reproducible matched-R0 external
mode gate, and an independent artifact validator. Materialization now requires
that gate to pass and carries an immutable copy into the artifact and serve
directory.
The final external KLD corpus/tools are pinned and a kquant-owned paired
3p09-versus-candidate, window-clustered promotion gate is implemented; its
39.98-GiB canonical reference payload is staged locally. The streamed official
PyTorch reference can now reconstruct active compressed and kept experts
directly from completed TP12 layer slabs, independently of the serving reader.
The fresh schema-v3 all-expert candidate-pool build is in progress. Its
postprocessing, validation, materialization, all-rank closure, streamed
comparison, paired KLD captures, and owned serving probe are dependency-queued.
Production-scale results, an actually materialized and served artifact,
end-to-end quality validation, and elimination of the remaining routed-kernel
performance gap are still required.

The current phase is **TP12 only**. No TP4 or TP16 design, implementation, or
benchmark is in scope. A TP-independent persistent representation is a later
phase and must not constrain the first TP12 runtime format.

## 1. Decision summary

The proposed successor to uniform EXL3-3 is an **expert-static rate-transfer
trellis codec**. It retains the parts of EXL3 that have measured value—dense
Hessian-aware LDLQ, Hadamard/sign regularization, and Viterbi trellis
quantization—but adds the rate-control structure used by successful image and
video codecs.

The initial design is:

1. Use the resident interim EXL3 model to capture exact routing, applied gate
   weights, conditional activation moments, and raw Hessian samples. Never try
   to keep the official model resident during capture.
2. Treat each routed expert's 3072-dimensional post-SiTU intermediate axis as
   a set of coupled neurons shared by `w1`, `w3`, and `w2`.
3. Rank those neurons offline using routed calibration data, initially in
   small groups, and form partitions aligned to 128-channel records.
4. The reference encoder supports the monotone `R_r` ladder: among
   24 importance-ranked records, the first `r` use K2, the final `r` use K4,
   and the rest use K3, for `0 <= r <= 12`. The old U4, T8, and M4 experiments
   are compatibility names for `R0`, `R3`, and `R6`, not three unrelated
   codec families. The frozen initial artifact alphabet is
   `{R0,R1,R2,R3,R5}`; the full ladder remains available for offline studies.
5. Preserve the complete dense Hessian and LDLQ feedback while varying the
   trellis rate. Do not quantize contexts independently.
6. Build the encoding Hessians from routed fit documents with fixed shrinkage:
   25% expert-conditional input covariance plus 75% layer-global covariance
   for the shared `w1`/`w3` H13, and 75% expert-conditional official-source
   post-SiTU covariance plus 25% layer-global covariance for `w2` H2.
7. Bake one common intermediate-neuron permutation into the output rows of
   `w1` and `w3` and the input columns of `w2`. This is exactly
   function-preserving because SiTU acts coordinatewise.
8. Keep 16x16 as the elementary trellis coding tile, but use a 128-channel
   slab as the mode, storage, movement, and TP-placement record.
9. Store records in fixed K2/K4 or K3/K3 pairs. Both pair types contain three
   trellis bits per weight and give each TP12 rank exactly 256 local channels
   at the same byte budget.
10. Select one shared mode per layer/expert assignment offline using routed
    rate-distortion optimization. The schema retains separate `(r13, r2)`
    values for controlled studies, but the measured 0.078% incremental gain
    does not justify enabling them in the initial artifact.
11. For the in-flight phase-1 artifact, jointly compare the selected compressed
    candidate against the original MXFP4 expert without changing its frozen
    schema. For the successor frontier, **adopt** the bit-exact
    compressed-MXFP4 `X4` representation as the high tier. Uniform K4 is not a
    production endpoint. The high-tier decision and the within-TSH mode
    decision remain distinct.

The design deliberately does **not** optimize for compatibility with the
current EXL3 checkpoint layout. A new tensor schema and kernel contract are
acceptable if they make the decoder regular, balanced, and measurably faster
or better per byte.

## 2. Baseline being replaced

Kimi-K3 has 92 MoE layers, 896 routed experts per MoE layer, and therefore
82,432 layer/expert assignments. In the current `3p09` artifact:

- 7,007 assignments (8.5%) retain their original MXFP4 weights;
- 75,425 assignments are encoded as EXL3;
- every EXL3 `w1`, `w3`, and `w2` matrix uses the same hard-coded `K = 3`;
- the current production artifact used identity Hessians; and
- the keep set was selected using the L0 router-bias traffic proxy.

The encoder does not currently estimate K from weights, Hessians, routing, or
tile statistics. The pack recipe fixes `BITS = 3`, passes that value to every
EXL3 call, and lets Viterbi find the best path conditional on that rate. The
only coarse rate adaptation is the whole-expert choice between retained
MXFP4 and uniform K3.

Ignoring format overhead, the nominal expert-weight rate of that mixture is

```text
0.085 * 4 + 0.915 * 3 = 3.085 bits/weight,
```

which explains the `3p09` label after rounding. Exact physical rate is higher
because MXFP4 scales, EXL sign/scale vectors, manifests, alignment, and index
metadata also consume bytes.

The baseline's central limitation is not that K3 is intrinsically poor. It is
that it spends the same K3 budget on channels whose routed sensitivity can
differ by orders of magnitude.

## 3. Terminology and geometry

### 3.1 Expert matrices and channels

In ordinary PyTorch `[out, in]` order, each K3 routed expert has:

```text
w1: [3072, 3584]
w3: [3072, 3584]
w2: [3584, 3072]
```

The word **channel** in this plan means one coordinate on the shared
3072-dimensional intermediate axis:

- one output row of `w1`;
- the matching output row of `w3`;
- one post-SiTU scalar; and
- the matching input column of `w2`.

It does not mean a token, an expert, a matrix row in every orientation, or an
MMA lane.

The initial format has exactly one supported tensor-parallel geometry:

| TP | Local intermediate channels | 128-channel records | Record pairs |
|---:|---:|---:|---:|
| 12 | 256 | 2 | 1 |

Phase 1 is TP12-only. The schema, reference decoder, CUDA decoder, packaging,
and validation code must reject every other TP size rather than implying
compatibility. A TP-independent storage format is a separate later phase and
is not part of this design's initial implementation.

### 3.2 What K means

K is the fixed trellis payload rate in bits per weight. It is unrelated to
Kimi-K3's name, top-K expert routing, or the inner dimension conventionally
called K in GEMM notation.

EXL jointly quantizes a 16x16 tile, or 256 coefficients, as a path through a
fixed trellis. For a chosen K, the packed path occupies

```text
256 * K bits = 16 * K int16 words.
```

| K | Payload per 16x16 tile | Packed words |
|---:|---:|---:|
| 2 | 64 bytes | 32 int16 |
| 3 | 96 bytes | 48 int16 |
| 4 | 128 bytes | 64 int16 |

K is therefore a path-rate budget, not an independent `2^K`-level scalar
codebook for every coefficient. Viterbi selects the jointly optimized path;
it does not select K.

### 3.3 Three distinct granularities

The design uses three levels deliberately:

| Level | Granularity | Decision |
|---|---|---|
| Expert schedule | one layer/expert assignment | shared `r` or `(r13, r2)` chosen offline |
| Rate record | 128 intermediate channels | K2, K3, or K4 from the mode schedule |
| Coding tile | 16x16 weights | one Viterbi path at the record's K |

A 128-channel record in any of the three expert matrices covers

```text
128 * 3584 = 458,752 weights
```

and contains

```text
(128 / 16) * (3584 / 16) = 1,792
```

elementary 16x16 trellis tiles. Every one of those tiles uses the record's K.

The corresponding ideal trellis payloads are:

| Record | Payload |
|---|---:|
| K2 | 112 KiB |
| K3 | 168 KiB |
| K4 | 224 KiB |

Thus a K2/K4 pair and a K3/K3 pair are both exactly 336 KiB before scale and
alignment overheads.

The 16x16 tile is a **coding** choice, not a claim about the granularity of
all SM120 MMA atoms. A kernel may decode several coding tiles and feed any
legal MMA shapes and operand layouts. The 128-channel record is the unit that
should align storage, descriptors, TMA movement, and TP placement.

## 4. Why a coupled neuron permutation is exact

Let the expert input be a column vector `x`, and write the unquantized expert
as

```text
g = W1 x
u = W3 x
h = SiTU(g, u)
y = W2 h.
```

K3's SiTU is coordinatewise. With `beta = 4` and `linear_beta = 25`, the
implementation is

```text
SiTU(g, u) =
    [beta * tanh(g / beta) * sigmoid(g)]
    .* [linear_beta * tanh(u / linear_beta)].
```

The two branches are not interchangeable, but applying the same permutation
to both branches is valid. For any 3072x3072 permutation matrix `P`, define

```text
W1' = P W1
W3' = P W3
W2' = W2 P^T.
```

Then

```text
SiTU(Pg, Pu) = P SiTU(g, u)
```

because every output coordinate depends only on the two matching input
coordinates. Consequently,

```text
W2' SiTU(W1' x, W3' x)
  = W2 P^T P SiTU(W1 x, W3 x)
  = W2 SiTU(W1 x, W3 x).
```

The unquantized expert function is unchanged. In PyTorch storage this is the
same permutation applied to `w1` and `w3` output rows and `w2` input columns.
If a future architecture adds expert biases or channel-indexed auxiliary
parameters, those must be permuted as well.

The permutation does not improve compression by itself. Its value is that it
turns an arbitrary importance classification into a regular physical layout:

- K2, K3, and K4 channels become complete contiguous records;
- the decoder derives K from a record index instead of reading a per-channel
  map;
- `w2` needs no runtime column gather;
- `w1` and `w3` produce the corresponding physical SiTU order directly;
- TP shards own complete records; and
- records can be paired to equalize bytes per rank.

It is useful to separate two operations that are both informally called
"reordering":

1. **Importance ordering for LDLQ.** This can change distortion. LDLQ makes
   sequential, approximate decisions, so its elimination order matters. For
   `w2`, placing high-importance channels at the end of the permuted input
   axis makes the backward recursion quantize them first. Their error then
   influences compensation of lower-priority channels rather than the other
   way around.
2. **Post-encoding physical record ordering.** This cannot change the already
   fixed reconstruction. It moves complete 128-channel records into P24/P33
   pairs for constant-stride addressing, TP balance, and regular decoding.

Eliminating rate metadata is real but not, by itself, a sufficient reason for
the permutation. A four-context assignment at the experiment's four-channel
group granularity would cost 768 two-bit entries, or 192 bytes per expert. A
rate map for only the 24 final records would cost about six bytes. The larger
benefit is removing the consequences of an arbitrary map: mixed-K tiles,
variable control flow, scattered payloads, and either a gather between the
SiTU output and `w2` or an equivalent indirection inside the GEMM.

Grouping similarly sensitive channels may provide a secondary coding benefit
by making each fixed-K trellis/transform record more homogeneous. Hadamard
mixing and scale search could also make that effect neutral or negative, so
it is an ablation to measure rather than part of the core justification.

Full-precision algebraic equivalence does not excuse implementation closure.
BF16 rounding, transform ordering, scale placement, and TP reduction must all
be tested against the ordinary expert path.

The EXL Hadamard/sign gauges are distinct from this semantic permutation.
They do not commute with SiTU, and the working B12X ABI does not ask them to:

```text
canonical x
  -> w1/w3 input transform -> trellis GEMM -> output inverse transform
  -> canonical g,u -> SiTU
  -> w2 input transform -> trellis GEMM -> output inverse transform.
```

Thus SiTU already sees canonical gate/up coordinates, and its canonical
output is separately transformed for `w2`. This is a verified implementation
invariant rather than an unresolved algorithmic risk. The mixed-rate kernel
must preserve the same order, and closure tests must fail if a hidden-axis
Hadamard, sign vector, or scale is accidentally folded through SiTU.

## 5. Candidate modes and physical schedules

### 5.1 The `R_r` rate-transfer ladder

Rank the 24 complete 128-channel records from low to high proposed
importance. Define `R_r` by

```text
records [0, r)       -> K2
records [r, 24-r)    -> K3
records [24-r, 24)   -> K4,
```

where `0 <= r <= 12`. Every mode has `r` donor records, `r` recipient
records, and exactly three trellis bits per weight. The previous names map to
this one-dimensional family:

| Current name | Ladder name | K2 / K3 / K4 records | Role |
|---|---|---:|---|
| U4 | `R0` | `0 / 24 / 0` | uniform captured-H control |
| T8 | `R3` | `3 / 18 / 3` | mild transfer |
| M4 | `R6` | `6 / 12 / 6` | medium transfer |
| `2,2,4,4` | `R12` | `12 / 0 / 12` | maximum transfer |

The representative study should search all 13 values, or a measured subset
dense enough to reconstruct the frontier. Only after that study should the
small production alphabet be selected. The current two-mode table is a
prototype, not a format decision.

On a manageable subset, also encode an equal-count unconstrained record-map
oracle. Its donor and recipient records need not be the tails of the initial
scalar importance order. The gap between that oracle and the best monotone
`R_r` candidate is **ranking/partition regret**; mode-table regret is the
separate gap between the full `R0..R12` ladder and a restricted set of `r`
values.

At 128-channel granularity, an arbitrary *shared* donor/recipient map does not
itself require hot per-record metadata. The selected records can be placed at
the K2 and K4 ends of a new common neuron permutation and then packed by the
ordinary `R_r` P24/P33 layout. This makes the oracle potentially deployable,
not merely an unreachable upper bound, provided `w1`, `w3`, and `w2` share
the map. Separate matrix-specific maps would break that property or require
additional runtime reordering.

### 5.2 Common permutation, optionally separate schedules

The same ranked physical neuron order is mandatory for `w1`, `w3`, and `w2`.
Their rate thresholds need not be identical. The initial comparison is:

```text
shared:   (r13, r2) = (r, r)
two-mode: r13 chosen for fused w1/w3; r2 chosen independently for w2
oracle:   separate r1, r3, and r2, measured only as an upper bound.
```

Separate `(r13, r2)` does not introduce a SiTU gather: every matrix uses the
same physical neuron permutation, and only the record decoder schedule
differs. Tying `w1` and `w3` preserves a regular fused gate/up kernel. Whether
one shared `r` is sufficient is empirical, not an algebraic consequence of
the shared permutation.

### 5.3 Encoding order and storage order are different

LDLQ processes the relevant input axis backward. For `w2`, the encoder puts
the contexts in low-to-high importance order so the most important context is
fixed first and its error is visible to later compensation. The `w2` Hessian
is transformed consistently as

```text
H2' = P H2 P^T.
```

This preserves the quadratic distortion geometry. Splitting the contexts
into independent EXL calls would discard off-diagonal Hessian blocks and is
not acceptable.

Once quantization has produced a fixed reconstruction, complete 128-channel
records can be moved into a TP-balanced storage order without changing that
reconstruction. The same final physical neuron order is baked into all three
matrices.

For `R_r`, final storage contains:

```text
r      * (K2, K4) record pairs
(12-r) * (K3, K3) record pairs.
```

Every TP12 rank receives one 256-channel pair with an average trellis rate of
three bits per weight.

Equal bytes do not prove equal latency. K2 and K4 may have different decode
instruction counts, register pressure, or scheduling behavior than two K3
records. That question belongs to the kernel microbenchmark gate.

If P24 and P33 latency differ, pair ownership should be cyclically rotated by
expert ID so hot experts do not systematically make the same TP12 ranks the
stragglers. This rotation is part of the common physical permutation and must
be applied consistently to all three matrices. It is not a license for a
runtime gather.

### 5.4 Runtime K selection

There is no runtime importance calculation. The experimental metadata stores
either one shared `r` or the pair `(r13, r2)`. A frozen production alphabet
may later remap the retained combinations to smaller IDs. The decoder performs
the conceptual lookup

```text
K = schedule[matrix_family][expert_mode][physical_record_index].
```

The selected K then applies to every 16x16 coding tile in that record. A
per-tile or per-channel K map is intentionally absent.

## 6. Why uneven rate can beat uniform K3

### 6.1 The discrete exchange argument

As a motivating separable approximation, let `D_c(K)` be the routed
distortion attributed to record `c` when encoded at K. Replacing low record
`l` and high record `h` from K3/K3 to K2/K4 preserves the trellis payload and
looks beneficial when

```text
D_l(2) + D_h(4) < D_l(3) + D_h(3),
```

or equivalently when

```text
D_l(2) - D_l(3) < D_h(3) - D_h(4).
```

The left side is the damage caused by removing one bit from the low context;
the right side is the damage recovered by adding one bit to the high context.
Uniform K3 is favored when these marginal rate-distortion slopes are similar;
unequal slopes motivate transferring rate.

This is the discrete analogue of transform-codec bit allocation or
water-filling. High-rate quantization models often approximate distortion as
decreasing exponentially with rate and proportional to a source-sensitivity
constant. That model predicts more bits for more sensitive coefficients. EXL
is a stateful trellis quantizer inside a sequential LDLQ recursion, so
`D_c(K)` is not an intrinsic additive curve under a dense Hessian. The
inequality is a proposal heuristic, not an exact selection rule.

### 6.2 Hessian-aware distortion

For a linear matrix `W`, input covariance or Hessian `H`, quantized matrix
`Q`, and error `E = W - Q`, the standard local distortion proxy is

```text
D(Q) = trace(E H E^T).
```

Identity-H weight MSE assumes every input direction occurs equally and is
equally important. Routed experts violate both assumptions. Conditional
activation statistics and applied gate weights can change the useful metric
by orders of magnitude.

LDLQ uses the dense Hessian to feed quantization error from already-fixed
coordinates into later decisions. If contexts are encoded independently,
all cross-context terms in `H` disappear. The pilot confirmed that preserving
the full Hessian is essential; context-specific codebooks without dense-H
feedback were not competitive with captured-H EXL.

More explicitly, partition `w2` error and its input Hessian by records:

```text
E = [E1, ..., EC],  H = [Hcd]

trace(E H E^T)
  = sum_c trace(Ec Hcc Ec^T)
  + sum_{c != d} trace(Ec Hcd Ed^T).
```

Changing K for one `w2` record also changes the LDLQ feedback received by
later records. Therefore the exact exchange score is a complete
counterfactual encode:

```text
Delta_e(l,h | k) =
    D_e(Q_e(k with l->2 and h->4)) - D_e(Q_e(k)).
```

Cheap donor costs and recipient gains may propose candidates, but only the
full heterogeneous encode and held-out functional score authorize a mode.

The matrix axes matter. For `w2`, records lie on the Hessian-coupled input
axis, so independent calls discard `H2` cross-record structure. For `w1` and
`w3`, records lie on the output-row axis while `H13` is over the shared
3584-dimensional input. Under the standard identity output metric, those
rows are separable and may use different K while each retains the complete
input Hessian. They become output-coupled only when a SiTU/downstream Fisher
or other non-identity output metric is introduced. The implementation and
claims must preserve this distinction.

### 6.3 Why the final decision must be functional

Neither weight MSE nor the encoder's current Hessian proxy is a reliable
expert-mode selector by itself. Quantization errors pass through SiTU, `w2`,
the routing gate, and the routed expert mixture. The authoritative distortion
must therefore be evaluated on captured routed activations.

For expert `e`, mode `m`, routed sample `t`, and applied gate `a_te`, an
isolated conditional objective can be written as

```text
D_e(m) =
  sum_t a_te^2 ||f_e(x_t) - fhat_e,m(x_t)||_M^2
  ------------------------------------------------,
  sum_t a_te^2 ||f_e(x_t)||_M^2
```

where `M` is initially the identity output metric and may later be replaced
by a downstream Hessian approximation. A stronger closed-loop objective
compares the complete reconstructed routed mixture

```text
sum_e a_te fhat_e,m_e(x_t)
```

against the paired pre-RMSNorm teacher target. This preserves interactions
and error cancellation among the 16 selected experts.

### 6.4 Why a measured small mode table is preferable to fine-grained bit maps

An unconstrained 16x16-tile optimizer could assign K2/K3/K4 more precisely,
but it would require a large map, irregular offsets, divergent decode paths,
and poor random access. That objection does not apply to arbitrary selection
among the 24 complete 128-channel records when one common permutation can
bake the selected shared map into all three matrices. Image and video codecs
still show the value of a small set of legal *payload schedules*: the encoder
may search broadly over which neurons occupy each record while the decoder
sees only P24/P33 pairs and a small `r` alphabet.

The full `R0..R12` ladder is the encoder's schedule search space, not
necessarily the production decoder alphabet. After measuring the frontier,
retain the smallest subset whose validation loss is acceptably close to the
best ladder candidate. Independently measure whether optimizing the shared
record permutation closes material ranking/partition regret. The physical
permutation removes that record map from the served hot path; a provisional
pair of four-bit fields can represent all `(r13, r2)` combinations while the
final alphabet remains unfrozen.

## 7. Mode selection and global allocation

### 7.1 Two decisions, not one

The complete encoder makes two nested decisions:

1. **Within the compressed tier:** which validated `(r13, r2)` schedule should
   this expert use?
2. **Across the whole model:** should this expert use its best compressed
   candidate or remain in original MXFP4?

The first redistributes a nominal three-bit budget inside an expert. The
second spends additional model bytes on the assignments where compression
causes the greatest traffic-weighted damage.

### 7.2 Per-expert mode objective

For a candidate mode `m`, use a rate-distortion-compute objective of the form

```text
J_e(m) = D_e(m)
       + lambda_B * B_e(m)
       + lambda_T * T_e(m)
       + rho * U_e(m).
```

Here:

- `D_e(m)` is routed functional distortion on a selection fold;
- `B_e(m)` is exact stored bytes including mode metadata, offsets, scales,
  alignment, and padding;
- `T_e(m)` is measured or conservatively modeled decoder cost; and
- `U_e(m)` penalizes calibration uncertainty, low route counts, unstable
  rankings, or train/validation disagreement.

All `R_r` values have the same ideal trellis payload, so `D` dominates the
first experiments. Exact bytes and measured latency must be included before
choosing a production format.

The conservative selection rule should be:

1. Require minimum routed sample support.
2. Encode all enabled candidates from the same source weights and calibration
   fold.
3. Select provisionally on the training/selection fold.
4. Estimate paired per-document deltas on a document-disjoint validation
   fold with a cluster bootstrap or equivalent cluster-aware interval.
5. Choose nonzero transfer only when the conservative confidence bound clears
   a preset margin; otherwise fall back to `R0`.

Mode selection is offline and fixed in the checkpoint. It is never performed
per token.

### 7.3 Coupled three-matrix selection

The current pilot selects a mode for `w2` only. The production target couples
all three matrices through one physical neuron order and one functional
objective, but it must compare a shared `r` with separate `(r13, r2)` rather
than assuming their best schedules coincide.

For every candidate pair, the encoder must therefore:

- apply one common neuron permutation;
- encode `w1`/`w3` with `R_r13` and `w2` with `R_r2`;
- run the actual SiTU expert function on captured inputs; and
- score the reconstructed expert output or routed mixture.

The reference encoder now varies K on the correct logical intermediate axis
for all three matrices. Representative functional selection remains open.

The first encoder should quantize `w1` and `w3`, regenerate the candidate's
actual `hhat`, and at least test quantizing `w2` with the covariance of that
actual input. A later, explicitly regularized ablation may quantize `w2`
against the teacher expert target so it compensates upstream error. That
target-aware reconstruction deliberately changes the best linear center and
must not be adopted without document-disjoint validation.

### 7.4 Model-wide allocation

After each expert has a selected compressed candidate, choose MXFP4 keeps by
solving the discrete constrained problem

```text
minimize   sum_e expected_damage(e, selected_format_e)
subject to sum_e exact_bytes(selected_format_e) <= model_budget.
```

The candidate pool should be generated before allocation so keep fractions,
traffic estimates, and Lagrange multipliers can change without requantizing
the whole model. Expected damage must combine conditional reconstruction
error with exact measured route mass and applied gate-square mass.

The additive solution should then be refined layerwise against cached routed
mixture residuals. Because only 16 experts are active per token, coordinate
updates can include expert-error cross terms and cancellation without making
mode choice token-dependent. Live-routing end-to-end validation remains the
final check because the offline replay initially freezes teacher routes.

## 8. Importance ranking and record construction

The production alphabet is not yet fixed, but each expert needs one offline
mapping from its original intermediate neurons to 24 ranked records.

### 8.1 Calibration inputs

The implemented capture pipeline can collect from the resident interim EXL3
teacher:

- exact route counts;
- applied gate sums and gate-square sums;
- conditional `w1`/`w3` input moments;
- canonical post-SiTU `w2` input moments;
- raw `w1`/`w3` input samples;
- raw per-route `w2` input samples with expert IDs and gates; and
- paired routed-latent teacher targets.

The production target remains approximately ten million executed tokens from
a representative mixture, with prompts/documents split deterministically so
selection and validation do not share documents. Capture runs the interim
model; official checkpoint weights are only an offline, streamed encoder
source later.

Keep two logical calibration views:

- a **natural-routing view** for production route mass, gate-square mass,
  expert co-occurrence, expected damage, and keep allocation; and
- an **expert-support view** that stratifies or oversamples retained rows for
  tail-expert ranking and covariance support.

Any balanced/support sample must be reweighted to natural routing before it
contributes to expected production damage. The current implementation builds
one dense `H13` and `H2` pair per layer, not per expert assignment. Those
matrices cost exactly 85 MiB per layer and 7.64 GiB for all 92 MoE layers;
the 7.35-TB estimate applies only to a future per-expert dense-H design, which
is not the current plan.

The current all-layer fit bundle is already materialized. It uses 94 of the
115 training documents, leaves 21 training-corpus documents for independent
mode confirmation, and never touches the separate 27-document external
validation capture. Every layer has 12,292 H13 rows; H2 support ranges from
12,957 to 13,597 routed rows. Its 92 FP32 layer files total 8,199,879,040
bytes.

### 8.2 Initial ranking statistic

The simplest initial middle-channel score is gate-square-weighted conditional
post-SiTU energy,

```text
s_ej = E[a_e^2 * h_ej^2 | expert e is routed].
```

Channels should initially be ranked in four-channel groups to reduce noise
and preserve useful local structure. Context boundaries are then rounded or
constructed so every final partition consists of complete 128-channel
records.

Activation energy is a proposal generator, not the final mode objective. A
stronger coupled saliency estimate should account for:

- the norm and geometry of the corresponding `w2` column;
- derivatives of both SiTU branches on routed samples;
- quantization residuals of the matching `w1` and `w3` rows;
- off-diagonal Hessian interactions; and
- the applied router gate and route frequency.

The practical proposal hierarchy is:

1. Rank four-channel groups initially by activation energy and form 24
   complete 128-channel records.
2. On a smaller sample, measure conditional donor cost
   `C_j = D_j(2) - D_j(3)` and recipient gain
   `G_j = D_j(3) - D_j(4)`.
3. Use those slopes, derivative/output geometry, and support to propose a
   revised order.
4. Run the complete heterogeneous full-H encode for every retained `R_r`
   candidate.
5. Select only by held-out functional distortion.

The donor/recipient values are proposal statistics rather than additive
truth. Also ablate the semantic LDLQ order: high-importance first,
low-importance first, random, Hessian diagonal, and a pivoted-Cholesky or
Schur-complement order. Backward traversal alone does not prove which
semantic order is best.

### 8.3 Statistical safeguards

Long-tail experts will not all receive equal support. The encoder must record
route counts and effective sample size, shrink noisy moments toward a layer
or expert-family prior, regularize dense Hessians, and use `R0` when estimates
are unreliable. Record distinct-document count, gate-square effective sample
size, covariance effective rank, post-regularization condition number,
mode-effect standard error, and train/validation ranking agreement.

Primary summaries are traffic- and gate-square-weighted arithmetic excess
distortion, median paired improvement, high-percentile regressions, and the
worst regression among high-traffic experts. Geometric-mean ratios remain
descriptive only because tiny denominators can make them unstable.

The interim teacher is the only resident model. Before global encoding, run a
limited **offline-streamed** official comparison on selected documents and
layers: route overlap, gate correlation, hidden/post-SiTU covariance,
record-ranking correlation, selected modes, and keep value. If agreement is
poor, progressively recapture from a newly built artifact; never make the
official checkpoint resident merely for calibration.

## 9. Proposed checkpoint and bitstream contract

This section defines the intended structure, not final tensor names.

### 9.1 Global metadata

A new schema version should record:

- codec name and schema version;
- experimental `R_r` ladder and any frozen production mode table;
- supported K values and trellis codebook identifier;
- coding-tile shape, record width, and pair layout;
- alignment and padding rules;
- explicit TP12-only runtime geometry;
- Hessian/capture provenance and encoder revision;
- scale/sign representation and shared-vector contract; and
- exact byte ledger by tensor class.

Do not overload the current uniform-EXL manifest and infer mixed-rate behavior
from tensor shapes accidentally.

### 9.2 Per-expert metadata

The served hot path should need only:

- retained-MXFP4 versus compressed-format selection;
- provisional `(r13, r2)` values or a later frozen mode ID;
- base payload address or fixed record-pair offset; and
- any matrix/rank descriptor that cannot be derived from the global mode.

There should be no served per-channel context map. For reproducibility, the
offline artifact may retain the original-to-physical permutation or its
derivation metadata in a cold sidecar. The serve directory does not need to
load that map because all three matrices are already physically permuted.

### 9.3 Fixed-size pair containers

The materialized reference schema is `kquant_mixed_exl3_tp12_proto_v3`. It is
deliberately TP12-only and rejects any other runtime TP size. The reusable
candidate stream is allocation-independent and keeps twelve pairs in logical
neuron order. A materialized matrix payload contains those twelve pair
containers in physical-rank order. Each pair holds 256 intermediate channels
and its trellis portion always contains the same number of bytes:

```text
P33 = K3 record || K3 record
P24 = K2 record || K4 record.
```

For Kimi-K3's `[3072, 3584]` / `[3584, 3072]` expert matrices, the axis
orthogonal to the rate axis contains 224 coding tiles. One 128-channel record
therefore contains `8 * 224 = 1792` elementary 16x16 tiles. The exact sizes
are:

| Object | int16 words | bytes |
|---|---:|---:|
| K2 record | 57,344 | 114,688 |
| K3 record | 86,016 | 172,032 |
| K4 record | 114,688 | 229,376 |
| P24 or P33 pair | 172,032 | 344,064 |
| one matrix, 12 pairs | 2,064,384 | 4,128,768 |
| all three expert matrices | 6,193,152 | 12,386,304 |

Logical pair `p` starts at word `p * 172032` in the allocation-independent
candidate. P24's internal boundary is after 57,344 words; P33's is after
86,016 words. The pair type comes from the expert mode and logical pair index,
so neither pair offsets nor record offsets are stored.

The current all-expert workers were launched with the explicit logical
descriptor `kquant_mixed_exl3_tp12_proto_v2`. That is not silently relabeled.
Proto-v3 changed only the outer logical-pair-to-physical-rank placement; the
logical candidate bytes, record order, and trellis interpretation are
identical. Pool finalization validates one consistent v2-or-v3 logical source
schema and includes it in the pool-content digest. Held-out scoring decodes
under that recorded source schema. Materialization then records the source
schema, applies the v3 rank rotation below, and emits only a v3 layer header.
This is an explicit source-to-output conversion, not a mixed-schema artifact.

Schema v3 distributes mixed-mode work across TP12 ranks without metadata. For
layer `l`, expert `e`, and physical rank `q`, the stored logical pair is

```text
p = (q - ((5e + l) mod 12)) mod 12.
```

Multiplication by five is a permutation modulo twelve. Thus every expert still
places each coupled 256-neuron pair exactly once, while consecutive expert IDs
rotate P24 ownership nearly perfectly across ranks. The same mapping is used
for `w1`, `w3`, `w2`, and all three local scale slices, so it is simply an
offline TP partition of the common physical neuron permutation: it requires no
SiTU-boundary gather and does not alter the expert function. The current
layer-layout prototype includes fixed 4-KiB sections, rank-major payloads,
pooled scale storage, alignment padding, and an exact byte ledger. Its metadata
table is deliberately an experimental superset for `(r13, r2)`; it must not be
mistaken for a frozen production alphabet.

The materialized layer ABI is, in order: one 4-KiB header, one 4-KiB format
section, one 24-KiB layer-shared scale section, and twelve fixed-stride rank
sections. Within each rank section, compressed experts appear in ascending
expert ID and store `w1`, `w3`, then `w2` pair payloads; their local FP16
scale slices follow in the same matrix order. Retained experts then appear in
ascending expert ID. Each retained rank-local matrix stores the official
`weight_packed` bytes first and the official E8M0 `weight_scale` bytes second.
All integral payloads are little-endian, padding is zero, and tier-local slots
are derived from the format table rather than stored as an offset array.

The logical rate axis in EXL encoder orientation `[K, N]` is:

```text
w1: N
w3: N
w2: K
```

Within a record, tiles retain `[k_tile, n_tile]` row-major order. A K-axis
record spans eight consecutive K tiles and every N tile. An N-axis record
spans every K tile and eight consecutive N tiles. Each tile is then packed
exactly like native EXL: sixteen 16-value spans, K low edge bits per value,
big-endian bit concatenation into K 16-bit words per span, followed by the
native adjacent-word `SWAP16` ordering.

This is a validated reference layout, not yet the final TMA transaction
layout. B12X may add pair-level alignment or an equivalent fixed-stride
physical swizzle, but any change must preserve the schema's exact reference
mapping or introduce a new schema version.

### 9.4 Scale and transform data

The codec still needs the EXL input/output sign or scale representation and
the selected trellis codebook. Any channel-indexed vector must follow the
same physical permutation as its matrix axis. Moving records must never
separate a trellis payload from the scales used to reconstruct it.

The reference implementation therefore postpacks `svh` with `w1`/`w3`
records and `suh` with `w2` records. The other vector remains on the hidden
axis and is not moved.

The existing safe transform order is part of the ABI: `w1`/`w3` complete
their output inverse transforms before SiTU, and `w2` begins a separate input
transform after SiTU. Only the common coordinate permutation spans the three
semantic matrices. No dense Hadamard, sign flip, or diagonal scale may be
treated as though it commutes through SiTU.

The current experiment retains K3 global-scale search while changing only
the tile quantizer's K. Production candidates must test whether scale search
needs to be mode-aware or K-aware. Its exact metadata cost belongs in the
rate ledger.

Shared transform metadata must also be bitwise shared, not merely
algebraically equivalent. The schema-v3 encoder therefore never divides and
then re-multiplies the shared hidden-side scale during global-scale search;
the inverse factor is placed directly in the expert-local vector. Input and
output sign draws use independent deterministic layer/matrix seeds so cached
Hessian finalization and the number of evaluated R modes cannot change the
transform attached to a selected payload.

## 10. Encoder plan

### Stage A: representative capture

1. Launch the interim `/models/Kimi-K3-EXL3-3p09-serve` teacher under TP12
   with capture enabled.
2. Use a diverse corpus and document-disjoint training and validation folds.
3. Capture approximately ten million executed tokens with the documented
   route, moment, raw-Hessian, and paired-target sampling rates.
4. Finalize every TP shard and reject any capture with dropped rows,
   mismatched observation joins, incomplete layers, or tier-dependent scale
   anomalies.
5. Merge shards and build dense `H13` and `H2` matrices from training rows
   only.

The 65,536-token training and 16,384-token validation captures described in
Section 12 now bridge this stage for representative experiments. They prove
the document/provenance and dense-H analysis path across all 92 MoE layers,
but they do not replace the planned production-scale capture. In particular,
the 16,384-token validation corpus is now classified as a **pilot validation
gate only**: it may validate the scorer, expose a clear codec failure, and
estimate runtime, but it must not freeze the production keep set. Before
materialization, repeat selected-candidate and matched-R0 scoring on a fresh
document-disjoint validation capture with at least 131,072 prompt tokens and
at least 200 whole documents. Preserve the 2:1:1 diverse/deep/agentic token
mix, explicitly report code/prose/multilingual/tool-use coverage, and exclude
every document used by either the training capture or this pilot validation.
The earlier all-rates-one preflight proved only TP joins and capture plumbing.

### Stage B: layer-scale candidate study

1. Start with every available retained expert in representative layers 1,
   24, and 40, then expand to a stratified sample that is not biased by the
   current keep tier.
2. Construct one stable 24-record common importance order.
3. Encode the `R0..R12` ladder for all three matrices, first with shared `r`
   and then with separate `(r13, r2)` using cached per-matrix candidates.
4. Preserve dense-H feedback and use identical source weights, seeds, scale
   search, and evaluation samples across modes.
5. Compare activation, measured donor/recipient, and derivative/output-aware
   rankings; measure monotone-ladder regret against an unconstrained
   equal-count record-map oracle on a smaller subset.
6. Compare teacher `H2` with candidate-`hhat` covariance for `w2`, and keep
   target-compensating `w2` reconstruction as a later regularized ablation.
7. Select on the training fold and report untouched validation results,
   stratified by traffic, energy skew, route count, layer, and expert family.
8. Fit only conservative predictors used to skip obviously unhelpful mixed
   candidates; never replace functional validation with the predictor.

The go/no-go result is a positive document-clustered lower confidence bound
for traffic-weighted routed-mixture improvement over captured-H `R0` at the
same exact physical rate—not whether a hand-picked expert improves.

### Stage C: mode-alphabet selection

Plot distortion against `r`, support, sensitivity skew, and kernel cost.
Measure the regret of shared `r`, separate `(r13, r2)`, the retained ladder
subset, and the arbitrary record-map oracle. Freeze the smallest production
alphabet only after these results. The provisional metadata/schema may
change at this gate without invalidating the fixed P24/P33 payload primitive.

### Stage D: reference format and closure

1. Implement a standalone mixed-K packer and CPU decoder for K2/K3/K4
   trellis records.
2. Support record-rate maps on `w2`'s logical input columns and `w1`/`w3`'s
   logical output rows.
3. Emit fixed-size P24/P33 pair containers with explicit schema metadata.
4. Prove that unpacking reconstructs every cyclic trellis state and the
   serialized-scale decoded weight with the expected FP16 serialization
   tolerance.
5. Prove unquantized permutation closure for SiTU and quantized reference
   closure for every mode.
6. Produce an exact byte ledger including all descriptors, scales, padding,
   indexes, and alignment.

No full checkpoint should be generated before this reference closure passes.

### Stage E: SM120 kernel

Implement a new B12X path around the new format. The kernel should consume a
pair descriptor, decode its two records at the prescribed K values, and feed
hardware-appropriate MMA atoms. It should not materialize a runtime neuron
permutation or perform a per-channel gather.

Microbenchmark at least:

- uniform P33;
- mixed P24;
- P33 carried by a mode-bearing expert;
- real mixtures of routed `R_r` experts, including distinct `r13`/`r2`;
- all twelve TP12 rank-local pair positions; and
- CUDA graph capture and replay.

Measure payload bytes, effective bandwidth, decode instructions, register
use, occupancy, achieved tensor-core throughput, rank imbalance, preparation
time, graph memory, and end-to-end fused-MoE latency. Byte equality alone is
not an adequate performance result.

The TP12 decoder and both benchmark harnesses now exist. Isolated compact
P24/P33 linears are effectively at parity (within about 0.3% across the tested
axes and row counts), which validates the fixed-payload primitive. In the
complete routed MoE benchmark, however, alternating P24/P33 experts remain
about 6-7% slower than uniform P33 for 1, 4, and 16 tokens. Implementation is
therefore complete, but the performance acceptance gate remains open. A
follow-up static audit found that the expert-uniform dynamic-mode choice was
specialized at CTA entry but then converted back to runtime state inside the
repeated staging and register-load helpers. The mode is now propagated as a
compile-time value through those helpers. The revised benchmark adds the
expected sparse-P24 workload, raw interleaved graph timings, exact source
provenance, and CUDA register/local-memory reporting. It now also consumes a
manifest-bound kquant fixture containing all 896 expert mode flags for one
logical rank plus naturally captured route IDs and applied gates. Successive
route windows are copied into graph-stable buffers outside the timed events,
so the gate measures the actual sparse hit/miss distribution rather than a
single frozen route row. Fixture loading and malformed-input rejection have
CPU tests, including an actual cross-repository fixture load. Rebenchmarking
on an idle SM120 device is required before attributing any performance change
to this correction; the active all-expert encode currently owns all GPUs.

Freeze the initial acceptance rule before that rerun. Every case must have
bit-exact eager/graph replay, stable graph-time allocation, available CUDA
resource attributes, and zero reported local-memory spill. For the isolated
pair decoder, the upper bound of the paired 95% bootstrap interval for
`P24/P33` hot latency must be at most `1.03`. For the complete routed kernel,
the promotion comparison is the real captured sparse-mode/route replay against
the same schema carrying uniform P33—not the deliberately adversarial 50/50
alternating case. Across 1, 4, and 16 tokens, its median slowdown must be at
most 1% and the paired 95% upper bound at most 2%, with no logical TP12 rank
outside that bound. The alternating benchmark remains a useful dispatch stress
diagnostic but is not weighted as though half of production routes use P24.

### Stage F: teacher-proxy validation

Compare interim-teacher traces with an official checkpoint streamed offline
one layer at a time on selected documents. Either justify the interim proxy
quantitatively or perform one progressive recapture from a fresh generated
artifact. The official model is never loaded resident.

The first, deliberately small anchor is complete across all 92 MoE layers for
the existing 12-token correctness prompt. It is descriptive only. The next
run is now specified and queued: twelve independent 128-token validation-fold
documents, stratified 6/3/3 across the source corpora, are batched without
cross-document attention and traced at layers 1, 24, and 40. Exact suite hashes
bind both streamed runs. The suite gate uses whole-document bootstrap bounds
and pre-registered minimums of 0.85 route retention, 0.90 sparse-gate cosine,
0.75 record-rank Spearman, and one-third `R3` donor/recipient overlap. The run
also gates pooled per-expert coverage, rank correlation, and `R3` overlap so a
layer aggregate cannot conceal expert-level disagreement. It starts only after
candidate encoding and held-out scoring release the GPUs.

### Stage G: all-expert candidate pool and allocation

1. Stream official source weights offline through the encoder; do not load
   the official model as the resident calibration teacher.
2. Quantize all 82,432 assignments into reusable retained-ladder candidates,
   subject to
   practical candidate-pruning results from Stage B.
3. Record exact functional damage, residual summaries, mode, rate, and kernel
   cost estimates.
4. Select the best compressed mode conservatively.
5. Decode the persisted selected candidates on the untouched validation
   capture and estimate document-disjoint keep-tier damage without changing
   any mode.
6. Compare selection-corpus and validation rankings, then solve the global
   compressed-versus-MXFP4 allocation under the target byte and runtime
   budget. Prefer the validation damage when support and ranking stability
   close.
7. Materialize a fresh artifact and serve directory under a new tag and
   schema. Never reuse the current `3p09` destination.

The candidate-pool builder implements this flow with atomic layer outputs,
strict resume checks, one-matrix-at-a-time official-source residency, compact
one-pass capture indexing per worker, deterministic fit/confirmation scoring,
and exact shared-transform checks. Candidate payloads are written through a
predeclared safetensors header as each expert completes, so an 11-GiB layer
shard no longer has to remain resident in host memory. The full build is
running into a fresh candidate destination. During the run,
`scripts/summarize_mixed_exl3_pool.py` reads only atomically completed metric
and selection ledgers. It reports population mode frequencies, support,
fit-versus-confirmation gain, accepted confirmation regressions, NMSE, and
damage distributions without reading active partials or multi-gigabyte
payload contents. These are selection diagnostics, not a substitute for the
untouched validation capture. The allocator validates every layer
payload/metric ledger before solving either an exact keep count or an exact
aligned-byte target; it cannot be run on partial pool output.

Once every atomic triplet exists, `scripts/finalize_mixed_exl3_pool.py`
independently re-derives the selector/header closure and computes parallel
SHA-256 digests for all payload, metric, and selection files. The resulting
completion index commits to one pool-content digest and stable local file
identities. Held-out scoring, allocation, materialization, and artifact
validation all require that same digest; this prevents a manifest-identical
but byte-different candidate source from entering the pipeline.
The completion index also binds the pool-wide logical trellis schema, so the
running v2-logical pool can only be consumed through the audited v2-logical to
v3-physical materialization path.

The reusable document-disjoint scorer is also implemented. It mmaps each
completed candidate layer, decodes only the selected payload, streams the
official checkpoint one expert matrix at a time, and writes an independent
per-document validation-damage sidecar. The scorer executes the compressed
expert directly in its common physical neuron order, so it needs neither an
inverse permutation nor a serving-time gather. The allocator can consume this
strictly bound sidecar with `--validation-scores`; the initial allocation will
compare it with the selection-corpus ranking before choosing the damage
source. Using this capture for the keep allocation does not consume the later
full-model KLD/evaluation datasets used for the final quality gate.
The sidecar loader now computes one canonical SHA-256 over its manifest,
completion marker, and all 92 metric/ledger pairs. The allocation records this
score-set digest, and materialization reopens the sidecar, revalidates every
per-document score, and requires the digest to match.
`scripts/summarize_mixed_exl3_validation.py` compares both damage fields at
the exact old-artifact byte target, including global/per-layer rank
correlation, keep-set overlap, support, and the validation regret of retaining
the selection-corpus ranking. It additionally resamples whole validation
documents and reruns the exact aligned-byte optimizer, exposing keep-set
Jaccard, assignment churn, selection probabilities, and full-validation regret
at the promotion boundary rather than treating one noisy top-7,000 list as
certain.

Selected-candidate scoring answers the keep-tier question, but it does not by
itself prove that a selected `R>0` payload beats the same expert encoded as
uniform `R0`. A separate matched-R0 scorer therefore re-encodes every selected
mixed assignment with the identical dense Hessians, shared-scale scope,
permutation, signs, and quantizer source. It first requires exact reproduction
of the stored training R0 metrics and shared hidden transforms, then scores
the counterfactual on the untouched documents. Uniform assignments reuse their
already authenticated R0 score. The policy gate resamples whole documents and
requires both the aggregate and every selected `R_r` group to have positive
point improvement and a positive 95% lower bound. External validation decides
only whether the frozen codec policy is acceptable; it never retunes individual
expert modes. The summary loader reconstructs the decision from the complete
score sidecar, so editing the summary cannot manufacture a pass.

A persisted-payload scorer smoke also closes the real path on the held-out
capture. Layer 10/expert 0 from the earlier stream-writer shard had 70 routed
rows across 25 validation documents; its stored R0 candidate decoded and
compared with one-matrix-streamed official MXFP4 at finite routed NMSE
0.00464. This is a data-path closure result, not a population quality result.

### Stage H: production validation

Run, in order:

1. structural validation of every expected tensor and descriptor;
2. CPU/reference pack-decode equality;
3. unquantized and quantized SiTU permutation closure;
4. source-MXFP4 and mixed-trellis numerical closure at TP12;
5. streamed PyTorch layer and final-logit comparison;
6. owned full TP12 vLLM correctness probes with kernel-path auditing;
7. representative perplexity, reasoning, code, tool-use, and long-context
   quality evaluation; and
8. throughput, latency, memory, and graph-replay comparison against both the
   current `3p09` artifact and a captured-H uniform-K3 control.

For the final end-to-end quality gate, integrate the external
[`kimi-k3-full-mxfp4-kld-reference-32x2048`](https://huggingface.co/datasets/festr2/kimi-k3-full-mxfp4-kld-reference-32x2048)
reference dataset and the Kimi-K3 evaluation tools in
[`rtx6kpro/models/kimi-k3/tools`](https://github.com/local-inference-lab/rtx6kpro/tree/docs/kimi-k3-tp16-exl3-onegrid-20260801/models/kimi-k3/tools).
These are final-model gates, not calibration data and not inputs to codec-mode
selection. Reuse the evaluation tooling only; the initial codec, artifact, and
kernel work remain TP12-only regardless of that repository branch name.

Pin that evaluation contract rather than following either moving branch:

- Hugging Face dataset revision
  `097b2775900c0940d31c6469c2e930be8d17b2f8`;
- rtx6kpro tools commit
  `ec33b89f3c3ef74557d01897587d9045d3317cc4`;
- suite-token SHA-256
  `a6856e1d0504fd00d13c67a5515c081f349088664d7ea0894dc4d15db2c7d209`;
- 32 independent 2,048-token windows, 65,504 scored positions, and 163,840
  logits per position; and
- `KL(full original MXFP4 reference || candidate)` as the primary direction.

The tool copies stored in that dataset revision were verified byte-for-byte
against the pinned rtx6kpro commit on 2026-08-01. The reference logits were
captured at TP16, but that does not make TP16 a codec or kernel requirement:
they are a fixed functional target. Both 3p09 and the new artifact must be
captured at TP12, under the same owned vLLM/B12X runtime, so their paired
difference does not confound the codec comparison with a baseline TP change.

The upstream comparator answers how far one run is from full MXFP4. It does
not directly test whether the new artifact improves on 3p09. Capture and
finalize 3p09 and the candidate separately, passing an owned serving-run JSON
to `--runtime-manifest`, then create both upstream reports. Finally run:

```bash
.venv/bin/python scripts/summarize_k3_kld_gate.py \
  --reference-root /data/kld/kimi-k3-reference \
  --baseline-dir /data/kld/3p09/ref \
  --candidate-dir /data/kld/mixed-tp12/ref \
  --baseline-comparison /data/kld/3p09/kld-vs-full-mxfp4.json \
  --candidate-comparison /data/kld/mixed-tp12/kld-vs-full-mxfp4.json \
  --baseline-label 3p09 \
  --candidate-label mixed-exl3-tp12 \
  --output out/kld-3p09-vs-mixed-exl3-tp12.json
```

This kquant-owned gate revalidates the suite/token/logit manifests, requires
serving provenance, binds the published suite, reference manifest, and tool
files to their pinned SHA-256 values, rehashes every selected logit payload,
and resamples whole windows on the paired quantity

\[
\Delta_i = \operatorname{KL}_i(\text{MXFP4}\|\text{3p09})
          - \operatorname{KL}_i(\text{MXFP4}\|\text{candidate}).
\]

Positive \(\Delta\) is improvement. Its default KLD subgate passes only when
the paired 95% lower confidence bound is positive, the mean improvement
exceeds the reference's documented `0.004` same-model runtime-noise scale, and
no domain has a negative point estimate. A statistically supported overall or
domain regression fails; ambiguous or sub-noise results require another
capture. Corpus bootstrap uncertainty and runtime nondeterminism are reported
separately because resampling windows cannot measure the latter. Use windows
0--7 only as an early corruption gate; all 32 are mandatory for promotion.
`--skip-logit-rehash` exists only to shorten a development smoke and is not
valid for a promotion report.
Run the same paired analysis against a materialized captured-H uniform-`R0`
control to isolate the mixed-rate gain from the gain due to replacing identity
H alone.

### Stage I: three-to-four-bpw expert frontier

After the in-flight three-bpw pool has produced its independently useful
phase-1 checkpoint, replace the old binary K3-versus-uncompressed-MXFP4
allocation with a measured per-expert frontier. Do not assume in advance that
the upper endpoint must be another lossy trellis code.

Official MXFP4 uses four E2M1 index bits plus one UE8M0 scale byte per 32
weights, or 4.25 physical bpw before outer alignment. The scale byte stream is
highly redundant. A new exact reference codec leaves every four-bit index
unchanged and codes each scale row as:

- one dominant exponent plus a direction bit for its normally adjacent second
  exponent;
- one selector bit per scale;
- a two-bit common exception count with explicit count escapes;
- sparse `(position, exponent)` exceptions; and
- one independently decodable offset per 16-row tile.

Across every expert in layers 1, 24, and 40, this format reconstructed all
8,064 matrix scale planes bit-for-bit at 4.03819 bpw aggregate. Only 203 of
26,148,864 rows needed a non-adjacent-palette escape. It is therefore an
exact compressed-MXFP4 endpoint, provisionally named `X4`, rather than a new
quantizer. Layer-specific rates were 4.03804, 4.03787, and 4.03867 bpw.

Uniform transformed K4 TSH remains a useful systems control at 4.00967 bpw.
The completed layer-24 study encoded 887 supported experts from official
MXFP4 source weights and assigned the nine experts with insufficient
confirmation-document support to the exact fallback. On 131,072 untouched
validation tokens, K3 and K4 had routed-expert NMSE of 1.07458% and 0.275955%
respectively. K4 therefore removed 74.32% of K3 error; the isolated
one-expert-replacement mixture NMSE likewise fell from 0.0544322% to
0.0139784%.

Let `r3`, `r4`, and `rx` be the physical rates and let exact `X4` have zero
source-relative distortion. K4 lies on the per-expert lower convex hull only
if

\[
\frac{D_4}{D_3}
\le
\frac{r_x-r_4}{r_x-r_3}.
\]

At the measured rates the right side is about 0.0274, far below the observed
K4/K3 ratio of 0.2568. K4 was off the per-expert lower convex hull for all 887
supported experts under both confirmation and validation routed SSE. Thus a
selected mixture of K3 and exact `X4` has lower distortion than K3/K4 at the
same average bytes, even though `X4` costs only 0.02820 bpw more than K4.

The confirmation-selected held-out frontier is:

| Target bpw | K3/K4 routed NMSE | K3/X4 routed NMSE | X4 reduction |
|---:|---:|---:|---:|
| 3.25 | 0.72727% | 0.61476% | 15.5% |
| 3.50 | 0.52595% | 0.34837% | 33.8% |
| 3.75 | 0.37968% | 0.15495% | 59.2% |
| 4.00 | 0.27775% | 0.01215% | 95.6% |

Selection used only the 21-document confirmation fold; the 223 unique
external-validation documents were used only to report these values and a
validation-oracle regret diagnostic. Unsupported experts default to exact
`X4`. The table conditions NMSE on the 887 supported experts, so the full
layer's zero-error fallback experts can only reduce the error fraction after
their exact bytes are included in the final ledger.

The adopted successor upper-level codec is consequently:

1. mixed-`R` K3 TSH for the lossy tier;
2. exact index-plus-compressed-scale `X4` for the high tier; and
3. expert-static allocation by confirmation-fold routed gain per added byte,
   evaluated only on untouched validation documents.

Pure K4 is rejected as a rate-distortion production mode. K3/K4/K5
(`P35/P44`) remains an optional all-TSH systems experiment only if one decoder
has enough measured runtime value to justify quality below the K3/X4 hull.
Learned FP16 and E4M3 K4 tables have changed the K4 endpoint by only roughly
one to two percent, not the order-of-magnitude reduction needed to reach the
exact-tier convex hull. Keep all high-rate format and kernel work separate
from the initial `P24/P33` production gate so the in-flight checkpoint is not
delayed or invalidated.

The X4 adoption has four remaining implementation gates:

1. freeze its canonical independently decodable container, alignment, and
   malformed-input rules;
2. integrate X4 as a zero-distortion candidate in exact-byte allocation,
   materialization, artifact validation, and serve packaging;
3. implement and benchmark TP12 scale reconstruction feeding the production
   W4A16 path, including rank-tail and routed-mixture latency; and
4. produce a fresh TSH/X4 artifact and pass structural, numerical, KLD,
   full-model, and performance gates.

## 11. Validation gates

The project should advance only when each gate is satisfied.

| Gate | Required evidence |
|---|---|
| Capture integrity | all 92 MoE layers complete; exact TP joins; zero dropped rows; train/validation provenance recorded |
| Statistical adequacy | document-disjoint corpus; support and confidence quantified; unstable experts fall back to `R0` |
| Transform ABI | actual w1/w3 inverse transforms finish before SiTU and the separate w2 input transform begins after it |
| Permutation correctness | full-precision expert equivalence and BF16/TP12 closure with K3 SiTU branch order preserved |
| Format closure | mixed payload decodes exactly to encoder reconstruction; malformed descriptors are rejected |
| Rate closure | exact on-disk and resident byte ledger, including alignment and metadata |
| Quality | positive clustered-confidence result over captured-H `R0`; no material high-traffic regression hidden by an aggregate |
| Mode alphabet | retained ladder subset has acceptably small regret to the full ladder and record-map oracle |
| Kernel | P24 and mixed-expert workloads sustain acceptable end-to-end MoE performance without rank imbalance |
| Artifact | structural validator, streamed reference, TP closures, and full serving probes pass |

Thresholds for "acceptable" quality and latency should be fixed before the
full artifact run, after the representative layer-scale study exposes normal
variance. They should not be weakened after seeing the final model.

## 12. Current evidence

All results in this section are hypothesis-forming.

### 12.1 Data boundary

The completed retained and coupled experiments did not instantiate the
official Kimi-K3 model. They used:

- retained original MXFP4 expert weights available inside
  `/models/Kimi-K3-EXL3-3p09-serve`;
- routed samples captured from that same interim EXL3 teacher; and
- the local exllamav3 encoder implementation without instantiating a serving
  model.

The offline source-weight path subsequently added for the unbiased study
resolves the official checkpoint index but opens and dequantizes only one
expert matrix at a time on CPU. It never constructs or runs the official
model; routing, activations, Hessians, and validation targets continue to come
from the resident interim teacher.

The current study has two disjoint capture artifacts:

- training: 65,536 prompt tokens in 115 whole documents, excluding document
  fold 0 of 8;
- validation: 131,072 prompt tokens in 226 captured requests representing
  223 unique whole documents, containing only fold 0 of 8; and
- zero document-hash overlap.

The all-expert encoder further partitions the 115-document training capture
into 94 fit documents and 21 confirmation documents. The separate
223-document validation capture remains untouched by candidate generation and
allocation. Three duplicate requests in the corpus plan are deduplicated by
document hash before evidence is aggregated.

The earlier 16,384-token, 27-document validation pilot is no longer the
allocation authority. The 131,072-token sidecar satisfies the initial size
gate and is the current authenticated external damage source. It is still a
calibration-scale sample rather than a substitute for the planned full KLD
suite and end-to-end checkpoint evaluation.

Every one of the twelve capture shards finalized all 92 MoE layers with exact
input/middle joins and zero dropped rows. Each server run also produced one
epoch-zero infrastructure probe; raw observations and Hessians used here
strictly admit request epochs 1 through the corpus request count, so that
probe cannot contaminate ranking or evaluation.

This is materially stronger than the original eight-case pilot, but it is
still calibration evidence rather than a complete model-quality evaluation.
The first ladder and coupled studies below use retained experts; Section 12.4
separately removes that population and source-weight bias.

### 12.2 Dense Hessians matter

Captured-H EXL improved routed-output error over identity-H EXL in seven of
eight cases. Its geometric-mean error ratio was 0.148. Some apparent gains
were extreme because four-token Hessians are low-rank, which is simultaneously
evidence that calibration matters and a warning that the pilot can overfit.

For the first three-layer study, a fresh document-only dense-H bundle was
built from all eligible training rows. `H13` had 15,180 rows in each of layers
1, 24, and 40; `H2` had 16,256, 16,101, and 16,271 rows respectively.

The production-candidate study now uses a stricter fit-only bundle covering
all 92 MoE layers. It has 12,292 H13 rows per layer and between 12,957 and
13,597 H2 rows per layer, derived from 94 fit documents. All 92 FP32 matrix
pairs are finite, symmetric, and positive on the diagonal. Neither the 21
confirmation documents nor the 27 external-validation documents are
accumulated into these Hessians.

### 12.3 Uneven rate is promising

A separate activation-context palette experiment assigned unequal rates at
the same average of three bits per weight. It beat its uniform palette in
seven of eight held-out cases with a 0.643 geometric-mean error ratio. It was
still 6.14 times worse than captured-H EXL, arguing for unequal rate around
EXL rather than replacing EXL with that palette.

The full-H mixed-rate EXL experiment selected between the old U4 and M4
(`R0` and `R6`) using the
tiny training split. On validation it improved seven of eight cases and had a
0.776 geometric-mean ratio to ordinary captured-H EXL, corresponding to a
22.4% reduction. M4 was selected for the three examples with the clearest
activation skew; U4 was selected for the other five.

The simulated `w2` rate was:

- exactly 3.0 trellis bits per weight;
- 3.0096757 bits per weight including `suh`, `svh`, and the pilot's one-bit
  U4/M4 selector;
- not yet inclusive of real stream alignment, offset, padding, or
  kernel-descriptor costs.

The all-retained follow-up encoded every `R0..R12` counterfactual for 316
experts across layers 1, 24, and 40. Two additional retained layer-24 experts
had no eligible training middle rows and were explicitly skipped. All 4,108
encoded candidates passed reference edge/state closure and had identical
trellis, scale, and provisional format-field bytes within an expert.

On the untouched validation documents, the primary arithmetic sum of
gate-square-weighted routed-output SSE gave the following result:

- fixed `R2`: 5.64% improvement over captured-H `R0`, with a 95% paired
  document-bootstrap interval of 4.31% to 6.85%;
- full-ladder training argmin: 6.41% improvement, interval 4.16% to 8.71%;
- independent training split-confirmed selection: 67 of 316 experts changed
  from `R0`, for 2.15% improvement, interval 1.09% to 3.67%; and
- the old `R0`/`R6` training selector: 2.80% improvement, demonstrating that
  the coarse two-point table left substantial gain unused.

The layer frontier is not constant. Layer 1 favors approximately `R5`/`R6`,
layer 24 favors `R3`/`R4`, and layer 40 favors `R1`/`R2`; in layer 40, fixed
`R2` improves 3.27% while fixed `R6` regresses 6.51%. This directly supports
the rate-transfer ladder and rejects a globally fixed transfer strength.

An exhaustive training-only search over global mode alphabets found that
`{R0,R1,R2,R3,R5}` delivers 6.47% held-out improvement. The two-mode
`{R0,R2}` fallback already delivers 5.66%. These are candidate alphabets, not
frozen format decisions: arbitrary record-map regret remains outstanding.

### 12.4 Keep-tier-blind streamed-official evidence

The unbiased follow-up selected 48 experts in each of layers 1, 24, and 40 by
natural applied-gate-square traffic strata and deterministic within-stratum
sampling. Selection never inspected the current keep tier. Of the 144
experts, 22 happen to be retained and 122 are compressed in the interim
artifact. Each `w2` source matrix was streamed individually from the official
checkpoint on CPU; the official model was never instantiated. All 1,872
`R0..R12` candidates passed exact reference closure and equal-byte checks.

On the untouched validation documents:

- fixed `R2` improved routed `w2` SSE by 7.86%, with a 95% paired
  document-bootstrap interval of 3.38% to 13.98%;
- fixed `R3` improved it by 7.88%, interval 3.79% to 13.60%;
- the full training-selected ladder improved it by 7.98%, interval 3.98% to
  13.66%; and
- independent split-confirmed selection changed only 28 of 144 experts and
  still improved 2.17%, interval 0.76% to 3.90%.

The fixed-mode result is important because it cannot be attributed to
per-expert winner's curse. Transfer strength remains bounded: `R9` through
`R12` regress, and `R12` increases aggregate error by 47.3%. Layer 1 peaks
near `R3`, while layers 24 and 40 peak near `R2`.

The compact `{R0,R1,R2,R3}` alphabet achieved 8.18% held-out improvement,
slightly more than the full training-selected ladder because it avoided some
selection overfit. The coupled evidence below still finds value in `R5`, so
the provisional cross-matrix alphabet remains `{R0,R1,R2,R3,R5}` until the
record-map and candidate-`hhat` gates are complete.

### 12.5 Coupled full-expert mode evidence

A representative study then encoded `w1`, `w3`, and `w2` together for twelve
well-supported retained experts, four from each of layers 1, 24, and 40. The
median support was 102.5 training documents and 498 routed training rows per
expert, with 24 validation documents and 109.5 routed validation rows.

The training-selected shared full ladder reduced held-out routed functional
SSE by 4.98%, with a positive document-bootstrap interval. Fixed shared `R2`
reduced it by 2.73%. The selected result remained positive in each studied
layer, although the best transfer strength again differed substantially by
layer and expert.

Allowing separate `(r13, r2)` schedules improved the held-out aggregate by
only 0.078% over the unrestricted shared-ladder selector, with a 95% paired
document-bootstrap interval from -0.208% to 0.411%. Ten of twelve experts did
select a smaller `r13` than `r2`, so the matrices have measurably different
preferred schedules, but this sample does not show that the extra production
mode field earns its complexity. The schema should remain capable of separate
IDs; the provisional production decision remains one shared `r` unless a
larger unbiased study produces a clear gain.

Within this coupled sample, the compact shared alphabet
`{R0,R1,R2,R3,R5}` achieved 4.85% held-out improvement versus 4.98% for the
full ladder. Together with the unbiased-source and record-map results below,
this freezes `{R0,R1,R2,R3,R5}` as the initial encoder alphabet. The payload
schema continues to represent the full ladder so later evidence can revise
the encoder policy without inventing another pair primitive.

### 12.6 Record-map regret and the source of the gain

The arbitrary-map study used seven training-selected, official-source `w2`
experts spanning layers 1, 24, and 40 and modes `R1`, `R2`, `R3`, and `R5`.
It encoded 259 reference-closed candidates. Cheap independent-record
K2/K3/K4 trials proposed non-monotone donor/recipient maps, but every proposal
was authorized or rejected by a complete dense-H LDLQ counterfactual encode.

On external validation, the training-selected monotone maps improved 10.72%
over canonical-order `R0`, with a document-bootstrap interval of 4.23% to
18.72%. Allowing the shortlisted arbitrary maps changed that result to
10.71%. An arbitrary map was selected for only one expert and was slightly
worse than its monotone counterpart on validation.

Matched-order `R0` controls changed error by only 0.034%, with an interval
crossing zero, while the unequal-rate effect at the same order improved
10.68%, interval 4.17% to 18.64%. Thus the measured benefit comes from moving
trellis bits from low-importance to high-importance records, not merely from
reordering LDLQ traversal. Activation-ranked monotone records capture
essentially all observed allocation gain, so arbitrary maps are deferred from
the initial selector and require no extra serving metadata.

### 12.7 Expert-local Hessian and shrinkage evidence

The first candidate-H2 summary accidentally labeled a split as confirmation
even though its covariance had been built from every training document. That
result was not used to freeze policy. The corrected study deterministically
split the 115-document training corpus into 94 fit documents and 21 genuine
confirmation documents, materialized new dense H13/H2 matrices using only
the 94 fit documents, and retained the separate 27-document validation corpus
for external evaluation.

Across seven official-source coupled experts, blending 75% routed
expert-local source H2 with 25% fit-fold layer-global H2 reduced complete
expert error by 28.88% on confirmation, interval 23.17% to 34.32%, and 28.08%
on external validation, interval 24.62% to 32.51%. All seven experts improved.
Pure local H2 overfit the fit fold and delivered only about 20.4% externally,
so fixed shrinkage is materially better than taking the fit argmin.

Recomputing H2 from the quantized `w1`/`w3` activations changed the covariance
by a median 3.52% in relative Frobenius norm but did not improve the codec.
At alpha 0.75 it was 0.14% worse than the official-source local covariance on
validation, with an interval from 0.40% worse to 0.08% better. Candidate-`hhat`
is therefore deferred: it remains a target-aware research option, not an
initial encoder requirement.

A follow-up held that source-local H2 blend fixed and varied the shared H13
for `w1`/`w3`. A conservative 25% expert-local / 75% layer-global H13 blend
improved complete expert error by 2.57% on confirmation, interval 1.85% to
3.50%, and 2.88% on validation, interval 2.19% to 3.82%, with all seven
experts improving. Alpha 0.5 gave a slightly larger 3.19% validation aggregate
but regressed one expert; pure local H13 regressed all seven and increased
aggregate validation error by 9.47%. The initial policy therefore fixes H13
alpha at 0.25 and H2 alpha at 0.75 rather than selecting shrinkage per expert.

These gains are incremental at a constant byte payload. Covariance choice is
an offline encoder control and introduces no serving metadata or decoder
branch. The seven-expert population is still biased toward well-supported
cases, so production calibration must retain support thresholds and fall back
to layer-global covariance and `R0` when conditional estimates are weak.

### 12.8 TP12 reference implementation closure

The TP12 reference implementation now retains real EXL trellis states,
selects K on EXL `N` for `w1`/`w3` and EXL `K` for `w2`, postpacks complete
128-channel records and their aligned scale vector, and emits the fixed P24 /
P33 payload described above. Unit tests cover the legacy U4/M4/T8 schedules,
both logical axes, malformed descriptors, split-record rejection,
native word-order fixtures, and coupled SiTU permutation closure. A direct
CUDA comparison also matched exllamav3's native `pack_trellis` output for K2,
K3, and K4.

An interim-only layer-1/expert-15 smoke encoded all three matrices in both U4
and M4. Each matrix emitted exactly 4,128,768 trellis bytes. Every one of the
twelve physical pairs had a three-bpw mean, and coupled physical permutation
closure was about `1.7e-7` relative L2. These results establish plumbing and
layout closure only: that expert had one training row and one validation row,
so its U4/M4 comparison is not evidence for mode selection.

### 12.9 TP12 B12X implementation and performance

B12X now accepts compact P24 and P33 pairs on either logical rate axis,
reconstructs the native cyclic trellis states, supports per-expert dynamic
pair descriptors in the fused routed MoE path, and closes CUDA graph replay.
Persisted candidate validation agrees with the reference decoder to the
expected floating-point tolerance.

The isolated TP12 linear benchmark is encouraging: P24/P33 ratios stay within
about 0.3% for the tested one-, two-, four-, and eight-row cases, and compact
P33 is effectively tied with the existing K3 control. The routed benchmark is
the unresolved part. Across 1, 4, and 16 tokens, an alternating P24/P33 expert
mix is roughly 6-7% slower than uniform P33. This isolates the remaining work
to mode handling and scheduling in the routed kernel rather than the pair
decoder itself. Schema v3 separately removes deterministic cross-rank
imbalance by rotating each expert's P24 ownership; that should reduce TP tail
effects, but it does not by itself close the measured single-rank scheduling
gap. The first audit-driven kernel revision keeps the CTA-uniform P24/P33
choice compile-time throughout the inner load/dequant pipeline instead of
reintroducing runtime conditionals. Its acceptance run must use an isolated
fresh compile (`SPARKINFER_COMPILE_DISK_CACHE=0`), require known CUDA function
attributes and zero local memory per thread, retain the 50/50 mixed stress
case, and add a sparse-P24 case representative of the selected-mode frequency.
The natural-route fixture path is now implemented and CPU-validated. It binds
the candidate/capture hashes, exact rank-local mode tables, source row indices,
and route/gate tensors into the benchmark result. Until its idle-GPU
graph-replay run is complete, the earlier 6-7% alternating result remains the
last measured routed result and is not a production-frequency estimate.

A natural-route weighting over the first twelve atomically completed
candidate layers provides the first production-frequency estimate. Those
layers contain 140 mixed experts and 144 P24 expert/rank assignments: 0.1116%
of all static expert/rank pairs. On the untouched validation capture, P24
accounts for 0.1304% of rank-local routed expert executions. Because every
token routes to sixteen experts, 17.16% of layer/token rows nevertheless touch
at least one mixed expert somewhere in TP12. The per-rank token-hit fractions
range from 1.21% to 3.54%; across the TP12 slowest-rank view, 82.84% of rows
contain zero P24 routes, 16.35% contain one, 0.80% contain two, and 0.02%
contain three. Layer 6 is the current outlier, with 66 selected mixed experts
and a 72.7% any-mixed token rate. These are provisional partial-pool results,
not a final workload distribution, and all selected mixed modes still face
the untouched matched-R0 gate. They establish that both the all-P33 miss case
and the conditional sparse-hit case are required for latency modeling; the
50/50 mix remains a deliberately adversarial dispatch stress case.

### 12.10 All-expert encoder and allocator closure

The all-expert path streams official MXFP4 source matrices one at a time while
using only the interim-teacher capture for routes, activations, and targets.
It performs dense-H heterogeneous LDLQ for all three matrices, selects from
`{R0,R1,R2,R3,R5}` with document-disjoint confirmation, writes atomic layer
payload/metric/selection triplets, and supports strict same-manifest resume.

Two implementation hazards were found before committing a full layer. First,
selected capture slices retained their much larger backing tensors; compacting
the slice reduced a one-expert worker peak to about 2.1-2.5 GiB. Second, the
shared hidden scale was only algebraically, not bitwise, restored after the
per-expert global-scale search, and cached Hessian reuse shifted the random
output-sign stream between R modes. The encoder now places the global scale
directly in the expert-local vector and gives input/output signs independent
layer/matrix seeds. A 12-worker concurrent preflight passed layers 1-12, and a
layer-4 four-expert test selected both R0 and R1 while retaining exact shared
transform closure.

The strict allocator independently re-derives each supported fit argmin and
confirmation-CI acceptance decision, recomputes selected
fit-plus-confirmation excess SSE, rejects incomplete or malformed layer
pools, and requires the duplicated per-expert JSON selection evidence to
match the tensor ledger's selected/proposed modes, document support, CI,
evaluated-mode set, and canonical descriptors under one explicitly recorded
v2-or-v3 logical candidate schema. The persisted B12X
candidate probe requires both sidecars for the same reason. The allocator
treats the already natural-route/applied-gate-weighted SSE as the direct
MXFP4 keep benefit. It accounts TP12 rank-scale alignment exactly; a
compressed expert costs
12,404,736 marginal bytes after layer-shared scale deduplication, while an
MXFP4 expert costs 17,547,264 bytes before layer alignment. Global allocation
will run only after all 92 atomic layer outputs exist.

For the old-artifact target, 7,007 raw promotions leave 2,340,864 bytes for
per-layer alignment residuals. Residual padding is non-negative and depends
only on each layer's keep count modulo eight. If the global top-7,007 set fits,
it is therefore the proved global optimum: no 7,008th promotion can fit. If it
does not fit, the existing deterministic repair is not claimed to be globally
optimal. The production allocator now fails closed in that case; the repair
can be enabled only as an explicitly diagnostic result while an exact repair
solver is added.

The post-allocation materializer is implemented but has not yet been run on a
complete pool. It rejects any allocation whose tier partition, format code,
selected mode, source revision, candidate path, or exact aligned byte total
does not close. It also authenticates the named damage source, recomputes the
deterministic aligned-budget optimum, and re-derives every damage-ledger
scalar. A validation-backed allocation must reopen the complete score sidecar
and match its canonical score-set digest. It additionally requires a
reproducible matched-R0 summary with status `pass`, copies that summary into
the artifact, and binds its hash and complete score-set digest into the build
contract. Artifact and serve validation reopen the copied summary and
recompute the gate. It then streams one candidate tensor or one original
packed MXFP4 matrix at a time into explicit TP12 offsets,
validates every fixed section and zero-padding range, and atomically publishes
each layer. An immutable build request plus allocation and candidate-manifest
hashes make resume valid only for the identical fresh artifact. Before
publication, its reference reader rejoins all twelve rank slices and requires
every compressed trellis/scale tensor and every retained MXFP4 code/scale
tensor to be bit-exact with its candidate or official source. The standalone
artifact validator then rechecks the immutable provenance, complete 92-layer
inventory, exact logical filesystem bytes, per-layer layouts, padding, and
recorded full-payload closure; an expensive option independently rereads and
compares every source payload again. Each write-time comparison now also
produces a crash-safe per-layer closure receipt bound to the exact slab
identity and immutable build hash. A resumed build can no longer turn a
bit-exact result into header-only manifest metadata: it must load the matching
receipt or repeat the complete source comparison.

The fresh serve-directory packager and independent validator are implemented.
They publish symlinks only after all 92 slab targets, the allocation-derived
hybrid map, the MXFP8 non-expert index, tokenizer/auxiliary links, and exact
directory inventory close. Standalone serve validation re-runs live artifact
validation rather than trusting the cached packaging verdict. The independent
vLLM reader parses the frozen v3
ABI without importing kquant and emits B12X P24/P33 inputs plus production
MXFP4 kept-tier tensors for one physical TP12 rank.

### 12.11 Interim-teacher drift anchor

The existing full-layer streamed official and packaged-3p09 traces were
compared with `scripts/compare_k3_teacher_proxy.py`. Both checkpoint revision
directories resolve to the same config and safetensors-index payload. Across
92 MoE layers and the 12-token fixed correctness prompt:

- 16,252 of 17,664 top-16 assignments were retained (92.01%);
- the mean route-set Jaccard was 0.859, while 269 of 1,104 token/layer rows
  retained the entire route set;
- router-logit Pearson correlation was 0.9976;
- the sparse 896-expert applied-gate vectors had cosine 0.9670; and
- weights on routes common to both runs had Pearson correlation 0.9904.

This rules out a grossly unrelated calibration proxy on that prompt, but it
does not justify the interim teacher: 7.99% of assignments changed, and a
single 12-token prompt cannot estimate production routing or neuron-ranking
stability. The report records this limitation explicitly at
`out/correctness/exl3-pytorch/official-vs-interim-teacher-proxy-anchor.json`.

`scripts/stream_k3_pytorch.py --capture-routed-post-situ` now optionally
captures `[token, top_k, 3072]` post-SiTU values aligned back from the
official expert-grouped execution order. The flag is opt-in because its trace
storage scales as tokens times 16 routes times 3072 channels. This enables the
document-diverse comparison without ever making the official model resident.
`--input-ids-file` accepts the authenticated rectangular suite and
`--trace-layers` bounds trace storage while all preceding layers still execute.
The suite and document-bootstrap summarizer are implemented and unit-tested;
the official/interim GPU runs remain pending behind the active all-expert job.

### 12.12 Zero channels are a secondary mode

Among the 7,007 retained assignments, 640 (9.13%) had at least one exact-zero
`w2` input column. The mean was 2.194 zero columns per expert, only 0.0714%
of the middle axis, while a few outliers reached 1,071 columns. Zero/skip
records may be useful for outliers after permutation, but zeros are too rare
for a universal primary codec and the retained-tier frequency cannot be
projected onto all experts.

### 12.13 What has not been demonstrated

The following are still open:

- a production-scale capture beyond the completed 65,536-token representative
  study;
- stable all-expert mode-selection statistics;
- the physical permutation in a served checkpoint;
- routed P24/P33 performance parity or an accepted latency threshold;
- a packaged artifact whose filesystem size matches the closed prototype
  ledger; and
- end-to-end model quality.

## 13. Cross-discipline theoretical basis

The useful ideas come primarily from image, texture, video, and signal
coding—not from applying another LLM quantization paper unchanged.

### 13.1 BC7/BPTC and ASTC: small mode alphabets

The successors to DXT do not use one universal block representation. BC7
selects among a small number of modes with different partitions, endpoint
precision, and index allocation. ASTC expands this idea with explicit block
footprints, partitions, endpoint modes, and compact integer encodings.

The transferable lesson is the control structure: search several structured
encodings offline, signal a tiny mode ID, and keep the decoder's legal cases
finite. Direct endpoint-line and two-subset coding were tested on weight tiles
and did not justify their side information, but the mode architecture maps
well to a measured subset of the `R_r` ladder.

### 13.2 UASTC: optimize the intermediate for its consumer

UASTC demonstrates that an intermediate texture representation can be
designed for predictable decoding and transcoding rather than strict legacy
bitstream compatibility. Here the consumer is a fused TP/MMA kernel, so the
right outer record is determined by random access, TMA movement, rank
geometry, and decode regularity. Preserving the current EXL tensor format has
no intrinsic value.

### 13.3 VVC dependent quantization and AV2: trellis plus RDO

Modern video coding combines dependent or trellis quantization with
rate-distortion-selected transforms, partitions, quantization matrices,
contexts, and skip decisions. The trellis is not expected to solve rate
allocation by itself. This is the closest conceptual precedent for retaining
EXL's Viterbi core while adding context ranking and expert-level mode RDO.

### 13.4 Learned image/audio codecs: energy compaction and mode ladders

ELIC-style channel compaction, neural image codecs, and scalable residual-VQ
audio codecs all exploit unequal latent importance and discrete rate ladders.
Shared RVQ and gain/shape variants did not beat calibrated EXL in the interim
experiments, but they reinforce the narrower conclusion that different
channels should not automatically receive equal rate.

### 13.5 Why not runtime entropy coding

DeepCABAC, MPEG neural-network compression, and learned entropy models can
reduce archival size by exploiting symbol probabilities. A fused MoE GEMM
needs bounded-latency, independently addressable weight records. A serial or
context-dependent entropy stream can save disk bytes while losing effective
bandwidth, parallel decode, and random expert access. Entropy coding is
therefore deferred unless it can wrap independently decodable fixed-size
records without entering the hot kernel path.

Primary references:

- [Microsoft BC7 format](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format)
- [Arm ASTC format overview](https://chromium.googlesource.com/external/github.com/ARM-software/astc-encoder/+/HEAD/Docs/FormatOverview.md)
- [UASTC texture specification](https://github.com/BinomialLLC/basis_universal/wiki/UASTC-Texture-Specification/b624c07ad3c659e7b0f0badcb36e9a6b8820a99d)
- [Fast dependent quantization using trellis pruning, forward context adaptation, and vectorization](https://publica.fraunhofer.de/entities/publication/d5ef917c-1476-416d-95de-091f6935b6a0)
- [Transform and entropy coding in AV2](https://arxiv.org/html/2601.02712v2)
- [ELIC](https://arxiv.org/abs/2203.10886)
- [SoundStream](https://arxiv.org/abs/2107.03312)
- [JPEG AI documentation](https://jpeg.org/jpegai/documentation.html)
- [DeepCABAC](https://publica.fraunhofer.de/entities/publication/5b0bcbc9-ca77-4190-bb61-c44f457949f3)

## 14. Rejected or deferred alternatives

The following ideas should not distract the initial implementation:

- **Tile endpoint/line modes:** tested weight tiles were not sufficiently
  image-like, and endpoint side information erased the rate advantage.
- **Tile-local low rank:** factors and scales were too expensive at useful
  reconstruction error.
- **Shared RVQ or PVQ as the primary codec:** useful research controls, but
  worse than captured-H EXL on routed output.
- **Naive KLT/Cholesky whitening plus VQ:** did not reproduce LDLQ's dense
  error feedback.
- **Inter-expert prediction:** nearest or permuted expert residuals remained
  too large for a practical reference codec.
- **Sparse residual enhancement:** selected residuals improved, but position
  and value streams did not yet justify their cost.
- **Independent-context `w2` EXL:** explicitly rejected because it discards
  input-Hessian cross-record terms. Different output-row rates for `w1`/`w3`
  do not have that defect under an identity output metric.
- **Arbitrary per-tile K maps:** too much metadata and runtime irregularity
  for the first format.
- **Token-dependent mode selection:** incompatible with fixed checkpoint
  weights and predictable fused decoding.
- **Zero-format-change compatibility:** not a requirement.
- **Treating 16x16 as an MMA mandate:** the trellis tile and hardware MMA atom
  are separate design choices.
- **Minimal-state native-E2M1 K2/K4 transfer with unchanged source scales:** a
  complete layer-24 study learned useful K-specific transition tables and
  exploited both zero labels, but uniform K3 still measured 3.20% weight NMSE
  and even a global half-tile K2/K4 oracle was 1.682x worse than K3. No one of
  2,688 expert matrices won. Keep native MX output as a later hardware/codebook
  research branch, not as a replacement for the initial MCG TSH encoder. See
  [`native-mxfp4-trellis-investigator-brief.md`](native-mxfp4-trellis-investigator-brief.md).

## 15. Open questions and decisive experiments

1. Which channel statistic best proposes contexts beyond the now-validated
   post-SiTU energy order: a derivative-weighted score, measured ablation
   damage, or a learned layer-shared predictor?
2. At production support, does a small learned shrinkage rule improve on the
   frozen H13/H2 alphas without creating winner's curse?
3. How much quality is lost by one shared `r` compared with separate
   `(r13, r2)` while retaining the same common physical permutation?
4. Should scale search remain global-K3-like, or should K2/K3/K4 records use
   mode-specific scale optimization?
5. Can P24 be decoded with the same or better effective bandwidth and
   occupancy as P33 on SM120?
6. Is the fixed pair container sufficient to eliminate offsets after real
   alignment and TMA constraints are applied?
7. How should routed experts with insufficient calibration support borrow a
   layer prior without creating optimistic mode assignments?
8. Does a regularized teacher-target `w2` center generalize even though plain
   candidate-`hhat` covariance did not improve the initial study?
9. Does a zero/skip record mode pay for its signaling and padding on the rare
   outlier experts?
10. Is 16x16 still the best elementary trellis tile once a new decoder exists,
   or should that become a later, separately controlled experiment?
11. Does cyclic P24 ownership remove TP12 tail latency if P24 and P33 have
    unequal execution cost?
12. What canonical representation should a later TP-independent storage
    format use without constraining the TP12 runtime layout?

## 16. Initial implementation checklist

- [x] Complete and validate the representative interim-teacher capture.
- [x] Build fit-only dense Hessians for all 92 MoE layers while keeping the
      confirmation and external-validation documents untouched.
- [x] Scale the `R0..R12` `w2` study to all retained experts in layers 1, 24,
      and 40.
- [x] Complete the keep-tier-unbiased, streamed-official stratified
      `R0..R12` sample.
- [x] Implement common `w1`/`w3`-row and `w2`-column permutation closure.
- [x] Extend mixed-K encoding to the logical intermediate axis of all three
      matrices.
- [x] Compare shared `r` with separate `(r13, r2)` coupled expert modes.
- [x] Measure monotone-ladder versus shortlisted arbitrary-map regret and
      freeze the initial `{R0,R1,R2,R3,R5}` shared-mode alphabet.
- [x] Run leak-free expert-local H2 and H13 shrinkage studies; freeze source
      H2 alpha 0.75 and H13 alpha 0.25 for the initial encoder.
- [x] Specify the provisional TP12 schema and exact P24/P33 pair layout.
- [x] Implement reference pack/unpack, cyclic-state reconstruction, and
      malformed-input validation.
- [x] Close reference reconstruction and exact prototype byte accounting.
- [ ] Validate interim-teacher rankings against selected official layers
      streamed offline; never load the official model resident.
- [x] Freeze the authenticated 12-document teacher-proxy suite, selected-layer
      batched stream contract, and document-bootstrap decision gate.
- [x] Add a full-layer, single-prompt official/interim route-and-activation
      drift anchor plus route-aligned post-SiTU stream instrumentation.
- [x] Implement and benchmark the B12X SM120 decoder at TP12.
- [x] Implement a manifest-bound all-896-expert natural-route fixture and
      graph-stable route-window replay for the TP12 routed benchmark.
- [ ] Close or explicitly accept the remaining routed mixed-mode performance
      gap relative to P33.
- [ ] Generate an all-expert candidate pool from offline-streamed source
      weights. (Fresh schema-v3 logical candidate build in progress.)
- [x] Implement candidate-pool content finalization with all-file SHA-256 and
      require its digest throughout scoring, allocation, and materialization.
- [x] Implement document-disjoint selected-candidate scoring and an optional
      held-out-damage input to the global keep allocator.
- [x] Implement matched-R0 re-encoding for every selected mixed assignment,
      exact training/transform closure, whole-document bootstrap gating, and
      authenticated score/summary loading.
- [x] Make a reproducible matched-R0 `pass` mandatory for materialization;
      copy and revalidate the summary through artifact and serve packaging.
- [x] Implement one fail-closed TP12 performance verdict over the isolated
      N/K-axis decoder matrix and all requested natural-route layer/rank/token
      cases; independently recompute paired bootstraps from raw timings.
- [x] Make authenticated teacher-proxy and combined TP12 performance `pass`
      receipts mandatory for materialization, artifact validation, and serve
      packaging alongside the matched-R0 quality gate.
- [x] Bind held-out allocations to a canonical score-set digest and rederive
      the exact keep decision during materialization and artifact validation.
- [ ] Run held-out selected-candidate scoring after the complete pool closes
      and compare its keep ranking with the selection-corpus estimate.
- [ ] Run the matched-R0 external mode gate after selected-candidate scoring;
      reject or repeat the frozen mode policy unless aggregate and per-mode
      confidence gates pass.
- [x] Implement and unit-validate the exact equal-promotion-cost allocation
      when the globally best keep set closes under the physical layer ledger;
      fail closed rather than publish a greedy alignment repair.
- [ ] Implement the exact aligned repair solver only if the real held-out
      ranking makes the globally best keep set exceed the target through
      4-KiB layer padding.
- [x] Implement the bounded-memory, atomic allocation-to-TP12 layer-slab
      materializer with strict provenance and byte closure.
- [x] Implement reference slab extraction and independent structural/full-
      payload artifact validation.
- [x] Implement the independent vLLM rank-slab loader and fresh atomic serve
      packager/validator for the v3 TP12 ABI.
- [x] Implement a materialized-rank gate that cross-checks the production vLLM
      and kquant readers bit-for-bit, samples all retained-matrix orientations,
      and compares B12X FC1/SiTU/FC2 output with an independent PyTorch oracle
      under eager execution and CUDA graph replay.
- [x] Extend the streamed official PyTorch reference to reconstruct routed
      compressed and retained experts directly from completed TP12 slabs,
      while preserving the existing indexed interim-artifact path.
- [x] Make the kept/compressed TP reduction ownership fail-closed: both expert
      tiers must emit rank-local latent partials, followed by Kimi's latent
      reduction and exactly one post-projection FusedMoE reduction.
- [x] Pin the full-MXFP4 32x2048 KLD suite and implement a manifest-bound,
      paired 3p09-versus-candidate window-bootstrap subgate with a separate
      runtime-noise decision.
- [x] Test the no-Hadamard, unchanged-scale native-E2M1 small-state K2/K3/K4
      side branch across all layer-24 experts; reject raw K2/K4 transfer after
      its global tile oracle lost to K3 for every sampled expert matrix.
- [ ] Run that allocation at the target budget after the complete pool closes.
- [ ] Package only into a fresh artifact and serve directory.
- [ ] Run the materialized production-reader/B12X oracle on every logical TP12
      rank, then run structural, full-model, quality, and performance gates.
- [x] Implement and bit-exactly validate the compact `X4` MXFP4 scale plane on
      every expert in layers 1, 24, and 40 (8,064 matrices; 4.03819 bpw).
- [x] Complete the held-out all-expert layer-24 endpoint and compare K3/K4
      TSH against K3/`X4` at identical exact byte targets; K3/`X4` wins and
      K4 is off every supported expert's confirmation and validation hull.
- [x] Reject uniform K4 as a rate-distortion production mode after it fell
      off every supported expert's K3/`X4` convex hull.
- [x] Adopt exact `X4`, rather than uniform K4 TSH or raw 4.25-bpw MXFP4, as
      the successor high-quality allocation endpoint.
- [ ] Freeze the canonical X4 container, alignment, TP12 placement, and
      malformed-input contract.
- [ ] Integrate X4 into exact-byte allocation, materialization, artifact
      validation, and serve packaging as a zero-distortion candidate.
- [ ] Implement and benchmark TP12 X4 scale reconstruction feeding the normal
      W4A16 path, including natural-route and rank-tail measurements.
- [ ] Generate a fresh TSH/X4 checkpoint and pass structural, numerical,
      full-model, KLD, quality, and performance gates.
- [ ] Decide whether K3/K4/K5 (`P35/P44`) deserves a systems-only all-TSH
      branch despite the superior K3/`X4` rate-distortion frontier.
- [ ] Later phase: design a TP-independent storage format.

Until the remaining items pass, expert-static `R_r` transfer remains a
well-motivated and reference-closed codec design—not a replacement for the
validated `3p09` serving artifact.
