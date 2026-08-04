# Trellis window and FP8 reconstruction side study

Date: 2026-08-02

## Decision summary

Five conclusions survived disjoint studies over official Kimi-K3 expert
weights.

1. **L14 is a credible encoder-speed trade.** Relative to L16, it reduces the
   Viterbi transition work and state memory by 4x while increasing local tile
   distortion by about 2.8% at K2, 4.6% at K3, and 5.4% at K4. L12 supplies a
   16x reduction, but its corresponding penalties are about 6.9%, 10.8%, and
   13.9%.

2. **A learned E4M3 reconstruction table is now a credible K3 candidate.** A
   post-hoc cast is misleading: after holding the L16/K3 graph fixed and
   alternating Viterbi assignment with label fitting, one table shared across
   layers 1, 24, and 40 reduces blind tile SSE by 0.691% versus procedural MCG
   and wins 53 of 54 matrix groups. Its paired 95% interval is
   `[0.99194, 0.99417]` times MCG error.

3. **Fine local scale freedom is valuable, but UE8M0 is too coarse after EXL
   normalization.** A continuous scale per 32 experimental sequence values
   reduces learned-E4M3 SSE by another 4.329%. Integer power-of-two deltas
   select zero everywhere. Three fractional log-scale bits per 32 recover
   94.8% of the continuous improvement at 3.09375 nominal bpw. This is an
   endpoint-plus-index result, not yet a compute-native format: its grouping
   and scale alphabet do not match SM120 `mxf8f6f4` scale factors.

4. **Lookup-free MUL1 is the production-leading E4M3 base.** Projecting EXL's
   existing MUL1 reconstruction to E4M3 and rerunning Viterbi reaches
   `0.997069` times procedural-MCG SSE without any transition table. A search
   over 4,096 odd 32-bit multipliers plus local mutations decisively retained
   EXL's published `0x83DCD12D` multiplier; refitting the affine map was
   slightly worse. An 8.25-KiB learned correction improves the MUL1 result to
   `0.993710`, but it is only a quality oracle until its lookup and
   shared-memory cost beats the procedural decoder in the complete W4A8
   kernel.

5. **MUL1 has a measured sparse-decode advantage, not merely a simpler-looking
   formula.** On this host's exact TP12 Kimi-K3 matrix shapes, EXL's current
   integer-activation GEMV is 1.37--1.42x faster for the 7168-to-3072 projection
   and 1.50--1.67x faster for the 3072-to-7168 projection at `m=1..2`. The path
   fuses MUL1's byte reduction with the integer dot product through `dp4a`.
   At `m>=4`, that special path is not selected and the measured
   implementations tie. These numbers establish the value of the procedural
   primitive; they do not predict the speed of the future E4M3 Tensor Core
   path.

The FP8 result directly supports the proposed mechanism: the smaller value
alphabet leaves the trellis free to choose different state trajectories. It
does not merely round the trajectory selected for FP16.

These are tile-quantizer results. They justify full-Hessian counterfactual
encodes and a kernel feasibility study; they do not yet authorize changing the
active TrellisShift build or its production bitstream.

## Experimental contract

The two confirmation runs use:

- official MXFP4 source weights, never the interim EXL checkpoint;
- layers 1, 24, and 40;
- two disjoint sets of six experts per layer;
- `w1`, `w3`, and `w2` for every selected expert;
- 108 matrix groups in total;
- eight matrix-local gain-fitting tiles and 32 untouched evaluation tiles per
  matrix;
- 884,736 held-out scalar coefficients in aggregate;
- deterministic signs, EXL input-RMS normalization, and 128x128 block
  Hadamards;
- output-channel scaling disabled, as in the current routed-expert encoder;
- a matrix-local global gain selected at K3, then held fixed across K2/K3/K4;
  and
- EXL's two-pass, half-tile-rotated tail-biting approximation.

The reference dynamic program accumulates costs in FP32. On matched L16
samples, its error is within about 0.2% of EXL's production half-cost encoder,
and normally slightly lower, as expected from the higher-precision search.

The evaluated FP8 family is deliberately conservative. It projects every MCG
reconstruction value to E4M3 and then reruns the complete Viterbi search. It
does not yet learn a new hash, transition labelling, or FP8 centroid
distribution. At L16 the 65,536 transition labels produce 10,746 distinct MCG
FP16 values but only 145 distinct E4M3 values after projection.

Raw results:

- `out/trellis-window-fp8-l12-l14-l16-matrix-local-v2.json`
- `out/trellis-window-fp8-l12-l14-l16-matrix-local-v3.json`
- `out/trellis-learned-codebooks-l1-l24-l40-shared-v1.json`
- `out/trellis-learned-codebooks-l1-l24-l40-shared-v1-summary.json`
- `out/trellis-e4m3-scales-l1-l24-l40-shared-v2.json`
- `out/trellis-e4m3-state-scales-c512-v1.json`
- `out/trellis-e4m3-binary-subcode-v1.json`

The implementation is in `kquant/trellis_window.py` and
`scripts/experiment_trellis_window_fp8.py`.

## Window-length result

The table reports aggregate held-out SSE divided by the matched L16 SSE. The
confidence intervals are paired 95% matrix-cluster bootstrap intervals over
the 108 matrix groups.

| Window | Transition labels per step | Relative DP work | K2 | K3 | K4 |
|---|---:|---:|---:|---:|---:|
| L16 | 65,536 | 1x | 1.0000 | 1.0000 | 1.0000 |
| L14 | 16,384 | 1/4 | 1.0277 `[1.0257, 1.0296]` | 1.0459 `[1.0440, 1.0479]` | 1.0539 `[1.0486, 1.0592]` |
| L12 | 4,096 | 1/16 | 1.0694 `[1.0670, 1.0716]` | 1.1079 `[1.1055, 1.1103]` | 1.1393 `[1.1331, 1.1455]` |

The shorter-window penalty is monotone and stable across both replications.
It is smallest at K2 and largest at K4. That makes a **rate-specific window**
more interesting than replacing L16 everywhere:

- keep K3 and K4 at L16;
- first test K2 at L14; and
- test K2 at L12 only on the deliberately low-importance donor records.

Because K already identifies the decoder path, a globally fixed window per K
would require no per-record metadata. It would concentrate the encoder speed
and memory reduction on K2, where the current L16 graph has the most overlap
states, without weakening the high-value K4 recipient records or the R0 K3
control. This is especially well aligned with P24.

## FP8 result

The table reports aggregate held-out SSE divided by the matched FP16-MCG SSE
at L16.

| Reconstruction experiment | K2 | K3 | K4 |
|---|---:|---:|---:|
| Cast the chosen FP16 path afterward | 1.00961 | 1.03944 | 1.14794 |
| Re-run Viterbi in E4M3, same matrix scale | 0.99981 | 0.99960 | 0.99655 |
| Re-run Viterbi in E4M3, refit matrix scale | 1.00038 | 0.99939 | 0.99631 |

Paired matrix-cluster bootstrap intervals for the fixed-scale reroute are:

- K2: `[0.99941, 1.00021]`;
- K3: `[0.99898, 1.00021]`; and
- K4: `[0.99534, 0.99775]`.

Thus K2 and K3 are statistically indistinguishable from the FP16 procedural
alphabet in this study. K4's small improvement is stable. The scale-refit
variant is noisier because each matrix has only eight gain-fit tiles; the
fixed-scale comparison is both cleaner scientifically and closer to the
requested “keep the scales the same” control.

The E4M3-aware paths differ from the FP16 paths on 50.8% of transition labels
at K2, 71.5% at K3, and 86.4% at K4. This is not evidence that the functions
change by those percentages: one changed edge changes several rolling state
labels. It is evidence that Viterbi makes extensive use of alternate routes.

### Why rerouting can erase the cast penalty

The path code still supplies K new bits per step, regardless of how many
distinct numerical reconstruction values exist. When many transition labels
alias to the same E4M3 value, they are numerically interchangeable at the
current coefficient but lead to different future overlap states. Viterbi can
choose among those aliases according to the future sequence.

A post-hoc cast throws that freedom away: it first chooses a route for precise
FP16 values and only then moves those values. The fresh E4M3 search chooses a
different route under the actual constrained alphabet. The effect becomes
larger with K because the FP16 path had more freedom to exploit fine value
differences; at K4, post-hoc casting is especially misleading.

This also explains why “FP16 is a superset, so it must be better” does not
settle this particular comparison. A freely optimized FP16 transition table
would contain the E4M3 solution and could not be worse. MCG is not a freely
optimized table: it is a fixed procedural labelling. Projecting it to E4M3
changes the centroid distribution and transition aliases, and Viterbi can
occasionally use that changed fixed codebook more effectively.

### Quantifying the hidden steering reserve

The duplicate-label explanation can be stated more precisely. Let a trellis
state be `s`, let the K-bit outgoing branch be `e`, and let `Y(s,e)` be the
numeric reconstruction emitted by that edge. If outgoing branches are treated
as uniformly likely, the branch carries K raw path bits, while the current
numeric output exposes only

\[
H(Y\mid S=s).
\]

The difference

\[
A(s)=H(E\mid S=s,Y)
\]

is current-output-preserving state freedom. If `m(s,Y)` branches from state
`s` share the selected label, then

\[
A(s)=2^{-K}\sum_e \log_2 m(s,Y(s,e)).
\]

This is a local structural ceiling, not a claim that the source supplies that
many independent free bits. Actual paths are nonuniform, tail-biting couples
the sequence, and one branch choice changes several later rolling labels.

`scripts/analyze_trellis_aliases.py` computes this quantity directly on the
exact MCG graph. Its raw output is
`out/trellis-e4m3-alias-structure.json`. At L16:

| Rate | Alphabet | States with an outgoing alias | Edges with an alias | Uniform local steering reserve |
|---|---|---:|---:|---:|
| K2 | FP16 MCG | 0.02% | 0.01% | 0.0001 bit/edge |
| K2 | projected E4M3 | 3.49% | 1.75% | 0.0175 bit/edge |
| K3 | FP16 MCG | 0.29% | 0.07% | 0.0007 bit/edge |
| K3 | projected E4M3 | 20.54% | 5.46% | 0.0552 bit/edge |
| K4 | FP16 MCG | 1.39% | 0.18% | 0.0018 bit/edge |
| K4 | projected E4M3 | 74.95% | 14.98% | 0.1548 bit/edge |

The rate trend agrees with the empirical result: E4M3's clearest advantage is
at K4, where three quarters of states have at least one same-output successor
choice. It also corrects an overinterpretation of the path-change rates. A
86.4% K4 transition-index change does not imply nearly one free steering bit
per coefficient; the local uniform ceiling is about 0.155 bit, and a changed
rolling history propagates into many later transition indices.

The analysis also refines states by their labelled right context: two states
remain equivalent at depth `d` only if every identical sequence of the next
`d` branch symbols emits the same numeric sequence. All L16 E4M3 outgoing
aliases lead to successor states that are distinguishable after just one more
step. At K4 all 4,096 successor states already have unique one-step labelled
contexts; at K3, 8,191 of 8,192 do; and at K2, 16,015 of 16,384 do, reaching
all 16,384 after two steps. Thus the measured aliases are not redundant copies
leading to equivalent futures: they are genuine current-output-preserving
choices among different future reconstruction contexts.

The information-theoretic framing suggests a new control. E4M3 is not the only
way to obtain useful non-injective labels. A learned small set of unrestricted
FP16 centroids, with many transitions tied to each centroid, can preserve the
same hidden path multiplicity without accepting the E4M3 grid. The next study
should therefore compare:

1. procedural MCG;
2. a free FP16 label for every transition;
3. a learned finite FP16 centroid alphabet with tied transition labels; and
4. learned E4M3 transition labels.

The free table is the same-graph quality ceiling. The tied-centroid control
separates the benefit of **alphabet collapse and state steering** from the
cost of the particular E4M3 numerical grid. VVC dependent quantization and
JPEG 2000 TCQ support the general construction: the useful object is a
state-conditioned family of overlapping reconstruction sets, jointly searched
by a trellis encoder, rather than a sorted scalar codebook alone. See the
[Fraunhofer VVC dependent-quantization description](https://www.hhi.fraunhofer.de/en/departments/vca/research-groups/video-coding-technologies/research-topics/transform-coefficient-quantization-and-coding.html)
and [ITU-T T.801 TCQ specification](https://www.itu.int/rec/T-REC-T.801).

### Same-graph dominance and the next quality gate

For a fixed graph, traversal, scale family, and distortion objective, the
E4M3 table is a subset of a free FP16 table. Therefore the converged training
optimum obeys

\[
D^*_{\mathrm{free\ FP16}} \leq D^*_{\mathrm{E4M3}}.
\]

The current E4M3 win over procedural MCG is consequently evidence about MCG's
fixed labelling, not evidence that fewer reconstruction values have a higher
fundamental quality ceiling. The practical question is whether E4M3 recovers
enough of the free-table improvement to justify a faster compute path.

`scripts/experiment_trellis_learned_codebooks.py` implements the first
controlled test at L16/K3. It uses disjoint matrix-local tiles for gain fitting,
transition-centroid fitting, ridge/iteration selection, and blind evaluation.
Both learned families use identical Viterbi reassignment and weighted Lloyd
updates, and the matrix-local gain is held fixed. This prevents a 65,536-entry
table from winning by memorizing the evaluation sample.

The first experiment is deliberately diagonal-metric. A surviving family must
then be retrained inside the production Hadamard and dense-H BlockLDLQ loop;
feedback changes the target after every committed error, so raw-weight
centroids are not the final production centroids.

### Learned-table result

The controlled L16/K3 study covers six experts in each of layers 1, 24, and
40, all three expert matrices, 54 independently evaluated matrix groups, and
1,769,472 blind coefficients. The source is the official MXFP4 checkpoint.
Gain fitting, transition fitting, ridge/iteration selection, and evaluation
use disjoint tiles. One transition table is fitted jointly across all three
layers, so the result does not rely on per-layer or per-expert tables.

| Fixed L16/K3 graph reconstruction | SSE / procedural MCG | Wins | Paired 95% interval |
|---|---:|---:|---:|
| Project MCG labels to E4M3, reroute | 0.998911 | 39/54 | `[0.998398, 0.999397]` |
| Learned E4M3 labels | 0.993089 | 53/54 | `[0.991941, 0.994170]` |
| Learned tied FP16, 145 labels | 0.992316 | 52/54 | `[0.991002, 0.993564]` |
| Learned free FP16 transition table | 0.992208 | 54/54 | `[0.990908, 0.993411]` |

Learned E4M3 recovers 86.85% of the projected-E4M3-to-free-table gap. The
free table remains better, as feasible-set inclusion requires, but only by
0.089% relative SSE. The 145-level tied-FP16 control is statistically
indistinguishable from the free table and beats E4M3 by 0.078%. This assigns
most of the gain to a **finite, deliberately tied label alphabet and the state
steering it creates**, with a smaller residual cost from the E4M3 grid.

The pooled shared table is slightly better than fitting separate tables to
each layer (`0.993089` versus `0.993581` aggregate). The layer-local tables
have only about 41% exact label agreement but correlations above 0.997. More
training support therefore regularizes the same underlying transition
geometry; layer-specific metadata is not currently justified.

### Learned binary subcode result

A flat state-palette factorization first established the storage frontier. A
512-row palette occupies 13 KiB packed and only ties procedural MCG. Increasing
it to 2,048 rows occupies 27 KiB packed and recovers 96.9% of the full learned
table's measured gain. Giving each C512 state a three- or four-bit
power-of-two exponent produces 16- or 17-KiB packed formats (20 KiB with a
simple uint16 descriptor), but the held-out selector retains the unscaled
C512 initialization. Magnitude normalization therefore does not explain the
missing row diversity.

The successful factorization instead uses procedural MCG as a predictor. For
transition index `t`, let

\[
a_t=\operatorname{E4M3}(\operatorname{MCG}(t)).
\]

The format stores one learned bit `r_t` for every one of the 65,536 L16
transitions and a 256-entry table `z[a]` containing one alternate E4M3 byte:

\[
y_t=
\begin{cases}
a_t,&r_t=0,\\
z[a_t],&r_t=1.
\end{cases}
\]

The exact decoder is a procedural MCG evaluation, E4M3 projection, one packed
bit extraction, and one conditional 256-byte lookup. The bit plane is 8,192
bytes and the alternate table is 256 bytes, for 8.25 KiB total. If the
alternate table is kept in constant storage, the hot random-access SMEM
footprint is exactly 8 KiB.

The table below uses the same disjoint 54-group evaluation as the learned-table
study:

| Fixed L16/K3 reconstruction | Bytes | SSE / procedural MCG | Wins |
|---|---:|---:|---:|
| Projected MCG E4M3 | procedural | 0.998911 | 39/54 |
| Full learned E4M3 table | 65,536 | 0.993089 | 53/54 |
| MCG anchor + learned alternate | 8,448 | **0.992999** | 53/54 |
| Two learned outputs per MCG anchor | 8,704 | 0.993006 | 54/54 |

The 8.25-KiB candidate is selected after one fitting-path refit. Only 5,203 of
65,536 transitions (7.94%) select the alternate, and the bit plane's marginal
entropy is 0.400 bit per transition. It exactly matches the full learned E4M3
table on 68.1% of transitions. The slight blind improvement over the full
table is plausibly a regularization effect from tying rare transitions to the
procedural predictor; it is too small to claim as a statistically meaningful
quality win. The important result is practical equivalence at one eighth the
reconstruction storage.

The predictor comparison changes the production disposition. Using exactly
the same fitting and blind-evaluation split gives:

| Fixed L16/K3 reconstruction | Bytes | SSE / procedural MCG | Wins |
|---|---:|---:|---:|
| Procedural MUL1 FP16 | procedural | 0.997373 | 37/54 |
| Projected MUL1 E4M3, rerouted | procedural | **0.997069** | 35/54 |
| MUL1 anchor + learned alternate | 8,448 | 0.993710 | 45/54 |
| MCG anchor + learned alternate | 8,448 | 0.992909 | 54/54 |

The corrected MUL1 form is only `1.000806` times the corrected MCG error, so
the predictor choice is nearly irrelevant after the 8-KiB correction. Without
that correction, however, MUL1 is both better than projected MCG and far
cheaper to carry into a direct decoder. The correction buys about 0.337%
relative SSE over projected MUL1. That is now the price/quality trade to test,
not 8 KiB versus a 64-KiB flat table.

### MUL1 procedural-family search

The lookup-free family was searched in the form

\[
p_t = \operatorname{uint32}(t m),\qquad
z_t = \sum_{i=0}^{3}\operatorname{byte}_i(p_t),\qquad
y_t = \operatorname{E4M3}(a z_t+b).
\]

The search covered 4,096 seeded odd multipliers, all single-bit mutations and
several arithmetic neighborhoods of the published multiplier, then reran
Viterbi and refit matrix gains for the finalists. `0x83DCD12D` won both the
transition-weighted proxy and the held-out path-selection score by a large
margin. The best least-squares affine refit (`a=0.00676568`, `b=-3.45230`)
reached `0.997374` times MCG SSE, slightly worse than projecting the exact
published MUL1 half reconstruction (`0.997069`). Therefore the codec should
retain the published multiplier and half constants. There are no learned
procedural parameters to serialize.

Entropy-coding the sparse bit plane is deferred. The 8-KiB fixed bitmap gives
constant-time indexing with no offsets or serial decode, matching the serving
requirements. The reference pack/unpack and fixed-primary decoder close
exactly in `kquant/e4m3_subcode.py`.

### Scale-resolution result

`scripts/experiment_trellis_e4m3_scales.py` adds coefficient-dependent scales
inside the Viterbi cost, rather than rescaling a path afterward. With the
shared learned E4M3 table held fixed:

| Scale family per 32 experimental values | Nominal bpw | SSE / factorized E4M3 | Continuous-gain recovery |
|---|---:|---:|---:|
| Existing factorized matrix scale | 3.00000 | 1.000000 | 0% |
| UE8M0-like integer power-of-two delta | 3.25000 | 1.000000 | 0% |
| 3-bit fractional log2 delta, step 1/32 | 3.09375 | 0.958945 | 94.8% |
| 4-bit fractional log2 delta, step 1/64 | 3.12500 | 0.957726 | 97.7% |
| Continuous FP16 K32 scale oracle | 3.50000 | 0.956710 | 100% |

The final continuous log2 scale ratio has standard deviation 0.043; its 1st
and 99th percentiles are -0.105 and +0.102. The useful corrections cluster
tightly around one, exactly between UE8M0's adjacent factor-of-two values.
This is the classical endpoint-plus-index pattern: once a trellis path chooses
the indices, a few locally fitted endpoint bits can be worth much more than a
wide but coarse exponent range.

This result is a representation upper bound, not yet an SM120 block-scale
result. Consecutive groups in the current trellis sequence cover two N columns
and 16 K rows. PTX `mxf8f6f4` instead provides one one-byte scale per B-matrix
row and 32-value K block, and its scale type is UE8M0. The
[PTX scale-factor-B contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-b-layout-1x)
therefore requires a new K32-by-N-aligned control. Fractional scales would
also need a separate multiply/factorization path; applying them directly to
the E4M3 labels would forfeit the native E4M3 operand.

### SM120 E4M3 compute contract

The quality experiment and a native FP8 kernel are separate claims. The
current [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
defines ordinary FP8 MMA with E4M3/E5M2 multiplicands and defines BF16 MMA
separately; it does not expose a BF16-by-E4M3 Tensor Core form. Its SM120
block-scaled `mxf8f6f4` forms likewise consume low-precision operands plus
scale vectors. A direct E4M3 weight path must therefore quantize the activation
operand into a compatible FP8-family representation.

This creates two mandatory numerical controls:

1. decode the candidate E4M3 weights, widen them, and multiply by BF16
   activations, isolating weight-codec distortion; and
2. run the actual E4M3 activation/weight MMA path, measuring the additional
   activation-quantization and scaling error.

The throughput incentive is substantial but not automatic. NVIDIA lists FP8
peak throughput at twice FP16/BF16 peak for the
[RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/),
but trellis decode, activation conversion, scale preparation, register
pressure, and routed expert sizes can dominate the realized kernel.

There are two plausible scale contracts:

- **Factorized EXL scales:** keep the existing input-axis and output-axis
  factors. Fold the global codebook gain into one factor, apply the input-axis
  factor while preparing the activation operand, emit raw E4M3 table values,
  and apply the output-axis factor after accumulation. This adds no K32 scale
  stream but needs a fused activation-preparation path.
- **MXFP8-style block scaling:** add one hardware-aligned UE8M0 scale per 32 K
  values for each B-matrix row and use block-scaled MMA. This is nominally
  3.25 bpw for a K3 path before other metadata. The sequence-local scale
  experiment shows that UE8M0 may be too coarse, so this family must earn its
  bytes in the exact hardware grouping rather than inheriting the continuous
  scale oracle's gain.

The second option is consistent with the per-32 local-scaling rationale in
[NVIDIA Transformer Engine's MXFP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html),
but it is not presumed superior for static trellis weights.

## Implications for TrellisShift

The immediate algorithmic implication is not “switch everything to L14 and
FP8.” It is a narrower pair of follow-ups.

### 1. Rate-specific window counterfactual

Add L14-K2 and L12-K2 candidates while keeping K3/K4 at L16. Run the actual
dense-H BlockLDLQ traversal on representative P24 records. The decisive metric
is routed functional distortion for a complete R1/R2 expert, not scalar tile
MSE. K2 is assigned to low-importance donors, so its 2.8% or 6.9% local error
increase may have much less than proportional functional cost.

If L14-K2 survives, its DP has four times fewer transition evaluations and
four times fewer overlap states than L16-K2. L12-K2 offers 16x reductions.
Those are algorithmic work ratios, not measured production speedups; a
specialized CUDA encoder must establish wall-clock scaling.

### 2. Native E4M3 reconstruction candidate

Keep the current matrix scales and rerun full-H K2/K3/K4 encodes with the
lookup-free projected-MUL1 E4M3 codebook. Score R0--R5 candidates on routed
validation rows with separate `r13` and `r2` decisions. The encoder must use
the E4M3 reconstruction inside Viterbi and BlockLDLQ; casting an already chosen
FP16 path remains an invalid control.

Implement the decoder as the exact published MUL1 state hash followed by a
native E4M3 conversion. The 8.25-KiB MUL1-anchor binary subcode remains an
optional quality arm, not the default format. It advances only if the complete
kernel measurement shows that its extra lookup/SMEM path recovers enough
checkpoint quality to offset its latency and occupancy cost.

The direct generator must be benchmarked against the actual SM120 MMA operand
and scaling path. Merely storing an FP8-looking value is not sufficient if the
kernel immediately widens it or requires an expensive mixed-type conversion.

## Limits

- This study measures regularized 16x16 tile error, not dense-H LDLQ error,
  routed expert-function error, logits, or end-to-end quality.
- The source matrices are official MXFP4, but the sampled transform signs are
  deterministic experimental signs rather than the exact signs of a packaged
  candidate.
- Eight training tiles per matrix make the independently refit gain noisy.
  The fixed-scale FP8 control is more trustworthy.
- The 8.25-KiB binary subcode still evaluates MCG, projects it to an E4M3 byte,
  extracts a bit, and conditionally looks up an alternate. Its instruction,
  constant-cache, and SMEM costs remain unmeasured.
- EXL's measured MUL1 speedups use an integer-activation sparse GEMV and cannot
  be transferred numerically to E4M3/E4M3 MMA. The new W4A8 arm needs its own
  graph-replay timing at the actual routed shapes.
- The fractional K32 scale result uses sequence-local groups that do not match
  the SM120 B-operand scale layout and cannot be claimed as compute-native.
- The reference timing is not a kernel benchmark. L-dependent speed claims
  must come from specialized CUDA encoder kernels.

## Recommended disposition

- Let the in-flight L16/FP16 MCG R0/R1/R2 validation finish as an independent
  reference; do not confuse it with the new production candidate.
- Promote **L14 for K2 only** to the next dense-H side experiment.
- Keep **L12 for K2 only** as an aggressive donor-record candidate.
- Promote **lookup-free L16 MUL1-to-E4M3** to the production-transform,
  dense-H R0--R5 experiment. Retain the 8.25-KiB corrected MUL1 form and the
  64-KiB learned table only as reconstruction-quality oracles until kernel
  timing justifies either lookup.
- Run a hardware-aligned K32-by-N scale control. Do not spend bytes on UE8M0
  merely because the continuous scale oracle is strong; the tested UE8M0-like
  delta earned no gain.
- Treat fractional 3/4-bit local scales as a codec-quality lead whose compute
  factorization is unresolved, not as a bitstream commitment.
- Do not spend kernel effort on a post-hoc FP8 cast; it measures the wrong
  route.
- Freeze no new bitstream field yet. Both a rate-specific L and a globally
  fixed reconstruction family can be introduced without per-record metadata
  if their functional and kernel gates pass.
