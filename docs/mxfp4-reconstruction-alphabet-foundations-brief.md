# Research brief: reconstruction alphabets for trellis-requantized MXFP4 weights

Status: investigator brief, 2026-08-02

Companion empirical report:
[`native-mxfp4-trellis-investigator-brief.md`](native-mxfp4-trellis-investigator-brief.md).
This document is deliberately about foundations and next questions rather than
about defending the first native-E2M1 experiment.

## Mission

Determine the theoretically and practically best reconstruction family for
compressing Kimi-K3's already-quantized MXFP4 expert weights with a
fixed-rate trellis codec.

The central question is not merely whether the stored source values are
four-bit. It is:

> At approximately three stored path bits per weight, should the trellis
> reconstruct general FP16 values, the original E2M1 values, a refitted
> E2M1-plus-scale representation, or some intermediate hardware-native
> alphabet?

Answer this as a rate-distortion-compute problem. The desired result is a
Pareto frontier over:

1. exact physical bytes;
2. source-, Hessian-, and routed-function distortion;
3. encoder complexity;
4. TP12 decoder and fused-MoE throughput; and
5. the extra retained-MXFP4 capacity needed to recover any quality loss.

Do not make LLM quantization literature the center of the search. QTIP and
EXL3 define the local implementation context, but the likely foundational
insight should come from classical quantization, finite-state source coding,
requantization, video/image/audio codecs, constrained coding, or
hardware-native numerical formats.

## Concrete system context

The official Kimi-K3 expert source is MXFP4. Each group of 32 weights contains
E2M1 values under one E8M0 power-of-two scale. The wire alphabet has sixteen
codes but only fifteen numerical values because `+0` and `-0` are distinct
codes:

```text
+0, +0.5, +1, +1.5, +2, +3, +4, +6,
-0, -0.5, -1, -1.5, -2, -3, -4, -6
```

The source costs `4 + 8/32 = 4.25` bits per weight. Kimi-K3 expert matrices
are:

```text
w1: [3072, 3584]
w3: [3072, 3584]
w2: [3584, 3072]
```

The production target for the initial codec is TP12 only. The shared
intermediate dimension is therefore 256 channels per TP rank. Activations are
BF16 in the surrounding model. A native E2M1 weight MMA path would require
the complete cost of any activation conversion, scale preparation, trellis
decode, and tensor-core feed to be included in its benchmark.

The current EXL3 MCG representation does not reconstruct source nibbles. A
coefficient is represented by K new path bits; an overlapping 16-bit state is
mapped procedurally through multiplier `0xCBAC1FED` and a small sequence of
integer/half operations to an FP16 value. There are 65,536 possible state
values, but only `2^K` outgoing choices from the current history at each step.

TrellisShift uses K2/K4 rate transfers around a K3 baseline. Over 24 logical
128-neuron records, mode `Rr` assigns K2 to `r` low-priority records, K4 to
`r` high-priority records, and K3 to the remaining `24 - 2r`. K2/K4 and
K3/K3 record pairs have equal path payload. The initial all-expert study uses
R0, R1, and R2, with separate decisions for `w1`/`w3` (`r13`) and `w2`
(`r2`) under one common neuron permutation.

The native-output question is a possible later codec branch. It must not
invalidate, replace, or delay the current MCG TrellisShift baseline.

## Formalize four different objects

Much of the intuitive confusion comes from calling all of the following a
"codebook." Keep them separate.

1. **Source alphabet:** values present in the official checkpoint, here
   scaled E2M1.
2. **Path alphabet:** the `2^K` branch symbols physically stored at each
   trellis step.
3. **State graph:** the finite-state transition rule that turns branch
   symbols plus history into transition labels.
4. **Reconstruction alphabet:** numerical values emitted by those labelled
   transitions.

The state graph can make a low-rate path behave like a high-dimensional
vector quantizer, but it does not increase the number of stored branch bits.
Likewise, a 65,536-entry procedural reconstruction mapping does not mean the
encoder has 16 free bits at every coefficient.

For a sequence of length `n`, a state graph with `S` possible initial states
and `2^K` outgoing edges per state has at most

```text
S * 2^(K*n)
```

candidate paths before closure constraints. Its rate is therefore at most

```text
K + log2(S)/n
```

bits per coefficient if an initial state is transmitted, and exactly K path
bits per coefficient in our tail-bitten format with no state header. A larger
state provides better path geometry or longer memory, not free per-symbol
information.

## Two inherited choices, not constants of nature

### Why is the rolling codeword 16 bits?

In QTIP/EXL notation, `L=16` means that the numerical reconstruction is a
function of a rolling 16-bit path window. It does **not** mean that each weight
stores 16 bits. Each weight still stores K new path bits.

Successive windows overlap by `L-K` bits for a scalar (`V=1`) trellis. There
are `2^L` possible labelled transitions and `2^(L-K)` overlap states in the
Viterbi dynamic program. At K3, the candidate widths imply:

| L | Labelled transitions | K3 overlap states | Recent K3 symbols touching a window | Encoder-state work relative to L16 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 65,536 | 8,192 | 6, including one partial symbol | 1x |
| 14 | 16,384 | 2,048 | 5, including one partial symbol | 1/4x |
| 12 | 4,096 | 512 | 4 | 1/16x |
| 10 | 1,024 | 128 | 4, including one partial symbol | 1/64x |
| 8 | 256 | 32 | 3, including one partial symbol | 1/256x |

The exact encoder cost is implementation-dependent, but its dominant state
work grows exponentially with L. Decode storage remains K bits per weight,
and a bitshift decoder can reconstruct any of these widths with similar word
operations. Reducing L is consequently much more likely to accelerate the
offline Viterbi encoder than to reduce checkpoint bytes.

QTIP chose L16 because quality improved with state width and its computed
codebook removed the otherwise impossible lookup memory. Its published K2
ablation was:

| L | Llama-2-7B WikiText-2 perplexity | C4 perplexity |
| ---: | ---: | ---: |
| 8 | 7.83 | 10.30 |
| 10 | 7.49 | 9.67 |
| 12 | 6.97 | 9.21 |
| 16 | 6.83 | 8.92 |

That supports “larger helps, with diminishing returns.” It does not establish
16 as optimal, and it provides no L14 result. L16 was also a convenient
machine boundary: a rolling index fits one halfword, a 32-bit multiply hashes
it cheaply, and the resulting word can be treated as two 16-bit fields by the
3INST/MCG decoder. It was an attractive quality/implementation point on the
hardware and source distribution QTIP studied.

For Kimi-K3, L8/L10/L12/L14/L16 is a required ablation. K3 has more outgoing
branches than QTIP's headline K2 study; the source is already MXFP4; dense-H
targets differ from an i.i.d. Gaussian; and offline encode time is presently
important. L12 or L14 could retain nearly all of L16's distortion performance
while cutting Viterbi state work by 16x or 4x respectively. The current L16
TrellisShift run remains the control and should not be changed mid-run.

### Why does MCG reconstruct FP16 rather than FP8?

FP16 is also an inherited implementation choice, separate from L16. QTIP's
3INST construction hashes the rolling index into a 32-bit word, transforms its
two 16-bit halves into two approximate half-precision random variables, and
adds them. This cheaply approximates the Gaussian reconstruction distribution
expected after Hadamard incoherence processing. EXL's MCG variant retains that
basic machine-shaped construction.

The FP16 values are generated; they are not stored at 16 bits per weight.
Their original justification was therefore:

- a dense, approximately Gaussian reconstruction distribution;
- more numerical freedom than the path can usually exploit;
- cheap half/half2 generation and consumption on the target GPU path; and
- no extra checkpoint payload.

It was not that the source or activations had to be FP16. Indeed, QTIP reports
that its alternative 1MAD generator has only about `2^10` representable
values even when `L>10`, without an empirical quality penalty in that study.
That is direct evidence that 65,536 distinct FP16 results are not inherently
necessary.

FP8 is therefore a serious candidate for this hardware, but “cast MCG's FP16
result to FP8” is the wrong experiment. That would add rounding error without
letting the graph or scale adapt. Instead compare equally optimized families:

1. a procedural E4M3 or E5M2 distribution with a refitted scale;
2. a learned 256-entry FP8-constrained transition table;
3. an 8-bit index into 256 freely learned centroids, as a quality control;
4. E2M1 plus one or more refitted block scales; and
5. free FP16 centroids on the same graph.

FP16 remains a superset-quality control: with the same graph and a genuinely
free optimizer, it can reproduce every FP8 or E2M1 value and cannot have worse
minimum distortion. FP8 or FP4 can win the complete design only through a
cheaper register/decode/MMA path, a better-constrained procedural mapping, or
finite-data regularization. On SM120, the comparison must include compatible
activation conversion, block-scale preparation, actual MMA throughput, and
the removal or retention of any materialized FP16 weight tile.

## The first foundational result: source and reconstruction alphabets differ

An already-quantized source does not imply that a lossy requantizer should
reconstruct only source symbols.

Under squared error, the best reconstruction value for a fixed cluster is
its conditional mean. For samples `x_i` assigned to transition `t`, with
nonnegative weights `a_i`, the optimum free reconstruction is

```text
y_t = sum_i a_i*x_i / sum_i a_i.
```

That centroid will generally lie between source symbols. A minimal example
is an equally likely source containing only 0 and 6. If both must share one
reconstruction value, emitting 3 gives mean squared error 9. Restricting the
output to either source value, 0 or 6, gives mean squared error 18.

This leads to an important dominance test:

> Hold the graph, path rate, scale model, traversal, and objective fixed. If
> a free FP16 transition table is optimized globally, it cannot have higher
> training distortion than an E2M1-constrained table, because FP16 exactly
> contains every scaled E2M1 reconstruction available to the constrained
> table.

Therefore a smaller output alphabet does not fundamentally give the same
branches more freedom. The number of paths is unchanged. A constrained
alphabet can still win in practice for four distinct reasons:

- it uses a different and better transition topology;
- the unconstrained procedural family is not actually a superset of the
  constrained family;
- the constraint regularizes a finite-data or imperfect optimization
  problem; or
- it enables a materially faster decoder/MMA path, so a modest distortion
  loss is worthwhile.

MCG is a fixed procedural mapping, not a freely learned 65,536-entry FP16
table. A native E2M1 graph and MCG are consequently non-nested codec families.
An apples-to-apples foundational comparison must first put native E2M1 and
free FP16 reconstruction values on the **same graph**. Only after that should
it compare either with MCG or MUL1.

## Dense-H error feedback makes the target even less source-native

The raw checkpoint value is not always the value the sequential encoder
should reproduce. The local dense-H objective is

```text
D(W_hat) = tr((W_hat - W) H (W_hat - W)^T).
```

Off-diagonal entries of `H` couple coefficient errors. BlockLDLQ changes later
quantization targets to compensate for errors already committed elsewhere.
Those counterfactual targets are generally continuous even when every entry
of `W` began on an E2M1 grid.

For fixed trellis paths, fixed transforms, and fixed scales, reconstruction is
linear in the transition values. The dense-H loss is therefore a quadratic
function of those values:

```text
D(y) = y^T A y - 2 b^T y + constant.
```

The best unrestricted table can be found by solving `A y = b` with suitable
regularization. This suggests an alternating algorithm:

1. hold reconstruction values fixed and find paths with Viterbi/LDLQ;
2. hold paths fixed and solve the weighted centroid or dense quadratic
   problem for reconstruction values;
3. repeat to convergence from several initializations;
4. project or refit the result into each hardware-constrained family.

This is the correct controlled way to ask how much quality is lost by native
E2M1 output. Comparing a hand-labelled E2M1 graph directly with procedural MCG
confounds reconstruction constraints, topology, and optimization.

## What the duplicate zero can and cannot buy

The two zero wire codes are a genuine finite-state resource. Distinct
zero-labelled transitions can incur no current numerical error while taking
the path into different future states. The correct object is a labelled
transition graph, not a sorted list of scalar levels.

The duplicate does not create a free bit at every coefficient. As a generous
local upper bound, if a fraction `p0` of coefficients are exact zeros and both
zero labels could be chosen independently at every such location, their path
multiplicity would contribute at most `p0` bits per source coefficient. The
layer-24 study observed about 13.1% numerical zeros, so this thought experiment
is about 0.13 bit per weight, not the full one-bit gap between K3 and K4.
Actual graph closure and state reachability reduce the independent freedom.

The investigator should replace this loose argument with a graph-theoretic
one. Relevant quantities include:

- the topological entropy of the transition graph;
- the number of paths mapping to the same numerical reconstruction sequence;
- state reachability conditioned on zero and near-zero outputs;
- labelled-graph quotients after merging `+0` and `-0`; and
- how zero-transition placement changes future distortion.

There is no a priori reason that the two zeros should occupy opposite binary
indices. In our learned layer-24 table, opposite positions won for K3 but not
for K2. Topology is rate-dependent and should be optimized, not inferred from
visual symmetry.

## Why equal-average K2/K4 can lose badly to K3

For a homogeneous source, time-sharing a low-rate and a high-rate code is a
feasible code at the intermediate average rate. The optimal distortion at the
intermediate rate can therefore be no worse than that time-sharing line.
Unequal rate becomes useful only when contexts have different
rate-distortion slopes or different functional importance.

For an exchange around K3, define:

```text
donor cost    = distortion(K2) - distortion(K3)
recipient gain = distortion(K3) - distortion(K4).
```

A K2/K4 exchange helps only when the recipient gain exceeds the donor cost
under the complete objective. With dense-H feedback this must ultimately be
measured by a full candidate encode, not by adding independent per-record
curves.

In the raw native-E2M1 layer-24 study:

```text
D2 = 0.118864 normalized MSE
D3 = 0.032006 normalized MSE
D4 = 0       (exact source closure)
```

Thus the K2 donor cost is approximately `0.086858`, whereas the entire K3 to
K4 recipient gain is only `0.032006`: the donor penalty is about 2.71 times
larger. Assigning half the tiles to K2 and half to exact K4 predicts

```text
0.5 * D2 / D3 = 1.857,
```

which is the observed fixed-pair result. Even assigning K2 to the globally
easiest half of sampled tiles remained 1.682 times worse than uniform K3.

This does not prove that TrellisShift's MCG K2/K4 exchange is unsound. It says
that the unchanged-scale, no-transform, native-grid distortion curve is too
steep at K2. MCG has different K2, K3, and K4 reconstruction errors, and the
real selector allocates rate according to captured functional sensitivity,
not raw source error alone.

## "Native" is not one design

The search should not be framed as only E2M1 versus FP16. Separate at least
these reconstruction families:

| Family | Transition output | Scale | Main purpose |
| --- | --- | --- | --- |
| Source-native | E2M1 | original E8M0/K32 | exact K4 closure and simplest falsification test |
| Compute-native refit | E2M1 | refitted E8M0/K32 | retain MX MMA while moving the grid |
| Compute-native fine scale | E2M1 | E4M3, FP16, or denser grouping | approximate free centroids with native mantissas |
| Multi-grid native | E2M1 plus a small grid/scale selector | record-, state-, or subblock-dependent | add centroid freedom without FP16 weights |
| FP8 reconstruction | suitable FP8 values | explicit shared scale if needed | intermediate precision and possible tensor-core feed |
| Free small table | learned FP16 transition centroids | fixed/refitted | isolate the cost of the alphabet constraint |
| Procedural | MCG or MUL1 | current EXL scaling | production quality/decoder control |

The most promising unexplored region may be **compute-native but not
source-native**. The original E2M1 nibble and scale are merely one
factorization of each source value. Refitting the scale, using two nearby
grids, or learning an affine reconstruction can place effective codepoints
near the optimal centroids while still emitting an E2M1 operand to hardware.

Every added scale or mode bit must be charged exactly. A scheme that improves
path distortion by silently spending extra scale bytes is not an equal-rate
result.

## Transforms and SiTU

Do not treat the existing Hadamard path as a correctness mystery. Its transform
placement and cancellation already work through the implemented Kimi-K3 SiTU
expert path and have numerical serving closure. The native experiment removed
Hadamard and sign transforms only to isolate the source-grid hypothesis.

The research question is instead empirical and representational:

- how much does incoherence improve each reconstruction family;
- whether a global, 128-value, K32, or record-local transform is best;
- whether a compute-native output can preserve a cheap inverse-transform path;
  and
- how transforms change the optimal scale and transition-value distribution.

Scaling and rotation are not substitutes. Scaling changes local dynamic range;
Hadamard mixing changes coordinate distribution, correlations, and the targets
seen by the trellis. Once a transform or LDLQ feedback makes the targets
continuous, the fact that the original stored weight was E2M1 has less direct
importance.

## Most relevant non-LLM research traditions

### 1. Classical scalar and vector quantization

Start with Lloyd-Max conditions, generalized Lloyd/LBG algorithms, fixed-rate
versus entropy-constrained quantization, and high-dimensional shaping. The
questions are how optimal reconstruction centroids depend on an already
discrete source, a weighted quadratic metric, and a constrained output set.

Search terms:

```text
discrete source continuous reproduction alphabet squared error
constrained reproduction alphabet rate distortion
generalized Lloyd algorithm weighted distortion
entropy constrained vector quantization finite state
quantizer codebook projection hardware constrained
```

### 2. Trellis-coded and finite-state quantization

Study TCQ as a way to separate path rate from effective vector dimension.
Focus on set partitioning, state count versus shaping gain, predictive TCQ,
finite-state ECVQ, tail-biting, and learned transition labels. The concrete
question is whether MCG's 16-bit memory is near a useful saturation point or
merely one implementation choice.

Search terms:

```text
trellis coded quantization state complexity shaping gain
finite state vector quantizer state conditioned codebook
predictive trellis coded quantization Gauss Markov
tail biting source coding trellis quantizer
trellis label optimization reproduction levels
```

### 3. VVC dependent quantization

This is the closest mature codec analogy. VVC switches between quantizers with
a small parity-driven state machine; both component quantizers include zero.
Investigate why that construction improves low/medium-rate transform coding,
how its reconstruction levels and state transitions were chosen, how RDOQ
performs the trellis search, and what was learned from later fast/parallel DQ
encoders.

Do not copy VVC's graph mechanically. Its coefficient distribution, entropy
coder, scan order, and distortion model differ from ours. Extract the design
principles: overlapping quantizer subsets, deliberately duplicated zero,
state-dependent cosets, and joint path/rate optimization.

### 4. Requantization and successive degradation

The checkpoint is itself the output of an earlier quantizer, making this a
tandem or source-requantization problem. Study when requantizing a codeword
sequence is equivalent to quantizing the latent original, the penalty caused
by not planning the first code for later degradation, and whether nested or
successively refinable codebooks can improve K2/K3/K4 behavior.

One concrete alternative is a progressive graph in which K2 is a base path,
K3 adds one refinement bit, and K4 adds another. This guarantees a nested
syntax but does not guarantee the best distortion at each rate. Quantify the
price of nesting and whether its simpler mixed-rate decoder is valuable.

### 5. Image and texture block formats

BC7 and ASTC are relevant as fixed-payload mode systems, but their deeper
lesson here is endpoint-plus-interpolation reconstruction. They do not merely
select among source colors; they fit a small hardware-friendly parametric
reconstruction family per block. Explore analogues in which a trellis emits
E2M1 indices while a block carries learned endpoints, scale, offset, or one of
a tiny set of grids.

### 6. Predictive and noise-shaping audio codecs

DPCM, ADPCM, predictive TCQ, and noise-shaped quantization deliberately move
instantaneous reconstruction error so it matters less after a downstream
filter. This is conceptually close to LDLQ using covariance to feed error into
later coordinates. Look for stable finite-state designs, centroid optimization
under prediction, and how reconstruction constraints interact with feedback.

### 7. Constrained coding and symbolic dynamics

Run-length-limited storage codes and labelled finite-state encoders provide
tools for measuring graph capacity, equivalent labels, state splitting, and
path multiplicity. This is the right language for determining exactly what
two numerical zeros contribute and whether alternative duplicated or
near-duplicated levels would improve state steering.

### 8. Hardware-native block floating point

Study OCP MX formats, Blackwell block-scaled MMA, and compressed-domain matrix
multiplication. Distinguish storage format from reconstruction format and MMA
operand format. Determine whether codepoints that are mathematically FP16 can
be factored cheaply into E2M1 plus a scale, generated in registers, or consumed
through another low-precision tensor-core path without materializing an FP16
weight tile.

## Required theoretical questions

The investigator should answer these explicitly.

1. Under a fixed graph and quadratic objective, prove the dominance relation
   between unrestricted FP16 transition values and E2M1-constrained values.
2. For fixed paths, derive the optimal transition values under diagonal
   weights and under full dense-H distortion.
3. Bound the benefit available from duplicated reconstruction labels, both by
   simple path counting and by labelled-graph entropy.
4. Characterize how state width changes path count, effective dimension,
   minimum distance, shaping gain, and decoder/encoder cost. Explain why a
   16-bit window might be justified—or not—rather than assuming it.
5. Estimate the empirical entropy rate and higher-order dependence of source
   nibbles in each relevant traversal order. Marginal symbol entropy alone is
   insufficient.
6. Compute a lower bound or small-block optimum for the empirical 15-value
   source at K2, K3, and K4 under both raw weighted SSE and simplified Hessian
   metrics. Blahut-Arimoto, exhaustive block coding, or dynamic programming
   may provide useful bounds, but label clearly which operational constraints
   they omit.
7. Determine whether the steep native K2-to-K3 distortion slope is inherent
   to the source distribution or caused by our minimal 4/2-state graph.
8. Quantify the constraint loss for E2M1 with original scales, refitted scales,
   multiple grids, FP8 output, and free FP16 centroids.
9. Determine whether nested K2/K3/K4 paths help or merely constrain three
   independently optimized rate points.
10. Derive the hardware break-even curve: how much extra distortion can a
    native-output codec tolerate at a given speedup before the additional
    MXFP4 keep tier erases its byte or throughput advantage?

## Decisive experimental program

### Phase A: information and scalar bounds

Use official source weights, with train/validation experts separated.

1. Measure E2M1 wire entropy, numerical entropy after merging zeros, entropy
   rate, run statistics, sign predictability, and mutual information over
   several lags and traversal orders.
2. Fit optimal 4-, 8-, and 16-level scalar quantizers with unrestricted
   centroids to the normalized source distribution.
3. Repeat with codepoints restricted to E2M1 and with refitted scale.
4. Compute empirical rate-distortion lower bounds for the finite source.
5. Stratify by matrix family, layer, scale exponent, and Hessian sensitivity.

This phase tells us how much loss exists before any trellis or kernel detail.

### Phase B: hold the graph fixed

Use one or more fixed bitshift graphs and compare:

1. learned native E2M1 transition labels with original scales;
2. the same labels with refitted scales;
3. freely learned FP16 transition centroids;
4. centroids projected to E2M1 plus one scale;
5. two-grid/state-conditioned E2M1;
6. FP8-constrained transition values; and
7. duplicated or near-duplicated centroid variants designed for state
   steering.

Alternate Viterbi path assignment and reconstruction-value optimization. Use
multiple seeds and disjoint validation experts. This phase isolates the
reconstruction alphabet from graph topology.

### Phase C: vary graph memory and topology

Sweep practical state widths and graph constructions rather than comparing
only the minimal native graph with MCG's 16-bit window. Include at least:

- minimal 4-K carried-state bits;
- 4-, 8-, 10-, 12-, and 16-bit histories where computationally feasible;
- standard set-partitioned TCQ graphs;
- VVC-like dependent-quantization graphs;
- learned transition labellings;
- tail-biting and fixed-start controls; and
- multiple traversal orders.

Report quality versus Viterbi work, decoder instructions, table bytes,
register pressure, and parallel-decode depth. State count is not free even
when the runtime codebook is procedural.

### Phase D: controlled transform and dense-H comparison

For the surviving families, make MCG, MUL1, native, and free-centroid encodes
identical in every respect except reconstruction family:

- same source experts;
- same captured training Hessian and untouched validation documents;
- same transform and scale policy;
- same tile/traversal order;
- same K and exact metadata bytes;
- same BlockLDLQ feedback; and
- same candidate-specific `w2` input distribution.

Evaluate uniform K3 first. Then evaluate full-record R1/R2 with separate
`r13` and `r2` selection. A raw native K2/K4 tile oracle already failed, so a
new native mixed-rate branch should proceed only if a modified construction
materially changes the K2/K3/K4 curve.

### Phase E: TP12 kernel frontier

Implement only enough reference kernel work to measure the real opportunity.
Compare:

- MCG decode to its current weight operand;
- E2M1 decode plus original/refitted block scale;
- FP8 reconstruction if a relevant MMA path exists;
- BF16 activation handling and any E4M3 quantization cost;
- isolated K2, K3, and K4 records;
- P24 versus P33 pair balance;
- routed mixtures at realistic expert/token counts; and
- complete fused-MoE latency, not decoder throughput alone.

Measure graph-captured TP12 latency, tensor-core utilization, instructions,
registers, occupancy, preparation traffic, and rank-tail behavior. Do not
generalize from TP4 or TP16.

### Phase F: functional and global byte tradeoff

For Pareto candidates, replay routed expert functions and then perform a small
live-model evaluation. Report:

- dense-H reconstruction loss;
- routed expert-output and full-mixture loss;
- route and gate drift;
- teacher-logit divergence;
- exact average bytes including scales/tables/modes; and
- how many additional experts must remain MXFP4 to match the MCG checkpoint's
  quality.

A native decoder that is faster per compressed expert can still lose globally
if its quality deficit requires enough extra retained experts to increase
memory traffic or reduce the compressed fraction.

## Existing evidence to reproduce, not explain away

The completed layer-24 source-domain study covers all 896 experts, all three
expert matrices, and approximately 44.0 million sampled coefficients. It kept
the official E8M0 scales, used no Hadamard/sign transform, and optimized
minimal native transition tables on disjoint experts.

Learned graph labels reduced NMSE relative to native nibble ordering by 25.1%
at K2 and 34.8% at K3. This validates the hypothesis that transition topology
and the duplicate zero matter.

Nevertheless:

```text
uniform native K2 NMSE             11.886%
uniform native K3 NMSE              3.201%
native K4 NMSE                      0.000%
fixed half-K2 / half-K4 vs K3       1.857x
best-within-pair oracle vs K3       1.731x
global easiest-half oracle vs K3    1.682x
expert matrices where oracle won    0 / 2688
```

The negative result is highly informative: learning a better 16-transition
native graph is insufficient. A successful hardware-native successor must
change at least one material assumption—scale freedom, reconstruction grid,
state topology, transform, dense-H compensation, or the quality/speed
objective.

The source output and exact learned tables are in
`out/e2m1-trellis-layer24-native-v1.json`. The reference implementation is
`kquant/e2m1_trellis.py`; the study driver is
`scripts/experiment_e2m1_trellis.py`.

## Controls that prevent misleading conclusions

- Compare every native family with a free-centroid version of the same graph.
- Compare MCG and MUL1 under identical transforms, scales, Hessian, and bytes.
- Separate table-fitting experts from mode-selection and validation experts.
- Do not use K4 exactness as evidence that K2/K4 time-sharing is efficient.
- Charge scale, table, padding, mode, and initial-state metadata exactly.
- Distinguish raw weight SSE, dense-H loss, routed functional loss, and final
  model quality.
- Do not let a tile oracle masquerade as a deployable fixed-record format.
- Report optimizer regret with multiple initializations before attributing a
  difference to the alphabet.
- Do not assume marginal nibble entropy is available to a fixed-rate random
  access decoder.
- Do not claim that the duplicated zero yields one free bit per weight.
- Do not assume FP16 numerical output necessarily requires an FP16 stored
  table or a fully materialized FP16 tile; investigate factorized generation.
- Treat the existing Hadamard/SiTU serving path as working evidence and study
  its coding effect, rather than reopening an abstract transform objection.

### 2026-08-02 empirical update

The first controlled L12/L14/L16 and E4M3 reconstruction study is complete.
Across two disjoint official-weight replications, L14 costs roughly 3-5% local
tile distortion for a 4x smaller dynamic program, while fresh E4M3-aware paths
recover essentially the entire penalty implied by casting an FP16-selected
path. See [the side-study report](trellis-window-fp8-side-study.md) for the
contract, paired bootstrap intervals, raw-result paths, and proposed
rate-specific-window follow-up.

## Expected investigator deliverables

1. A concise literature map organized by the eight non-LLM traditions above,
   emphasizing primary sources and mechanisms rather than paper counts.
2. A formal note proving the fixed-graph dominance result, deriving optimal
   transition centroids under diagonal and dense quadratic metrics, and
   bounding duplicated-label freedom.
3. A reproducible notebook or small program that computes empirical entropy,
   scalar/vector lower bounds, and state-width curves from the recorded
   layer-24 sample.
4. A ranked shortlist of no more than five new reconstruction families with
   exact bit accounting and plausible SM120 decode paths.
5. A proposed controlled experiment matrix in which alphabet, graph, scale,
   transform, and distortion metric are varied one at a time.
6. A go/no-go recommendation for a compute-native TrellisShift branch,
   including the quality/throughput break-even point.

## Starting primary sources

- S. P. Lloyd, ["Least Squares Quantization in
  PCM"](https://doi.org/10.1109/TIT.1982.1056489), 1982. The foundational
  centroid and nearest-neighbor conditions.
- R. M. Gray and D. L. Neuhoff,
  ["Quantization"](https://www.math.ucdavis.edu/~saito/data/quantization/44it06-gray.pdf),
  1998. Broad historical and theoretical survey of scalar, vector, predictive,
  and finite-state quantization.
- M. W. Marcellin and T. R. Fischer,
  ["Trellis Coded Quantization of Memoryless and Gauss-Markov
  Sources"](https://doi.org/10.1109/26.46532), 1990. Classical TCQ foundation.
- P. A. Chou, T. Lookabaugh, and R. M. Gray,
  ["Entropy-Constrained Vector
  Quantization"](https://doi.org/10.1109/29.17498), 1989. Joint rate and
  distortion optimization and generalized codebook training.
- H. Schwarz et al.,
  ["Quantization and Entropy Coding in the Versatile Video Coding (VVC)
  Standard"](https://refubium.fu-berlin.de/bitstream/handle/fub188/33130/Quantization_and_Entropy_Coding_in_the_VVC_Standard.pdf),
  2021. Dependent quantization, duplicated zero, state transitions, and RDOQ.
- A. S. Cohen, S. C. Draper, E. Martinian, and G. W. Wornell,
  ["Source Requantization: Successive Degradation and Bit
  Stealing"](https://sia.mit.edu/wp-content/uploads/2015/04/2002-cohen-draper-martinian-wornell-dcc.pdf),
  2002. Compression of an already-quantized source without advance planning.
- W. H. R. Equitz and T. M. Cover,
  ["Successive Refinement of
  Information"](https://isl.stanford.edu/~cover/papers/paper94.pdf), 1991.
  Conditions under which nested descriptions reach the ordinary
  rate-distortion bound.
- Open Compute Project,
  ["Microscaling Formats (MX) Specification
  v1.0"](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
  Source-format and block-scale semantics.
- NVIDIA,
  ["Parallel Thread Execution ISA"](https://docs.nvidia.com/cuda/parallel-thread-execution/).
  Authoritative block-scaled MMA operand and instruction contracts.
- A. Tseng et al.,
  ["QTIP: Quantization with Trellises and Incoherence
  Processing"](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf),
  2024. The immediate bitshift-trellis, procedural-codebook, Hadamard, and
  BlockLDLQ foundation; use it as an anchor rather than the center of the
  literature review.

## Decision standard

The investigator should not try to prove that four-bit output is inherently
better because the source is four-bit. That proposition is false in general
under lossy squared-error reconstruction.

The useful question is narrower and more promising:

> Can a hardware-factorized reconstruction family recover most of the
> free-centroid/MCG rate-distortion performance while enabling a sufficiently
> faster TP12 decode-and-MMA path to improve the complete checkpoint Pareto
> frontier?

The raw source-native experiment says "not with unchanged scales, no
transform, and the minimal 16-transition graph." It does not answer the
compute-native question.
