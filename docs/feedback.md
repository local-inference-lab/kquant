## Overall assessment

My judgment is positive, with two important qualifications.

First, the strongest contribution is not the general proposition that sensitive channels deserve more bits. That territory is already crowded. The distinctive part is the way you propose to make unequal rate compatible with a regular serving format:

* heterogeneous K inside an EXL/QTIP-style trellis quantizer;
* dense-Hessian error feedback rather than independently quantized partitions;
* a common, function-preserving intermediate-neuron permutation across `w1`, `w3`, and `w2`;
* fixed-size K2/K4 and K3/K3 containers;
* expert-static offline mode selection based on routed functional distortion; and
* a layout built around TP12 rather than around the existing checkpoint representation.

That is a coherent algorithm–format–kernel co-design, and I did not find a paper that combines that exact package. Your decision summary captures it accurately. 

Second, the empirical case is currently much weaker than the design. Eight retained `w2` cases, generally with two training rows and one validation row, are enough to justify the next experiment but not enough to estimate the expected model-wide gain. The reported 0.776 geometric-mean ratio is encouraging, especially because the mode selector chose M4 only for the three visibly skewed cases, but it remains a hypothesis-forming result from a biased subset. 

I would continue with the design, but I would revise several theoretical claims and expand the candidate structure before committing to the bitstream.

## Where it sits in the literature

QTIP is the direct foundation. It contributes the high-dimensional trellis quantizer, random-Hadamard incoherence processing, and the use of TCQ as the rounding operation inside BlockLDLQ. Its standard local objective is the familiar activation-weighted error

[
\operatorname{tr}!\left((\widehat W-W)H(\widehat W-W)^T\right),
]

and its implementation traverses the input blocks backward while feeding previous reconstruction error through the LDL factor. ([NSF Public Access Repository][1])

The closest prior art to your mixed-rate idea is **Q-Palette**. Its “half-TCQ” explicitly partitions a matrix row-wise, applies different TCQ bit widths to the partitions, and decodes them in a fused CUDA kernel. It also performs resource-constrained selection among quantizer schemes. Therefore, I would not claim that this is the first mixed-rate TCQ within a matrix. The meaningful distinction is that Q-Palette uses mixed TCQ primarily to realize fractional average rates, whereas your proposal transfers rate from sensitivity-ranked K2 records to K4 records at exactly the same nominal three-bit budget, while retaining dense-H feedback and imposing a shared gated-neuron layout across three matrices. ([arXiv][2])

There is also substantial overlap with recent hardware-aligned mixed-precision work. **ScaleBITS** clusters sensitive rows and columns through bidirectional reordering, then assigns one precision to each hardware-aligned block. **PolyQ** compiles irregular per-channel bit assignments into contiguous bit-homogeneous blocks and propagates permutations through adjacent operators; it explicitly observes that coordinatewise nonlinearities commute with channel permutations. **CMPQ** allocates 2-, 3-, and 4-bit precision by channel and, importantly, reports that the damage from moving channels from three to two bits can exceed the benefit of moving an equal number from three to four bits. These papers weaken broad novelty claims about sensitivity ordering, channel permutation, or equal-average 2/4-bit exchange, but they support the design direction. ([arXiv][3])

On the MoE side, **MxMoE** already combines expert/block sensitivity, routing frequency, hardware constraints, and mixed-precision Group-GEMM generation. **GEMQ** performs global expert-level bit allocation and shows that expert quantization can materially change later routing behavior. **MoEQuant** addresses the calibration imbalance caused by sparse routing with expert-balanced sampling and affinity weighting. Your expert-static mode decision and global compressed-versus-MXFP4 allocation therefore fit an active line of work, but your within-expert hidden-neuron allocation is a finer level than those methods generally target. ([arXiv][4])

The rate-distortion analogy is also established in LLM work. **Radio** formulates LLM quantization directly as a rate-distortion allocation problem, and **RateQuant** fits quantizer-specific distortion curves and performs reverse-waterfilling allocation. Your use of actual candidate encodes rather than an idealized analytic distortion curve is appropriate because TCQ plus LDLQ is stateful and nonseparable. 

The BC7/ASTC comparison is useful primarily as a **format-design analogy**. BC7 uses fixed 128-bit blocks with one of eight legal modes. ASTC likewise keeps every compressed block fixed at 128 bits while allowing the encoder to allocate those bits differently per block, preserving random access and bounded decode cost. That strongly supports your “small mode alphabet plus fixed container” philosophy, but it is not the theoretical basis for why mixed rate should improve LLM weights. The theory comes from TCQ, mixed precision, Hessian-aware rounding, and rate-distortion allocation; the image codecs contribute the decoder-control architecture. ([Microsoft Learn][5])

A defensible novelty statement would be something like:

> An expert-static, fixed-payload mixed-rate trellis codec for gated MoE experts, using a shared function-preserving hidden-neuron permutation, heterogeneous-rate dense-H BlockLDLQ, routed-function rate-distortion selection, and TP-balanced P24/P33 containers.

That is narrower than “first mixed-rate TCQ,” but more technically meaningful. This is a literature assessment rather than a patent-clearance search.

## What is strongest in the current design

The **P24/P33 equality** is probably the most valuable systems insight. A 128-channel K2 record and K4 record together have the same ideal payload as two K3 records, so a pair container can retain constant-stride addressing while supporting unequal rate. That avoids the usual variable-bitstream consequences: offset arrays, serial parsing, irregular gathers, and poorly bounded random access. 

The **common neuron permutation** is also sound. For a true permutation matrix (P),

[
W_1'=PW_1,\qquad
W_3'=PW_3,\qquad
W_2'=W_2P^T
]

leaves the expert function unchanged because SiTU is coordinatewise and the same (P) is applied to both branches. The permutation is useful not because it saves a few bytes of metadata, but because it makes the rate regions contiguous in all three matrices and removes any serving-time shuffle between SiTU and `w2`. 

Preserving **dense-H feedback across the heterogeneous-rate traversal** is the right choice for `w2`. Independent K2/K3/K4 EXL calls would discard the off-diagonal covariance structure precisely where the quantizer is supposed to use it. The fact that your simpler context codebook experiment remained much worse than captured-H EXL is consistent with that view. 

The design also correctly distinguishes the **within-compressed-tier mode decision** from the **compressed-versus-MXFP4 keep decision**. These are different optimization problems: one redistributes a fixed three-bit payload, while the other adds model bytes. Keeping those candidate pools separate will let you change the global memory budget without requantizing every expert. 

## The main theoretical correction: dense-H distortion is not separable

The exchange argument in Section 6.1 is a good intuition, but the equations are not literally exact under a dense Hessian.

Partition the `w2` error and Hessian by rate record:

[
E=[E_1,\ldots,E_C],\qquad H=[H_{cd}].
]

Then

[
\operatorname{tr}(EHE^T)
========================

\sum_c \operatorname{tr}(E_cH_{cc}E_c^T)
+
\sum_{c\ne d}\operatorname{tr}(E_cH_{cd}E_d^T).
]

The cross-context terms do not permit an intrinsic scalar distortion curve (D_c(K)) for each context unless (H) is block diagonal or (D_c) is defined conditionally on the complete reconstruction. Furthermore, in BlockLDLQ, changing K for one record changes its error feedback and can therefore change the reconstruction errors of later records. Your document itself emphasizes the importance of those off-diagonal terms, so the separable exchange equation conflicts slightly with the stronger dense-H claim. 

I would recast that section as follows:

1. The separable inequality is a **motivating approximation** under weak cross-context coupling.
2. The exact quantity is a full counterfactual encode:

[
\Delta_e(l,h\mid\mathbf{k})
===========================

## D_e!\left(Q_e(\mathbf{k}_{l\leftarrow2,h\leftarrow4})\right)

D_e!\left(Q_e(\mathbf{k})\right).
]

3. The K2/K4 exchange is beneficial when this complete, full-H delta is negative.
4. Cheap marginal estimates can propose exchanges, but only full candidate encoding authorizes the mode.

There is also an important matrix-specific nuance. For `w2`, the intermediate records lie on the **input axis**, which is the axis coupled by (H_2), so independent context calls really do lose Hessian cross terms. For `w1` and `w3`, the intermediate records lie on the **output-row axis**, while the ordinary local Hessian (H_{13}) is over their shared 3584-dimensional input. Under the standard identity output metric, those output rows are separable, and different row groups can use different trellis rates while sharing the complete input Hessian. They become coupled only when you introduce an output-side functional metric, such as a SiTU/downstream Fisher metric. The implementation and paper should distinguish these two cases rather than issuing one blanket rule about independent contexts.

## The most important correctness risk: transforms across SiTU

The permutation proof is correct, but it proves equivalence only for an ordinary permutation. It does not justify propagating arbitrary Hadamard transforms, sign flips, or diagonal scalings across SiTU.

QTIP-style incoherence processing transforms weights and Hessians using orthogonal matrices such as

[
\widetilde W = U W V^T,\qquad
\widetilde H = VHV^T.
]

That is harmless inside a linear operation when the corresponding transformations are applied or cancelled on the activations. It is not generally harmless across a nonlinearity. ([NSF Public Access Repository][1])

In particular, even applying the same sign flip to both SiTU inputs is not a symmetry:

[
\operatorname{SiTU}(-g,-u)\ne\operatorname{SiTU}(g,u)
]

because (\sigma(-g)=1-\sigma(g)). A dense Hadamard is even less compatible:

[
\operatorname{SiTU}(Rg,Ru)\ne R\operatorname{SiTU}(g,u)
]

for a general orthogonal (R). Only the common coordinate permutation is exact.

This matters because the new format associates one semantic intermediate neuron with a `w1` row, `w3` row, and `w2` column. If EXL3 applies a hidden-axis output transform to `w1`/`w3` or an input transform to `w2`, you must be explicit about whether that transform:

* is reversed before SiTU;
* is performed online between SiTU and `w2`;
* is confined within a record and cancelled locally; or
* changes the coordinate system in which the rate records are defined.

A permanent hidden-axis Hadamard cannot simply be folded through SiTU in the way the permutation can. Likewise, any hidden-axis `suh`/`svh` sign vector must be undone or explicitly applied at the appropriate linear boundary; it cannot be silently treated as part of the function-preserving neuron permutation.

Recent rotation/group-quantization work independently finds that global rotations can conflict with localized quantization groups and recommends aligning the rotation scope to the quantization structure. That is another reason to test record-local or record-aligned transforms rather than inheriting a global transform without an ablation. ([arXiv][6])

I would add a formal **transform compatibility gate** before any mixed-format experiment:

* state the coordinate system of every rate record;
* list the left and right transforms for all three matrices;
* prove where each transform is cancelled;
* prohibit non-permutation hidden-axis transforms from crossing SiTU;
* test full-precision and BF16 closure with the actual serving transform order;
* compare global, 128-channel-local, and no hidden-axis transform where applicable.

This is more fundamental than the mode-selection statistics. A subtle transform mismatch could make a locally promising encoder impossible to serve exactly.

## Replace U4/M4/T8 with a rate-transfer ladder

Your current candidate family has a simpler underlying structure than the names suggest.

There are 24 records and 12 record pairs per expert matrix. After ranking records from low to high importance, define mode (R_r) as:

* the first (r) records use K2;
* the final (r) records use K4;
* the remaining (24-2r) records use K3;
* each K2 record is physically paired with a K4 record;
* the remaining records form K3/K3 pairs.

Every (R_r), for (0\le r\le12), has exactly three trellis bits per weight and exactly 12 fixed-size pair containers.

Under this notation:

* U4 is (R_0);
* T8 is (R_3);
* M4 is (R_6);
* the deferred `2,2,4,4` pattern is (R_{12}).

That relationship is visible in the record counts already derived in the document. 

This suggests that the first study should not treat U4, M4, and T8 as three unrelated codec modes. They are points on a one-dimensional **rate-transfer ladder**. Adding another (r) value does not introduce another decoder primitive: the kernel still sees only P24 or P33. It changes only how many of each pair an expert contains.

A four-bit expert field could represent all 13 possible values of (r). You probably would not retain all 13 in production, but searching them on the representative study would reveal the actual frontier. A plausible final alphabet might be something like (R_0,R_1,R_3,R_6), chosen after measuring mode-table regret. The current jump from (R_0) directly to (R_6) may simply be too coarse for many experts.

I would also estimate an **unconstrained record-map oracle** on a manageable subset. Let it choose any equal number of K2 and K4 records, subject to the same byte budget, then compare it with the best monotone (R_r) mode. The difference is the cost of the small mode alphabet. If that regret is small, you have strong evidence that the regular format captures most of the available allocation gain. If it is large, the ranking or mode family needs work before the kernel is frozen.

There is one systems consequence: at TP12, an (R_r) expert gives (r) ranks a P24 pair and (12-r) ranks a P33 pair. Equal bytes do not guarantee equal rank completion time. The post-encoding record order should rotate P24 ownership across expert IDs so that hot experts do not systematically make the same ranks slower. If P24 and P33 remain materially different in execution time, a 64-channel subrecord or larger super-record may eventually be needed for per-rank decode balance.

I would also rename the modes. “U4” is easily read as uniform four-bit quantization, and “mode-adaptive” sounds like a runtime decision. `R0`, `R3`, and `R6`, or explicit names such as `U3333` and `M2334`, make the rate semantics clearer. “Expert-static mixed-rate trellis codec” is less ambiguous than “mode-adaptive.”

## A shared permutation does not require a shared rate mode

The same physical neuron permutation must be used by `w1`, `w3`, and `w2`. It does not follow that all three matrices must use the same K schedule.

A neuron may be sensitive because:

* its `w2` column strongly affects the expert output;
* the `w1` branch is operating in a steep part of SiTU;
* the `w3` branch is operating in a steep part of SiTU;
* one matrix has unusually hard-to-quantize residual structure; or
* errors in the three matrices partially cancel or reinforce.

Consequently, “the neuron is low importance” does not imply that the same record should be K2 in every matrix. The current one-mode design deliberately imposes that constraint. 

I would initially use two mode IDs:

[
r_{13}\quad\text{for the fused `w1`/`w3` pair},\qquad
r_2\quad\text{for `w2`}.
]

Tying `w1` and `w3` preserves a regular fused gate/up kernel. Letting `w2` choose separately costs only a few cold metadata bits and does not require a runtime neuron gather, because the physical neuron order remains common. Each matrix still averages exactly three trellis bits.

The key experiment is:

* one shared (r) for all three matrices;
* separate (r_{13}) and (r_2);
* optionally fully separate (r_1,r_3,r_2) as an oracle.

If the shared mode is nearly as good, retain it for simplicity. If the two-mode version is materially better, the current coupling rule is throwing away quality for very little systems benefit.

## Improve the importance proposal, but keep functional encoding authoritative

The proposed score

[
s_{ej}=E[a_e^2h_{ej}^2\mid e\text{ routed}]
]

is a reasonable first score for the `w2` input columns. It measures how much gated energy enters each down-projection column. It is not yet a joint saliency measure for the complete gated neuron. 

Write

[
h_j=A(g_j)B(u_j),
]

where (A) is the gate-side SiTU term and (B) is the up-side term. To first order,

[
\delta h_j
\approx
A'(g_j)B(u_j),\delta g_j
+
A(g_j)B'(u_j),\delta u_j,
]

and

[
\delta y
\approx
W_2[:,j]\delta h_j
+
h_j,\delta W_2[:,j].
]

A useful joint proposal score therefore depends on more than (a^2h_j^2). It should include:

* applied gate-square mass;
* the downstream metric and norm/geometry of `W2[:,j]`;
* both SiTU derivatives;
* `w1` and `w3` quantization residuals;
* input covariance;
* branch-error covariance; and
* the measured K2/K3/K4 residuals, not merely source activation magnitude.

ScaleBITS reaches a related conclusion in the ordinary linear case: its sensitivity decomposes into an output-gradient factor, an input-activation factor, and the actual local quantization error. ([arXiv][3])

A practical hierarchy would be:

1. Use activation energy to get a cheap initial ordering.
2. Quantize a small sample of records at K2, K3, and K4 and estimate each record’s **donor cost**
   [
   C_j=D_j(2)-D_j(3)
   ]
   and **recipient gain**
   [
   G_j=D_j(3)-D_j(4).
   ]
3. Use those conditional slopes to propose low/high records.
4. Perform the complete heterogeneous full-H encode.
5. Select only by held-out functional distortion.

Because of dense-H coupling, (C_j) and (G_j) are proposal statistics, not additive truth. But they are closer to the real allocation question than raw activation energy.

The assumed LDLQ ordering also needs an ablation. Putting high-importance `w2` channels at the end of the permuted input axis makes the backward traversal quantize them first, which is plausible, but it is not guaranteed to be optimal under a dense Hessian. Compare at least:

* high-importance first;
* low-importance first;
* random order;
* Hessian-diagonal order;
* a pivoted-Cholesky or Schur-complement-based order.

QTIP’s backward traversal specifies the traversal direction, not the optimal semantic ordering of the coordinates. ([NSF Public Access Repository][1])

## “Coupled encoding” should eventually mean more than shared rates

The current plan couples the three matrices through their permutation, rate schedule, and final functional evaluation. The matrices are still reconstructed primarily as independent approximations of their source weights.

There is a stronger candidate-specific procedure:

1. Quantize `w1` and `w3`.
2. Run the captured inputs through those quantized matrices and obtain (\widehat h).
3. Quantize `w2` against the teacher expert output (y=W_2h), using (\widehat h) as its actual input.

The objective is then

[
\min_{\widehat W_2\in\mathcal C}
\sum_t\left|y_t-\widehat W_2\widehat h_t\right|^2_M,
]

rather than merely

[
\min_{\widehat W_2\in\mathcal C}
\sum_t\left|W_2\widehat h_t-\widehat W_2\widehat h_t\right|^2.
]

The unconstrained least-squares center for the first objective is

[
W_{2,\mathrm{LS}}
=================

Y\widehat H^T
\left(\widehat H\widehat H^T+\gamma I\right)^{-1}.
]

Trellis-quantizing a regularized version of this target would allow `w2` to compensate for some upstream `w1`/`w3` error. It changes the interpretation from “three independently compressed tensors” to “a compressed implementation of the complete expert function,” but that is already the direction of your functional objective. It should be an ablation after the basic reference codec works, with strong regularization and document-disjoint validation to prevent calibration overfit.

At minimum, `w2` should be quantized using the candidate’s actual (\widehat h) covariance. Using the teacher’s canonical (h) Hessian for every candidate can favor modes whose upstream errors change the `w2` input distribution.

## Calibration: separate coverage from production weighting

Your planned capture includes the right raw information: routes, gates, moments, Hessian rows, per-route `w2` inputs, and paired routed targets. The fallback and shrinkage language is also appropriate. 

I would use two logically separate calibration streams:

**The natural-routing stream** estimates the true deployment distribution: route mass, gate-square mass, expert co-occurrence, final expected damage, and global keep value.

**The expert-balanced stream** obtains sufficient Hessian and ranking support for tail experts. MoEQuant’s expert-balanced sampling is relevant here, but balanced samples must be reweighted back to the natural routing distribution when estimating production damage. ([arXiv][7])

The capture specification should report not only “ten million tokens,” but also, for every layer/expert assignment:

* routed row count;
* gate-square effective sample size;
* number of distinct documents;
* covariance effective rank;
* condition number after regularization;
* mode-effect standard error;
* train/validation ranking agreement.

Ten million model tokens could imply a large average number of routes per expert because each token executes 16 experts per layer, but the tail and document correlation determine whether those rows are actually informative.

There is also a practical storage issue. One FP32 `H13` plus `H2` is about 85 MiB per assignment:

[
(3584^2+3072^2)\times4\text{ bytes}.
]

Across 82,432 assignments, materializing all of them would be roughly 7.35 TB. That follows directly from the model dimensions and assignment count in the design.  

The all-expert pipeline should therefore keep compact sample reservoirs, covariance accumulators, or factor sketches and materialize/factorize one expert or batch of experts at a time. `w1` and `w3` can share the same `H13` factorization.

### The interim-teacher mismatch needs its own experiment

The capture plan uses the resident interim EXL3 artifact as teacher while the eventual encoder streams official source weights. The current pilot likewise used routes and inputs from the interim model and only the original weights that happened to survive in its retained tier.  

That is operationally understandable, but it creates a distribution mismatch:

* upstream hidden states come from an already quantized model;
* router decisions may differ from the official model;
* the neuron ranking is estimated under those altered states;
* the candidate weights are nevertheless reconstructed from the official checkpoint.

GEMQ’s results are a warning that quantization can change token-to-expert assignments and that importance estimates can change as the quantized model moves away from the original. ([arXiv][8])

Before relying on the interim teacher globally, I would run a limited streamed-official comparison on selected layers and documents. Measure:

* top-16 route overlap;
* applied-gate correlation;
* hidden-state and post-SiTU covariance differences;
* record-ranking Spearman correlation;
* agreement of selected (R_r) modes;
* whether the same experts appear high-value in the keep allocation.

If the agreement is high, the interim artifact is a justified calibration proxy. If not, use progressive recapture: construct an initial new artifact, capture again from that artifact, and re-encode once or twice. That is much less expensive than keeping the official model resident and addresses some of the distribution drift.

## Mode selection needs protection against winner’s curse

With 82,432 independent mode decisions, even a nominally held-out selector can accumulate many optimistic choices. The current tiny pilot is an extreme illustration: one validation row cannot distinguish a real expert-level effect from sample noise. 

I would make the conservative selection rule operational:

* split at the document level;
* select the mode on one fold;
* estimate paired per-document mode deltas on another;
* use a document-cluster bootstrap or other cluster-aware confidence interval;
* choose mixed rate only when the lower confidence bound clears a preset margin;
* shrink uncertain mode gains toward zero or a layer/family prior;
* use uniform (R_0) when effective support is poor.

The geometric mean of error ratios is useful as a descriptive statistic but should not be the principal optimization target. Ratios become unstable when the uniform-mode denominator is extremely small, and a geometric mean can understate a serious regression in a high-traffic expert. The primary summaries should be:

* traffic- and gate-square-weighted arithmetic excess distortion;
* median paired improvement;
* 90th and 99th percentile regression;
* worst regression among high-traffic experts;
* per-layer aggregates;
* mode frequency as a function of support and sensitivity skew.

The document’s validation gate already says not to hide a material regression behind a geometric mean; the reported experiment tables should implement that literally. 

## Score the routed mixture, not only isolated experts

For fixed teacher routing, the isolated gate-square objective is the correct self-term. But the complete routed error is

[
\left|\sum_e a_e\epsilon_e\right|^2_M
=====================================

\sum_e a_e^2|\epsilon_e|^2_M
+
2\sum_{e<f}a_ea_f\langle\epsilon_e,\epsilon_f\rangle_M.
]

The cross terms can produce either cancellation or reinforcement among the 16 selected experts. Your proposed complete-mixture teacher target recognizes this. 

Because each token activates only 16 experts, the full mixture does not make offline allocation intractable. For each MoE layer, you can:

1. initialize every expert to its best isolated mode;
2. cache the teacher and candidate expert outputs on routed samples;
3. maintain the current mixture residual;
4. update one expert’s mode using the exact incremental mixture score over all tokens that route to it;
5. sweep until no mode changes.

That coordinate descent captures co-occurrence and error cancellation while retaining one static mode per expert. The same idea can refine the final MXFP4 keep allocation after the additive knapsack solution.

This layer-replay objective still freezes teacher routing. End-to-end validation must then enable normal routing and measure:

* top-K route Jaccard overlap;
* applied-gate divergence;
* changed expert traffic;
* downstream recurrence of route differences;
* teacher-logit KL;
* final task quality.

GEMQ demonstrates that router shifts can be important in aggressively quantized MoEs. Its router fine-tuning is not necessarily appropriate for your codec, because it deliberately changes the model rather than preserving it, but it is a useful upper bound on how much of a regression is caused by routing adaptation rather than irrecoverable expert error. ([arXiv][8])

## Do not make local activation Hessians the unquestioned endpoint

Captured (H=E[xx^T]) is clearly much better than identity H in your pilot. That does not establish it as the best final distortion geometry.

YAQA directly challenges the usual local activation proxy. It approximates the Hessian of the end-to-end teacher KL using input- and output-side Kronecker factors and reports substantially lower teacher KL than ordinary LDLQ across QTIP and conventional quantizers. Its central point is that minimizing immediate layer output error need not minimize final model divergence. ([arXiv][9])

You do not need to replace the entire EXL/LDLQ encoder with YAQA in the first implementation. A sensible progression is:

1. captured input Hessian for the reference codec;
2. routed expert-output metric (M);
3. pre-RMSNorm or block-output Fisher metric for mode selection;
4. teacher-logit KL ranking on a representative subset;
5. a YAQA-style output factor as an eventual rounding control.

For `w1` and `w3`, an output-side metric is especially valuable because the ordinary input Hessian does not distinguish important output neurons. A diagonal or small-block approximation based on the SiTU Jacobian and `w2` geometry may provide most of the benefit without constructing a full output Hessian.

## Kernel and format feedback

The fixed pair format is well motivated, and the literature suggests that mixed-TCQ and mixed-precision fused kernels are feasible. Q-Palette implements fused mixed-rate TCQ, and MxMoE generates mixed-precision grouped GEMMs. Neither result guarantees that P24 is as fast as P33 on SM120, but they reduce the architectural uncertainty. ([arXiv][2])

The kernel study should answer four distinct questions.

**Decode work:** Is the K2-plus-K4 instruction count actually comparable to two K3 decodes? The K4 side may cost more than the K2 side saves.

**Register and pipeline behavior:** Equal bytes can still produce different live ranges, bit-extraction schedules, register pressure, or tensor-core feed rates.

**Rank balance:** At TP12, some ranks receive P24 and others P33 for a given expert. Measure collective tail latency, not just average per-rank throughput. Rotate physical pair types across experts to distribute any imbalance.

**Metadata closure:** If K-aware scale searches eventually use different metadata sizes or alignment, force the final pair container to one fixed physical size, even if that requires small padding. The constant-stride property is more valuable than saving a fraction of a percent of cold storage.

The benchmark matrix in the document is appropriate: isolated P24/P33, real routed U/M mixtures, TP4/TP12, graph capture, register use, occupancy, preparation time, and end-to-end fused-MoE latency. 

I also agree with deferring entropy coding. The weights are being optimized for a random-access fused consumer, not for archival size. Serial probability models would attack the wrong bottleneck unless confined behind independently decodable fixed records.

## The revised design I would test

I would preserve the core format but modify the encoder contract as follows.

1. Use one exact common physical neuron permutation across all matrices. Explicitly exclude hidden-axis sign or dense rotations from that equivalence proof.

2. Replace U4/M4/T8 with a monotone (R_r) mode family. Search all or most values of (r) during the representative study, then retain a small Pareto-optimal subset.

3. Store separate (r_{13}) and (r_2) mode IDs initially. Tie `w1` and `w3` for fused execution, but do not unnecessarily tie the down projection to them.

4. Construct the ranking with a cheap joint saliency proposal, then refine it using measured K2/K3/K4 donor and recipient deltas.

5. Preserve the full input-H LDLQ traversal for `w2`. For `w1`/`w3`, use the full common input Hessian for every output record and evaluate whether a diagonal output-side SiTU metric improves the row decisions.

6. Quantize `w1`/`w3` first, regenerate candidate-specific (\widehat h), and quantize `w2` using that actual input distribution. Test target-aware `w2` reconstruction as a later enhancement.

7. Select modes with full routed-mixture replay and document-clustered confidence bounds. Use isolated expert scores only for candidate pruning.

8. Use a natural-routing stream for expected damage and a balanced stream for covariance support. Reweight the latter and record effective sample size.

9. Solve the model-wide MXFP4 allocation additively, then perform one or more layerwise coordinate-refinement sweeps using the full mixture residual.

10. Validate with live routing and a teacher-logit metric before generating the production artifact.

## Decisive experiment sequence

The fastest path to a real answer is not yet a full 82,432-expert encode.

**Gate 0: algebra and format closure.** Implement the CPU mixed-K packer/decoder, one heterogeneous-K LDLQ traversal, exact P24/P33 byte accounting, the full transform audit, and BF16/TP permutation closure. No quality experiment can rescue a format that is not exact here.

**Gate 1: representative algorithm study.** Select hundreds of experts across early, middle, and late layers, stratified by natural route mass, effective sample size, activation skew, and current retained/compressed status. Do not restrict the study to retained MXFP4 experts. Compare:

* identity-H K3;
* captured-H (R_0);
* fixed (R_r) modes;
* selected (R_r);
* random record assignments;
* activation-only versus derivative/residual-based rankings;
* shared (r) versus ((r_{13},r_2));
* teacher-H versus candidate-(\widehat h) `w2`;
* independent-context EXL as a negative control;
* an arbitrary record-map oracle on a smaller subset.

The real go/no-go result is a positive document-bootstrap lower confidence bound for traffic-weighted routed-mixture improvement over **captured-H (R_0)** at equal exact physical rate. The current identity-H artifact is not the correct scientific control.

**Gate 2: mode-alphabet selection.** Plot distortion against (r), mode-table regret, support, and sensitivity skew. Choose the smallest subset of (R_r) modes that is close to the record-map oracle. This is the point at which the production mode table should be frozen.

**Gate 3: kernel closure.** Compare P24 and P33 at the kernel and routed-workload level. If P24 is materially slower or causes TP tail latency, revise the outer record before quantizing the complete model.

**Gate 4: teacher-proxy validation.** Compare interim-teacher and streamed-official traces on selected layers. Either justify the interim teacher quantitatively or add progressive recapture.

**Gate 5: all-expert allocation and live-model validation.** Only after the preceding gates should you generate the reusable candidate pool, solve the global keep allocation, refine it with mixture replay, and run final-logit, perplexity, reasoning, code, tool-use, and long-context evaluations.

## Bottom line

The core idea is technically credible and worth pursuing. Its strongest aspect is the fixed-payload systems construction, not the generic mixed-precision premise.

The five most important changes are:

1. Treat the exchange equation as a heuristic and use full-mode counterfactual encodes as the exact criterion.
2. Audit Hadamard and sign-transform placement; only the plain common permutation is guaranteed to commute with SiTU.
3. Generalize U4/M4/T8 into the (R_r) rate-transfer ladder.
4. Test a common permutation with separate gate/up and down-projection mode IDs.
5. Make calibration drift, statistical selection bias, routed-mixture cross terms, and live-router changes first-class validation targets.

The most plausible successful result is not that M4 should replace K3 everywhere. It is that a conservatively selected minority of experts with stable sensitivity skew can use nonzero (R_r), while uncertain or flat experts remain (R_0). That is already what the tiny pilot tentatively suggests. The main ways the idea could fail are also now fairly crisp: K2 damage may dominate K4 recovery, all-three-matrix coupling may erase the `w2` gain, hidden-axis transforms may wash out or invalidate the neuron structure, P24 may be slower than P33, or the selector may not generalize beyond the interim-teacher calibration distribution.

If those failure modes are ruled out at representative scale, the result would support a credible quantization-and-systems contribution rather than merely another mixed-precision heuristic.

[1]: https://par.nsf.gov/servlets/purl/10677431 "https://par.nsf.gov/servlets/purl/10677431"
[2]: https://arxiv.org/pdf/2509.20214 "https://arxiv.org/pdf/2509.20214"
[3]: https://arxiv.org/html/2602.17698v1 "https://arxiv.org/html/2602.17698v1"
[4]: https://arxiv.org/pdf/2505.05799 "MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design"
[5]: https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format "https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format"
[6]: https://arxiv.org/html/2607.27694v1 "https://arxiv.org/html/2607.27694v1"
[7]: https://arxiv.org/abs/2505.03804 "https://arxiv.org/abs/2505.03804"
[8]: https://arxiv.org/pdf/2605.23078 "https://arxiv.org/pdf/2605.23078"
[9]: https://arxiv.org/html/2505.22988v1 "Model-Preserving Adaptive Rounding"

