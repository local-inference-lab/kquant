# Investigator brief: unified SQG-Cheb for K2/K3/K4

Status: research request, 2026-08-03.

## Objective

Revisit SQG-Cheb as a **unified K2/K3/K4 reconstruction design**.  The prior
study developed and tuned K3/K4/K5, but K2 was only evaluated afterward by
applying the K3/K4 normal staircase to the K2 graph.  That is not a sufficient
K2 investigation because K2 is the donor side of every equal-rate K2/K4
exchange in SQRT-C.

The preferred outcome is one common rank-to-E4M3 law, or one common
construction with minimal shared parameters, that works well at all three
rates.  Independently optimized K2, K3, and K4 laws should still be measured
as diagnostic oracles: they quantify the quality cost of unification and may
justify a rate-specific fallback if the shared solution leaves a material
gap.

## Frozen constraints

- Rates are K2, K3, and K4.  K5 is out of scope.
- Keep L16.
- Keep the existing SQG history mixer, syndrome mixer, branch permutation,
  successor graph, and baseline phase rule at every rate.
- Do not introduce K-specific phase shifts, Gray-XOR phase edits, or different
  graph topologies.
- Output finite E4M3FN with round-to-nearest-even semantics.
- Refit the ordinary encoder scales independently for every candidate and
  rate; reusing a K3 scale at K2 is not a controlled comparison.
- A 65,536-byte rank LUT is acceptable as a research artifact.  A production
  proposal should preferably have a compact procedural or coefficient form,
  ideally no more than roughly 8 KiB of shared data.
- Do not alter the currently running all-expert SQG-R44 candidate pool.

## Why K2 is the key test

At K2, each state exposes only four outgoing reconstruction choices, one per
quantile stratum, and new bits replace the L16 window more slowly than at K3
or K4.  Tail allocation and central resolution can therefore have a different
tradeoff.  The unclipped SQG-Cheb law restored tail resolution and clearly
helped K4, but it did not reliably help K2.

On the current 24-expert production-path panel, replacing clipped R44 with the
shared full-tail Chebyshev staircase produced:

| Rate | Dense-H SSE change | Validation routed-SSE change | Wins |
| --- | ---: | ---: | ---: |
| K2 | +0.013% | -0.197% | 9/24 |
| K3 | -0.193% | -0.276% | 17/24 |
| K4 | -0.685% | -0.811% | 22/24 |

The K2 aggregate improved slightly, but its median expert regressed by 0.147%
and the K2-donor/K4-recipient curvature was essentially unchanged.  The next
study must include K2 during fitting and candidate selection, rather than
extrapolating a law selected at other rates.

## Requested investigation

1. Search reconstruction laws jointly across K2/K3/K4.  Include clipped R44
   and the current full-tail SQG-Cheb law as controls.
2. Explore principled compromises between K2 central resolution and K4 tail
   resolution.  Reasonable families include soft tail compression,
   generalized-normal or mixture companders, learned monotone staircases, and
   E4M3-aware Chebyshev approximations synthesized against rounding intervals.
3. Alternate path assignment, scale fitting, and reconstruction-law fitting
   where practical.  The law should be optimized for trellis paths, not only
   for scalar CDF matching.
4. Compare three levels explicitly:
   - one identical rank law for K2/K3/K4;
   - one unified formula with a very small number of rate parameters; and
   - independently learned per-rate laws as an oracle.
5. Include uniform K2/K3/K4 measurements, but make the actual SQRT-C criterion
   the K2 donor cost versus K4 recipient gain:

   ```text
   donor cost    = D(K2) - D(K3)
   recipient gain = D(K3) - D(K4)
   ```

   These slopes are screening statistics.  Full mixed K2/K4 encodes remain
   authoritative because dense-H BlockLDLQ couples records.
6. Use disjoint fitting, selection, and blind-validation samples.  Synthetic
   searches are useful, but do not claim a production winner from synthetic
   NMSE alone.  Supply the best candidate rank laws so we can replay them in
   the real Hadamard, dense-H BlockLDLQ, and routed-function harness.

## Required output

- A concise report explaining the best unified design and its K2/K3/K4
  tradeoff.
- Same-harness results for both existing controls and all finalists.
- The best shared candidate plus the best per-rate oracle, with exact
  65,536-entry E4M3 rank LUTs and machine-readable parameters.
- A compact procedural/Chebyshev implementation for any production candidate,
  with exhaustive all-rank agreement against its LUT.
- Per-rate NMSE, dense-H or supplied-trace loss, donor/recipient slopes,
  win rate, median change, and important tail regressions.
- Exact table/coefficient bytes, checksums, seeds, and data splits.

The success condition is not merely lower average K2 NMSE.  A useful result
must reduce held-out K2 donor damage without surrendering the established K3
and K4 gains, and it must improve the number and margin of viable equal-byte
K2/K4 exchanges.  If no shared law does that, quantify the shared-versus-
per-rate regret clearly rather than forcing a unified conclusion.
