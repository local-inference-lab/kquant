# Kimi-K3 broad calibration corpus and expert-stratified H2 plan

## Current broad production candidate

The source-controlled broad corpus is planned and integrity-checked:

| Split | Report | Tokens | Documents |
| --- | --- | ---: | ---: |
| training | `out/k3-denseh-broad-v5-train-corpus.json` | 20,000,000 | 23,712 |
| mode selection | `out/k3-denseh-broad-v5-selection-corpus.json` | 2,000,000 | 2,382 |
| final validation | `out/k3-denseh-broad-v5-final-validation-corpus.json` | 2,000,000 | 2,310 |

The training split uses 75% of the content-hash space and the two held-out
plans draw only from the reserved quarter. The reports also authenticate and
exclude every document and tokenized prompt from the earlier v3 plans. The
combined six-report validator result is
`out/k3-denseh-broad-v5-validation.json`: there are zero raw-record and zero
post-tokenization prompt collisions within or across the old and new splits.

The v5 token mixture is:

| Lane | Share | Per-document cap |
| --- | ---: | ---: |
| FineWeb-Edu explanatory prose | 15.0% | 2,048 |
| OpenWebMath mathematics/science | 16.0% | 3,072 |
| FineWeb2 multilingual text | 20.0% | 1,024 |
| UltraChat dialogue | 15.0% | 2,048 |
| SWE-agent trajectories | 8.0% | 4,096 |
| OpenHands trajectories | 4.0% | 4,096 |
| APIGen + ToolACE | 2.0% | 2,048 / 4,096 |
| local diverse/general tasks | 9.0% | 1,024 |
| local generic reasoning | 4.5% | 768 |
| local coding/agentic v3 | 3.0% | 1,024 |
| deep structured contexts | 3.2% | 4,096 |
| long agent traces | 0.3% | 8,192 |

The multilingual share contains Chinese at 4% and Spanish, French, German,
Japanese, Korean, Russian, Arabic, and Hindi at 2% each. These are explicit
token quotas, not proportions inherited from a source dataset.

## Earlier local seed

The first reproducible local-only seed remains useful as an independent prior
study and as an exclusion set:

| Split | Report | Tokens | Documents |
| --- | --- | ---: | ---: |
| training | `out/k3-denseh-broad-v3-corpus.json` | 10,000,000 | 18,983 |
| mode selection | `out/k3-denseh-broad-v3-selection-corpus.json` | 1,000,000 | 1,882 |
| final validation | `out/k3-denseh-broad-v3-final-validation-corpus.json` | 1,000,000 | 1,915 |

All three splits have zero pairwise overlap in raw-record hashes and in the
actual token sequence presented to Kimi after chat templating and truncation.
The validator result is `out/k3-denseh-broad-v3-validation.json`.

The seed mixture is:

| Source | Share | Per-document cap |
| --- | ---: | ---: |
| diverse calibration | 45% | 768 tokens |
| generic/agentic calibration | 28% | 512 tokens |
| coding/agentic v3 | 20% | 768 tokens |
| deep structured/tool records | 7% | 4,096 tokens |

This seed is no longer the preferred production capture distribution. Source
names are not reliable semantic labels: for example, the generic set contains
substantial general reasoning and classification, while the deep set is
heavily structured/tool-oriented. Coverage must be measured from content and
routing, not inferred from filenames.

## Staged source material

A local GLMFlash corpus contains 120,000 prompts spanning UltraChat,
SWE-agent, OpenHands, APIGen, ToolACE, and the four existing local sets over a
128-to-32K context distribution. The K3 planner now accepts its
`conversations` schema and can filter the embedded `source` field. The
byte-preserving sharder is `scripts/shard_calibration_corpus.py`.

The complete 120,000-document training inventory is split by semantic source
at:

```text
/data/kquant/corpora/k3-hybrid-v2-train-by-source-v1/
```

Its manifest authenticates the 2.2 GiB input and all ten output shards. This
prevents the original hybrid's heavily agentic aggregate distribution from
silently determining the K3 mixture.

The 2,048-document held-out portion has been split and hashed at:

```text
/data/kquant/corpora/k3-hybrid-v2-validation-by-source-v1/
```

An initial controlled 1M-token pilot is planned at
`out/k3-denseh-broad-v4-source-pilot-corpus.json`. It contributes:

| Lane | Tokens | Documents | Cap |
| --- | ---: | ---: | ---: |
| UltraChat | 350,000 | 334 | 2,048 |
| SWE-agent | 250,000 | 68 | 4,096 |
| OpenHands | 100,000 | 28 | 4,096 |
| diverse local | 150,000 | 257 | 768 |
| deep structured/tool | 100,000 | 28 | 4,096 |
| agentic/coding v3 | 50,000 | 104 | 768 |

It is also raw-record- and prompt-disjoint from all three v3 splits. This
pilot proves source ingestion and splitting; its long agent trajectories make
it deliberately unsuitable as the final production proportions.

Revision-pinned external shards are staged at:

```text
/data/kquant/corpora/k3-broad-external-v1/
```

They include 5,000 FineWeb-Edu documents, 5,000 OpenWebMath documents, 2,000
Chinese FineWeb2 documents, and 1,000 documents for each of eight additional
FineWeb2 language/script configurations. Every JSONL has an adjacent manifest
with the exact upstream dataset, revision, configuration, filters, document
count, byte count, and SHA-256. The pinned revisions are:

- FineWeb-Edu: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- FineWeb2: `af9c13333eb981300149d5ca60a8e9d659b276b9`
- OpenWebMath: `fde8ef8de2300f5e778f56261843dab89f230815`

## Target breadth

Before the production capture, assemble independently controllable lanes for:

- ordinary explanatory prose and multi-turn dialogue;
- code in several languages plus realistic repository-agent trajectories;
- tool/API calling and structured extraction;
- mathematics, science, and symbolic notation;
- Chinese and other multilingual text;
- creative writing, classification, and short instructions;
- long technical documents and long multi-turn contexts.

The existing local material covers code, agentic work, creative prompts, and
some structured/tool use well. The largest gaps are independently sourced
general prose, multilingual material, and math/science. Candidate additions
are revision-pinned samples from FineWeb-Edu, FineWeb2, and OpenWebMath, plus
the APIGen and ToolACE records already present in the local hybrid candidate
pool. Every normalized shard must retain upstream dataset ID, revision,
configuration, source-row identity where available, byte hash, license, and
normalization version.

Do not freeze the final token proportions from intuition alone. Run an exact
route/gate census first, then adjust lanes that add expert coverage or alter
the tail-expert support distribution. A lane that contributes many tokens but
no new routing support should not crowd out a smaller, complementary lane.

## Split and integrity contract

1. Hash and assign complete source records before token truncation.
2. Also hash the final token sequence after chat templating and truncation.
3. Deduplicate both hashes within and across every source.
4. Keep training, mode selection, and final validation document-disjoint and
   prompt-disjoint. Final validation is never used for Hessians, permutations,
   mode selection, codebook choice, or keep allocation.
5. Pin every source file by SHA-256 and every upstream source by revision.
6. Give long-context lanes explicit caps rather than allowing a few records to
   consume a domain's entire token quota.
7. Validate all plans with `scripts/validate_calibration_corpus_plans.py`
   before starting vLLM.

## Expert-stratified capture

The new corpus fixes diversity and route volume; it does not by itself make a
layer-global `H2` valid. Intermediate-neuron axes are expert-local. The capture
sequence is therefore:

1. Run a cheap natural-routing census over the full training plan. Record
   exact route count, gate mass, gate-square mass, distinct documents, and
   source lane for every layer/expert assignment.
2. Allocate document-aware reservoir targets per assignment. Favor tail
   experts and cap very hot experts instead of applying one global route
   sampling denominator.
3. Capture canonical post-SiTU `h_e` rows with expert ID, gate, document ID,
   source lane, and observation ID. Keep the natural sampling probability so
   expected-damage estimates can be reweighted correctly.
4. Encode each supported `r13` candidate, decode its actual stored `w1/w3`,
   replay the expert reservoir through SiTU, and build `H2[e,r13]` one
   candidate at a time. Shrink it toward
   `trace(H2[e,r13]) / 3072 * I`, encode the corresponding `w2` candidates,
   and discard the dense factor. Do not persist 82,432 dense
   3072-by-3072 matrices (or three times that many conditional matrices).
5. Use identity H whenever support, document diversity, effective rank, or
   held-out covariance prediction is inadequate.

A useful initial storage frontier is 512 to 2,048 BF16 `h` rows per
assignment, roughly 260 GiB to 1.04 TiB model-wide. The route census should
choose where inside that range each expert lands. The old 65,536-token pilot
averaged only about 18 sampled `w2` rows per expert; increasing the corpus
without changing the sampling policy would leave the tail uncontrolled.

For each expert report at least:

- natural route and gate-square mass;
- raw row count, gate-square effective sample size, and distinct documents;
- source-lane coverage;
- covariance effective rank and condition after shrinkage;
- validation prediction error for identity, diagonal, low-rank/shrunk, and
  dense candidate metrics;
- stability of neuron ranking and `(r13, r2)` mode selection across document
  folds.

The production acceptance criterion is held-out routed-function improvement
over the identity-H QSRT control. Dense-H training loss alone is not evidence
that a covariance estimate generalizes.
