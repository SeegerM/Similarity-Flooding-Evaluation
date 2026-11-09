# SemStruct-reimpl: a preregistered reconstruction, specified before any accuracy was seen

**Status when written: the algorithm is implemented, its unit and control tests
pass, its cost is measured, and no score matrix has been compared against
ground truth.** The order matters. Two constants below are chosen by rules that
read seconds and node counts, and both rules had to be fixed while the only
numbers in hand were seconds and node counts.

SemStruct is Kang, Srinivas, Mihindukulasooriya, Shirai, Ram, Samulowitz and
Seneviratne, *"SemStruct: Contextualizing Semantic Embeddings with Structural
Information for Schema Matching"*, KDD '26; the paper is
`papers/3770855.3817963.pdf`. The implementation is `code/semstruct`, the
driver is `semstruct_run.py`, and the frozen configuration with a provenance
marker on every field is `code/semstruct/semstruct/config.py`.

## 1. Why this is a reconstruction rather than a run

There is no implementation to run. Searches for the title, for `SemStruct`, for
the authors, for the first author's repositories and for IBM Research's public
repositories return nothing; the first author's publication page links the
paper and an arXiv preprint but no code, and CatalyzeX still shows "Request
Code" rather than a repository. That is not proof that no private prototype
exists — KDD '26 is recent and an artifact release may simply be lagging, and
the cheapest way to improve on this document remains asking the authors — but
as of 2026-09-08 SemStruct is paper-only here.

So every number this produces is **our reconstruction of SemStruct, not
SemStruct**, and must be labelled that way wherever it is reported. The name
used in tables is `SemStruct-reimpl`.

The paper is unusually tractable for this. Unlike REMA (`REMA_REPRODUCTION.md`),
which is four pages, SemStruct's appendix quotes the backbone, its depth and
width, the optimiser, the learning rate, the batch size, the epoch count, the
early-stopping patience, the discretisation constant, the merge threshold, the
loss, its margin, and its mining rule. What follows separates what the paper
fixes from what we had to decide, and it is deliberately unflattering about the
size of the second list.

## 2. The single frozen variant

The paper is largely an ablation study: three row initialisations, four
serialisations, five PLM backbones, three GNN architectures, and a component
ladder. Reproducing the grid would be a search, and this project's standing
rule is that searches do not pay. **One configuration is built, chosen from the
paper's own reported best, and it is fixed before any evaluation scenario is
scored:**

| choice | value | the paper's own justification |
| --- | --- | --- |
| semantic model | frozen `e5-small` | A.1.2, and S5.5 finds E5 best overall |
| column serialisation | `header` | S5.4 and Table 6: lowest average rank across datasets, including on the opaque-header benchmark |
| row initialisation | zero | S5.3 and Table 4: best overall, and the "topological conduit" reading the paper's own conclusion rests on |
| GNN | 3-layer GraphSAGE, hidden 384 | A.1.2 |
| discretisation | k = 8 | S3.1, A.1.2 |
| merge threshold | tau = 0.9 | S3.2.1 — but see S4.2, this one does not survive contact with the checkpoint |
| loss | cosine triplet, margin 0.2, hardest positive and negative | S3.4.1 |
| optimiser | Adam, lr 5e-4, batch 16 tables, 10 epochs (Valentine) | A.1.2 |

Header-only serialisation is worth a sentence of its own, because it is the
choice a reader is most likely to query. It is defensible not only because the
paper ranks it first, but because it makes the reconstruction's structural
claim *harder* to satisfy: with `verbose` serialisation a column node already
carries a summary of its own values, so part of what the GNN is supposed to
supply is present before the first message is passed. Under `header`, anything
the column node knows about its own data arrived through the graph.

The `GAT` and `HGT` arms are excluded, and this is the paper's own reason
rather than ours: it reports that they ran into memory problems while
GraphSAGE scaled, which is why GraphSAGE is the backbone in A.1.2.

## 3. What the paper leaves open, and what we fix

One value each, fixed now, before any scenario was scored. The scoreboard rule
is the one `EMBEDDING_BASELINE.md` and `REMA_REPRODUCTION.md` already use:
trying several and reporting the best is the failure mode this document exists
to prevent. Each row is also marked `PREREG` in `config.py`, so the code and
this document cannot drift apart silently.

### 3.1 The semantic side

| decision | frozen value | why this one |
| --- | --- | --- |
| E5 prefix | `"query: "` on **every** text | E5 is trained with an asymmetric instruction prefix and produces degraded vectors without one. The model card prescribes `query:` on both sides for symmetric similarity, which is what column-to-column matching is. The paper does not mention the prefix at all. |
| pooling | attention-masked mean, then L2 | E5's own published convention; the paper restates neither. |
| truncation | 64 word-pieces | bounds the encoder, which is the only expensive stage. A header plus a value fits comfortably; long free-text values are the case this loses, and it is declared rather than hidden. |
| checkpoint version | `intfloat/e5-small`, snapshot `e272f3049e853b47cb5ca3952268c6662abda68f` | the exact revision is recorded so a rerun cannot silently pick up a re-uploaded checkpoint. |
| where L2 normalisation is applied | node features at initialisation; merged nodes after averaging; scoring by cosine. **Not** inside the GNN (`normalize=False`) | the paper never says. PyG's `SAGEConv` defaults to no normalisation, and the residual in Eq. 2 adds a GNN output to a unit-norm vector, so normalising inside the layers would make the two sides of that sum incommensurable. |

### 3.2 The graph

| decision | frozen value | why this one |
| --- | --- | --- |
| value-node scope | **per column** | forced by the paper's own value serialisation, `"column: {}, value: {}"`: the same string under two headers already has two different embeddings, so a global key would give one node two contradictory features. Note this differs from REMA, where global value identity is the entire mechanism. |
| merge scope | within a column | merging across columns would fuse two attributes into one node, which is the opposite of what a matcher comparing attributes needs. |
| merge algorithm | leader-based greedy, values visited by descending frequency then string | the paper says "merge cell node pairs" without saying whether merging is transitive. Transitive closure at a high threshold chains: a~b~c collapses even when a and c are unrelated. Leader-based merging cannot chain, and its order is deterministic without a seed. |
| merged node feature | mean of members, renormalised | the natural reading of "merge"; the alternative (keep the leader's vector) discards the members. |
| nulls | no value node, no edge | the paper is silent. This project's `is_null` rule is reused verbatim, so a cell counted absent by the recoded-corpus builder is counted absent here. The alternative — one shared null node — is a hub adjacent to nearly every row, which is FINDINGS 33 in graph form. |
| duplicate quantile boundaries | collapsed | a column that is 90% one value produces coincident quantiles. Collapsing gives it fewer than k bins; not collapsing gives it empty bins no value can occupy. |
| numeric detection | every non-null cell parses **and** more than k distinct values | the paper says only "numerical columns". Requiring every cell to parse keeps mixed columns categorical; the cardinality condition stops binning a coded flag that already has at most k categories. |
| graph type for the GNN | one homogeneous graph; node and edge types share parameters | GraphSAGE is a homogeneous operator. Typed parameters are what HGT is for, and the paper reports HGT as an arm it did not adopt. |
| row cap | see S4.1 | a cost rule, not an accuracy choice. |

### 3.3 The encoder and the objective

| decision | frozen value | why this one |
| --- | --- | --- |
| `SAGEConv` form | `lin_l(mean_j x_j) + lin_r(x_i)`, bias on `lin_l` only, `aggr='mean'`, `root_weight=True`, `normalize=False`, `project=False` | PyG's defaults, which is what "a 3-layer GraphSAGE backbone" means in a PyG paper. PyG is not installed here and pulling it in for four matrix multiplies is not worth the dependency, so the operator is reimplemented in plain torch and checked against the definition by `test_sage_matches_reference`. |
| activations | ReLU between layers, none after the last | PyG's own examples; the paper does not say. |
| dropout / normalisation | none | not mentioned; adding either would be a second method. |
| MLP | `Linear(384,384) - ReLU - Linear(384,384)` | "a simple 2-layer feedforward network that projects the GNN output" (Eq. 2). The output dimension is forced by the residual. |
| views per source table | 4 | the paper says "a set of augmented views" and gives no count. |
| row / column sampling | uniform in [0.5, 1.0] of each, minimum 2 columns | "randomly sampling rows and columns", no rates given. |
| header noise | one of {typo, format, acronym}, uniformly, on each kept column with p = 0.5 | the three kinds are the paper's; the rate is ours. |
| value noise | a kept column is noisy with p = 0.3; within it a cell is perturbed with p = 0.3 | as above. |
| noise operators | transpose / delete / duplicate / adjacent-key substitution; case, separator and numeric-format changes; initialism or vowel-dropping | written from the descriptions, deliberately **not** by reading Valentine's fabricator, so the model is not trained on the exact transformation it is tested against. |
| batch construction | 8 source tables x (original + one sampled view) = 16 graphs | the smallest construction satisfying both "batch size of 16 tables" (A.1.2) and "every table has at least one positive match" (S3.4.2). |
| loss reduction | **mean** over anchors, not sum | Eq. 3 is written as a sum. With a sum the effective step size scales with how many columns the sampled tables happen to have, so the paper's single learning rate would mean different things on a 6-column Magellan table and a 38-column OpenData one. |
| Adam | torch defaults, `weight_decay=0` | the paper names the optimiser only. |
| gradient clipping | norm 1.0 | not mentioned; stability, and fixed rather than tuned. |
| validation split | 10% of training tables, patience 5, best state restored | the paper uses "the validation set" with patience 5 (A.1.2) but Valentine has no such split, so one is carved out of the training tables. |
| seeds | 20260908 everywhere | the paper reports none. |
| training data | distinct **source** tables only, de-duplicated by content | the paper's leakage rule (S3.4) is that targets are never seen; de-duplication is ours, because the corpus reuses one source table across many scenarios and training on it thirty times would reweight the objective toward whichever family fabricates the most scenarios. |
| scoring | raw cosine over final column embeddings | see S5. |
| the paper's own extraction | softmax over targets at temperature 1, then Magneto's bipartite reranking, transcribed from `code/magneto-matcher/.../bp_reranker.py` | the paper says "softmax-normalized cosine similarities" and "the same bipartite reranking algorithm used in some Magneto variants", so the algorithm is recoverable but the axis and temperature are not. Secondary arm only. |

## 4. The two constants fixed by measurement

### 4.1 The row cap

The paper ran whole tables on an H100; the corpus here has a median of 7,500
rows per table and this machine has no GPU. The expensive stage is not the
3-layer GraphSAGE — it is embedding one value node per distinct value per
column, before any merging. Measured on this machine, `e5-small` encodes about
170 short texts per second at batch 128 on 8 threads.

**The rule, fixed before the cap was chosen:** take the largest round prefix
cap whose complete two-regime run over all 382 scenarios finishes within one
overnight window, measured on a family-spanning sample covering every (family,
scenario type) class in the cohort, in both regimes. The recoded regime is
measured because it is the expensive one: hashing removes every repeated value,
so a capped column contributes its full row count in distinct value nodes.

*What the rule selected, and the measurement it read, are recorded in S4.3
below and in `reports/semstruct_20260908/cost.json`.*

Prefix rather than a random sample, for the reasons `REMA_REPRODUCTION.md` S4
gives: deterministic without a seed, and consistent with the row-cap precedent
already in this repository (REMA 1,000, LIMA profiling 1,000).

### 4.2 The merge threshold does not survive contact with the checkpoint

This is the one place where a quoted constant from the paper cannot be applied
as written, and it was found by measurement before any scenario was scored.

PAPER S3.2.1 sets tau = 0.9 on the cosine similarity between value-node
embeddings. But a cosine threshold is only meaningful relative to the scale of
the embedding space, and that scale is a property of the checkpoint, not of the
rule. Frozen `e5-small` is strongly anisotropic — the phenomenon the paper
itself cites (its reference [6], Ethayarajh 2019) — so two arbitrary short
strings already sit near cosine 0.95. Measured on three families of this
corpus, over the paper's own value serialisation:

| family | mean fraction of value pairs at raw cosine >= 0.9 | the same on column-centred cosine |
| --- | ---: | ---: |
| Wikidata | 0.962 | 0.001 |
| ChEMBL | 0.874 | 0.001 |
| TPC-DI | 0.756 | 0.013 |

Read the two columns together. Applied literally, tau = 0.9 merges three
quarters to nineteen twentieths of all value pairs, and a whole table's value
nodes collapse to a handful — on one Wikidata table, 716 distinct value tokens
became 21 nodes. That does not "preserve the semantic relationships between
similar values" (S3.2.1); it deletes them. Applied to centred cosine, which
removes the common direction responsible for the anisotropy, the same threshold
merges about one pair in a hundred and the step becomes a no-op.

Neither reading does what the paper describes the merge as doing. So the
threshold becomes a **declared three-arm ladder, fixed here before any
scenario was scored**, rather than a silent correction:

| arm | rule | status |
| --- | --- | --- |
| `raw` | tau = 0.9 on raw cosine | **primary.** The literal reading of the published constant. |
| `centered` | tau = 0.9 on column-centred cosine | the anisotropy-corrected reading the paper's own citation motivates |
| `none` | no merging | the limit the `centered` arm nearly reaches, stated exactly rather than approximately |

The primary arm stays literal on purpose. If the reconstruction underperforms
there and recovers under `none`, the finding is "the published threshold is not
transferable to the published checkpoint", which is a reproducibility result
about the paper; silently substituting a corrected threshold would convert that
result into a number nobody could interpret. Both readings are on the table,
and neither was chosen after looking at accuracy.

### 4.3 What the measurement returned

Twenty-six graph-pair builds per cap — every (family, scenario type) class in
the cohort, in both regimes — with a cold cache.
`reports/semstruct_20260908/cost.csv`, raw run in `logs_semstruct_cost.txt`.

| cap | mean s/pair | median | slowest | plain | recoded | projected h (382 x 2 regimes) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 7.3 | 6.0 | 26.6 | 3.6 | 11.1 | 1.6 |
| 200 | 11.5 | 9.0 | 41.9 | 6.2 | 16.8 | 2.4 |
| 400 | 22.7 | 17.5 | 88.8 | 10.0 | 35.3 | 4.8 |

The slowest class is TPC-DI Unionable under the recoding at every cap, and the
recoded regime is 3.5x the plaintext one at cap 400. That gap is the predicted
one: hashing removes every repeated value, so a capped column contributes its
full row count in distinct value texts to embed.

**What the rule selected: 400 rows per side.** Three declarations about how
that number was reached, because the reader should be able to check it rather
than take it:

1. The candidate ladder `{100, 200, 400}` was fixed when the pilot was
   launched, before any of its output existed. Extrapolating the measured
   trend (about 1.8x per doubling in the recoded regime) puts cap 800 near 8.7
   hours for scoring alone, which does not leave an overnight window once
   training is added; cap 400 is the largest rung of the declared ladder, and
   the extrapolation says it would also be the largest rung of a longer one.
2. Scoring all four arms costs little more than one: `semstruct`, `lm_only` and
   `zero_delta` share their graphs exactly, and `permuted` changes only the
   *order* of cells within a column, so its value texts are identical and the
   encoder cache serves them. The three merge arms likewise share every
   embedding; only the merge and the edge lists are recomputed.
3. `config.py` already carried `row_cap = 400` as an unvalidated placeholder
   when the pilot was launched, marked `COST` and pointing here. The rule was
   declared before the measurement and the measurement agreed with the
   placeholder; it was not chosen afterwards to match it. Recorded because the
   coincidence is exactly the kind a reader is right to be suspicious of.

**And the merge collapse, now measured across the whole class sample** rather
than the three families of S4.2. Value tokens presented to the merge, against
value nodes surviving it, at cap 400 over all 26 (class, regime) source tables:

| arm | value nodes kept | share of tokens |
| --- | ---: | ---: |
| `raw` (tau = 0.9, literal) | 1,531 | **3.8%** |
| `centered` (tau = 0.9, anisotropy-corrected) | 39,156 | 96.1% |
| `none` | 40,759 | 100% |

The literal reading discards 96% of the value nodes, and the corrected reading
discards 4%. The extremes are worse than the average: under the recoding,
Magellan's 1,097 value tokens become **4** nodes, and TPC-DI Unionable's 4,188
become 22. A column whose values all collapse into one node contributes a
single constant neighbour to every row it touches, which is not a sparsity
reduction — it is the removal of the signal the row nodes exist to carry.

This is measured before any scenario was scored, and it is why S4.2's ladder
exists. It also sharpens prediction P5 below: on the recoded regime the `raw`
arm has almost no graph left, so if it nonetheless matches `none` there, the
explanation cannot be that value nodes were carrying the result.

## 5. Arms

Both regimes, all 382 scenarios, the standing 44 exploration / 338 confirmation
cohorts every other table in this project uses.

1. `semstruct` — the reconstruction at the frozen configuration.
2. `lm_only` — **the frozen-E5 column baseline**, and the paper's own `LM`
   comparison in spirit. It is not a separately written baseline: it is the
   same graphs, the same serialisation and the same code path with the GNN and
   the MLP skipped, so a difference between it and arm 1 cannot come from a
   difference in preprocessing.
3. `permuted` — **null control for the structural claim.** Cells are permuted
   independently within each column, so every column keeps its exact multiset
   of values, its cardinality, its value nodes and its column-value edges,
   while row co-occurrence is destroyed. This is the arm that decides whether
   the reconstruction uses the information SemStruct claims to use.
4. `zero_delta` — MLP(.) forced to zero, which by Eq. 2 must reproduce arm 2
   exactly. A blocking self-check on the pipeline, not a result.

Each of the three merge arms in S4.2 is a separate directory and a separate
checkpoint; a checkpoint trained under one graph construction is refused
against another by a configuration fingerprint.

Beside these, and reported separately rather than mixed in:

* `*_native` — the paper's own softmax + bipartite reranking applied to the
  same matrices, so system-faithful numbers exist alongside the ones the common
  extractors produce.
* a recoded-training arm — the primary checkpoint is trained on plaintext
  source tables and applied to **both** regimes, which is what deploying one
  trained matcher means and is the paper's own single-training-run setting.
  `--train-regime audited_md5` trains a second checkpoint on the recoded source
  tables, so the alternative reading is available rather than assumed away.
* a held-out-family arm — train with one family withheld, evaluate on it. The
  paper trains on Valentine source tables and evaluates on Valentine pairs, so
  its own protocol has no family holdout; this is the leakage-free reading, and
  it is an extension beyond the paper's protocol, exactly as the held-out-family
  arm in `SIMA_INTEGRATION.md` is.

## 6. The validation ladder, and what each rung can catch

Four checks. The first three are mechanical and run before any accuracy is
reported; the fourth is the sanity comparison.

1. **LM-only control.** Disabling the GNN must reduce exactly to the frozen E5
   column representation. Asserted by
   `tests/test_semstruct.py::test_lm_only_control_is_exactly_the_frozen_embedding`
   and reported per-scenario by `semstruct_run.py verify`.
2. **Row permutation.** Independently permuting cells within each column
   destroys row co-occurrence and preserves column marginals exactly. The
   structural gain over arm 2 ought to shrink strongly. This is the rung that
   matters most: it is the difference between "the reconstruction uses
   structure" and "the reconstruction gained from some other preprocessing
   choice", and no accuracy table can distinguish those two on its own.
3. **Zero structural delta.** Zeroing the MLP's last layer must reproduce the
   semantic initialisation bit for bit. Asserted by
   `test_zero_structural_delta_reproduces_semantics` and checked corpus-wide by
   `verify`, which **blocks**: a mismatch means something outside the residual
   is moving the scores.
4. **Published-benchmark sanity.** Run the Valentine families the paper
   reports. Exact numbers are not expected — a different row budget, no GPU,
   and a dozen unspecified details stand between us and them — but the
   qualitative relation `SemStruct > frozen E5` should appear, most clearly on
   structurally ambiguous cases.

## 7. Metrics and extraction

The standing set, unchanged: Hungarian F1 and mutual-top1 F1, pessimistic
Recall@GT and pessimistic MRR, on the same cohorts, with paired deltas and
win/loss counts. Matrices failing `matrix_health.health` are excluded, as
FINDINGS 77 requires. Single-correspondence scenarios are reported separately,
per FINDINGS 89.

The primary artefact is the **raw cosine matrix**, not the paper's
softmax-and-rerank output. That is this harness's whole point: SemStruct
naturally produces final column embeddings, so its score-generation stage plugs
into the common extractors directly, and running a competitor under its native
extractor while running ours under ours confounds scoring quality with
selection policy. The native transformation is reproduced as a separate arm so
the confound can be measured rather than argued about.

Note also that Recall@GT and MRR — the paper's own two metrics — are already in
this harness and are extractor-free, so on those two the comparison is direct.
One wording check, since it decides what the numbers mean: the paper describes
Recall@GT as top-k "for a given source column", but the definition it cites is
Valentine's, which takes the |GT| best-scoring cells of the **whole** matrix.
`smx.core.recall_at_ground_truth` implements Valentine's, and reports it under
both tie orders — which matters more here than usual, because the raw-merge arm
is the one most likely to produce a flat matrix.

## 8. Predictions, fixed before any comparison

Written now so that no outcome can be reinterpreted afterwards.

* **P1 (blocking).** `zero_delta` and `lm_only` are identical to the last
  written digit on every scenario in both regimes. If not, nothing else here
  may be reported until it is explained.
* **P2.** On the recoded (MD5) regime, `semstruct` and `lm_only` are both near
  chance and close to each other. SemStruct reads headers and values through a
  language model; under a per-side salted recoding there is no linguistic
  content left on either side, and the graph structure that survives is
  identical in shape but carries meaningless features. A contemporary semantic
  matcher having no answer to opaque schemas is the scope limit this project
  already documents for embedding baselines, and it is a prediction, not a
  criticism.
* **P3.** On the plaintext regime, `semstruct` beats `lm_only` on MRR and
  Recall@GT, and the margin is larger on Semantically-Joinable and
  View-Unionable than on Unionable. Unionable pairs share a schema, so header
  semantics alone nearly solve them and there is little room above the LM
  baseline; the paper's own Table 10 shows exactly this shape, with its
  baselines weakest on Semantically-Joinable.
* **P4.** The gain in P3 shrinks under `permuted`. If it does not, the
  reconstruction is not using row co-occurrence and P3's gain has some other
  cause, which must be found before the number is reported.
* **P5.** The `raw` merge arm underperforms `none`. This follows from the S4.2
  measurement rather than from any result: at raw tau = 0.9 a table's value
  nodes collapse to a handful, so the column nodes have almost nothing
  distinct to aggregate. If `raw` nonetheless matches `none`, then value nodes
  contribute little in this corpus and the structural claim rests on row
  topology alone — which would be a finding worth reporting on its own.

## 9. Running it

Needs the embedding environment (`torch`, `transformers`) and about 130 MB of
checkpoint, already fetched:

```powershell
harness\.venv-embed\Scripts\python.exe code\semstruct\tests\run_tests.py
harness\.venv-embed\Scripts\python.exe harness\semstruct_run.py cost
harness\.venv-embed\Scripts\python.exe harness\semstruct_run.py train
harness\.venv-embed\Scripts\python.exe harness\semstruct_run.py score --regime plain
harness\.venv-embed\Scripts\python.exe harness\semstruct_run.py verify
harness\.venv-embed\Scripts\python.exe harness\semstruct_run.py report
```

`--merge centered|none` selects a sensitivity arm; `--hold-out-family` trains
the leakage-free arm; `--native` also writes the paper's own softmax + rerank
matrices.

## 10. What must be said wherever these numbers appear

Three sentences, and they are not optional.

1. This is a reimplementation from the paper, not the authors' code, which has
   not been released.
2. Every implementation choice the paper does not specify was fixed before any
   evaluation scenario was scored and is listed in S3 of this document.
3. The published merge threshold could not be applied as written to the
   published checkpoint (S4.2); the primary arm keeps it literal and two
   declared alternatives are reported beside it.

That turns a missing upstream artifact from a weakness of the comparison into a
reproducible reimplementation exercise, which is the register the E&A paper is
already written in.
