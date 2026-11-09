# Pre-registration: the Melnik reference, layer 1 and layer 3

Written 2026-10-02, 21:00, **before any layer-1 or layer-3 result was scored** (details under
"What had been seen"). It covers the initial-matcher axis (RQ1), the graph-construction axis (RQ2),
the parameter arms outside the tuning protocol (RQ3a), the data axis (RQ4) and efficiency (RQ5).
The tuning protocol (RQ3b: screen, dev, lockbox) stays governed by `tuning/PREREGISTRATION.md`
and its amendments; nothing here changes it. Deviations from this file are appended as dated
amendments; its SHA-256 is logged when the first layer is scored.

## 1. Reference configuration

SF as published (Melnik, Garcia-Molina, Rahm, ICDE 2002), as close as Schematch allows; details
and the paper mapping in `reference/README.md`:
- the OIM-style schema graph, no dependency edges; formula C; inverse-average coefficients;
  normalisation by the maximum; stop when the Euclidean residual is below 0.05 (at most 200 steps);
- the initial matcher's matrix on column-name pairs (NAME attachment), and the paper's prefix/suffix
  StringMatch on every other pair of non-ID nodes;
- the paper's selection, SelectThreshold with t_rel = 1.0, is mutual top-1 (F1_M).

Known, unavoidable deviations: Valentine's CSV tables declare no keys (the paper's graph has
declared keys); data types are inferred from values.

## 2. Designs

Every arm differs from the reference in exactly one module; all comparisons are paired on the same
initial matrix and scenario. Initial matrices are the seed study's stored first-line matrices (C2 in
`sanity_checks.py`: SF reproduces the seed study from them exactly).

**Layer 1** (`make_layer1_configs.py`, ARMS): `melnik`, `coef_prod`, `fd`, `coef_prod_fd`,
`attach_id`, `lit_none`.

**Layer 3** (ARMS_L3), graph construction:
- schema graph, one information family added: `ucc`, `ind`, `fd_ucc_ind`; representation
  `fd_reified` (COMPLETE); strength `fd_damped` (CONSTANT 0.35); `fd` is layer 1's arm;
- column-identity graph (no schema or type hubs; channels computed from the data): `col_fd`
  (projected FDs), `col_gpdep` (FD strength), `col_gpdep_gcov` (+ within-table value inclusion),
  `col_fd_gcov`, `col_fd_gcov_u` (+ uncertainty), `col_fd_jaccard` (+ Jaccard value overlap).
  These encodings stop at a residual of 1e-4 and have no seed attachment or literal seeds; they
  are compared with `melnik` as alternative graphs, not as one-module deviations, and reported as
  a separate block.

**Initial matchers:** every seed of the seed study that finished in a regime: 90 (plain),
86 + 2 later (MD5: `fuzzy_jaccard`, `valentine_distribution` join when finished), ISResMat when its
matrices exist (flagged as added later). Controls (`constant`, `random`, `embedding_random`,
`oracle*`) are reported separately and never enter means. Seeds that do not cover every scenario
(ReMa: not TPC-DI-Semantically-Joinable; SiMa: TPC-DI only; PRISMA, SemStruct: their coverage) are
reported separately with their coverage.

**Data:** Valentine audited corpus, 382 scenarios, plain and MD5 regimes; the independent-sampling
corpus (81 multi-table scenarios, plain), layer 1 FD-free arms only (`melnik`, `coef_prod`,
`attach_id`, `lit_none`) until it has FD/UCC/IND profiles. Nothing is tuned on the independent
corpus.

## 3. Metrics and aggregation

Per scenario and matrix: F1_Hr (Hungarian F1, random tie order, 50 seeded orders, Amendment 2),
R_exp (expected Recall@GT), MRR_exp, F1_M (mutual top-1 = Melnik's filter); reported alongside:
F1_H, R_pess, MRR_pess (index-order or pessimistic; shown to document the tie and diagonal bias,
never as headline). Headline objective J_tf = (ΔR_exp + ΔF1_Hr) / 2 as in Amendment 2.

Contrasts, all paired per scenario:
- **SF effect:** SF(arm) − initial matrix;
- **arm effect:** SF(arm) − SF(melnik).

Aggregation: mean per family of scenarios, then mean over families with at least 5 scenarios
(Valentine: ChEMBL, Magellan, OpenData, TPC-DI; independent: AdventureWorks, Chinook, TPC-H).
Uncertainty: 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples,
seed 20261002. Results are reported per initial matcher, per family of initial matchers
(MATCHERS.md), and as the mean over non-control matchers with full coverage. All arms and all
initial matchers are reported; nothing is selected.

## 4. Hypotheses

**H1 (Proposition 1, deterministic).** On single-table scenarios, every arm whose graph has no
dependency or column-relation edges (`melnik`, `coef_prod`, `attach_id`, `lit_none`) yields an SF
output that is affine in the initial matrix within each datatype-pair block,
S_ij = a·q_ij + b(dt_i, dt_j), a > 0. Test: zero within-block order inversions (tolerance 1e-6) on
every scenario and initial matcher, datatype classes taken from the `datatype` seed's matrix. Any
violation is reported as a defect of the implementation or of the proposition's assumptions.
Arms with dependency edges are expected to show inversions.

**H2 (RQ1).** No directional prediction for the mean SF effect. Predicted by H1: on Valentine
(single tables), the ranking part (ΔR_exp, ΔMRR_exp) of SF as published is small for every
initial matcher (|macro ΔR_exp| < 0.03), because without dependencies SF can only re-weight
datatype blocks; the constant control under `melnik` behaves as a datatype matcher.

**H3 (RQ2).** Information families that distinguish columns within a table (dependency edges,
column-identity channels) change the ranking more than the parameter arms do: the mean |arm
effect on R_exp| over initial matchers is larger for every layer-3 dependency arm and for `fd` than
for each of `coef_prod`, `attach_id` and `lit_none`. Direction of the change is not predicted.

**H4 (RQ4).** The SF effect of `melnik` (J_tf) is larger on the independent multi-table corpus than
on Valentine plain, macro over initial matchers present in both.

**H5 (RQ4, regimes).** Under MD5, label and lexical initial matchers lose most of their accuracy;
no prediction for the SF effect (disclosed: an earlier analysis on 14 MD5 initial matchers under
the production configuration saw SF gain about 0.03 pessimistic Recall@GT on label matchers).

**RQ5 (descriptive).** SF iterations, convergence, graph sizes and time per scenario
(`sf_stats.tsv`; schema-graph arms only) and wall time, peak RSS and CPU time per task (`DONE`), by
arm and corpus. Timings from the shared cluster are under load; a dedicated timing pass gives the
reported absolute numbers.

## 5. What had been seen when this was written

- Layer 1: nothing scored except two `sima_tpcdi` configurations (TPC-DI-Unionable only), used to
  test the scorer. Layer 3: a smoke run of all arms with `levenshtein` on Magellan (only the
  maximum difference to `melnik` was looked at).
- Proposition 1 was checked on 40 scenarios with `levenshtein` (Magellan, TPC-DI-Unionable): zero
  inversions without dependency edges, 27,155 with them. H1 extends this to every initial matcher.
- Independent corpus: one TPC-H scenario with `levenshtein` (seed F1_Hr 0.025, `melnik` 0.317), and
  the 64 Java seeds on that scenario under the production configuration (smoke test).
- Earlier harness pilots (seed calibration, channel budgets, non-backtracking, formula variants;
  `harness/SF_*.md`) are known and motivate H1–H3; they are not re-run here.

## Amendment 1 (2026-10-02, 23:40): how H1's datatype classes are identified

**Written after H1 was run and seen.** H1 as registered took the datatype classes from the `datatype`
seed's matrix (identical rows and columns). Run that way, the FD-free arms showed no inversions on MD5
(0 of 11.9·10⁹ ordered pairs, all 86 initial matchers) but some on plain (`melnik`: 232,057 of 1.4·10⁹,
in 63 of 90 initial matchers; the other FD-free arms alike).

The cause is the proxy, not SF. The `datatype` seed scores type *similarity*, so two different datatypes
whose similarity rows agree on the target's types fall into one class, and pairs from two real blocks
are compared. The seed also has non-finite values in 86 plain scenarios.

**Change:** the classes are read from SF itself, from its output on the `constant` initial matrix under
`melnik`. By Proposition 1 that output is b(dt(i), dt(j)), and the literal seeds separate the datatype
names, so identical rows and columns are exactly the datatype classes of the graph. Scenarios where that
output is not finite are skipped. `check_prop1.py --classes constant`; the registered variant stays
available as `--classes datatype` and is reported next to it.

**Result under the change:** 0 inversions for `melnik`, `coef_prod`, `attach_id` and `lit_none` on every
initial matcher: plain 0 of 1.52·10⁹ ordered pairs (90 matchers, 33,766 scenario runs), MD5 0 of
1.21·10¹⁰ (86 matchers, 32,587 runs). With FD edges: plain 4.8·10⁸ (`fd`) and 1.1·10⁸ (`coef_prod_fd`)
inversions, MD5 4.3·10⁹ and 1.0·10⁹, as expected.

Both variants are reported. The registered one is the weaker test, because its classes can merge
datatypes. The amended one is the exact test of the proposition.

## Disclosure (2026-10-03, 00:45): layer-1 outputs seen by another session before H3 is analysed

Session C (layer 3b, `SYNC.md`) re-ran the registered `melnik` and `fd` arms locally for its R7 smoke
test and so saw `fd` against `melnik` on the 60 screen scenarios for 4 initial matchers (levenshtein, pmf,
prefix_suffix, role_gcov): mean J_tf −0.076. This touches H3. The screen scenarios had already been used
by the tuning. Plain layer 1 had been scored at 23:19 (`layer1_plain_scores`); A had read its report
table, not the per-matcher contrasts H3 is tested on. Nothing in H3's test or its decision rule changes.

## Amendment 2 (2026-10-03, appended by A before queueing): R7, dependency closure and minimal cover (layer 3b)

Approved by the user on 2026-10-03 (about 02:00). Written by session C as `layer3b/AMENDMENT_R7_DRAFT.md`
(SHA-256 ed6f018cd9fd1a64493cc562e7b3ca7c3831e37bada3d25770195144b9978e4b), appended here verbatim, §1–8.

### Draft amendment: layer 3b, R7 (saturated and reduced FD encodings)

**Status: approved by the user on 2026-10-03 (about 02:00), including §7. Draft by C, 2026-10-02 23:48. Binding once A appends it, dated, to `reference/PREREGISTRATION_LAYERS.md` (amended hash 8d555849…).** It becomes binding only when A appends
it, dated, to `reference/PREREGISTRATION_LAYERS.md` (C does not edit that file). Written **before
any output of the R7 smoke test was scored**. What had been seen is listed in §6.

#### 1. Question

SF's dependency edges are the one graph module that changes the ranking (H3; the MD5 layer 1: `fd`
against `melnik` +0.095 F1_Hr). Does it matter *which* FD set is encoded, beyond its size? Falleri
et al. (MODELS 2008) found a saturated encoding best and a minimal one worst on metamodels. The
corpus's FDs are approximate (relaxed HyFD, threshold 0.95, determinants of at most 2 columns), so
they are neither closed under Armstrong's axioms nor free of redundancy, and both directions can be
tested on the same profiles.

#### 2. Arms

Two FD sets, derived per table from G, the set the registered arms put into the graph (FDs whose
determinant has 1 to 3 columns and is not a UCC):
- **closure**: G plus every FD X → A implied by G, with 1 ≤ |X| ≤ 2, X not a UCC, and X minimal for A
  under G (the saturated encoding);
- **cover**: a non-redundant subset of G: an FD is dropped when the remaining ones imply it; larger
  determinants first, then by text (the reduced encoding). No left-reduction, so every FD in the
  cover was found by the profiler.

They enter through a parallel corpus copy (only `FD_results.txt` and its checksum change;
`layer3b/r7_build.py`). Four arms, each differing from a registered arm in the FD set only:

| Arm | Registered base | Encoding |
|---|---|---|
| `fd_closure` | `fd` (layer 1) | QUICK, projected |
| `fd_cover` | `fd` | QUICK, projected |
| `fd_reified_closure` | `fd_reified` (layer 3) | COMPLETE, reified |
| `fd_reified_cover` | `fd_reified` | COMPLETE, reified |

**Control (deterministic):** `fd_same` rewrites the FD file from the parsed G without changing it.
SF on it must equal `fd` cell by cell. It runs on the screen scenarios only, as a check of the
pipeline, and is not reported as an arm.

#### 3. Data, initial matchers, metrics

As in §2 and §3 of `PREREGISTRATION_LAYERS.md`: every seed of the layer-3 set, all 382 Valentine
scenarios, plain and MD5. The independent corpus is out until it has FD profiles. The metrics,
contrasts (arm effect = SF(arm) − SF(base), paired), aggregation (families with ≥ 5 scenarios),
bootstrap (10,000 resamples, stratified, seed 20261002) and reporting rules are unchanged. The
Amendment 3 tie-breaker is computed for plain only; on MD5 not before Amendment 3's primary test.

#### 4. Hypotheses

The structure of the two FD sets is known from the profiles alone (§6), so the hypotheses follow it:

- **H-R7a (closure is nearly inert).** The closure changes the projected graph by about 2% of its
  edges, so under QUICK it changes little: the mean over initial matchers of |arm effect on J_tf| of
  `fd_closure` is below 0.005 (equivalence bound; tested by the 95% bootstrap interval of the macro
  mean of |effect| lying below 0.005).
- **H-R7b (the cover changes more than the closure).** The mean |arm effect on R_exp| over initial
  matchers is larger for `fd_cover` than for `fd_closure`, in each encoding.
- **H-R7c (the cover reorders more inside Proposition 1's blocks).** The dense projected FD graph
  (mean density 0.46, up to 0.97) makes many columns of a table structurally interchangeable (colour
  refinement separates 58% of the columns, against 81% under the cover). Proposition 1's symmetry
  returns in the limit of a complete graph. Prediction: the share of ordered within-block pairs of q
  that SF inverts (blocks from SF(constant) under `melnik`, as in `check_prop1.py --classes
  constant`) is higher under `fd_cover` than under `fd`, macro over initial matchers.
- **No directional prediction** for the effect of the cover on J_tf (Falleri predicts a loss, the
  symmetry argument a gain in tie-breaking). It is reported with its interval.
- **RQ5 (descriptive).** Pairwise-graph size, iterations and time per arm, from `sf_stats.tsv`. The
  reified closure is the expensive arm. If a scenario exceeds the task limits, it is reported as not
  run and excluded pairwise from its arm.

#### 5. What does not change

The layer-1 and layer-3 arms, their hypotheses H1–H5, the tuning protocol and its amendments.
Nothing here is a selection candidate.

#### 6. What had been seen when this was written

- **Structural diagnostics of the FD sets** on the 60 screen scenarios (`layer3b/r7_screen_diagnostics.csv`;
  no SF output):
  - mean FDs per table: G 339, closure 380, cover 50;
  - projected edges 229 / 235 / 61;
  - projected density 0.46 / 0.47 / 0.11;
  - share of columns separated by colour refinement 0.58 / 0.56 / 0.81.
  - Every profile has threshold 0.95 and max arity 2.
- **The R7 smoke test** (4 initial matchers: Levenshtein, prefix/suffix, PMF, gcov role × 8 arms, on
  the 60 screen scenarios, plain) was **running, unscored**, when this was written. The only thing
  read from it was the "Filtered FDs" counts in the logs: they match the builder's FD sets in every
  case checked (the pipeline check).
- The screen scenarios are part of the dev split, and so of the registered runs' data. The smoke test
  is exploratory and is reported as such (EXPLORATORY.md).

##### Addendum (2026-10-02 23:55, still before any smoke-test output was scored)

Full-corpus diagnostics of the FD sets (`r7_build.py --cases all --dry-run` →
`layer3b/r7_all_diagnostics.csv`; 382 scenarios, 764 tables, structure only):

| Family | Tables | FDs per table G / closure / cover | Projected density G / closure / cover | Columns separated by colour refinement G / closure / cover |
|---|---|---|---|---|
| ChEMBL | 172 | 36 / 38 / 18 | 0.11 / 0.11 / 0.06 | 0.47 / 0.46 / 0.60 |
| Magellan | 14 | 10 / 11 / 6 | 0.37 / 0.37 / 0.26 | 0.67 / 0.67 / 0.74 |
| OpenData | 222 | 535 / 637 / 114 | 0.25 / 0.26 / 0.08 | 0.76 / 0.71 / 0.87 |
| TPC-DI | 352 | 355 / 385 / 30 | 0.76 / 0.76 / 0.16 | 0.57 / 0.56 / 0.87 |
| Wikidata | 4 | 27 / 30 / 13 | 0.20 / 0.22 / 0.12 | 0.60 / 0.54 / 0.67 |

Reified pairwise-graph load (FD nodes source × target per scenario): median 51,824 / 63,798 / 986,
maximum 1.54 M / 2.22 M / 39,767. The reified closure is about 1.4 times the registered
`fd_reified` at the maximum.

**H-R7c, family form (added here, before scoring):** the increase in within-block inversions from
`fd` to `fd_cover` is largest on TPC-DI, where the projected FD graph is nearly complete
(density 0.76 → 0.16).

#### 7. Proposed changes after the smoke test (2026-10-03 01:45; disclosed post-hoc)

Everything above §7 was fixed before any smoke-test output was scored (SHA-256 ba8d73bb… in SYNC.md,
00:25). The smoke test is EXPLORATORY.md X8. What it showed, and what C proposes in consequence:

1. **Add a size-matched random null**, `fd_thin` and `fd_reified_thin`: a random subset of G with as
   many FDs as the cover, per table, **three draws** (seeded by SHA-256 of dataset/scenario/side/draw;
   the smoke test used one). In the smoke test the random subset recovered +0.030 of the reified
   cover's +0.049. Without it the registered test cannot tell "which FDs" from "how many". Added
   **after** seeing the smoke results.
   - Contrast for that question, added here: cover − thin (mean over draws), paired, on J_tf,
     R_exp and F1_Hr. No directional prediction.
2. **Keep H-R7a–c unchanged.** H-R7c (the cover reorders more within blocks) failed in the smoke
   test, in every family: the inversion rate fell from 0.30 to 0.16 (QUICK) and from 0.13 to 0.06
   (reified). It stays registered as written, and its result is reported as it comes.
3. **Primary data for the confirmatory reading: the 322 scenarios outside the screen.** The smoke
   test used the 60 screen scenarios. The full 382 are reported as well.
4. **Memory:** the reified arms need more than 4 GB of heap on the largest OpenData-Unionable scenarios.
   Locally, 10 GB was enough.

#### 8. Implementation (2026-10-03 02:20, after approval; no new data seen)

- **Same FD sets in both regimes.** §2's tie-break "by text" and the random draws (§7) use the FD's
  text under the **plain** column names. The MD5 regime's names are salted tokens, and
  `transform-manifest.json` maps them back. Without this, MD5 would get different covers and draws than
  plain, a confound for the regime comparison. On plain this is the rule the smoke test used, so plain
  is unchanged (checked byte for byte). Verified: for every variant and table, the MD5 FD set mapped to
  plain names equals the plain set (`layer3b/r7_verify_bundles.py`).
- **Arms, as run:** `fd_closure`, `fd_cover`, `fd_thin1..3` (base `fd`, layer 1) and
  `fd_reified_closure`, `fd_reified_cover`, `fd_reified_thin1..3` (base `fd_reified`, layer 3), for every
  initial matcher with both base configurations: 90 on plain, 87 on MD5. Each configuration is a copy of
  its base with only the corpus path changed (`layer3b/make_r7_cluster_configs.py` asserts this).
- **Pipeline check:** `levenshtein__fd_same` in each regime must equal layer 1's `levenshtein__fd` cell
  for cell. If it does not, nothing of R7 is scored. In a local end-to-end test (bundle → apply script →
  cluster configuration), `fd_same` and `fd_reified_cover` reproduced the smoke test exactly on
  60 scenarios.
- **The registered scorer** is written by C after collection, before any R7 result is read. Contrasts as
  in §3 and §7, metrics as in `PREREGISTRATION_LAYERS.md` §3. The tie-breaker is computed on plain only
  (§3).

## Amendment 3 (2026-10-03, 14:10, appended by A before queueing): R8, restricted propagation scope (layer 3b)

Approved by the user on 2026-10-03 (about 11:05). Written by session C as `layer3b/AMENDMENT_R8_DRAFT.md` (SHA-256 1b54bd7d782f6d2147c847dca9a702b5cd53442731e3d57393d4a92218632520), appended here verbatim, all sections; headings demoted by two levels.

### Draft amendment: layer 3b, R8 (restricted propagation scope)

**Status: approved by the user on 2026-10-03 (about 11:05), including §7 and §8. Draft by C, 2026-10-03 03:02, written before any output of the R8
exploratory grid was scored. Binding once A appends it, dated, to `reference/PREREGISTRATION_LAYERS.md`.** What had been seen is listed in §6.

#### 1. Question

Melnik's pairwise connectivity graph pairs every two nodes joined by edges with the same label, so SF
propagates over all column pairs. Lily (Wang & Xu 2009) propagates only inside small subgraphs, and
blocking is standard in matching. Does restricting propagation to the candidate pairs an initial matcher
proposes change SF's quality, and what does it save (RQ3, RQ5)?

#### 2. Arms

`-Dsmx.sf.pairScope` (Schematch; off by default) removes the pruned column pairs, with their name
pairs, from the pairwise connectivity graph before the propagation graph is induced. Coefficients stay
those of the two schema graphs. Pruned pairs output 0.
- `QUANTILE p`: keep the pairs with q at or above the p-quantile of all column pairs of the scenario,
  p ∈ {0.5, 0.75, 0.9};
- `TOPK k`: keep a pair if it is among the k highest of its row or its column, k ∈ {1, 3}.

Each scope is crossed with the reference `melnik` (no dependency edges) and with `fd` (layer 1).
**Control (scoring only):** q_s, the initial matrix with the same cells pruned and no propagation. It
separates what propagation adds from what blocking alone does.

#### 3. Data, matchers, metrics

As in Amendment 2 §3: every initial matcher of layer 1 with full coverage, 382 scenarios, plain and MD5.
The metrics, aggregation and bootstrap are those of `PREREGISTRATION_LAYERS.md` §3. Primary data: the
322 scenarios outside the screen; the exploratory grid used the screen.

#### 4. Hypotheses

- **H-R8a (deterministic; Proposition 1).** Under `melnik` with any scope, SF inverts no
  within-block pair among the kept cells (blocks of SF(constant) under `melnik`; tolerance 1e-6).
- **H-R8b (Proposition 1, consequence).** Under `melnik`, the scope cannot reorder the kept cells
  within a block, so what propagation adds on the candidate set is close to what it adds without the
  scope: |(SF_s − q_s) − (SF − q)| on R_exp is below 0.01, macro mean over matchers, for every scope.
  (Prediction for the ranking part only; the assignment may differ through the block intercepts.)
- **H-R8c (dependency edges).** Under `fd`, the scope changes the order within blocks: the inversion
  share among the kept cells is below `fd`'s unrestricted share for QUANTILE 0.9 and TOPK 1 (fewer
  dependency paths), and the scope's effect on SF (SF_s − SF on J_tf) is larger in absolute value under
  `fd` than under `melnik` for every scope.
- **RQ5 (descriptive, with a direction):** pairwise-graph size and iteration time fall monotonically
  from no scope to QUANTILE 0.5, 0.75, 0.9, and from TOPK 3 to TOPK 1.
- **No direction** for the scope's effect on J_tf.

#### 5. What does not change

Layers 1 and 3, Amendment 2 (R7), the tuning protocol. Nothing here is a selection candidate.

#### 6. What had been seen when this was written

- The R8 implementation checks (`layer3b/r8_smoke.py check`; `levenshtein`, 60 screen scenarios):
  the default output is unchanged; `melnik` with QUANTILE 0.5 inverts 0 of 1,000,084 within-block
  pairs among the kept cells (H-R8a seen for one matcher and one scope); `fd` with TOPK 1 ran, and only
  its pruning was checked (zero outputs and kept counts), not its scores.
- The R7 smoke test's `melnik` and `fd` outputs on the screen (EXPLORATORY X8).
- The R8 exploratory grid (4 matchers × {melnik, fd} × 5 scopes, screen, plain) was **running, unscored**.

#### 7. After the exploratory grid (2026-10-03 03:40; disclosed post-hoc)

Everything above §7 was fixed before the grid was scored (SHA-256 ca57bc4a…, SYNC 03:04). The grid is
EXPLORATORY.md X11. How it went against the hypotheses: H-R8a held (0 inversions in 20 runs); H-R8b
held for the quantile scopes and failed for TOPK (+0.032 and +0.011 against the 0.01 bound); H-R8c and
the RQ5 direction held. C proposes:
1. **Keep H-R8a–c as written**, including H-R8b, which is now expected to fail for TOPK.
2. **RQ5 reports iteration time and pair-graph size, not wall time.** Pruning happens after the full
   graph is built, so the wall time does not fall. A construction-time filter would be a further Java
   change and is not part of this amendment.
3. **Registration needs a cluster build** of `SimilarityFlooding.java` with `smx.sf.pairScope`
   (installed locally at 03:00; source 3351d7bd…). Cost from the grid, scaled to 382 scenarios: about
   3.5 CPU-hours per initial matcher and regime for the 5 scopes × 2 arms, so about 300 CPU-hours per
   regime.

#### 8. Implementation (2026-10-03 11:10, after approval)

- **Build:** the cluster needs `SimilarityFlooding.java` with `smx.sf.pairScope` (source SHA-256
  3351d7bddc2812e47f32f5df896dcaeba94995149e72bf4e15beda392a12bf79; locally compiled class 8ff306c1…).
  The default path is unchanged (checked cell for cell locally, SYNC 03:02).
- **Configurations** (`layer3b/make_r8_cluster_configs.py`): for every initial matcher with a layer-1
  configuration (90 plain, 87 MD5), `<seed>__melnik_<scope>` and `<seed>__fd_<scope>` with scope in q050,
  q075, q090 (QUANTILE 0.5, 0.75, 0.9) and top1, top3 (TOPK 1, 3). Each is a copy of its layer-1 base; only
  `resultsDir` and the two JVM properties change (asserted). 901 plain, 871 MD5, including the build check.
- **Checks, in the scorer, before anything is scored:**
  1. `levenshtein__fd_scopenone` (`pairScope=NONE` set explicitly on the cluster build) equals layer 1's
     `levenshtein__fd` cell for cell;
  2. every cell the scope prunes is exactly 0 in every scoped output.
- **Scorer** `layer3b/score_r8.py` (SHA-256 0998bec6…), frozen before any registered R8 result.
  Operational choices, fixed here:
  - H-R8b is confirmed if the 95% interval of the mean difference lies inside (−0.01, +0.01).
  - H-R8c(1) uses the per-scenario inversion share, macro-averaged (as in the R7 scorer). Disclosed:
    on the screen it gives q090 +0.032 [−0.004, +0.065] and top1 −0.018 [−0.033, −0.002], while the
    exploratory grid's pooled shares were lower for both (0.26 and 0.25 against 0.30). Both versions were
    seen before this choice was written down. The per-scenario version is kept because it is the R7
    scorer's definition, and it is fixed before any registered result.
  - H-R8c(2): the lower bound of mean |SF_s − SF| on J_tf, `fd` minus `melnik`, above 0.
  - RQ5 "held" if graph size and iteration time do not rise along none → q050 → q075 → q090 and
    none → top3 → top1.
- **Primary data:** the 322 scenarios outside the screen; all 382 and the screen are reported as well.
  No tie-breaker on MD5.
- **Heap:** 6 GB was enough for every local run (the scope only shrinks the graph).

## Amendment 4 (2026-10-03, 14:10, appended by A before queueing): R8b, propagation restricted to mutually best candidates (layer 3b)

Requested by the user for registration on 2026-10-03 (about 13:10). Written by session C as `layer3b/AMENDMENT_R8B_DRAFT.md` (SHA-256 f5df80fa340aff3e94dcb8fca648d2fcc40e28b96af434c40c0e0eec4285e587), appended here verbatim, all sections; headings demoted by two levels.

### Draft amendment: layer 3b, R8b (propagation restricted to mutually best candidates)

**Status: requested by the user for registration on 2026-10-03, about 13:10 ("draft the amendment for
mutual top-1 and hand it to A"). Drafted by C, 13:15 to 13:41. Binding once A appends it, dated, to
`reference/PREREGISTRATION_LAYERS.md` (as its Amendment 4).**

#### 1. Question

R8 (registered as Amendment 3) restricts SF's pairwise graph by QUANTILE or TOPK. In an exploratory follow-up
on the screen (EXPLORATORY.md X17), **mutual top-1** was the best of 9 selection modes: keep a column pair only
if it is the best of its row and of its column in the initial matrix q (Melnik's SelectThreshold at
t_rel = 1, applied before propagation instead of after). It brought `fd` from −0.042 to +0.065 J_tf
against q, and `melnik` from +0.034 to +0.063, with about half of the gain from the blocking alone. This
amendment tests whether that holds on data the choice did not see.

#### 2. Arms

`-Dsmx.sf.pairScope=MUTUALTOPK` with `smx.sf.pairScopeParam=1` (R8b build; §8). The pruned column pairs and
their name pairs are removed from the pairwise connectivity graph before propagation; pruned pairs output 0.
Ties at a row's or column's best value are kept.
- `melnik_mutual1`: layer 1's `melnik` with the scope;
- `fd_mutual1`: layer 1's `fd` with the scope.

**Control (scoring only):** q_s, q with the same cells pruned and no propagation.

#### 3. Data, matchers, metrics

Every initial matcher with a layer-1 configuration (90 plain, 87 MD5), 382 scenarios, plain and MD5. Metrics,
aggregation (families with ≥ 5 scenarios), matcher inclusion (non-controls with full coverage) and bootstrap
(10,000 stratified resamples, seed 20261002) as in `PREREGISTRATION_LAYERS.md` §3. **Primary data: the 322
scenarios outside the screen**, which X17 did not use; the full 382 and the screen are reported as well. No
tie-breaker on MD5 is needed for any hypothesis; it is reported on plain only.

#### 4. Hypotheses (directional, from X17; on J_tf unless stated)

- **H-R8b-a (Proposition 1, deterministic).** Under `melnik_mutual1`, SF inverts no within-block pair among
  the kept cells (blocks of SF(constant) under `melnik`; tolerance 1e-6).
- **H-R8b-1 (the FD edges' loss is repaired).** `fd_mutual1` − `fd` > 0.
- **H-R8b-2 (better than the initial matcher).** `fd_mutual1` − q > 0, and `melnik_mutual1` − q > 0.
- **H-R8b-3 (propagation adds on the candidate set).** SF_s − q_s > 0 under both `melnik_mutual1` and
  `fd_mutual1`.
- **H-R8b-4 (better than SF as published).** `melnik_mutual1` − `melnik` > 0.

Each is confirmed if the lower bound of the 95% interval of the mean over included matchers is above 0
(H-R8b-a: zero inversions). Each hypothesis is tested separately in each regime; the plain and MD5 verdicts
are reported side by side. No correction for multiplicity is applied; there are seven directional tests per
regime, which is disclosed with the results.

**Reported, no hypothesis:** `fd_mutual1` − `melnik_mutual1`; the registered R8 TOPK 1 arm against mutual
top-1 (descriptive, same scenarios); R_exp, F1_Hr, MRR_exp and F1_M for every contrast; pair-graph size and
iteration time (RQ5).

#### 5. What does not change

Layers 1 and 3, Amendments 2 (R7) and 3 (R8). Nothing here is a selection candidate.

#### 6. What had been seen when this was written

- **EXPLORATORY.md X11 and X17** (plain, 60 screen scenarios, 4 matchers: levenshtein, prefix_suffix, pmf,
  role_gcov): every number in §1, for mutual top-1 and the 8 other modes. **Mutual top-1 was chosen
  because it was best there.** The hypotheses follow those results, so the screen scenarios are not
  independent. That is why the primary data excludes them.
- Layer 1 (plain and MD5) and layer 3 plain are scored. C has seen layer 1's `melnik` and `fd` only on the
  screen, through the smoke tests (disclosed in `PREREGISTRATION_LAYERS.md`, 2026-10-03 00:45).
- Nothing of MD5 under any pair scope.

#### 7. Build

The R8b build changes only `SimilarityFlooding.java` (C's copy: source SHA-256
8f0d1bed5fcb5882ff4435e3fca03da686fe63c031f30e578ea4bbce8cef12fe). It adds the modes ROWTOPK, COLTOPK,
MUTUALTOPK, GLOBALTOPK and ROWREL to R8's `smx.sf.pairScope`. NONE, QUANTILE and TOPK keep their code path.
The release classes are verified against earlier runs, cell for cell (`layer3b/r8b_release_check.py`):
the default path, QUANTILE 0.9 and TOPK 1 against the shipped R8 class, and MUTUALTOPK 1 against the dev
build.

#### 8. Implementation (2026-10-03 13:41)

- **Release classes to ship** (`layer3b/classes_r8b_release/`; only `SimilarityFlooding*.class`, compiled
  from the source in §7 against the shared classes, without C's family-6 builder copy):
  - `SimilarityFlooding.class` 1f38fcf2a73d16d1aa42b3fda63c6ed2880b435422786ba9b397a9dff687df9f
  - `SimilarityFlooding$PairScope.class` 0ebcf04d95b2790b06db3f549145cf3c87651fb84b8d9ddaf83d627f37552ff1
  - `$Param` b8482e05…, `$ChannelBudget` ce5210b1…, `$Normalise` ed9f9b50…, `$SeedAttachment` 9082902d…
- **Release check** (`layer3b/r8b_release_check.py`, 60 screen scenarios, levenshtein; max |difference| 0.0
  in all five): no scope = the shared build's `melnik`; QUANTILE 0.9 and TOPK 1 = the shipped R8 class;
  MUTUALTOPK 1 = the dev build, under `melnik` and `fd`.
- **Configurations** (`layer3b/make_r8b_cluster_configs.py`; copies of layer 1 with only `resultsDir` and the
  scope properties changed, asserted): `r8b_plain` 182, `r8b_md5` 176. They include two checks per regime:
  - `levenshtein__fd_scopenone`, the gate: it must equal layer 1's `levenshtein__fd`;
  - `levenshtein__fd_top1check`: it must equal R8's registered `levenshtein__fd_top1`.
- **Order:** after R8. The R8b classes would also run R8's arms identically (checked above), but R8 runs on
  the class confirmed for it (8ff306c1…). Ship the R8b classes once R8's tasks are finished.
- **Scorer** `layer3b/score_r8b.py`, SHA-256 b35e4604141f4f0a26c7ee4a7f248e7a72b23a17196195d922a0bd4ee0a02a01,
  frozen before any R8b result. It uses `score_r7.py` (761f2c14…), `score_r8.py` (0998bec6…) and `r8b.py`
  (c3b0d9ea…, the rule mirror), all frozen with it. It checks the gate, the TOPK 1 check (once R8's output
  exists) and that every pruned cell is 0, and scores nothing if one fails.
  - Tested on the exploratory runs: it reproduces X17. Negative test: with the gate pointed at another run
    it refuses and writes nothing.
- **Cost:** about 40 CPU-minutes per initial matcher and regime (two QUICK arms), so about 65 CPU-hours per
  regime. 6 GB heap is enough.

## Amendment 5 (2026-10-03, 17:40, appended by A before `layer1_indep` was collected): additive datatype-and-class term against SF as published, independent corpus

Approved by the user on 2026-10-03 (B, SYNC 17:10). Written by session B as `reference/AMENDMENT_ADDITIVE_DRAFT.md` (SHA-256 43b6495278c79a4ad6c1ed3ab3f1607625ca2e64dc0ac162b7494ae5cf2731f3), appended here verbatim; its headings are already at sub-section level. At appending, `layer1_indep` had not started on the cluster (996 tasks queued, 0 run), and every code hash in its §4 was checked and matches.

### Draft amendment (Amendment 5): an additive datatype-and-class term against SF as published, on the independent corpus



Written by B on 2026-10-03, at the user's request ("register 1"). It must be appended **before**

`layer1_indep` is collected or scored.



#### 1. What has been seen



- **Exploratory results on Valentine, plain, which motivate the hypotheses** (EXPLORATORY.md X19, X25, X26):

  - Proposition 1 makes SF without dependency edges a·q + b(datatype block). The residual check holds it

    to ≤ 1.2e-5 in all 21,450 dev fits.

  - An additive term q + β·D, with D the datatype matcher, beats layer 1's `melnik` on the 226 non-screen

    dev scenarios: +0.032 against +0.016 J_tf over q, ahead for 62 of 75 matchers.

  - With a "same class" term, β and γ chosen on the screen, it reaches +0.072 (pattern classes, ahead for 69

    of 75) and +0.069 (rule-based semantic types, ahead for 70 of 75).

  - On MD5 there are no datatype blocks (median 1 per scenario); SF gains +0.009, q + β·D +0.010.

- **The weights below are those chosen on the Valentine screen in X26.** Nothing here is chosen on the

  independent corpus.

- **On the independent corpus (81 multi-table scenarios: AdventureWorks, Chinook, TPC-H; plain):**

  - No layer-1 output exists. `tuning/results/layer1_indep` is absent; the 996 tasks are queued behind R7.

  - No metric of q + β·D or of the class terms has been computed.

  - `score_additive_indep.py --check-inputs` only loads the matrices and class labels and checks that they

    align; it computes no metric and does not read SF output.

  - The seed study on this corpus (`results/indep/scores.csv`, standard configuration, scored 02:45 by A)

    exists. B has not read it. It reports the initial matchers' quality and the standard configuration's

    effect, not the `melnik` arm or any additive term.

- **The datatype matcher used is the NaN-free one of X30.** 27 of the 81 scenarios had NaN cells before; the

  fix sets exactly those cells to 0 and leaves every finite cell unchanged.



#### 2. Hypotheses



On the contrast A − SF(`melnik`) in J_tf = (R_exp + F1_Hr)/2, per scenario and initial matcher:



- **HB1 (primary):** A_p = q + 5·D + 0.2·[same pattern class] beats SF: A_p − SF > 0.

- **HB2:** A_s = q + 5·D + 0.1·[same rule-based semantic type] beats SF: A_s − SF > 0.

- **HB3:** A_d = q + 5·D beats SF: A_d − SF > 0.



Pattern classes are `layer3b/f7.py` `column_classes` (the dominant value shape, ≥ 50%, else MIXED).

Semantic types are `layer3b/f13.py` `column_types` (≥ 60%, else mixed). Both are computed per table on

the scenario's CSVs and keyed by `table.column`.



**Why this is a test, not a repetition.** These scenarios are multi-table. Proposition 1 then gains a

table-pair term, and SF can reorder across tables. The additive term has no table-pair term, so SF may win

here although it loses on single tables.



#### 3. Statistic and decision



- **Statistic:** per scenario, the mean over the included initial matchers; then the mean within each

  database; then the mean over the three databases.

- **Interval:** 95% percentile bootstrap over scenarios within database, 10,000 resamples, seed 20261003.

- **Decision:** a hypothesis is **confirmed** if the lower bound is > 0 **and** the estimate is > 0 in each

  of the three databases. The corpus has three independent sources (ER4), so the scenario bootstrap

  alone would overstate the evidence.

- **Also reported, not part of the decision:** the matchers with a positive contrast; the R_exp and F1_Hr

  parts; every term against q.



**Included initial matchers:** RQ1's non-control set (the paper's points) with finite q and D on all 81 scenarios: 74 matchers, frozen in

`reference/additive_indep_matchers.json`. Of these, the ones with an SF output on all 81 scenarios are used;

the others are listed and not used.



#### 4. Code (hashes recorded at appending)



| file | SHA-256 |

|---|---|

| `reference/score_additive_indep.py` (scorer; refuses to run before this amendment is in the file) | d815d580a6ee0392270369fab8e54f21cdfe32b4aee7ee8550f2feadb06ab2c4 |

| `run_datatype_nanfix.py` (produces `reference/seeds/indep/datatype_nanfix`) | 1df5ac92221c085bc873a8d930ae3067ef9fa7742e1b8acc4466d7eb75567886 |

| `devjava_b/de/uni_marburg/schematch/matching/metadata/DatatypeMatcher.java` (the fix, `-Dsmx.datatype.nanAsZero`) | 010b597845eade3bc3efcb85af5a89c7278302476d91bb5fffb67edfafe9e688 |

| `layer3b/f7.py` / `layer3b/f13.py` (class labels; imported, not changed) | 692e554fce246959407f18efa7c7bc00260f7e96e25dc3e0e0d6a44e295d72fc / 34882a0086e0f5f818b303eb037b7cd03d1c09a786f21a07e9255044631ab3c0 |

| `reference/additive_indep_matchers.json` (the 74 matchers, frozen) | c83fd152616bfba11f3ebd17b004f797bff0bd56fe9f8fb7b1fe76f0913daf82 |

| `tuning/score_screen_metrics.py` (metrics) | 8357ad2f149ce100c943177e2026e7f7607e46d2509fcfc205e761939473a01c |



**Run after `layer1_indep` is collected:**



    ../harness/.venv-bdi/Scripts/python.exe reference/score_additive_indep.py



Output: `tuning/results/layer1_indep_scores/additive_report.md`.

## Amendment 6 (2026-10-04, ~12:10, appended by A before queueing): T1, table-membership ablation (layer 3b)

Approved for registration by the user (~11:50, "register T1 and hand it to A", relayed by C). Written by session C as `layer3b/AMENDMENT_T1_DRAFT.md` (SHA-256 2470f1de381cb4fa13231beb85d2193f44d06170bcde05423b04056c84844a6b), appended verbatim, all sections; headings demoted by two levels. C's exploratory screen (4 matchers, all 81 scenarios) was seen before §6 was written; the draft's §4–5 disclose it. (B's E2 draft, not yet appended, calls itself Amendment 6; it becomes Amendment 7.)

Frozen code at appending:

| file | SHA-256 |
|---|---|
| `layer3b/score_t1.py` | 19bf84b881807ac8d27887c89e577e4cd255600e2624fdb5963e540cbf8e9977 |
| `layer3b/make_t1_cluster_configs.py` | 1d000f506c9ad39e4b6baeb58fd13a864d19ce46fa24aff62da9ee72106f64a7 |
| `reference/score_additive_indep.py` | d815d580a6ee0392270369fab8e54f21cdfe32b4aee7ee8550f2feadb06ab2c4 |
| `layer3b/score_r7.py` | 761f2c146ece3a99447234565228db821e3d81871a55925fce82725235945930 |
| `layer3b/r8_score.py` | 9c22852bc887592bd9fe5bc74fba49ecc39002e0193a2fc31f9fe5795c0d8860 |
| `tuning/score_screen_metrics.py` | 8357ad2f149ce100c943177e2026e7f7607e46d2509fcfc205e761939473a01c |
| `score_seeds.py` | 76c6141dc9e0f5df8e5759577e8af1cec6db607454df0a40a50bcae8513cd25d |
| `layer3b/classes_tm/…/SchemaGraphBuilder.class` (overlay_t1) | 5c6c6c1a9ca268816fe88ab1a51ca5fe0d4d79ff81396e000a6cb631899ea425 |

### Draft: T1, table-membership ablation (layer 3b; the fifth external review, priority 1)

**Status: approved for registration by the user on 2026-10-04 (about 11:50, "register T1 and hand it to A"); §6 is the registered analysis. Draft by C, 2026-10-04 11:34, written before any T1 output existed. Exploratory screen first (4 matchers,
all 81 scenarios of the independent corpus); a registered full-panel version needs the user's go and A.**

#### 1. Question

Does correct table organisation carry matching evidence for SF, or does "propagation helps more on multi-table
schemas" only reflect a different corpus? By Proposition 1 the table pair is the only structural channel of the
dependency-free schema graph: on a single table its term is constant (SF = a·q + b(datatype pair)); on several
tables it becomes b(table pair, datatype pair).

#### 2. Arms

On the independent corpus (81 multi-table scenarios; AdventureWorks, Chinook, TPC-H), with q, ground truth,
datatypes, class nodes and the seed attachment fixed, `-Dsmx.sf.tableMembership` (C's `classes_tm` build) is one of:
- ORIGINAL: the real table membership (= layer1_indep / ER10);
- COLLAPSED: one table node per schema, all columns under it;
- PERMUTED: the original table nodes, columns reassigned at random with every table's size kept (seeded per
  scenario and side).

Each crossed with `melnik`, `melnik` + pattern classes (ER10 `shape`), `melnik` + semantic types (ER10 `semtype`).
Baselines without propagation, as in Amendment 5: A_d = q + 5·D, A_p = A_d + 0.2·eq_p, A_s = A_d + 0.1·eq_s.

#### 3. Hypotheses (J_tf; macro over the three databases; on the exploratory screen descriptive only)

- **T1a (deterministic, Proposition 1).** Under COLLAPSED, `melnik` inverts no within-block pair (blocks from
  SF(constant) under `melnik`, COLLAPSED).
- **T1b (table organisation carries information).** SF(ORIGINAL) − SF(COLLAPSED) > 0 under `melnik`.
- **T1c (it is the correct organisation that matters).** SF(ORIGINAL) − SF(PERMUTED) > 0 under `melnik`.
- **T1d (the propagation advantage over additive needs the tables).** ER10 found SF + classes ahead of A_p / A_s on
  this corpus; X24 found the reverse on single tables. Prediction: under COLLAPSED, A_p − SF+P ≥ 0 and
  A_s − SF+S ≥ 0 (the advantage disappears); under ORIGINAL the ER10 sign reappears.
- No direction for PERMUTED against COLLAPSED.

#### 4. What had been seen

ER10 (SF + classes vs A_p / A_s, ORIGINAL), Amendment 5 (A_p / A_s / A_d vs SF), X24 (single-table screen), the
review. Nothing under COLLAPSED or PERMUTED.

#### 5. After the exploratory screen (2026-10-04 11:46; disclosed)

Everything above was fixed before any T1 output (SHA-256 835281a2…, SYNC 11:35). Screen result (EXPLORATORY X39):
T1a–T1d all held. SF(ORIGINAL) − SF(COLLAPSED) +0.049 J and − SF(PERMUTED) +0.060 under `melnik`, positive in 3 of 3
databases and 4 of 4 matchers. The additive baselines beat SF under COLLAPSED and PERMUTED, and lose to it under
ORIGINAL. With classes, the ORIGINAL advantage over A_p / A_s comes mostly from TPC-H.

Proposed for a registered full-panel version (the user decides):
- every matcher of Amendment 5's frozen set (`reference/additive_indep_matchers.json`) × {melnik, shape, semtype} ×
  {COLLAPSED, PERMUTED}; ORIGINAL comes from layer1_indep and ER10;
- **three PERMUTED draws.** That needs a seed-salt property in `classes_tm`, a small change to make before
  shipping, after which the screen gates are re-run;
- hypotheses T1a–T1d as written, with Amendment 5's statistic (database macro, scenario bootstrap within database,
  a positive estimate in each database required); primary T1b and T1d (`melnik`);
- the 4 screen matchers reported but not excluded (disclosed: this screen used them on all 81 scenarios; there is no
  untouched subset of this corpus).

#### 6. Registered hypotheses and analysis (2026-10-04 11:57, after the user's approval)

**Data.** The independent corpus, all 81 scenarios (AdventureWorks, Chinook, TPC-H, 27 each), plain. The initial
matchers are Amendment 5's frozen set (`reference/additive_indep_matchers.json`, 74), of which those with finite q and
D (as Amendment 5) and an output in every arm of a contrast on all 81 scenarios are used.

**Arms** (`layer3b/make_t1_cluster_configs.py`; C's `classes_tm` as overlay `overlay_t1`):
- `melnik`, `shape`, `semtype` × COLLAPSED and PERMUTED, with three draws d0, d1, d2 (`smx.sf.tableMembershipDraw`;
  d0 is the screen's permutation);
- ORIGINAL comes from layer1_indep (`melnik`) and ER10 (`shape`, `semtype`);
- PERMUTED is the mean of the three draws per scenario.

**Statistic** (as Amendment 5): J_tf = (R_exp + F1_Hr)/2 on q and SF clipped at 0. Per scenario, the mean over the
included matchers of the contrast; the mean within each database; the mean over the three databases. 95% percentile
bootstrap resampling scenarios within each database, 10,000 resamples, seed 20261003. **A hypothesis is confirmed if
the lower bound is above 0 and the estimate is positive in each of the three databases.**

**Hypotheses:**
- **T1a (deterministic):** `melnik` under COLLAPSED inverts no within-block pair (blocks from SF(constant) under
  `melnik`, COLLAPSED; tolerance 1e-6), for every included matcher and scenario.
- **T1b (primary):** SF_melnik(ORIGINAL) − SF_melnik(COLLAPSED) > 0.
- **T1d (primary):** A_d − SF_melnik(COLLAPSED) > 0. With the original tables, Amendment 5 found the opposite sign
  (HB3: A_d − SF −0.031), so T1d tests the sign flip.
- **T1c (secondary):** SF_melnik(ORIGINAL) − SF_melnik(PERMUTED) > 0.
- **T1e (secondary):** A_p − SF_shape(COLLAPSED) > 0 and A_s − SF_semtype(COLLAPSED) > 0. ER10 found the opposite
  sign with the original tables (−0.010 / −0.016).
- **T1f (secondary):** SF_shape(ORIGINAL) − SF_shape(COLLAPSED) > 0 and the same for `semtype`.
- **Reported, no hypothesis:** PERMUTED − COLLAPSED; every contrast's R_exp and F1_Hr parts; per database; the number
  of matchers with a positive contrast; the spread over the three PERMUTED draws.

**Checks before scoring** (nothing is scored if one fails): the gates `levenshtein__melnik_original` = layer1_indep
`levenshtein__melnik` and `levenshtein__shape_original` = ER10 `levenshtein__melnik_shape`, cell for cell; and
`levenshtein__melnik_permuted_d0` = the screen's `levenshtein__melnik__permuted` (`layer3b/results/t1_indep`).

**Disclosed, what was known when this was written:**
- Amendment 5's ORIGINAL contrasts on the full panel (HB1–HB3);
- ER10;
- the screen of §5, which used 4 of the matchers (levenshtein, prefix_suffix, pmf, role_gcov) on all 81 scenarios and
  one PERMUTED draw (d0). They are not excluded; the corpus has no untouched subset. Results are also reported without
  them.

**Build** (`layer3b/t1_release_check.py`, levenshtein, 81 scenarios; all exact or as required):
- ORIGINAL = layer1_indep and ER10;
- COLLAPSED and PERMUTED d0 = the screen;
- d1 ≠ d0.

**Scorer** `layer3b/score_t1.py`, SHA-256 19bf84b881807ac8d27887c89e577e4cd255600e2624fdb5963e540cbf8e9977, frozen
before any registered T1 output. It imports the frozen `reference/score_additive_indep.py` (Amendment 5),
`layer3b/score_r7.py` (761f2c14…) and `layer3b/r8_score.py`. It requires three PERMUTED draws.
- Tested on the screen (`layer3b/make_t1_scorer_test.py`, one draw allowed): it reproduces §5.
- With a missing gate output it refuses and writes nothing.

**Erratum to Amendment 6 (A, 2026-10-04 12:02):** its heading says "~12:10"; it was appended at 12:00. Nothing else changes; it was appended before any T1 configuration was queued.

## Amendment 8 (2026-10-04, 15:27, appended by A before any classfull output was read): class nodes on the full panel (families 7 and 13)

The user chose "register, then score" (~14:35) and approved this draft after seeing it (to A, directly). Written by session C as `layer3b/AMENDMENT_CLASSFULL_DRAFT.md` (SHA-256 8f5fabc39e2cc379725a2f42cbadb131fc4b1143abcffaac2fe8fb203b8670d8), appended verbatim, headings demoted by two levels. (Amendment 7 is reserved for B's E2, appended when its configuration generator exists.)

Frozen at appending:

| file | SHA-256 |
|---|---|
| `layer3b/score_classfull.py` | 2f0157474d02edcda6025be8ecbea8285ce3db319e386cf5f1feb85a9c7fb8e4 |
| `layer3b/score_r7.py` | 761f2c146ece3a99447234565228db821e3d81871a55925fce82725235945930 |
| `servers/make_classfull_configs.py` | 908a198842c013055f29308638e5b860233030eb1ba618baa991faef61e9c008 |

### Draft amendment (Amendment 8 of PREREGISTRATION_LAYERS.md): class nodes on the full panel (families 7 and 13)

**Status: draft by C, 2026-10-04 14:43. The user chose "register, then score" (A, SYNC 14:37). Written before any
`classfull_plain` or `classfull_md5` output was read; nobody has read them. Binding once A appends it, after the user
has seen it.**

#### 1. Question

Do pattern-class and semantic-type nodes help SF as published on all of Valentine (single-table scenarios), beyond
what a "same class" bonus without propagation gives? On the 4-matcher screen, class nodes raised SF by +0.048 / +0.064
J_tf (X22, X23), but late fusion matched or beat them (X24). On the multi-table corpus, propagation beat the additive
terms (ER10), and T1 traced that to the table structure. Extended Proposition 1 predicts that on single tables class
nodes act as an additive bonus per (datatype, class) block.

#### 2. Arms and data (already run; `servers/make_classfull_configs.py`)

- **Initial matchers:** every layer-1 matcher.
- **Arms:** `melnik_shape`, `melnik_shape_shuffled`, `melnik_semtype`, `melnik_semtype_shuffled` (C's dev build
  as overlay `overlay_er10`).
  - Classes per column: `layer3b/f7.column_classes` (dominant value shape, else MIXED) and `layer3b/f13.column_types`
    (rule-based semantic types, a proxy).
  - Shuffled: permuted within each table.
  - Edge files: `data/classfull_<regime>/`.
- **Reference:** layer 1's `melnik`. The datatype blocks come from layer 1's `constant__melnik`.
- **Corpus:** all 382 Valentine scenarios, plain and MD5.
- **Primary data:** the 322 scenarios outside the screen (X22–X24 used the screen); all 382 are reported as well.

**Late fusion, computed in the scorer with the weights fixed here:**
- L_p = q + 0.2 · eq_p and L_s = q + 0.2 · eq_s (X24's form). eq = 1 where the source and target column have the same
  class in the edge files; MIXED = MIXED counts as equal.
- B_p = SF + 0.2 · max(SF) · eq_p and B_s likewise (the bonus added after propagation, X24's form (b)).
- Reported only: Amendment 5's / X26's terms A_p = q⁺ + 5 D + 0.2 eq_p, A_s = q⁺ + 5 D + 0.1 eq_s
  (D = `reference/seeds/<regime>/datatype_nanfix`, q⁺ = q clipped at 0).

#### 3. Statistic (as `PREREGISTRATION_LAYERS.md` §3 and the R7–R8b scorers)

J_tf = (R_exp + F1_Hr)/2 (`score_screen_metrics.all_metrics`, 50 seeded tie orders).
- **Per matcher:** the macro over the families with at least 5 scenarios in the case set.
- **Then:** the mean over the included matchers, i.e. non-controls with an output in every arm of the contrast on
  every scenario of the case set.
- **Uncertainty:** 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples, seed 20261002.
- **Confirmed** if the lower bound is above 0 (equivalence: the interval inside (−0.01, +0.01)).
- **Reported:** the R_exp and F1_Hr parts, and the number of matchers with a positive macro effect.

#### 4. Checks before any number is computed (the scorer refuses otherwise)

1. **Gate per regime:** `levenshtein__melnik_devnone` equals layer 1's `levenshtein__melnik` cell for cell.
2. **The class edges reached SF.** The cluster's collection drops run.log, so the "Extra edges from …" lines are not
   available locally. Instead, for every class configuration and scenario, `sf_stats.tsv` must show the schema graph
   larger than layer 1's `melnik` by exactly what the edge files imply:
   - Δvertices = Σ_side (2·k_side + 1);
   - Δedges = Σ_side (columns_side + 2·k_side);
   - k_side = the number of distinct classes of that side's edge file.

   The formula was validated on C's local family-7/13 screen runs (180 of 180 scenario runs exact).

   **Where it is checked:** wherever layer 1's `<m>__melnik` has an output (B's review: six layer-1 matchers have
   partial coverage).
   - **Refused:** a class arm with an output there but no statistics row, or a wrong growth.
   - **Counted and reported, not refused:** scenarios without a class-arm output, which the inclusion rule
     already removes from every contrast of that matcher; and class outputs without a layer-1 output, which
     cannot enter a contrast against `melnik`.

#### 5. Hypotheses

**Plain (primary):**
- **H8a (deterministic, extended Proposition 1):** under `melnik_shape` and `melnik_semtype`, 0 within-block
  inversions inside the true refined blocks (datatype block × class pair from the edge files; tolerance 1e-6), for
  every matcher and scenario. Also for MD5.
- **H8b (the classes carry information):** `melnik_shape` − `melnik_shape_shuffled` > 0, and the same for `semtype`.
- **H8c (class nodes help SF as published):** `melnik_shape` − `melnik` > 0, and the same for `semtype`.
- **H8d (late fusion beats class nodes on single tables):** L_p − `melnik_shape` > 0 and L_s − `melnik_semtype` > 0.
- **H8e (the bonus after propagation beats propagating it):** B_p − `melnik_shape` > 0 and B_s − `melnik_semtype` > 0.

**MD5 (its own prediction; the rule as above):**
- **H8f (semantic types collapse on hashed values; 592 of 764 tables have one type):** `melnik_semtype` − `melnik`
  lies within ±0.01: the 95% interval inside (−0.01, +0.01).
- **H8a** as above.
- Everything else on MD5 is reported without a hypothesis. The pattern classes keep about 5 classes per table there,
  so no direction is predicted.

#### 6. What had been seen

- the screen (X22–X24, 4 matchers, 60 scenarios);
- X25/X26 (additive class terms, plain dev and MD5);
- ER10 and X37 (independent corpus);
- T1's screen;
- the class statistics of `data/classfull_*/class_stats.txt` (inputs).

**No classfull output** (matrices, scores, `sf_stats.tsv` of the classfull runs) was read. The screen's 4 matchers are
part of the panel and are not excluded; the primary set excludes the screen's scenarios.

#### 7. Scorer

`layer3b/score_classfull.py`, frozen with this draft (its hash is recorded in SYNC when it is frozen). It imports
`layer3b/score_r7.py` (761f2c14…, frozen since R7). Tested only on C's local screen runs, never on classfull outputs
(`layer3b/make_classfull_scorer_test.py`, 4 matchers, 60 scenarios). On those it reproduces X22–X24:
- shape − melnik +0.0476; semtype − melnik +0.0642;
- L_p − shape +0.0274; L_s − semtype +0.0017 [−0.0130, +0.0140];
- B_p − shape +0.0413; B_s − semtype +0.0131;
- the edge check is exact in 960 of 960;
- H8a: 0 inversions.

Negative test: a "class" arm without class edges fails the edge check (60 of 960), and the scorer writes nothing.
Partial-coverage test (after B's review): one matcher with 5 scenarios removed from its layer-1 and class outputs,
and another with 3 removed from one class arm. The scorer runs and reports 23 cells without output; the first
matcher drops out of every contrast and the second out of that arm's contrasts.
**B checked the statistic and the report against §3 (SYNC 2026-10-04).**

**Disclosed:** on the screen, H8d (semtype) sits at +0.002 with an interval across 0. It is registered anyway as a
directional prediction from X24.

## Erratum 1 to Amendment 8 (2026-10-04, 15:48, appended by A before any classfull number existed)

Written by C as `layer3b/AMENDMENT_CLASSFULL_ERRATUM1.md` (SHA-256 b047a561ea22135c707049ddebdd58c84ab12a2e43e12b0853780d6aef2dadee), rechecked and approved by B (SYNC 15:55), appended verbatim. The scorer frozen for Amendment 8 is now `layer3b/score_classfull.py` SHA-256 63ac2d642349166fefa57d1a6349e849a890ec03bb03db5bfb342408ccff65f3 (with `score_r7.py` 761f2c14…). The earlier run (scorer 2f015747…) wrote no number; its console log is kept as `tuning/results/classfull_plain/score_classfull_output.txt`, renamed to `…_crashed_2f015747.txt`.

### Erratum 1 to Amendment 8 (class nodes on the full panel): scorer fix before any number existed

**Written by C on 2026-10-04 after A's crash report; no classfull number had been computed or written.**

**What happened.** The frozen scorer `layer3b/score_classfull.py` (SHA-256
2f0157474d02edcda6025be8ecbea8285ce3db319e386cf5f1feb85a9c7fb8e4) passed both checks on `classfull_plain`:
- the gate, max |difference| 0;
- the class edges, 135,064 checked, 0 wrong; 2,456 cells without a class output, reported only.

It then crashed (ZeroDivisionError in `mrr_both`) before writing anything. The cause: layer 1's `melnik` output S
was used without the finiteness and shape check that the class arms get. `extrema__melnik` has 120 non-finite
outputs where q is finite (A's scan). So S.max() and the B arms became NaN.

**The change** (and nothing else):
1. A scenario whose layer-1 `melnik` output is non-finite or misshapen is skipped for that matcher, before any arm, L,
   B, A or H8a block is built. This is §3's inclusion rule: a matcher without a finite effect on every scenario of the
   case set leaves the contrast, as `score_layer1.py` treats the same outputs.
2. The same check for `constant__melnik` (the H8a blocks), which would otherwise mis-block silently (B's review).
3. The report lists the skipped cells per reason next to the checks:
   - layer-1 `melnik` output not finite or misshapen;
   - class-arm output not finite or misshapen;
   - `constant__melnik` not finite or misshapen;
   - D missing or not finite.

Hypotheses, statistic, case sets, arms and the edge check are unchanged.

**Expected effect:** `extrema` leaves the plain contrasts (as it already leaves the layer-1 means through
partial coverage).

**Tests** (C's local screen runs only):
- the regular test reproduces X22–X24 exactly, as before the change;
- with one layer-1 output made non-finite, the scorer runs, reports 1 skipped cell, and that matcher leaves every
  contrast.

**MD5 checked before the re-freeze** (`layer3b/scan_nonfinite.py --regime md5`): no matcher has a finite q with a
non-finite layer-1 output; only `datatype`'s 86 non-finite q, which the scorer already skipped.

**New scorer:** `layer3b/score_classfull.py`, SHA-256 63ac2d642349166fefa57d1a6349e849a890ec03bb03db5bfb342408ccff65f3, frozen with `score_r7.py` (761f2c14…).
B rechecks it. A appends this erratum after Amendment 8 and runs the new scorer once per regime.

## Amendment 9 (2026-10-04, 18:37, appended by A before any table-context metric was computed): the table-context control

The user saw the draft and approved it (to A, directly). Written by session B as `reference/AMENDMENT_TABLECTX_DRAFT.md` (SHA-256 6ac4925ee9b65f57e07129923455d3f9bbb8eebeabde26527bcc34d24e890bd9), appended verbatim, headings demoted by two levels. (Amendment 7 remains reserved for E2.)

Frozen at appending:

| file | SHA-256 |
|---|---|
| `reference/score_tablectx.py` | d8e5f5f08bdc08fc6614a87f0c6edc3311e96002ed122b89f741416ff5397cb4 |
| `reference/score_additive_indep.py` | d815d580a6ee0392270369fab8e54f21cdfe32b4aee7ee8550f2feadb06ab2c4 |
| `reference/additive_indep_matchers.json` | c83fd152616bfba11f3ebd17b004f797bff0bd56fe9f8fb7b1fe76f0913daf82 |
| `reference/score_layer1.py` | 468034256c1c2711f11cb43a76786e0c7bd92ef9228f9e66846710d2e4bf4b1f |
| `tuning/score_screen_metrics.py` | 8357ad2f149ce100c943177e2026e7f7607e46d2509fcfc205e761939473a01c |

### Draft amendment (Amendment 9 of PREREGISTRATION_LAYERS.md): the table-context control

**Status: draft by B, 2026-10-04, at the user's request ("prepare 1 and 2"; the external reviewer's point 2 of
17:57). Written before any table-context metric was computed; nobody has computed one. Binding once A appends it,
after the user has seen it.** Amendment 7 is reserved for E2; Amendment 8 is the class-node test.

#### 1. Question

T1 (Amendment 6) showed on the independent multi-table corpus that SF beats the column-level additive baselines of
Amendment 5 only with the correct table organisation. With the columns collapsed into one table, SF is the additive
form again and loses. Those baselines see no table context. So T1 shows that SF uses the table structure, not that
iterated propagation is needed to use it.

This amendment asks: does a **non-iterative baseline that sees the same table context** explain SF's advantage? The
answer separates two explanations:
- local aggregation over tables is enough;
- repeated flooding adds beyond it.

#### 2. The table-aware baseline (no iteration, no ground truth)

Per initial matcher and scenario, with q⁺ = max(q, 0) and table(x) the column label's prefix before the first '.':

- **Table-pair score.** For source table a and target table b:
  T(a, b) = ½ (mean over a's columns of their best q⁺ in b + mean over b's columns of their best q⁺ in a).
  It is normalised by its maximum over the scenario's table pairs (0 if that maximum is 0), giving T̂.
- **Context term:** C_ij = T̂(table(i), table(j)). Every column pair inherits how well its two tables match as wholes.
- **Baselines.** Amendment 5's terms, plus λ C:
  - A_dt = q⁺ + 5 D + λ C, against SF as published (layer1_indep `melnik`);
  - A_pt = q⁺ + 5 D + 0.2 eq_p + λ C, against SF with pattern-class nodes (ER10 `melnik_shape`);
  - A_st = q⁺ + 5 D + 0.1 eq_s + λ C, against SF with semantic-type nodes (ER10 `melnik_semtype`).
- **D, eq_p, eq_s and the weights 5 / 0.2 / 0.1** are Amendment 5's, imported unchanged from
  `score_additive_indep.py` (d815d580…).
- **λ** comes from the grid {0, 0.1, 0.2, 0.5, 1, 2, 5}, chosen per term by leave-one-database-out. For each held-out
  database, the λ with the highest mean over the included matchers of the database-macro J_tf on the other two
  databases is used for the held-out database's scenarios (ties: the smaller λ). No λ is chosen on its own
  database's scenarios. The chosen λ values are reported.

The one-step aggregation mirrors what SF's graph makes possible: the evidence of column pairs flows to their table
pair and back.

#### 3. Data and matchers

- The 81 scenarios of the independent corpus (AdventureWorks, Chinook, TPC-H; plain).
- Amendment 5's frozen matcher list (`additive_indep_matchers.json`, c83fd152…).
- Included: matchers with q, D and all three SF arms finite on all 81 scenarios. At drafting, 74 of 74 have all
  outputs present (located, not read).

#### 4. Statistic (Amendment 5's)

- J_tf = (R_exp + F1_Hr)/2.
- Per scenario, the mean over included matchers; then the mean within database; then the mean over the three
  databases.
- 95% percentile bootstrap over scenarios within database: 10,000 resamples, seed 20261007.
- Also reported: the number of matchers with a positive macro difference, per-database estimates, and the R_exp
  and F1_Hr parts as gains over q.

#### 5. Hypotheses and decisions

- **HC0 (mechanism check):** A_dt − A_d > 0. Confirmed if the lower bound is > 0 and the estimate is positive in
  every database.
- **HC1 (primary):** SF − A_dt, with a three-way decision fixed here:
  - **"propagation adds beyond table context"** if the lower bound is > 0 and the estimate is positive in every
    database;
  - **"table context suffices"** if the upper bound is < 0 and the estimate is negative in every database;
  - otherwise **"inconclusive"**, marked **"practically equivalent"** when the interval lies inside (−0.01, +0.01).
- **HC2:** SF+P − A_pt, same rule.
- **HC3:** SF+S − A_st, same rule.

No direction is predicted for HC1–HC3. B expects HC0 to be confirmed.

**Reported only:**
- every λ on all databases;
- the gains over q of every matrix.

#### 6. What had been seen

- Amendment 5 (A_d, A_p, A_s against SF), ER10, X37 (anchors and profiles on this corpus), T1 and its sensitivity
  analysis.
- No metric of any table-context term has been computed. The scorer's self-test uses synthetic matrices only, and
  its input check locates the SF outputs without reading them.

#### 7. Scorer

- `reference/score_tablectx.py`, SHA-256 d8e5f5f08bdc08fc…, frozen with this draft. It imports
  `score_additive_indep.py` (d815d580…) unchanged.
- **Modes:**
  - `--selftest` (synthetic; checks the table-pair term, the leave-one-database-out choice and the decision rule;
    passed);
  - `--check-inputs` (74 of 74 complete; no labels without a table prefix);
  - the registered run, which refuses unless this amendment is in `PREREGISTRATION_LAYERS.md`.
- **Output:** `tuning/results/tablectx_indep_scores/`.
- It needs no cluster run; A runs it once, after appending.

## Amendment 10 (2026-10-04, 20:58, appended by A before any Sherlock SF output was scored): Sherlock types as class nodes (family 13)

The user decided to register ("register it, let C draft the amendment") and approved the draft after seeing it (to A, directly). Written by session C as `layer3b/AMENDMENT_SHERLOCK_DRAFT.md` (SHA-256 b5063efb93478c847d427d5bd2f3e4d5a67fbeb4bcc5d8382dd1e2399dd54bf3), checked by B, appended verbatim, headings demoted by two levels. (Amendment 7 remains reserved for E2.)

Frozen at appending:

| file | SHA-256 |
|---|---|
| `layer3b/score_sherlock.py` | 510f35766dc02402e6a4b2893cdbb39dad44de40c4c050b8b5b36c00af720cbd |
| `layer3b/score_classfull.py` | 63ac2d642349166fefa57d1a6349e849a890ec03bb03db5bfb342408ccff65f3 |
| `layer3b/score_r7.py` | 761f2c146ece3a99447234565228db821e3d81871a55925fce82725235945930 |
| `layer3b/sherlock_edges.py` | c94df06a955508d63e98f63d874b8ac0e5de1976fc88e478a988734ce3c07f2e |
| `data/sherlock_train_columns_means.csv` | e1dce7eaf23b8d1e488d5baaad5d2cff2c990ee0b16f3cb5378d13e264dea434 |
| `servers/sherlock_run.py` | df01d43b71333deec942ed8d7edf7db0cbac0b12980684e8bf664be686b40a2b |

### Draft amendment (Amendment 10 of PREREGISTRATION_LAYERS.md): family 13 with Sherlock's types

**Status: draft by C, 2026-10-04. The user chose "register, then score" (to C, and to A: "register it, let C draft the
amendment"). Written before any Sherlock SF run existed; no Sherlock SF output exists or has been read. Binding once A
appends it, after the user has seen it. Numbering: 7 = E2 (reserved), 8 = classfull, 9 = table context.**

#### 1. Question

Amendment 8 used a rule-based semantic typer as a stand-in for family 13. Do the types of a published semantic-type
model, Sherlock, help SF the same way when they are added as class nodes? On plain, the rule-based types raised SF by
+0.027 J_tf (H8c), and late fusion was about equal (H8d semtype +0.003, across 0). On MD5 they collapsed (H8f).

#### 2. Inputs and arms (configurations built, nothing run)

- **Sherlock** (mitmedialab/sherlock-project, commit f194013e), run by A on <gpu-host> as in its notebook 00:
  `extract_features` with its ≤ 1,000-value sample (seed 13), then `predict_proba`.
  - NaN features (the 200 word-embedding features when no value has a GloVe token) are filled with the training-set
    means, as Sherlock's notebook 01 does. A re-extracted the training columns with Sherlock's own pipeline.
  - Provenance: `data/sherlock_HASHES.txt` and `data/sherlock_train_columns_means.csv`. The invalid first run is kept as
    `data/sherlock_*_INVALID_nan`.
  - Outputs: `data/sherlock_<regime>/`. Validated against the corpus by `layer3b/sherlock_edges.py check`.
- **Classes** (`layer3b/sherlock_edges.py`, rules fixed before any Sherlock output existed):
  - a column's class is Sherlock's top-1 type, with no probability threshold;
  - a column without values is `mixed_`, as in the rule-based typer (Sherlock's prediction there is ignored);
  - shuffled control: classes permuted within each table;
  - edge label `semtype`, as Amendment 8's semtype arm, so only the class names differ;
  - edge files: `data/sherlock_edges_<regime>/{sherlock,sherlock_shuffled}`.
- **Arms:** `<m>__melnik_sherlock` and `<m>__melnik_sherlock_shuffled` for every layer-1 initial matcher (plain 90,
  MD5 87), plus the gate `levenshtein__melnik_devnone`; `tuning/configs/sherlock_<regime>` (181 and 175
  configurations). They are copies of layer 1's `<m>__melnik`, with only the results directory and two JVM properties
  changed: the extra-edges directory, and the classpath overlay `overlay_er10` (as classfull).
- **References:**
  - layer 1's `melnik`;
  - the H10a blocks from layer 1's `constant__melnik`;
  - the rule-based types: Amendment 8's `<m>__melnik_semtype` (`tuning/results/classfull_<regime>`).
- **Corpus:** all 382 Valentine scenarios, plain and MD5.

**Late fusion and additive terms, computed in the scorer** (eq = 1 where source and target column have the same
Sherlock class):
- L_k = q + 0.2 · eq;
- B_k = SF + 0.2 · max(SF) · eq;
- A_k = q⁺ + 5 D + 0.1 eq.

#### 3. Statistic (as Amendment 8 §3)

J_tf = (R_exp + F1_Hr)/2 (50 seeded tie orders).
- **Per matcher:** the macro over the families with at least 5 scenarios.
- **Then:** the mean over the included matchers, i.e. non-controls with an output in every arm of the contrast on
  every scenario of the case set.
- **Uncertainty:** 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples, seed 20261002.
- **Confirmed** if the lower bound is above 0.
- **Reported as well:** the R_exp and F1_Hr parts, and the number of matchers with a positive effect.

**Case sets:**
- **primary:** all 382 scenarios. No Sherlock output has been seen on any of them.
- **secondary:** the 322 outside the screen, which is Amendment 8's primary set, for the comparison with it.

**Verdicts are defined on the primary set only.** The secondary set is reported without verdicts (B's review).

#### 4. Checks before any number is computed (the scorer refuses otherwise; as Amendment 8 after its erratum 1)

1. **Gate per regime:** `levenshtein__melnik_devnone` equals layer 1's `levenshtein__melnik` cell for cell.
2. **The Sherlock edges reached SF.** For every arm and scenario where layer 1's `<m>__melnik` has an output,
   `sf_stats.tsv` must show the schema graph larger than layer 1's by exactly Δv = Σ_side (2k + 1) and
   Δe = Σ_side (columns + 2k), where k is the number of distinct classes in that side's edge file.
   - **Refused:** a wrong growth, or an arm output without a statistics row.
   - **Counted, not refused:** arms without output, and arm outputs without a layer-1 output.
3. **Skip and count:**
   - a scenario whose layer-1 `melnik` output is not finite or misshapen drops out for that matcher;
   - a non-finite or misshapen arm, rules-arm or `constant__melnik` output, or a missing D, is counted, and that
     arm, block or term is left out.

#### 5. Hypotheses

**Plain:**
- **H10a (deterministic, extended Proposition 1):** under `melnik_sherlock`, 0 within-block inversions inside the true
  refined blocks (datatype block × Sherlock class pair; tolerance 1e-6), for every matcher and scenario. Also on MD5.
- **H10b (the Sherlock classes carry information):** `melnik_sherlock` − `melnik_sherlock_shuffled` > 0.
- **H10c (Sherlock class nodes help SF as published):** `melnik_sherlock` − `melnik` > 0.

**MD5:**
- **H10d:** `melnik_sherlock` − `melnik_sherlock_shuffled` > 0.
- **H10e:** `melnik_sherlock` − `melnik` > 0.
- **Why MD5 gets directional predictions here, unlike H8f:** Sherlock keeps 7.33 classes per table on MD5 (the rules
  kept 1.23). Hashing preserves cardinality, duplicates, value lengths and empty values, and Sherlock's statistics
  features use them.
- H10a as above.

**Reported, without a hypothesis (both regimes):**
- L_k − sherlock, B_k − sherlock, A_k − sherlock;
- sherlock − semtype_rules (Sherlock against the rule-based types);
- sherlock_shuffled − melnik.

C's prediction for L_k − sherlock (≥ 0, near 0) is not directional enough to test.

**C's expected sizes (SYNC 19:45, not part of the verdicts):** plain H10c +0.02 to +0.04; MD5 H10e +0.005 to +0.015.

#### 6. What had been seen

- Amendment 8's results on both regimes (classfull, scored), and X22–X24 (screen);
- Sherlock's class statistics on both regimes, and their agreement with the rule-based types
  (`layer3b/sherlock_diag_<regime>.log`, inputs only);
- the invalid first Sherlock run, refused by the validation before any edge file existed.

**No Sherlock SF output was run or read.** The predictions in SYNC (19:45) were written before the edge files and
configurations were built.

#### 7. Scorer

`layer3b/score_sherlock.py`, frozen with this draft. It imports:
- `score_classfull.py` (63ac2d64…, frozen; `edge_classes`, `expected_growth`, `graph_sizes`);
- `score_r7.py` (761f2c14…, frozen).

It is tested only on C's local f13 screen runs, with the rule-based semtype runs standing in for the Sherlock arms
(`layer3b/make_sherlock_scorer_test.py`, 4 matchers, 60 scenarios). It reproduces Amendment 8's semtype screen numbers:
- sherlock − melnik +0.0642;
- L_k − sherlock +0.0017;
- B_k − sherlock +0.0131;
- sherlock − semtype_rules exactly 0;
- the edge check exact in 480 of 480;
- H10a: 0 inversions.

**Negative test:** arms without class edges fail the edge check (240 of 480), and nothing is scored.

## Amendment 7 (2026-10-04, 21:07, appended by A before any initial-matcher or SF run on these corpora): E2, external replication on dependency-profiled corpora

Approved by the user ("yes to E1 and E2", relayed by B, 2026-10-04 00:00; and to A directly at ~21:05: "prepare E2 and queue it after X42"). Written by session B as `reference/AMENDMENT_E2_DRAFT.md` (SHA-256 41dc665f582bf18a8e0f5be7e04ed30983437f02dd45b685b467fee9b2963fbe), appended verbatim, headings demoted by two levels. Numbered 7 as reserved (appended after 8–10).

**A's notes at appending:** (1) Feasibility (§3): all 10 frozen matchers, both controls and `datatype_nanfix` can run on ctl (the Python ones through the seed server, `datatype_nanfix` with `-Dsmx.datatype.nanAsZero=true` and B's overlay `overlay_gvb`); none is dropped. (2) t2d (stage 2) has no FD/UCC/IND profiles; it is not run now and enters only if it is profiled before any SF run on it, as §2 requires. (3) The configuration generator is `servers/make_e2_configs.py` (stage 1: the initial matchers on ctl; stage 2: seed files and the SF arms as clones of the registered layer-1/3 configurations, only corpus, seed directory, datasets and resultsDir changed, plus R8's TOPK 1 for `fd_top1`).

Frozen at appending:

| file | SHA-256 |
|---|---|
| `reference/score_e2.py` | 26a03aa99c6ac6f370562143b2110923bb84657d6e0f0ff6c193f662b379e04e |
| `servers/make_e2_configs.py` | 172b88be51a218e8c0f1410749e5740206091a858fee2a775c7d81815e42f7fd |
| `make_configs.py` | 087b981543fd8c3cd9ec06e2e24ac450149d994fbd97b15701521952ec49fc6f |
| `servers/export_seed_study.py` | 40e3e5c242008acbf209a7568a341908c54163d93d6373c0d3084ba3964c6c45 |
| `reference/score_layer1.py` | 468034256c1c2711f11cb43a76786e0c7bd92ef9228f9e66846710d2e4bf4b1f |
| `tuning/score_screen_metrics.py` | 8357ad2f149ce100c943177e2026e7f7607e46d2509fcfc205e761939473a01c |

##### Draft amendment (PREREGISTRATION_LAYERS.md, Amendment 7): external replication of the headline predictions on dependency-profiled corpora

Written by B on 2026-10-04 at the user's request ("yes to E2", backlog E2). **A reviews it, fills in the hashes
of the configuration generator and appends it before any initial-matcher or SF run on these corpora.**

###### 1. Why, and what has been seen
- **Why:** Valentine's 382 scenarios come from about 10 independent sources (X18, X35). The independent
  multi-table corpus has three databases and no dependency profiles. So the dependency claims have never been
  tested outside Valentine.
- **What has been seen on these corpora:** only file listings, the profile status files (controlled: complete;
  T2D: none) and the scenario names. No initial matcher or SF has run on them in this study, and no metric
  exists.

###### 2. Corpora
- **ctl:** `harness/generated/controlled-imported-v3`.
  - Lahman and MusicBrainz, each with the 4 Valentine relationship types in two evidence regimes: plain, and
    opaque (MusicBrainz: independent keyed recoding; Lahman: monotone numeric transforms, categoricals
    recoded). 16 single-table scenarios, FD/UCC/IND profiles complete.
  - The opaque regime is the external counterpart of MD5.
- **t2d** (stage 2): `harness/generated/t2d-sm-wh`, 178 single-table web-table scenarios. It enters only if A
  materialises its FD/UCC/IND profiles with the same harness pipeline **before any SF run** on it. Otherwise
  P1, P3 and P4 are decided on ctl-plain alone and reported as such.

###### 3. Initial matchers and arms (frozen now)
- **Initial matchers (10, chosen to span weak to strong and label / instance / dependency):**
  - levenshtein, prefix_suffix, jaro_winkler, sbert_mpnet, pmf, ks, jsd_values, kang_naughton, role_gcov,
    datatype;
  - controls, reported separately: constant, oracle_noise050;
  - plus `datatype_nanfix` (X30's NaN-free datatype matcher) for the additive term.
  - A matcher that cannot run on a corpus is dropped **before any output** and listed. Nothing is chosen
    after outputs exist.
- **Arms:** clones of the registered configurations, only the corpus and seed directory changed.
  - layer 1's `melnik`, `fd`, `coef_prod_fd` (standard);
  - layer 3's `ucc`;
  - `fd_top1` = layer 1's `fd` with R8's TOPK 1 pair scope (registered build);
  - plus the additive term q + 5·D (Python), with β = 5 fixed from X19/X26.
- **Paths:**
  - initial matrices: `reference/seeds/<corpus>/<matcher>/…/seed.csv`;
  - SF outputs: `tuning/results/e2_<corpus>/<matcher>__<arm>/`.

###### 4. Predictions (frozen from Valentine), statistic, decision
Statistic, as in Amendment 5:
- per scenario, the mean over matchers;
- then the mean within family (Lahman, MusicBrainz; T2D);
- then the mean over families;
- 95% bootstrap over scenarios within family, 10,000 resamples, seed 20261005.

The ≥ 5-scenario family rule does not apply (ctl has 4 scenarios per family and regime). Per-family estimates are
reported.

- **P1:** the better the initial matcher, the less SF adds: Spearman over matchers of (initial R_exp,
  SF(melnik) − initial on R_exp) < 0. **Decided on t2d** (upper bound < 0), reported on ctl.
- **P2:** FD edges cost the ranking on readable data (fd − melnik on R_exp < 0 on ctl-plain and t2d) and
  help on recoded data (fd − melnik on J_tf > 0 on ctl-opaque).
- **P3:** on single tables the additive datatype term does at least as well as SF: (q + 5D) − SF(melnik) on
  J_tf > 0 on t2d and ctl-plain.
- **P4:** the top-1 scope repairs FD edges' ranking loss: fd_top1 − fd on R_exp > 0 on ctl (both regimes) and
  t2d.
- **Decision:** confirmed if the interval's bound is on the predicted side (lower bound > 0, or upper bound < 0
  for "< 0"). With only 8 ctl scenarios per regime, ctl's intervals are wide; a "not confirmed" there is
  reported as inconclusive, not as refuted, unless the interval lies on the other side.
- **Reported, not tested:** coef_prod_fd − melnik and ucc − melnik on J_tf.

###### 5. Code
| file | SHA-256 |
|---|---|
| `reference/score_e2.py` (scorer; imports `score_layer1.score_seed` unchanged) | 26a03aa99c6ac6f3… (full hash at appending) |
| `reference/score_layer1.py` | … (A) |
| configuration generator for `e2_<corpus>` | … (A) |
| `tuning/score_screen_metrics.py` | 8357ad2f149ce100… |

**Run order:**
1. Seeds (the initial matchers, incl. `datatype_nanfix` with `-Dsmx.datatype.nanAsZero=true` and B's devclasses_b
   overlay) on ctl.
2. The SF arms on ctl.
3. T2D profiling, then seeds and arms on t2d.
4. `reference/score_e2.py --corpus ctl`, then `--corpus t2d`.

## Amendment 11 (2026-10-05, 00:21, appended by A before any role-term metric on t2d or ctl): registered replication of the dependency-role term

Requested by the user ("draft the registered replication of the role term", to B) and approved after seeing the draft (to A, directly: "yes, append it and set up the arms"). Written by session B as `reference/AMENDMENT_ROLES_DRAFT.md` (SHA-256 f33d59bee4a01198f1a1effb6be8efd5d62f5c0b6811d4752398dc7717a1f703), appended verbatim, headings demoted by two levels. §4's arms are built by `servers/make_e2x_configs.py` (listed below), exactly as X33 built them, from each corpus's own E2 configurations; the t2d corpus is A's exploratory T2D run (`servers/make_t2d_configs.py`, reusing the registered E2 generator unchanged).

Frozen at appending:

| file | SHA-256 |
|---|---|
| `reference/score_roles_reg.py` | f3d8eedb8877bddec3a79727d819d1b689aa73f3aba7754ca78c412cf96c74d9 |
| `explore_deproles.py` | 9f17972254462c114d86e10175c6f50c72db2c4f61cfa7485ab2e2a71c34e7b9 |
| `reference/score_layer1.py` | 468034256c1c2711f11cb43a76786e0c7bd92ef9228f9e66846710d2e4bf4b1f |
| `tuning/score_screen_metrics.py` | 8357ad2f149ce100c943177e2026e7f7607e46d2509fcfc205e761939473a01c |
| `servers/make_e2x_configs.py` | e6d0e9980f7606d0a374426eea585ed6881e595aa4ff81a5449a852868fe10a9 |
| `servers/make_t2d_configs.py` | 6fbae7aa07c42d809f124a4de4a7f20c8bb56b06034519124551887effa9adbc |
| `servers/make_e2_configs.py` | 172b88be51a218e8c0f1410749e5740206091a858fee2a775c7d81815e42f7fd |

### Draft amendment (Amendment 11 of PREREGISTRATION_LAYERS.md): registered replication of the dependency-role term

**Status: draft by B, 2026-10-05, at the user's request ("draft the registered replication of the role term").
Written before any role-term metric on these corpora was computed. Binding once A appends it, after the user has seen
it. Numbering: 7 = E2, 8 = class nodes, 9 = table context, 10 = Sherlock.**

#### 1. Question

X46 (exploratory, Valentine, 382 scenarios) found that an additive term on each column's dependency roles, added to
the datatype term, is level with SF's best dependency graphs and beats the others:
- **Readable:** +0.038 over the datatype term; level with the top-1 key (+0.009) and the key with weighted FD edges
  (+0.007).
- **Recoded:** +0.151 over the datatype term; level with FD+UCC+IND (+0.022) and the key with weighted FDs (+0.009).

The role features and the weights were developed on Valentine. This amendment tests, with every weight fixed, whether
the result holds on data the term was not developed on.

#### 2. Corpora

- **t2d (primary):** `harness/generated/t2d-usable`, the 175 single-table WDC T2D scenarios.
  - It is A's profiled re-import (harness pipeline, the controlled corpus's parameters; 3 of 178 scenarios refused
    by the harness, listed in SYNC 23:42).
  - FD profiles exist for all 175, UCCs for 154.
  - Readable data. One family.
- **ctl (secondary):** `harness/generated/controlled-imported-v3`, E2's corpus.
  - Lahman and MusicBrainz, 8 plain and 8 opaque scenarios, with FD/UCC/IND profiles.
  - Reported per regime with the same rules. With 4 scenarios per family and regime its intervals are wide, so it
    does not decide.

#### 3. The terms (weights fixed, nothing chosen on these corpora)

With q⁺ = max(q, 0) and D the NaN-free datatype matcher:
- **A_d** = q⁺ + 5D (Amendment 5's term).
- **A_role** = q⁺ + 5D + γR, where:
  - R is the role similarity, computed by `explore_deproles.py` (imported unchanged): R_ij = 1 − mean |r(i) − r(j)|
    over six features, each normalised per side (FD left-hand-side count, FD right-hand-side count, UCC count, key
    flag = in a smallest-arity UCC, IND out-degree, IND in-degree);
  - **γ = 0.2 on readable data** (t2d, ctl-plain) and **γ = 1.0 on recoded data** (ctl-opaque), X46's choice on all
    Valentine families;
  - the one-step neighbour term is not used (it added nothing in X46).

#### 4. SF comparators

All use the same initial matrices, metrics and tie orders (`score_layer1.score_seed`).
- **melnik and fd:** E2's arms (`tuning/results/e2_ctl`) and A's exploratory t2d run (`tuning/results/e2_t2d`).
- **ucc_top1 and ucc1_fd_topk:** new for this amendment, built by A as X33 built them, under
  `tuning/results/e2x_<corpus>/<m>__<arm>`:
  - **ucc_top1:** QUICK, `uccFilterThreshold 0.0`, `dependencyTopK 1`, on the shared build;
  - **ucc1_fd_topk:** as in `gvfull`: `smx.sf.topK=1`, `smx.sf.uccTopK=1`, `smx.sf.uccUnweighted=true`,
    `overlay_gvb`;
  - otherwise each is a copy of the corpus's `melnik` configuration, with only these settings and resultsDir
    changed;
  - gate: none is needed for new arms, but A checks one `melnik` twin against the existing run, cell for cell.
- **Initial matchers:** E2's panel (the 10 matchers and the controls; `datatype_nanfix` for D).

#### 5. Hypotheses (decided on t2d)

| | contrast | rule | X46 plain estimate |
|---|---|---|---|
| **HR1 (primary)** | A_role − A_d | lower bound > 0 | +0.038 |
| HR2 | A_role − SF(melnik) | lower bound > 0 | +0.051 |
| HR3 | A_role − SF(fd) | lower bound > 0 | +0.148 |
| HR4a | A_role − SF(ucc_top1) | non-inferiority: lower bound > −0.01 | +0.009 |
| HR4b | A_role − SF(ucc1_fd_topk) | non-inferiority: lower bound > −0.01 | +0.007 |

HR4a and HR4b test the claim "level with the best graphs" as non-inferiority with a margin of 0.01 in J_tf. They do not
test equivalence.

On ctl the same five contrasts are reported per regime, with γ = 0.2 on plain and γ = 1.0 on opaque. They are
secondary, and a "not confirmed" there is weak evidence either way.

#### 6. Statistic

- J_tf = (R_exp + F1_Hr)/2.
- Per scenario, the mean over the included matchers (non-controls with q, D and every SF arm on every scenario of the
  cell).
- Then the mean within family (t2d: one family; ctl: Lahman, MusicBrainz), then the mean over families.
- 95% percentile bootstrap over scenarios within family, 10,000 resamples, seed 20261012.
- Also reported: per-family estimates and the number of matchers with a positive difference.

#### 7. What had been seen

- X46 on Valentine (all numbers above), X32, X33, and Amendments 5, 6 and 9.
- On t2d and ctl: the initial matrices exist (E2 stage 1, and A's exploratory t2d stage 1), and some SF outputs or
  E2/T2D verdicts may exist or be scored before this is appended (A's exploratory T2D run; E2). Those concern A_d and
  SF, not the role term.
- **No role-term metric has been computed on t2d or ctl.** The scorer's self-test uses synthetic numbers only; its
  input check counts files without reading SF outputs.

#### 8. Scorer

- `reference/score_roles_reg.py`, SHA-256 f3d8eedb8877bdde…, frozen with this draft.
- **Imports, unchanged:**
  - `explore_deproles.py` (9f17972254462c11…): profiles, roles;
  - `reference/score_layer1.py` (468034256c1c2711…);
  - `tuning/score_screen_metrics.py`.
- **Modes:**
  - `--selftest` (synthetic: confirms and refuses as designed; passed);
  - `--check-inputs` (t2d: 175 scenarios, 175 with FDs; ctl: 16; 13 seed directories each; SF directories not yet
    present);
  - `--corpus t2d|ctl`, which refuses unless this amendment is in `PREREGISTRATION_LAYERS.md`.
- **Output:** `tuning/results/rolesreg_<corpus>_scores/`.
- A runs it once per corpus, after the SF arms of §4 are collected.


## Erratum 1 to Amendment 11 (2026-10-06, 11:08, appended by A; the user approved the parser fix via B): the role term's FD parse

- **Defect (found by B, 2026-10-06):** the FD-file parser of `explore_deproles.py` (registered 9f17972254462c11…; R,
  the role similarity, is computed with it) read an FD's right-hand side with `\S+`. A right-hand-side column whose
  name contains a space was therefore cut at the first space (e.g. "Bilateral aid (international marker)" →
  "Bilateral").
  - On t2d (this amendment's primary corpus) this concerns 1,090 FD lines in 125 of 175 scenarios (17.9% of FDs).
  - ctl parses identically under both versions, so its secondary results are unchanged.
  - Schematch's Java FD reader is not affected (SYNC 2026-10-06, A → B): the SF arms of §4 used the full names.
- **Fix (B):** `explore_deproles.py` now reads the right-hand side up to " (pdep" (SHA-256 4325b9fe91e183c6…).
  `DEPROLES_FROZEN_FD_PARSE=1` restores the registered parse. The frozen scorer `reference/score_roles_reg.py`
  (f3d8eedb8877bdde…) is unchanged.
- **Check:** with `DEPROLES_FROZEN_FD_PARSE=1`, the scorer reproduces the registered t2d report **byte for byte**
  (`tuning/results/rolesreg_t2d_scores_frozenparse/report.md`). Every registered row of scores.csv is unchanged; the
  file has additional rows only for initial matchers added to reference/seeds/t2d after the registration (X52, ISResMat),
  which are not in the registered panel.
  - The registered output `tuning/results/rolesreg_t2d_scores/` is left as it was.
- **Corrected parse** (`tuning/results/rolesreg_t2d_scores_fixedparse/report.md`; same scorer, same panel of 9 initial
  matchers, same γ):

| hypothesis | registered (frozen parse) | corrected parse | verdict |
|---|---|---|---|
| HR1 (primary): A_role − A_d | +0.0175 [+0.0071, +0.0282] | +0.0188 [+0.0081, +0.0298] | CONFIRMED (both) |
| HR2: A_role − melnik | -0.0095 [-0.0281, +0.0082] | -0.0082 [-0.0272, +0.0099] | not confirmed (both) |
| HR3: A_role − fd | +0.0215 [-0.0031, +0.0453] | +0.0228 [-0.0018, +0.0467] | not confirmed (both) |
| HR4a: A_role − ucc_top1 (margin −0.01) | +0.0364 [+0.0136, +0.0596] | +0.0376 [+0.0147, +0.0606] | CONFIRMED (both) |
| HR4b: A_role − ucc1_fd_topk (margin −0.01) | +0.0316 [+0.0081, +0.0550] | +0.0328 [+0.0092, +0.0558] | CONFIRMED (both) |

  **Every verdict is unchanged**; the estimates move by +0.0012 to +0.0013.
- **γ = 0.2** was frozen from X46's choice, which was made under the defective parse. The registered test keeps γ = 0.2.
  B is re-running X46 to report (exploratory) whether the corrected parse would have chosen a different γ.
- **Reporting:** the registered verdicts stand. The paper reports the corrected-parse estimates beside the registered
  ones and names this erratum.


## Amendment 12 (2026-10-06, 18:20, appended by A before any initial-matcher or SF run on these data): final validation of the frozen recommendations (X76)

Approved by the user ("Sounds good lets run it", relayed by B, SYNC 2026-10-06 16:13; Part B size: three seeded samples, B
16:16; Sakila from the canonical folder, B 17:22). At 18:05 the user asked A directly: "Can we run Part A already?" Written by
session B as `reference/AMENDMENT_FINAL_DRAFT.md` (SHA-256 70b65aa1f11f769b54e1a7142fb3e500987125d0c01f49972e0d98bdc12e056c),
appended verbatim below, headings demoted by two levels.

**A's notes at appending:**
1. **Staged appending.** Part A's data are complete and recorded here, and Part A's runs start after this appending. Part B's
   design is fixed here (builder hash below, `smx.controlled` unchanged), but its build was still running at appending. Part
   B's corpus record (the SHA-256 of controlled-benchmark.json and of the import report, the scenario count, the base-view
   hashes) is appended as "Amendment 12, Part B corpus record" **before any initial-matcher or SF run on Part B**. A does not
   change the builder after this appending. If the build fails, the fix is appended as an erratum before any Part B run.
   No Part A output is read before that record is appended, and nothing in it depends on Part A.
2. **Part A import** (`data/convert_final_indep.py`): 9 pairs, 0 refused; every truth CSV hash equals §3's table. Tables are
   copied row for row (all rows). Truth = the C + L rows; columns absent from a truth file are non-matches. One scenario per
   pair (`<pair>/full`). Profiles with `data/profile_multitable.py` (as Synthea: FD/UCC/IND per table, threshold 0.95, max
   arity 2, IND arity 1).

| pair | tables (source × target) | columns | positives C + L | ground_truth_full.csv SHA-256 |
|---|---|---|---|---|
| bib_s1a_s2b | 15 × 27 | 101 × 53 | 76 (69 C) | 4ceb6661e7ed492b6255a30df6905a1d03eade3f737772355c30dc907ef75682 |
| bib_s1a_s3b | 15 × 5 | 101 × 28 | 26 (25 C) | 5f2a250334154626b4b7b43067bb1febb5139c723a1bc13af6081a5161484c79 |
| bib_s1a_s4b | 15 × 9 | 101 × 42 | 64 (39 C) | cb427a7d215217b6bbe3701bc95d37e089e6da54c53b117615df21247182b23a |
| bib_s2b_s3a | 27 × 5 | 53 × 28 | 15 (15 C) | 2f8f70b5f5ebfc4a496ccbec2cad3cc2e3ed995642657fdf7b52ffb2829a2596 |
| bib_s2b_s4b | 27 × 9 | 53 × 42 | 15 (11 C) | c24db93b070e06befa195a99672e32c63fa9c1939d3a8b7782d908b59fadf48d |
| bib_s3a_s4b | 5 × 9 | 28 × 42 | 14 (12 C) | d398e8026c631b34ea2f949c1527c98a51456e6112a156f3c52cc044cc09ed41 |
| music_mb1_dis2 | 26 × 31 | 199 × 137 | 36 (15 C) | adfd283756b478ae62a7dcce1645ff50475a2ab99aa0301a1fa74844b0700a00 |
| music_mb1_fdb1 | 26 × 2 | 199 × 13 | 9 (4 C) | 28071c7899852675b3893bb8b5f43cf658354ac7d69b69629f7a6da6e43cad5a |
| music_dis2_fdb1 | 31 × 2 | 137 × 13 | 8 (1 C) | 57b161d5c78e379b60b4a81b09a82b534a2fd9f200d4f8ed057877b7cbf761fc |

3. **Part B design facts** (`data/build_final_controlled.py`, fixed before any matcher run; B accepted 1–3 on 2026-10-06, 17:22):
   1. **Inner join.** The driving table is over-sampled 3× by stable hash rank (`smx.controlled._bounded_sample`). The
      first 5,000 records whose required joined rows exist are kept: a bounded sample of the inner join, as the controlled
      Lahman view. AdventureWorks' Person stays a left join (store customers have no person). The skipped candidates are
      counted in the manifest.
   2. **SCOPe CSV repair.** Rows with more fields than the header (unescaped quotes inside free text) have the surplus
      joined back into the description column. That is 287 of 3,372,539 scop_comment rows and 29 of 2,058,199 scop_node
      rows; scop_node's last, truncated record is dropped. All counts are in the manifest.
   3. **Sakila** is read from the canonical `$DATA_ROOT/sakila`, not SakilaHPI (a truncated, denormalised
      variant).
      - Joins: rental → inventory → film → language, and rental → customer → address.
      - film.language is the language's name; customer carries address, district, city_id, postal_code and phone.
      - Every `last_update` column is left out, as the audit timestamps of the other domains.
      - Before the build, a check of all three seeds gave 5,000 complete rows each and 0 skipped candidates.
   4. **The opaque regime** is the controlled generator's structural regime, chosen as for the controlled corpus: monotone-numeric
      for the numeric-heavy domains (AdventureWorks, Sakila; as Lahman), independent-recode for IMDB and SCOPe (as
      MusicBrainz). Labels are hashed in both. Scenario folders are `<domain>_<type>_s<seed>_<regime>`, so the scorer's suffix
      rule reads `_plain` as plain and every other regime as opaque.
   5. Other fixed choices:
      - "NULL" and "\N" become empty.
      - Decimal-comma values are typed string.
      - Row ids, rowguids and audit timestamps are left out. The column lists are `COLUMNS` in the builder.
      - Import: `python -m smx.datasets import`. Profiling: `python -m smx.profiling --threshold 0.95 --max-arity 2
        --ind-arity 1 --dc-rows 1000 --dc-seed 20260904 --materialize`, as the controlled corpus.
4. **Runs.**
   - `servers/make_final_configs.py`:
     - stage 1 is E2's panel plus datatype_nanfix, `--no-sf`;
     - stage 2 clones layer1_plain (melnik, fd) and gvfull_plain (ucc_top1, ucc1_fd_topk). Only the corpus path, the
       datasets, the seed directory and resultsDir change, and the generator asserts this.
   - Gate: `levenshtein__melnik_twin` must equal `levenshtein__melnik` cell for cell before the other arms are queued.
   - Completeness is checked per configuration. A matcher missing on a cell is a deviation recorded in SYNC before scoring.

Frozen at appending:

| file | SHA-256 |
|---|---|
| `reference/score_final.py` | 01e159f573f01c545b57dd610f69b83ba3d8c8c7f1f78314b1cdfbd37b89aa0b |
| `reference/AMENDMENT_FINAL_DRAFT.md` | 70b65aa1f11f769b54e1a7142fb3e500987125d0c01f49972e0d98bdc12e056c |
| `data/convert_final_indep.py` (Part A importer) | 6506365c7b54df37ccd39b06a0efc3b9e54cb16f2b86e3c44ef98a65c715ab7f |
| `data/final_indep/import_manifest.json` | 6bdb5dbf6d7e6d84c473b431bd2810a7b87bf74c880a2b6f92b1713faf599a1e |
| `data/profile_multitable.py` | 159ec69233db90da8214c442dc5183aac99d3084858b59faea591103c3e6eef0 |
| `data/build_final_controlled.py` (Part B builder) | b2490dab3fe85298aece42b8a5c9339a9f12fd64426653d97edb077cdc084cb5 |
| `harness/smx/controlled.py` (unchanged) | 297debc0398d454cd5a175a966f4ac431623e1f5f12e84370242c66a8d0a6ee4 |
| `harness/smx/datasets.py` | e375af1c9aeaf6163743b4c8c91daf6c1a48dd610b9a6377251512af0c2c4082 |
| `servers/make_final_configs.py` | cd3e6443e5756eb9ebca0dab4e23f612aa26600bf2125c9294d25a4c55833917 |
| `servers/make_e2_configs.py` | 172b88be51a218e8c0f1410749e5740206091a858fee2a775c7d81815e42f7fd |
| `make_configs.py` | 087b981543fd8c3cd9ec06e2e24ac450149d994fbd97b15701521952ec49fc6f |
| `servers/export_seed_study.py` | 40e3e5c242008acbf209a7568a341908c54163d93d6373c0d3084ba3964c6c45 |
| `reference/score_tablectx.py` | d8e5f5f08bdc08fc6614a87f0c6edc3311e96002ed122b89f741416ff5397cb4 |

### Draft amendment (Amendment 12 of PREREGISTRATION_LAYERS.md): final validation of frozen recommendations on untouched data

**Status: draft by B, 2026-10-06, at the user's request.**
- The external reviewer's optional last experiment; the user chose: both parts; B drafts the truth and the user validates it;
  preregistered.
- Written before any matcher, profile or SF run on these data. Binding once A appends it, after the user has seen it.
- The plan is in FINAL_VALIDATION_PLAN.md (X76).

#### 1. Question

Do the paper's recommendations, with every weight frozen, hold on data the study has not touched? This includes independently
designed schemas, where Valentine's shared-source construction does not apply.

#### 2. The recommendations under test (nothing is fitted on these data)

With q⁺ = max(q, 0), D the NaN-free datatype matcher (datatype_nanfix), J_tf = (R_exp + F1_Hr)/2:

| id | recommendation | contrast | applies to |
|---|---|---|---|
| F1 | the datatype term A_d = q⁺ + 5D is at least as good as SF as published | A_d − SF(melnik) | Part B |
| F2 | the role term A_role = A_d + γR is no worse than SF's best single-table dependency graphs | A_role − SF(ucc_top1), A_role − SF(ucc1_fd_topk); margin −0.01 | Part B |
| F3 | SF's gain shrinks as the initial matcher improves | Spearman ρ over matchers of (R_exp of q, R_exp of SF(melnik) − R_exp of q) | Part B |
| F4 | on readable data, FD edges lower SF's ranking | R_exp of SF(fd) − R_exp of SF(melnik) | Part B, plain |
| F5 | on multi-table schemas, the one-step table-context term A_dt = A_d + 0.5C is at least as good as SF as published | A_dt − SF(melnik) | Part A |

- R is the role similarity of explore_deproles.py with the corrected FD parse (Erratum 1). γ = 0.2 readable and 1.0 opaque,
  Amendment 11's frozen values.
- C is the table-context term of score_tablectx.py with Amendment 9's λ = 0.5.
- The SF arms are the study's configurations: melnik = layer 1; fd = layer 1; ucc_top1 and ucc1_fd_topk = gvfull/E2x.

#### 3. Data

##### Part A: independently designed schema pairs (multi-table, readable; hand-annotated truth)

From $DATA_ROOT (read-only; not used by the study before).
- **Bibliography:** s1a, s2b, s3a/s3b (one design, two instances), s4b. Six cross-design pairs: s1a–s2b, s1a–s3b, s1a–s4b,
  s2b–s3a, s2b–s4b, s3a–s4b. All tables of each schema.
- **Music:** MusicBrainz (mb1), Discogs (dis2), FreeDB (fdb1). Three pairs: mb1–dis2, mb1–fdb1, dis2–fdb1.
  - All dis2 and fdb1 tables.
  - mb1 is restricted to 26 tables: the concept-map tables plus their direct lookup and link tables — artist, artist_alias,
    artist_credit, artist_credit_name, artist_tag, artist_type, gender, label, label_alias, label_type, release, release_group,
    release_group_primary_type, release_group_tag, release_country, release_label, release_status, area, medium, medium_format,
    medium_cdtoc, cdtoc, track, recording, isrc, tag.
  - The other 32 mb1 tables have no counterpart and are mostly empty or very large lookups (language, script, annotations,
    cover art).
- **Truth:** drafted by B from schema, table and column names and sample values, without any matcher output. The user validated it
  (decisions recorded in the generator scripts).
  - Each correspondence is C (certain) or L (likely).
  - **Primary truth: C + L. Sensitivity: C only.**
  - Frozen files (SHA-256):

| file | sha256 |
|---|---|
| data/final_gt/bib_s1a_s2b.csv | 20a604eb2906ca283925dcb27ebb3166fd38b30a1c9a168495b5ce43565e7fcd |
| data/final_gt/bib_s1a_s3b.csv | 28acddb539e1adc5a41091df6e85d4e06e0f9d84a259b24a801be611a947177e |
| data/final_gt/bib_s1a_s4b.csv | 1bbdea23c758f64bba6ed1e0419121c0fec01c86018f2cefcaabff6ba52362f0 |
| data/final_gt/bib_s2b_s3a.csv | 94af37dc92ecc61fa96aa43b8eb1c1aaf258bcefd790392138d7cd110c70fa82 |
| data/final_gt/bib_s2b_s4b.csv | d2289c84b9d998273b99f59f30ff2a8c1d7f4858c4fa8c332ee147cb91ef3045 |
| data/final_gt/bib_s3a_s4b.csv | 2b05df24cc5e3f8fa7c9edd590200484cb19de213ce1e1cd81501b77cb810491 |
| data/final_gt/music_mb1_dis2.csv | 04a74b59d043229c616ff53965eb9d3dbe1d4e4a8f2c53d635039fcfe4d898e3 |
| data/final_gt/music_mb1_fdb1.csv | c8e78712a59f38380e1131fd6039706c498c6c6111b31112e092fa5a49eedfb0 |
| data/final_gt/music_dis2_fdb1.csv | dd020649c7b5d4d6f5c84a559a979e7a42fa963e20a709743c8dfd65ea058f37 |
| data/final_gt/concepts.csv | 0ce8219dc404bc33f5cf18ebc95a6f276c001083aedf691f76fec595dec81067 |
| data/final_gt/music_concepts.csv | 80951d68a1d5c84623dec71394c0cde7a35811ac172cab56ea9655a9af3bcf49 |
| data/final_gt/draft_bib_gt.py | 867816df9fc67a5002bfc70ef64c2f8cd369139ff689eb8da464a75a3ca5f11d |
| data/final_gt/draft_music_gt.py | 1d8938549d236019619b03f30b2fe91d085a9073d07a4e8849f669dad5416232 |

  - A's importer turns each pair into the harness's multi-table Valentine format: one folder per side, with
    "table.column" labels in the truth matrix. A records the importer version and the matrices' hashes before any matcher
    runs. Columns absent from a truth file are non-matches.

##### Part B: Valentine-style scenarios from untouched sources (single table per side; plain and opaque)

- Built with harness/smx/controlled exactly as the controlled Lahman/MB corpus:
  - hash-ranked base rows;
  - the four relationship types (Unionable, View-Unionable, Joinable, Semantically-Joinable);
  - plain, and opaque (labels hashed, values independently recoded).
- Base relations, chosen before any matcher result:
  - **AdventureWorks:** Sales.SalesOrderDetail → Sales.SalesOrderHeader → Sales.Customer (→ Person.Person where needed);
  - **IMDB:** principals → name → basics;
  - **Sakila:** rental → inventory → film (+ customer);
  - **SCOPe:** scop_comment → scop_node → scop_level.
- **Size (the user's decision, 2026-10-06, before the append):** three seeded row samples per relationship type and regime (seeds
  20261006, 20261007, 20261008; 5,000 rows each).
  - That is 4 types × 2 regimes × 3 = 24 scenarios per domain, so 12 per domain and regime, and 96 in total.
  - The controlled generator's single sample per type (8 per domain) would leave 4 values per domain and regime for the
    within-domain bootstrap.
  - A fixes the remaining parameters in the importer manifest before any matcher runs.

#### 4. Panel and runs

- **Initial matchers:** E2's panel (layer-1 matchers without controls, LLM matchers or matchers needing approval).
- **Profiles:** FD/UCC/IND with the harness's standard profilers.
- **SF arms:** melnik and fd (layer-1 configs); ucc_top1 and ucc1_fd_topk (gvfull/E2x configs); unchanged.
- The additive terms (A_d, A_role, A_dt) are computed offline from the same initial matrices.
- **Included matchers per cell:** every arm on every scenario of the cell.

#### 5. Statistic and decision rules

- **Part B:**
  - per scenario, the mean over included matchers; mean within domain; mean over domains;
  - 95% percentile bootstrap over scenarios within domain, 10,000 resamples, seed 20261076;
  - reported per regime (plain, opaque).
  - **Decisions (plain is primary; opaque reported with the same rules):**
    - F1 lower bound > 0;
    - F2 lower bounds > −0.01 for both graphs;
    - F3 ρ < 0 with its bootstrap interval (resampling matchers) entirely below 0;
    - F4 upper bound < 0 (plain only).
- **Part A:**
  - per pair, J_tf as the mean over included matchers;
  - report each pair, the number of pairs in each direction, and the mean over the 9 pairs with a bootstrap over pairs.
  - **F5 is supported if ≥ 7 of 9 pairs have A_dt − SF(melnik) > 0 and the mean's lower bound is > 0;**
  - otherwise "not supported" (reported with the per-pair values). C-only truth is reported as sensitivity.
- **Also reported, not decided:**
  - A_role and A_dt on Part A;
  - SF(fd) on Part A;
  - the per-domain and per-type breakdowns.

#### 6. Predictions (B, written before any run)

- **F1** holds on Part B plain. On opaque it may fail; on Valentine md5 the datatype term was inconclusive.
- **F2** holds in both regimes.
- **F3** holds.
- **F4** holds on plain.
- **F5** is the riskiest. Independently designed schemas are where table context helped least (Synthea), and the bibliographic designs
  partition the same entities very differently. B expects the direction in at least 5 of 9 pairs, with support uncertain.

#### 7. Scorer and deviations

- **Scorer:** reference/score_final.py (sha256 01e159f573f01c545b57dd610f69b83ba3d8c8c7f1f78314b1cdfbd37b89aa0b), written by B before any real output existed.
  - `--selftest` passes on synthetic data. It checks aggregation, the inclusion rule, the F1–F5 decisions (including failures below
    the margin) and the truth mapping, which refuses a truth file naming columns absent from the matrices.
  - Part A also checks that the imported truth matrix equals the frozen CSV (C + L); a mismatch stops the scorer.
- **Layouts the scorer expects:**
  - Part A: `sf-matcher-study/data/final_indep/<pair>/<scenario>/`, where `<pair>` is the truth file stem (e.g. bib_s1a_s2b);
  - Part B: `harness/generated/final-controlled/<Domain>-<Type>/<scenario>_{plain,opaque}/`;
  - seeds: `reference/seeds/final_indep|final_controlled/<m>/`;
  - SF runs: `tuning/results/final_indep|final_controlled/<m>__<arm>`.
- **Deviations:** any departure (a refused scenario, a matcher missing on a cell, a profile timeout) is recorded in SYNC before
  scoring and reported.
- No result of this amendment changes a decision of an earlier amendment.


## Amendment 12, Part B corpus record (2026-10-06, 19:00, appended by A before any initial-matcher or SF run on Part B)

Promised by Amendment 12, note 1. The builder is the registered one (`data/build_final_controlled.py`
b2490dab3fe85298aece42b8a5c9339a9f12fd64426653d97edb077cdc084cb5, unchanged since the appending). No Part A output has
been read (Part A's SF arms were still being collected at this appending).

- **Build** 17:24–18:46. Import (`python -m smx.datasets import`) 18:47. Profiling (`python -m smx.profiling`, the parameters
  of note 3.5) 18:47–18:52, exit 0. **96 scenarios**; FD, UCC, IND and DC are `complete` on both sides of all 96.
- **Layout and scorer:**
  - Layout: `harness/generated/final-controlled/<Domain>-<Type>/<domain>_<type>_s<seed>_<regime>/`, two levels, as
    controlled-imported-v3.
  - §7's example name `s<seed>_<regime>` differs in spelling only; B agreed to keep the import's names (SYNC 2026-10-06 18:50).
  - The frozen scorer's `cases("B")`, run read-only (no metric computed), lists 96 rows: 48 plain and 48 opaque; 12 per domain
    and regime (AdventureWorks, IMDB, Sakila, SCOPe); 6 per dataset (16 datasets = 4 domains × 4 types).
  - `cases("A")` lists the 9 pairs (scenario `full`, plain).
- **Inner join, skipped candidates** (of 15,000 hash-ranked candidates per seed):
  - AdventureWorks and Sakila 0;
  - IMDB 1 per seed;
  - SCOPe 4,515 / 4,665 / 4,724 (comments whose node is not in scop_node).
- **SCOPe repairs** (per seed, the same files): scop_comment 287 rows; scop_node 29 rows plus its truncated last record
  (dropped).

| file | SHA-256 |
|---|---|
| `harness/generated/final-controlled-raw/controlled-benchmark.json` (build manifest) | abe85ef866fd4ce88ce0f79f30f78e94d1133a4c60ba079c7d7f5d9b10a31d66 |
| `harness/reports/final_controlled_import_report.json` | 3912f2a689d0d048888f44f29f497d029c4d312559a45563ad1ab0213615feef |
| `harness/reports/final_controlled_tree.sha256` (SHA-256 of each of the corpus's 2,208 files, sorted by path) | 19aa2c3585ef9012ec073bbc83e2635862480d0b9465f1c235e924662464c14d |
| `harness/smx/profiling.py` | 05a2f739a6114d787f685a6c8aa448eb9e477e2b6c9928e37c85b879bf1127ae |

Base views (5,000 hash-ranked rows each; `controlled-benchmark.json` holds the raw-file hashes):

| domain | seed | rows × columns | opaque regime | candidates / skipped | base view SHA-256 |
|---|---|---|---|---|---|
| AdventureWorks | 20261006 | 5000 × 37 | monotone-numeric | 15,000 / 0 | 3e6e64d7a155215b6c19315387b450b9e806c01c65b612985a155be3a0c5d2ae |
| AdventureWorks | 20261007 | 5000 × 37 | monotone-numeric | 15,000 / 0 | d2629e4bac71488f89e384ae7b24d4eb874795a576cb701adffa82078414b2eb |
| AdventureWorks | 20261008 | 5000 × 37 | monotone-numeric | 15,000 / 0 | e12b0222ef78ea6066552762e1197d8b156ee1d8cd9345e1cd7291f9bcdeefed |
| IMDB | 20261006 | 5000 × 19 | independent-recode | 15,000 / 1 | f745856aa5ea6fc063c96499c3477a85ceb6c64d17cf238e6a4eb9e721670ac6 |
| IMDB | 20261007 | 5000 × 19 | independent-recode | 15,000 / 1 | 0077866fabdd93b2489682c4dee26c62e0b04adec16ec8a6de7848b47340239f |
| IMDB | 20261008 | 5000 × 19 | independent-recode | 15,000 / 1 | 1f660eac8febc3ead287c9d657f8491df616808dd23934f6e75e7b3baf92b3f6 |
| Sakila | 20261006 | 5000 × 27 | monotone-numeric | 15,000 / 0 | a511e5e0a15de8e1bc0cc627184649f03fead31267cdf97fbb7a53032ed80a95 |
| Sakila | 20261007 | 5000 × 27 | monotone-numeric | 15,000 / 0 | cf0c6f277155fdeaf5e7cd71400c8b060c81624f7fe37ce4f7f6823e77f57865 |
| Sakila | 20261008 | 5000 × 27 | monotone-numeric | 15,000 / 0 | e4eed0c61814268ae6e668b8bf2e863eb3fa448e7d553e730b024b974e2d8c07 |
| SCOPe | 20261006 | 5000 × 18 | independent-recode | 15,000 / 4,515 | 5dbca91018a310e2407a39a0e6334ec5e11f8cdd797f3045272f6b094afa8745 |
| SCOPe | 20261007 | 5000 × 18 | independent-recode | 15,000 / 4,665 | a00f176715433b55aac3e12ea067b5db73096250fb8e8e1532ca9103b82e9cb6 |
| SCOPe | 20261008 | 5000 × 18 | independent-recode | 15,000 / 4,724 | 9d020c7be9ffa65560cbb537b3f88eb9d7b031c5e1dd6924c55d4dfe2722b339 |
