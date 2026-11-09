# SF parameter tuning: pre-registration

Written 2026-10-02, before any tuning run. The protocol, the search space, the objective and the
decision rules below are fixed. `make_tuning_configs.py` records this file's SHA-256 in every
configuration it writes. A change to anything here is an amendment: append a dated section at the
end, and never edit the text above it.

## Question

Which Similarity Flooding parameters maximise SF's gain over its own seed? Is the best setting the
same for every seed, or does it depend on seed quality, in a way that a quantity computed without
ground truth can predict?

The harness motivates the second part:
- SF costs about 0.17 of global recall against not propagating (FINDINGS 103).
- Controlling propagation strength recovers that, but this was held out on one seed only
  (FINDINGS 105).
- The best restart α is 0 for the production seed (SF_PROPERTIES.md).

## Objective

For each scenario s, seed q and configuration c:

    ΔR(s,q,c) = RecallAtGT_pessimistic(SF_c(q)) − RecallAtGT_pessimistic(q)
    ΔF(s,q,c) = F1_hungarian(SF_c(q)) − F1_hungarian(q)
    J(s,q,c)  = (ΔR + ΔF) / 2

Both metrics come from `harness/smx/core.py`. Pessimistic Recall@GT is included because Hungarian
F1 rewards ties: the constant seed scores F1 = 1.0 by landing on the diagonal. J is aggregated as
a **macro mean over families** (ChEMBL, OpenData, TPC-DI, Magellan, Wikidata), so that the large
families do not decide alone. Ties within 0.002 go to the simpler configuration: no restart, then
the production values.

## Splits

They are fixed by `make_splits.py` and written to `splits.json`:
- **Lockbox:** 25% of the 382 audited cases, stratified by (family, scenario type), with the order
  set by SHA-256 of the case id. It is never used for tuning, and it is scored exactly once, for the
  configurations chosen at the end.
- **Dev:** the other 75%.
- **Screen:** about 60 dev cases, stratified the same way.
- **Folds:** leave-one-family-out over dev. With 5 families there are 5 folds.

## Seed panel

The panel is fixed, and it spans seed quality. Every seed is read as a FILE seed from the full
seed-study run, so SF varies and the seed does not.

| Seed | Why it is in the panel |
|---|---|
| `oracle_noise050` | strong but noisy reference (a control) |
| `levenshtein` | the standard lexical seed |
| `prefix_suffix` | SF's own seed (Melnik / Valentine) |
| `semstruct` | strong learned seed |
| `pmf` | weak instance seed, the harness's base |
| `ks` | weak distribution seed |
| `kang_naughton` | dependency-graph seed |
| `role_gcov` | within-table role seed, the thesis's family |
| `constant` | no information: structure only (a control) |

## Search space (108 configurations per seed)

- **Operator** (9):
  - The historical fixpoint with `fixpoint` ∈ {BASIC, A, B, C}.
  - Restart `x = (1−α) q + α P x` with α ∈ {0.15, 0.30, 0.50, 0.70, 0.85}, set through
    `smx.sf.restartAlpha`.
  - α = 0 is SF switched off (output = seed). It is evaluated once per seed as the
    no-propagation control, not as a search point.
- **Coefficient policy** (2): INV_PROD, INV_AVG.
- **Seed attachment** (2): NAME, ID (`smx.sf.seedAttachment`). SF_SEED_ATTACHMENT.md reports a
  sign reversal with the fixpoint.
- **Dependency strength** (3):
  - `full`: `constraintWeighting: NONE`.
  - `damped`: CONSTANT with `smx.sf.constantWeight` 0.35.
  - `off`: CONSTANT with weight 0.0.

**Held fixed**, because the harness found them irrelevant or settled (FINDINGS 94, 104):
- QUICK encoding, `normalise` GLOBAL, the iteration cap of 200, `consistentEdgeIdentity` true.
- FD threshold 0.0, UCC and IND off.

Graph construction is the next axis, and it is not tuned here.

## Stages and decision rules

1. **Screen.** All 108 configurations, plus the α = 0 control, for every panel seed, on the screen
   cases. For each fold and seed, take the configurations ranked top 10 by the macro J over that
   fold's *training* families. The union of these over folds and seeds goes forward
   (`keep.json`). Pruning per fold keeps the held-out family out of the pruning decision.
2. **Dev.** The kept configurations on all dev cases.
   - Per fold: choose the best configuration per seed, and the best single configuration across
     seeds (by the mean of the per-seed macro J), on the training families only.
   - Report J on the held-out family against two baselines: the production configuration (C,
     INV_PROD, NAME, full, no restart) and α = 0.
   - The LOFO table is the tuning result.
3. **Final choice.** On all dev cases, choose one configuration per seed and one for all seeds.
   Score them once on the lockbox (`chosen.json`). The lockbox number is the headline.

## Adaptive propagation (decided in advance)

For the restart configurations at the production values of the other factors:
- Record α\* for each seed, and each scenario's best α.
- Compute the Spearman correlation between α\* and seed quality (the seed's pessimistic
  Recall@GT), and between each scenario's best α and three ground-truth-free predictors of the seed
  matrix (`CompositeMatcher`): DOMINANTS, STDEV and MAXMEAN.
- **Continue to an adaptive-α design** only if |ρ| ≥ 0.5 for some predictor at the scenario level,
  with the same sign in every LOFO fold. Otherwise, report that α\* is not predictable from the
  seed alone.

## Reporting

All configurations are reported, not only the winners, with:
- main effects and pairwise interactions of the four factors on J (dev);
- the per-family rank stability of the top 10;
- the screen-to-dev rank agreement.

## Amendment 1 (2026-10-02, before any tuning run; found by the 4-configuration smoke test)

The no-propagation control was specified as restart α = 0 at the production attachment (NAME).
Restart at α = 0 returns the *attached* seed, and SF reads its output from the column (ID) pairs.
With NAME attachment the column pairs receive no seed mass, so this "control" produced J between
−0.27 and −1.0 on the smoke test, instead of 0.

The control and the restart-α curve therefore use **ID attachment**, which is what the harness's
restart arms (`restart_a*`) always used. Everything else is unchanged. The production
configuration stays at NAME, and restart × NAME stays in the search space as an ordinary point.
The smoke-test results were deleted, and none of them were used for any decision.

## Amendment 2 (2026-10-02): a tie-free objective for dev and lockbox

Written after the screen was scored, and before any dev configuration was generated or any dev
or lockbox run took place. No dev or lockbox data has been seen.

### What changes

From the dev stage on, selection and the headline use the **tie-free objective**:

    J_tf(s,q,c) = (ΔR_exp + ΔF_Hr) / 2

    R_exp  = Recall@GT, with ties at the cut-off filled at random: the exact expected value
             (hits above the cut-off, plus the free slots times the share of true pairs among
             the tied cells), divided by |GT|.
    F_Hr   = Hungarian F1 with ties broken at random: the mean F1 over 50 random row and column
             orders of the matrix. The same 50 orders are used for the initial similarities and
             for SF's output, so the difference stays paired. The orders are seeded by the
             SHA-256 of the scenario id, so the score is deterministic. Pairs with score ≤ 1e-9
             are dropped, as in harness/smx/core.py's hungarian.

`ΔR_exp` and `ΔF_Hr` are SF(q) − q, as before. Everything else about the objective is unchanged:
the macro mean over families, the tie rule (within 0.002, the simpler configuration), the
leave-one-family-out folds, the choice per initial matcher and of one configuration for all of
them, and the single scoring of the lockbox. The adaptive-α test uses each scenario's best α
under J_tf, with the same threshold (|ρ| ≥ 0.5, same sign in every fold).

The code is in `score_tuning.py` (selection objective `J_tf` for `--stage dev` and
`--stage lockbox`) and `score_screen_metrics.py` (the metric functions).

### What does not change

- **The screen and its keep set.** The screen stays scored by the registered J, and `keep.json`
  (168 pairs) is the one it produced at 15:26 and again at 15:34. Re-running the amended scorer
  on the screen reproduces it byte for byte.
- **The splits, the seed panel, the search space and the stages.**

### What is reported

The registered J is reported next to J_tf at every stage, together with the four components
(ΔR_pess, ΔR_exp, ΔF_H, ΔF_Hr), and with ΔF1 after mutual top-1 and ΔMRR (pessimistic and
expected) as secondary metrics.

### Why

The registered J averages two tie-sensitive metrics, which are biased in opposite directions:

- Hungarian F1 breaks ties by row and column index (scipy's `linear_sum_assignment`). On
  Valentine's near-diagonal ground truth, a tied matrix therefore lands on the true pairs by
  position. On the 60 screen scenarios, the constant control scores 0.80 Hungarian F1 with
  index ties and 0.08 with random ties, and the within-table inclusion matcher 0.56 against
  0.17. Untied matchers are unaffected (Levenshtein 0.62 against 0.62).
- Pessimistic Recall@GT counts every tie against the ground truth.

When SF breaks the ties of a tied input, the first bias registers as a loss and the second as
a gain. J therefore partly measures tie-breaking rather than propagation. The bias was known in
principle when this protocol was written, which is why Recall@GT was added next to F1. Its size,
and the fact that the two biases do not cancel, only showed up when the finished screen was
re-scored on every metric (`score_screen_metrics.py`, `results/screen_plain_metrics/report.md`).

### Disclosure: this change was made after seeing the screen

On the screen:
- Under J, no configuration beats switching SF off (best propagating −0.016, standard −0.043).
- Under J_tf, the best configuration reaches +0.015 and the standard one +0.010.

The change is therefore made with knowledge of the screen, and it moves the screen's verdict in
favour of propagation. Three things limit what that knowledge can buy:

- **The reason for the change is a measurement defect**, the index-order tie-breaking, not the
  direction of a result. J_tf is the registered J with its two tie-sensitive parts replaced by
  their tie-invariant versions; no new component or weighting is introduced.
- **The registered J is still reported at every stage.** Any conclusion that holds under J_tf but
  not under J is to be stated as such.
- **The screen only pruned.** The ranking of configurations under J_tf agrees with J at Spearman
  ρ = 0.99. The best configuration of every initial matcher under J_tf is in the keep set
  (9 of 9).

### Hash

The screen configurations record this file's SHA-256 before the amendment:
3152f8db78c8886239c5438b64a1ba6b10f40d779fe10da49659b33fc4fad797. Dev and lockbox
configurations will record the amended hash.

## Amendment 3 (2026-10-02): "SF as tie-breaker" as a pre-specified secondary analysis, and the Melnik reference as a report-only baseline

Written before:
- any MD5 or lockbox data was scored for the tie-breaker;
- any lockbox run;
- the dev stage was scored.

It adds analyses. It changes no selection rule, no objective and no split.

### A. SF as tie-breaker

**Origin (exploratory).** Rescoring the plain initial-matcher study (all 382 scenarios, standard
configuration, 74 initial matchers) showed one way of using SF's output that improves every
tie-free metric: keep the order of the initial similarities and let SF order only the pairs they
tie (`explore_posthoc.py`). It raised J_tf by +0.009, tie-free F1 by +0.013 and expected
Recall@GT by +0.004, in each of ChEMBL, OpenData, TPC-DI and Magellan. A fresh random tie-breaker,
as a control, gains nothing. Because these 382 scenarios include the 96 lockbox scenarios, a
lockbox test of the tie-breaker is not fully independent of its origin; the MD5 test below is.

**Construction (fixed).** `tiebreak(q, S)` in `explore_posthoc.py` (SHA-256
a199fa19b28ecc409bb15a4b46144a5964fda3672c25920fb3dd79cc7b23e605):
- cells with q ≤ 1e-9 stay 0;
- positive cells become q + δ · S/max(S), with δ a quarter of the smallest gap between distinct
  positive values of q, divided by min(n, m).

It has no parameter.

**Hypotheses**, on J_tf = (R_exp + F1_Hr)/2 and its ranking part:
- H1: tie-breaker − initial matcher alone > 0;
- H2: tie-breaker − SF as run > 0;
- H3 (non-inferiority): R_exp(tie-breaker) − R_exp(initial matcher) > −0.01.

**Analysis (fixed).** `confirm_tiebreak.py` (SHA-256
24ab193c48dafbcfd7df2734700060873fd94e43dae5b572c1d419e438ce26ea):
- per scenario, the mean difference over the included initial matchers (all non-controls with
  full coverage);
- then the mean over the scenarios of each family, then over the families with at least 5
  scenarios;
- a 95% percentile bootstrap, stratified by family, 10,000 resamples, seed 20261002.

H1 and H2 are confirmed if the lower bound is above 0. H3 is confirmed if the lower bound is above
−0.01.

**Tests:**
1. **Primary: the MD5 regime of the initial-matcher study.** All 382 scenarios, every finished
   non-control initial matcher, standard configuration. Scores from
   `explore_posthoc.py --suffix md5`. The MD5 regime has not been analysed for the tie-breaker.
   - Disclosed: 14 of its initial matchers were scored earlier on the registered metrics, which
     showed SF gaining about 0.03 pessimistic Recall@GT on the label matchers.
   - Precondition: `results/md5` holds only valid runs when the test is run. Magneto enters only
     from its patched rerun, because every Magneto result from before 2026-10-02 is invalid.
2. **Secondary: the lockbox.**
   - For each panel initial matcher, its own configuration in `chosen.json` (role "own"); the
     universal configuration is reported as well (role "universal").
   - Scores from `score_tuning.py --stage lockbox`, which writes `tiebreak_scores.csv` from the
     dev stage on, as a report only.
   - Because of the overlap noted above, a lockbox confirmation is reported as supporting
     evidence, not as an independent replication.

The plain regime remains exploratory; `confirm_tiebreak.py` was run on its dev scenarios only to
test the code.

### B. The Melnik reference as a report-only baseline

The reference configuration of the paper is the Melnik et al. (2002) setting as implemented in
`reference/make_layer1_configs.py` (arm "melnik"):
- schema graph without dependency edges;
- formula C, inverse-average coefficients, NAME attachment;
- prefix/suffix StringMatch on the other literal pairs;
- residual 0.05.

Its layer-1 runs cover all 382 scenarios. For dev and lockbox they are reported next to the
production configuration and the no-propagation control. They are never a selection candidate.
Their lockbox scenarios must not inform any selection, and the dev stage selects on dev scenarios
only, as before.

### Hash

Before this amendment the file's SHA-256 was
3d99900485492a5326a844f2e2b1c2ccd2867c231fadb3d0b5911e3e04d21a3f.

## Amendment 4 (2026-10-04, ~00:45, appended by A before any lockbox run): clean nested selection as the primary dev result, a non-degenerate universal configuration, and the single lockbox scoring

Approved by the user ("yes to E1", relayed by B, SYNC 2026-10-04 00:00). Written by B as `tuning/AMENDMENT_E1_DRAFT.md`, appended verbatim (§1–4), followed by A's review (§5) and the code hashes (§6).

### Draft amendment (tuning PREREGISTRATION.md, Amendment 4): clean nested selection as the primary dev result, a non-degenerate configuration for all matchers, and the single lockbox scoring

Written by B on 2026-10-04 at the user's request ("yes to E1", backlog E1). **A reviews it, fills in the hashes
and appends it before any lockbox run.** No lockbox data has been run or scored.

#### 1. What has been seen
- **The dev stage (registered `dev_plain`) and its LOFO table.**
- **The supplementary selection analyses (ER1, ER2; A's `supp_selection.py`, report
  `tuning/results/supp_selection_plain/report.md`):**
  - the per-fold masks (each fold's own screen top 10 plus production);
  - the corrected per-matcher LOFO: +0.0100 J_tf, or +0.0206 with abstention;
  - the universal shortlist of 20 configurations, screened on their own across-matcher objective, with the
    universal choice `fpC__INV_PROD__ID__fdoff` (also with abstention).
- Nothing on the lockbox.

#### 2. What changes, and why
1. **The leak and its scope.** The registered LOFO table selects from the union of all folds' shortlists, so in
   9 of 45 choices a fold used a configuration that only another fold's screen kept, and that screen had seen
   the held-out family (ER2). The leak affects **only the LOFO estimates**: the final per-matcher choice is
   made on all dev families with no family held out, so `chosen.json`'s per-matcher entries never touched
   lockbox data and stay valid.
2. **The registered universal choice was degenerate.** After per-matcher pruning, only production was kept
   for every matcher (ER1).
3. **Therefore:**
   - **(a) Primary dev result:** the corrected LOFO table, in which each fold selects only from its own screen
     shortlist (+ production). The registered LOFO table is reported in the supplement as the historical
     analysis, with the leak described.
   - **(b) The universal configuration for the lockbox:** `fpC__INV_PROD__ID__fdoff` (ER1, chosen on all dev
     families from the 20-configuration universal shortlist; abstention gives the same). It **replaces**
     production as the universal entry. Production stays in the lockbox as a baseline, as registered.
   - **(c) The per-matcher entries of `chosen.json` are unchanged.** `chosen.json` itself is not rewritten; the
     clean universal choice is read from `universal_choice.json` (hash below).

#### 3. The lockbox (scored once)
- **Scenarios:** the 96 lockbox scenarios, plain.
- **Matchers:** the 9 panel matchers; the controls are reported separately.
- **Arms per matcher:**
  - its `chosen.json` per-matcher configuration ("own");
  - the clean universal configuration;
  - production;
  - α = 0 (no propagation);
  - layer 1's `melnik` (report-only, Amendment 3);
  - the tie-breaker (Amendment 3's secondary analysis).
- **Statistics:** as registered. J_tf (Amendment 2), macro over families with ≥ 5 scenarios, mean over the
  non-control matchers, 95% stratified bootstrap over scenarios, seed 20261002. Added as sensitivity: the
  column-split cluster bootstrap (X18's `cluster_of`).
- **Hypotheses** (none were registered for the lockbox beyond "the lockbox number is the headline"; these are
  stated now, before any lockbox data):
  - **HL1:** own − production > 0.
  - **HL2:** own − α = 0 > 0. The corrected dev stage gives +0.010 against 0.000, a narrow margin, so HL2 is a
    real test.
  - **HL3:** own − universal > 0. The dev stage says yes: +0.023.
  - **HL4:** universal − production > 0. The dev stage says yes: +0.013.
  - Confirmed if the lower bound of the stratified bootstrap is > 0. The cluster bootstrap is reported
    alongside.
  - All per-matcher values are reported; none is selected.

#### 4. Code (A fills in)
| file | SHA-256 |
|---|---|
| `tuning/score_tuning.py` (`--stage lockbox`) | … |
| `tuning/supp_selection.py` | … |
| `tuning/results/supp_selection_plain/universal_choice.json` | … |
| `tuning/chosen.json` (unchanged, 8562afd9…) | 8562afd9… |
| `tuning/keep.json` (unchanged, 6a2a5862…) | 6a2a5862… |
| `confirm_tiebreak.py` (Amendment 3; `--scenarios lockbox --chosen tuning/chosen.json --role own`) | … |

#### 5. A's review (appended with this amendment, 2026-10-04, before any lockbox configuration was generated or run)
- **§2's claim checked: `chosen.json`'s per-matcher entries are free of lockbox information.**
  - `splits.json`: lockbox 96 scenarios, dev 286, screen 60 ⊂ dev; lockbox ∩ dev = lockbox ∩ screen = ∅.
  - `keep.json` comes from the screen only; `chosen.json`'s choices come from all dev families with no family
    held out, from `keep.json` ∪ {production}.
  - Neither step reads a lockbox scenario. ER2's leak concerns only the LOFO estimates, as §2 says.
- **Disclosure: earlier exposure of the lockbox scenarios (not selection).** The lockbox scenarios are part of the
  382-scenario protocol, so many configurations have run on them inside aggregates over all 382:
  - the seed study's standard configuration, which is production;
  - layer 1 (including `melnik`), layer 3, R7, R8, and the nanfix and family reruns.
  - Those aggregates have been read; no lockbox-only number has been computed. The `melnik` arm here is read
    from layer 1's existing outputs on the 96 scenarios.
- **Families:** by the registered ≥ 5 rule the lockbox uses TPC-DI (44), OpenData (28) and ChEMBL (22). Magellan's 2
  scenarios are reported per matcher but are not in the macro.
- **"Own"** follows `confirm_tiebreak.py --role own`: the `chosen.json` entry that is not `chosen.json`'s universal one
  (production), or production if they coincide.
- **Tie-breaker (Amendment 3):** computed as registered with `--role own`. The pinned script's `--role universal` reads
  `chosen.json`'s universal entry (production), not the clean universal. If it is run, it is reported as the
  tie-breaker for production.
- **Run order:**
  1. `make_tuning_configs.py --stage lockbox --regime plain` (unchanged: `chosen.json`'s configurations,
     production, α = 0).
  2. `make_lockbox_universal.py` (adds the clean universal per matcher).
  3. Run.
  4. `score_tuning.py --stage lockbox`. Its leave-one-family-out section re-selects on lockbox families; it is not
     used and not reported as a selection.
  5. `score_lockbox_hl.py` (HL1–HL4 and the cluster sensitivity).
  6. `confirm_tiebreak.py --scores tuning/results/lockbox_plain/tiebreak_scores.csv --scenarios lockbox --chosen
     tuning/chosen.json --role own`.
- **The lockbox is run only on the user's direct go to A.** It is run once.

#### 6. Code hashes (at appending)
| file | SHA-256 |
|---|---|
| `tuning/AMENDMENT_E1_DRAFT.md` (this text, §1–4 as drafted by B) | 622f9a8fc4118f51c949bdc329db7f4a0be9c53a1dbc4ccdaa585d9f06acb82f |
| `tuning/score_tuning.py` (`--stage lockbox`) | e5c698bdcde393f8b10bfc941ff82405b6c25bcf037f603e51a35ae560e326fb |
| `tuning/supp_selection.py` | e11e74763d7533c65ca543e00aa7f117179e0328b6e638b8e273254235e56822 |
| `tuning/results/supp_selection_plain/universal_choice.json` | 0b7210a980c8d42859273e801c373cc0056ae8331cb3eed7ea3789240d475d98 |
| `tuning/chosen.json` (unchanged) | 8562afd9441de8c85b850993246aa2ca8162d1c761697cefa95cb1a2a2c7479f |
| `tuning/keep.json` (unchanged) | 6a2a5862b646cde6871c9d2134a18f713beb26a7778ecdaffdecf9216bc4dc7d |
| `tuning/make_tuning_configs.py` | 177e0f9660cd4b1efa64cb2ee3091f17fd6a139997b07e2e39b57d9c8f393330 |
| `tuning/make_lockbox_universal.py` | f2e56947ea89e6f1a4b3043108fea8748d64b6ee7d20c3bc90abea1e8fc211c3 |
| `tuning/score_lockbox_hl.py` (HL1–HL4; tested only on synthetic dev data) | cebdb8af62f7130f7e8c34b939149ae8ff06221302f6f631f651cd2fec491b93 |
| `explore_cluster_bootstrap.py` (`cluster_of`, imported) | 2d684277dd27fa1a359456e6e5e98270c3880b661faf38d688d8d5ef0359f1a7 |
| `confirm_tiebreak.py` (Amendment 3, pinned) | 24ab193c48dafbcfd7df2734700060873fd94e43dae5b572c1d419e438ce26ea |

**Erratum to Amendment 4 (appended by A, 2026-10-04 ~00:25):** its heading says "~00:45"; it was appended at about 00:07 (A misread the clock). The content, the hashes in its §6 and the order of events are unchanged: appended before any lockbox configuration was generated or run.
