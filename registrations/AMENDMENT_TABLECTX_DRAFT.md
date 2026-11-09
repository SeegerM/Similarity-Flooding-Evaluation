# Draft amendment (Amendment 9 of PREREGISTRATION_LAYERS.md): the table-context control

**Status: draft by B, 2026-10-04, at the user's request ("prepare 1 and 2"; the external reviewer's point 2 of
17:57). Written before any table-context metric was computed; nobody has computed one. Binding once A appends it,
after the user has seen it.** Amendment 7 is reserved for E2; Amendment 8 is the class-node test.

## 1. Question

T1 (Amendment 6) showed on the independent multi-table corpus that SF beats the column-level additive baselines of
Amendment 5 only with the correct table organisation. With the columns collapsed into one table, SF is the additive
form again and loses. Those baselines see no table context. So T1 shows that SF uses the table structure, not that
iterated propagation is needed to use it.

This amendment asks: does a **non-iterative baseline that sees the same table context** explain SF's advantage? The
answer separates two explanations:
- local aggregation over tables is enough;
- repeated flooding adds beyond it.

## 2. The table-aware baseline (no iteration, no ground truth)

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

## 3. Data and matchers

- The 81 scenarios of the independent corpus (AdventureWorks, Chinook, TPC-H; plain).
- Amendment 5's frozen matcher list (`additive_indep_matchers.json`, c83fd152…).
- Included: matchers with q, D and all three SF arms finite on all 81 scenarios. At drafting, 74 of 74 have all
  outputs present (located, not read).

## 4. Statistic (Amendment 5's)

- J_tf = (R_exp + F1_Hr)/2.
- Per scenario, the mean over included matchers; then the mean within database; then the mean over the three
  databases.
- 95% percentile bootstrap over scenarios within database: 10,000 resamples, seed 20261007.
- Also reported: the number of matchers with a positive macro difference, per-database estimates, and the R_exp
  and F1_Hr parts as gains over q.

## 5. Hypotheses and decisions

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

## 6. What had been seen

- Amendment 5 (A_d, A_p, A_s against SF), ER10, X37 (anchors and profiles on this corpus), T1 and its sensitivity
  analysis.
- No metric of any table-context term has been computed. The scorer's self-test uses synthetic matrices only, and
  its input check locates the SF outputs without reading them.

## 7. Scorer

- `reference/score_tablectx.py`, SHA-256 d8e5f5f08bdc08fc…, frozen with this draft. It imports
  `score_additive_indep.py` (d815d580…) unchanged.
- **Modes:**
  - `--selftest` (synthetic; checks the table-pair term, the leave-one-database-out choice and the decision rule;
    passed);
  - `--check-inputs` (74 of 74 complete; no labels without a table prefix);
  - the registered run, which refuses unless this amendment is in `PREREGISTRATION_LAYERS.md`.
- **Output:** `tuning/results/tablectx_indep_scores/`.
- It needs no cluster run; A runs it once, after appending.
