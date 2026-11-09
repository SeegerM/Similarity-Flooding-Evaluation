# Draft amendment (Amendment 8 of PREREGISTRATION_LAYERS.md): class nodes on the full panel (families 7 and 13)

**Status: draft by C, 2026-10-04 14:43. The user chose "register, then score" (A, SYNC 14:37). Written before any
`classfull_plain` or `classfull_md5` output was read; nobody has read them. Binding once A appends it, after the user
has seen it.**

## 1. Question

Do pattern-class and semantic-type nodes help SF as published on all of Valentine (single-table scenarios), beyond
what a "same class" bonus without propagation gives? On the 4-matcher screen, class nodes raised SF by +0.048 / +0.064
J_tf (X22, X23), but late fusion matched or beat them (X24). On the multi-table corpus, propagation beat the additive
terms (ER10), and T1 traced that to the table structure. Extended Proposition 1 predicts that on single tables class
nodes act as an additive bonus per (datatype, class) block.

## 2. Arms and data (already run; `servers/make_classfull_configs.py`)

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

## 3. Statistic (as `PREREGISTRATION_LAYERS.md` §3 and the R7–R8b scorers)

J_tf = (R_exp + F1_Hr)/2 (`score_screen_metrics.all_metrics`, 50 seeded tie orders).
- **Per matcher:** the macro over the families with at least 5 scenarios in the case set.
- **Then:** the mean over the included matchers, i.e. non-controls with an output in every arm of the contrast on
  every scenario of the case set.
- **Uncertainty:** 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples, seed 20261002.
- **Confirmed** if the lower bound is above 0 (equivalence: the interval inside (−0.01, +0.01)).
- **Reported:** the R_exp and F1_Hr parts, and the number of matchers with a positive macro effect.

## 4. Checks before any number is computed (the scorer refuses otherwise)

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

## 5. Hypotheses

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

## 6. What had been seen

- the screen (X22–X24, 4 matchers, 60 scenarios);
- X25/X26 (additive class terms, plain dev and MD5);
- ER10 and X37 (independent corpus);
- T1's screen;
- the class statistics of `data/classfull_*/class_stats.txt` (inputs).

**No classfull output** (matrices, scores, `sf_stats.tsv` of the classfull runs) was read. The screen's 4 matchers are
part of the panel and are not excluded; the primary set excludes the screen's scenarios.

## 7. Scorer

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
