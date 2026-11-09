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
