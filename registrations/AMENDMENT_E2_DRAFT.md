### Draft amendment (PREREGISTRATION_LAYERS.md, Amendment 7): external replication of the headline predictions on dependency-profiled corpora

Written by B on 2026-10-04 at the user's request ("yes to E2", backlog E2). **A reviews it, fills in the hashes
of the configuration generator and appends it before any initial-matcher or SF run on these corpora.**

#### 1. Why, and what has been seen
- **Why:** Valentine's 382 scenarios come from about 10 independent sources (X18, X35). The independent
  multi-table corpus has three databases and no dependency profiles. So the dependency claims have never been
  tested outside Valentine.
- **What has been seen on these corpora:** only file listings, the profile status files (controlled: complete;
  T2D: none) and the scenario names. No initial matcher or SF has run on them in this study, and no metric
  exists.

#### 2. Corpora
- **ctl:** `harness/generated/controlled-imported-v3`.
  - Lahman and MusicBrainz, each with the 4 Valentine relationship types in two evidence regimes: plain, and
    opaque (MusicBrainz: independent keyed recoding; Lahman: monotone numeric transforms, categoricals
    recoded). 16 single-table scenarios, FD/UCC/IND profiles complete.
  - The opaque regime is the external counterpart of MD5.
- **t2d** (stage 2): `harness/generated/t2d-sm-wh`, 178 single-table web-table scenarios. It enters only if A
  materialises its FD/UCC/IND profiles with the same harness pipeline **before any SF run** on it. Otherwise
  P1, P3 and P4 are decided on ctl-plain alone and reported as such.

#### 3. Initial matchers and arms (frozen now)
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

#### 4. Predictions (frozen from Valentine), statistic, decision
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

#### 5. Code
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
