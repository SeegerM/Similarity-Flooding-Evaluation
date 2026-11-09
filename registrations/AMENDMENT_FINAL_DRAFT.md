# Draft amendment (Amendment 12 of PREREGISTRATION_LAYERS.md): final validation of frozen recommendations on untouched data

**Status: draft by B, 2026-10-06, at the user's request.**
- The external reviewer's optional last experiment; the user chose: both parts; B drafts the truth and the user validates it;
  preregistered.
- Written before any matcher, profile or SF run on these data. Binding once A appends it, after the user has seen it.
- The plan is in FINAL_VALIDATION_PLAN.md (X76).

## 1. Question

Do the paper's recommendations, with every weight frozen, hold on data the study has not touched? This includes independently
designed schemas, where Valentine's shared-source construction does not apply.

## 2. The recommendations under test (nothing is fitted on these data)

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

## 3. Data

### Part A: independently designed schema pairs (multi-table, readable; hand-annotated truth)

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

### Part B: Valentine-style scenarios from untouched sources (single table per side; plain and opaque)

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

## 4. Panel and runs

- **Initial matchers:** E2's panel (layer-1 matchers without controls, LLM matchers or matchers needing approval).
- **Profiles:** FD/UCC/IND with the harness's standard profilers.
- **SF arms:** melnik and fd (layer-1 configs); ucc_top1 and ucc1_fd_topk (gvfull/E2x configs); unchanged.
- The additive terms (A_d, A_role, A_dt) are computed offline from the same initial matrices.
- **Included matchers per cell:** every arm on every scenario of the cell.

## 5. Statistic and decision rules

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

## 6. Predictions (B, written before any run)

- **F1** holds on Part B plain. On opaque it may fail; on Valentine md5 the datatype term was inconclusive.
- **F2** holds in both regimes.
- **F3** holds.
- **F4** holds on plain.
- **F5** is the riskiest. Independently designed schemas are where table context helped least (Synthea), and the bibliographic designs
  partition the same entities very differently. B expects the direction in at least 5 of 9 pairs, with support uncertain.

## 7. Scorer and deviations

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
