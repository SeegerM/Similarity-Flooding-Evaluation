# X88: held-out validation of "stop SF as published after two updates" (EXPLORATORY, pre-specified rule)

t2 − converged, per scenario the mean over initial matchers, macro over groups; 95% bootstrap (clusters if ≥ 5).
Non-inferior: lower bound > −0.01. Superior: lower bound > 0. Predictions: EXPLORATORY.md X88.

| corpus | matchers | scenarios | J_tf t2 − conv [95%] | verdict | R_exp t2 − conv | F1_Hr t2 − conv | matchers t2 ≥ conv (J) | median iterations to converge | interval over |
|---|---|---|---|---|---|---|---|---|---|
| T2D web tables | 9 | 175 | +0.0021 [-0.0006, +0.0051] | non-inferior | +0.0028 [-0.0015, +0.0074] | +0.0014 [-0.0013, +0.0044] | 78% | 6 | 169 table-sharing clusters |
| Lahman/MusicBrainz single-table | 10 | 16 | -0.0039 [-0.0094, +0.0015] | non-inferior | -0.0109 [-0.0234, -0.0002] | +0.0030 [+0.0003, +0.0062] | 50% | 17 | 14 table-sharing clusters |
| Synthea-OMOP | 73 | 39 | +0.0146 [+0.0084, +0.0193] | superior | +0.0271 [+0.0194, +0.0304] | +0.0020 [-0.0060, +0.0091] | 71% | 10 | 9 table-sharing clusters |
| Lahman/MusicBrainz multi-table | 73 | 54 | +0.0100 [+0.0084, +0.0119] | superior | +0.0038 [+0.0018, +0.0058] | +0.0163 [+0.0148, +0.0181] | 63% | 39 | 6 table-sharing clusters |

**Rule VALIDATED**: non-inferior on 4 of 4 corpora (superior on 2).

## Trajectory (gain over q in R_exp / J_tf; macro over groups of scenario means)

| corpus | t1 | t2 | t3 | t5 | t10 | conv |
|---|---|---|---|---|---|---|
| T2D web tables | +0.0000 / -0.0062 | +0.0204 / +0.0213 | +0.0179 / +0.0195 | +0.0176 / +0.0189 | +0.0177 / +0.0192 | +0.0176 / +0.0191 |
| Lahman/MusicBrainz single-table | +0.0000 / -0.0007 | +0.0023 / +0.0053 | +0.0081 / +0.0062 | +0.0119 / +0.0084 | +0.0134 / +0.0093 | +0.0131 / +0.0092 |
| Synthea-OMOP | +0.0000 / -0.0037 | +0.0743 / +0.0455 | +0.0374 / +0.0252 | +0.0422 / +0.0279 | +0.0512 / +0.0331 | +0.0472 / +0.0309 |
| Lahman/MusicBrainz multi-table | +0.0001 / -0.0025 | +0.0780 / +0.0914 | +0.0310 / +0.0529 | +0.0409 / +0.0604 | +0.0850 / +0.0894 | +0.0742 / +0.0814 |
