# Registered replication of the dependency-role term: corpus t2d

J_tf; per scenario the mean over included matchers, mean within family, mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261012. γ fixed: 0.2 readable, 1.0 recoded. Decided on t2d.

## t2d plain (175 scenarios)

included initial matchers: 9 (jaro_winkler, jsd_values, kang_naughton, ks, levenshtein, pmf, prefix_suffix, role_gcov, sbert_mpnet)

| hypothesis | contrast | estimate [95% CI] | T2D | matchers > 0 | verdict |
|---|---|---|---|---|---|
| HR1 (primary) | A_role − A_d | +0.0175 [+0.0071, +0.0282] | +0.0175 | 8 of 9 | CONFIRMED |
| HR2 | A_role − melnik | -0.0095 [-0.0281, +0.0082] | -0.0095 | 3 of 9 | not confirmed |
| HR3 | A_role − fd | +0.0215 [-0.0031, +0.0453] | +0.0215 | 6 of 9 | not confirmed |
| HR4a | A_role − ucc_top1 (margin −0.01) | +0.0364 [+0.0136, +0.0596] | +0.0364 | 9 of 9 | CONFIRMED |
| HR4b | A_role − ucc1_fd_topk (margin −0.01) | +0.0316 [+0.0081, +0.0550] | +0.0316 | 9 of 9 | CONFIRMED |

