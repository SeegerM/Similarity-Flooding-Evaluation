# Registered replication of the dependency-role term: corpus t2d

J_tf; per scenario the mean over included matchers, mean within family, mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261012. γ fixed: 0.2 readable, 1.0 recoded. Decided on t2d.

## t2d plain (175 scenarios)

included initial matchers: 9 (jaro_winkler, jsd_values, kang_naughton, ks, levenshtein, pmf, prefix_suffix, role_gcov, sbert_mpnet)

| hypothesis | contrast | estimate [95% CI] | T2D | matchers > 0 | verdict |
|---|---|---|---|---|---|
| HR1 (primary) | A_role − A_d | +0.0188 [+0.0081, +0.0298] | +0.0188 | 8 of 9 | CONFIRMED |
| HR2 | A_role − melnik | -0.0082 [-0.0272, +0.0099] | -0.0082 | 3 of 9 | not confirmed |
| HR3 | A_role − fd | +0.0228 [-0.0018, +0.0467] | +0.0228 | 6 of 9 | not confirmed |
| HR4a | A_role − ucc_top1 (margin −0.01) | +0.0376 [+0.0147, +0.0606] | +0.0376 | 9 of 9 | CONFIRMED |
| HR4b | A_role − ucc1_fd_topk (margin −0.01) | +0.0328 [+0.0092, +0.0558] | +0.0328 | 9 of 9 | CONFIRMED |

