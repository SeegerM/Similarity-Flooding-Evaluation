# Registered replication of the dependency-role term: corpus ctl

J_tf; per scenario the mean over included matchers, mean within family, mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261012. γ fixed: 0.2 readable, 1.0 recoded. Secondary corpus: reported per regime with the same rules.

## ctl opaque (8 scenarios)

included initial matchers: 10 (datatype, jaro_winkler, jsd_values, kang_naughton, ks, levenshtein, pmf, prefix_suffix, role_gcov, sbert_mpnet)

| hypothesis | contrast | estimate [95% CI] | Lahman | MusicBrainz | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| HR1 (primary) | A_role − A_d | +0.0964 [+0.0545, +0.1480] | +0.0719 | +0.1208 | 8 of 10 | CONFIRMED |
| HR2 | A_role − melnik | +0.0995 [+0.0581, +0.1537] | +0.0802 | +0.1189 | 8 of 10 | CONFIRMED |
| HR3 | A_role − fd | +0.0567 [+0.0302, +0.0961] | +0.1005 | +0.0128 | 8 of 10 | CONFIRMED |
| HR4a | A_role − ucc_top1 (margin −0.01) | +0.0859 [+0.0467, +0.1381] | +0.0802 | +0.0916 | 8 of 10 | CONFIRMED |
| HR4b | A_role − ucc1_fd_topk (margin −0.01) | +0.0369 [+0.0094, +0.0710] | +0.0536 | +0.0201 | 8 of 10 | CONFIRMED |

## ctl plain (8 scenarios)

included initial matchers: 10 (datatype, jaro_winkler, jsd_values, kang_naughton, ks, levenshtein, pmf, prefix_suffix, role_gcov, sbert_mpnet)

| hypothesis | contrast | estimate [95% CI] | Lahman | MusicBrainz | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| HR1 (primary) | A_role − A_d | +0.0521 [+0.0297, +0.0750] | +0.0608 | +0.0433 | 4 of 10 | CONFIRMED |
| HR2 | A_role − melnik | +0.0671 [+0.0497, +0.0857] | +0.0796 | +0.0547 | 6 of 10 | CONFIRMED |
| HR3 | A_role − fd | +0.2757 [+0.2582, +0.2928] | +0.3409 | +0.2105 | 10 of 10 | CONFIRMED |
| HR4a | A_role − ucc_top1 (margin −0.01) | +0.0636 [+0.0480, +0.0814] | +0.0796 | +0.0476 | 6 of 10 | CONFIRMED |
| HR4b | A_role − ucc1_fd_topk (margin −0.01) | +0.0719 [+0.0576, +0.0842] | +0.0924 | +0.0514 | 8 of 10 | CONFIRMED |

