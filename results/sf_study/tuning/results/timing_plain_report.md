# RQ5 timing pass (plain; descriptive)

Each task (one configuration on one dataset) ran with at most 4 timing tasks and nothing else of ours on its host (scheduler `--solo 4`). Wall and CPU time are summed over a configuration's dataset tasks; peak RSS is the maximum. SFSTATS columns are per-scenario medians over the configuration's scenarios.

| configuration | tasks | wall s | CPU s | peak RSS MB | scenarios | median iterations | median PCG vertices | median PCG edges | median build ms | median iterate ms |
|---|---|---|---|---|---|---|---|---|---|---|
| kang_naughton__seedstudy | 13 | 10020 | 3620 | 2456 | 382 | 14 | 1119 | 53293 | 5898 | 18 |
| ks__seedstudy | 13 | 3396 | 3601 | 2373 | 382 | 22 | 1119 | 53293 | 6015 | 20 |
| levenshtein__attach_id__L1 | 13 | 1235 | 1411 | 2262 | 382 | 2 | 1119 | 3822 | 1744 | 2 |
| levenshtein__coef_prod__L1 | 13 | 1216 | 1394 | 2296 | 382 | 9 | 1119 | 3822 | 1722 | 2 |
| levenshtein__coef_prod_fd__L1 | 13 | 3445 | 3635 | 2466 | 382 | 7 | 1119 | 53293 | 5893 | 19 |
| levenshtein__col_fd__L3 | 13 | 2890 | 3044 | 2237 | 0 |  |  |  |  |  |
| levenshtein__col_fd_gcov__L3 | 13 | 2968 | 3128 | 2273 | 0 |  |  |  |  |  |
| levenshtein__col_fd_gcov_u__L3 | 13 | 4849 | 5027 | 2452 | 0 |  |  |  |  |  |
| levenshtein__col_fd_jaccard__L3 | 13 | 2907 | 3070 | 2231 | 0 |  |  |  |  |  |
| levenshtein__col_gpdep__L3 | 13 | 1666 | 1774 | 2232 | 0 |  |  |  |  |  |
| levenshtein__col_gpdep_gcov__L3 | 13 | 1810 | 1927 | 2383 | 0 |  |  |  |  |  |
| levenshtein__fd__L1 | 13 | 3416 | 3615 | 2325 | 382 | 5 | 1119 | 53293 | 5824 | 18 |
| levenshtein__fd_damped__L3 | 13 | 3447 | 3644 | 2413 | 382 | 8 | 1119 | 53293 | 6082 | 20 |
| levenshtein__fd_reified__L3 | 13 | 11157 | 11367 | 14094 | 382 | 13 | 66124 | 553231 | 8988 | 480 |
| levenshtein__fd_ucc_ind__L3 | 13 | 3851 | 4054 | 2434 | 382 | 5 | 1120 | 53316 | 6056 | 20 |
| levenshtein__ind__L3 | 13 | 1736 | 1917 | 2280 | 382 | 11 | 1119 | 5006 | 2259 | 2 |
| levenshtein__lit_none__L1 | 13 | 1237 | 1411 | 2273 | 382 | 17 | 1119 | 3822 | 1667 | 2 |
| levenshtein__melnik__L1 | 13 | 1237 | 1412 | 2287 | 382 | 13 | 1119 | 3822 | 1722 | 2 |
| levenshtein__seedstudy | 13 | 3385 | 3585 | 2422 | 382 | 21 | 1119 | 53293 | 6161 | 22 |
| levenshtein__ucc__L3 | 13 | 1237 | 1416 | 2295 | 382 | 13 | 1120 | 3826 | 1700 | 2 |
| pmf__attach_id__L1 | 13 | 1254 | 1435 | 2292 | 382 | 1 | 1119 | 3822 | 1673 | 2 |
| pmf__coef_prod__L1 | 13 | 1247 | 1423 | 2262 | 382 | 20 | 1119 | 3822 | 1720 | 2 |
| pmf__coef_prod_fd__L1 | 13 | 3481 | 3674 | 2389 | 382 | 16 | 1119 | 53293 | 5995 | 21 |
| pmf__col_fd__L3 | 13 | 2843 | 2995 | 2246 | 0 |  |  |  |  |  |
| pmf__col_fd_gcov__L3 | 13 | 2958 | 3120 | 2241 | 0 |  |  |  |  |  |
| pmf__col_fd_gcov_u__L3 | 13 | 4737 | 4909 | 2476 | 0 |  |  |  |  |  |
| pmf__col_fd_jaccard__L3 | 13 | 2906 | 3066 | 2232 | 0 |  |  |  |  |  |
| pmf__col_gpdep__L3 | 13 | 1681 | 1790 | 2378 | 0 |  |  |  |  |  |
| pmf__col_gpdep_gcov__L3 | 13 | 1806 | 1922 | 2476 | 0 |  |  |  |  |  |
| pmf__fd__L1 | 13 | 3372 | 3568 | 2432 | 382 | 9 | 1119 | 53293 | 5992 | 20 |
| pmf__fd_damped__L3 | 13 | 3438 | 3631 | 2439 | 382 | 14 | 1119 | 53293 | 6092 | 19 |
| pmf__fd_reified__L3 | 13 | 11772 | 11976 | 13753 | 382 | 10 | 66124 | 553231 | 8634 | 426 |
| pmf__fd_ucc_ind__L3 | 13 | 3875 | 4078 | 2441 | 382 | 9 | 1120 | 53316 | 6174 | 22 |
| pmf__ind__L3 | 13 | 1691 | 1868 | 2238 | 382 | 20 | 1119 | 5006 | 2188 | 2 |
| pmf__lit_none__L1 | 13 | 1228 | 1403 | 2271 | 382 | 29 | 1119 | 3822 | 1698 | 2 |
| pmf__melnik__L1 | 13 | 1235 | 1411 | 2295 | 382 | 22 | 1119 | 3822 | 1710 | 2 |
| pmf__seedstudy | 13 | 3711 | 3920 | 2424 | 382 | 47 | 1119 | 53293 | 6074 | 24 |
| pmf__ucc__L3 | 13 | 1253 | 1430 | 2304 | 382 | 22 | 1120 | 3826 | 1674 | 2 |
| prefix_suffix__seedstudy | 13 | 3475 | 3679 | 2293 | 382 | 17 | 1119 | 53293 | 6217 | 21 |
| sbert_mpnet__seedstudy | 13 | 3803 | 3709 | 2470 | 382 | 25 | 1119 | 53293 | 6143 | 20 |
