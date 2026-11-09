# X46 (plain): an additive term with dependency information against SF with dependency edges (EXPLORATORY)

76 initial matchers, 549 scenarios. Weights chosen leave-one-family-out from G = (0, 0.1, 0.2, 0.5, 1); chosen (γ, λ) per held-out family:

- role: ChEMBL (0.2, 0.0); Magellan (0.5, 0.0); OpenData (0.2, 0.0); TPC-DI (0.5, 0.0); _other (0.2, 0.0)
- nb: ChEMBL (0.0, 0.0); Magellan (0.0, 0.1); OpenData (0.0, 0.1); TPC-DI (0.0, 0.1); _other (0.0, 0.1)
- both: ChEMBL (0.2, 0.0); Magellan (0.5, 0.0); OpenData (0.2, 0.1); TPC-DI (0.5, 0.0); _other (0.2, 0.0)

| contrast | J_tf [95% CI] | matchers > 0 |
|---|---|---|
| A_role − A_d | +0.0366 [+0.0306, +0.0423] | 67 of 76 |
| A_nb − A_d | -0.0035 [-0.0054, -0.0017] | 23 of 76 |
| A_both − A_d | +0.0264 [+0.0204, +0.0319] | 64 of 76 |
| SF melnik − A_d | -0.0133 [-0.0162, -0.0101] | 8 of 76 |
| SF fd − A_d | -0.1228 [-0.1418, -0.1054] | 5 of 76 |
| SF ucc_top1 − A_d | +0.0361 [+0.0286, +0.0439] | 74 of 76 |
| SF ucc1_fd_topk − A_d | +0.0326 [+0.0219, +0.0422] | 69 of 76 |
| A_role − SF melnik | +0.0500 [+0.0432, +0.0565] | 73 of 76 |
| A_role − SF fd | +0.1595 [+0.1443, +0.1774] | 76 of 76 |
| A_role − SF ucc_top1 | +0.0005 [-0.0094, +0.0101] | 37 of 76 |
| A_role − SF ucc1_fd_topk | +0.0041 [-0.0065, +0.0158] | 40 of 76 |
| A_both − SF melnik | +0.0397 [+0.0321, +0.0471] | 66 of 76 |
| A_both − SF fd | +0.1492 [+0.1342, +0.1680] | 76 of 76 |
| A_both − SF ucc_top1 | -0.0097 [-0.0201, +0.0004] | 30 of 76 |
| A_both − SF ucc1_fd_topk | -0.0062 [-0.0168, +0.0052] | 28 of 76 |
