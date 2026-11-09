# X46 (md5): an additive term with dependency information against SF with dependency edges (EXPLORATORY)

76 initial matchers, 549 scenarios. Weights chosen leave-one-family-out from G = (0, 0.1, 0.2, 0.5, 1); chosen (γ, λ) per held-out family:

- role: ChEMBL (1.0, 0.0); Magellan (1.0, 0.0); OpenData (1.0, 0.0); TPC-DI (1.0, 0.0); _other (1.0, 0.0)
- nb: ChEMBL (0.0, 0.1); Magellan (0.0, 0.1); OpenData (0.0, 0.1); TPC-DI (0.0, 0.2); _other (0.0, 0.1)
- both: ChEMBL (1.0, 0.0); Magellan (1.0, 0.0); OpenData (1.0, 0.0); TPC-DI (1.0, 0.0); _other (1.0, 0.0)

| contrast | J_tf [95% CI] | matchers > 0 |
|---|---|---|
| A_role − A_d | +0.1314 [+0.1049, +0.1589] | 71 of 76 |
| A_nb − A_d | +0.0017 [+0.0009, +0.0026] | 49 of 76 |
| A_both − A_d | +0.1314 [+0.1058, +0.1593] | 71 of 76 |
| SF melnik − A_d | -0.0006 [-0.0010, -0.0003] | 18 of 76 |
| SF fd − A_d | +0.0443 [+0.0218, +0.0650] | 66 of 76 |
| SF ucc_top1 − A_d | +0.1113 [+0.0997, +0.1237] | 76 of 76 |
| A_role − SF melnik | +0.1321 [+0.1061, +0.1594] | 71 of 76 |
| A_role − SF fd | +0.0871 [+0.0663, +0.1099] | 76 of 76 |
| A_role − SF ucc_top1 | +0.0201 [-0.0043, +0.0454] | 60 of 76 |
| A_both − SF melnik | +0.1321 [+0.1052, +0.1590] | 71 of 76 |
| A_both − SF fd | +0.0871 [+0.0661, +0.1099] | 76 of 76 |
| A_both − SF ucc_top1 | +0.0201 [-0.0051, +0.0461] | 60 of 76 |
