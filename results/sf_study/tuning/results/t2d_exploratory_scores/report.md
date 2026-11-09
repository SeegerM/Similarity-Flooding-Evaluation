> EXPLORATORY: the frozen E2 scorer run on A's profiled re-import of T2D (harness/generated/t2d-usable). E2's registered verdicts are on ctl only. The additive term skips scenarios whose initial matrix is not finite (as the SF rows already do).

# External replication (E2), corpus t2d: EXPLORATORY replication (not registered; corpus t2d-usable)

| prediction | matchers | estimate | 95% CI | per family | verdict |
|---|---|---|---|---|---|
| P1 ρ(initial R_exp, ΔR_exp melnik) (plain) | 9 | -0.633 | [-0.750, -0.417] | | consistent with the prediction |
| **P2** fd − melnik (plain, R_exp) | 9 | -0.0380 | [-0.0594, -0.0163] | T2D -0.0380 | consistent with the prediction |
| **P3** additive_d − melnik (plain, J) | 9 | -0.0270 | [-0.0418, -0.0132] | T2D -0.0270 | not consistent |
| **P4** fd_top1 − fd (plain, R_exp) | 9 | +0.0424 | [+0.0282, +0.0567] | T2D +0.0424 | consistent with the prediction |

Reported, not tested: coef_prod_fd − melnik and ucc − melnik (J_tf), per regime.
| coef_prod_fd − melnik (plain, J) | 9 | -0.0078 | [-0.0194, +0.0039] | T2D -0.0078 | (descriptive) |
| ucc − melnik (plain, J) | 9 | -0.0401 | [-0.0576, -0.0237] | T2D -0.0401 | (descriptive) |
