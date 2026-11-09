# Amendment 12, part B (registered): frozen recommendations on untouched data

## Part B, plain: 48 scenarios, 4 domains, 9 matchers

| hypothesis | contrast | estimate [95% CI] | verdict |
|---|---|---|---|
| F1 | A_d − SF(melnik), J_tf | +0.0037 [+0.0004, +0.0070] | CONFIRMED |
| F2a | A_role − SF(ucc_top1), J_tf (margin −0.01) | +0.0341 [+0.0279, +0.0402] | CONFIRMED |
| F2b | A_role − SF(ucc1_fd_topk), J_tf (margin −0.01) | +0.0342 [+0.0284, +0.0401] | CONFIRMED |
| F3 | Spearman ρ(q quality, SF gain), R_exp | -0.4167 [-0.8167, +0.8500] | not confirmed |
| F4 | SF(fd) − SF(melnik), R_exp | -0.3545 [-0.3654, -0.3428] | CONFIRMED |

## Part B, opaque: 48 scenarios, 4 domains, 9 matchers

| hypothesis | contrast | estimate [95% CI] | verdict |
|---|---|---|---|
| F1 | A_d − SF(melnik), J_tf | +0.0074 [+0.0058, +0.0089] | CONFIRMED |
| F2a | A_role − SF(ucc_top1), J_tf (margin −0.01) | +0.1333 [+0.1133, +0.1533] | CONFIRMED |
| F2b | A_role − SF(ucc1_fd_topk), J_tf (margin −0.01) | +0.0862 [+0.0692, +0.1032] | CONFIRMED |
| F3 | Spearman ρ(q quality, SF gain), R_exp | +0.4333 [-0.2833, +0.9500] | not confirmed |
| F4 | SF(fd) − SF(melnik), R_exp | +0.0192 [+0.0103, +0.0281] | reported only |

