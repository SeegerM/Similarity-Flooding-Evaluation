# X81: Amendment 12's tests on the external corpora (EXPLORATORY)

Fixed HyFD profiles: FD arms from x81fix_<c>, role vectors from harness/generated_hyfdfix/<c> (SMX_VALENTINE=549).
Frozen terms (A_d, A_role γ 0.2, A_dt λ 0.5); 15-matcher panel; SF arms of x81_<c>. Rules as Amendment 12: E1 lower > 0; E2 lower > −0.01; E3 upper < 0; E4 upper < 0; E5' (single-table A_dt) lower > 0. Predictions: EXPLORATORY.md X81.

## gdcsm

- scenarios 10; groups ['gdcsm']; matchers included 15 (FD arms), 15 (other arms); FD-arm exclusions 0
- mean J_tf: q 0.1270, A_d 0.1187, A_role 0.1169, A_dt 0.1184, melnik 0.1246, ucc_top1 0.1241
- mean J_tf, FD arms: fd 0.0664, ucc1_fd_topk 0.1243

| test | estimate [95%] | rule met |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0059 [-0.0126, +0.0011] | no |
| E2a A_role − SF(ucc_top1) | -0.0072 [-0.0148, +0.0008] | no |
| E5' A_dt − SF(melnik) | -0.0062 [-0.0129, +0.0010] | no |
| E3 ρ(q R_exp, melnik gain R_exp) | +0.1750 [-0.3608, +0.6214] | no |
| E2b A_role − SF(ucc1_fd_topk) | -0.0073 [-0.0146, +0.0005] | no |
| E4 R_exp SF(fd) − SF(melnik) | -0.1074 [-0.1394, -0.0726] | yes |

Cluster-aware intervals (10 clusters; resampling whole clusters, 10,000 draws; X86 clustering):

| test | estimate [95% cluster] | rule met (cluster) |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0059 [-0.0126, +0.0011] | no |
| E2a A_role − SF(ucc_top1) | -0.0072 [-0.0149, +0.0009] | no |
| E5' A_dt − SF(melnik) | -0.0062 [-0.0128, +0.0007] | no |
| E2b A_role − SF(ucc1_fd_topk) | -0.0073 [-0.0146, +0.0003] | no |
| E4 R_exp SF(fd) − SF(melnik) | -0.1074 [-0.1392, -0.0718] | yes |

## madi_indep

- scenarios 12; groups ['companies', 'games', 'music', 'papers']; matchers included 15 (FD arms), 15 (other arms); FD-arm exclusions 0
- mean J_tf: q 0.4445, A_d 0.5076, A_role 0.5593, A_dt 0.5072, melnik 0.5142, ucc_top1 0.5989
- mean J_tf, FD arms: fd 0.4144, ucc1_fd_topk 0.6036

| test | estimate [95%] | rule met |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0066 [-0.0155, +0.0016] | no |
| E2a A_role − SF(ucc_top1) | -0.0396 [-0.0630, -0.0150] | no |
| E5' A_dt − SF(melnik) | -0.0070 [-0.0154, +0.0011] | no |
| E3 ρ(q R_exp, melnik gain R_exp) | -0.4786 [-0.8857, +0.1250] | no |
| E2b A_role − SF(ucc1_fd_topk) | -0.0444 [-0.0648, -0.0210] | no |
| E4 R_exp SF(fd) − SF(melnik) | -0.1313 [-0.1822, -0.0855] | yes |

Cluster-aware intervals (4 clusters; resampling whole clusters, 10,000 draws; X86 clustering):

| test | estimate [95% cluster] | rule met (cluster) |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0066 (only 4 clusters; no interval) | – |
| E2a A_role − SF(ucc_top1) | -0.0396 (only 4 clusters; no interval) | – |
| E5' A_dt − SF(melnik) | -0.0070 (only 4 clusters; no interval) | – |
| E2b A_role − SF(ucc1_fd_topk) | -0.0444 (only 4 clusters; no interval) | – |
| E4 R_exp SF(fd) − SF(melnik) | -0.1313 (only 4 clusters; no interval) | – |

## madi_products

- scenarios 6; groups ['products']; matchers included 15 (FD arms), 15 (other arms); FD-arm exclusions 0
- mean J_tf: q 0.6220, A_d 0.6829, A_role 0.7002, A_dt 0.6829, melnik 0.6528, ucc_top1 0.6586
- mean J_tf, FD arms: fd 0.4760, ucc1_fd_topk 0.6678

| test | estimate [95%] | rule met |
|---|---|---|
| E1 A_d − SF(melnik) | +0.0301 [+0.0184, +0.0411] | yes |
| E2a A_role − SF(ucc_top1) | +0.0415 [+0.0207, +0.0629] | yes |
| E5' A_dt − SF(melnik) | +0.0301 [+0.0184, +0.0412] | yes |
| E3 ρ(q R_exp, melnik gain R_exp) | -0.6607 [-0.8572, -0.2000] | yes |
| E2b A_role − SF(ucc1_fd_topk) | +0.0324 [+0.0215, +0.0423] | yes |
| E4 R_exp SF(fd) − SF(melnik) | -0.2818 [-0.3181, -0.2466] | yes |

Cluster-aware intervals (1 clusters; resampling whole clusters, 10,000 draws; X86 clustering):

| test | estimate [95% cluster] | rule met (cluster) |
|---|---|---|
| E1 A_d − SF(melnik) | +0.0301 (only 1 clusters; no interval) | – |
| E2a A_role − SF(ucc_top1) | +0.0415 (only 1 clusters; no interval) | – |
| E5' A_dt − SF(melnik) | +0.0301 (only 1 clusters; no interval) | – |
| E2b A_role − SF(ucc1_fd_topk) | +0.0324 (only 1 clusters; no interval) | – |
| E4 R_exp SF(fd) − SF(melnik) | -0.2818 (only 1 clusters; no interval) | – |

## goby

- scenarios 200; groups ['goby']; matchers included 15 (FD arms), 15 (other arms); FD-arm exclusions 0
- mean J_tf: q 0.3971, A_d 0.3945, A_role 0.3883, A_dt 0.3922, melnik 0.4077, ucc_top1 0.4081
- mean J_tf, FD arms: fd 0.3490, ucc1_fd_topk 0.4167

| test | estimate [95%] | rule met |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0133 [-0.0183, -0.0084] | no |
| E2a A_role − SF(ucc_top1) | -0.0198 [-0.0257, -0.0140] | no |
| E5' A_dt − SF(melnik) | -0.0155 [-0.0202, -0.0107] | no |
| E3 ρ(q R_exp, melnik gain R_exp) | -0.7571 [-0.8964, -0.3571] | yes |
| E2b A_role − SF(ucc1_fd_topk) | -0.0284 [-0.0349, -0.0218] | no |
| E4 R_exp SF(fd) − SF(melnik) | -0.0874 [-0.1000, -0.0747] | yes |

Cluster-aware intervals (124 clusters; resampling whole clusters, 10,000 draws; X86 clustering):

| test | estimate [95% cluster] | rule met (cluster) |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0133 [-0.0184, -0.0083] | no |
| E2a A_role − SF(ucc_top1) | -0.0198 [-0.0259, -0.0139] | no |
| E5' A_dt − SF(melnik) | -0.0155 [-0.0204, -0.0106] | no |
| E2b A_role − SF(ucc1_fd_topk) | -0.0284 [-0.0349, -0.0218] | no |
| E4 R_exp SF(fd) − SF(melnik) | -0.0874 [-0.1002, -0.0747] | yes |

## mimic

- scenarios 1; groups ['mimic']; matchers included 15 (FD arms), 15 (other arms); FD-arm exclusions 0
- mean J_tf: q 0.1395, A_d 0.1416, A_role 0.1469, A_dt 0.1461, melnik 0.1368, ucc_top1 0.1364
- mean J_tf, FD arms: fd 0.1366, ucc1_fd_topk 0.1348
- E5 (descriptive, n = 1): A_dt − SF(melnik) mean over 15 matchers +0.0093; positive for 10 of 15
- E1 A_d − SF(melnik) +0.0048; E2 A_role − SF(ucc_top1) +0.0105

## hdxsm

- scenarios 199; groups ['hdxsm']; matchers included 15 (FD arms), 15 (other arms); FD-arm exclusions 6
- mean J_tf: q 0.4686, A_d 0.4852, A_role 0.4939, A_dt 0.4840, melnik 0.4911, ucc_top1 0.4736
- mean J_tf, FD arms: fd 0.4135, ucc1_fd_topk 0.4795

| test | estimate [95%] | rule met |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0059 [-0.0094, -0.0028] | no |
| E2a A_role − SF(ucc_top1) | +0.0203 [+0.0139, +0.0270] | yes |
| E5' A_dt − SF(melnik) | -0.0071 [-0.0103, -0.0041] | no |
| E3 ρ(q R_exp, melnik gain R_exp) | -0.8857 [-0.9643, -0.5893] | yes |
| E2b A_role − SF(ucc1_fd_topk) | +0.0128 [+0.0051, +0.0203] | yes |
| E4 R_exp SF(fd) − SF(melnik) | -0.1227 [-0.1352, -0.1104] | yes |

Cluster-aware intervals (32 clusters; resampling whole clusters, 10,000 draws; X86 clustering):

| test | estimate [95% cluster] | rule met (cluster) |
|---|---|---|
| E1 A_d − SF(melnik) | -0.0059 [-0.0124, +0.0002] | no |
| E2a A_role − SF(ucc_top1) | +0.0203 [+0.0066, +0.0407] | yes |
| E5' A_dt − SF(melnik) | -0.0071 [-0.0135, -0.0016] | no |
| E2b A_role − SF(ucc1_fd_topk) | +0.0128 [-0.0053, +0.0383] | yes |
| E4 R_exp SF(fd) − SF(melnik) | -0.1227 [-0.1481, -0.0966] | yes |

