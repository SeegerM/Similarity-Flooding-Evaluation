# R8 (restricted propagation scope), regime md5: registered analysis

build check: fd_scopenone vs layer 1 fd on 382 scenarios, max |difference| 0.0 — passed. pruning check: scoped outputs with a non-zero pruned cell: 0 — passed.

J = (R_exp + F1_Hr) / 2. SF_s: scoped output; q_s: q with the same cells pruned, no propagation; SF: layer 1's unrestricted arm. Macro over families with ≥ 5 scenarios, mean over included matchers; 95% stratified bootstrap, 10,000 resamples, seed 20261002. No tie-breaker on MD5 (Amendment 3).

## Case set: primary (non-screen) (322 scenarios; families ChEMBL 72, Magellan 6, OpenData 94, TPC-DI 148)

| arm | scope | kept | SF_s − q_s J | SF_s − SF J | R_exp | F1_Hr | q_s − q J |
|---|---|---|---|---|---|---|---|
| melnik | q050 | 0.65 | +0.0096 [+0.0084, +0.0107] | -0.0024 [-0.0047, -0.0002] | +0.0012 [+0.0008, +0.0017] | -0.0060 [-0.0107, -0.0016] | -0.0025 [-0.0049, -0.0002] |
| melnik | q075 | 0.45 | +0.0094 [+0.0084, +0.0103] | -0.0060 [-0.0076, -0.0044] | +0.0017 [+0.0012, +0.0022] | -0.0136 [-0.0168, -0.0104] | -0.0059 [-0.0076, -0.0043] |
| melnik | q090 | 0.31 | +0.0085 [+0.0077, +0.0094] | -0.0166 [-0.0222, -0.0113] | -0.0042 [-0.0089, +0.0001] | -0.0290 [-0.0367, -0.0220] | -0.0157 [-0.0211, -0.0104] |
| melnik | top1 | 0.30 | +0.0088 [+0.0078, +0.0098] | -0.0036 [-0.0054, -0.0018] | +0.0023 [+0.0007, +0.0041] | -0.0094 [-0.0131, -0.0058] | -0.0029 [-0.0048, -0.0012] |
| melnik | top3 | 0.45 | +0.0098 [+0.0085, +0.0113] | +0.0003 [-0.0012, +0.0021] | +0.0014 [+0.0008, +0.0019] | -0.0007 [-0.0037, +0.0028] | -0.0001 [-0.0013, +0.0013] |
| fd | q050 | 0.65 | +0.0395 [+0.0174, +0.0583] | -0.0198 [-0.0240, -0.0149] | -0.0076 [-0.0175, +0.0036] | -0.0320 [-0.0444, -0.0213] | -0.0025 [-0.0049, -0.0002] |
| fd | q075 | 0.45 | +0.0316 [+0.0221, +0.0408] | -0.0312 [-0.0436, -0.0186] | -0.0100 [-0.0240, +0.0051] | -0.0523 [-0.0718, -0.0342] | -0.0059 [-0.0076, -0.0043] |
| fd | q090 | 0.31 | +0.0232 [+0.0185, +0.0278] | -0.0493 [-0.0680, -0.0309] | -0.0209 [-0.0344, -0.0077] | -0.0778 [-0.1070, -0.0502] | -0.0157 [-0.0211, -0.0104] |
| fd | top1 | 0.30 | +0.0280 [+0.0216, +0.0351] | -0.0317 [-0.0441, -0.0164] | -0.0038 [-0.0180, +0.0117] | -0.0596 [-0.0819, -0.0385] | -0.0029 [-0.0048, -0.0012] |
| fd | top3 | 0.45 | +0.0401 [+0.0217, +0.0570] | -0.0168 [-0.0211, -0.0130] | -0.0084 [-0.0158, -0.0021] | -0.0252 [-0.0298, -0.0211] | -0.0001 [-0.0013, +0.0013] |

Hypotheses:

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R8a | melnik: 0 within-block inversions among kept cells (all matchers, scopes) | 87 | +0.0000 | confirmed (0 of 4322167247 ordered pairs) |
| H-R8b q050 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0012 [+0.0008, +0.0017] | confirmed |
| H-R8b q075 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0017 [+0.0012, +0.0022] | confirmed |
| H-R8b q090 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0011 [+0.0007, +0.0016] | confirmed |
| H-R8b top1 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0014 [+0.0009, +0.0019] | confirmed |
| H-R8b top3 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0015 [+0.0010, +0.0020] | confirmed |
| H-R8c(1) q090 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 79 | -0.0250 [-0.0450, -0.0045] | confirmed |
| H-R8c(1) top1 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 79 | -0.0475 [-0.0574, -0.0362] | confirmed |
| H-R8c(2) q050 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0157 [+0.0096, +0.0206] | confirmed |
| H-R8c(2) q075 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0272 [+0.0156, +0.0371] | confirmed |
| H-R8c(2) q090 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0335 [+0.0156, +0.0455] | confirmed |
| H-R8c(2) top1 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0399 [+0.0265, +0.0492] | confirmed |
| H-R8c(2) top3 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0183 [+0.0135, +0.0229] | confirmed |

## Case set: all (382 scenarios; families ChEMBL 86, Magellan 7, OpenData 111, TPC-DI 176)

| arm | scope | kept | SF_s − q_s J | SF_s − SF J | R_exp | F1_Hr | q_s − q J |
|---|---|---|---|---|---|---|---|
| melnik | q050 | 0.65 | +0.0096 [+0.0086, +0.0107] | -0.0022 [-0.0042, -0.0003] | +0.0012 [+0.0008, +0.0016] | -0.0055 [-0.0097, -0.0018] | -0.0023 [-0.0045, -0.0004] |
| melnik | q075 | 0.45 | +0.0093 [+0.0084, +0.0101] | -0.0060 [-0.0074, -0.0046] | +0.0016 [+0.0012, +0.0021] | -0.0136 [-0.0164, -0.0108] | -0.0058 [-0.0074, -0.0044] |
| melnik | q090 | 0.31 | +0.0084 [+0.0077, +0.0092] | -0.0169 [-0.0218, -0.0122] | -0.0041 [-0.0083, -0.0003] | -0.0296 [-0.0364, -0.0231] | -0.0158 [-0.0207, -0.0112] |
| melnik | top1 | 0.30 | +0.0087 [+0.0078, +0.0096] | -0.0038 [-0.0053, -0.0022] | +0.0017 [+0.0001, +0.0035] | -0.0092 [-0.0125, -0.0061] | -0.0030 [-0.0047, -0.0015] |
| melnik | top3 | 0.45 | +0.0099 [+0.0086, +0.0112] | +0.0005 [-0.0010, +0.0021] | +0.0013 [+0.0008, +0.0018] | -0.0003 [-0.0032, +0.0027] | +0.0000 [-0.0011, +0.0011] |
| fd | q050 | 0.65 | +0.0424 [+0.0224, +0.0585] | -0.0222 [-0.0280, -0.0165] | -0.0112 [-0.0217, +0.0004] | -0.0333 [-0.0441, -0.0236] | -0.0023 [-0.0045, -0.0004] |
| fd | q075 | 0.45 | +0.0337 [+0.0248, +0.0418] | -0.0344 [-0.0459, -0.0223] | -0.0128 [-0.0255, +0.0014] | -0.0560 [-0.0733, -0.0386] | -0.0058 [-0.0074, -0.0044] |
| fd | q090 | 0.31 | +0.0252 [+0.0202, +0.0299] | -0.0530 [-0.0697, -0.0352] | -0.0238 [-0.0365, -0.0108] | -0.0822 [-0.1084, -0.0564] | -0.0158 [-0.0207, -0.0112] |
| fd | top1 | 0.30 | +0.0297 [+0.0233, +0.0362] | -0.0356 [-0.0477, -0.0205] | -0.0085 [-0.0225, +0.0073] | -0.0628 [-0.0818, -0.0430] | -0.0030 [-0.0047, -0.0015] |
| fd | top3 | 0.45 | +0.0443 [+0.0263, +0.0599] | -0.0180 [-0.0219, -0.0143] | -0.0100 [-0.0165, -0.0037] | -0.0261 [-0.0302, -0.0224] | +0.0000 [-0.0011, +0.0011] |

Hypotheses:

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R8a | melnik: 0 within-block inversions among kept cells (all matchers, scopes) | 87 | +0.0000 | confirmed (0 of 5057428149 ordered pairs) |
| H-R8b q050 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0012 [+0.0008, +0.0016] | confirmed |
| H-R8b q075 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0016 [+0.0012, +0.0021] | confirmed |
| H-R8b q090 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0010 [+0.0006, +0.0014] | confirmed |
| H-R8b top1 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0012 [+0.0008, +0.0016] | confirmed |
| H-R8b top3 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0014 [+0.0010, +0.0019] | confirmed |
| H-R8c(1) q090 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 79 | -0.0280 [-0.0459, -0.0091] | confirmed |
| H-R8c(1) top1 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 79 | -0.0435 [-0.0544, -0.0321] | confirmed |
| H-R8c(2) q050 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0190 [+0.0116, +0.0268] | confirmed |
| H-R8c(2) q075 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0314 [+0.0186, +0.0417] | confirmed |
| H-R8c(2) q090 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0369 [+0.0194, +0.0487] | confirmed |
| H-R8c(2) top1 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0429 [+0.0295, +0.0517] | confirmed |
| H-R8c(2) top3 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0197 [+0.0147, +0.0241] | confirmed |

## Case set: screen (60 scenarios; families ChEMBL 14, OpenData 17, TPC-DI 28)

| arm | scope | kept | SF_s − q_s J | SF_s − SF J | R_exp | F1_Hr | q_s − q J |
|---|---|---|---|---|---|---|---|
| melnik | q050 | 0.65 | +0.0082 [+0.0054, +0.0108] | -0.0002 [-0.0011, +0.0006] | +0.0017 [+0.0010, +0.0025] | -0.0022 [-0.0035, -0.0009] | -0.0008 [-0.0014, -0.0002] |
| melnik | q075 | 0.45 | +0.0075 [+0.0050, +0.0098] | -0.0021 [-0.0034, -0.0008] | +0.0018 [+0.0009, +0.0028] | -0.0060 [-0.0088, -0.0033] | -0.0020 [-0.0032, -0.0008] |
| melnik | q090 | 0.31 | +0.0067 [+0.0043, +0.0089] | -0.0042 [-0.0063, -0.0020] | +0.0010 [+0.0004, +0.0016] | -0.0093 [-0.0137, -0.0050] | -0.0033 [-0.0053, -0.0013] |
| melnik | top1 | 0.30 | +0.0065 [+0.0040, +0.0088] | -0.0023 [-0.0039, -0.0005] | +0.0013 [-0.0000, +0.0026] | -0.0058 [-0.0089, -0.0027] | -0.0012 [-0.0027, +0.0003] |
| melnik | top3 | 0.45 | +0.0074 [+0.0047, +0.0098] | -0.0018 [-0.0030, -0.0006] | +0.0013 [+0.0004, +0.0023] | -0.0049 [-0.0070, -0.0028] | -0.0016 [-0.0026, -0.0006] |
| fd | q050 | 0.65 | +0.0326 [+0.0217, +0.0432] | -0.0110 [-0.0147, -0.0073] | -0.0085 [-0.0125, -0.0045] | -0.0134 [-0.0179, -0.0091] | -0.0008 [-0.0014, -0.0002] |
| fd | q075 | 0.45 | +0.0249 [+0.0162, +0.0335] | -0.0198 [-0.0262, -0.0135] | -0.0111 [-0.0172, -0.0049] | -0.0286 [-0.0366, -0.0206] | -0.0020 [-0.0032, -0.0008] |
| fd | q090 | 0.31 | +0.0208 [+0.0144, +0.0270] | -0.0252 [-0.0347, -0.0157] | -0.0096 [-0.0178, -0.0013] | -0.0409 [-0.0526, -0.0292] | -0.0033 [-0.0053, -0.0013] |
| fd | top1 | 0.30 | +0.0181 [+0.0115, +0.0243] | -0.0258 [-0.0345, -0.0171] | -0.0100 [-0.0176, -0.0027] | -0.0417 [-0.0527, -0.0306] | -0.0012 [-0.0027, +0.0003] |
| fd | top3 | 0.45 | +0.0212 [+0.0129, +0.0291] | -0.0231 [-0.0303, -0.0160] | -0.0141 [-0.0211, -0.0072] | -0.0321 [-0.0407, -0.0238] | -0.0016 [-0.0026, -0.0006] |

Hypotheses:

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R8a | melnik: 0 within-block inversions among kept cells (all matchers, scopes) | 87 | +0.0000 | confirmed (0 of 735260902 ordered pairs) |
| H-R8b q050 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0017 [+0.0010, +0.0025] | confirmed |
| H-R8b q075 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0018 [+0.0009, +0.0028] | confirmed |
| H-R8b q090 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0010 [+0.0004, +0.0016] | confirmed |
| H-R8b top1 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0006 [-0.0004, +0.0015] | confirmed |
| H-R8b top3 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 75 | +0.0016 [+0.0007, +0.0026] | confirmed |
| H-R8c(1) q090 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 79 | +0.0116 [+0.0039, +0.0188] | not confirmed |
| H-R8c(1) top1 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 79 | -0.0498 [-0.0543, -0.0453] | confirmed |
| H-R8c(2) q050 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0106 [+0.0067, +0.0140] | confirmed |
| H-R8c(2) q075 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0218 [+0.0159, +0.0271] | confirmed |
| H-R8c(2) q090 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0297 [+0.0222, +0.0365] | confirmed |
| H-R8c(2) top1 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0328 [+0.0243, +0.0403] | confirmed |
| H-R8c(2) top3 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 75 | +0.0229 [+0.0163, +0.0290] | confirmed |

## RQ5 (all scenarios of the analysed set)

| arm | scope | runs | pair-graph vertices | iterate ms | build ms |
|---|---|---|---|---|---|
| melnik | none | 32969 | 1074 | 3 | 5249 |
| melnik | q050 | 32969 | 735 | 3 | 5523 |
| melnik | q075 | 32969 | 561 | 2 | 5528 |
| melnik | q090 | 32969 | 425 | 2 | 5466 |
| melnik | top1 | 32969 | 408 | 2 | 5491 |
| melnik | top3 | 32969 | 537 | 2 | 5476 |
| fd | none | 32969 | 1074 | 43 | 17114 |
| fd | q050 | 32969 | 735 | 16 | 15357 |
| fd | q075 | 32969 | 561 | 7 | 15494 |
| fd | q090 | 32969 | 425 | 3 | 15538 |
| fd | top1 | 32969 | 408 | 2 | 15477 |
| fd | top3 | 32969 | 537 | 6 | 15537 |
- RQ5 monotone, melnik, pcg_v: held
- RQ5 monotone, melnik, iterate_ms: held
- RQ5 monotone, fd, pcg_v: held
- RQ5 monotone, fd, iterate_ms: held
