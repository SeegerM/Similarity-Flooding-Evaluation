# R8 (restricted propagation scope), regime plain: registered analysis

build check: fd_scopenone vs layer 1 fd on 382 scenarios, max |difference| 0.0 — passed. pruning check: scoped outputs with a non-zero pruned cell: 0 — passed.

J = (R_exp + F1_Hr) / 2. SF_s: scoped output; q_s: q with the same cells pruned, no propagation; SF: layer 1's unrestricted arm. Macro over families with ≥ 5 scenarios, mean over included matchers; 95% stratified bootstrap, 10,000 resamples, seed 20261002.

## Case set: primary (non-screen) (322 scenarios; families ChEMBL 72, Magellan 6, OpenData 94, TPC-DI 148)

| arm | scope | kept | SF_s − q_s J | SF_s − SF J | R_exp | F1_Hr | q_s − q J | tb − q J |
|---|---|---|---|---|---|---|---|---|
| melnik | q050 | 0.68 | +0.0188 [+0.0139, +0.0239] | -0.0035 [-0.0045, -0.0025] | +0.0009 [-0.0001, +0.0020] | -0.0079 [-0.0093, -0.0065] | -0.0026 [-0.0036, -0.0018] | +0.0117 [+0.0096, +0.0139] |
| melnik | q075 | 0.45 | +0.0167 [+0.0121, +0.0218] | -0.0091 [-0.0113, -0.0070] | -0.0022 [-0.0054, +0.0008] | -0.0160 [-0.0190, -0.0131] | -0.0064 [-0.0079, -0.0046] | +0.0116 [+0.0095, +0.0137] |
| melnik | q090 | 0.25 | +0.0173 [+0.0128, +0.0225] | -0.0297 [-0.0387, -0.0193] | -0.0257 [-0.0366, -0.0138] | -0.0338 [-0.0424, -0.0239] | -0.0279 [-0.0344, -0.0200] | +0.0113 [+0.0093, +0.0135] |
| melnik | top1 | 0.25 | +0.0195 [+0.0148, +0.0247] | +0.0103 [+0.0079, +0.0125] | +0.0272 [+0.0238, +0.0304] | -0.0065 [-0.0100, -0.0031] | +0.0104 [+0.0089, +0.0118] | +0.0119 [+0.0098, +0.0142] |
| melnik | top3 | 0.48 | +0.0190 [+0.0136, +0.0251] | -0.0060 [-0.0074, -0.0045] | +0.0043 [+0.0024, +0.0064] | -0.0162 [-0.0181, -0.0145] | -0.0054 [-0.0062, -0.0046] | +0.0119 [+0.0097, +0.0142] |
| fd | q050 | 0.68 | -0.0583 [-0.0772, -0.0395] | +0.0213 [+0.0170, +0.0253] | +0.0505 [+0.0434, +0.0579] | -0.0080 [-0.0115, -0.0035] | -0.0026 [-0.0036, -0.0018] | +0.0139 [+0.0121, +0.0157] |
| fd | q075 | 0.45 | -0.0313 [-0.0443, -0.0195] | +0.0453 [+0.0333, +0.0553] | +0.0988 [+0.0795, +0.1160] | -0.0083 [-0.0173, +0.0040] | -0.0064 [-0.0079, -0.0046] | +0.0132 [+0.0114, +0.0148] |
| fd | q090 | 0.25 | -0.0009 [-0.0051, +0.0031] | +0.0547 [+0.0291, +0.0780] | +0.1265 [+0.0911, +0.1578] | -0.0172 [-0.0386, +0.0102] | -0.0279 [-0.0344, -0.0200] | +0.0135 [+0.0117, +0.0154] |
| fd | top1 | 0.25 | +0.0116 [+0.0068, +0.0163] | +0.1051 [+0.0854, +0.1254] | +0.1975 [+0.1699, +0.2214] | +0.0127 [-0.0044, +0.0401] | +0.0104 [+0.0089, +0.0118] | +0.0142 [+0.0125, +0.0159] |
| fd | top3 | 0.48 | -0.0392 [-0.0601, -0.0195] | +0.0380 [+0.0330, +0.0429] | +0.0873 [+0.0768, +0.0978] | -0.0114 [-0.0153, -0.0079] | -0.0054 [-0.0062, -0.0046] | +0.0142 [+0.0124, +0.0160] |

Hypotheses:

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R8a | melnik: 0 within-block inversions among kept cells (all matchers, scopes) | 90 | +0.0000 | confirmed (0 of 823395593 ordered pairs) |
| H-R8b q050 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0009 [-0.0001, +0.0020] | confirmed |
| H-R8b q075 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | -0.0022 [-0.0054, +0.0008] | confirmed |
| H-R8b q090 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | -0.0048 [-0.0084, -0.0011] | confirmed |
| H-R8b top1 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0065 [+0.0036, +0.0096] | confirmed |
| H-R8b top3 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0041 [+0.0022, +0.0062] | confirmed |
| H-R8c(1) q090 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 82 | -0.0331 [-0.0482, -0.0204] | confirmed |
| H-R8c(1) top1 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 82 | -0.0395 [-0.0502, -0.0264] | confirmed |
| H-R8c(2) q050 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0193 [+0.0145, +0.0244] | confirmed |
| H-R8c(2) q075 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0380 [+0.0259, +0.0491] | confirmed |
| H-R8c(2) q090 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0338 [+0.0171, +0.0582] | confirmed |
| H-R8c(2) top1 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0895 [+0.0694, +0.1092] | confirmed |
| H-R8c(2) top3 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0303 [+0.0250, +0.0353] | confirmed |

## Case set: all (382 scenarios; families ChEMBL 86, Magellan 7, OpenData 111, TPC-DI 176)

| arm | scope | kept | SF_s − q_s J | SF_s − SF J | R_exp | F1_Hr | q_s − q J | tb − q J |
|---|---|---|---|---|---|---|---|---|
| melnik | q050 | 0.68 | +0.0182 [+0.0137, +0.0228] | -0.0036 [-0.0045, -0.0027] | +0.0009 [+0.0000, +0.0019] | -0.0081 [-0.0095, -0.0067] | -0.0025 [-0.0033, -0.0017] | +0.0117 [+0.0097, +0.0137] |
| melnik | q075 | 0.45 | +0.0159 [+0.0117, +0.0205] | -0.0098 [-0.0118, -0.0077] | -0.0023 [-0.0052, +0.0003] | -0.0172 [-0.0205, -0.0141] | -0.0066 [-0.0080, -0.0050] | +0.0114 [+0.0095, +0.0134] |
| melnik | q090 | 0.25 | +0.0170 [+0.0128, +0.0215] | -0.0304 [-0.0384, -0.0213] | -0.0256 [-0.0354, -0.0153] | -0.0352 [-0.0434, -0.0261] | -0.0285 [-0.0343, -0.0215] | +0.0113 [+0.0094, +0.0133] |
| melnik | top1 | 0.25 | +0.0190 [+0.0147, +0.0236] | +0.0100 [+0.0078, +0.0119] | +0.0270 [+0.0238, +0.0300] | -0.0070 [-0.0102, -0.0039] | +0.0101 [+0.0088, +0.0114] | +0.0118 [+0.0098, +0.0140] |
| melnik | top3 | 0.47 | +0.0184 [+0.0134, +0.0237] | -0.0063 [-0.0076, -0.0050] | +0.0041 [+0.0024, +0.0059] | -0.0167 [-0.0186, -0.0150] | -0.0054 [-0.0061, -0.0047] | +0.0119 [+0.0099, +0.0141] |
| fd | q050 | 0.68 | -0.0545 [-0.0725, -0.0374] | +0.0195 [+0.0147, +0.0241] | +0.0487 [+0.0417, +0.0561] | -0.0097 [-0.0140, -0.0050] | -0.0025 [-0.0033, -0.0017] | +0.0137 [+0.0121, +0.0153] |
| fd | q075 | 0.45 | -0.0297 [-0.0414, -0.0190] | +0.0411 [+0.0292, +0.0521] | +0.0954 [+0.0782, +0.1118] | -0.0132 [-0.0245, -0.0001] | -0.0066 [-0.0080, -0.0050] | +0.0127 [+0.0110, +0.0144] |
| fd | q090 | 0.25 | -0.0008 [-0.0047, +0.0031] | +0.0485 [+0.0251, +0.0715] | +0.1201 [+0.0885, +0.1491] | -0.0231 [-0.0436, +0.0033] | -0.0285 [-0.0343, -0.0215] | +0.0135 [+0.0119, +0.0152] |
| fd | top1 | 0.25 | +0.0125 [+0.0079, +0.0171] | +0.1003 [+0.0818, +0.1197] | +0.1922 [+0.1674, +0.2150] | +0.0083 [-0.0073, +0.0330] | +0.0101 [+0.0088, +0.0114] | +0.0143 [+0.0128, +0.0158] |
| fd | top3 | 0.47 | -0.0352 [-0.0549, -0.0172] | +0.0362 [+0.0314, +0.0412] | +0.0854 [+0.0758, +0.0955] | -0.0129 [-0.0165, -0.0095] | -0.0054 [-0.0061, -0.0047] | +0.0143 [+0.0127, +0.0158] |

Hypotheses:

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R8a | melnik: 0 within-block inversions among kept cells (all matchers, scopes) | 90 | +0.0000 | confirmed (0 of 965454319 ordered pairs) |
| H-R8b q050 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0009 [+0.0000, +0.0019] | confirmed |
| H-R8b q075 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | -0.0023 [-0.0052, +0.0003] | confirmed |
| H-R8b q090 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | -0.0043 [-0.0078, -0.0007] | confirmed |
| H-R8b top1 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0066 [+0.0038, +0.0093] | confirmed |
| H-R8b top3 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0039 [+0.0022, +0.0057] | confirmed |
| H-R8c(1) q090 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 82 | -0.0337 [-0.0472, -0.0224] | confirmed |
| H-R8c(1) top1 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 82 | -0.0397 [-0.0482, -0.0276] | confirmed |
| H-R8c(2) q050 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0187 [+0.0143, +0.0233] | confirmed |
| H-R8c(2) q075 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0344 [+0.0238, +0.0458] | confirmed |
| H-R8c(2) q090 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0308 [+0.0165, +0.0524] | confirmed |
| H-R8c(2) top1 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0851 [+0.0668, +0.1040] | confirmed |
| H-R8c(2) top3 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0285 [+0.0233, +0.0335] | confirmed |

## Case set: screen (60 scenarios; families ChEMBL 14, OpenData 17, TPC-DI 28)

| arm | scope | kept | SF_s − q_s J | SF_s − SF J | R_exp | F1_Hr | q_s − q J | tb − q J |
|---|---|---|---|---|---|---|---|---|
| melnik | q050 | 0.68 | +0.0142 [+0.0051, +0.0257] | -0.0027 [-0.0047, -0.0005] | +0.0009 [-0.0008, +0.0029] | -0.0062 [-0.0098, -0.0026] | -0.0022 [-0.0035, -0.0008] | +0.0124 [+0.0073, +0.0192] |
| melnik | q075 | 0.45 | +0.0130 [+0.0036, +0.0251] | -0.0070 [-0.0112, -0.0028] | -0.0008 [-0.0039, +0.0031] | -0.0133 [-0.0206, -0.0063] | -0.0052 [-0.0081, -0.0025] | +0.0125 [+0.0073, +0.0194] |
| melnik | q090 | 0.25 | +0.0171 [+0.0073, +0.0300] | -0.0032 [-0.0124, +0.0062] | -0.0018 [-0.0076, +0.0050] | -0.0046 [-0.0204, +0.0114] | -0.0059 [-0.0121, +0.0004] | +0.0124 [+0.0070, +0.0197] |
| melnik | top1 | 0.25 | +0.0144 [+0.0050, +0.0267] | +0.0119 [+0.0081, +0.0160] | +0.0280 [+0.0213, +0.0349] | -0.0042 [-0.0123, +0.0038] | +0.0120 [+0.0092, +0.0150] | +0.0124 [+0.0071, +0.0197] |
| melnik | top3 | 0.47 | +0.0130 [+0.0037, +0.0250] | -0.0088 [-0.0120, -0.0050] | +0.0043 [+0.0003, +0.0096] | -0.0219 [-0.0257, -0.0181] | -0.0071 [-0.0085, -0.0058] | +0.0125 [+0.0073, +0.0195] |
| fd | q050 | 0.68 | -0.0489 [-0.0673, -0.0317] | +0.0119 [+0.0036, +0.0223] | +0.0332 [+0.0209, +0.0508] | -0.0094 [-0.0159, -0.0032] | -0.0022 [-0.0035, -0.0008] | +0.0128 [+0.0102, +0.0155] |
| fd | q075 | 0.45 | -0.0333 [-0.0499, -0.0178] | +0.0247 [+0.0120, +0.0382] | +0.0661 [+0.0492, +0.0869] | -0.0167 [-0.0293, -0.0050] | -0.0052 [-0.0081, -0.0025] | +0.0126 [+0.0100, +0.0153] |
| fd | q090 | 0.25 | -0.0048 [-0.0182, +0.0074] | +0.0529 [+0.0341, +0.0729] | +0.1099 [+0.0876, +0.1363] | -0.0040 [-0.0245, +0.0163] | -0.0059 [-0.0121, +0.0004] | +0.0152 [+0.0124, +0.0181] |
| fd | top1 | 0.25 | +0.0070 [-0.0064, +0.0192] | +0.0826 [+0.0673, +0.0989] | +0.1670 [+0.1420, +0.1945] | -0.0018 [-0.0146, +0.0112] | +0.0120 [+0.0092, +0.0150] | +0.0142 [+0.0115, +0.0171] |
| fd | top3 | 0.47 | -0.0235 [-0.0412, -0.0070] | +0.0327 [+0.0229, +0.0438] | +0.0864 [+0.0689, +0.1068] | -0.0210 [-0.0275, -0.0148] | -0.0071 [-0.0085, -0.0058] | +0.0129 [+0.0102, +0.0157] |

Hypotheses:

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R8a | melnik: 0 within-block inversions among kept cells (all matchers, scopes) | 90 | +0.0000 | confirmed (0 of 142058726 ordered pairs) |
| H-R8b q050 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0009 [-0.0008, +0.0029] | confirmed |
| H-R8b q075 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | -0.0008 [-0.0039, +0.0031] | confirmed |
| H-R8b q090 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | -0.0018 [-0.0076, +0.0050] | confirmed |
| H-R8b top1 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0071 [+0.0023, +0.0126] | not confirmed |
| H-R8b top3 | melnik: (SF_s − q_s) − (SF − q) on R_exp within ±0.01 (interval) | 76 | +0.0040 [-0.0000, +0.0093] | confirmed |
| H-R8c(1) q090 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 82 | -0.0206 [-0.0283, -0.0132] | confirmed |
| H-R8c(1) top1 | fd: inversion share among kept cells − unrestricted share < 0 (upper bound) | 82 | -0.0464 [-0.0553, -0.0368] | confirmed |
| H-R8c(2) q050 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0098 [+0.0047, +0.0189] | confirmed |
| H-R8c(2) q075 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0161 [+0.0076, +0.0294] | confirmed |
| H-R8c(2) q090 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0448 [+0.0266, +0.0606] | confirmed |
| H-R8c(2) top1 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0659 [+0.0505, +0.0807] | confirmed |
| H-R8c(2) top3 | mean |SF_s − SF| on J: fd − melnik > 0 (lower bound) | 76 | +0.0211 [+0.0134, +0.0310] | confirmed |

## RQ5 (all scenarios of the analysed set)

| arm | scope | runs | pair-graph vertices | iterate ms | build ms |
|---|---|---|---|---|---|
| melnik | none | 33766 | 1119 | 7 | 3886 |
| melnik | q050 | 33766 | 937 | 4 | 3599 |
| melnik | q075 | 33766 | 695 | 3 | 3592 |
| melnik | q090 | 33766 | 551 | 2 | 3611 |
| melnik | top1 | 33766 | 527 | 2 | 3593 |
| melnik | top3 | 33766 | 683 | 3 | 3630 |
| fd | none | 33766 | 1119 | 53 | 14450 |
| fd | q050 | 33766 | 937 | 18 | 13206 |
| fd | q075 | 33766 | 695 | 8 | 13038 |
| fd | q090 | 33766 | 551 | 3 | 13120 |
| fd | top1 | 33766 | 527 | 3 | 13199 |
| fd | top3 | 33766 | 683 | 7 | 13002 |
- RQ5 monotone, melnik, pcg_v: held
- RQ5 monotone, melnik, iterate_ms: held
- RQ5 monotone, fd, pcg_v: held
- RQ5 monotone, fd, iterate_ms: held
