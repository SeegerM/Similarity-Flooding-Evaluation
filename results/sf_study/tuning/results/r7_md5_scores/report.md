# R7 (Amendment 2), regime md5: registered analysis

pipeline check: fd_same vs layer 1 fd on 382 scenarios, max |difference| 0.0 — passed.

J = (R_exp + F1_Hr) / 2. Effects are paired per scenario, macro over families with ≥ 5 scenarios, then the mean over included matchers (non-controls with every scenario in every arm of the contrast). 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples, seed 20261002. `thin` = mean of the three random draws. No tie-breaker on MD5 (Amendment 3 blinding).

## Case set: primary (non-screen) (322 scenarios; families ChEMBL 72, Magellan 6, OpenData 94, TPC-DI 148)

Arm effects on SF's output (estimate [95% CI]):

| Contrast | matchers | J | R_exp | F1_Hr | MRR_exp | F1_M |
|---|---|---|---|---|---|---|
| fd_closure − fd | 75 | +0.0009 [-0.0001, +0.0018] | +0.0010 [-0.0004, +0.0023] | +0.0008 [-0.0002, +0.0018] | -0.0011 [-0.0018, -0.0005] | -0.0010 [-0.0025, +0.0005] |
| fd_cover − fd | 75 | +0.0008 [-0.0104, +0.0103] | -0.0044 [-0.0132, +0.0041] | +0.0059 [-0.0101, +0.0211] | -0.0124 [-0.0266, +0.0003] | -0.0196 [-0.0330, -0.0071] |
| fd_thin − fd | 75 | -0.0112 [-0.0188, -0.0051] | -0.0021 [-0.0085, +0.0039] | -0.0203 [-0.0301, -0.0132] | +0.0050 [-0.0026, +0.0127] | -0.0313 [-0.0438, -0.0188] |
| fd_reified_closure − fd_reified | 75 | +0.0023 [-0.0004, +0.0065] | -0.0008 [-0.0033, +0.0011] | +0.0055 [-0.0014, +0.0161] | -0.0004 [-0.0023, +0.0012] | +0.0015 [-0.0017, +0.0057] |
| fd_reified_cover − fd_reified | 75 | -0.0031 [-0.0120, +0.0063] | -0.0029 [-0.0099, +0.0048] | -0.0033 [-0.0182, +0.0143] | -0.0431 [-0.0603, -0.0265] | -0.0215 [-0.0421, +0.0008] |
| fd_reified_thin − fd_reified | 75 | -0.0166 [-0.0221, -0.0115] | -0.0045 [-0.0106, +0.0014] | -0.0286 [-0.0424, -0.0188] | -0.0258 [-0.0339, -0.0179] | -0.0320 [-0.0468, -0.0177] |
| fd_cover − fd_thin | 75 | +0.0120 [+0.0003, +0.0226] | -0.0023 [-0.0113, +0.0069] | +0.0263 [+0.0087, +0.0439] | -0.0174 [-0.0278, -0.0077] | +0.0117 [-0.0017, +0.0265] |
| fd_reified_cover − fd_reified_thin | 75 | +0.0134 [+0.0080, +0.0205] | +0.0016 [-0.0078, +0.0101] | +0.0253 [+0.0137, +0.0373] | -0.0173 [-0.0288, -0.0071] | +0.0105 [-0.0037, +0.0269] |
| fd − melnik | 75 | +0.0474 [+0.0275, +0.0652] | +0.0190 [+0.0005, +0.0367] | +0.0757 [+0.0466, +0.1068] | +0.0393 [+0.0271, +0.0514] | +0.0374 [+0.0047, +0.0635] |
| fd_reified − melnik | 75 | +0.0509 [+0.0310, +0.0713] | +0.0204 [+0.0056, +0.0357] | +0.0814 [+0.0503, +0.1156] | +0.0732 [+0.0584, +0.0878] | +0.0471 [+0.0134, +0.0826] |

Hypotheses (Amendment 2, §4):

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R7a | mean |effect of fd_closure on J_tf| < 0.005 (upper bound) | 75 | +0.0014 [+0.0008, +0.0033] | confirmed |
| H-R7b (QUICK) | mean |R_exp effect| cover > closure (lower bound > 0) | 75 | +0.0066 [+0.0049, +0.0130] | confirmed |
| H-R7b (reified) | mean |R_exp effect| cover > closure (lower bound > 0) | 75 | +0.0109 [+0.0080, +0.0184] | confirmed |
| H-R7c | inversion share cover > fd (lower bound > 0) | 79 | -0.0791 [-0.0885, -0.0702] | not confirmed |
| H-R7c, family form | increase largest on TPC-DI (point estimate) | 79 | -0.0617 | not confirmed (largest: ChEMBL; TPC-DI largest in 7.8% of resamples; ChEMBL -0.0547, Magellan -0.0599, OpenData -0.1342, TPC-DI -0.0617) |
| H-R7c (reified, secondary) | inversion share cover > fd (lower bound > 0) | 79 | -0.0906 [-0.0965, -0.0837] | not confirmed |
| H-R7c (reified, secondary), family form | increase largest on TPC-DI (point estimate) | 79 | -0.1579 | not confirmed (largest: ChEMBL; TPC-DI largest in 0.0% of resamples; ChEMBL -0.0407, Magellan -0.0574, OpenData -0.1011, TPC-DI -0.1579) |

Reported per matcher only (controls or partial coverage): constant, datatype, embedding_random, oracle, oracle_decoy100, oracle_erase050, oracle_noise025, oracle_noise050, prisma, random, rema, rema_shuffled.

## Case set: all (382 scenarios; families ChEMBL 86, Magellan 7, OpenData 111, TPC-DI 176)

Arm effects on SF's output (estimate [95% CI]):

| Contrast | matchers | J | R_exp | F1_Hr | MRR_exp | F1_M |
|---|---|---|---|---|---|---|
| fd_closure − fd | 75 | +0.0009 [+0.0001, +0.0017] | +0.0010 [-0.0002, +0.0022] | +0.0008 [-0.0001, +0.0017] | -0.0010 [-0.0016, -0.0005] | -0.0008 [-0.0022, +0.0006] |
| fd_cover − fd | 75 | +0.0001 [-0.0096, +0.0083] | -0.0021 [-0.0111, +0.0067] | +0.0024 [-0.0119, +0.0168] | -0.0105 [-0.0254, +0.0026] | -0.0223 [-0.0345, -0.0102] |
| fd_thin − fd | 75 | -0.0152 [-0.0245, -0.0070] | -0.0063 [-0.0154, +0.0017] | -0.0241 [-0.0344, -0.0150] | +0.0049 [-0.0026, +0.0125] | -0.0319 [-0.0428, -0.0211] |
| fd_reified_closure − fd_reified | 75 | +0.0021 [-0.0003, +0.0057] | -0.0005 [-0.0026, +0.0012] | +0.0047 [-0.0013, +0.0137] | -0.0003 [-0.0019, +0.0010] | +0.0010 [-0.0017, +0.0046] |
| fd_reified_cover − fd_reified | 75 | -0.0049 [-0.0129, +0.0042] | -0.0027 [-0.0089, +0.0039] | -0.0071 [-0.0208, +0.0094] | -0.0387 [-0.0548, -0.0245] | -0.0228 [-0.0407, -0.0029] |
| fd_reified_thin − fd_reified | 75 | -0.0201 [-0.0282, -0.0130] | -0.0084 [-0.0176, -0.0003] | -0.0318 [-0.0441, -0.0211] | -0.0237 [-0.0309, -0.0169] | -0.0321 [-0.0448, -0.0193] |
| fd_cover − fd_thin | 75 | +0.0153 [+0.0031, +0.0262] | +0.0042 [-0.0085, +0.0189] | +0.0264 [+0.0111, +0.0419] | -0.0154 [-0.0257, -0.0067] | +0.0096 [-0.0025, +0.0234] |
| fd_reified_cover − fd_reified_thin | 75 | +0.0152 [+0.0092, +0.0219] | +0.0057 [-0.0043, +0.0158] | +0.0246 [+0.0148, +0.0350] | -0.0151 [-0.0262, -0.0058] | +0.0093 [-0.0030, +0.0238] |
| fd − melnik | 75 | +0.0529 [+0.0331, +0.0699] | +0.0238 [+0.0057, +0.0402] | +0.0820 [+0.0540, +0.1095] | +0.0400 [+0.0288, +0.0512] | +0.0421 [+0.0119, +0.0651] |
| fd_reified − melnik | 75 | +0.0559 [+0.0361, +0.0745] | +0.0250 [+0.0096, +0.0399] | +0.0869 [+0.0575, +0.1160] | +0.0720 [+0.0582, +0.0853] | +0.0528 [+0.0197, +0.0846] |

Hypotheses (Amendment 2, §4):

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R7a | mean |effect of fd_closure on J_tf| < 0.005 (upper bound) | 75 | +0.0014 [+0.0009, +0.0030] | confirmed |
| H-R7b (QUICK) | mean |R_exp effect| cover > closure (lower bound > 0) | 75 | +0.0060 [+0.0042, +0.0121] | confirmed |
| H-R7b (reified) | mean |R_exp effect| cover > closure (lower bound > 0) | 75 | +0.0116 [+0.0088, +0.0182] | confirmed |
| H-R7c | inversion share cover > fd (lower bound > 0) | 79 | -0.0763 [-0.0855, -0.0681] | not confirmed |
| H-R7c, family form | increase largest on TPC-DI (point estimate) | 79 | -0.0602 | not confirmed (largest: Magellan; TPC-DI largest in 5.1% of resamples; ChEMBL -0.0531, Magellan -0.0518, OpenData -0.1341, TPC-DI -0.0602) |
| H-R7c (reified, secondary) | inversion share cover > fd (lower bound > 0) | 79 | -0.0892 [-0.0950, -0.0831] | not confirmed |
| H-R7c (reified, secondary), family form | increase largest on TPC-DI (point estimate) | 79 | -0.1583 | not confirmed (largest: ChEMBL; TPC-DI largest in 0.0% of resamples; ChEMBL -0.0395, Magellan -0.0519, OpenData -0.1015, TPC-DI -0.1583) |

Reported per matcher only (controls or partial coverage): constant, datatype, embedding_random, oracle, oracle_decoy100, oracle_erase050, oracle_noise025, oracle_noise050, prisma, random, rema, rema_shuffled.

## Case set: screen (60 scenarios; families ChEMBL 14, OpenData 17, TPC-DI 28)

Arm effects on SF's output (estimate [95% CI]):

| Contrast | matchers | J | R_exp | F1_Hr | MRR_exp | F1_M |
|---|---|---|---|---|---|---|
| fd_closure − fd | 75 | +0.0014 [+0.0001, +0.0026] | +0.0015 [-0.0004, +0.0036] | +0.0012 [-0.0015, +0.0041] | -0.0007 [-0.0019, +0.0005] | +0.0005 [-0.0039, +0.0054] |
| fd_cover − fd | 75 | +0.0009 [-0.0087, +0.0106] | +0.0006 [-0.0111, +0.0124] | +0.0011 [-0.0100, +0.0118] | -0.0189 [-0.0457, +0.0032] | -0.0161 [-0.0278, -0.0044] |
| fd_thin − fd | 75 | -0.0174 [-0.0255, -0.0092] | -0.0105 [-0.0209, +0.0015] | -0.0242 [-0.0344, -0.0145] | -0.0070 [-0.0198, +0.0036] | -0.0209 [-0.0301, -0.0107] |
| fd_reified_closure − fd_reified | 75 | +0.0008 [-0.0012, +0.0027] | +0.0016 [-0.0003, +0.0039] | -0.0000 [-0.0035, +0.0034] | +0.0002 [-0.0017, +0.0019] | -0.0023 [-0.0050, +0.0001] |
| fd_reified_cover − fd_reified | 75 | -0.0079 [-0.0155, -0.0007] | +0.0074 [-0.0010, +0.0164] | -0.0233 [-0.0330, -0.0138] | -0.0163 [-0.0441, +0.0064] | -0.0271 [-0.0466, -0.0101] |
| fd_reified_thin − fd_reified | 75 | -0.0227 [-0.0296, -0.0165] | -0.0060 [-0.0118, -0.0006] | -0.0395 [-0.0492, -0.0304] | -0.0068 [-0.0209, +0.0050] | -0.0279 [-0.0460, -0.0125] |
| fd_cover − fd_thin | 75 | +0.0182 [+0.0084, +0.0278] | +0.0111 [-0.0020, +0.0227] | +0.0253 [+0.0143, +0.0364] | -0.0119 [-0.0293, +0.0029] | +0.0047 [-0.0073, +0.0166] |
| fd_reified_cover − fd_reified_thin | 75 | +0.0148 [+0.0098, +0.0198] | +0.0134 [+0.0067, +0.0203] | +0.0162 [+0.0091, +0.0229] | -0.0096 [-0.0249, +0.0034] | +0.0008 [-0.0061, +0.0077] |
| fd − melnik | 75 | +0.0351 [+0.0231, +0.0468] | +0.0136 [+0.0010, +0.0260] | +0.0566 [+0.0430, +0.0700] | +0.0272 [+0.0015, +0.0564] | +0.0246 [+0.0120, +0.0372] |
| fd_reified − melnik | 75 | +0.0368 [+0.0259, +0.0476] | +0.0064 [-0.0056, +0.0180] | +0.0671 [+0.0553, +0.0792] | +0.0325 [+0.0087, +0.0599] | +0.0423 [+0.0229, +0.0635] |

Hypotheses (Amendment 2, §4):

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R7a | mean |effect of fd_closure on J_tf| < 0.005 (upper bound) | 75 | +0.0020 [+0.0017, +0.0034] | confirmed |
| H-R7b (QUICK) | mean |R_exp effect| cover > closure (lower bound > 0) | 75 | +0.0090 [+0.0054, +0.0174] | confirmed |
| H-R7b (reified) | mean |R_exp effect| cover > closure (lower bound > 0) | 75 | +0.0122 [+0.0087, +0.0171] | confirmed |
| H-R7c | inversion share cover > fd (lower bound > 0) | 79 | -0.0763 [-0.0862, -0.0662] | not confirmed |
| H-R7c, family form | increase largest on TPC-DI (point estimate) | 79 | -0.0518 | not confirmed (largest: ChEMBL; TPC-DI largest in 31.1% of resamples; ChEMBL -0.0448, OpenData -0.1339, TPC-DI -0.0518) |
| H-R7c (reified, secondary) | inversion share cover > fd (lower bound > 0) | 79 | -0.0989 [-0.1032, -0.0946] | not confirmed |
| H-R7c (reified, secondary), family form | increase largest on TPC-DI (point estimate) | 79 | -0.1603 | not confirmed (largest: ChEMBL; TPC-DI largest in 0.0% of resamples; ChEMBL -0.0331, OpenData -0.1039, TPC-DI -0.1603) |

Reported per matcher only (controls or partial coverage): constant, datatype, embedding_random, oracle, oracle_decoy100, oracle_erase050, oracle_noise025, oracle_noise050, prisma, random, rema, rema_shuffled.

## RQ5: medians per scenario run (sf_stats.tsv)

| arm | runs | iterations | pairwise-graph vertices | edges | build ms | iterate ms |
|---|---|---|---|---|---|---|
| fd_closure | 32969 | 5 | 1074 | 52948 | 15544 | 37 |
| fd_cover | 32969 | 10 | 1074 | 6446 | 7122 | 6 |
| fd_thin1 | 32969 | 9 | 1074 | 7570 | 7100 | 7 |
| fd_thin2 | 32969 | 9 | 1074 | 7596 | 7082 | 7 |
| fd_thin3 | 32969 | 9 | 1074 | 7406 | 7083 | 7 |
| fd_reified_closure | 32969 | 11 | 78284 | 640714 | 24362 | 1122 |
| fd_reified_cover | 32969 | 14 | 3369 | 14286 | 7182 | 16 |
| fd_reified_thin1 | 32969 | 14 | 3369 | 16256 | 7163 | 18 |
| fd_reified_thin2 | 32969 | 14 | 3369 | 16270 | 7126 | 18 |
| fd_reified_thin3 | 32969 | 14 | 3369 | 16458 | 7092 | 17 |
| fd | 32969 | 5 | 1074 | 52948 | 17114 | 43 |
| melnik | 32969 | 15 | 1074 | 3730 | 5249 | 3 |
| fd_reified | 32969 | 12 | 63947 | 544184 | 24576 | 1013 |
