# R7 (Amendment 2), regime plain: registered analysis

pipeline check: fd_same vs layer 1 fd on 382 scenarios, max |difference| 0.0 — passed.

J = (R_exp + F1_Hr) / 2. Effects are paired per scenario, macro over families with ≥ 5 scenarios, then the mean over included matchers (non-controls with every scenario in every arm of the contrast). 95% percentile bootstrap over scenarios, stratified by family, 10,000 resamples, seed 20261002. `thin` = mean of the three random draws.

## Case set: primary (non-screen) (322 scenarios; families ChEMBL 72, Magellan 6, OpenData 94, TPC-DI 148)

Arm effects on SF's output (estimate [95% CI]):

| Contrast | matchers | J | R_exp | F1_Hr | MRR_exp | F1_M |
|---|---|---|---|---|---|---|
| fd_closure − fd | 76 | +0.0001 [-0.0005, +0.0008] | +0.0005 [-0.0003, +0.0014] | -0.0002 [-0.0010, +0.0007] | -0.0003 [-0.0013, +0.0010] | +0.0002 [-0.0009, +0.0014] |
| fd_cover − fd | 76 | +0.0116 [-0.0011, +0.0268] | +0.0245 [+0.0034, +0.0528] | -0.0012 [-0.0091, +0.0054] | +0.0347 [+0.0239, +0.0448] | +0.0012 [-0.0100, +0.0128] |
| fd_thin − fd | 76 | +0.0098 [+0.0004, +0.0211] | +0.0390 [+0.0246, +0.0585] | -0.0194 [-0.0284, -0.0111] | +0.0393 [+0.0325, +0.0465] | -0.0037 [-0.0147, +0.0087] |
| fd_reified_closure − fd_reified | 76 | -0.0053 [-0.0077, -0.0032] | -0.0069 [-0.0082, -0.0056] | -0.0036 [-0.0078, +0.0001] | -0.0025 [-0.0031, -0.0018] | -0.0037 [-0.0087, +0.0012] |
| fd_reified_cover − fd_reified | 76 | +0.0465 [+0.0389, +0.0546] | +0.0905 [+0.0773, +0.1049] | +0.0025 [-0.0037, +0.0104] | +0.0209 [+0.0124, +0.0294] | +0.0237 [+0.0102, +0.0377] |
| fd_reified_thin − fd_reified | 76 | +0.0404 [+0.0339, +0.0474] | +0.0893 [+0.0775, +0.1026] | -0.0085 [-0.0121, -0.0049] | +0.0267 [+0.0205, +0.0328] | +0.0175 [+0.0057, +0.0301] |
| fd_cover − fd_thin | 76 | +0.0018 [-0.0064, +0.0103] | -0.0146 [-0.0253, -0.0032] | +0.0181 [+0.0107, +0.0275] | -0.0046 [-0.0120, +0.0021] | +0.0050 [-0.0054, +0.0168] |
| fd_reified_cover − fd_reified_thin | 76 | +0.0061 [+0.0017, +0.0116] | +0.0011 [-0.0035, +0.0060] | +0.0111 [+0.0054, +0.0184] | -0.0057 [-0.0105, -0.0011] | +0.0062 [-0.0000, +0.0146] |
| fd − melnik | 76 | -0.1035 [-0.1252, -0.0827] | -0.2008 [-0.2284, -0.1718] | -0.0063 [-0.0354, +0.0124] | -0.1091 [-0.1223, -0.0957] | -0.0804 [-0.1112, -0.0561] |
| fd_reified − melnik | 76 | -0.0805 [-0.0965, -0.0639] | -0.1685 [-0.1895, -0.1464] | +0.0076 [-0.0126, +0.0230] | -0.0534 [-0.0636, -0.0424] | -0.0367 [-0.0602, -0.0136] |

Tie-breaker (Amendment 3's construction, plain only):

| Contrast (tie-breaker built from each arm) | J |
|---|---|
| fd_closure − fd | -0.0000 [-0.0002, +0.0001] |
| fd_cover − fd | -0.0012 [-0.0019, -0.0004] |
| fd_thin − fd | -0.0009 [-0.0019, +0.0002] |
| fd_reified_closure − fd_reified | -0.0002 [-0.0005, +0.0004] |
| fd_reified_cover − fd_reified | -0.0028 [-0.0039, -0.0017] |
| fd_reified_thin − fd_reified | -0.0019 [-0.0029, -0.0005] |
| fd_cover − fd_thin | -0.0002 [-0.0013, +0.0007] |
| fd_reified_cover − fd_reified_thin | -0.0009 [-0.0027, +0.0006] |

Hypotheses (Amendment 2, §4):

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R7a | mean |effect of fd_closure on J_tf| < 0.005 (upper bound) | 76 | +0.0012 [+0.0010, +0.0024] | confirmed |
| H-R7b (QUICK) | mean |R_exp effect| cover > closure (lower bound > 0) | 76 | +0.0329 [+0.0182, +0.0603] | confirmed |
| H-R7b (reified) | mean |R_exp effect| cover > closure (lower bound > 0) | 76 | +0.0830 [+0.0700, +0.0974] | confirmed |
| H-R7c | inversion share cover > fd (lower bound > 0) | 82 | -0.0670 [-0.0745, -0.0595] | not confirmed |
| H-R7c, family form | increase largest on TPC-DI (point estimate) | 82 | -0.0548 | not confirmed (largest: ChEMBL; TPC-DI largest in 2.8% of resamples; ChEMBL -0.0446, Magellan -0.0550, OpenData -0.1229, TPC-DI -0.0548) |
| H-R7c (reified, secondary) | inversion share cover > fd (lower bound > 0) | 82 | -0.0617 [-0.0689, -0.0548] | not confirmed |
| H-R7c (reified, secondary), family form | increase largest on TPC-DI (point estimate) | 82 | -0.1088 | not confirmed (largest: ChEMBL; TPC-DI largest in 0.0% of resamples; ChEMBL -0.0255, Magellan -0.0500, OpenData -0.0659, TPC-DI -0.1088) |

Reported per matcher only (controls or partial coverage): constant, datatype, embedding_random, extrema, oracle, oracle_decoy100, oracle_erase050, oracle_noise025, oracle_noise050, prisma, random, rema, rema_shuffled, sima_tpcdi.

## Case set: all (382 scenarios; families ChEMBL 86, Magellan 7, OpenData 111, TPC-DI 176)

Arm effects on SF's output (estimate [95% CI]):

| Contrast | matchers | J | R_exp | F1_Hr | MRR_exp | F1_M |
|---|---|---|---|---|---|---|
| fd_closure − fd | 76 | +0.0001 [-0.0004, +0.0006] | +0.0004 [-0.0003, +0.0013] | -0.0003 [-0.0010, +0.0005] | -0.0004 [-0.0013, +0.0007] | +0.0002 [-0.0008, +0.0013] |
| fd_cover − fd | 76 | +0.0097 [-0.0013, +0.0229] | +0.0219 [+0.0034, +0.0469] | -0.0024 [-0.0094, +0.0033] | +0.0312 [+0.0216, +0.0400] | -0.0028 [-0.0129, +0.0080] |
| fd_thin − fd | 76 | +0.0069 [-0.0015, +0.0170] | +0.0345 [+0.0214, +0.0524] | -0.0207 [-0.0284, -0.0132] | +0.0358 [+0.0298, +0.0421] | -0.0064 [-0.0161, +0.0045] |
| fd_reified_closure − fd_reified | 76 | -0.0051 [-0.0073, -0.0034] | -0.0070 [-0.0082, -0.0058] | -0.0033 [-0.0070, -0.0002] | -0.0024 [-0.0030, -0.0018] | -0.0037 [-0.0081, +0.0005] |
| fd_reified_cover − fd_reified | 76 | +0.0453 [+0.0381, +0.0532] | +0.0890 [+0.0767, +0.1030] | +0.0015 [-0.0040, +0.0087] | +0.0214 [+0.0137, +0.0291] | +0.0232 [+0.0112, +0.0357] |
| fd_reified_thin − fd_reified | 76 | +0.0391 [+0.0319, +0.0466] | +0.0882 [+0.0763, +0.1013] | -0.0101 [-0.0143, -0.0060] | +0.0268 [+0.0210, +0.0325] | +0.0168 [+0.0061, +0.0282] |
| fd_cover − fd_thin | 76 | +0.0028 [-0.0050, +0.0105] | -0.0126 [-0.0231, -0.0024] | +0.0182 [+0.0113, +0.0265] | -0.0046 [-0.0114, +0.0013] | +0.0036 [-0.0056, +0.0141] |
| fd_reified_cover − fd_reified_thin | 76 | +0.0062 [+0.0020, +0.0111] | +0.0008 [-0.0036, +0.0053] | +0.0116 [+0.0061, +0.0181] | -0.0053 [-0.0098, -0.0013] | +0.0064 [+0.0009, +0.0137] |
| fd − melnik | 76 | -0.0976 [-0.1187, -0.0776] | -0.1936 [-0.2212, -0.1661] | -0.0016 [-0.0275, +0.0152] | -0.1037 [-0.1164, -0.0912] | -0.0752 [-0.1031, -0.0539] |
| fd_reified − melnik | 76 | -0.0766 [-0.0932, -0.0601] | -0.1645 [-0.1862, -0.1420] | +0.0113 [-0.0077, +0.0252] | -0.0518 [-0.0625, -0.0410] | -0.0344 [-0.0568, -0.0142] |

Tie-breaker (Amendment 3's construction, plain only):

| Contrast (tie-breaker built from each arm) | J |
|---|---|
| fd_closure − fd | -0.0000 [-0.0001, +0.0001] |
| fd_cover − fd | -0.0011 [-0.0018, -0.0004] |
| fd_thin − fd | -0.0012 [-0.0021, -0.0001] |
| fd_reified_closure − fd_reified | -0.0002 [-0.0005, +0.0003] |
| fd_reified_cover − fd_reified | -0.0027 [-0.0037, -0.0017] |
| fd_reified_thin − fd_reified | -0.0021 [-0.0033, -0.0007] |
| fd_cover − fd_thin | +0.0001 [-0.0009, +0.0011] |
| fd_reified_cover − fd_reified_thin | -0.0006 [-0.0022, +0.0007] |

Hypotheses (Amendment 2, §4):

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R7a | mean |effect of fd_closure on J_tf| < 0.005 (upper bound) | 76 | +0.0012 [+0.0010, +0.0021] | confirmed |
| H-R7b (QUICK) | mean |R_exp effect| cover > closure (lower bound > 0) | 76 | +0.0292 [+0.0174, +0.0537] | confirmed |
| H-R7b (reified) | mean |R_exp effect| cover > closure (lower bound > 0) | 76 | +0.0818 [+0.0699, +0.0955] | confirmed |
| H-R7c | inversion share cover > fd (lower bound > 0) | 82 | -0.0651 [-0.0722, -0.0584] | not confirmed |
| H-R7c, family form | increase largest on TPC-DI (point estimate) | 82 | -0.0534 | not confirmed (largest: ChEMBL; TPC-DI largest in 1.8% of resamples; ChEMBL -0.0434, Magellan -0.0493, OpenData -0.1235, TPC-DI -0.0534) |
| H-R7c (reified, secondary) | inversion share cover > fd (lower bound > 0) | 82 | -0.0601 [-0.0669, -0.0539] | not confirmed |
| H-R7c (reified, secondary), family form | increase largest on TPC-DI (point estimate) | 82 | -0.1083 | not confirmed (largest: ChEMBL; TPC-DI largest in 0.0% of resamples; ChEMBL -0.0248, Magellan -0.0449, OpenData -0.0658, TPC-DI -0.1083) |

Reported per matcher only (controls or partial coverage): constant, datatype, embedding_random, extrema, oracle, oracle_decoy100, oracle_erase050, oracle_noise025, oracle_noise050, prisma, random, rema, rema_shuffled, sima_tpcdi.

## Case set: screen (60 scenarios; families ChEMBL 14, OpenData 17, TPC-DI 28)

Arm effects on SF's output (estimate [95% CI]):

| Contrast | matchers | J | R_exp | F1_Hr | MRR_exp | F1_M |
|---|---|---|---|---|---|---|
| fd_closure − fd | 76 | -0.0002 [-0.0015, +0.0010] | +0.0002 [-0.0015, +0.0016] | -0.0006 [-0.0026, +0.0013] | -0.0010 [-0.0024, +0.0006] | +0.0003 [-0.0032, +0.0044] |
| fd_cover − fd | 76 | -0.0105 [-0.0215, +0.0002] | -0.0095 [-0.0281, +0.0087] | -0.0115 [-0.0205, -0.0026] | +0.0137 [-0.0019, +0.0282] | -0.0228 [-0.0384, -0.0076] |
| fd_thin − fd | 76 | -0.0073 [-0.0166, +0.0021] | +0.0135 [-0.0015, +0.0285] | -0.0280 [-0.0363, -0.0199] | +0.0238 [+0.0103, +0.0371] | -0.0138 [-0.0305, +0.0017] |
| fd_reified_closure − fd_reified | 76 | -0.0060 [-0.0088, -0.0035] | -0.0099 [-0.0144, -0.0062] | -0.0022 [-0.0053, +0.0005] | -0.0031 [-0.0051, -0.0012] | -0.0053 [-0.0081, -0.0027] |
| fd_reified_cover − fd_reified | 76 | +0.0527 [+0.0429, +0.0627] | +0.1073 [+0.0918, +0.1238] | -0.0020 [-0.0091, +0.0048] | +0.0340 [+0.0165, +0.0488] | +0.0279 [+0.0078, +0.0457] |
| fd_reified_thin − fd_reified | 76 | +0.0514 [+0.0411, +0.0622] | +0.1175 [+0.1000, +0.1369] | -0.0148 [-0.0218, -0.0081] | +0.0400 [+0.0298, +0.0495] | +0.0236 [+0.0037, +0.0410] |
| fd_cover − fd_thin | 76 | -0.0032 [-0.0129, +0.0059] | -0.0230 [-0.0377, -0.0101] | +0.0165 [+0.0079, +0.0253] | -0.0100 [-0.0206, +0.0005] | -0.0091 [-0.0208, +0.0031] |
| fd_reified_cover − fd_reified_thin | 76 | +0.0013 [-0.0053, +0.0075] | -0.0102 [-0.0207, -0.0009] | +0.0129 [+0.0062, +0.0196] | -0.0060 [-0.0153, +0.0016] | +0.0043 [-0.0023, +0.0113] |
| fd − melnik | 76 | -0.0794 [-0.1073, -0.0552] | -0.1727 [-0.2232, -0.1295] | +0.0139 [+0.0015, +0.0262] | -0.0947 [-0.1191, -0.0703] | -0.0401 [-0.0659, -0.0118] |
| fd_reified − melnik | 76 | -0.0812 [-0.1067, -0.0596] | -0.1857 [-0.2334, -0.1461] | +0.0233 [+0.0130, +0.0334] | -0.0652 [-0.0836, -0.0459] | -0.0212 [-0.0452, +0.0044] |

Tie-breaker (Amendment 3's construction, plain only):

| Contrast (tie-breaker built from each arm) | J |
|---|---|
| fd_closure − fd | +0.0000 [-0.0003, +0.0004] |
| fd_cover − fd | -0.0009 [-0.0027, +0.0008] |
| fd_thin − fd | -0.0020 [-0.0036, -0.0006] |
| fd_reified_closure − fd_reified | -0.0004 [-0.0008, +0.0000] |
| fd_reified_cover − fd_reified | -0.0023 [-0.0041, -0.0006] |
| fd_reified_thin − fd_reified | -0.0027 [-0.0041, -0.0012] |
| fd_cover − fd_thin | +0.0011 [-0.0006, +0.0028] |
| fd_reified_cover − fd_reified_thin | +0.0004 [-0.0012, +0.0018] |

Hypotheses (Amendment 2, §4):

| Hypothesis | Test | matchers | estimate [95% CI] | verdict |
|---|---|---|---|---|
| H-R7a | mean |effect of fd_closure on J_tf| < 0.005 (upper bound) | 76 | +0.0018 [+0.0017, +0.0032] | confirmed |
| H-R7b (QUICK) | mean |R_exp effect| cover > closure (lower bound > 0) | 76 | +0.0302 [+0.0248, +0.0439] | confirmed |
| H-R7b (reified) | mean |R_exp effect| cover > closure (lower bound > 0) | 76 | +0.0969 [+0.0827, +0.1114] | confirmed |
| H-R7c | inversion share cover > fd (lower bound > 0) | 82 | -0.0676 [-0.0762, -0.0588] | not confirmed |
| H-R7c, family form | increase largest on TPC-DI (point estimate) | 82 | -0.0465 | not confirmed (largest: ChEMBL; TPC-DI largest in 22.0% of resamples; ChEMBL -0.0369, OpenData -0.1264, TPC-DI -0.0465) |
| H-R7c (reified, secondary) | inversion share cover > fd (lower bound > 0) | 82 | -0.0632 [-0.0662, -0.0605] | not confirmed |
| H-R7c (reified, secondary), family form | increase largest on TPC-DI (point estimate) | 82 | -0.1054 | not confirmed (largest: ChEMBL; TPC-DI largest in 0.0% of resamples; ChEMBL -0.0212, OpenData -0.0656, TPC-DI -0.1054) |

Reported per matcher only (controls or partial coverage): constant, datatype, embedding_random, extrema, oracle, oracle_decoy100, oracle_erase050, oracle_noise025, oracle_noise050, prisma, random, rema, rema_shuffled, sima_tpcdi.

## RQ5: medians per scenario run (sf_stats.tsv)

| arm | runs | iterations | pairwise-graph vertices | edges | build ms | iterate ms |
|---|---|---|---|---|---|---|
| fd_closure | 33766 | 5 | 1119 | 53188 | 13470 | 41 |
| fd_cover | 33766 | 10 | 1119 | 6666 | 5404 | 7 |
| fd_thin1 | 33766 | 9 | 1119 | 7878 | 5342 | 8 |
| fd_thin2 | 33766 | 9 | 1119 | 7836 | 5348 | 8 |
| fd_thin3 | 33766 | 9 | 1119 | 7646 | 5339 | 8 |
| fd_reified_closure | 33766 | 12 | 80482 | 643026 | 20797 | 1180 |
| fd_reified_cover | 33766 | 14 | 3527 | 14566 | 5280 | 18 |
| fd_reified_thin1 | 33766 | 14 | 3527 | 16828 | 5181 | 19 |
| fd_reified_thin2 | 33766 | 14 | 3527 | 16796 | 5162 | 19 |
| fd_reified_thin3 | 33766 | 14 | 3527 | 16898 | 5190 | 19 |
| fd | 33766 | 5 | 1119 | 53188 | 14450 | 53 |
| melnik | 33766 | 15 | 1119 | 3822 | 3886 | 7 |
| fd_reified | 33713 | 12 | 65704 | 549128 | 20903 | 1059 |
