# Amendment 10, family 13 with Sherlock's types: regime md5

- gate levenshtein__melnik_devnone = layer 1 levenshtein__melnik: max |difference| 0.0 — passed
- Sherlock class edges in the schema graph (sf_stats growth = edge files, where layer 1 has an output): 65938 checked, 0 wrong, 0 outputs without statistics; reported only: 530 (matcher, arm, scenario) without an arm output, 0 arm outputs without a layer-1 output — passed
- skipped cells per reason (reported only): layer-1 melnik output not finite or misshapen 0 (matcher, scenario; the scenario drops out for that matcher); Sherlock-arm output not finite or misshapen 0 (matcher, scenario, arm); rules-arm output not finite or misshapen 0; constant__melnik not finite or misshapen 0 (no H10a blocks there); datatype matrix D missing or not finite 0 (no A_k there) — passed

J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over the included matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. Confirmed: lower bound > 0. Columns R_exp and F1_Hr: estimates of the parts. Verdicts on the primary set only.

## primary: all (382 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| H10d: sherlock − sherlock_shuffled | 75 | +0.0224 [+0.0050, +0.0441] | +0.0268 | +0.0179 | 68 of 75 | CONFIRMED |
| H10e: sherlock − melnik | 75 | +0.0214 [+0.0097, +0.0365] | +0.0204 | +0.0223 | 63 of 75 | CONFIRMED |
| report: L_k − sherlock | 75 | +0.0049 [-0.0049, +0.0121] | +0.0053 | +0.0045 | 41 of 75 |  |
| report: B_k − sherlock | 75 | +0.0059 [-0.0043, +0.0131] | +0.0052 | +0.0067 | 43 of 75 |  |
| report: A_k − sherlock | 75 | +0.0043 [-0.0038, +0.0098] | +0.0031 | +0.0055 | 49 of 75 |  |
| report: sherlock − semtype_rules | 75 | +0.0207 [+0.0089, +0.0358] | +0.0199 | +0.0215 | 61 of 75 |  |
| report: sherlock_shuffled − melnik | 75 | -0.0010 [-0.0093, +0.0078] | -0.0064 | +0.0044 | 55 of 75 |  |

## secondary: non-screen (322 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| H10d: sherlock − sherlock_shuffled | 75 | +0.0268 [+0.0089, +0.0493] | +0.0291 | +0.0245 | 71 of 75 |  |
| H10e: sherlock − melnik | 75 | +0.0225 [+0.0087, +0.0394] | +0.0217 | +0.0233 | 63 of 75 |  |
| report: L_k − sherlock | 75 | +0.0036 [-0.0074, +0.0119] | +0.0041 | +0.0031 | 41 of 75 |  |
| report: B_k − sherlock | 75 | +0.0051 [-0.0065, +0.0135] | +0.0043 | +0.0060 | 41 of 75 |  |
| report: A_k − sherlock | 75 | +0.0038 [-0.0056, +0.0102] | +0.0022 | +0.0053 | 44 of 75 |  |
| report: sherlock − semtype_rules | 75 | +0.0218 [+0.0079, +0.0389] | +0.0211 | +0.0226 | 61 of 75 |  |
| report: sherlock_shuffled − melnik | 75 | -0.0044 [-0.0111, +0.0021] | -0.0075 | -0.0013 | 43 of 75 |  |

- H10a: 0 within-block inversions of 547988177 ordered pairs inside the true refined blocks (all matchers, all scenarios) — CONFIRMED.
