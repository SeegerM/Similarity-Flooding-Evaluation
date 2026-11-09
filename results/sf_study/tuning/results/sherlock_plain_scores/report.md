# Amendment 10, family 13 with Sherlock's types: regime plain

- gate levenshtein__melnik_devnone = layer 1 levenshtein__melnik: max |difference| 0.0 — passed
- Sherlock class edges in the schema graph (sf_stats growth = edge files, where layer 1 has an output): 67532 checked, 0 wrong, 0 outputs without statistics; reported only: 1228 (matcher, arm, scenario) without an arm output, 0 arm outputs without a layer-1 output — passed
- skipped cells per reason (reported only): layer-1 melnik output not finite or misshapen 120 (matcher, scenario; the scenario drops out for that matcher); Sherlock-arm output not finite or misshapen 0 (matcher, scenario, arm); rules-arm output not finite or misshapen 0; constant__melnik not finite or misshapen 0 (no H10a blocks there); datatype matrix D missing or not finite 0 (no A_k there) — passed

J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over the included matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. Confirmed: lower bound > 0. Columns R_exp and F1_Hr: estimates of the parts. Verdicts on the primary set only.

## primary: all (382 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| H10b: sherlock − sherlock_shuffled | 76 | +0.0796 [+0.0708, +0.0886] | +0.0802 | +0.0789 | 76 of 76 | CONFIRMED |
| H10c: sherlock − melnik | 76 | +0.0508 [+0.0445, +0.0573] | +0.0529 | +0.0488 | 74 of 76 | CONFIRMED |
| report: L_k − sherlock | 76 | +0.0040 [-0.0011, +0.0088] | -0.0021 | +0.0101 | 51 of 76 |  |
| report: B_k − sherlock | 76 | +0.0066 [+0.0021, +0.0107] | +0.0082 | +0.0049 | 55 of 76 |  |
| report: A_k − sherlock | 76 | +0.0227 [+0.0181, +0.0270] | +0.0206 | +0.0248 | 70 of 76 |  |
| report: sherlock − semtype_rules | 76 | +0.0257 [+0.0167, +0.0344] | +0.0297 | +0.0217 | 72 of 76 |  |
| report: sherlock_shuffled − melnik | 76 | -0.0287 [-0.0341, -0.0235] | -0.0272 | -0.0302 | 0 of 76 |  |

## secondary: non-screen (322 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| H10b: sherlock − sherlock_shuffled | 76 | +0.0807 [+0.0711, +0.0900] | +0.0812 | +0.0802 | 76 of 76 |  |
| H10c: sherlock − melnik | 76 | +0.0507 [+0.0434, +0.0578] | +0.0525 | +0.0488 | 74 of 76 |  |
| report: L_k − sherlock | 76 | +0.0031 [-0.0022, +0.0083] | -0.0033 | +0.0096 | 51 of 76 |  |
| report: B_k − sherlock | 76 | +0.0059 [+0.0011, +0.0103] | +0.0075 | +0.0043 | 49 of 76 |  |
| report: A_k − sherlock | 76 | +0.0229 [+0.0180, +0.0275] | +0.0209 | +0.0248 | 70 of 76 |  |
| report: sherlock − semtype_rules | 76 | +0.0242 [+0.0141, +0.0341] | +0.0275 | +0.0208 | 72 of 76 |  |
| report: sherlock_shuffled − melnik | 76 | -0.0300 [-0.0355, -0.0249] | -0.0287 | -0.0314 | 0 of 76 |  |

- H10a: 0 within-block inversions of 38810814 ordered pairs inside the true refined blocks (all matchers, all scenarios) — CONFIRMED.
