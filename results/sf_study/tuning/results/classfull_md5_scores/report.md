# Amendment 8, class nodes on the full panel: regime md5

- gate levenshtein__melnik_devnone = layer 1 levenshtein__melnik: max |difference| 0.0 — passed
- class edges in the schema graph (sf_stats growth = edge files, where layer 1 has an output): 131876 checked, 0 wrong, 0 outputs without statistics; reported only: 1060 (matcher, arm, scenario) without a class-arm output, 0 class outputs without a layer-1 output — passed
- skipped cells per reason (reported only): layer-1 melnik output not finite or misshapen 0 (matcher, scenario; the scenario drops out for that matcher); class-arm output not finite or misshapen 0 (matcher, scenario, arm); constant__melnik not finite or misshapen 0 (matcher, scenario; no H8a blocks there); datatype matrix D missing or not finite 0 (matcher, scenario; no A_p / A_s there) — passed

J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over the included matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. Confirmed: lower bound > 0 (H8f: interval inside (−0.01, +0.01)). Columns R_exp and F1_Hr: estimates of the parts.

## primary: non-screen (322 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| report: shape − shape_shuffled | 75 | +0.0225 [+0.0112, +0.0371] | +0.0235 | +0.0214 | 74 of 75 |  |
| report: semtype − semtype_shuffled | 75 | +0.0020 [+0.0012, +0.0027] | +0.0020 | +0.0019 | 71 of 75 |  |
| report: shape − melnik | 75 | +0.0146 [+0.0055, +0.0258] | +0.0130 | +0.0161 | 69 of 75 |  |
| H8f (semtype, equivalence ±0.01): semtype − melnik | 75 | +0.0006 [+0.0003, +0.0010] | +0.0006 | +0.0007 | 51 of 75 | CONFIRMED |
| report: L_p − shape | 75 | -0.0102 [-0.0157, -0.0054] | -0.0129 | -0.0076 | 7 of 75 |  |
| report: L_s − semtype | 75 | -0.0001 [-0.0005, +0.0003] | -0.0017 | +0.0015 | 54 of 75 |  |
| report: B_p − shape | 75 | -0.0045 [-0.0088, -0.0007] | -0.0081 | -0.0009 | 26 of 75 |  |
| report: B_s − semtype | 75 | +0.0012 [+0.0009, +0.0016] | +0.0006 | +0.0019 | 55 of 75 |  |
| report: A_p − shape | 75 | -0.0030 [-0.0073, +0.0009] | -0.0079 | +0.0018 | 33 of 75 |  |
| report: A_s − semtype | 75 | +0.0000 [-0.0004, +0.0004] | -0.0016 | +0.0017 | 56 of 75 |  |
| report: shape_shuffled − melnik | 75 | -0.0079 [-0.0119, -0.0047] | -0.0105 | -0.0053 | 17 of 75 |  |
| report: semtype_shuffled − melnik | 75 | -0.0013 [-0.0021, -0.0005] | -0.0014 | -0.0012 | 25 of 75 |  |

## secondary: all (382 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| report: shape − shape_shuffled | 75 | +0.0209 [+0.0111, +0.0339] | +0.0218 | +0.0199 | 74 of 75 |  |
| report: semtype − semtype_shuffled | 75 | +0.0020 [+0.0013, +0.0027] | +0.0021 | +0.0019 | 73 of 75 |  |
| report: shape − melnik | 75 | +0.0136 [+0.0058, +0.0238] | +0.0124 | +0.0149 | 68 of 75 |  |
| H8f (semtype, equivalence ±0.01): semtype − melnik | 75 | +0.0007 [+0.0003, +0.0010] | +0.0006 | +0.0008 | 50 of 75 | CONFIRMED |
| report: L_p − shape | 75 | -0.0097 [-0.0147, -0.0055] | -0.0120 | -0.0074 | 11 of 75 |  |
| report: L_s − semtype | 75 | -0.0001 [-0.0005, +0.0003] | -0.0016 | +0.0013 | 53 of 75 |  |
| report: B_p − shape | 75 | -0.0041 [-0.0080, -0.0007] | -0.0074 | -0.0008 | 28 of 75 |  |
| report: B_s − semtype | 75 | +0.0012 [+0.0009, +0.0015] | +0.0006 | +0.0018 | 53 of 75 |  |
| report: A_p − shape | 75 | -0.0027 [-0.0066, +0.0007] | -0.0070 | +0.0017 | 36 of 75 |  |
| report: A_s − semtype | 75 | +0.0000 [-0.0003, +0.0004] | -0.0015 | +0.0016 | 55 of 75 |  |
| report: shape_shuffled − melnik | 75 | -0.0072 [-0.0109, -0.0044] | -0.0094 | -0.0050 | 18 of 75 |  |
| report: semtype_shuffled − melnik | 75 | -0.0013 [-0.0020, -0.0006] | -0.0015 | -0.0012 | 23 of 75 |  |

- H8a (shape): 0 within-block inversions of 1782811428 ordered pairs inside the true refined blocks (all matchers, all scenarios) — CONFIRMED.
- H8a (semtype): 0 within-block inversions of 12110458445 ordered pairs inside the true refined blocks (all matchers, all scenarios) — CONFIRMED.
