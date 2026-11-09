# Amendment 8, class nodes on the full panel: regime plain

- gate levenshtein__melnik_devnone = layer 1 levenshtein__melnik: max |difference| 0.0 — passed
- class edges in the schema graph (sf_stats growth = edge files, where layer 1 has an output): 135064 checked, 0 wrong, 0 outputs without statistics; reported only: 2456 (matcher, arm, scenario) without a class-arm output, 0 class outputs without a layer-1 output — passed
- skipped cells per reason (reported only): layer-1 melnik output not finite or misshapen 120 (matcher, scenario; the scenario drops out for that matcher); class-arm output not finite or misshapen 0 (matcher, scenario, arm); constant__melnik not finite or misshapen 0 (matcher, scenario; no H8a blocks there); datatype matrix D missing or not finite 0 (matcher, scenario; no A_p / A_s there) — passed

J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over the included matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. Confirmed: lower bound > 0 (H8f: interval inside (−0.01, +0.01)). Columns R_exp and F1_Hr: estimates of the parts.

## primary: non-screen (322 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| H8b (shape): shape − shape_shuffled | 76 | +0.0457 [+0.0371, +0.0543] | +0.0407 | +0.0506 | 76 of 76 | CONFIRMED |
| H8b (semtype): semtype − semtype_shuffled | 76 | +0.0547 [+0.0456, +0.0638] | +0.0531 | +0.0562 | 76 of 76 | CONFIRMED |
| H8c (shape): shape − melnik | 76 | +0.0186 [+0.0142, +0.0237] | +0.0128 | +0.0243 | 67 of 76 | CONFIRMED |
| H8c (semtype): semtype − melnik | 76 | +0.0265 [+0.0178, +0.0326] | +0.0250 | +0.0280 | 62 of 76 | CONFIRMED |
| H8d (shape): L_p − shape | 76 | +0.0149 [+0.0087, +0.0204] | +0.0079 | +0.0219 | 57 of 76 | CONFIRMED |
| H8d (semtype): L_s − semtype | 76 | +0.0027 [-0.0028, +0.0077] | -0.0030 | +0.0084 | 47 of 76 | not confirmed |
| H8e (shape): B_p − shape | 76 | +0.0197 [+0.0147, +0.0242] | +0.0188 | +0.0205 | 64 of 76 | CONFIRMED |
| H8e (semtype): B_s − semtype | 76 | +0.0062 [-0.0012, +0.0120] | +0.0083 | +0.0041 | 50 of 76 | not confirmed |
| report: A_p − shape | 76 | +0.0276 [+0.0213, +0.0333] | +0.0240 | +0.0312 | 65 of 76 |  |
| report: A_s − semtype | 76 | +0.0160 [+0.0115, +0.0202] | +0.0126 | +0.0194 | 61 of 76 |  |
| report: shape_shuffled − melnik | 76 | -0.0271 [-0.0320, -0.0217] | -0.0279 | -0.0263 | 2 of 76 |  |
| report: semtype_shuffled − melnik | 76 | -0.0282 [-0.0348, -0.0207] | -0.0282 | -0.0282 | 0 of 76 |  |

## secondary: all (382 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 | verdict |
|---|---|---|---|---|---|---|
| H8b (shape): shape − shape_shuffled | 76 | +0.0427 [+0.0328, +0.0524] | +0.0376 | +0.0479 | 76 of 76 | CONFIRMED |
| H8b (semtype): semtype − semtype_shuffled | 76 | +0.0547 [+0.0468, +0.0629] | +0.0513 | +0.0582 | 76 of 76 | CONFIRMED |
| H8c (shape): shape − melnik | 76 | +0.0185 [+0.0146, +0.0232] | +0.0126 | +0.0243 | 67 of 76 | CONFIRMED |
| H8c (semtype): semtype − melnik | 76 | +0.0251 [+0.0175, +0.0316] | +0.0232 | +0.0271 | 61 of 76 | CONFIRMED |
| H8d (shape): L_p − shape | 76 | +0.0147 [+0.0092, +0.0199] | +0.0079 | +0.0214 | 56 of 76 | CONFIRMED |
| H8d (semtype): L_s − semtype | 76 | +0.0018 [-0.0032, +0.0068] | -0.0031 | +0.0068 | 48 of 76 | not confirmed |
| H8e (shape): B_p − shape | 76 | +0.0197 [+0.0154, +0.0239] | +0.0188 | +0.0207 | 65 of 76 | CONFIRMED |
| H8e (semtype): B_s − semtype | 76 | +0.0057 [-0.0007, +0.0111] | +0.0079 | +0.0036 | 48 of 76 | not confirmed |
| report: A_p − shape | 76 | +0.0279 [+0.0223, +0.0333] | +0.0242 | +0.0316 | 65 of 76 |  |
| report: A_s − semtype | 76 | +0.0162 [+0.0120, +0.0201] | +0.0130 | +0.0194 | 61 of 76 |  |
| report: shape_shuffled − melnik | 76 | -0.0243 [-0.0309, -0.0166] | -0.0250 | -0.0235 | 1 of 76 |  |
| report: semtype_shuffled − melnik | 76 | -0.0296 [-0.0355, -0.0225] | -0.0281 | -0.0311 | 0 of 76 |  |

- H8a (shape): 0 within-block inversions of 455507808 ordered pairs inside the true refined blocks (all matchers, all scenarios) — CONFIRMED.
- H8a (semtype): 0 within-block inversions of 491328972 ordered pairs inside the true refined blocks (all matchers, all scenarios) — CONFIRMED.
