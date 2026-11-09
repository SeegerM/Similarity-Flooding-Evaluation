# SemStruct information-matched ablation (plain)

EXPLORATORY (SEMSTRUCT_ABLATION.md, Amendment 1). Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples).

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 549 scenarios.

## remaining505: 505 scenarios, families ChEMBL 170, OpenData 168, TPC-DI 164

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.309 | 0.263 | 0.355 |
| lm_only | 0.379 | 0.409 | 0.348 |
| zero_delta | 0.379 | 0.409 | 0.348 |
| full_permuted | 0.308 | 0.261 | 0.355 |
| onehop | 0.493 | 0.516 | 0.470 |
| onehop_permuted | 0.493 | 0.516 | 0.470 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | -0.070 [-0.088, -0.053] | -0.145 | +0.006 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.184 [+0.168, +0.199] | +0.253 | +0.115 |
| onehop − lm_only | own values over the header alone | +0.114 [+0.096, +0.133] | +0.107 | +0.122 |
| full − full_permuted | row co-occurrence (full) | +0.001 [-0.001, +0.003] | +0.002 | -0.000 |
| onehop − onehop_permuted | value order (one hop; should be ~0) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |

## all: 549 scenarios, families ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.479 | 0.441 | 0.518 |
| lm_only | 0.535 | 0.556 | 0.514 |
| zero_delta | 0.535 | 0.556 | 0.514 |
| full_permuted | 0.479 | 0.439 | 0.518 |
| onehop | 0.620 | 0.635 | 0.605 |
| onehop_permuted | 0.620 | 0.635 | 0.605 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | -0.056 [-0.071, -0.042] | -0.115 | +0.004 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.140 [+0.128, +0.154] | +0.194 | +0.087 |
| onehop − lm_only | own values over the header alone | +0.084 [+0.071, +0.098] | +0.078 | +0.091 |
| full − full_permuted | row co-occurrence (full) | +0.001 [-0.001, +0.002] | +0.002 | -0.000 |
| onehop − onehop_permuted | value order (one hop; should be ~0) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |

