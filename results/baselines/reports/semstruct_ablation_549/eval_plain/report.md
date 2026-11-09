# SemStruct information-matched ablation (plain)

EXPLORATORY (SEMSTRUCT_ABLATION.md, Amendment 1). Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples).

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 549 scenarios.

## remaining505: 505 scenarios, families ChEMBL 170, OpenData 168, TPC-DI 164

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.412 | 0.417 | 0.407 |
| lm_only | 0.379 | 0.409 | 0.348 |
| zero_delta | 0.379 | 0.409 | 0.348 |
| full_permuted | 0.412 | 0.416 | 0.408 |
| onehop | 0.507 | 0.540 | 0.475 |
| onehop_permuted | 0.507 | 0.540 | 0.475 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.033 [+0.014, +0.053] | +0.008 | +0.059 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.095 [+0.084, +0.108] | +0.123 | +0.068 |
| onehop − lm_only | own values over the header alone | +0.129 [+0.111, +0.147] | +0.131 | +0.127 |
| full − full_permuted | row co-occurrence (full) | +0.000 [-0.000, +0.001] | +0.000 | -0.000 |
| onehop − onehop_permuted | value order (one hop; should be ~0) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |

## all: 549 scenarios, families ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.549 | 0.540 | 0.558 |
| lm_only | 0.535 | 0.556 | 0.514 |
| zero_delta | 0.535 | 0.556 | 0.514 |
| full_permuted | 0.549 | 0.540 | 0.558 |
| onehop | 0.631 | 0.653 | 0.609 |
| onehop_permuted | 0.631 | 0.653 | 0.609 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.014 [-0.003, +0.030] | -0.016 | +0.044 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.082 [+0.070, +0.096] | +0.113 | +0.051 |
| onehop − lm_only | own values over the header alone | +0.096 [+0.083, +0.109] | +0.097 | +0.095 |
| full − full_permuted | row co-occurrence (full) | +0.000 [-0.000, +0.000] | +0.000 | -0.000 |
| onehop − onehop_permuted | value order (one hop; should be ~0) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |

