# SemStruct, the revised setup (Amendment 2): plain, evaluation row cap 400

EXPLORATORY. Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples). Seeds present: full 3, valavg 3, onehop 1, mlp 1, lm_only 1, zero_delta 1, permuted 1

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 549 scenarios.

## remaining505: 505 scenarios, families ChEMBL 170, OpenData 168, TPC-DI 164

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.594 | 0.635 | 0.552 |
| valavg | 0.637 | 0.684 | 0.590 |
| onehop | 0.601 | 0.628 | 0.574 |
| mlp | 0.498 | 0.520 | 0.477 |
| lm_only | 0.391 | 0.437 | 0.345 |
| zero_delta | 0.391 | 0.437 | 0.345 |
| permuted | 0.609 | 0.658 | 0.560 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.202 [+0.184, +0.221] | +0.199 | +0.206 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.008 [-0.002, +0.017] | -0.007 | +0.023 |
| valavg − full | the authors' ValAvg minus message passing | +0.043 [+0.036, +0.052] | +0.049 | +0.038 |
| onehop − lm_only | own values (one hop) over the header alone | +0.210 [+0.193, +0.227] | +0.192 | +0.229 |
| valavg − lm_only | ValAvg over the header alone | +0.246 [+0.227, +0.263] | +0.247 | +0.244 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | -0.000 [-0.000, +0.000] | +0.000 | -0.000 |
| mlp − lm_only | ladder: MLP on the column embedding over the header alone | +0.107 [+0.094, +0.120] | +0.083 | +0.131 |
| valavg − mlp | ladder: the column's values (ValAvg) over the MLP | +0.139 [+0.126, +0.152] | +0.164 | +0.113 |
| full − valavg | ladder: message passing over ValAvg | -0.043 [-0.052, -0.035] | -0.049 | -0.038 |

Per seed (J_tf, point estimates):

- seed 1: full − lm_only +0.217; valavg − full +0.028
- seed 2: full − lm_only +0.186; valavg − full +0.066
- seed 3: full − lm_only +0.204; valavg − full +0.037

## all: 549 scenarios, families ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.693 | 0.720 | 0.666 |
| valavg | 0.727 | 0.760 | 0.694 |
| onehop | 0.701 | 0.720 | 0.682 |
| mlp | 0.624 | 0.639 | 0.610 |
| lm_only | 0.544 | 0.576 | 0.511 |
| zero_delta | 0.544 | 0.576 | 0.511 |
| permuted | 0.706 | 0.740 | 0.671 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.149 [+0.136, +0.163] | +0.144 | +0.154 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.008 [+0.001, +0.016] | +0.001 | +0.016 |
| valavg − full | the authors' ValAvg minus message passing | +0.035 [+0.029, +0.041] | +0.041 | +0.029 |
| onehop − lm_only | own values (one hop) over the header alone | +0.157 [+0.145, +0.169] | +0.145 | +0.170 |
| valavg − lm_only | ValAvg over the header alone | +0.184 [+0.170, +0.197] | +0.185 | +0.183 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | -0.000 [-0.000, +0.000] | +0.000 | -0.000 |
| mlp − lm_only | ladder: MLP on the column embedding over the header alone | +0.081 [+0.072, +0.089] | +0.063 | +0.098 |
| valavg − mlp | ladder: the column's values (ValAvg) over the MLP | +0.103 [+0.094, +0.113] | +0.121 | +0.085 |
| full − valavg | ladder: message passing over ValAvg | -0.035 [-0.041, -0.028] | -0.041 | -0.029 |

Per seed (J_tf, point estimates):

- seed 1: full − lm_only +0.162; valavg − full +0.021
- seed 2: full − lm_only +0.140; valavg − full +0.048
- seed 3: full − lm_only +0.146; valavg − full +0.035

