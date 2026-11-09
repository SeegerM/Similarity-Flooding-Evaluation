# SemStruct, the revised setup (Amendment 2): plain, evaluation row cap 2000

EXPLORATORY. Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples). Seeds present: full 3, valavg 3, onehop 1, mlp 1, lm_only 1, zero_delta 1, permuted 1

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 549 scenarios.

## remaining505: 505 scenarios, families ChEMBL 170, OpenData 168, TPC-DI 164

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.604 | 0.646 | 0.561 |
| valavg | 0.654 | 0.704 | 0.604 |
| onehop | 0.605 | 0.634 | 0.576 |
| mlp | 0.498 | 0.520 | 0.477 |
| lm_only | 0.391 | 0.437 | 0.345 |
| zero_delta | 0.391 | 0.437 | 0.345 |
| permuted | 0.617 | 0.668 | 0.567 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.212 [+0.193, +0.232] | +0.209 | +0.215 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.001 [-0.009, +0.012] | -0.012 | +0.015 |
| valavg − full | the authors' ValAvg minus message passing | +0.051 [+0.044, +0.058] | +0.058 | +0.043 |
| onehop − lm_only | own values (one hop) over the header alone | +0.214 [+0.197, +0.231] | +0.198 | +0.230 |
| valavg − lm_only | ValAvg over the header alone | +0.263 [+0.244, +0.282] | +0.268 | +0.258 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | +0.000 [+0.000, +0.001] | +0.000 | +0.000 |
| mlp − lm_only | ladder: MLP on the column embedding over the header alone | +0.107 [+0.094, +0.120] | +0.083 | +0.131 |
| valavg − mlp | ladder: the column's values (ValAvg) over the MLP | +0.156 [+0.143, +0.169] | +0.184 | +0.127 |
| full − valavg | ladder: message passing over ValAvg | -0.051 [-0.058, -0.043] | -0.058 | -0.043 |

Per seed (J_tf, point estimates):

- seed 1: full − lm_only +0.227; valavg − full +0.030
- seed 2: full − lm_only +0.187; valavg − full +0.089
- seed 3: full − lm_only +0.224; valavg − full +0.033

## all: 549 scenarios, families ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.701 | 0.728 | 0.673 |
| valavg | 0.740 | 0.775 | 0.705 |
| onehop | 0.705 | 0.725 | 0.684 |
| mlp | 0.624 | 0.639 | 0.610 |
| lm_only | 0.544 | 0.576 | 0.511 |
| zero_delta | 0.544 | 0.576 | 0.511 |
| permuted | 0.713 | 0.748 | 0.678 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.157 [+0.143, +0.171] | +0.153 | +0.162 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.004 [-0.004, +0.012] | -0.003 | +0.011 |
| valavg − full | the authors' ValAvg minus message passing | +0.040 [+0.034, +0.046] | +0.047 | +0.032 |
| onehop − lm_only | own values (one hop) over the header alone | +0.161 [+0.148, +0.174] | +0.150 | +0.172 |
| valavg − lm_only | ValAvg over the header alone | +0.197 [+0.182, +0.211] | +0.200 | +0.194 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |
| mlp − lm_only | ladder: MLP on the column embedding over the header alone | +0.081 [+0.072, +0.089] | +0.063 | +0.098 |
| valavg − mlp | ladder: the column's values (ValAvg) over the MLP | +0.116 [+0.107, +0.126] | +0.137 | +0.096 |
| full − valavg | ladder: message passing over ValAvg | -0.040 [-0.045, -0.034] | -0.047 | -0.032 |

Per seed (J_tf, point estimates):

- seed 1: full − lm_only +0.170; valavg − full +0.022
- seed 2: full − lm_only +0.141; valavg − full +0.066
- seed 3: full − lm_only +0.161; valavg − full +0.031

