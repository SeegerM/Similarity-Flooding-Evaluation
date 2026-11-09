# SemStruct, the revised setup (Amendment 2): plain, evaluation row cap 2000, no merging

EXPLORATORY. Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples). Seeds present: full 1, valavg 1, lm_only 1, zero_delta 1

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 549 scenarios.

## remaining505: 505 scenarios, families ChEMBL 170, OpenData 168, TPC-DI 164

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.577 | 0.614 | 0.539 |
| valavg | 0.641 | 0.688 | 0.594 |
| lm_only | 0.391 | 0.437 | 0.345 |
| zero_delta | 0.391 | 0.437 | 0.345 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.186 [+0.166, +0.206] | +0.178 | +0.194 |
| valavg − full | the authors' ValAvg minus message passing | +0.064 [+0.051, +0.076] | +0.073 | +0.054 |
| valavg − lm_only | ValAvg over the header alone | +0.250 [+0.230, +0.268] | +0.251 | +0.248 |
| full − valavg | ladder: message passing over ValAvg | -0.064 [-0.075, -0.052] | -0.073 | -0.054 |

## all: 549 scenarios, families ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.679 | 0.702 | 0.656 |
| valavg | 0.730 | 0.763 | 0.698 |
| lm_only | 0.544 | 0.576 | 0.511 |
| zero_delta | 0.544 | 0.576 | 0.511 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.135 [+0.119, +0.150] | +0.126 | +0.145 |
| valavg − full | the authors' ValAvg minus message passing | +0.051 [+0.041, +0.062] | +0.060 | +0.042 |
| valavg − lm_only | ValAvg over the header alone | +0.186 [+0.173, +0.200] | +0.187 | +0.186 |
| full − valavg | ladder: message passing over ValAvg | -0.051 [-0.062, -0.041] | -0.060 | -0.042 |

