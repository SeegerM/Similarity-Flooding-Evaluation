# SemStruct, the revised setup (Amendment 2): audited_md5, evaluation row cap 2000

EXPLORATORY. Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples). Seeds present: full 1, valavg 1, onehop 1, lm_only 1, zero_delta 1, permuted 1

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 549 scenarios.

## remaining505: 505 scenarios, families ChEMBL 170, OpenData 168, TPC-DI 164

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.035 | 0.032 | 0.038 |
| valavg | 0.035 | 0.030 | 0.040 |
| onehop | 0.036 | 0.036 | 0.036 |
| lm_only | 0.025 | 0.023 | 0.027 |
| zero_delta | 0.025 | 0.023 | 0.027 |
| permuted | 0.035 | 0.032 | 0.038 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.010 [+0.006, +0.014] | +0.008 | +0.011 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.001 [-0.004, +0.006] | +0.004 | -0.002 |
| valavg − full | the authors' ValAvg minus message passing | +0.000 [-0.004, +0.005] | -0.001 | +0.002 |
| onehop − lm_only | own values (one hop) over the header alone | +0.011 [+0.007, +0.015] | +0.013 | +0.009 |
| valavg − lm_only | ValAvg over the header alone | +0.010 [+0.007, +0.013] | +0.007 | +0.013 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | -0.000 [-0.000, +0.000] | +0.000 | -0.000 |
| full − valavg | ladder: message passing over ValAvg | -0.000 [-0.004, +0.004] | +0.001 | -0.002 |

## all: 549 scenarios, families ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.054 | 0.051 | 0.057 |
| valavg | 0.040 | 0.038 | 0.042 |
| onehop | 0.052 | 0.049 | 0.054 |
| lm_only | 0.043 | 0.040 | 0.047 |
| zero_delta | 0.043 | 0.040 | 0.047 |
| permuted | 0.054 | 0.051 | 0.057 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.011 [-0.000, +0.022] | +0.012 | +0.010 |
| onehop − full | information-matched: one hop (own values) minus message passing | -0.003 [-0.021, +0.013] | -0.003 | -0.002 |
| valavg − full | the authors' ValAvg minus message passing | -0.014 [-0.028, -0.000] | -0.013 | -0.015 |
| onehop − lm_only | own values (one hop) over the header alone | +0.008 [-0.013, +0.026] | +0.009 | +0.007 |
| valavg − lm_only | ValAvg over the header alone | -0.004 [-0.024, +0.011] | -0.002 | -0.005 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | -0.000 [-0.000, +0.000] | +0.000 | -0.000 |
| full − valavg | ladder: message passing over ValAvg | +0.014 [-0.000, +0.029] | +0.013 | +0.015 |

