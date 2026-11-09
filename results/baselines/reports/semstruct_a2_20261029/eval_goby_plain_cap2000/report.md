# SemStruct, the revised setup (Amendment 2): plain, evaluation row cap 2000

EXPLORATORY. Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples). Seeds present: full 1, valavg 1, onehop 1, mlp 1, lm_only 1, zero_delta 1, permuted 1

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 200 scenarios.

## all: 200 scenarios, families GOBY-Unionable 200; bootstrap over 124 clusters of scenarios that share a table

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.800 | 0.833 | 0.767 |
| valavg | 0.816 | 0.849 | 0.783 |
| onehop | 0.814 | 0.847 | 0.781 |
| mlp | 0.717 | 0.746 | 0.688 |
| lm_only | 0.734 | 0.745 | 0.724 |
| zero_delta | 0.734 | 0.745 | 0.724 |
| permuted | 0.800 | 0.833 | 0.767 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | +0.066 [+0.043, +0.088] | +0.088 | +0.043 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.014 [-0.000, +0.027] | +0.014 | +0.013 |
| valavg − full | the authors' ValAvg minus message passing | +0.015 [-0.003, +0.034] | +0.016 | +0.015 |
| onehop − lm_only | own values (one hop) over the header alone | +0.080 [+0.058, +0.101] | +0.103 | +0.057 |
| valavg − lm_only | ValAvg over the header alone | +0.081 [+0.064, +0.101] | +0.104 | +0.059 |
| full@0 − permuted | row co-occurrence (full, first seed: permuted exists for it only) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |
| mlp − lm_only | ladder: MLP on the column embedding over the header alone | -0.018 [-0.035, -0.001] | +0.001 | -0.037 |
| valavg − mlp | ladder: the column's values (ValAvg) over the MLP | +0.099 [+0.079, +0.121] | +0.103 | +0.095 |
| full − valavg | ladder: message passing over ValAvg | -0.015 [-0.035, +0.002] | -0.016 | -0.015 |

