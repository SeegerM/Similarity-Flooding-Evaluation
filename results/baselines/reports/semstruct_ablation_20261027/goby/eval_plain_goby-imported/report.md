# SemStruct information-matched ablation (plain, generated/goby-imported)

EXPLORATORY (SEMSTRUCT_ABLATION.md, Amendment 1). Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios stratified by family (2000 resamples).

Check: zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00 over 200 scenarios.

165 clusters of scenarios that share a table; bootstrap over clusters.
## all: 200 scenarios, families GOBY-Unionable 200

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| full | 0.508 | 0.494 | 0.522 |
| lm_only | 0.709 | 0.721 | 0.697 |
| zero_delta | 0.709 | 0.721 | 0.697 |
| full_permuted | 0.508 | 0.495 | 0.522 |
| onehop | 0.537 | 0.536 | 0.538 |
| onehop_permuted | 0.537 | 0.536 | 0.538 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| full − lm_only | gate: structure over the header alone | -0.201 [-0.230, -0.175] | -0.227 | -0.175 |
| onehop − full | information-matched: one hop (own values) minus message passing | +0.029 [+0.010, +0.049] | +0.043 | +0.016 |
| onehop − lm_only | own values over the header alone | -0.172 [-0.198, -0.145] | -0.185 | -0.160 |
| full − full_permuted | row co-occurrence (full) | -0.000 [-0.002, +0.001] | -0.001 | -0.000 |
| onehop − onehop_permuted | value order (one hop; should be ~0) | +0.000 [+0.000, +0.000] | +0.000 | +0.000 |

