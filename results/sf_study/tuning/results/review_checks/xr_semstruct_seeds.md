# SemStruct reconstruction: per-seed values (EXPLORATORY, external review 2026-11-03)

remaining505, plain, cap 2000; macro over ChEMBL, OpenData, TPC-DI (502 scenarios).

| arm | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| full | 0.618 | 0.578 | 0.615 | 0.604 |
| valavg | 0.648 | 0.666 | 0.648 | 0.654 |
| onehop | 0.605 | – | – | 0.605 |
| lm_only | 0.391 | – | – | 0.391 |
| mlp | 0.498 | – | – | 0.498 |

| per seed | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| value-average − full | +0.030 | +0.089 | +0.033 |
| full − header-only | +0.227 | +0.187 | +0.224 |

one-hop − full, first seed only: -0.013
