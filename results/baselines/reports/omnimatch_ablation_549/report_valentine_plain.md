# OmniMatch information-matched ablation (valentine_plain)

EXPLORATORY (OMNIMATCH_ABLATION.md). Each arm averaged over seeds [1, 2, 3] per scenario; paired differences; macro over families with >= 5 scenarios; 95% bootstrap (2,000).

## remaining505: 505 scenarios (ChEMBL 170, OpenData 168, TPC-DI 164)

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| F | 0.244 | 0.183 | 0.304 |
| N | 0.508 | 0.496 | 0.519 |
| A | 0.509 | 0.499 | 0.520 |
| RF | 0.522 | 0.549 | 0.496 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| F - N | gate: the edges carry something (F over no edges) | -0.264 [-0.278, -0.249] | -0.313 | -0.215 |
| A - F | information-matched: one-step edge term minus propagation | +0.266 [+0.251, +0.280] | +0.316 | +0.216 |
| RF - F | the authors' random forest on the pair similarities minus OmniMatch | +0.279 [+0.257, +0.300] | +0.366 | +0.192 |
| A - N | the one-step edge term over no edges | +0.002 [+0.001, +0.003] | +0.002 | +0.001 |

## all: 549 scenarios (ChEMBL 180, OpenData 180, TPC-DI 180, Magellan 7)

| arm | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| F | 0.364 | 0.274 | 0.453 |
| N | 0.604 | 0.573 | 0.636 |
| A | 0.606 | 0.575 | 0.637 |
| RF | 0.640 | 0.658 | 0.623 |

| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |
|---|---|---|---|---|
| F - N | gate: the edges carry something (F over no edges) | -0.241 [-0.267, -0.217] | -0.298 | -0.183 |
| A - F | information-matched: one-step edge term minus propagation | +0.242 [+0.220, +0.267] | +0.300 | +0.184 |
| RF - F | the authors' random forest on the pair similarities minus OmniMatch | +0.277 [+0.251, +0.304] | +0.384 | +0.170 |
| A - N | the one-step edge term over no edges | +0.001 [+0.001, +0.002] | +0.002 | +0.001 |

