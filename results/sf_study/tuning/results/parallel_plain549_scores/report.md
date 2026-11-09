# parallel (EXPLORATORY): the parallel-edge rule for FD edges, regime plain

- 549 run (SMX_VALENTINE=549): assembled rows; the gate and the schema-graph check were not repeated (they need the run folders)

J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over included non-control matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. No verdicts.

## all (549 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 78 | -0.0215 [-0.0265, -0.0176] | -0.0407 | -0.0023 | 5 of 78 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 78 | -0.1302 [-0.1504, -0.1106] | -0.2440 | -0.0163 | 6 of 78 |
| fd − melnik | 78 | -0.1087 [-0.1290, -0.0899] | -0.2033 | -0.0140 | 6 of 78 |

## ChEMBL (180 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | -0.0009 [-0.0023, +0.0004] | -0.0040 | +0.0022 | 33 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | -0.0800 [-0.0892, -0.0708] | -0.1627 | +0.0027 | 8 of 79 |
| fd − melnik | 79 | -0.0791 [-0.0879, -0.0704] | -0.1587 | +0.0005 | 8 of 79 |

## Magellan (7 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | -0.0131 [-0.0309, +0.0001] | -0.0187 | -0.0075 | 9 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | -0.1323 [-0.2113, -0.0569] | -0.2422 | -0.0224 | 8 of 79 |
| fd − melnik | 79 | -0.1192 [-0.1995, -0.0465] | -0.2235 | -0.0149 | 9 of 79 |

## OpenData (180 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | -0.0129 [-0.0150, -0.0108] | -0.0152 | -0.0106 | 9 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | -0.1661 [-0.1815, -0.1510] | -0.2840 | -0.0483 | 2 of 79 |
| fd − melnik | 79 | -0.1533 [-0.1686, -0.1384] | -0.2688 | -0.0377 | 2 of 79 |

## TPC-DI (180 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | -0.0579 [-0.0644, -0.0518] | -0.1228 | +0.0070 | 5 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | -0.1325 [-0.1437, -0.1216] | -0.2742 | +0.0092 | 9 of 79 |
| fd − melnik | 79 | -0.0746 [-0.0843, -0.0652] | -0.1514 | +0.0022 | 13 of 79 |

