# parallel (EXPLORATORY): the parallel-edge rule for FD edges, regime md5

- 549 run (SMX_VALENTINE=549): assembled rows; the gate and the schema-graph check were not repeated (they need the run folders)

J = (R_exp + F1_Hr)/2; per matcher the macro over families with ≥ 5 scenarios; mean over included non-control matchers; 95% bootstrap stratified by family, 10,000 resamples, seed 20261002. No verdicts.

## all (549 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 78 | +0.0044 [+0.0033, +0.0055] | -0.0005 | +0.0093 | 62 of 78 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 78 | +0.0502 [+0.0276, +0.0714] | +0.0118 | +0.0886 | 68 of 78 |
| fd − melnik | 78 | +0.0458 [+0.0232, +0.0667] | +0.0123 | +0.0794 | 68 of 78 |

## ChEMBL (180 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 78 | +0.0035 [+0.0022, +0.0048] | +0.0025 | +0.0046 | 71 of 78 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 78 | +0.0216 [+0.0138, +0.0297] | -0.0070 | +0.0502 | 63 of 78 |
| fd − melnik | 78 | +0.0181 [+0.0107, +0.0258] | -0.0095 | +0.0457 | 63 of 78 |

## Magellan (7 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | -0.0001 [-0.0014, +0.0014] | +0.0052 | -0.0054 | 41 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | +0.1460 [+0.0547, +0.2284] | +0.0948 | +0.1972 | 71 of 79 |
| fd − melnik | 79 | +0.1461 [+0.0552, +0.2281] | +0.0895 | +0.2026 | 72 of 79 |

## OpenData (180 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | -0.0007 [-0.0024, +0.0009] | -0.0037 | +0.0023 | 31 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | +0.0200 [+0.0155, +0.0245] | +0.0032 | +0.0369 | 69 of 79 |
| fd − melnik | 79 | +0.0208 [+0.0164, +0.0251] | +0.0069 | +0.0346 | 69 of 79 |

## TPC-DI (180 scenarios)

| contrast | matchers | estimate [95% CI] | R_exp | F1_Hr | matchers > 0 |
|---|---|---|---|---|---|
| fd_split − fd | 79 | +0.0150 [+0.0114, +0.0186] | -0.0057 | +0.0357 | 69 of 79 |
| fd_pavg − fd | 0 | – | – | – | – |
| fd_pmax − fd | 0 | – | – | – | – |
| fd_split − melnik | 79 | +0.0147 [+0.0108, +0.0185] | -0.0421 | +0.0715 | 67 of 79 |
| fd − melnik | 79 | -0.0003 [-0.0042, +0.0036] | -0.0364 | +0.0358 | 62 of 79 |

