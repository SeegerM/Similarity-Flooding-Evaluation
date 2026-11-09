# X85: the new corpora characterised as Valentine (EXPLORATORY)

Properties from X77/X81 (medians and quartiles over scenarios); constant matcher re-scored here. Predictions: EXPLORATORY.md X85.

## 1. Sources and shape

| corpus | scen. | distinct source / target tables | scen. per distinct table | tables/side | cols src×tgt (median) | rows src/tgt (median) | size ratio |
|---|---|---|---|---|---|---|---|
| valentine_plain | 382 | 287 / 307 | 1.3 | 1/1 | 19×20 | 11627/11250 | 1.00 |
| t2d | 175 | 169 / 175 | 1.0 | 1/1 | 5×5 | 100/132 | 0.75 |
| gdcsm | 10 | 10 / 1 | 1.8 | 1/1 | 42×736 | 131/4478 | 0.06 |
| hdxsm | 199 | 69 / 98 | 2.4 | 1/1 | 14×14 | 100/100 | 0.72 |
| goby | 200 | 182 / 183 | 1.1 | 1/1 | 11×11 | 634/560 | 0.77 |
| madi_indep | 12 | 8 / 8 | 1.5 | 1/1 | 9×8 | 20000/7877 | 0.77 |
| madi_products | 6 | 3 / 3 | 2.0 | 1/1 | 25×25 | 812/694 | 1.00 |
| mimic | 1 | 1 / 1 | 1.0 | 7/6 | 76×113 | 44477/70101 | 0.67 |

Distinct tables are counted by content (all CSV files of a side). Valentine's 382 scenarios come from 10 source tables (X77); its sides are distinct fragments, so the content count overstates its independence.

## 2. Position artefact (diagonal truth)

| corpus | diag share median | scen. with diag ≥ 0.9 | constant F1_H (index order) | constant F1_Hr (random ties) | gap |
|---|---|---|---|---|---|
| valentine_plain | 1.00 | 99% | 0.723 | 0.035 | +0.688 |
| t2d | 0.00 | 13% | 0.186 | 0.130 | +0.056 |
| gdcsm | 0.00 | 0% | 0.000 | 0.001 | -0.001 |
| hdxsm | 0.29 | 12% | 0.329 | 0.055 | +0.274 |
| goby | 0.12 | 0% | 0.117 | 0.065 | +0.052 |
| madi_indep | 0.60 | 8% | 0.580 | 0.092 | +0.488 |
| madi_products | 1.00 | 100% | 1.000 | 0.038 | +0.962 |
| mimic | nan | nan% | 0.000 | 0.006 | -0.006 |

## 3. Heterogeneity within each corpus

| corpus | ovl_true q25/q50/q75 | name_true q25/q50/q75 | split-like (ovl ≥ 0.3, name ≥ 0.9) | name_true = 1 | dist_true − dist_non |
|---|---|---|---|---|---|
| valentine_plain | 0.25/0.42/0.69 | 0.52/0.65/0.76 | 12% | 18% | +0.29 |
| t2d | 0.00/0.02/0.10 | 0.33/0.55/0.75 | 0% | 10% | +0.12 |
| gdcsm | 0.04/0.04/0.10 | 0.48/0.51/0.56 | 0% | 0% | -0.04 |
| hdxsm | 0.15/0.22/0.32 | 0.65/0.77/0.95 | 9% | 10% | +0.23 |
| goby | 0.00/0.00/0.00 | 0.64/0.76/0.85 | 0% | 3% | +0.15 |
| madi_indep | 0.13/0.16/0.20 | 0.29/0.42/0.53 | 0% | 0% | +0.22 |
| madi_products | 0.34/0.34/0.36 | 0.54/0.59/0.65 | 0% | 0% | +0.39 |
| mimic | 0.26/0.26/0.26 | 0.38/0.38/0.38 | 0% | 0% | +0.18 |

## 4. By domain (MaDI use cases)

| domain | scen. | ovl_true | name_true | diag | num share | constant F1_H − F1_Hr |
|---|---|---|---|---|---|---|
| companies | 3 | 0.18 | 0.26 | 0.50 | 0.15 | +0.304 |
| games | 3 | 0.15 | 0.49 | 0.67 | 0.17 | +0.582 |
| music | 3 | 0.13 | 0.29 | 0.60 | 0.14 | +0.585 |
| papers | 3 | 0.23 | 0.61 | 0.60 | 0.46 | +0.482 |
| products | 6 | 0.34 | 0.59 | 1.00 | 0.40 | +0.962 |

## HDXSM, the most split-like pairs (by ovl_true)

| pair | ovl_true | name_true | diag | cols |
|---|---|---|---|---|
| pair_15 | 0.69 | 0.81 | 0.44 | 27×10 |
| pair_74 | 0.67 | 1.00 | 1.00 | 9×9 |
| pair_47 | 0.65 | 1.00 | 0.18 | 96×87 |
| pair_159 | 0.63 | 1.00 | 0.77 | 13×13 |
| pair_52 | 0.62 | 1.00 | 0.15 | 93×95 |
| pair_43 | 0.61 | 1.00 | 0.07 | 96×95 |
| pair_50 | 0.61 | 1.00 | 0.35 | 96×84 |
| pair_41 | 0.58 | 1.00 | 0.60 | 10×10 |
