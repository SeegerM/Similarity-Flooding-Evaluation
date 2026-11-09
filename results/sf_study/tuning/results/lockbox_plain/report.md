# Tuning report: lockbox_plain

3168 scored (seed, configuration, scenario) triples; families ['ChEMBL', 'Magellan', 'OpenData', 'TPC-DI']; 0 problems.

Selection objective: **J_tf** (Amendment 2, tie-free); the registered J is reported next to it.

## Leave-one-family-out

| seed | held-out family | chosen (training families) | held-out J_tf | production | no propagation | held-out J of the chosen |
|---|---|---|---|---|---|---|
| constant | ChEMBL | fpBASIC__INV_PROD__NAME__fddamped | +0.1183 | +0.0976 | +0.0000 | -0.2218 |
| constant | Magellan | fpBASIC__INV_PROD__NAME__fddamped | +0.2052 | +0.1501 | +0.0000 | -0.0667 |
| constant | OpenData | fpBASIC__INV_PROD__NAME__fddamped | +0.1247 | +0.0878 | +0.0000 | -0.2138 |
| constant | TPC-DI | fpBASIC__INV_PROD__NAME__fddamped | +0.0423 | +0.0403 | +0.0000 | -0.2512 |
| kang_naughton | ChEMBL | fpC__INV_PROD__NAME__fddamped | +0.0812 | +0.0603 | +0.0000 | +0.0652 |
| kang_naughton | Magellan | fpC__INV_PROD__NAME__fddamped | +0.0710 | -0.0207 | +0.0000 | +0.0250 |
| kang_naughton | OpenData | fpC__INV_PROD__NAME__fddamped | +0.0169 | +0.0107 | +0.0000 | +0.0153 |
| kang_naughton | TPC-DI | fpC__INV_PROD__NAME__fddamped | -0.0348 | -0.0475 | +0.0000 | -0.0041 |
| ks | ChEMBL | fpC__INV_PROD__ID__fdoff | -0.0063 | +0.0115 | +0.0000 | +0.0132 |
| ks | Magellan | fpC__INV_PROD__ID__fdoff | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| ks | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0614 | -0.0614 | +0.0000 | -0.0569 |
| ks | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0154 | +0.0273 | +0.0000 | -0.0103 |
| levenshtein | ChEMBL | fpC__INV_PROD__NAME__fddamped | +0.0147 | +0.0208 | +0.0000 | +0.0193 |
| levenshtein | Magellan | fpC__INV_PROD__NAME__fdfull | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| levenshtein | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0080 | -0.0080 | +0.0000 | +0.0050 |
| levenshtein | TPC-DI | fpC__INV_PROD__NAME__fdfull | +0.0018 | +0.0018 | +0.0000 | +0.0065 |
| oracle_noise050 | ChEMBL | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1402 | +0.0000 | +0.0000 |
| oracle_noise050 | Magellan | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1333 | +0.0000 | +0.0000 |
| oracle_noise050 | OpenData | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1392 | +0.0000 | +0.0000 |
| oracle_noise050 | TPC-DI | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.0041 | +0.0000 | +0.0000 |
| pmf | ChEMBL | fpC__INV_PROD__ID__fdoff | +0.0159 | +0.0083 | +0.0000 | -0.0099 |
| pmf | Magellan | fpC__INV_PROD__ID__fdoff | +0.0460 | -0.0540 | +0.0000 | +0.0000 |
| pmf | OpenData | fpC__INV_PROD__ID__fdoff | +0.0462 | -0.1105 | +0.0000 | +0.0643 |
| pmf | TPC-DI | fpC__INV_PROD__ID__fdoff | +0.0021 | -0.1556 | +0.0000 | +0.0022 |
| prefix_suffix | ChEMBL | fpC__INV_AVG__NAME__fddamped | +0.0529 | +0.0411 | +0.0000 | +0.0158 |
| prefix_suffix | Magellan | fpC__INV_AVG__NAME__fddamped | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| prefix_suffix | OpenData | fpC__INV_AVG__NAME__fddamped | +0.0078 | -0.0055 | +0.0000 | +0.0104 |
| prefix_suffix | TPC-DI | fpC__INV_AVG__NAME__fddamped | -0.0069 | -0.0071 | +0.0000 | -0.0060 |
| role_gcov | ChEMBL | fpC__INV_AVG__NAME__fddamped | +0.0854 | +0.0758 | +0.0000 | -0.0565 |
| role_gcov | Magellan | fpC__INV_AVG__NAME__fddamped | +0.1914 | +0.1501 | +0.0000 | -0.2333 |
| role_gcov | OpenData | fpC__INV_AVG__NAME__fddamped | +0.0853 | +0.0927 | +0.0000 | +0.0546 |
| role_gcov | TPC-DI | fpC__INV_AVG__NAME__fddamped | +0.0516 | +0.0462 | +0.0000 | -0.1173 |
| semstruct | ChEMBL | fpC__INV_PROD__ID__fdoff | +0.0012 | -0.1606 | -0.0000 | -0.0030 |
| semstruct | Magellan | fpC__INV_PROD__ID__fdoff | -0.3667 | -0.4667 | +0.0000 | -0.3667 |
| semstruct | OpenData | fpC__INV_PROD__ID__fdoff | -0.0218 | -0.2271 | -0.0023 | -0.0442 |
| semstruct | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0156 | -0.2389 | -0.0015 | -0.0230 |
| ALL_SEEDS | ChEMBL | fpC__INV_PROD__ID__fdoff | +0.0129 | +0.0016 | -0.0000 | -0.0066 |
| ALL_SEEDS | Magellan | fpC__INV_PROD__ID__fdoff | +0.0022 | -0.0416 | +0.0000 | -0.0398 |
| ALL_SEEDS | OpenData | fpC__INV_PROD__ID__fdoff | +0.0160 | -0.0401 | -0.0003 | +0.0170 |
| ALL_SEEDS | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0092 | -0.0375 | -0.0002 | -0.0214 |

## Restart alpha (production coefficient and dependency strength, ID attachment)

| seed | seed R@GT | α=0.00 | α=0.15 | α=0.30 | α=0.50 | α=0.70 | α=0.85 | α* |
|---|---|---|---|---|---|---|---|---|
| constant | 0.000 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| kang_naughton | 0.131 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| ks | 0.304 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| levenshtein | 0.522 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| oracle_noise050 | 1.000 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| pmf | 0.410 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| prefix_suffix | 0.481 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| role_gcov | 0.000 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| semstruct | 0.382 | -0.0010 | +nan | +nan | +nan | +nan | +nan | 0.00 |

Spearman(α*, seed quality) across seeds: +nan

| predictor | ρ(best α, predictor), pooled scenarios | ρ per held-out family | same sign in every fold |
|---|---|---|---|
| DOMINANTS | +nan | ChEMBL +nan, Magellan +nan, OpenData +nan, TPC-DI +nan | no |
| STDEV | +nan | ChEMBL +nan, Magellan +nan, OpenData +nan, TPC-DI +nan | no |
| MAXMEAN | +nan | ChEMBL +nan, Magellan +nan, OpenData +nan, TPC-DI +nan | no |

**Adaptive-α rule (pre-registered: |ρ| ≥ 0.5, same sign in every fold):** not met; α* is not predictable from the seed alone on this stage.

## Main effects (macro J_tf, mean over seeds)

| factor | level | mean macro J_tf |
|---|---|---|
| operator_value | fixpoint=BASIC | +0.1226 |
| operator_value | fixpoint=C | -0.0027 |
| coeff | INV_AVG | +0.0584 |
| coeff | INV_PROD | -0.0025 |
| attach | ID | +0.0055 |
| attach | NAME | +0.0008 |
| fd | damped | +0.0554 |
| fd | full | -0.0294 |
| fd | off | +0.0049 |

## Report only: SF's output re-scaled by its row and column maxima (pa)

Never used for selection (EXPLORATORY.md X6). Macro J_tf over all families, for the chosen configurations.

| seed | configuration | J_tf SF | J_tf SF × pa | difference |
|---|---|---|---|---|
| constant | fpBASIC__INV_PROD__NAME__fddamped | +0.1226 | +0.1283 | +0.0056 |
| constant | fpC__INV_PROD__NAME__fdfull | +0.0940 | +0.1115 | +0.0175 |
| kang_naughton | fpC__INV_PROD__NAME__fddamped | +0.0336 | +0.0433 | +0.0097 |
| kang_naughton | fpC__INV_PROD__NAME__fdfull | +0.0007 | +0.0216 | +0.0209 |
| ks | fpC__INV_PROD__NAME__fdfull | -0.0056 | +0.0298 | +0.0354 |
| levenshtein | fpC__INV_PROD__NAME__fddamped | +0.0038 | +0.0160 | +0.0122 |
| levenshtein | fpC__INV_PROD__NAME__fdfull | +0.0036 | +0.0154 | +0.0117 |
| oracle_noise050 | fpC__INV_PROD__NAME__fdfull | -0.1042 | -0.0730 | +0.0311 |
| oracle_noise050 | fpC__INV_PROD__NAME__fdoff | +0.0000 | +0.0025 | +0.0025 |
| pmf | fpC__INV_PROD__ID__fdoff | +0.0276 | +0.0369 | +0.0094 |
| pmf | fpC__INV_PROD__NAME__fdfull | -0.0780 | -0.0922 | -0.0143 |
| prefix_suffix | fpC__INV_AVG__NAME__fddamped | +0.0134 | +0.0259 | +0.0125 |
| prefix_suffix | fpC__INV_PROD__NAME__fdfull | +0.0071 | +0.0164 | +0.0093 |
| role_gcov | fpC__INV_AVG__NAME__fddamped | +0.1035 | +0.1117 | +0.0082 |
| role_gcov | fpC__INV_PROD__NAME__fdfull | +0.0912 | +0.1099 | +0.0187 |
| semstruct | fpC__INV_PROD__ID__fdoff | -0.1007 | -0.0212 | +0.0795 |
| semstruct | fpC__INV_PROD__NAME__fdfull | -0.2733 | -0.2624 | +0.0109 |

Mean difference +0.0165; pa better in 16 of 17 (seed, configuration) pairs.
