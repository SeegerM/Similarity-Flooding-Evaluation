# Tuning report: screen_plain

58860 scored (seed, configuration, scenario) triples; families ['ChEMBL', 'Magellan', 'OpenData', 'TPC-DI']; 0 problems.

Selection objective: **J** (pre-registered).

## Leave-one-family-out

| seed | held-out family | chosen (training families) | held-out J | production | no propagation | held-out J_tf of the chosen |
|---|---|---|---|---|---|---|
| constant | ChEMBL | fpC__INV_PROD__ID__fdoff | -0.0165 | -0.2202 | +0.0000 | -0.0026 |
| constant | Magellan | fpC__INV_PROD__ID__fdoff | +0.0000 | +0.2000 | +0.0000 | +0.1760 |
| constant | OpenData | fpC__INV_PROD__ID__fdoff | +0.0000 | -0.2480 | +0.0000 | +0.0000 |
| constant | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0089 | -0.2970 | +0.0000 | -0.0000 |
| kang_naughton | ChEMBL | fpB__INV_PROD__ID__fdoff | +0.0676 | +0.0732 | +0.0000 | +0.0779 |
| kang_naughton | Magellan | fpC__INV_PROD__NAME__fddamped | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| kang_naughton | OpenData | fpC__INV_PROD__NAME__fddamped | +0.0392 | +0.0212 | +0.0000 | +0.0381 |
| kang_naughton | TPC-DI | fpB__INV_PROD__ID__fdoff | +0.0106 | +0.0595 | +0.0000 | +0.0119 |
| ks | ChEMBL | fpC__INV_PROD__NAME__fdfull | +0.0208 | +0.0208 | +0.0000 | +0.0044 |
| ks | Magellan | fpA__INV_PROD__ID__fdoff | +0.0000 | +0.2000 | +0.0000 | +0.0000 |
| ks | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0533 | -0.0533 | +0.0000 | -0.0645 |
| ks | TPC-DI | fpC__INV_PROD__NAME__fdfull | +0.0377 | +0.0377 | +0.0000 | +0.0561 |
| levenshtein | ChEMBL | fpC__INV_PROD__NAME__fdoff | +0.0281 | +0.0238 | +0.0000 | -0.0185 |
| levenshtein | Magellan | fpC__INV_PROD__NAME__fdoff | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| levenshtein | OpenData | fpB__INV_PROD__NAME__fdfull | +0.0096 | +0.0096 | +0.0000 | -0.0277 |
| levenshtein | TPC-DI | fpC__INV_PROD__NAME__fdoff | +0.0013 | +0.0045 | +0.0000 | -0.0001 |
| oracle_noise050 | ChEMBL | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1489 | +0.0000 | +0.0000 |
| oracle_noise050 | Magellan | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1000 | +0.0000 | +0.0000 |
| oracle_noise050 | OpenData | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1388 | +0.0000 | +0.0000 |
| oracle_noise050 | TPC-DI | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.0061 | +0.0000 | +0.0000 |
| pmf | ChEMBL | r0.15__INV_AVG__ID__fdfull | -0.0159 | +0.0281 | +0.0000 | +0.0193 |
| pmf | Magellan | fpC__INV_PROD__ID__fdoff | -0.1000 | +0.2000 | +0.0000 | -0.1000 |
| pmf | OpenData | r0.15__INV_AVG__ID__fdfull | -0.0326 | -0.0900 | +0.0000 | -0.0534 |
| pmf | TPC-DI | fpBASIC__INV_AVG__NAME__fddamped | -0.4772 | -0.0784 | +0.0000 | -0.4462 |
| prefix_suffix | ChEMBL | fpC__INV_PROD__NAME__fddamped | -0.0581 | -0.0410 | +0.0000 | +0.0301 |
| prefix_suffix | Magellan | fpB__INV_PROD__NAME__fddamped | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| prefix_suffix | OpenData | fpC__INV_AVG__NAME__fddamped | -0.0329 | -0.0402 | +0.0000 | -0.0297 |
| prefix_suffix | TPC-DI | fpB__INV_PROD__NAME__fddamped | -0.0022 | -0.0041 | +0.0000 | -0.0005 |
| role_gcov | ChEMBL | fpB__INV_PROD__NAME__fdfull | -0.0273 | -0.0205 | +0.0000 | +0.0623 |
| role_gcov | Magellan | fpC__INV_PROD__ID__fddamped | +0.2000 | +0.2000 | +0.0000 | +0.5000 |
| role_gcov | OpenData | fpC__INV_PROD__ID__fddamped | +0.0449 | +0.0635 | +0.0000 | +0.0841 |
| role_gcov | TPC-DI | fpB__INV_AVG__NAME__fdfull | -0.1258 | -0.1071 | +0.0000 | +0.0474 |
| semstruct | ChEMBL | fpC__INV_PROD__ID__fdoff | -0.0060 | -0.1227 | +0.0000 | +0.0002 |
| semstruct | Magellan | fpC__INV_PROD__ID__fdoff | -0.3000 | -0.3000 | +0.0000 | -0.3000 |
| semstruct | OpenData | fpC__INV_AVG__NAME__fdoff | -0.1781 | -0.3188 | +0.0015 | -0.1774 |
| semstruct | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0077 | -0.3375 | +0.0000 | -0.0007 |
| ALL_SEEDS | ChEMBL | fpC__INV_PROD__ID__fdoff | -0.0053 | -0.0453 | +0.0000 | +0.0066 |
| ALL_SEEDS | Magellan | fpC__INV_PROD__ID__fdoff | -0.0667 | +0.0444 | +0.0000 | -0.0284 |
| ALL_SEEDS | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0883 | -0.0883 | +0.0002 | -0.0547 |
| ALL_SEEDS | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0207 | -0.0810 | +0.0000 | -0.0044 |

## Restart alpha (production coefficient and dependency strength, ID attachment)

| seed | seed R@GT | α=0.00 | α=0.15 | α=0.30 | α=0.50 | α=0.70 | α=0.85 | α* |
|---|---|---|---|---|---|---|---|---|
| constant | 0.000 | +0.0000 | -0.2752 | -0.2759 | -0.2825 | -0.2798 | -0.3105 | 0.00 |
| kang_naughton | 0.133 | +0.0000 | +0.0164 | +0.0152 | -0.0380 | -0.0546 | -0.0925 | 0.15 |
| ks | 0.337 | +0.0000 | -0.0045 | -0.0110 | -0.0333 | -0.0510 | -0.0585 | 0.00 |
| levenshtein | 0.532 | +0.0000 | +0.0049 | +0.0057 | +0.0138 | +0.0105 | -0.0206 | 0.50 |
| oracle_noise050 | 1.000 | +0.0000 | -0.0038 | -0.0052 | -0.0073 | -0.0273 | -0.0466 | 0.00 |
| pmf | 0.403 | +0.0000 | +0.0103 | +0.0127 | -0.0038 | -0.0168 | -0.0185 | 0.30 |
| prefix_suffix | 0.538 | +0.0000 | -0.0257 | -0.0229 | -0.0239 | -0.0335 | -0.0501 | 0.00 |
| role_gcov | 0.000 | +0.0000 | -0.1084 | -0.1104 | -0.1344 | -0.1670 | -0.1704 | 0.00 |
| semstruct | 0.471 | +0.0004 | -0.4065 | -0.4093 | -0.4139 | -0.4177 | -0.4478 | 0.00 |

Spearman(α*, seed quality) across seeds: +0.080

| predictor | ρ(best α, predictor), pooled scenarios | ρ per held-out family | same sign in every fold |
|---|---|---|---|
| DOMINANTS | -0.325 | ChEMBL -0.33, Magellan -0.32, OpenData -0.31, TPC-DI -0.34 | yes |
| STDEV | +0.023 | ChEMBL +0.01, Magellan +0.03, OpenData +0.03, TPC-DI +0.02 | yes |
| MAXMEAN | -0.312 | ChEMBL -0.29, Magellan -0.31, OpenData -0.26, TPC-DI -0.42 | yes |

**Adaptive-α rule (pre-registered: |ρ| ≥ 0.5, same sign in every fold):** not met; α* is not predictable from the seed alone on this stage.

## Main effects (macro J, mean over seeds)

| factor | level | mean macro J |
|---|---|---|
| operator_value | fixpoint=A | -0.0709 |
| operator_value | fixpoint=B | -0.1035 |
| operator_value | fixpoint=BASIC | -0.2437 |
| operator_value | fixpoint=C | -0.0669 |
| operator_value | restart=0.15 | -0.1388 |
| operator_value | restart=0.3 | -0.1397 |
| operator_value | restart=0.5 | -0.1456 |
| operator_value | restart=0.7 | -0.1510 |
| operator_value | restart=0.85 | -0.1597 |
| coeff | INV_AVG | -0.1448 |
| coeff | INV_PROD | -0.1263 |
| attach | ID | -0.1133 |
| attach | NAME | -0.1578 |
| fd | damped | -0.1358 |
| fd | full | -0.1567 |
| fd | off | -0.1141 |

keep.json: 168 (seed, configuration) pairs go to dev.
