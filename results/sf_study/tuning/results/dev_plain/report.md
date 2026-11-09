# Tuning report: dev_plain

51480 scored (seed, configuration, scenario) triples; families ['ChEMBL', 'Magellan', 'OpenData', 'TPC-DI', 'Wikidata']; 0 problems.

Selection objective: **J_tf** (Amendment 2, tie-free); the registered J is reported next to it.

## Leave-one-family-out

| seed | held-out family | chosen (training families) | held-out J_tf | production | no propagation | held-out J of the chosen |
|---|---|---|---|---|---|---|
| constant | ChEMBL | fpBASIC__INV_PROD__NAME__fddamped | +0.1118 | +0.0971 | +0.0000 | -0.2197 |
| constant | Magellan | fpA__INV_AVG__NAME__fddamped | +0.3649 | +0.3638 | +0.0000 | +0.0967 |
| constant | OpenData | fpA__INV_AVG__NAME__fddamped | +0.1156 | +0.0883 | +0.0000 | -0.2242 |
| constant | TPC-DI | fpBASIC__INV_PROD__NAME__fddamped | +0.0525 | +0.0407 | +0.0000 | -0.2548 |
| constant | Wikidata | fpBASIC__INV_PROD__NAME__fddamped | +0.1744 | +0.1374 | +0.0000 | +0.0941 |
| kang_naughton | ChEMBL | fpC__INV_PROD__NAME__fddamped | +0.0852 | +0.0634 | +0.0000 | +0.0699 |
| kang_naughton | Magellan | fpC__INV_PROD__NAME__fddamped | +0.0213 | -0.0148 | +0.0000 | +0.0313 |
| kang_naughton | OpenData | fpC__INV_PROD__NAME__fddamped | +0.0296 | +0.0185 | +0.0000 | +0.0290 |
| kang_naughton | TPC-DI | fpC__INV_PROD__NAME__fddamped | +0.0186 | +0.0103 | +0.0000 | +0.0459 |
| kang_naughton | Wikidata | fpB__INV_PROD__ID__fdoff | +0.0703 | +0.1314 | +0.0000 | +0.1134 |
| ks | ChEMBL | fpC__INV_PROD__NAME__fdfull | +0.0209 | +0.0209 | +0.0000 | +0.0299 |
| ks | Magellan | fpC__INV_PROD__NAME__fdoff | +0.0000 | +0.0708 | +0.0000 | +0.0000 |
| ks | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0682 | -0.0682 | +0.0000 | -0.0601 |
| ks | TPC-DI | fpC__INV_PROD__NAME__fdoff | -0.0107 | +0.0401 | +0.0000 | -0.0086 |
| ks | Wikidata | fpC__INV_PROD__NAME__fdfull | -0.0558 | -0.0558 | +0.0000 | -0.0558 |
| levenshtein | ChEMBL | fpC__INV_PROD__NAME__fdoff | -0.0115 | +0.0089 | +0.0000 | +0.0176 |
| levenshtein | Magellan | fpA__INV_PROD__ID__fddamped | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| levenshtein | OpenData | r0.50__INV_PROD__ID__fdfull | -0.0033 | -0.0149 | +0.0000 | +0.0106 |
| levenshtein | TPC-DI | fpA__INV_PROD__ID__fddamped | +0.0016 | -0.0076 | +0.0000 | +0.0051 |
| levenshtein | Wikidata | fpA__INV_PROD__ID__fddamped | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| oracle_noise050 | ChEMBL | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1241 | +0.0000 | +0.0000 |
| oracle_noise050 | Magellan | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1083 | +0.0000 | +0.0000 |
| oracle_noise050 | OpenData | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.1399 | +0.0000 | +0.0000 |
| oracle_noise050 | TPC-DI | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.0052 | +0.0000 | +0.0000 |
| oracle_noise050 | Wikidata | fpC__INV_PROD__NAME__fdoff | +0.0000 | -0.0417 | +0.0000 | +0.0000 |
| pmf | ChEMBL | fpC__INV_PROD__ID__fdoff | +0.0198 | +0.0290 | +0.0000 | -0.0117 |
| pmf | Magellan | fpC__INV_PROD__ID__fdoff | +0.0309 | +0.0959 | +0.0000 | +0.0086 |
| pmf | OpenData | fpC__INV_PROD__ID__fdoff | +0.0520 | -0.1145 | +0.0000 | +0.0737 |
| pmf | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0000 | -0.1265 | +0.0000 | -0.0002 |
| pmf | Wikidata | fpC__INV_PROD__ID__fdoff | +0.0045 | -0.1667 | +0.0000 | -0.0227 |
| prefix_suffix | ChEMBL | fpC__INV_PROD__NAME__fddamped | +0.0129 | +0.0201 | +0.0000 | -0.0123 |
| prefix_suffix | Magellan | fpC__INV_AVG__NAME__fddamped | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| prefix_suffix | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0005 | -0.0005 | +0.0000 | -0.0008 |
| prefix_suffix | TPC-DI | fpC__INV_AVG__NAME__fddamped | -0.0166 | -0.0035 | +0.0000 | -0.0189 |
| prefix_suffix | Wikidata | fpC__INV_AVG__NAME__fddamped | -0.0275 | -0.0339 | +0.0000 | -0.0370 |
| role_gcov | ChEMBL | fpC__INV_AVG__NAME__fddamped | +0.0781 | +0.0614 | +0.0000 | -0.0357 |
| role_gcov | Magellan | fpC__INV_AVG__NAME__fddamped | +0.2839 | +0.3074 | +0.0000 | +0.1079 |
| role_gcov | OpenData | fpC__INV_AVG__NAME__fddamped | +0.0952 | +0.0996 | +0.0000 | +0.0525 |
| role_gcov | TPC-DI | fpC__INV_AVG__NAME__fddamped | +0.0644 | +0.0498 | +0.0000 | -0.1123 |
| role_gcov | Wikidata | fpC__INV_AVG__NAME__fddamped | +0.1468 | +0.0797 | +0.0000 | +0.1132 |
| semstruct | ChEMBL | fpC__INV_PROD__ID__fdoff | +0.0005 | -0.1859 | -0.0003 | -0.0052 |
| semstruct | Magellan | fpC__INV_PROD__ID__fdoff | -0.2271 | -0.3583 | +0.0000 | -0.2271 |
| semstruct | OpenData | fpC__INV_PROD__ID__fdoff | -0.0205 | -0.2593 | -0.0014 | -0.0442 |
| semstruct | TPC-DI | fpC__INV_PROD__ID__fdoff | -0.0107 | -0.3239 | -0.0006 | -0.0196 |
| semstruct | Wikidata | fpC__INV_PROD__ID__fdoff | +0.0000 | -0.2758 | +0.0000 | +0.0000 |
| ALL_SEEDS | ChEMBL | fpC__INV_PROD__NAME__fdfull | -0.0010 | -0.0010 | -0.0000 | -0.0566 |
| ALL_SEEDS | Magellan | fpC__INV_PROD__NAME__fdfull | +0.0396 | +0.0396 | +0.0000 | -0.0208 |
| ALL_SEEDS | OpenData | fpC__INV_PROD__NAME__fdfull | -0.0434 | -0.0434 | -0.0002 | -0.0811 |
| ALL_SEEDS | TPC-DI | fpC__INV_PROD__NAME__fdfull | -0.0362 | -0.0362 | -0.0001 | -0.0855 |
| ALL_SEEDS | Wikidata | fpC__INV_PROD__NAME__fdfull | -0.0250 | -0.0250 | +0.0000 | -0.0369 |

## Restart alpha (production coefficient and dependency strength, ID attachment)

| seed | seed R@GT | α=0.00 | α=0.15 | α=0.30 | α=0.50 | α=0.70 | α=0.85 | α* |
|---|---|---|---|---|---|---|---|---|
| constant | 0.000 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| kang_naughton | 0.124 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| ks | 0.309 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| levenshtein | 0.538 | +0.0000 | +nan | +nan | +0.0018 | -0.0189 | +nan | 0.50 |
| oracle_noise050 | 1.000 | +0.0000 | +nan | -0.0062 | +nan | +nan | +nan | 0.00 |
| pmf | 0.412 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| prefix_suffix | 0.549 | +0.0000 | +nan | +nan | -0.0093 | +nan | +nan | 0.00 |
| role_gcov | 0.000 | +0.0000 | +nan | +nan | +nan | +nan | +nan | 0.00 |
| semstruct | 0.460 | -0.0005 | +nan | +nan | +nan | +nan | +nan | 0.00 |

Spearman(α*, seed quality) across seeds: +0.275

| predictor | ρ(best α, predictor), pooled scenarios | ρ per held-out family | same sign in every fold |
|---|---|---|---|
| DOMINANTS | +nan | ChEMBL +nan, Magellan +nan, OpenData +nan, TPC-DI +nan, Wikidata +nan | no |
| STDEV | +nan | ChEMBL +nan, Magellan +nan, OpenData +nan, TPC-DI +nan, Wikidata +nan | no |
| MAXMEAN | +nan | ChEMBL +nan, Magellan +nan, OpenData +nan, TPC-DI +nan, Wikidata +nan | no |

**Adaptive-α rule (pre-registered: |ρ| ≥ 0.5, same sign in every fold):** not met; α* is not predictable from the seed alone on this stage.

## Main effects (macro J_tf, mean over seeds)

| factor | level | mean macro J_tf |
|---|---|---|
| operator_value | fixpoint=A | +0.0222 |
| operator_value | fixpoint=B | +0.0072 |
| operator_value | fixpoint=BASIC | +0.0019 |
| operator_value | fixpoint=C | +0.0090 |
| operator_value | restart=0.15 | -0.0238 |
| operator_value | restart=0.3 | -0.0102 |
| operator_value | restart=0.5 | -0.0095 |
| operator_value | restart=0.7 | -0.0204 |
| operator_value | restart=0.85 | -0.0291 |
| coeff | INV_AVG | +0.0109 |
| coeff | INV_PROD | +0.0024 |
| attach | ID | +0.0022 |
| attach | NAME | +0.0106 |
| fd | damped | +0.0250 |
| fd | full | +0.0210 |
| fd | off | -0.0250 |

chosen.json: universal configuration fpC__INV_PROD__NAME__fdfull; per seed: constant: fpBASIC__INV_PROD__NAME__fddamped, fpC__INV_PROD__NAME__fdfull; kang_naughton: fpC__INV_PROD__NAME__fddamped, fpC__INV_PROD__NAME__fdfull; ks: fpC__INV_PROD__NAME__fdfull; levenshtein: fpC__INV_PROD__NAME__fddamped, fpC__INV_PROD__NAME__fdfull; oracle_noise050: fpC__INV_PROD__NAME__fdfull, fpC__INV_PROD__NAME__fdoff; pmf: fpC__INV_PROD__ID__fdoff, fpC__INV_PROD__NAME__fdfull; prefix_suffix: fpC__INV_AVG__NAME__fddamped, fpC__INV_PROD__NAME__fdfull; role_gcov: fpC__INV_AVG__NAME__fddamped, fpC__INV_PROD__NAME__fdfull; semstruct: fpC__INV_PROD__ID__fdoff, fpC__INV_PROD__NAME__fdfull

## Report only: SF's output re-scaled by its row and column maxima (pa)

Never used for selection (EXPLORATORY.md X6). Macro J_tf over all families, for the chosen configurations.

| seed | configuration | J_tf SF | J_tf SF × pa | difference |
|---|---|---|---|---|
| constant | fpBASIC__INV_PROD__NAME__fddamped | +0.1672 | +0.1774 | +0.0102 |
| constant | fpC__INV_PROD__NAME__fdfull | +0.1455 | +0.1602 | +0.0147 |
| kang_naughton | fpC__INV_PROD__NAME__fddamped | +0.0572 | +0.0806 | +0.0233 |
| kang_naughton | fpC__INV_PROD__NAME__fdfull | +0.0417 | +0.0561 | +0.0144 |
| ks | fpC__INV_PROD__NAME__fdfull | +0.0016 | +0.0352 | +0.0336 |
| levenshtein | fpC__INV_PROD__NAME__fddamped | +0.0001 | +0.0194 | +0.0193 |
| levenshtein | fpC__INV_PROD__NAME__fdfull | -0.0027 | +0.0256 | +0.0284 |
| oracle_noise050 | fpC__INV_PROD__NAME__fdfull | -0.0838 | -0.0558 | +0.0280 |
| oracle_noise050 | fpC__INV_PROD__NAME__fdoff | +0.0000 | +0.0027 | +0.0027 |
| pmf | fpC__INV_PROD__ID__fdoff | +0.0214 | +0.0330 | +0.0115 |
| pmf | fpC__INV_PROD__NAME__fdfull | -0.0566 | -0.0324 | +0.0241 |
| prefix_suffix | fpC__INV_AVG__NAME__fddamped | -0.0007 | +0.0116 | +0.0123 |
| prefix_suffix | fpC__INV_PROD__NAME__fdfull | -0.0036 | -0.0022 | +0.0013 |
| role_gcov | fpC__INV_AVG__NAME__fddamped | +0.1337 | +0.1566 | +0.0229 |
| role_gcov | fpC__INV_PROD__NAME__fdfull | +0.1196 | +0.1467 | +0.0271 |
| semstruct | fpC__INV_PROD__ID__fdoff | -0.0516 | -0.0074 | +0.0441 |
| semstruct | fpC__INV_PROD__NAME__fdfull | -0.2806 | -0.2469 | +0.0337 |

Mean difference +0.0207; pa better in 17 of 17 (seed, configuration) pairs.
