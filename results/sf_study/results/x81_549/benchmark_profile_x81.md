# X81: X77 properties of the external corpora (EXPLORATORY)

Medians over scenarios. ovl = value-set Jaccard, name = name similarity, dist = profile similarity; true pairs vs sampled non-pairs. num share = columns ≥ 95% numeric; dtype H = entropy of (num, text, empty) in bits.

| corpus | scen. | cols (src×tgt) | true share | diag | ovl true / non | name true / non | dist true / non | num share | dtype H | FD/col | key share | IND/col |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| valentine_plain | 549 | 19×20 | 0.029 | 1.00 | 0.42 / 0.00 | 0.65 / 0.24 | 0.97 / 0.69 | 0.36 | 0.99 | 9.9 | 0.08 | 1.94 |
| t2d | 175 | 5×5 | 0.086 | 0.00 | 0.02 / 0.00 | 0.55 / 0.22 | 0.85 / 0.73 | 0.17 | 0.65 | 2.2 | 0.29 | 0.00 |
| final_controlled_plain | 48 | 19×18 | 0.041 | 0.07 | 0.79 / 0.00 | 1.00 / 0.42 | 1.00 / 0.70 | 0.51 | 1.00 | 8.6 | 0.04 | 0.24 |
| final_controlled_opaque | 48 | 19×18 | 0.041 | 0.07 | 0.00 / 0.00 | 0.27 / 0.27 | 1.00 / 0.77 | 0.18 | 0.47 | 8.6 | 0.04 | 0.00 |
| final_indep | 9 | 101×42 | 0.009 | nan | 0.03 / 0.00 | 0.62 / 0.21 | 0.86 / 0.63 | 0.28 | 1.28 | 2.3 | 0.39 | 1.14 |
| synthea | 39 | 8×18 | 0.062 | nan | 0.01 / 0.00 | 0.27 / 0.25 | 0.81 / 0.60 | 0.50 | 1.42 | 3.3 | 0.00 | 0.61 |
| gdcsm | 10 | 42×736 | 0.001 | 0.00 | 0.04 / 0.00 | 0.51 / 0.23 | 0.48 / 0.53 | 0.03 | 1.16 | 15.6 | 0.01 | 192.26 |
| hdxsm | 199 | 14×14 | 0.046 | 0.29 | 0.22 / 0.00 | 0.77 / 0.25 | 0.97 / 0.74 | 0.29 | 0.87 | 5.1 | 0.04 | 0.04 |
| goby | 200 | 11×11 | 0.054 | 0.12 | 0.00 / 0.00 | 0.76 / 0.27 | 0.85 / 0.69 | 0.08 | 0.55 | 5.7 | 0.00 | 0.44 |
| madi_indep | 12 | 9×8 | 0.093 | 0.60 | 0.16 / 0.00 | 0.42 / 0.25 | 0.91 / 0.69 | 0.17 | 0.65 | 2.8 | 0.13 | 0.00 |
| madi_products | 6 | 25×25 | 0.040 | 1.00 | 0.34 / 0.00 | 0.59 / 0.15 | 0.98 / 0.59 | 0.40 | 0.97 | 9.5 | 0.08 | 0.00 |
| mimic | 1 | 76×113 | 0.006 | nan | 0.26 / 0.00 | 0.38 / 0.29 | 0.83 / 0.66 | 0.44 | 1.45 | 5.5 | 0.10 | 0.90 |

549 run: valentine_plain = 381 stored + 168 new scenarios.
