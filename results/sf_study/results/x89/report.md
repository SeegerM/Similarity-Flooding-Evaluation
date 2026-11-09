# X89: RQ1 across corpora: coupling checks and the near-zero corpora (EXPLORATORY)

x = R_exp of the initial matcher, d = R_exp gain of SF as published; per matcher, macro over groups.

## Coupling checks and reliability

| corpus | matchers | ρ(x,d) | Oldham ρ | families (n) ρ | r_x | r_d | disattenuated ρ |
|---|---|---|---|---|---|---|---|
| final_controlled_opaque | 75 | +0.17 | +0.22 | (11) +0.39 | 0.89 | 0.89 | +0.20 |
| final_controlled_plain | 74 | -0.03 | +0.01 | (11) -0.28 | 1.00 | 0.97 | -0.03 |
| final_indep | 73 | -0.31 | -0.19 | (10) -0.45 | 0.88 | 0.56 | -0.45 |
| gdcsm | 73 | -0.07 | -0.06 | (11) -0.08 | 0.97 | 0.80 | -0.08 |
| goby | 73 | -0.75 | -0.73 | (11) -0.86 | 0.99 | 0.85 | -0.81 |
| hdxsm | 72 | -0.71 | -0.69 | (11) -0.67 | 0.99 | 0.90 | -0.75 |
| madi_indep | 75 | -0.66 | -0.55 | (11) -0.80 | 0.81 | 0.80 | -0.82 |
| madi_products | 75 | -0.43 | -0.35 | (11) -0.63 | 0.95 | 0.74 | -0.51 |
| mimic | 73 | -0.22 | -0.16 | (11) -0.44 | nan | nan | +nan |
| valentine_md5 | 75 | -0.18 | +0.01 | (12) -0.20 | 0.84 | 0.91 | -0.21 |
| valentine_plain | 76 | -0.69 | -0.65 | (13) -0.67 | 0.99 | 0.92 | -0.72 |

## Range of matcher quality and of gains

| corpus | SD x | IQR x [q25, q75] | ceiling (x > 0.9) | floor (x < 0.1) | mean d | SD d |
|---|---|---|---|---|---|---|
| final_controlled_opaque | 0.119 | 0.054 [0.04, 0.09] | 0% | 76% | +0.0054 | 0.0112 |
| final_controlled_plain | 0.299 | 0.434 [0.57, 1.00] | 55% | 4% | +0.0130 | 0.0604 |
| final_indep | 0.098 | 0.164 [0.07, 0.23] | 0% | 26% | +0.0231 | 0.0242 |
| gdcsm | 0.128 | 0.275 [0.01, 0.28] | 0% | 42% | +0.0047 | 0.0144 |
| goby | 0.256 | 0.521 [0.18, 0.70] | 0% | 10% | +0.0165 | 0.0204 |
| hdxsm | 0.202 | 0.285 [0.39, 0.68] | 0% | 4% | +0.0182 | 0.0344 |
| madi_indep | 0.153 | 0.208 [0.36, 0.56] | 0% | 0% | +0.0489 | 0.0407 |
| madi_products | 0.195 | 0.221 [0.47, 0.69] | 1% | 1% | +0.0175 | 0.0575 |
| mimic | 0.097 | 0.143 [0.04, 0.18] | 0% | 40% | +0.0019 | 0.0211 |
| valentine_md5 | 0.091 | 0.040 [0.07, 0.11] | 0% | 71% | +0.0048 | 0.0075 |
| valentine_plain | 0.174 | 0.221 [0.42, 0.64] | 0% | 3% | +0.0286 | 0.0425 |

## Range matching: ρ in a strong corpus, restricted to the matchers whose x lies in a zero corpus's IQR

| strong corpus | restricted to the x-range of | matchers kept | ρ restricted | ρ full |
|---|---|---|---|---|
| valentine_plain | final_controlled_plain [0.57, 1.00] | 34 | +0.01 | -0.69 |
| valentine_plain | final_controlled_opaque [0.04, 0.09] | 1 | +nan | -0.69 |
| valentine_plain | gdcsm [0.01, 0.28] | 10 | -0.21 | -0.69 |
| goby | final_controlled_plain [0.57, 1.00] | 41 | -0.23 | -0.75 |
| goby | final_controlled_opaque [0.04, 0.09] | 6 | +nan | -0.75 |
| goby | gdcsm [0.01, 0.28] | 24 | -0.21 | -0.75 |
| hdxsm | final_controlled_plain [0.57, 1.00] | 39 | -0.14 | -0.71 |
| hdxsm | final_controlled_opaque [0.04, 0.09] | 3 | +nan | -0.71 |
| hdxsm | gdcsm [0.01, 0.28] | 12 | -0.66 | -0.71 |
| madi_indep | final_controlled_plain [0.57, 1.00] | 19 | -0.12 | -0.66 |
| madi_indep | final_controlled_opaque [0.04, 0.09] | 0 | +nan | -0.66 |
| madi_indep | gdcsm [0.01, 0.28] | 5 | +nan | -0.66 |
