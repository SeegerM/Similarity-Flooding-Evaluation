# X90: melnik vs lit_none trajectories (EXPLORATORY)

| corpus | arm | t | cases | median resid | median s | mean o | mean R_exp gain | mean J gain |
|---|---|---|---|---|---|---|---|---|
| valentine_plain | melnik | t1 | 2196 | 9.04e-08 | 0.000 | -0.000 | +0.0000 | -0.0022 |
| valentine_plain | melnik | t2 | 2196 | 1.02e-07 | 0.141 | 0.468 | +0.0568 | +0.0454 |
| valentine_plain | melnik | t3 | 2196 | 1.17e-07 | 0.047 | 0.112 | +0.0054 | +0.0126 |
| valentine_plain | melnik | t5 | 2196 | 1.16e-07 | 0.045 | 0.103 | +0.0097 | +0.0155 |
| valentine_plain | melnik | t10 | 2196 | 1.17e-07 | 0.052 | 0.144 | +0.0367 | +0.0314 |
| valentine_plain | melnik | conv | 2196 | 1.14e-07 | 0.049 | 0.132 | +0.0277 | +0.0260 |
| valentine_plain | lit_none | t1 | 2196 | 3.23e-08 | 0.000 | -0.000 | +0.0000 | -0.0001 |
| valentine_plain | lit_none | t2 | 2196 | 8.28e-08 | 0.000 | 0.000 | +0.0000 | -0.0001 |
| valentine_plain | lit_none | t3 | 2196 | 1.10e-07 | 0.047 | 0.009 | -0.0200 | -0.0072 |
| valentine_plain | lit_none | t5 | 2196 | 1.12e-07 | 0.046 | 0.005 | -0.0176 | -0.0056 |
| valentine_plain | lit_none | t10 | 2196 | 1.09e-07 | 0.043 | -0.001 | -0.0174 | -0.0057 |
| valentine_plain | lit_none | conv | 2196 | 1.07e-07 | 0.044 | -0.002 | -0.0178 | -0.0059 |
| goby | melnik | t1 | 800 | 6.87e-08 | 0.000 | -0.000 | +0.0000 | -0.0031 |
| goby | melnik | t2 | 800 | 8.76e-08 | 0.155 | 0.120 | +0.0200 | +0.0165 |
| goby | melnik | t3 | 800 | 1.13e-07 | 0.103 | 0.071 | +0.0127 | +0.0102 |
| goby | melnik | t5 | 800 | 1.14e-07 | 0.108 | 0.072 | +0.0138 | +0.0108 |
| goby | melnik | t10 | 800 | 1.17e-07 | 0.109 | 0.078 | +0.0155 | +0.0121 |
| goby | melnik | conv | 800 | 1.19e-07 | 0.109 | 0.077 | +0.0149 | +0.0118 |
| goby | lit_none | t1 | 800 | 0.00e+00 | 0.000 | 0.000 | +0.0000 | +0.0000 |
| goby | lit_none | t2 | 800 | 5.99e-08 | 0.000 | -0.000 | +0.0000 | +0.0001 |
| goby | lit_none | t3 | 800 | 9.67e-08 | 0.094 | 0.039 | +0.0080 | +0.0045 |
| goby | lit_none | t5 | 800 | 1.02e-07 | 0.104 | 0.041 | +0.0088 | +0.0047 |
| goby | lit_none | t10 | 800 | 1.04e-07 | 0.099 | 0.036 | +0.0090 | +0.0046 |
| goby | lit_none | conv | 800 | 1.03e-07 | 0.099 | 0.036 | +0.0090 | +0.0051 |
| hdxsm | melnik | t1 | 796 | 7.00e-08 | 0.000 | 0.000 | +0.0000 | -0.0017 |
| hdxsm | melnik | t2 | 796 | 8.02e-08 | 0.156 | 0.258 | +0.0424 | +0.0351 |
| hdxsm | melnik | t3 | 796 | 9.44e-08 | 0.080 | 0.096 | +0.0105 | +0.0124 |
| hdxsm | melnik | t5 | 796 | 9.85e-08 | 0.080 | 0.096 | +0.0142 | +0.0149 |
| hdxsm | melnik | t10 | 796 | 9.65e-08 | 0.084 | 0.117 | +0.0203 | +0.0201 |
| hdxsm | melnik | conv | 796 | 9.70e-08 | 0.083 | 0.111 | +0.0197 | +0.0194 |
| hdxsm | lit_none | t1 | 796 | 0.00e+00 | 0.000 | -0.000 | +0.0000 | +0.0000 |
| hdxsm | lit_none | t2 | 796 | 6.45e-08 | 0.000 | -0.000 | +0.0000 | -0.0001 |
| hdxsm | lit_none | t3 | 796 | 8.52e-08 | 0.077 | 0.046 | -0.0019 | +0.0016 |
| hdxsm | lit_none | t5 | 796 | 8.45e-08 | 0.076 | 0.042 | -0.0029 | +0.0009 |
| hdxsm | lit_none | t10 | 796 | 8.77e-08 | 0.070 | 0.034 | -0.0021 | +0.0020 |
| hdxsm | lit_none | conv | 796 | 8.61e-08 | 0.070 | 0.034 | -0.0022 | +0.0019 |

## lit_none − melnik (mean over matchers per scenario, then scenarios; cluster 95% CI)

| corpus | contrast | estimate [CI] | scenarios / clusters |
|---|---|---|---|
| valentine_plain | Δ s_2 | -0.2488 [-0.3438, -0.1585] | 549 / 10 |
| valentine_plain | Δ J gain at t2 | -0.0455 [-0.0661, -0.0185] | 549 / 10 |
| valentine_plain | P melnik = J(t2) − J(conv) | +0.0195 [+0.0013, +0.0412] | 549 / 10 |
| valentine_plain | P lit_none | +0.0058 [-0.0071, +0.0219] | 549 / 10 |
| valentine_plain | ΔP = P(lit_none) − P(melnik) | -0.0137 [-0.0481, +0.0205] | 549 / 10 |
| goby | Δ s_2 | -0.1560 [-0.1713, -0.1429] | 200 / 124 |
| goby | Δ J gain at t2 | -0.0164 [-0.0225, -0.0111] | 200 / 124 |
| goby | P melnik = J(t2) − J(conv) | +0.0048 [+0.0018, +0.0078] | 200 / 124 |
| goby | P lit_none | -0.0050 [-0.0075, -0.0026] | 200 / 124 |
| goby | ΔP = P(lit_none) − P(melnik) | -0.0097 [-0.0135, -0.0063] | 200 / 124 |
| hdxsm | Δ s_2 | -0.1769 [-0.2405, -0.1360] | 199 / 32 |
| hdxsm | Δ J gain at t2 | -0.0352 [-0.0491, -0.0239] | 199 / 32 |
| hdxsm | P melnik = J(t2) − J(conv) | +0.0157 [+0.0078, +0.0225] | 199 / 32 |
| hdxsm | P lit_none | -0.0020 [-0.0074, +0.0010] | 199 / 32 |
| hdxsm | ΔP = P(lit_none) − P(melnik) | -0.0177 [-0.0237, -0.0115] | 199 / 32 |
