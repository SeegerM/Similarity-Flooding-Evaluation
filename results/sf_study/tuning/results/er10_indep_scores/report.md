# ER10 (EXPLORATORY): additive baselines against SF with the same class information, independent corpus

73 initial matchers with every method on all 81 scenarios; J_tf against the initial matrix, macro over AdventureWorks, Chinook, TPCH. Bootstrap: 10000 scenario draws stratified by database, seed 20261004.

| method | J_tf | AdventureWorks | Chinook | TPCH |
|---|---|---|---|---|
| SF | +0.0660 | +0.0612 | +0.0548 | +0.0821 |
| SF+P | +0.0856 | +0.0719 | +0.0715 | +0.1133 |
| SF+P~ | +0.0472 | +0.0420 | +0.0437 | +0.0560 |
| SF+S | +0.0902 | +0.0753 | +0.0756 | +0.1197 |
| SF+S~ | +0.0471 | +0.0433 | +0.0412 | +0.0569 |
| A_p | +0.0761 | +0.0669 | +0.0709 | +0.0904 |
| A_s | +0.0744 | +0.0707 | +0.0570 | +0.0955 |

| contrast | mean | 95% CI | AdventureWorks | Chinook | TPCH | matchers > 0 |
|---|---|---|---|---|---|---|
| A_p − SF+P | -0.0095 | [-0.0142, -0.0046] | -0.0050 | -0.0006 | -0.0229 | 24 of 73 |
| A_s − SF+S | -0.0158 | [-0.0205, -0.0109] | -0.0046 | -0.0187 | -0.0242 | 20 of 73 |
| SF+P − SF | +0.0196 | [+0.0184, +0.0207] | +0.0107 | +0.0167 | +0.0312 | 59 of 73 |
| SF+S − SF | +0.0242 | [+0.0234, +0.0250] | +0.0141 | +0.0209 | +0.0376 | 62 of 73 |
| SF+P − SF+P~ | +0.0384 | [+0.0357, +0.0411] | +0.0300 | +0.0278 | +0.0574 | 73 of 73 |
| SF+S − SF+S~ | +0.0431 | [+0.0407, +0.0455] | +0.0321 | +0.0344 | +0.0628 | 72 of 73 |
| A_p − SF | +0.0100 | [+0.0046, +0.0158] | +0.0057 | +0.0161 | +0.0083 | 32 of 73 |
| A_s − SF | +0.0084 | [+0.0031, +0.0138] | +0.0095 | +0.0022 | +0.0134 | 33 of 73 |
