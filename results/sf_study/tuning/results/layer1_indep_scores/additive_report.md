# Amendment 5: additive datatype and class terms against SF as published (independent corpus, plain)

73 initial matchers with full coverage, 81 scenarios in Indep-AdventureWorks, Indep-Chinook, Indep-TPCH. Not used (incomplete): extrema.

| contrast | estimate | 95% CI | Indep-AdventureWorks | Indep-Chinook | Indep-TPCH | matchers > 0 | verdict |
|---|---|---|---|---|---|---|---|
| HB1 A_p − SF | +0.0100 | [+0.0046, +0.0158] | +0.0057 | +0.0161 | +0.0083 | 32 of 73 | CONFIRMED |
| HB2 A_s − SF | +0.0084 | [+0.0030, +0.0138] | +0.0095 | +0.0022 | +0.0134 | 33 of 73 | CONFIRMED |
| HB3 A_d − SF | -0.0305 | [-0.0355, -0.0253] | -0.0220 | -0.0291 | -0.0402 | 11 of 73 | not confirmed |

Descriptive (macro over databases, mean over matchers), gains over q:

| matrix | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| sf | +0.0660 | +0.0431 | +0.0890 |
| A_d | +0.0356 | +0.0352 | +0.0359 |
| A_p | +0.0761 | +0.0817 | +0.0704 |
| A_s | +0.0744 | +0.0794 | +0.0694 |
