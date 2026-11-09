# Table-context control: a non-iterative table-aware baseline against SF (independent corpus, plain)

73 initial matchers, 81 scenarios in Indep-AdventureWorks, Indep-Chinook, Indep-TPCH. λ chosen leave-one-database-out per term: A_dt: Indep-AdventureWorks 0.5, Indep-Chinook 0.5, Indep-TPCH 0.5; A_pt: Indep-AdventureWorks 0.5, Indep-Chinook 0.5, Indep-TPCH 0.5; A_st: Indep-AdventureWorks 0.5, Indep-Chinook 0.5, Indep-TPCH 0.5.

| hypothesis | contrast | estimate | 95% CI | Indep-AdventureWorks | Indep-Chinook | Indep-TPCH | matchers > 0 | decision |
|---|---|---|---|---|---|---|---|---|
| HC0 | A_dt − A_d | +0.0769 | [+0.0702, +0.0836] | +0.0897 | +0.0543 | +0.0866 | 70 of 73 | CONFIRMED |
| HC1 (primary) | sf − A_dt | -0.0464 | [-0.0487, -0.0440] | -0.0677 | -0.0252 | -0.0464 | 6 of 73 | table context suffices |
| HC2 | sf_p − A_pt | -0.0584 | [-0.0607, -0.0560] | -0.0743 | -0.0424 | -0.0585 | 3 of 73 | table context suffices |
| HC3 | sf_s − A_st | -0.0531 | [-0.0555, -0.0506] | -0.0745 | -0.0320 | -0.0528 | 4 of 73 | table context suffices |

Reported only: gains over q (macro over databases, mean over matchers), with the held-out λ:

| matrix | J_tf | R_exp | F1_Hr |
|---|---|---|---|
| sf | +0.0660 | +0.0431 | +0.0890 |
| sf_p | +0.0856 | +0.0579 | +0.1133 |
| sf_s | +0.0902 | +0.0653 | +0.1151 |
| A_d | +0.0356 | +0.0352 | +0.0359 |
| A_p | +0.0761 | +0.0817 | +0.0704 |
| A_s | +0.0744 | +0.0794 | +0.0694 |
| A_dt | +0.1125 | +0.1157 | +0.1092 |
| A_pt | +0.1440 | +0.1479 | +0.1401 |
| A_st | +0.1433 | +0.1472 | +0.1394 |

Reported only: every λ on all databases (J_tf gain over q):

| term | 0.0 | 0.1 | 0.2 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|---|---|
| A_dt | +0.0356 | +0.1016 | +0.1076 | +0.1125 | +0.1086 | +0.0872 | +0.0366 |
| A_pt | +0.0761 | +0.1358 | +0.1409 | +0.1440 | +0.1399 | +0.1181 | +0.0683 |
| A_st | +0.0744 | +0.1334 | +0.1391 | +0.1433 | +0.1385 | +0.1157 | +0.0637 |
