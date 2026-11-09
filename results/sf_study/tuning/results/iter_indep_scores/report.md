# X42 (indep): SF's iteration trajectory (EXPLORATORY)

- gate levenshtein__melnik_gate = layer 1 levenshtein__melnik: max |difference| 0.0 — passed

Gain over the initial matrix q (t = number of updates; conv = the registered converged run). J_tf = (R_exp + F1_Hr)/2; per scenario the mean over included matchers, mean within family (database), mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261006.

## melnik (73 matchers)

| t | J_tf gain over q [95% CI] | R_exp | F1_Hr | minus converged J_tf [95% CI] | runs at the cap | median build ms | median iterate ms |
|---|---|---|---|---|---|---|---|
| 1 | -0.0008 [-0.0010, -0.0006] | -0.0000 | -0.0015 | -0.0668 [-0.0716, -0.0619] | 1.00 | 1471 | 46.0 |
| 2 | +0.0632 [+0.0579, +0.0685] | +0.0493 | +0.0772 | -0.0028 [-0.0051, -0.0004] | 1.00 | 1460 | 45.0 |
| 3 | +0.0519 [+0.0480, +0.0557] | +0.0220 | +0.0819 | -0.0141 [-0.0155, -0.0127] | 1.00 | 1476 | 46.0 |
| 5 | +0.0543 [+0.0500, +0.0583] | +0.0256 | +0.0829 | -0.0118 [-0.0127, -0.0108] | 1.00 | 1478 | 46.0 |
| 10 | +0.0744 [+0.0691, +0.0796] | +0.0585 | +0.0903 | +0.0084 [+0.0078, +0.0090] | 1.00 | 1456 | 47.0 |
| conv | +0.0660 [+0.0610, +0.0708] | +0.0431 | +0.0890 | — | — | 1408 | 52.0 |

## melnik_shape (73 matchers)

| t | J_tf gain over q [95% CI] | R_exp | F1_Hr | minus converged J_tf [95% CI] | runs at the cap | median build ms | median iterate ms |
|---|---|---|---|---|---|---|---|
| 1 | -0.0008 [-0.0009, -0.0006] | -0.0000 | -0.0015 | -0.0863 [-0.0906, -0.0821] | 1.00 | 1518 | 60.0 |
| 2 | +0.0870 [+0.0822, +0.0919] | +0.0674 | +0.1067 | +0.0015 [-0.0011, +0.0040] | 1.00 | 1527 | 62.0 |
| 3 | +0.0657 [+0.0623, +0.0691] | +0.0279 | +0.1035 | -0.0199 [-0.0209, -0.0189] | 1.00 | 1515 | 61.0 |
| 5 | +0.0702 [+0.0666, +0.0738] | +0.0346 | +0.1059 | -0.0154 [-0.0161, -0.0146] | 1.00 | 1520 | 64.0 |
| 10 | +0.0917 [+0.0873, +0.0960] | +0.0673 | +0.1161 | +0.0061 [+0.0056, +0.0066] | 0.98 | 1510 | 66.0 |
| conv | +0.0856 [+0.0813, +0.0897] | +0.0579 | +0.1133 | — | — | 1459 | 74.0 |

