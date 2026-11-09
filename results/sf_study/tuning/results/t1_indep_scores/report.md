# T1 (table-membership ablation), registered analysis

- levenshtein__melnik_original = layer1_indep levenshtein__melnik: max |difference| 0.0 — passed
- levenshtein__shape_original = ER10 levenshtein__melnik_shape: max |difference| 0.0 — passed
- levenshtein__melnik_permuted_d0 = screen levenshtein__melnik__permuted: max |difference| 0.0 — passed

J_tf; per scenario the mean over included matchers, mean within database, mean over databases; 95% bootstrap within databases, 10,000 resamples, seed 20261003. Confirmed: lower bound > 0 and positive in every database.

| | contrast | matchers | estimate [95% CI] | AdventureWorks | Chinook | TPCH | matchers > 0 | verdict |
|---|---|---|---|---|---|---|---|---|
| T1b (primary) | melnik_original − melnik_collapsed | 73 | +0.0493 [+0.0445, +0.0540] | +0.0492 | +0.0403 | +0.0585 | 73 | CONFIRMED |
| T1d (primary) | A_d − melnik_collapsed | 73 | +0.0189 [+0.0183, +0.0195] | +0.0272 | +0.0112 | +0.0182 | 67 | CONFIRMED |
| T1c | melnik_original − melnik_permuted | 73 | +0.0522 [+0.0471, +0.0571] | +0.0506 | +0.0430 | +0.0629 | 73 | CONFIRMED |
| T1e (A_p) | A_p − shape_collapsed | 73 | +0.0324 [+0.0314, +0.0335] | +0.0361 | +0.0315 | +0.0298 | 67 | CONFIRMED |
| T1e (A_s) | A_s − semtype_collapsed | 73 | +0.0271 [+0.0262, +0.0279] | +0.0335 | +0.0168 | +0.0308 | 68 | CONFIRMED |
| T1f (shape) | shape_original − shape_collapsed | 73 | +0.0419 [+0.0378, +0.0460] | +0.0410 | +0.0321 | +0.0527 | 73 | CONFIRMED |
| T1f (semtype) | semtype_original − semtype_collapsed | 73 | +0.0429 [+0.0386, +0.0471] | +0.0381 | +0.0355 | +0.0550 | 73 | CONFIRMED |
| report: melnik permuted − collapsed | melnik_permuted − melnik_collapsed | 73 | -0.0028 [-0.0040, -0.0016] | -0.0015 | -0.0026 | -0.0044 | 26 |  |
| report: A_d − SF(melnik) original | A_d − melnik_original | 73 | -0.0305 [-0.0355, -0.0254] | -0.0220 | -0.0291 | -0.0402 | 11 |  |
| report: A_p − SF(shape) original | A_p − shape_original | 73 | -0.0095 [-0.0143, -0.0046] | -0.0050 | -0.0006 | -0.0229 | 24 |  |
| report: A_s − SF(semtype) original | A_s − semtype_original | 73 | -0.0158 [-0.0206, -0.0110] | -0.0046 | -0.0187 | -0.0242 | 20 |  |
| report: A_d − SF(melnik) permuted | A_d − melnik_permuted | 73 | +0.0217 [+0.0203, +0.0231] | +0.0286 | +0.0139 | +0.0227 | 70 |  |

- T1a (deterministic): 0 within-block inversions of 4490192445 ordered pairs under `melnik` COLLAPSED — CONFIRMED.
