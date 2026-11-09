# X78 part (b): which effects depend on benchmark properties? (EXPLORATORY)

- Joined 1568 scenarios; unmatched effect rows per corpus: none.
- Scenarios per corpus × regime: {'ctl/opaque': 8, 'ctl/plain': 8, 'final_controlled/opaque': 48, 'final_controlled/plain': 48, 'final_indep/plain': 9, 'indep/plain': 81, 'indep2/plain': 54, 'synthea/plain': 39, 't2d/plain': 175, 'valentine_md5/md5': 549, 'valentine_plain/plain': 549}.
- Within-corpus groups for (1)/(2): ≥ 15 scenarios. CIs: bootstrap over source_id clusters within each group.

## (2) Pooled within-corpus Spearman ρ (ranks centred within corpus × regime)

Only |ρ| ≥ 0.15 with the interval excluding 0 are listed; the full grid is in benchmark_effects_within.csv.

| effect | property | ρ [95% CI] | scenarios | groups |
|---|---|---|---|---|
| E1 datatype − SF | src_tables | -0.20 [-0.30, -0.16] | 1543 | 8 |
| E2 role − key | true_share | +0.40 [+0.30, +0.66] | 1462 | 7 |
| E2 role − key | size_ratio | +0.17 [+0.03, +0.30] | 1462 | 7 |
| E2 role − key | log_cols | +0.42 [+0.28, +0.59] | 1462 | 7 |
| E2 role − key | value_gap | +0.21 [+0.04, +0.32] | 1462 | 7 |
| E2 role − key | ovl_true | +0.21 [+0.05, +0.32] | 1462 | 7 |
| E2 role − key | fd_per_col | +0.37 [+0.28, +0.53] | 1462 | 7 |
| E2 role − key | key_share | -0.26 [-0.38, -0.09] | 1462 | 7 |
| E2 role − key | ind_per_col | +0.32 [+0.23, +0.54] | 1462 | 7 |
| E3 fd − melnik (R_exp) | true_share | +0.26 [+0.12, +0.44] | 1369 | 5 |
| E3 fd − melnik (R_exp) | name_gap | -0.19 [-0.32, -0.08] | 1369 | 5 |
| E3 fd − melnik (J) | true_share | +0.27 [+0.12, +0.37] | 1462 | 7 |
| E3 fd − melnik (J) | name_gap | -0.17 [-0.28, -0.03] | 1462 | 7 |
| E4 SF − q | true_share | +0.34 [+0.09, +0.50] | 1543 | 8 |
| E4 SF − q | num_share | +0.21 [+0.02, +0.34] | 1543 | 8 |
| E4 SF − q | dtype_entropy | +0.38 [+0.07, +0.48] | 1543 | 8 |
| E4 SF − q | name_gap | -0.15 [-0.23, -0.03] | 1543 | 8 |
| E4 SF − q | src_tables | +0.17 [+0.13, +0.26] | 1543 | 8 |
| E5 table − datatype | diag_share | +0.71 [+0.62, +0.87] | 96 | 2 |
| E5 table − datatype | key_share | +0.33 [+0.21, +0.53] | 189 | 4 |
| E5 table − datatype | src_tables | +0.59 [+0.50, +0.65] | 189 | 4 |

## (1) Per corpus × regime Spearman ρ for the pooled associations above (descriptive signs)

- E1 datatype − SF ~ src_tables: indep/plain -0.48; indep2/plain -0.96; synthea/plain -0.28
- E2 role − key ~ true_share: final_controlled/opaque +0.20; final_controlled/plain +0.25; indep2/plain -0.18; synthea/plain +0.69; t2d/plain +0.02; valentine_md5/md5 +0.48; valentine_plain/plain +0.50
- E2 role − key ~ size_ratio: final_controlled/opaque -0.10; final_controlled/plain -0.10; indep2/plain +0.42; synthea/plain -0.66; t2d/plain +0.17; valentine_md5/md5 +0.22; valentine_plain/plain +0.21
- E2 role − key ~ log_cols: final_controlled/opaque +0.34; final_controlled/plain +0.22; indep2/plain +0.08; synthea/plain -0.24; t2d/plain -0.20; valentine_md5/md5 +0.59; valentine_plain/plain +0.55
- E2 role − key ~ value_gap: final_controlled/opaque +0.25; final_controlled/plain -0.29; indep2/plain +0.23; synthea/plain -0.18; t2d/plain +0.01; valentine_plain/plain +0.43
- E2 role − key ~ ovl_true: final_controlled/opaque +0.25; final_controlled/plain -0.29; indep2/plain +0.28; synthea/plain -0.15; t2d/plain +0.01; valentine_plain/plain +0.42
- E2 role − key ~ fd_per_col: final_controlled/opaque +0.31; final_controlled/plain +0.17; indep2/plain +0.56; synthea/plain +0.27; t2d/plain -0.16; valentine_md5/md5 +0.52; valentine_plain/plain +0.41
- E2 role − key ~ key_share: final_controlled/opaque +0.18; final_controlled/plain +0.37; indep2/plain -0.63; synthea/plain -0.25; t2d/plain +0.16; valentine_md5/md5 -0.39; valentine_plain/plain -0.33
- E2 role − key ~ ind_per_col: final_controlled/opaque +0.17; final_controlled/plain +0.18; indep2/plain +0.07; synthea/plain -0.01; t2d/plain +0.04; valentine_md5/md5 +0.34; valentine_plain/plain +0.43
- E3 fd − melnik (R_exp) ~ true_share: final_controlled/opaque +0.51; final_controlled/plain -0.43; t2d/plain +0.07; valentine_md5/md5 +0.17; valentine_plain/plain +0.44
- E3 fd − melnik (R_exp) ~ name_gap: final_controlled/opaque +0.13; final_controlled/plain -0.37; t2d/plain -0.14; valentine_md5/md5 -0.04; valentine_plain/plain -0.38
- E3 fd − melnik (J) ~ true_share: final_controlled/opaque +0.43; final_controlled/plain -0.56; indep2/plain +0.23; synthea/plain +0.03; t2d/plain +0.01; valentine_md5/md5 +0.27; valentine_plain/plain +0.43
- E3 fd − melnik (J) ~ name_gap: final_controlled/opaque +0.02; final_controlled/plain -0.16; indep2/plain -0.41; synthea/plain -0.20; t2d/plain -0.19; valentine_md5/md5 +0.02; valentine_plain/plain -0.34
- E4 SF − q ~ true_share: final_controlled/opaque -0.75; final_controlled/plain -0.63; indep/plain -0.32; indep2/plain +0.09; synthea/plain +0.01; t2d/plain +0.26; valentine_md5/md5 +0.71; valentine_plain/plain +0.31
- E4 SF − q ~ num_share: final_controlled/opaque +0.76; final_controlled/plain +0.50; indep/plain -0.04; indep2/plain +0.66; synthea/plain +0.28; t2d/plain +0.21; valentine_plain/plain +0.22
- E4 SF − q ~ dtype_entropy: final_controlled/opaque +0.76; final_controlled/plain +0.58; indep/plain -0.13; indep2/plain +0.52; synthea/plain +0.01; t2d/plain +0.19; valentine_md5/md5 +0.54; valentine_plain/plain +0.34
- E4 SF − q ~ name_gap: final_controlled/opaque -0.16; final_controlled/plain -0.00; indep/plain -0.29; indep2/plain -0.20; synthea/plain -0.51; t2d/plain -0.27; valentine_md5/md5 +0.00; valentine_plain/plain -0.23
- E4 SF − q ~ src_tables: indep/plain +0.49; indep2/plain +0.90; synthea/plain -0.38
- E5 table − datatype ~ diag_share: final_controlled/opaque +0.53; final_controlled/plain +0.87
- E5 table − datatype ~ key_share: final_controlled/opaque +0.48; final_controlled/plain +0.17; indep2/plain +0.59; synthea/plain -0.02
- E5 table − datatype ~ src_tables: indep2/plain +0.96; synthea/plain +0.44

## (3) Between corpora: corpus × regime effect means and property medians (descriptive, n = groups)

| group | n | E1 datatype − SF | E2 role − key | E3 fd − melnik (R_exp) | E3 fd − melnik (J) | E4 SF − q | E5 table − datatype | value_gap | fd_per_col | diag_share | dtype_entropy | name_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctl/opaque | 8 | +0.003 | +0.075 | +0.010 | +0.031 | +0.004 | – | 0.00 | 12.13 | 0.06 | 0.45 | -0.02 |
| ctl/plain | 8 | +0.017 | +0.029 | -0.446 | -0.250 | +0.017 | – | 0.78 | 12.09 | 0.06 | 0.88 | 0.56 |
| final_controlled/opaque | 48 | +0.007 | +0.133 | +0.019 | +0.048 | +0.012 | -0.000 | 0.00 | 8.55 | 0.07 | 0.47 | -0.00 |
| final_controlled/plain | 48 | +0.004 | +0.034 | -0.354 | -0.185 | +0.026 | -0.005 | 0.76 | 8.55 | 0.07 | 1.00 | 0.58 |
| final_indep/plain | 9 | -0.028 | -0.012 | -0.012 | -0.007 | +0.024 | +0.011 | 0.03 | 2.25 | – | 1.28 | 0.39 |
| indep/plain | 81 | -0.030 | – | – | – | +0.062 | – | 0.75 | 0.00 | – | 1.00 | 0.65 |
| indep2/plain | 54 | -0.060 | +0.028 | – | -0.060 | +0.080 | +0.090 | 0.59 | 14.78 | – | 1.07 | 0.68 |
| synthea/plain | 39 | +0.067 | +0.083 | – | -0.018 | +0.055 | -0.004 | 0.01 | 3.90 | – | 1.42 | 0.01 |
| t2d/plain | 175 | -0.027 | +0.036 | -0.038 | -0.031 | +0.019 | – | 0.02 | 2.25 | 0.00 | 0.65 | 0.34 |
| valentine_md5/md5 | 549 | +0.001 | -0.011 | -0.012 | +0.015 | +0.008 | – | 0.00 | 13.09 | 1.00 | -0.00 | 0.00 |
| valentine_plain/plain | 549 | +0.016 | +0.001 | -0.194 | -0.103 | +0.016 | – | 0.37 | 13.09 | 1.00 | 0.99 | 0.38 |

Between-corpus Spearman (effect mean ~ property median; descriptive):

- E1 datatype − SF: src_tables -0.75 (n 11); true_share +0.58 (n 11); log_cols -0.54 (n 11); key_share -0.47 (n 11)
- E2 role − key: true_share +0.81 (n 10); ind_per_col -0.74 (n 10); key_share -0.63 (n 10); log_cols -0.62 (n 10)
- E3 fd − melnik (R_exp): value_gap -0.88 (n 8); ovl_true -0.88 (n 8); name_gap -0.81 (n 8); num_share -0.62 (n 8)
- E3 fd − melnik (J): value_gap -0.94 (n 10); ovl_true -0.94 (n 10); name_gap -0.82 (n 10); num_share -0.75 (n 10)
- E4 SF − q: dtype_entropy +0.84 (n 11); name_gap +0.81 (n 11); size_ratio -0.65 (n 11); src_tables +0.60 (n 11)
- E5 table − datatype: src_tables +0.78 (n 5); true_share -0.72 (n 5); log_cols +0.72 (n 5); key_share +0.72 (n 5)
