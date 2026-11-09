# X58 (EXPLORATORY): matcher family and SF's gain, regime plain

- 78 non-control matchers with all 549 scenarios; families (≥ 3 matchers, others pooled): comp 7, dep 6, dist 6, emb 5, fmt 3, graph 4, lex 7, meta 6, other 4, ovl 10, str 20.
- Tie share of q (mean share of cells whose value repeats in its row): median 0.54, range 0.00–0.98.

## Arm `melnik`: Δ = J_tf(melnik) − J_tf(q)

- Mean Δ +0.0210; Spearman ρ(Δ, Q) -0.70.
- R²: quality (Q + Q²) 0.652; + family 0.798 (Δ +0.147, permutation p 0.0002); + input flags 0.697 (Δ +0.046, p 0.0395); + tie share 0.687 (Δ +0.036); + tie share + family 0.822 (family beyond Q + ties Δ +0.134, p 0.0001).
- Spearman ρ(residual of Δ ~ Q, tie share) -0.42.

| family | matchers | mean Q | mean Δ | mean residual (Δ − quality fit) | SE |
|---|---|---|---|---|---|
| dep | 6 | 0.249 | +0.0994 | +0.0242 | 0.0138 |
| meta | 6 | 0.374 | +0.0631 | +0.0126 | 0.0084 |
| comp | 7 | 0.657 | +0.0085 | +0.0085 | 0.0065 |
| other | 4 | 0.664 | +0.0059 | +0.0078 | 0.0090 |
| str | 20 | 0.634 | +0.0058 | +0.0038 | 0.0027 |
| emb | 5 | 0.542 | +0.0147 | +0.0016 | 0.0035 |
| dist | 6 | 0.427 | +0.0348 | +0.0001 | 0.0124 |
| lex | 7 | 0.515 | +0.0137 | -0.0042 | 0.0031 |
| fmt | 3 | 0.473 | +0.0100 | -0.0156 | 0.0076 |
| ovl | 10 | 0.517 | -0.0021 | -0.0202 | 0.0022 |
| graph | 4 | 0.384 | +0.0190 | -0.0295 | 0.0188 |

## Arm `fd`: Δ = J_tf(fd) − J_tf(q)

- Mean Δ -0.0877; Spearman ρ(Δ, Q) -0.80.
- R²: quality (Q + Q²) 0.755; + family 0.877 (Δ +0.122, permutation p 0.0001); + input flags 0.812 (Δ +0.057, p 0.0022); + tie share 0.876 (Δ +0.121); + tie share + family 0.915 (family beyond Q + ties Δ +0.039, p 0.0052).
- Spearman ρ(residual of Δ ~ Q, tie share) +0.76.

| family | matchers | mean Q | mean Δ | mean residual (Δ − quality fit) | SE |
|---|---|---|---|---|---|
| ovl | 10 | 0.517 | -0.0472 | +0.0427 | 0.0062 |
| other | 4 | 0.664 | -0.1091 | +0.0397 | 0.0256 |
| fmt | 3 | 0.473 | -0.0511 | +0.0194 | 0.0191 |
| dist | 6 | 0.427 | -0.0439 | +0.0051 | 0.0131 |
| str | 20 | 0.634 | -0.1334 | +0.0036 | 0.0070 |
| comp | 7 | 0.657 | -0.1476 | -0.0027 | 0.0119 |
| dep | 6 | 0.249 | +0.0324 | -0.0086 | 0.0061 |
| lex | 7 | 0.515 | -0.1035 | -0.0140 | 0.0069 |
| meta | 6 | 0.374 | -0.0399 | -0.0220 | 0.0159 |
| emb | 5 | 0.542 | -0.1352 | -0.0334 | 0.0170 |
| graph | 4 | 0.384 | -0.0918 | -0.0694 | 0.0385 |

