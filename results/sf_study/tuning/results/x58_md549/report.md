# X58 (EXPLORATORY): matcher family and SF's gain, regime md5

- 78 non-control matchers with all 549 scenarios; families (≥ 3 matchers, others pooled): comp 7, dep 6, dist 6, emb 5, fmt 3, graph 4, lex 7, meta 7, other 3, ovl 10, str 20.
- Tie share of q (mean share of cells whose value repeats in its row): median 0.28, range 0.00–1.00.

## Arm `melnik`: Δ = J_tf(melnik) − J_tf(q)

- Mean Δ +0.0088; Spearman ρ(Δ, Q) -0.30.
- R²: quality (Q + Q²) 0.247; + family 0.766 (Δ +0.519, permutation p 0.0001); + input flags 0.519 (Δ +0.272, p 0.0001); + tie share 0.364 (Δ +0.118); + tie share + family 0.777 (family beyond Q + ties Δ +0.413, p 0.0001).
- Spearman ρ(residual of Δ ~ Q, tie share) +0.27.

| family | matchers | mean Q | mean Δ | mean residual (Δ − quality fit) | SE |
|---|---|---|---|---|---|
| ovl | 10 | 0.045 | +0.0397 | +0.0225 | 0.0033 |
| dep | 6 | 0.248 | +0.0065 | +0.0085 | 0.0027 |
| fmt | 3 | 0.145 | +0.0042 | +0.0034 | 0.0015 |
| comp | 7 | 0.167 | +0.0048 | +0.0014 | 0.0036 |
| dist | 6 | 0.114 | +0.0114 | -0.0007 | 0.0061 |
| meta | 7 | 0.265 | -0.0001 | -0.0033 | 0.0041 |
| lex | 7 | 0.089 | +0.0026 | -0.0059 | 0.0009 |
| str | 20 | 0.077 | +0.0042 | -0.0065 | 0.0009 |
| graph | 4 | 0.068 | +0.0047 | -0.0077 | 0.0022 |
| other | 3 | 0.097 | -0.0000 | -0.0081 | 0.0041 |
| emb | 5 | 0.073 | +0.0028 | -0.0086 | 0.0014 |

## Arm `fd`: Δ = J_tf(fd) − J_tf(q)

- Mean Δ +0.0546; Spearman ρ(Δ, Q) -0.85.
- R²: quality (Q + Q²) 0.951; + family 0.972 (Δ +0.021, permutation p 0.0001); + input flags 0.965 (Δ +0.014, p 0.0002); + tie share 0.957 (Δ +0.006); + tie share + family 0.973 (family beyond Q + ties Δ +0.016, p 0.0004).
- Spearman ρ(residual of Δ ~ Q, tie share) +0.36.

| family | matchers | mean Q | mean Δ | mean residual (Δ − quality fit) | SE |
|---|---|---|---|---|---|
| ovl | 10 | 0.045 | +0.1080 | +0.0138 | 0.0020 |
| dep | 6 | 0.248 | -0.0119 | +0.0076 | 0.0056 |
| comp | 7 | 0.167 | +0.0332 | +0.0059 | 0.0043 |
| meta | 7 | 0.265 | -0.0290 | +0.0038 | 0.0057 |
| graph | 4 | 0.068 | +0.0863 | +0.0029 | 0.0011 |
| dist | 6 | 0.114 | +0.0579 | +0.0029 | 0.0030 |
| emb | 5 | 0.073 | +0.0766 | -0.0040 | 0.0047 |
| lex | 7 | 0.089 | +0.0666 | -0.0059 | 0.0017 |
| str | 20 | 0.077 | +0.0718 | -0.0067 | 0.0025 |
| fmt | 3 | 0.145 | +0.0323 | -0.0103 | 0.0081 |
| other | 3 | 0.097 | +0.0491 | -0.0185 | 0.0115 |

