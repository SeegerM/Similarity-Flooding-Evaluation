# X91: SF export check (EXPLORATORY)

- runs: 504
- replay: max relative deviation of x_t 9.33e-15; of M_t 1.27e-14
- decomposition (origin × path length) reconstructs x_t to 1.04e-14 (max relative)

## Proposition 1 on the graph's own blocks (all t)

| attach | lit | runs × t | flagged | max rel. resid (unflagged) | order violations / strict pairs | ties broken / ties |
|---|---|---|---|---|---|---|
| ID | NONE | 284 | 0 | 6.20e-14 | 0 / 1070881 | 0 / 637292 |
| ID | PREFIX_SUFFIX | 254 | 0 | 6.51e-14 | 0 / 1049525 | 0 / 505482 |
| NAME | NONE | 1135 | 0 | 2.04e-14 | 0 / 7676547 | 0 / 1656192 |
| NAME | PREFIX_SUFFIX | 856 | 0 | 2.02e-14 | 0 / 5961153 | 0 / 1221946 |
| SPLIT | NONE | 322 | 0 | 3.05e-14 | 0 / 1227693 | 0 / 704200 |
| SPLIT | PREFIX_SUFFIX | 283 | 0 | 3.18e-14 | 0 / 1201177 | 0 / 585538 |

## Offset strength s_t on the real blocks (median)

| attach | lit | t1 | t2 | t3 | final |
|---|---|---|---|---|---|
| ID | NONE | 0.000 | 0.050 | 0.103 | 0.059 |
| ID | PREFIX_SUFFIX | 0.000 | 0.081 | 0.123 | 0.082 |
| NAME | NONE | 0.000 | 0.000 | 0.044 | 0.049 |
| NAME | PREFIX_SUFFIX | 0.000 | 0.124 | 0.077 | 0.078 |
| SPLIT | NONE | 0.000 | 0.046 | 0.083 | 0.058 |
| SPLIT | PREFIX_SUFFIX | 0.000 | 0.080 | 0.114 | 0.082 |

## Block-offset contributions at t = 2 by origin and path length (median offset SD in units of a·SD q; mean o)

| attach | lit | origin | path | median offset SD | mean o |
|---|---|---|---|---|---|
| ID | NONE | name | 0 | 0.000 | +0.000 |
| ID | NONE | name | 1 | 0.000 | +0.000 |
| ID | NONE | name | 2 | 0.050 | +0.004 |
| ID | PREFIX_SUFFIX | literal | 0 | 0.000 | +0.000 |
| ID | PREFIX_SUFFIX | literal | 1 | 0.000 | -0.000 |
| ID | PREFIX_SUFFIX | literal | 2 | 0.015 | +0.107 |
| ID | PREFIX_SUFFIX | name | 0 | 0.000 | +0.000 |
| ID | PREFIX_SUFFIX | name | 1 | 0.000 | +0.000 |
| ID | PREFIX_SUFFIX | name | 2 | 0.049 | +0.002 |
| NAME | NONE | name | 0 | 0.000 | +0.000 |
| NAME | NONE | name | 1 | 0.000 | +0.000 |
| NAME | NONE | name | 2 | 0.000 | +0.000 |
| NAME | PREFIX_SUFFIX | literal | 0 | 0.000 | +0.000 |
| NAME | PREFIX_SUFFIX | literal | 1 | 0.000 | +0.000 |
| NAME | PREFIX_SUFFIX | literal | 2 | 0.124 | +0.195 |
| NAME | PREFIX_SUFFIX | name | 0 | 0.000 | +0.000 |
| NAME | PREFIX_SUFFIX | name | 1 | 0.000 | +0.000 |
| NAME | PREFIX_SUFFIX | name | 2 | 0.000 | +0.000 |
| SPLIT | NONE | name | 0 | 0.000 | +0.000 |
| SPLIT | NONE | name | 1 | 0.000 | +0.000 |
| SPLIT | NONE | name | 2 | 0.046 | +0.001 |
| SPLIT | PREFIX_SUFFIX | literal | 0 | 0.000 | +0.000 |
| SPLIT | PREFIX_SUFFIX | literal | 1 | 0.000 | -0.000 |
| SPLIT | PREFIX_SUFFIX | literal | 2 | 0.027 | +0.126 |
| SPLIT | PREFIX_SUFFIX | name | 0 | 0.000 | +0.000 |
| SPLIT | PREFIX_SUFFIX | name | 1 | 0.000 | +0.000 |
| SPLIT | PREFIX_SUFFIX | name | 2 | 0.044 | -0.001 |
