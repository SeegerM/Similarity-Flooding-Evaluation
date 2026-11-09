# X42 (md5, 549 scenarios): SF's iteration trajectory (EXPLORATORY)

- 549 run (SMX_VALENTINE=549): rows assembled by reference/assemble_549.py from iter_<r> (381) and iter_<new> (168), converged melnik = layer-1 melnik; the gate was checked when the 382 batch was scored

Gain over the initial matrix q (t = number of updates; conv = the registered converged run). J_tf = (R_exp + F1_Hr)/2; per scenario the mean over included matchers, mean within family (families with >= 5 scenarios), mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261006.

## melnik (76 matchers)

| t | J_tf gain over q [95% CI] | R_exp | F1_Hr | minus converged J_tf [95% CI] | runs at the cap | median build ms | median iterate ms |
|---|---|---|---|---|---|---|---|
| 1 | +0.0047 [+0.0040, +0.0054] | +0.0000 | +0.0094 | -0.0042 [-0.0048, -0.0035] | 1.00 | 5738 | 5.0 |
| 2 | +0.0126 [+0.0112, +0.0139] | +0.0107 | +0.0145 | +0.0037 [+0.0030, +0.0044] | 1.00 | 5720 | 5.0 |
| 3 | +0.0091 [+0.0082, +0.0101] | +0.0055 | +0.0127 | +0.0003 [-0.0002, +0.0008] | 1.00 | 5725 | 5.0 |
| 5 | +0.0091 [+0.0083, +0.0099] | +0.0054 | +0.0127 | +0.0002 [-0.0003, +0.0007] | 1.00 | 5734 | 5.0 |
| 10 | +0.0093 [+0.0085, +0.0102] | +0.0051 | +0.0136 | +0.0005 [+0.0004, +0.0006] | 0.87 | 5743 | 5.0 |
| conv | +0.0089 [+0.0080, +0.0097] | +0.0042 | +0.0135 | — | — | 5169 | 4.0 |

## fd: no matcher with every arm on every scenario

## Proposition 1 at every t: not computed on 549 (needs the melnik_t* matrices of the 168 new scenarios; pending)

