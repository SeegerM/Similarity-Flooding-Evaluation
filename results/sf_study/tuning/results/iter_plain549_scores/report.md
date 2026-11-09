# X42 (plain, 549 scenarios): SF's iteration trajectory (EXPLORATORY)

- 549 run (SMX_VALENTINE=549): rows assembled by reference/assemble_549.py from iter_<r> (381) and iter_<new> (168), converged melnik = layer-1 melnik; the gate was checked when the 382 batch was scored

Gain over the initial matrix q (t = number of updates; conv = the registered converged run). J_tf = (R_exp + F1_Hr)/2; per scenario the mean over included matchers, mean within family (families with >= 5 scenarios), mean over families; 95% bootstrap within family, 10,000 resamples, seed 20261006.

## melnik (76 matchers)

| t | J_tf gain over q [95% CI] | R_exp | F1_Hr | minus converged J_tf [95% CI] | runs at the cap | median build ms | median iterate ms |
|---|---|---|---|---|---|---|---|
| 1 | -0.0070 [-0.0076, -0.0064] | -0.0001 | -0.0140 | -0.0273 [-0.0319, -0.0232] | 1.00 | 3783 | 6.0 |
| 2 | +0.0326 [+0.0276, +0.0378] | +0.0483 | +0.0170 | +0.0123 [+0.0109, +0.0139] | 1.00 | 3772 | 6.0 |
| 3 | +0.0098 [+0.0062, +0.0140] | +0.0110 | +0.0087 | -0.0105 [-0.0122, -0.0088] | 1.00 | 3762 | 6.0 |
| 5 | +0.0146 [+0.0106, +0.0189] | +0.0197 | +0.0096 | -0.0057 [-0.0063, -0.0051] | 1.00 | 3772 | 6.0 |
| 10 | +0.0227 [+0.0186, +0.0271] | +0.0334 | +0.0120 | +0.0024 [+0.0021, +0.0026] | 0.92 | 3756 | 6.0 |
| conv | +0.0203 [+0.0161, +0.0248] | +0.0295 | +0.0111 | — | — | 3758 | 6.0 |

## fd: no matcher with every arm on every scenario

## Proposition 1 at every t: not computed on 549 (needs the melnik_t* matrices of the 168 new scenarios; pending)

