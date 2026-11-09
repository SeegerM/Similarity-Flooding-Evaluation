# Answer (A): artefact checks, Valentine plain (EXPLORATORY review check)

77 non-control initial matchers, 547 scenarios in 4 families (ChEMBL 180, Magellan 7, OpenData 180, TPC-DI 180); SF as published (layer 1, arm melnik). Quality = the matcher's expected Recall@GT alone. 1000 random splits / permutations, seed 20261026.

| gain in | published rho | rho, this panel | split-half rho: median [2.5%, 97.5%] (share < 0) | rho with gain / headroom | ceiling-null rho: median [2.5%, 97.5%] | P(null <= observed) |
|---|---|---|---|---|---|---|
| expected Recall@GT | -0.67 | -0.67 | -0.66 [-0.71, -0.59] (100%) | -0.48 | -0.69 [-0.78, -0.57] | 0.592 |
| tie-free F1 | -0.49 | -0.49 | -0.48 [-0.54, -0.43] (100%) | -0.45 | -0.67 [-0.77, -0.54] | 0.995 |

Relative gain under the same ceiling null (gain / headroom of each permuted, clipped draw):

| gain in | rho with gain / headroom | null: median [2.5%, 97.5%] | P(null <= observed) |
|---|---|---|---|
| expected Recall@GT | -0.48 | +0.06 [-0.14, +0.26] | 0.0010 |
| tie-free F1 | -0.45 | -0.49 [-0.63, -0.32] | 0.6983 |

Reading: if the relation were only shared noise, the split-half rho would shrink towards 0; if it were only the ceiling, the observed rho would lie inside the ceiling-null band and the headroom-normalised rho would vanish.
