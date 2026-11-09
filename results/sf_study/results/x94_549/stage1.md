# X94 stage 1: headroom of per-scenario weights for D and L (EXPLORATORY)

- frozen_V (best on the 60 Valentine screen scenarios): w_D = 2.0, w_L = 0.1; rule (b): kappa = 0.05 with that w_D.
- rule (a) check: max |J_a − J_sf_t2| = 1.80e-01 over 14640 cells; c/a median 0.300 (IQR 0.294–0.304).

## Q1 headroom and Q3 rules (J_tf; mean over matchers per scenario, then scenarios; Valentine without the screen)

| corpus | oracle_ms | oracle_s | best global (on corpus) | frozen_V | rule (a) = SF(t2) | rule (b) | SF conv | headroom ms / s |
|---|---|---|---|---|---|---|---|---|
| valentine_plain | 0.4863 | 0.4683 | 0.4532 | 0.4531 | 0.4459 | 0.4514 | 0.4279 | +0.0332 / +0.0152 |
| goby | 0.4477 | 0.4311 | 0.4138 | 0.3988 | 0.4084 | 0.3966 | 0.4077 | +0.0339 / +0.0173 |
| hdxsm | 0.5419 | 0.5245 | 0.5080 | 0.4946 | 0.5037 | 0.4916 | 0.4911 | +0.0339 / +0.0166 |
| gdcsm | 0.1353 | 0.1316 | 0.1298 | 0.1203 | 0.1247 | 0.1203 | 0.1246 | +0.0054 / +0.0018 |
| madi_indep | 0.5570 | 0.5413 | 0.5333 | 0.5254 | 0.5226 | 0.5175 | 0.5142 | +0.0237 / +0.0080 |
| madi_products | 0.7045 | 0.6849 | 0.6837 | 0.6775 | 0.6638 | 0.6809 | 0.6528 | +0.0208 / +0.0012 |

Contrasts against frozen_V (cluster 95% CI where ≥ 5 clusters):

- valentine_plain: rule (a) − frozen_V -0.0072 [-0.0173, +0.0028] (10 clusters)
- valentine_plain: rule (b) − frozen_V -0.0017 [-0.0069, +0.0009] (10 clusters)
  - rule (b)'s share of rule (a)'s gain: 0.24
- goby: rule (a) − frozen_V +0.0096 [+0.0047, +0.0147] (124 clusters)
- goby: rule (b) − frozen_V -0.0020 – (124 clusters)
  - rule (b)'s share of rule (a)'s gain: -0.23
- hdxsm: rule (a) − frozen_V +0.0091 [+0.0037, +0.0149] (32 clusters)
- hdxsm: rule (b) − frozen_V -0.0015 – (32 clusters)
  - rule (b)'s share of rule (a)'s gain: -0.32
- gdcsm: rule (a) − frozen_V +0.0045 [-0.0023, +0.0106] (10 clusters)
- gdcsm: rule (b) − frozen_V +0.0000 [-0.0000, +0.0000] (10 clusters)
  - rule (b)'s share of rule (a)'s gain: 0.00
- madi_indep: rule (a) − frozen_V -0.0027 – (4 clusters)
- madi_indep: rule (b) − frozen_V -0.0079 – (4 clusters)
  - rule (b)'s share of rule (a)'s gain: 2.87
- madi_products: rule (a) − frozen_V -0.0136 – (1 clusters)
- madi_products: rule (b) − frozen_V +0.0034 – (1 clusters)
  - rule (b)'s share of rule (a)'s gain: -0.25

Oracle w_D (share of cells with w_D ≤ 0.5): valentine_plain 86%; goby 94%; hdxsm 94%; gdcsm 91%; madi_indep 93%; madi_products 82%

## Q2 predictability: depth-3 trees on ground-truth-free features, leave one corpus out

| held-out | variant | J_pred | J_frozen(train) | oracle_ms | recovered share | share vs best global (on corpus) |
|---|---|---|---|---|---|---|
| valentine_plain | features | 0.4484 | 0.4454 | 0.4863 | 0.07 | -0.14 |
| goby | features | 0.4116 | 0.4080 | 0.4477 | 0.09 | -0.06 |
| hdxsm | features | 0.5002 | 0.4979 | 0.5406 | 0.05 | -0.24 |
| gdcsm | features | 0.1284 | 0.1295 | 0.1353 | -0.20 | -0.27 |
| madi_indep | features | 0.5058 | 0.5310 | 0.5570 | -0.96 | -1.16 |
| valentine_plain | features + c/a | 0.4484 | 0.4454 | 0.4863 | 0.07 | -0.14 |
| goby | features + c/a | 0.4116 | 0.4080 | 0.4477 | 0.09 | -0.06 |
| hdxsm | features + c/a | 0.5002 | 0.4979 | 0.5406 | 0.05 | -0.24 |
| gdcsm | features + c/a | 0.1284 | 0.1295 | 0.1353 | -0.20 | -0.27 |
| madi_indep | features + c/a | 0.5058 | 0.5310 | 0.5570 | -0.96 | -1.16 |

## Stop rules (D's)

- headroom oracle_ms − best global on the independent corpora: goby +0.0339, hdxsm +0.0339, gdcsm +0.0054 → stop rule 1 not met
- learned weights (without c/a) recover ≥ 1/3 on 0 of 3; an analytic rule beats frozen_V on some independent corpus: True → stop rule 2 not met
- **verdict: go to stage 2 (after a pre-registration)**
