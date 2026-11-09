# Additive datatype baseline against the reference SF (plain; exploratory, X19, X26)

75 initial matchers; beta chosen on the screen scenarios, evaluated on 226 dev scenarios outside the screen (lockbox not read). Macro over families (>= 5 scenarios), then the mean over matchers. D's NaN cells set to 0 in 64 scenarios.

Universal beta: 5. Per-matcher beta: 0.02 (4), 0.05 (3), 0.1 (9), 0.2 (15), 0.5 (8), 1 (5), 2 (8), 5 (11), 1000 (12)

| method (gain over q) | J_tf | R_exp | F1_Hr | matchers with J gain > 0 |
|---|---|---|---|---|
| SF reference (melnik) | +0.0158 | +0.0280 | +0.0036 | 53 / 75 |
| q + beta D, per-matcher beta | +0.0322 | +0.0394 | +0.0249 | 64 / 75 |
| q + beta D, universal beta | +0.0323 | +0.0393 | +0.0253 | 65 / 75 |
| q + beta D + gamma [same class], pattern classes (beta, gamma = 5, 0.2) | +0.0717 | +0.0777 | +0.0657 | 67 / 75 |
| q + beta D + gamma [same class], semantic types (beta, gamma = 5, 0.1) | +0.0694 | +0.0762 | +0.0626 | 68 / 75 |

With pattern classes: ahead of SF for 69 of 75 matchers; minus q + beta D (universal) +0.0394.
With semantic types: ahead of SF for 70 of 75 matchers; minus q + beta D (universal) +0.0371.

SF minus baseline (J_tf): per-matcher beta mean -0.0163, SF ahead for 13, baseline ahead for 62; universal beta mean -0.0165, SF ahead 13, baseline ahead 62.

Residual of SF = a q + b_block, blocks from SF on the constant matcher (Amendment 1): 21450 fits; max relative residual median 7.89e-08, 95th pct 2.16e-07, max 1.20e-05; share below 1e-4: 1.000.
Residual of SF = a q + b_block, blocks from the datatype matcher: 21450 fits; max relative residual median 8.28e-08, 95th pct 4.41e-02, max 9.38e-01; share below 1e-4: 0.909.
Blocks per fit (constant partition): median 9, max 25; blocks per cell median 0.022, max 0.562 (a fit has blocks + 1 parameters).
