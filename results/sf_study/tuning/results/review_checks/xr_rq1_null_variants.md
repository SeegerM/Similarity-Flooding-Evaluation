# Ceiling-only null: robustness variants (EXPLORATORY, external review 2026-11-03)

77 matchers, 547 scenarios, 1000 draws. 'below' = observed rho under the null's 2.5% bound.

| variant | row | observed | null median [2.5%, 97.5%] | below |
|---|---|---|---|---|
| published | raw ranking | -0.67 | -0.69 [-0.78, -0.57] | no |
| published | rel ranking | -0.48 | +0.06 [-0.14, +0.26] | yes |
| published | raw assignment | -0.49 | -0.67 [-0.77, -0.54] | no |
| published | rel assignment | -0.45 | -0.49 [-0.63, -0.32] | no |
| V1_matcher_level | raw ranking | -0.67 | -0.00 [-0.22, +0.21] | yes |
| V1_matcher_level | rel ranking | -0.48 | +0.26 [+0.06, +0.45] | yes |
| V1_matcher_level | raw assignment | -0.49 | -0.01 [-0.24, +0.21] | yes |
| V1_matcher_level | rel assignment | -0.45 | +0.06 [-0.17, +0.29] | yes |
| V2_floor_scaled | raw ranking | -0.67 | -0.69 [-0.78, -0.57] | no |
| V2_floor_scaled | rel ranking | -0.48 | +0.06 [-0.14, +0.26] | yes |
| V2_floor_scaled | raw assignment | -0.49 | -0.67 [-0.77, -0.54] | no |
| V2_floor_scaled | rel assignment | -0.43 | -0.49 [-0.63, -0.32] | no |
| V3_nonnegative | raw ranking | -0.69 | -0.69 [-0.80, -0.57] | no |
| V3_nonnegative | rel ranking | -0.47 | +0.08 [-0.12, +0.27] | yes |
| V3_nonnegative | raw assignment | -0.83 | -0.70 [-0.81, -0.53] | yes |
| V3_nonnegative | rel assignment | -0.74 | -0.50 [-0.68, -0.27] | yes |

V3 keeps 71 (ranking) / 53 (assignment) matchers.

Reproduction of the published null: {"R_exp": {"raw_null": [-0.69, -0.78, -0.57], "rel_null": [0.06, -0.14, 0.26], "obs_raw": -0.67, "obs_rel": -0.48}, "F1_Hr": {"raw_null": [-0.67, -0.77, -0.54], "rel_null": [-0.49, -0.63, -0.32], "obs_raw": -0.49, "obs_rel": -0.45}}
