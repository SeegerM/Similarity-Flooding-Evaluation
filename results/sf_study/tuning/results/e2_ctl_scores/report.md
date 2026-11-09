# External replication (E2), corpus ctl: registered analysis

| prediction | matchers | estimate | 95% CI | per family | verdict |
|---|---|---|---|---|---|
| P1 ρ(initial R_exp, ΔR_exp melnik) (plain) | 10 | -0.248 | [-0.636, -0.066] | | CONFIRMED (decided on t2d) |
| P1 ρ(initial R_exp, ΔR_exp melnik) (opaque) | 10 | -0.345 | [-0.612, +0.297] | | not confirmed (decided on t2d) |
| **P2** fd − melnik (plain, R_exp) | 10 | -0.3884 | [-0.4554, -0.3191] | Lahman -0.4407, MusicBrainz -0.3361 | CONFIRMED |
| **P2** fd − melnik (opaque, J) | 10 | +0.0429 | [+0.0207, +0.0679] | Lahman -0.0203, MusicBrainz +0.1061 | CONFIRMED |
| **P3** additive_d − melnik (plain, J) | 10 | +0.0151 | [+0.0079, +0.0238] | Lahman +0.0188, MusicBrainz +0.0114 | CONFIRMED |
| **P4** fd_top1 − fd (plain, R_exp) | 10 | +0.3317 | [+0.2714, +0.3924] | Lahman +0.4011, MusicBrainz +0.2623 | CONFIRMED |
| **P4** fd_top1 − fd (opaque, R_exp) | 10 | +0.0187 | [-0.0013, +0.0372] | Lahman +0.0453, MusicBrainz -0.0079 | not confirmed |

Reported, not tested: coef_prod_fd − melnik and ucc − melnik (J_tf), per regime.
| coef_prod_fd − melnik (plain, J) | 10 | -0.0632 | [-0.0853, -0.0394] | Lahman -0.0626, MusicBrainz -0.0639 | (descriptive) |
| ucc − melnik (plain, J) | 10 | +0.0035 | [+0.0000, +0.0071] | Lahman +0.0000, MusicBrainz +0.0071 | (descriptive) |
| coef_prod_fd − melnik (opaque, J) | 10 | +0.0543 | [+0.0345, +0.0771] | Lahman +0.0171, MusicBrainz +0.0914 | (descriptive) |
| ucc − melnik (opaque, J) | 10 | +0.0136 | [+0.0000, +0.0272] | Lahman +0.0000, MusicBrainz +0.0272 | (descriptive) |
