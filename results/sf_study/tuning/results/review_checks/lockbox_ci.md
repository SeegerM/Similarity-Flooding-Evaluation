# Lockbox: intervals per row (EXPLORATORY review check; the registered HL1-HL4 are in lockbox_plain/hl_report.md)

94 scenarios (ChEMBL 22, OpenData 28, TPC-DI 44), 7 panel matchers (kang_naughton, ks, levenshtein, pmf, prefix_suffix, role_gcov, semstruct). Mean J_tf gain over the initial matcher; 95% percentile bootstrap, stratified by family (10000, seed 20261002), and over column-split clusters.

| arm | mean | 95% CI (stratified) | 95% CI (clusters) | matchers > 0 |
|---|---|---|---|---|
| Tuned, per matcher | +0.0172 | [+0.0093, +0.0244] | [+0.0053, +0.0268] | 5 of 7 |
| Tuned, one for all matchers | +0.0084 | [+0.0012, +0.0166] | [-0.0020, +0.0191] | 4 of 7 |
| Standard | -0.0298 | [-0.0432, -0.0166] | [-0.0475, -0.0129] | 4 of 7 |
| No propagation (alpha = 0) | -0.0002 | [-0.0004, -0.0000] | [-0.0004, -0.0000] | 0 of 7 |
| SF as published (report only) | +0.0380 | [+0.0255, +0.0523] | [+0.0257, +0.0535] | 6 of 7 |

## SF as published against the tuned configurations (descriptive: not a tuning candidate)

| contrast | mean | 95% CI (stratified) | 95% CI (clusters) | matchers > 0 |
|---|---|---|---|---|
| SF as published - Tuned, per matcher | +0.0208 | [+0.0097, +0.0341] | [+0.0084, +0.0370] | 5 of 7 |
| SF as published - Tuned, one for all matchers | +0.0296 | [+0.0217, +0.0377] | [+0.0230, +0.0375] | 6 of 7 |
| SF as published - No propagation (alpha = 0) | +0.0382 | [+0.0256, +0.0525] | [+0.0259, +0.0537] | 6 of 7 |
| SF as published - Standard | +0.0678 | [+0.0541, +0.0835] | [+0.0510, +0.0873] | 6 of 7 |

Non-inferiority margin used elsewhere in the paper (evidence matrix): -0.01 in J_tf.
