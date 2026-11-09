### Draft amendment (tuning PREREGISTRATION.md, Amendment 4): clean nested selection as the primary dev result, a non-degenerate configuration for all matchers, and the single lockbox scoring

Written by B on 2026-10-04 at the user's request ("yes to E1", backlog E1). **A reviews it, fills in the hashes
and appends it before any lockbox run.** No lockbox data has been run or scored.

#### 1. What has been seen
- **The dev stage (registered `dev_plain`) and its LOFO table.**
- **The supplementary selection analyses (ER1, ER2; A's `supp_selection.py`, report
  `tuning/results/supp_selection_plain/report.md`):**
  - the per-fold masks (each fold's own screen top 10 plus production);
  - the corrected per-matcher LOFO: +0.0100 J_tf, or +0.0206 with abstention;
  - the universal shortlist of 20 configurations, screened on their own across-matcher objective, with the
    universal choice `fpC__INV_PROD__ID__fdoff` (also with abstention).
- Nothing on the lockbox.

#### 2. What changes, and why
1. **The leak and its scope.** The registered LOFO table selects from the union of all folds' shortlists, so in
   9 of 45 choices a fold used a configuration that only another fold's screen kept, and that screen had seen
   the held-out family (ER2). The leak affects **only the LOFO estimates**: the final per-matcher choice is
   made on all dev families with no family held out, so `chosen.json`'s per-matcher entries never touched
   lockbox data and stay valid.
2. **The registered universal choice was degenerate.** After per-matcher pruning, only production was kept
   for every matcher (ER1).
3. **Therefore:**
   - **(a) Primary dev result:** the corrected LOFO table, in which each fold selects only from its own screen
     shortlist (+ production). The registered LOFO table is reported in the supplement as the historical
     analysis, with the leak described.
   - **(b) The universal configuration for the lockbox:** `fpC__INV_PROD__ID__fdoff` (ER1, chosen on all dev
     families from the 20-configuration universal shortlist; abstention gives the same). It **replaces**
     production as the universal entry. Production stays in the lockbox as a baseline, as registered.
   - **(c) The per-matcher entries of `chosen.json` are unchanged.** `chosen.json` itself is not rewritten; the
     clean universal choice is read from `universal_choice.json` (hash below).

#### 3. The lockbox (scored once)
- **Scenarios:** the 96 lockbox scenarios, plain.
- **Matchers:** the 9 panel matchers; the controls are reported separately.
- **Arms per matcher:**
  - its `chosen.json` per-matcher configuration ("own");
  - the clean universal configuration;
  - production;
  - α = 0 (no propagation);
  - layer 1's `melnik` (report-only, Amendment 3);
  - the tie-breaker (Amendment 3's secondary analysis).
- **Statistics:** as registered. J_tf (Amendment 2), macro over families with ≥ 5 scenarios, mean over the
  non-control matchers, 95% stratified bootstrap over scenarios, seed 20261002. Added as sensitivity: the
  column-split cluster bootstrap (X18's `cluster_of`).
- **Hypotheses** (none were registered for the lockbox beyond "the lockbox number is the headline"; these are
  stated now, before any lockbox data):
  - **HL1:** own − production > 0.
  - **HL2:** own − α = 0 > 0. The corrected dev stage gives +0.010 against 0.000, a narrow margin, so HL2 is a
    real test.
  - **HL3:** own − universal > 0. The dev stage says yes: +0.023.
  - **HL4:** universal − production > 0. The dev stage says yes: +0.013.
  - Confirmed if the lower bound of the stratified bootstrap is > 0. The cluster bootstrap is reported
    alongside.
  - All per-matcher values are reported; none is selected.

#### 4. Code (A fills in)
| file | SHA-256 |
|---|---|
| `tuning/score_tuning.py` (`--stage lockbox`) | … |
| `tuning/supp_selection.py` | … |
| `tuning/results/supp_selection_plain/universal_choice.json` | … |
| `tuning/chosen.json` (unchanged, 8562afd9…) | 8562afd9… |
| `tuning/keep.json` (unchanged, 6a2a5862…) | 6a2a5862… |
| `confirm_tiebreak.py` (Amendment 3; `--scenarios lockbox --chosen tuning/chosen.json --role own`) | … |
