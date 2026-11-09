# Erratum 1 to Amendment 8 (class nodes on the full panel): scorer fix before any number existed

**Written by C on 2026-10-04 after A's crash report; no classfull number had been computed or written.**

**What happened.** The frozen scorer `layer3b/score_classfull.py` (SHA-256
2f0157474d02edcda6025be8ecbea8285ce3db319e386cf5f1feb85a9c7fb8e4) passed both checks on `classfull_plain`:
- the gate, max |difference| 0;
- the class edges, 135,064 checked, 0 wrong; 2,456 cells without a class output, reported only.

It then crashed (ZeroDivisionError in `mrr_both`) before writing anything. The cause: layer 1's `melnik` output S
was used without the finiteness and shape check that the class arms get. `extrema__melnik` has 120 non-finite
outputs where q is finite (A's scan). So S.max() and the B arms became NaN.

**The change** (and nothing else):
1. A scenario whose layer-1 `melnik` output is non-finite or misshapen is skipped for that matcher, before any arm, L,
   B, A or H8a block is built. This is §3's inclusion rule: a matcher without a finite effect on every scenario of the
   case set leaves the contrast, as `score_layer1.py` treats the same outputs.
2. The same check for `constant__melnik` (the H8a blocks), which would otherwise mis-block silently (B's review).
3. The report lists the skipped cells per reason next to the checks:
   - layer-1 `melnik` output not finite or misshapen;
   - class-arm output not finite or misshapen;
   - `constant__melnik` not finite or misshapen;
   - D missing or not finite.

Hypotheses, statistic, case sets, arms and the edge check are unchanged.

**Expected effect:** `extrema` leaves the plain contrasts (as it already leaves the layer-1 means through
partial coverage).

**Tests** (C's local screen runs only):
- the regular test reproduces X22–X24 exactly, as before the change;
- with one layer-1 output made non-finite, the scorer runs, reports 1 skipped cell, and that matcher leaves every
  contrast.

**MD5 checked before the re-freeze** (`layer3b/scan_nonfinite.py --regime md5`): no matcher has a finite q with a
non-finite layer-1 output; only `datatype`'s 86 non-finite q, which the scorer already skipped.

**New scorer:** `layer3b/score_classfull.py`, SHA-256 63ac2d642349166fefa57d1a6349e849a890ec03bb03db5bfb342408ccff65f3, frozen with `score_r7.py` (761f2c14…).
B rechecks it. A appends this erratum after Amendment 8 and runs the new scorer once per regime.
