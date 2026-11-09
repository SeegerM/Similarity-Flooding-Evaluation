"""Tie-free Hungarian F1 under separate extraction policies, to split the old-vs-invariant F1 difference into its
causes (2026-11-06; plan step 1; EXPLORATORY).

All variants call score_screen_metrics.hungarian_random (50 fixed random row/column orders) on a transformed matrix:

  old    S as stored; cells <= 1e-9 are never assigned.
  round  S / max|S|, rounded to 1e-6. Removes SF's 32-bit storage noise (about 7 significant digits) relative to the
         matrix's own scale; the abstention rule is unchanged (cells <= 0 are never assigned). Scale-invariant, not
         offset-invariant.
  shift  S - min(S), not rounded: cells at the matrix minimum are never assigned. Changes only the abstention rule
         (subtracting a constant shifts every complete assignment's sum by the same amount).
  inv    shift and round (f1_invariant.hungarian_invariant): both changes.
  full   as inv, but every assigned pair is kept (no abstention at all). Invariant under a*S + b, a > 0.

old -> round  isolates the storage noise (ties between equally good assignments that 32-bit storage breaks);
round -> inv  isolates the abstention rule (dropping pairs at the matrix minimum);
full          is the invariant alternative that never abstains.
"""
import sys
from pathlib import Path

import numpy as np

ST = Path(__file__).resolve().parent.parent.parent / 'sf-matcher-study'
sys.path.insert(0, str(ST / 'tuning'))
from score_screen_metrics import hungarian_random  # noqa: E402

ROUND = 6
POLICIES = ('old', 'round', 'shift', 'inv', 'full')


def _scaled(S):
    S = np.asarray(S, dtype=float)
    m = float(np.max(np.abs(S))) if S.size else 0.0
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(S)
    return np.round(S / m, ROUND)


def _shifted(S, rnd):
    S = np.asarray(S, dtype=float)
    lo, hi = float(np.min(S)), float(np.max(S))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(S)
    T = (S - lo) / (hi - lo)
    return np.round(T, ROUND) if rnd else T


def f1_policy(S, gt, perms, policy):
    if policy == 'old':
        return hungarian_random(np.asarray(S, dtype=float), gt, perms)
    if policy == 'round':
        return hungarian_random(_scaled(S), gt, perms)
    if policy == 'shift':
        return hungarian_random(_shifted(S, False), gt, perms)
    if policy == 'inv':
        return hungarian_random(_shifted(S, True), gt, perms)
    if policy == 'full':
        # a constant matrix is all ties: a complete assignment then picks one at random (50 fixed orders), like any
        # other tie (fixed 2026-11-06 after the ChEMBL report; before, constant matrices assigned nothing)
        return hungarian_random(_shifted(S, True), gt, perms, eps=-1.0)
    raise ValueError(policy)


def all_policies(S, gt, perms):
    return {p: f1_policy(S, gt, perms, p) for p in POLICIES}


if __name__ == '__main__':
    from score_screen_metrics import permutations
    gt = np.eye(2, dtype=bool)
    S = np.array([[1.0, 0.5], [0.5, 0.5]])
    pr = permutations(S.shape, 'counterexample')
    print('counterexample', {p: round(v, 3) for p, v in all_policies(S, gt, pr).items()})
    rng = np.random.default_rng(1)
    worst = {'round': 0.0, 'inv': 0.0, 'full': 0.0}
    for t in range(200):
        n, m = rng.integers(2, 9), rng.integers(2, 9)
        S = np.round(rng.random((n, m)), 2) * (rng.random((n, m)) > 0.3)
        g = np.zeros((n, m), dtype=bool)
        for i in range(min(n, m)):
            g[i, rng.integers(0, m)] = True
        pr = permutations((n, m), f'selftest/{t}')
        a, b = rng.uniform(0.01, 10), rng.uniform(0, 5)
        worst['round'] = max(worst['round'], abs(f1_policy(S, g, pr, 'round') - f1_policy(a * S, g, pr, 'round')))
        for p in ('inv', 'full'):
            worst[p] = max(worst[p], abs(f1_policy(S, g, pr, p) - f1_policy(a * S + b, g, pr, p)))
    print('max change under rescaling (round: a*S; inv, full: a*S + b):', worst)
