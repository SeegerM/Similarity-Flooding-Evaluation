"""Exact additive-channel alignment null and GT-free gated assignment.

This is a conditional *channel agreement* null, not a proven semantic-nonmatch
model. Shared biases between evidence channels can invalidate its use as a
correspondence p-value. BH/BY refer to endpoint tests, not pair-level FDR.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import false_discovery_control

EPS = 1e-12


def channel_components(base, evidence, weight=.35):
    """Reproduce existing range-aligned fusion in base-range units.

Positive affine score changes do not affect the null. If either channel is
constant, the existing fusion ignores it; the null must then have no power.
"""
    base, evidence = np.asarray(base, float), np.asarray(evidence, float)
    if base.shape != evidence.shape or base.ndim != 2 or not base.size:
        raise ValueError("Expected same-sized nonempty 2-D channels")
    if not np.isfinite(base).all() or not np.isfinite(evidence).all():
        raise ValueError("Nonfinite channel")
    if not 0 <= weight <= 1:
        raise ValueError("Weight outside [0,1]")
    br, er = float(np.ptp(base)), float(np.ptp(evidence))
    lo = float(base.min())
    if br <= EPS:
        return np.zeros_like(base), np.zeros_like(base), base.copy(), False
    a = (base-lo) / br
    if er <= EPS:
        return a, np.zeros_like(base), base.copy(), False
    b = (evidence-float(evidence.min())) / er
    first, second = (1-weight)*a, weight*b
    # Same algebra as scale_aligned_fuse; the unnormalised result is checked
    # against the stored matrix by the experiment, never substituted for it.
    return first, second, (1-weight)*base + weight*(b*br+lo), 0 < weight < 1


def max_alignment_tail(first, second, thresholds):
    """P[max_j(first_j + second_perm(j)) >= t], inclusive of score ties.

Exact, without enumerating permutations. For max < t, row j admits c_j
second-channel entries below t-first_j. These admissible sets are nested.
Sorting c ascending, the number of valid permutations is product(c_i-i),
using zero-based i. Divide by n! to obtain the CDF, then take its complement.
Duplicates are separate slots, as under a uniform permutation of indices.
"""
    a, b, t = np.asarray(first, float), np.sort(np.asarray(second, float)), np.atleast_1d(thresholds).astype(float)
    if a.ndim != 1 or b.shape != a.shape or not a.size:
        raise ValueError("Expected equal nonempty vectors")
    if not all(np.isfinite(x).all() for x in (a, b, t)):
        raise ValueError("Nonfinite input")
    n = a.size
    allowed = np.sort(np.searchsorted(b, t[:, None]-a[None, :]-EPS, side="left"), axis=1)
    choices = allowed - np.arange(n)[None, :]
    valid = (choices > 0).all(axis=1)
    tail = np.ones(t.size)
    # Ratios keep all intermediate quantities bounded; log1p/expm1 retain
    # precision for small tails instead of subtracting nearly equal numbers.
    if valid.any():
        ratios = choices[valid] / (n-np.arange(n))[None, :]
        log_cdf = np.log(ratios).sum(axis=1)
        tail[valid] = -np.expm1(log_cdf)
    return np.clip(tail, 0, 1)


def alignment_pvalues(base, evidence, weight=.35):
    """Row-search- and column-search-adjusted tails for every candidate.

Each row permutes only its auxiliary-channel target slots. Each column uses
the analogous source-slot null separately; there is no asserted joint null
over the whole matrix. Endpoints are tested by their best observed candidate.
"""
    a, b, fused, informative = channel_components(base, evidence, weight)
    s = a+b
    row_p, col_p = np.ones(s.shape), np.ones(s.shape)
    if informative:
        for i in range(s.shape[0]):
            row_p[i] = max_alignment_tail(a[i], b[i], s[i])
        for j in range(s.shape[1]):
            col_p[:, j] = max_alignment_tail(a[:, j], b[:, j], s[:, j])
    return row_p, col_p, fused, informative


def endpoint_cutoff(row_p, col_p, alpha=.05, correction="bh"):
    if not 0 < alpha < 1:
        raise ValueError("alpha outside (0,1)")
    if correction not in ("none", "bh", "by"):
        raise ValueError(correction)
    endpoints = np.concatenate((row_p.min(axis=1), col_p.min(axis=0)))
    if correction == "none":
        return alpha, int((endpoints <= alpha).sum())
    adjusted = false_discovery_control(endpoints, method=correction)
    rejected = adjusted <= alpha
    return (float(endpoints[rejected].max()) if rejected.any() else 0.0), int(rejected.sum())


def gated_assignment(scores, allowed):
    """Maximum ORIGINAL-score 1:1 assignment; zero-cost unmatched dummies.

No cardinality target, bonus, GT count, or p-value re-ranking is introduced.
Forbidden edges are strictly worse than leaving either endpoint unmatched.
"""
    s, allowed = np.asarray(scores, float), np.asarray(allowed, bool)
    if s.ndim != 2 or s.shape != allowed.shape or not np.isfinite(s).all():
        raise ValueError("Invalid assignment input")
    n, m = s.shape
    out = np.zeros(s.shape, dtype=bool)
    if not n or not m:
        return out
    costs = np.zeros((n+m, n+m))
    costs[:n, :m] = np.where(allowed & (s > 0), s, -1.)
    rows, cols = linear_sum_assignment(costs, maximize=True)
    for i, j in zip(rows, cols):
        if i < n and j < m and allowed[i, j] and s[i,j] > 0:
            out[i, j] = True
    return out


def null_assignment(scores, row_p, col_p, alpha=.05, correction="bh"):
    cutoff, rejected = endpoint_cutoff(row_p, col_p, alpha, correction)
    allowed = (row_p <= cutoff) & (col_p <= cutoff) if cutoff > 0 else np.zeros(scores.shape, bool)
    return gated_assignment(scores, allowed), {"cutoff": cutoff, "rejected_endpoints": rejected,
                                               "eligible_pairs": int(allowed.sum())}
