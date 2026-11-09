"""Grain inference from within-table dependencies, rules G1 and G2.

Preregistered in `GRAIN_PILOT.md` section 3 (commit 1e8be7b). The thresholds
below are the registered ones and are not to be tuned. Sensitivity runs pass
other values explicitly and are reported, never selected from.

A column's *grain* is the coarsest entity whose anchor genuinely and
near-functionally determines it. Inputs are one table's values only: no labels,
no ground truth, no other table, no matcher output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from smx.afd_measures import _positive_ratio, afd_matrices
from smx.relational_roles import MI_EPS, _codes, _entropy, aligned_values

MU_MIN = 0.50
G3N_MIN = 0.90
NEARKEY_MAX = 0.90
FACT = "FACT"


@dataclass(frozen=True)
class Grain:
    """The G1/G2 result for one table. Column indices refer to input order."""

    width: int
    rows: int
    distinct: tuple            # per-column non-null distinct count
    determined: dict           # anchor (tuple of indices) -> frozenset of dependents
    home: tuple                # per column: anchor tuple, or None for FACT
    edges: frozenset           # genuine unary edges (x, a), for 1-WL

    @property
    def anchors(self) -> tuple:
        return tuple(sorted(self.determined, key=lambda k: (len(k), k)))

    def class_key(self, column: int) -> str:
        anchor = self.home[column]
        return FACT if anchor is None else "+".join(str(i) for i in anchor)

    def classes(self) -> dict:
        out: dict = {}
        for column in range(self.width):
            out.setdefault(self.class_key(column), []).append(column)
        return {key: tuple(members) for key, members in out.items()}

    def summary(self) -> dict:
        sizes = [len(members) for members in self.classes().values()]
        return {
            "G": len(sizes),
            "G_eff": sum(size >= 2 for size in sizes),
            "spread": 1.0 - max(sizes) / self.width if self.width else 0.0,
            "W": wl_classes(self.width, self.edges),
            "anchors": len(self.determined),
            "genuine_edges": len(self.edges),
        }


def _directed_scores(x: np.ndarray, y: np.ndarray, y_total: int):
    """(mu_plus, g3n) for X -> Y on pair-complete rows.

    A transcription of the per-direction arithmetic in `afd_matrices`, needed for
    composite determinants that are not columns. `tests/test_grain.py` requires it
    to reproduce `afd_matrices` on unary pairs in both directions, to 1e-12: the
    reverse direction sums the same joint cells in a different order.
    """
    valid = (x >= 0) & (y >= 0)
    n = int(valid.sum())
    if n < 2:
        return 0.0, 0.0
    a, b = x[valid], y[valid]
    cx, cy = np.bincount(a), np.bincount(b)
    if _entropy(cx) <= MI_EPS or _entropy(cy) <= MI_EPS:
        return 0.0, 0.0
    nx = int(np.count_nonzero(cx))
    keys, counts = np.unique(a * y_total + b, return_counts=True)
    ax = keys // y_total
    degenerate = [0]
    pdep = float(np.sum(counts.astype(float) ** 2 / cx[ax]) / n)
    collision = float(np.sum((cy / n) ** 2))
    expected = collision + (nx - 1) / (n - 1) * (1 - collision)
    signed = pdep - expected
    if abs(signed) <= MI_EPS:
        signed = 0.0
    mu = 0.0 if nx == n else _positive_ratio(signed, 1 - expected, degenerate)
    group_max = np.zeros(len(cx))
    np.maximum.at(group_max, ax, counts)
    conform = float(group_max.sum()) / n
    modal = float(cy.max()) / n
    return mu, _positive_ratio(conform - modal, 1 - modal, degenerate)


def _composite_codes(coded, members):
    """Joint codes of several columns; null wherever any member is null."""
    parts = [coded[i][0] for i in members]
    valid = np.all([p >= 0 for p in parts], axis=0)
    joint = np.full(len(parts[0]), -1, dtype=np.int64)
    if valid.any():
        _, inverse = np.unique(np.stack([p[valid] for p in parts], axis=1),
                               axis=0, return_inverse=True)
        joint[valid] = inverse.ravel()
    return joint


NEARKEY_GRID_MAX = 0.95  # largest near-key cut in the registered sensitivity grid


@dataclass(frozen=True)
class Measures:
    """Everything the rule reads from one table, computed once.

    Thresholds are applied afterwards by `grain_from_measures`, so the registered
    sensitivity grid is a re-thresholding, not a recomputation.
    """

    rows: int
    distinct: tuple
    mu: np.ndarray
    g3n: np.ndarray
    composites: dict  # (i, j) -> (d, mu vector over columns, g3n vector over columns)


def measure_table(columns: Sequence[Sequence[str]],
                  composites: Optional[Iterable[Sequence[int]]] = None) -> Measures:
    width = len(columns)
    if width == 0:
        raise ValueError("a table needs at least one column")
    rows = len(columns[0])
    matrices, cov, _ = afd_matrices([list(c) for c in columns], with_rfi=False)
    distinct = tuple(int(cov["distinct"][i, i]) for i in range(width))
    scored: dict = {}
    if composites is not None:
        coded = [_codes(c) for c in aligned_values([list(c) for c in columns])]
        for members in sorted({tuple(sorted(m)) for m in composites}):
            if len(members) != 2 or len(set(members)) != 2:
                raise ValueError(f"G2 composites must be column pairs, got {members}")
            joint = _composite_codes(coded, members)
            d = len(set(joint[joint >= 0].tolist()))
            mu = np.zeros(width)
            g3n = np.zeros(width)
            # A composite at or above the largest registered near-key cut can
            # never be a candidate under any grid setting; skip scoring it.
            if d >= 2 and rows and d / rows < NEARKEY_GRID_MAX:
                for a in range(width):
                    if a not in members:
                        mu[a], g3n[a] = _directed_scores(joint, coded[a][0], max(coded[a][1], 1))
            scored[members] = (d, mu, g3n)
    return Measures(rows=rows, distinct=distinct, mu=matrices["mu_plus"],
                    g3n=matrices["g3n"], composites=scored)


def grain_from_measures(m: Measures, *, use_composites: bool = False,
                        mu_min: float = MU_MIN, g3n_min: float = G3N_MIN,
                        nearkey_max: float = NEARKEY_MAX) -> Grain:
    """Rule G1, or G2 with `use_composites` (arity-2 determinant candidates).

    1. Candidate anchor X: 2 <= d(X) and d(X)/n < nearkey_max.
    2. Genuine X -> a: mu_plus >= mu_min and g3n >= g3n_min, a not in X.
    3. X is an anchor if it genuinely determines at least one column.
    4. home(a): among anchors containing or determining a, the smallest d(X);
       ties by arity, then by column indices. No such anchor: FACT.
    """
    width, rows = len(m.distinct), m.rows

    def candidate(d: int) -> bool:
        return d >= 2 and rows > 0 and d / rows < nearkey_max

    determined: dict = {}
    anchor_distinct: dict = {}
    edges = set()
    for x in range(width):
        if not candidate(m.distinct[x]):
            continue
        dependents = frozenset(a for a in range(width)
                               if a != x and m.mu[x, a] >= mu_min and m.g3n[x, a] >= g3n_min)
        edges.update((x, a) for a in dependents)
        if dependents:
            determined[(x,)] = dependents
            anchor_distinct[(x,)] = m.distinct[x]
    if use_composites:
        for members, (d, mu, g3n) in sorted(m.composites.items()):
            if not candidate(d):
                continue
            dependents = frozenset(a for a in range(width) if a not in members
                                   and mu[a] >= mu_min and g3n[a] >= g3n_min)
            if dependents:
                determined[members] = dependents
                anchor_distinct[members] = d

    home = []
    for a in range(width):
        owners = [x for x, deps in determined.items() if a in x or a in deps]
        home.append(min(owners, key=lambda x: (anchor_distinct[x], len(x), x)) if owners else None)
    return Grain(width=width, rows=rows, distinct=m.distinct, determined=determined,
                 home=tuple(home), edges=frozenset(edges))


def infer_grain(columns: Sequence[Sequence[str]],
                composites: Optional[Iterable[Sequence[int]]] = None, *,
                mu_min: float = MU_MIN, g3n_min: float = G3N_MIN,
                nearkey_max: float = NEARKEY_MAX) -> Grain:
    """Rule G1, or G2 when `composites` is given. See `grain_from_measures`."""
    return grain_from_measures(measure_table(columns, composites),
                               use_composites=composites is not None, mu_min=mu_min,
                               g3n_min=g3n_min, nearkey_max=nearkey_max)


def wl_classes(width: int, edges: Iterable[tuple]) -> int:
    """Stable 1-WL colour count on a directed graph, from uniform colours."""
    out = [[] for _ in range(width)]
    into = [[] for _ in range(width)]
    for x, a in edges:
        out[x].append(a)
        into[a].append(x)
    colours = [0] * width
    count = 1 if width else 0
    while True:
        signatures = [(colours[v], tuple(sorted(colours[u] for u in out[v])),
                       tuple(sorted(colours[u] for u in into[v]))) for v in range(width)]
        palette = {s: i for i, s in enumerate(sorted(set(signatures)))}
        colours = [palette[s] for s in signatures]
        if len(palette) == count:
            return count
        count = len(palette)


def adjusted_rand_index(truth: Sequence, predicted: Sequence) -> float:
    """Hubert–Arabie ARI. Two trivial partitions that agree score 1.0."""
    if len(truth) != len(predicted):
        raise ValueError("partitions differ in length")
    n = len(truth)
    if n < 2:
        return 1.0

    def pairs(k):
        return k * (k - 1) / 2.0

    joint: dict = {}
    rows: dict = {}
    cols: dict = {}
    for t, p in zip(truth, predicted):
        joint[t, p] = joint.get((t, p), 0) + 1
        rows[t] = rows.get(t, 0) + 1
        cols[p] = cols.get(p, 0) + 1
    index = sum(pairs(v) for v in joint.values())
    a = sum(pairs(v) for v in rows.values())
    b = sum(pairs(v) for v in cols.values())
    expected = a * b / pairs(n)
    maximum = (a + b) / 2.0
    if maximum == expected:
        return 1.0
    return float((index - expected) / (maximum - expected))


__all__ = ["FACT", "G3N_MIN", "MU_MIN", "NEARKEY_MAX", "Grain", "Measures", "adjusted_rand_index",
           "grain_from_measures", "infer_grain", "measure_table", "wl_classes"]
