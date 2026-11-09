"""Candidate SF update formulas F1 and F2, and the operator plumbing they share.

Specified by section 5 of `THREE_PAPER_EXPERIMENT_DESIGN_20260922.md`. These are
**proposed operators under test**, not established algorithms and not a claim of
novelty over personalised propagation or confidence-weighted diffusion.

The coordinate space is candidate pairs `(source column, target column)`. A
propagation operator `P` is built from typed schema edges exactly as Similarity
Flooding builds its propagation graph: coordinate `(a, b)` receives support from
`(a', b')` when `(a, l, a')` is a source edge and `(b, l, b')` is a target edge
carrying the same label `l`. Per-relation budgets are applied, then rows are
made substochastic, so `||P||_inf <= 1` holds by construction rather than by
assumption.

Four updates are provided, and the three controls exist because the design
document requires the candidate to beat them before anything is claimed:

    f1_restart          x <- (I - D) q + D P x            confidence-dependent restart
    f2_null             x <- clip(q + eta D H x)          the candidate
    f2_one_step         x <- clip(q + eta D H q)          is iteration doing anything?
    uncentered          x <- clip(q + eta D P x)          is subtraction doing anything?

`H = P_real - P_null` where `P_null` is the mean of `B` edge-rewired operators.
Rewiring is degree-preserving double-edge swap on the FD channel only; other
channels are carried through unchanged, so their one-step contributions cancel
in `H`. That cancellation is a property of this construction and is the reason
F2 is a proposal about FD *arrangement* rather than a replacement for SF.

Randomness is keyed by a caller-supplied stable string identifier hashed with
BLAKE2b, never by Python's process-dependent `hash()`, per the execution
contract in section 2 of the design document.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Default per-relation budgets. A removed FD budget is *not* silently
#: redistributed to the other channels -- section 5.1 forbids that, because it
#: would confound "FD removed" with "other channels strengthened".
DEFAULT_BUDGETS: dict[str, float] = {}


def rng_for(identifier: str) -> np.random.Generator:
    """Deterministic generator keyed by a stable string.

    BLAKE2b rather than `hash()`: the built-in is salted per process, so a rerun
    would silently draw a different null ensemble.
    """
    digest = hashlib.blake2b(identifier.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# --------------------------------------------------------------------- graphs

@dataclass(frozen=True)
class SchemaGraph:
    """Typed directed edges over named nodes."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]   # (tail, label, head)

    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({l for _, l, _ in self.edges}))

    def of_label(self, label: str) -> tuple[tuple[str, str], ...]:
        return tuple((t, h) for t, l, h in self.edges if l == label)

    def replace_label(self, label: str,
                      pairs: Iterable[tuple[str, str]]) -> "SchemaGraph":
        kept = tuple(e for e in self.edges if e[1] != label)
        return SchemaGraph(self.nodes, kept + tuple((t, label, h) for t, h in pairs))


@dataclass
class PairSpace:
    """The candidate coordinates and their index."""

    source: tuple[str, ...]
    target: tuple[str, ...]

    def __post_init__(self):
        self.index = {(a, b): i * len(self.target) + j
                      for i, a in enumerate(self.source)
                      for j, b in enumerate(self.target)}

    @property
    def n(self) -> int:
        return len(self.source) * len(self.target)

    def matrix(self, v: np.ndarray) -> np.ndarray:
        return v.reshape(len(self.source), len(self.target))

    def vector(self, m: np.ndarray) -> np.ndarray:
        return m.reshape(-1)


def build_operator(src: SchemaGraph, tgt: SchemaGraph, space: PairSpace,
                   budgets: Mapping[str, float] | None = None,
                   reverse: bool = True) -> np.ndarray:
    """The propagation operator, row-substochastic.

    One entry per (label, source edge, target edge) agreement, as in the
    propagation graph. `reverse=True` adds the opposite orientation, which is
    Melnik's construction and this project's frozen policy.
    """
    budgets = dict(budgets or {})
    P = np.zeros((space.n, space.n))
    for label in sorted(set(src.labels()) & set(tgt.labels())):
        w = float(budgets.get(label, 1.0))
        if w == 0.0:
            continue
        L = np.zeros_like(P)
        for (sa, sb) in src.of_label(label):
            for (ta, tb) in tgt.of_label(label):
                i, j = space.index.get((sa, ta)), space.index.get((sb, tb))
                if i is None or j is None:
                    continue
                L[i, j] += 1.0
                if reverse:
                    L[j, i] += 1.0
        P += w * L
    return row_substochastic(P)


def row_substochastic(P: np.ndarray) -> np.ndarray:
    """Scale every row with mass above one down to one; leave the rest alone.

    Rows are *not* forced up to one. A coordinate with little structural support
    should keep little, and normalising it up would invent support -- the same
    error `SF_NORMALISATION` documents for the global maximum.
    """
    s = P.sum(axis=1, keepdims=True)
    scale = np.where(s > 1.0, 1.0 / np.maximum(s, 1e-300), 1.0)
    return P * scale


def canonicalise(q: np.ndarray) -> np.ndarray:
    """q / max(q) when nonzero, else q. Section 5.1's positive-scale removal."""
    m = float(np.max(q)) if q.size else 0.0
    return q / m if m > 0 else q.copy()


# ------------------------------------------------------------------- rewiring

def rewire_pairs(pairs: Sequence[tuple[str, str]], rng: np.random.Generator,
                 attempts_per_edge: int = 20) -> tuple[tuple[str, str], ...]:
    """Degree-preserving double-edge swap.

    Out-degree of every tail and in-degree of every head are preserved exactly.
    Self-loops and multi-edges are rejected, so a graph too constrained to
    randomise simply stops early -- the caller must check, per section 5.3
    ("record the degeneracy and abstain"), rather than relax the invariant.
    """
    edges = [list(p) for p in pairs]
    if len(edges) < 2:
        return tuple(tuple(e) for e in edges)
    present = {tuple(e) for e in edges}
    swaps = 0
    for _ in range(attempts_per_edge * len(edges)):
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            continue
        (t1, h1), (t2, h2) = tuple(edges[i]), tuple(edges[j])
        if t1 == t2 or h1 == h2:
            continue
        if t1 == h2 or t2 == h1:          # would create a self-loop
            continue
        if (t1, h2) in present or (t2, h1) in present:
            continue
        present.discard((t1, h1))
        present.discard((t2, h2))
        present.add((t1, h2))
        present.add((t2, h1))
        edges[i] = [t1, h2]
        edges[j] = [t2, h1]
        swaps += 1
    return tuple(tuple(e) for e in edges)


def rewire_fidelity(original: Sequence[tuple[str, str]],
                    rewired: Sequence[tuple[str, str]]) -> dict:
    """Constraint fidelity and mixing, both of which section 5.3 asks for."""
    def deg(pairs):
        out, inn = {}, {}
        for t, h in pairs:
            out[t] = out.get(t, 0) + 1
            inn[h] = inn.get(h, 0) + 1
        return out, inn
    o_out, o_in = deg(original)
    r_out, r_in = deg(rewired)
    shared = len(set(original) & set(rewired))
    return {"n_edges": len(original),
            "out_degree_preserved": o_out == r_out,
            "in_degree_preserved": o_in == r_in,
            "edges_unchanged": shared,
            "fraction_rewired": (1.0 - shared / len(original)) if original else 0.0}


def components(graph: SchemaGraph) -> dict[str, tuple[str, ...]]:
    """Connected components over the undirected edge set, keyed per node.

    The key is the component's sorted node tuple, which is stable under adding a
    *disjoint* component elsewhere in the graph. That stability is what makes
    component-local rewiring satisfy the locality condition in section 5.4.6.
    """
    parent = {n: n for n in graph.nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t, _, h in graph.edges:
        if t in parent and h in parent:
            rt, rh = find(t), find(h)
            if rt != rh:
                parent[rt] = rh
    groups: dict[str, list[str]] = {}
    for n in graph.nodes:
        groups.setdefault(find(n), []).append(n)
    return {n: tuple(sorted(members))
            for members in groups.values() for n in members}


def rewire_label(graph: SchemaGraph, label: str, identifier: str,
                 draw: int, component_local: bool = True):
    """Rewire one label's edges, within components when asked.

    Global rewiring mixes edges across disconnected parts of the graph, which
    makes the null operator -- and therefore every score -- depend on what else
    happens to be in the same input. Section 5.4.6 states the condition; the S1
    locality check measures what violating it costs.
    """
    pairs = graph.of_label(label)
    if not component_local:
        return rewire_pairs(pairs, rng_for(f"{identifier}|{label}|{draw}"))
    comp = components(graph)
    buckets: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for t, h in pairs:
        buckets.setdefault(comp.get(t, (t,)), []).append((t, h))
    out: list[tuple[str, str]] = []
    for key in sorted(buckets):
        tag = "".join(key)
        out.extend(rewire_pairs(
            buckets[key], rng_for(f"{identifier}|{label}|{tag}|{draw}")))
    return tuple(out)


def null_operator(src: SchemaGraph, tgt: SchemaGraph, space: PairSpace,
                  label: str, identifier: str, B: int = 5,
                  budgets: Mapping[str, float] | None = None,
                  reverse: bool = True,
                  component_local: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Mean of `B` operators with `label` rewired on both sides.

    Returns the averaged operator and one fidelity record per draw. The mean of
    row-substochastic matrices is row-substochastic, so `||P_null||_inf <= 1`
    survives the averaging and the F2 bounds hold.
    """
    mats, records = [], []
    for b in range(B):
        s_pairs = rewire_label(src, label, identifier, b, component_local)
        t_pairs = rewire_label(tgt, label, identifier, b, component_local)
        records.append({"draw": b,
                        "source": rewire_fidelity(src.of_label(label), s_pairs),
                        "target": rewire_fidelity(tgt.of_label(label), t_pairs)})
        mats.append(build_operator(src.replace_label(label, s_pairs),
                                   tgt.replace_label(label, t_pairs),
                                   space, budgets, reverse))
    return (np.mean(mats, axis=0) if mats else np.zeros((space.n, space.n)),
            records)


# -------------------------------------------------------------------- updates

@dataclass
class Fixpoint:
    x: np.ndarray
    iterations: int
    residual: float
    applications: int                     # sparse operator applications, all of them
    converged: bool
    clipped_fraction: float = 0.0
    trace: list[float] = field(default_factory=list)


def _iterate(step, x0, tol, max_iter, applications_per_step):
    x = x0.copy()
    trace = []
    for k in range(1, max_iter + 1):
        nxt = step(x)
        res = float(np.max(np.abs(nxt - x))) if x.size else 0.0
        trace.append(res)
        x = nxt
        if res <= tol:
            return Fixpoint(x, k, res, k * applications_per_step, True, trace=trace)
    return Fixpoint(x, max_iter, trace[-1] if trace else 0.0,
                    max_iter * applications_per_step, False, trace=trace)


def f1_restart(q: np.ndarray, P: np.ndarray, D: np.ndarray | float,
               tol: float = 1e-10, max_iter: int = 10_000) -> Fixpoint:
    """x <- (I - D) q + D P x. Equal diagonal entries reduce this to restart."""
    d = np.full_like(q, float(D)) if np.isscalar(D) else np.asarray(D, dtype=float)
    return _iterate(lambda x: (1.0 - d) * q + d * (P @ x), q, tol, max_iter, 1)


def f2_null(q: np.ndarray, H: np.ndarray, eta: float,
            D: np.ndarray | float = 1.0, tol: float = 1e-10,
            max_iter: int = 10_000) -> Fixpoint:
    """x <- clip(q + eta D H x), x0 = q. The candidate formula."""
    d = np.full_like(q, float(D)) if np.isscalar(D) else np.asarray(D, dtype=float)
    clipped = {"n": 0, "total": 0}

    def step(x):
        raw = q + eta * d * (H @ x)
        out = np.clip(raw, 0.0, 1.0)
        clipped["n"] += int(np.count_nonzero(raw != out))
        clipped["total"] += raw.size
        return out

    fp = _iterate(step, q, tol, max_iter, 2)     # H costs two applications
    fp.clipped_fraction = (clipped["n"] / clipped["total"]) if clipped["total"] else 0.0
    return fp


def f2_one_step(q: np.ndarray, H: np.ndarray, eta: float,
                D: np.ndarray | float = 1.0) -> Fixpoint:
    """clip(q + eta D H q). The control for "is iteration useful at all?"."""
    d = np.full_like(q, float(D)) if np.isscalar(D) else np.asarray(D, dtype=float)
    raw = q + eta * d * (H @ q)
    out = np.clip(raw, 0.0, 1.0)
    return Fixpoint(out, 1, float(np.max(np.abs(out - q))) if q.size else 0.0,
                    2, True, float(np.count_nonzero(raw != out)) / max(1, raw.size))


def uncentered(q: np.ndarray, P: np.ndarray, eta: float,
               D: np.ndarray | float = 1.0, tol: float = 1e-10,
               max_iter: int = 10_000) -> Fixpoint:
    """clip(q + eta D P x). The same bounded map without the null subtraction."""
    d = np.full_like(q, float(D)) if np.isscalar(D) else np.asarray(D, dtype=float)
    clipped = {"n": 0, "total": 0}

    def step(x):
        raw = q + eta * d * (P @ x)
        out = np.clip(raw, 0.0, 1.0)
        clipped["n"] += int(np.count_nonzero(raw != out))
        clipped["total"] += raw.size
        return out

    fp = _iterate(step, q, tol, max_iter, 1)
    fp.clipped_fraction = (clipped["n"] / clipped["total"]) if clipped["total"] else 0.0
    return fp


# ------------------------------------------------------------------- readouts

def margin_gate(q: np.ndarray, space: PairSpace, lo: float = 0.0,
                hi: float = 1.0) -> np.ndarray:
    """D from the normalised top-two row margin, per section 5.2.

    A confident row gets a small `d` (little revision permitted); an uncertain
    row gets a large one. The gate is applied to every candidate coordinate of
    that source row, and its functional form is fixed here before any outcome
    is compared, which is the condition section 5.2 states.
    """
    M = space.matrix(q)
    d_rows = np.empty(M.shape[0])
    for i, row in enumerate(M):
        if row.size < 2:
            d_rows[i] = hi
            continue
        top = np.sort(row)[::-1]
        denom = top[0] if top[0] > 0 else 1.0
        margin = float((top[0] - top[1]) / denom)
        d_rows[i] = hi - (hi - lo) * min(max(margin, 0.0), 1.0)
    return space.vector(np.repeat(d_rows[:, None], M.shape[1], axis=1))
