"""The final transferability table: three dependency classes, one protocol.

FINDINGS 30 measured functional dependencies on the 335 confirmation scenarios,
31 measured denial constraints on the same cohort, and 17 measured order -- but
order was measured on 36 ChEMBL scenarios against a different seed, so the three
results have never appeared in one comparable table. This produces that table.

Every class gets the identical treatment: the frozen 65% Levenshtein plus 35%
PMF seed read from the cached per-case matrices, the same role descriptor per
class, one 0.35 fusion, the same six arms, and the same 335 scenarios.

  none      no dependency evidence
  all       every discovered dependency
  random    a size-matched random subset, separating *which* from *how many*
  anchor    the unsupervised ranking at a budget fixed on the 44 exploration
            cases -- the deployable arm, no ground truth anywhere
  oracle    dependencies whose image under the ground-truth correspondence
            exists on the other side; the ceiling

Order dependencies come from thresholding the continuous relation, which is what
makes them computable at these schema widths at all (FINDINGS 16), and their
images preserve operand position, since an order dependency is not symmetric.

Emits a LaTeX table alongside the console summary.

Usage:
    python transferability_table.py [-o reports/transferability_table.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_paths import resolve  # noqa: E402

from anchor_generalization import (anchor_matrix, bare, correspondence,  # noqa: E402
                                   load_side, oracle_signatures)
from class_comparison_conditioned import read_columns  # noqa: E402
from compose_gate import fuse  # noqa: E402
from dc_gate import anchor_ranking as dc_anchor_ranking  # noqa: E402
from dc_gate import build_similarity, load_dcs, oracle_subset  # noqa: E402
from dc_gate import pair_score as dc_pair_score  # noqa: E402
from od_role import cosine_similarity, dc_profile, od_profile, scale_align  # noqa: E402
from order_relation import order_matrix, to_ordered  # noqa: E402
from order_transferability import threshold_ods  # noqa: E402
from relevance import build_similarity as fd_similarity  # noqa: E402
from relevance import determinant_similarity, relevance, signature  # noqa: E402
from transferability_classes import fd_profile  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate, extract, match_metrics  # noqa: E402

TAU = 0.99

# Overlap-free column similarities for the anchor ranking (OVERLAP_FREE_SELECTOR.md).
# "seed" reads the --base-name matrix itself.
SELECTOR_SOURCES = {"pmf": "pmf.csv", "kn": "kn_graph.csv", "seed": None}
SELECTOR_TRANSFORMS = ("raw", "mutual3")


def selector_matrix(path, s_cols, t_cols, transform):
    """A cached score matrix aligned to the scenario's columns by bare label.

    `raw` scales by the maximum (Amendment 1): the FD pairing multiplies one
    similarity per determinant column, so a matrix whose values sit near 0.02
    would penalise wide determinants through its scale alone. `mutual3` keeps
    an entry only if it is among the three largest of both its row and its
    column. A label that does not align is an error, never a silent zero.
    """
    m, rows, cols = read_matrix(path)
    if rows is None or cols is None:
        raise RuntimeError(f"{path}: selector matrix carries no labels")
    r_at = {bare(r.strip('"')): i for i, r in enumerate(rows)}
    c_at = {bare(c.strip('"')): j for j, c in enumerate(cols)}
    missing = [c for c in s_cols if c not in r_at] + [c for c in t_cols if c not in c_at]
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} columns do not align, e.g. {missing[:3]}")
    m = np.asarray(m, dtype=float)[np.ix_([r_at[c] for c in s_cols], [c_at[c] for c in t_cols])]
    m = np.nan_to_num(m, nan=0.0)
    top = m.max() if m.size else 0.0
    m = np.clip(m / top, 0.0, 1.0) if top > 0 else np.zeros_like(m)
    if transform == "mutual3":
        k_r, k_c = min(3, m.shape[1]), min(3, m.shape[0])
        row_kth = np.sort(m, axis=1)[:, -k_r][:, None]
        col_kth = np.sort(m, axis=0)[-k_c, :][None, :]
        m = np.where((m >= row_kth) & (m >= col_kth), m, 0.0)
    return m


def image_filter(cls, own, other, mapping):
    """Dependencies of `own` whose image under `mapping` lies in `other`: the
    oracle test, run against an arbitrary subset of the other side."""
    if cls == "FD":
        key = lambda fd: signature(list(fd[0]), fd[1])  # noqa: E731
        sig = oracle_signatures(own, other, mapping)
        return [f for f in own if key(f) in sig]
    test = oracle_subset if cls == "DC" else od_oracle
    return list(test(list(own), list(other), mapping))


def position_hash(deps, cols):
    """16-hex SHA-256 of `positional`, which is what the committed reports keep."""
    return hashlib.sha256(positional(deps, cols).encode()).hexdigest()[:16]


def paired_subsets(cls, ev, sub_s, sub_t):
    """How many selected dependencies of each side have an image among the
    other side's *selected* dependencies."""
    return (len(image_filter(cls, sub_s, sub_t, ev["map_st"])),
            len(image_filter(cls, sub_t, sub_s, ev["map_ts"])))


def pair_candidates(cls, own_s, own_t, matrix, s_cols, t_cols, partners=5):
    """Candidate dependency pairs, scored with the anchor ranking's own pair
    score: each source dependency's `partners` best targets plus each target
    dependency's `partners` best sources, zero scores dropped, sorted by
    descending score with ties in candidate order (OVERLAP_FREE_SELECTOR.md,
    follow-up 2)."""
    if cls == "FD":
        sim = fd_similarity(matrix, s_cols, t_cols)

        def score(a, b):
            if len(a[0]) != len(b[0]):
                return 0.0
            dep = sim(a[1], b[1], "st")
            return dep * determinant_similarity(a[0], b[0], sim, "st") if dep > 0 else 0.0
    elif cls == "DC":
        sim = build_similarity(matrix, s_cols, t_cols)

        def score(a, b):
            return dc_pair_score(a, b, sim, "st")
    else:
        sim = build_similarity(matrix, s_cols, t_cols)

        def score(a, b):
            (ls, rs), (lt, rt) = a, b
            if len(ls) != len(lt) or len(rs) != len(rt):
                return 0.0
            out = 1.0
            for x, y in zip((*ls, *rs), (*lt, *rt)):
                out *= sim(x, y, "st")
                if out <= 0.0:
                    return 0.0
            return out

    keep = set()
    by_target = [[] for _ in own_t]
    for i, a in enumerate(own_s):
        row = []
        for j, b in enumerate(own_t):
            s = score(a, b)
            if s > 0.0:
                row.append((s, -j))
                heap = by_target[j]
                if len(heap) < partners:
                    heapq.heappush(heap, (s, -i))
                elif (s, -i) > heap[0]:
                    heapq.heapreplace(heap, (s, -i))
        keep.update((s, i, -nj) for s, nj in heapq.nlargest(partners, row))
    for j, heap in enumerate(by_target):
        keep.update((s, -ni, j) for s, ni in heap)
    return sorted(keep, key=lambda c: (-c[0], c[1], c[2]))


def greedy_pairs(candidates, own_s, own_t, budget_s, budget_t):
    """One-to-one greedy pair selection; stops when either side's budget is
    reached, because a pair needs both ends."""
    used_s, used_t, pick_s, pick_t = set(), set(), [], []
    for _, i, j in candidates:
        if len(pick_s) >= budget_s or len(pick_t) >= budget_t:
            break
        if i in used_s or j in used_t:
            continue
        used_s.add(i)
        used_t.add(j)
        pick_s.append(own_s[i])
        pick_t.append(own_t[j])
    return pick_s, pick_t


DOSE_P = (0.0, 0.25, 0.5, 0.75)
DOSE_Q = (0.25, 0.5, 0.75)


def dose_arms(cls, ev, case_id, src, tgt, s_orc, t_orc, replicates, seed):
    """Synthetic subsets of controlled precision and completeness
    (OVERLAP_FREE_SELECTOR.md, dose-response). Each arm draws from its own
    hashed stream, so no existing arm's sample moves."""
    s_in, t_in = set(s_orc), set(t_orc)
    s_out = [d for d in src if d not in s_in]
    t_out = [d for d in tgt if d not in t_in]
    k_s, k_t = len(s_orc), len(t_orc)
    for r in range(replicates):
        for mode in ("indep", "paired"):
            for kind, levels in (("noise", DOSE_P), ("clean", DOSE_Q)):
                for level in levels:
                    arm = f"{kind}-{'p' if kind == 'noise' else 'q'}{round(level * 100):02d}-{mode}"
                    digest = hashlib.sha256(f"{case_id}|{cls}|{arm}|{r}|{seed}".encode()).hexdigest()
                    rng = random.Random(int(digest[:16], 16))
                    o_s = rng.sample(s_orc, round(level * k_s))
                    if mode == "paired":
                        o_t = image_filter(cls, t_orc, o_s, ev["map_ts"])
                    else:
                        o_t = rng.sample(t_orc, round(level * k_t))
                    if kind == "clean":
                        yield f"{arm}-r{r}", o_s, o_t
                        continue
                    n_s = rng.sample(s_out, min(len(s_out), k_s - len(o_s)))
                    n_t = rng.sample(t_out, min(len(t_out), max(0, k_t - len(o_t))))
                    yield f"{arm}-r{r}", o_s + n_s, o_t + n_t


def positional(deps, cols):
    """A dependency subset with column names replaced by their positions.

    Per-side recoding renames every column, so subsets from two regimes are
    only comparable through positions.
    """
    at = {c: i for i, c in enumerate(cols)}

    def canon(x):
        if isinstance(x, str):
            return at.get(x, x)
        if isinstance(x, (frozenset, set)):
            return tuple(sorted((canon(y) for y in x), key=repr))
        if isinstance(x, (tuple, list)):
            return tuple(canon(y) for y in x)
        return x

    return json.dumps(sorted(repr(canon(d)) for d in deps))



# ---------------------------------------------------------------- controls
def _endpoints(cls, dep):
    """Columns a dependency touches, per class."""
    if cls == "FD":
        determinant, dependent = dep
        return set(determinant) | {dependent}
    if cls == "DC":
        return {c for c, _ in dep}
    return set(dep[:2]) if isinstance(dep, (tuple, list)) else set()


def _arity(cls, dep):
    """Determinant arity for FDs, predicate count for DCs; ODs are fixed."""
    if cls == "FD":
        return len(dep[0])
    if cls == "DC":
        return len(dep)
    return 2


def _degree(cls, deps, columns):
    counts = {c: 0 for c in columns}
    for d in deps:
        for c in _endpoints(cls, d):
            if c in counts:
                counts[c] += 1
    return counts


def matched_controls(cls, pool, oracle, columns, rng):
    """One subset per control, plus how well each matched. Never reads truth."""
    out, diag = {}, {}
    want = len(oracle)
    if want == 0 or not pool:
        return out, diag

    # ---- arity: match the multiset of arities exactly, or fail
    need = Counter(_arity(cls, d) for d in oracle)
    have = defaultdict(list)
    for d in pool:
        have[_arity(cls, d)].append(d)
    if cls == "OD":
        diag["arity"] = {"admissible": False, "why": "arity is fixed at 2 for OD"}
    elif all(len(have[a]) >= n for a, n in need.items()):
        picked = []
        for a, n in sorted(need.items()):
            picked += rng.sample(have[a], n)
        out["arity-matched"] = picked
        diag["arity"] = {"admissible": True,
                         "exact": Counter(_arity(cls, d) for d in picked) == need,
                         "size_ratio": len(picked) / want}
    else:
        short = {a: n - len(have[a]) for a, n in need.items() if len(have[a]) < n}
        diag["arity"] = {"admissible": False, "why": "pool too small per arity",
                         "shortfall": short}

    # ---- endpoint coverage: draw only from dependencies inside the oracle's
    # endpoint set, so the covered columns cannot exceed it
    covered = set()
    for d in oracle:
        covered |= _endpoints(cls, d)
    inside = [d for d in pool if _endpoints(cls, d) <= covered]
    if len(inside) >= want:
        picked = rng.sample(inside, want)
        got = set()
        for d in picked:
            got |= _endpoints(cls, d)
        out["endpoint-matched"] = picked
        diag["endpoint"] = {"admissible": True,
                            "exact": got == covered,
                            "jaccard": len(got & covered) / max(1, len(got | covered)),
                            "oracle_endpoints": len(covered),
                            "control_endpoints": len(got)}
    else:
        diag["endpoint"] = {"admissible": False,
                            "why": "too few dependencies inside the oracle's columns",
                            "available": len(inside), "needed": want}

    # ---- degree: greedy toward the oracle's per-column participation vector
    target = _degree(cls, oracle, columns)
    if len(pool) >= want:
        current = {c: 0 for c in columns}
        remaining = list(pool)
        picked = []
        for _ in range(want):
            best, best_cost = None, None
            for d in remaining:
                cost = 0
                ends = _endpoints(cls, d)
                for c in columns:
                    after = current[c] + (1 if c in ends else 0)
                    cost += abs(target[c] - after)
                if best_cost is None or cost < best_cost:
                    best, best_cost = d, cost
            if best is None:
                break
            picked.append(best)
            remaining.remove(best)
            for c in _endpoints(cls, best):
                if c in current:
                    current[c] += 1
        out["degree-matched"] = picked
        l1 = sum(abs(target[c] - current[c]) for c in columns)
        diag["degree"] = {"admissible": True, "l1": l1,
                          "l1_per_column": l1 / max(1, len(columns)),
                          "size_ratio": len(picked) / want}
    else:
        diag["degree"] = {"admissible": False, "why": "pool smaller than oracle"}

    return out, diag


def order_sets(cols, labels, seed):
    prepared = [to_ordered(c, False)[0] for c in cols]
    n = len(cols[0]) if cols else 0
    if n < 2:
        return np.zeros((len(labels), len(labels))), []
    rng = np.random.default_rng(seed)
    k = min(60000, max(1000, n * 20))
    pi, pj = rng.integers(0, n, k), rng.integers(0, n, k)
    keep = pi != pj
    m = order_matrix(prepared, pi[keep], pj[keep])
    return m, threshold_ods(m, labels, TAU)


def od_oracle(own, other, mapping):
    """Order dependencies whose image exists on the other side.

    Position is preserved -- an order dependency is not symmetric, so mapping a
    left operand onto a right one would be a different dependency. The
    correspondence may be 1:n, so every image is enumerated and any hit counts,
    which is the same convention the FD oracle uses.
    """
    have = set(other)
    keep = []
    for lhs, rhs in own:
        if any(c not in mapping for c in lhs) or any(c not in mapping for c in rhs):
            continue
        left = [()]
        for c in lhs:
            left = [p + (img,) for p in left for img in mapping[c]]
        right = [()]
        for c in rhs:
            right = [p + (img,) for p in right for img in mapping[c]]
        if any((l, r) in have for l in left for r in right):
            keep.append((lhs, rhs))
    return keep


def od_anchor_ranking(own, other, sim, order):
    """Pair order dependencies by anchor evidence, operand position preserved."""
    out = {}
    for lhs, rhs in own:
        best = 0.0
        for lt, rt in other:
            if len(lhs) != len(lt) or len(rhs) != len(rt):
                continue
            score = 1.0
            for a, b in zip(lhs, lt):
                score *= sim(a, b, order)
            for a, b in zip(rhs, rt):
                score *= sim(a, b, order)
            best = max(best, score)
        out[(lhs, rhs)] = best
    return out


def _images(f, other, mapping):
    """Signatures of `other`'s FDs that are an image of FD f under `mapping` (as oracle_signatures)."""
    have = {signature(list(x), a): (x, a) for x, a in other}
    det, dep = f
    out = set()
    if dep not in mapping or any(c not in mapping for c in det):
        return out
    combos = [[]]
    for c in det:
        combos = [p + [img] for p in combos for img in mapping[c]]
    for dep_img in mapping[dep]:
        for combo in combos:
            sig = signature(combo, dep_img)
            if len(set(combo)) == len(combo) and sig in have:
                out.add(have[sig])
    return out


def _oracle_units(ev, s_orc, t_orc):
    """Connected components of the image relation between transferable source and target FDs (ORACLE_VARIANTS.md)."""
    s_list, t_list = list(dict.fromkeys(s_orc)), list(dict.fromkeys(t_orc))
    parent = {("s", i): ("s", i) for i in range(len(s_list))}
    parent.update({("t", j): ("t", j) for j in range(len(t_list))})

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    t_index = {f: j for j, f in enumerate(t_list)}
    s_index = {f: i for i, f in enumerate(s_list)}
    for i, f in enumerate(s_list):
        for g in _images(f, t_list, ev["map_st"]):
            parent[find(("s", i))] = find(("t", t_index[g]))
    for j, g in enumerate(t_list):
        for f in _images(g, s_list, ev["map_ts"]):
            parent[find(("t", j))] = find(("s", s_index[f]))
    groups = defaultdict(lambda: ([], []))
    for i, f in enumerate(s_list):
        groups[find(("s", i))][0].append(f)
    for j, g in enumerate(t_list):
        groups[find(("t", j))][1].append(g)
    units = [u for _, u in sorted(groups.items(), key=lambda kv: kv[0])]
    return units


def oracle_norms(rec, ev, s_orc, t_orc, s_prof, t_prof, weight):
    """ORACLE_VARIANTS.md Part B: the full oracle under other normalisations and aggregations of the role channel."""
    from scipy.stats import rankdata
    base, truth = ev["combo_base"], ev["truth"]
    units = _oracle_units(ev, s_orc, t_orc)
    fs = [f for u in units for f in u[0]]
    ft = [g for u in units for g in u[1]]
    P_s, P_t = s_prof(fs), t_prof(ft)
    R = cosine_similarity(P_s, P_t)
    per_unit = [cosine_similarity(s_prof(u[0]), t_prof(u[1])) for u in units]
    lo, hi = float(base.min()), float(base.max())
    flat = R.ravel()
    ranks = ((rankdata(flat) - 1) / max(1, flat.size - 1)).reshape(R.shape) if flat.size > 1 else np.zeros_like(R)
    umax = np.max(per_unit, axis=0)
    arms = {"minmax": fuse(base, [R], weight),
            "raw": base if hi <= lo else (1.0 - weight) * base + weight * (lo + R * (hi - lo)),
            "rank": fuse(base, [ranks], weight),
            "dot": fuse(base, [P_s @ P_t.T], weight),
            "unit-channels": fuse(base, per_unit, weight),
            "unit-max": fuse(base, [umax], weight)}
    pairs = list(zip(*np.nonzero(truth)))

    def diag(M):
        top = tie = zero = 0
        for i, j in pairs:
            others = np.delete(M[i], j)
            best = others.max() if others.size else -np.inf
            top += M[i, j] > best + 1e-12
            tie += abs(M[i, j] - best) <= 1e-12 and M[i, j] > 0
            zero += M[i, j] <= 1e-12
        n = max(1, len(pairs))
        return {"role_min": float(M.min()), "role_max": float(M.max()), "role_top1": top / n,
                "role_tie": tie / n, "role_zero": zero / n}
    d_pool, d_umax = diag(R), diag(umax)
    out = []
    for name, m in arms.items():
        for r in evaluate(ScoreMatrix(scores=m, ground_truth=truth), policies=["hungarian", "mutual_top1"]):
            out.append({"id": rec["id"], "cohort": rec["cohort"], "arm": name, "extractor": r["extractor"],
                        "f1": r["f1"], "recall_at_gt": r.get("recall_at_gt_pessimistic", float("nan")),
                        "mrr": r.get("mrr_pessimistic", float("nan")), "units": len(units),
                        **{f"pooled_{k}": v for k, v in d_pool.items()},
                        **{f"unitmax_{k}": v for k, v in d_umax.items()}})
    return out


def oracle_variants(rec, ev, s_orc, t_orc, s_prof, t_prof, weight):
    """ORACLE_VARIANTS.md: oracle, targeted, targeted-uncertain, contrastive and utility subsets of the FD oracle."""
    base, truth, s_cols, t_cols = ev["combo_base"], ev["truth"], ev["s_cols"], ev["t_cols"]
    units = _oracle_units(ev, s_orc, t_orc)

    S = np.asarray(base, dtype=float)
    rmax, cmax = S.max(axis=1, keepdims=True), S.max(axis=0, keepdims=True)
    pairs = list(zip(*np.nonzero(truth)))
    prob_s, prob_t, unc_s, unc_t = set(), set(), set(), set()
    srt = np.sort(S, axis=1)
    margin = srt[:, -1] - (srt[:, -2] if S.shape[1] > 1 else 0.0)
    med = float(np.median(margin[[i for i, _ in pairs]])) if pairs else 0.0
    for i, j in pairs:
        if S[i, j] < rmax[i, 0] or S[i, j] < cmax[0, j]:
            prob_s.add(s_cols[i])
            prob_t.add(t_cols[j])
        if margin[i] < med:
            unc_s.add(s_cols[i])
            unc_t.add(t_cols[j])

    def cols_of(fds):
        return {c for det, dep in fds for c in (*det, dep)}

    def touches(u, ps, pt):
        return bool(cols_of(u[0]) & ps) or bool(cols_of(u[1]) & pt)

    def role(sel):
        fs = [f for u in sel for f in u[0]]
        ft = [g for u in sel for g in u[1]]
        return cosine_similarity(s_prof(fs), t_prof(ft))

    def score(sel):
        m = fuse(base, [role(sel)], weight)
        return {r["extractor"]: r for r in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                                                       policies=["hungarian", "mutual_top1"])}

    def f1m(sel):
        m = fuse(base, [role(sel)], weight) if sel else base
        return match_metrics(extract(m, "mutual_top1"), truth)["f1"]



    def contrast(u):
        R = role([u])
        touched_s, touched_t = cols_of(u[0]), cols_of(u[1])
        vals = []
        for i, j in pairs:
            if s_cols[i] in touched_s or t_cols[j] in touched_t:
                others = np.delete(R[i], j)
                vals.append(R[i, j] - (others.max() if others.size else 0.0))
        return float(np.mean(vals)) if vals else 0.0

    targeted = [u for u in units if touches(u, prob_s, prob_t)]
    targeted_u = [u for u in units if touches(u, prob_s | unc_s, prob_t | unc_t)]
    contrastive = [u for u in units if contrast(u) > 0]
    def greedy(pool):
        chosen, current = [], f1m([])
        remaining = list(range(len(pool)))
        while remaining and len(chosen) < 30:
            best, best_k = None, None
            for k in remaining:
                v = f1m([pool[x] for x in chosen] + [pool[k]])
                if best is None or v > best:
                    best, best_k = v, k
            if best is None or best <= current + 1e-12:
                break
            chosen.append(best_k)
            remaining.remove(best_k)
            current = best
        return [pool[k] for k in chosen]

    utility = greedy(units)
    # Amendment A1: the same greedy search over random (source FD, target FD) pairs from the full pools, as many
    # pseudo-units as there are units; seeded per scenario. Separates FD usefulness from in-sample fitting.
    all_s, all_t = ev["fd"]
    null_rng = random.Random(int(hashlib.sha256(f"{rec['id']}|oracle-null".encode()).hexdigest()[:16], 16))
    null_pool = [([null_rng.choice(all_s)], [null_rng.choice(all_t)]) for _ in units]
    utility_null = greedy(null_pool)
    # Amendment A1: contrast on the fused matrix. A unit is kept if fusing it alone raises the mean margin of the
    # reference pairs (true score minus best competitor in its row) above fusing an uninformative channel.
    def mean_margin(m):
        vals = []
        for i, j in pairs:
            others = np.delete(m[i], j)
            vals.append(m[i, j] - (others.max() if others.size else 0.0))
        return float(np.mean(vals)) if vals else 0.0
    flat = mean_margin(fuse(base, [np.zeros_like(S)], weight))
    contrastive_base = [u for u in units if mean_margin(fuse(base, [role([u])], weight)) > flat + 1e-12]
    t_ids = {id(u) for u in targeted}
    c_ids = {id(u) for u in contrastive}

    out = []
    for name, sel in (("base", None), ("oracle", units), ("targeted", targeted), ("targeted-uncertain", targeted_u),
                      ("contrastive", contrastive), ("utility", utility),
                      ("contrastive-base", contrastive_base), ("utility-null", utility_null)):
        if sel is None:
            res = {r["extractor"]: r for r in evaluate(ScoreMatrix(scores=base, ground_truth=truth),
                                                          policies=["hungarian", "mutual_top1"])}
        else:
            res = score(sel)
        meta = {"units": len(units) if sel is None else len(sel), "source_fds": sum(len(u[0]) for u in (sel or [])),
                "target_fds": sum(len(u[1]) for u in (sel or [])),
                "util_in_targeted": sum(id(u) in t_ids for u in utility) if name == "utility" else "",
                "util_in_contrastive": sum(id(u) in c_ids for u in utility) if name == "utility" else "",
                "all_targeted": len(targeted), "all_contrastive": len(contrastive), "all_units": len(units),
                "mean_arity": (float(np.mean([len(f[0]) for u in sel for f in u[0]])) if sel and any(u[0] for u in sel) else "")}
        for ext, r in res.items():
            out.append({"id": rec["id"], "cohort": rec["cohort"], "variant": name, "extractor": ext, "f1": r["f1"],
                        "recall_at_gt": r.get("recall_at_gt_pessimistic", float("nan")),
                        "mrr": r.get("mrr_pessimistic", float("nan")), **meta})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="reports/valentine_full_382_20260903")
    ap.add_argument("--data", default="generated/benchmark-full")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", default="reports/transferability_table.csv")
    ap.add_argument("--tex", default="../latex/table_transferability.tex")
    ap.add_argument("--combinations", default=None,
                    help="also write the FD x OD x DC factorial to this CSV")
    ap.add_argument("--base-name", default=None,
                    help="fuse the factorial onto this cached matrix "
                         "(e.g. sf.csv) from --base-dir instead of the seed")
    ap.add_argument("--base-dir", default="reports/valentine_full_382_20260903",
                    help="where --base-name is read from")
    ap.add_argument("--matched-controls", default=None,
                    help="write the size/arity/degree/endpoint-matched "
                         "oracle controls to this CSV (see "
                         "MATCHED_ORACLE_CONTROLS.md)")
    ap.add_argument("--budgets", default=None,
                    help="use fixed budgets instead of fitting them, e.g. "
                         "'FD=0.282,DC=0.576,OD=0.423'. Needed on cohorts with "
                         "no exploration split, where fitting would be in-sample.")
    ap.add_argument("--labels", choices=("seed", "tables"), default="seed",
                    help="column identities from the seed matrix (historical default) "
                         "or from the scenario's own table headers (FINDINGS 113)")
    ap.add_argument("--strict-join", action="store_true",
                    help="fail if a side has FDs or DCs and none of them join to its columns")
    ap.add_argument("--join-report", default=None,
                    help="write per-scenario FD/DC join counts to this CSV")
    ap.add_argument("--selectors", default=None,
                    help="extra factorial variants that run the anchor ranking on an "
                         "overlap-free column similarity, e.g. 'pmf:raw,kn:mutual3'. "
                         "Sources: " + ", ".join(SELECTOR_SOURCES) + "; transforms: "
                         + ", ".join(SELECTOR_TRANSFORMS) + ". Read from the regime "
                         "directory of --base-name (OVERLAP_FREE_SELECTOR.md).")
    ap.add_argument("--paired-selectors", default=None,
                    help="paired top-k variants as source:transform:scale, e.g. "
                         "'pmf:raw:1.0,anchor:none:0.5'; 'anchor' uses the value-overlap "
                         "anchor matrix. Only FD+OD+DC is scored for them.")
    ap.add_argument("--size-matched", action="store_true",
                    help="with --paired-selectors: add per-side controls cut to each "
                         "paired variant's exact per-side pick counts")
    ap.add_argument("--any-classes", action="store_true",
                    help="also write combination rows for scenarios carrying one or two "
                         "classes, fusing the classes present (follow-up 5)")
    ap.add_argument("--selection-report", default=None,
                    help="write the FD/DC subsets each selector picked, as column "
                         "positions, to this CSV (cross-regime invariance check)")
    ap.add_argument("--dose-response", type=int, default=0, metavar="REPLICATES",
                    help="add the synthetic precision/completeness arms with this many "
                         "replicates; only FD+OD+DC is scored for them")
    ap.add_argument("--kn-host", default=None, metavar="PATH",
                    help="also run every FD arm's subsets inside the Kang-Naughton graph "
                         "match (restricted and emphasized edges) and write the scores "
                         "to PATH (KN_HOST_SELECTION.md); other outputs are unchanged")
    ap.add_argument("--oracle-variants", default=None, metavar="PATH",
                    help="score refined FD oracles (targeted, contrastive, utility) on the --base-name matrix "
                         "and write them to PATH (ORACLE_VARIANTS.md); other outputs are unchanged")
    ap.add_argument("--oracle-norms", default=None, metavar="PATH",
                    help="score the full FD oracle under other role normalisations (ORACLE_VARIANTS.md Part B)")
    ap.add_argument("--paired-aggregation", default=None, metavar="PATH",
                    help="per-pair FD roles for the paired selectors (PAIRED_AGGREGATION.md) -> PATH")
    ap.add_argument("--kn-full-rows", action="store_true",
                    help="with --kn-host: read full tables for the host's mutual information "
                         "instead of read_columns' first 5,000 rows (KN_HOST_SELECTION.md A2)")
    ap.add_argument("--export-subsets", default=None, metavar="PATH",
                    help="write every FD arm's (source, target) subsets as JSON lines to PATH "
                         "(PRISMA_HOST_SELECTION.md); other outputs are unchanged")
    ap.add_argument("--score-cohort", default=None,
                    help="score only scenarios of this cohort, e.g. inspected44; "
                         "budgets are fitted exactly as without it")
    args = ap.parse_args()

    selectors = []
    if args.selectors:
        if not args.base_name:
            ap.error("--selectors needs --base-name, whose directory names the regime")
        for part in args.selectors.split(","):
            source, _, transform = part.partition(":")
            if source not in SELECTOR_SOURCES or transform not in SELECTOR_TRANSFORMS:
                ap.error(f"unknown selector {part!r}")
            selectors.append((f"{source}-{transform}", source, transform))
    paired = []
    if args.paired_selectors:
        if not args.base_name:
            ap.error("--paired-selectors needs --base-name, whose directory names the regime")
        for part in args.paired_selectors.split(","):
            source, transform, scale = part.split(":")
            if source == "anchor":
                name = f"paired-anchor-b{round(float(scale) * 100)}"
            elif source in SELECTOR_SOURCES and transform in SELECTOR_TRANSFORMS:
                name = f"paired-{source}-{transform}-b{round(float(scale) * 100)}"
            else:
                ap.error(f"unknown paired selector {part!r}")
            paired.append((name, source, transform, float(scale)))
    selection_rows = []
    kn_rows = []
    ov_rows = []
    pa_rows = []
    on_rows = []
    subset_rows = []
    if args.kn_host:
        from kang_naughton import fd_edge_mask, kn_graph_scores, kn_graph_scores_masked, mi_matrix

    rng = random.Random(args.seed)
    join_rows = []
    cases_dir, data = Path(args.cases), Path(args.data)
    index = list(csv.DictReader(open(cases_dir / "cases.csv", encoding="utf-8")))
    if args.limit:
        index = index[:args.limit]

    shares = defaultdict(list)
    rows, skipped = [], defaultdict(int)
    combo_rows = []
    control_rows, control_metrics = [], []

    def prepare(rec):
        parts = rec["id"].split("/")
        # FINDINGS 67: the legacy rule drops the Magellan and Wikidata
        # families. Default is unchanged; SMX_RESOLVER=index corrects it.
        scenario = resolve(data, rec["id"])
        digest = hashlib.sha256(rec["id"].encode()).hexdigest()[:16]
        seed_path = cases_dir / "cases" / digest / "seed.csv"
        gt_f = sorted((scenario / "ground_truth").glob("*.csv"))
        if not scenario.is_dir() or not seed_path.is_file() or not gt_f:
            return None
        base, row_labels, col_labels = read_matrix(seed_path)
        gt, _, _ = read_matrix(gt_f[0])
        if base.shape != gt.shape:
            return None
        # Only the factorial uses this; a case without it still contributes to
        # budget fitting and to the per-class table. SF was run on the
        # confirmation cohort only, so requiring it here would empty the
        # exploration cohort the budgets are fitted on.
        combo_base = base
        if args.base_name:
            alt = Path(args.base_dir) / "cases" / digest / args.base_name
            combo_base = None
            if alt.is_file():
                candidate = read_matrix(alt)[0]
                if candidate.shape == gt.shape:
                    combo_base = candidate
        src = sorted((scenario / "source").glob("*.csv"))
        tgt = sorted((scenario / "target").glob("*.csv"))
        if not src or not tgt:
            return None
        s_names, s_raw = read_columns(src[0])
        t_names, t_raw = read_columns(tgt[0])
        if len(s_names) != gt.shape[0] or len(t_names) != gt.shape[1]:
            return None
        if args.labels == "tables":
            # Column identities from the scenario's own headers, which are the
            # label space its profiles use. The seed matrix is positionally
            # aligned (shapes checked above). Needed when the seeds come from a
            # different representation than --data, e.g. plain seeds against
            # the audited MD5 corpus, where the seed's plain labels never match
            # the hashed profile labels and every dependency silently drops out
            # (FINDINGS 113).
            s_cols = [bare(c) for c in s_names]
            t_cols = [bare(c) for c in t_names]
        else:
            s_cols = [bare(c) for c in row_labels]
            t_cols = [bare(c) for c in col_labels]
        s_vals, s_fds = load_side(scenario, "source")
        t_vals, t_fds = load_side(scenario, "target")
        _, s_dcs = load_dcs(scenario, "source")
        _, t_dcs = load_dcs(scenario, "target")
        join = {}
        for side, cols, fds, dcs in (("source", s_cols, s_fds, s_dcs), ("target", t_cols, t_fds, t_dcs)):
            known = set(cols)
            fd_ok = sum(1 for det, dep in fds if dep in known and all(c in known for c in det))
            dc_ok = sum(1 for dc in dcs if all(c in known for c, _ in dc))
            join[side] = {"fds": len(fds), "fds_joined": fd_ok, "dcs": len(dcs), "dcs_joined": dc_ok}
            if args.strict_join and ((fds and not fd_ok) or (dcs and not dc_ok)):
                raise RuntimeError(f"{rec['id']} {side}: dependencies do not join to the columns "
                                   f"({fd_ok}/{len(fds)} FDs, {dc_ok}/{len(dcs)} DCs)")
        join_rows.append({"id": rec["id"], **{f"{s}_{k}": v for s, d in join.items() for k, v in d.items()}})
        anchors = anchor_matrix(s_vals, t_vals, s_cols, t_cols)
        _, s_ods = order_sets(s_raw, s_cols, args.seed)
        _, t_ods = order_sets(t_raw, t_cols, args.seed)
        regime_dir = Path(args.base_dir) / "cases" / digest / Path(args.base_name or ".").parent
        selector_sims = {}
        needed = {(s, t) for _, s, t in selectors} | {(s, t) for _, s, t, _ in paired
                                                       if s != "anchor"}
        for source, transform in sorted(needed):
            path = regime_dir / (Path(args.base_name).name if source == "seed"
                                 else SELECTOR_SOURCES[source])
            selector_sims[f"{source}-{transform}"] = (
                selector_matrix(path, s_cols, t_cols, transform) if path.is_file() else None)
        kn_cols = None
        if args.kn_host:
            s_kn, t_kn = s_raw, t_raw
            if args.kn_full_rows:
                s_kn, t_kn = read_columns(src[0], max_rows=10**9)[1], read_columns(tgt[0], max_rows=10**9)[1]
            s_by, t_by = dict(zip((bare(c) for c in s_names), s_kn)), dict(zip((bare(c) for c in t_names), t_kn))
            if all(c in s_by for c in s_cols) and all(c in t_by for c in t_cols):
                kn_cols = ([s_by[c] for c in s_cols], [t_by[c] for c in t_cols])
        return {"base": base, "combo_base": combo_base, "kn_cols": kn_cols,
                "selectors": selector_sims,
                "truth": gt > 0.5, "s_cols": s_cols, "t_cols": t_cols,
                "anchors": anchors, "map_st": correspondence(gt, s_cols, t_cols),
                "map_ts": correspondence(gt.T, t_cols, s_cols),
                "fd": (s_fds, t_fds), "dc": (s_dcs, t_dcs), "od": (s_ods, t_ods)}

    def class_arms(ev, cls, budget, similarity=None):
        """Both sides of a class: sets, rankings, oracle subsets, profilers.

        Filtering has to apply to *both* schemas, which is the protocol
        FINDINGS 15 established and 30 and 31 used. An arm that thins only the
        source and leaves every target dependency in place measures something
        much weaker -- it understated the FD oracle by a factor of four when
        this was first written the other way.

        `similarity` replaces the anchor matrix in the rankings only; the
        oracle subsets never read it.
        """
        pairing = ev["anchors"] if similarity is None else similarity
        if cls == "FD":
            s, t = ev["fd"]
            if not s or not t:
                return None
            sim = fd_similarity(pairing, ev["s_cols"], ev["t_cols"])
            key = lambda fd: signature(list(fd[0]), fd[1])  # noqa: E731
            s_sig = oracle_signatures(s, t, ev["map_st"])
            t_sig = oracle_signatures(t, s, ev["map_ts"])
            s_orc = [f for f in s if key(f) in s_sig]
            t_orc = [f for f in t if key(f) in t_sig]
            raw_s = relevance(s, t, {}, {}, sim, "st", "anchor")
            raw_t = relevance(t, s, {}, {}, sim, "ts", "anchor")
            s_rank = {f: raw_s.get(key(f), 0.0) for f in s}
            t_rank = {f: raw_t.get(key(f), 0.0) for f in t}
            s_prof = lambda xs: fd_profile(xs, ev["s_cols"])   # noqa: E731
            t_prof = lambda xs: fd_profile(xs, ev["t_cols"])   # noqa: E731
        elif cls == "DC":
            s, t = ev["dc"]
            if not s or not t:
                return None
            sim = build_similarity(pairing, ev["s_cols"], ev["t_cols"])
            s_rank = dc_anchor_ranking(s, t, sim, "st")
            t_rank = dc_anchor_ranking(t, s, sim, "ts")
            s_orc = oracle_subset(s, t, ev["map_st"])
            t_orc = oracle_subset(t, s, ev["map_ts"])
            shape = lambda ds: [([c for c, _ in d], len(d),  # noqa: E731
                                 sum(1 for _, op in d if op == "=")) for d in ds]
            s_prof = lambda xs: dc_profile(shape(xs), ev["s_cols"])  # noqa: E731
            t_prof = lambda xs: dc_profile(shape(xs), ev["t_cols"])  # noqa: E731
        else:
            s, t = ev["od"]
            if not s or not t:
                return None
            sim = build_similarity(pairing, ev["s_cols"], ev["t_cols"])
            s_rank = od_anchor_ranking(s, t, sim, "st")
            t_rank = od_anchor_ranking(t, s, sim, "ts")
            s_orc = od_oracle(s, t, ev["map_st"])
            t_orc = od_oracle(t, s, ev["map_ts"])
            s_prof = lambda xs: od_profile(xs, ev["s_cols"])   # noqa: E731
            t_prof = lambda xs: od_profile(xs, ev["t_cols"])   # noqa: E731
        return s, t, s_rank, t_rank, s_orc, t_orc, s_prof, t_prof

    # ---- budgets fixed on the exploration cohort only
    for rec in [r for r in index if r["cohort"] != "remaining338"]:
        ev = prepare(rec)
        if ev is None:
            continue
        for cls in ("FD", "OD", "DC"):
            got = class_arms(ev, cls, None)
            if got is None:
                continue
            src, _, _, _, s_orc, _, _, _ = got
            shares[cls].append(len(s_orc) / max(1, len(src)))
    budgets = {c: (mean(v) if v else 0.25) for c, v in shares.items()}
    if args.budgets:
        supplied = dict(part.split("=") for part in args.budgets.split(","))
        budgets = {c: float(supplied.get(c, budgets.get(c, 0.25)))
                   for c in ("FD", "OD", "DC")}
        print("budgets supplied, not fitted: "
              + "  ".join(f"{c} {b:.3f}" for c, b in sorted(budgets.items())))
    print("budgets fixed on the exploration cohort: "
          + "  ".join(f"{c} {b:.3f}" for c, b in sorted(budgets.items())) + "\n")

    # The per-class table is confirmation-only. The factorial, when asked for,
    # is computed for every scenario so it can be reported per family; rows for
    # the per-class table are still gated on the cohort below.
    scored_index = index if args.combinations else [
        r for r in index if r["cohort"] == "remaining338"]
    if args.score_cohort:
        scored_index = [r for r in scored_index if r["cohort"] == args.score_cohort]
    for rec in scored_index:
        confirmation = rec["cohort"] == "remaining338"
        ev = prepare(rec)
        if ev is None:
            if confirmation:
                skipped["no evidence"] += 1
            continue
        roles = {}
        pair_roles = {}
        for cls in ("FD", "OD", "DC"):
            got = class_arms(ev, cls, budgets.get(cls))
            if got is None:
                if confirmation:
                    skipped[f"{cls} absent"] += 1
                continue
            src, tgt, s_rank, t_rank, s_orc, t_orc, s_prof, t_prof = got

            def top(rank, pool):
                budget = max(1, round(budgets[cls] * len(pool)))
                return sorted(rank, key=lambda x: -rank[x])[:budget]

            # Both schemas are filtered, as in FINDINGS 15, 30 and 31. Thinning
            # only the source leaves every target dependency in place and
            # measures a much weaker effect.
            arms = {
                "all": (src, tgt),
                "oracle": (s_orc, t_orc),
                "anchor": (top(s_rank, src), top(t_rank, tgt)),
                "random": (rng.sample(src, min(len(s_orc), len(src))),
                           rng.sample(tgt, min(len(t_orc), len(tgt)))),
            }
            picked = ({"anchor": arms["anchor"], "oracle": arms["oracle"]}
                      if selectors or args.dose_response or paired else {})
            for name, sub_s, sub_t in dose_arms(cls, ev, rec["id"], src, tgt, s_orc, t_orc,
                                                args.dose_response, args.seed):
                roles[(cls, name)] = cosine_similarity(s_prof(sub_s), t_prof(sub_t))
                picked[name] = (sub_s, sub_t)
            for name, _, _ in selectors:
                matrix = ev["selectors"].get(name)
                if matrix is None:
                    continue
                _, _, sel_s, sel_t, _, _, _, _ = class_arms(ev, cls, budgets.get(cls), matrix)
                sub_s, sub_t = top(sel_s, src), top(sel_t, tgt)
                roles[(cls, name)] = cosine_similarity(s_prof(sub_s), t_prof(sub_t))
                picked[name] = (sub_s, sub_t)
            cand_cache, rank_cache = {}, {}
            for name, source, transform, scale in paired:
                matrix = (ev["anchors"] if source == "anchor"
                          else ev["selectors"].get(f"{source}-{transform}"))
                if matrix is None:
                    continue
                if (source, transform) not in cand_cache:
                    cand_cache[(source, transform)] = pair_candidates(
                        cls, src, tgt, matrix, ev["s_cols"], ev["t_cols"])
                sub_s, sub_t = greedy_pairs(
                    cand_cache[(source, transform)], src, tgt,
                    max(1, round(scale * budgets[cls] * len(src))),
                    max(1, round(scale * budgets[cls] * len(tgt))))
                roles[(cls, name)] = cosine_similarity(s_prof(sub_s), t_prof(sub_t))
                picked[name] = (sub_s, sub_t)
                if args.paired_aggregation and cls == "FD":
                    # PAIRED_AGGREGATION.md: one role per selected pair, and the same FDs with the pairing
                    # broken by a separately seeded permutation (no shared RNG stream is touched).
                    shuffled = list(sub_t)
                    random.Random(int(hashlib.sha256(f"{rec['id']}|{name}|pair-shuffle".encode())
                                      .hexdigest()[:16], 16)).shuffle(shuffled)
                    pair_roles[name] = (
                        [cosine_similarity(s_prof([a]), t_prof([b])) for a, b in zip(sub_s, sub_t)],
                        [cosine_similarity(s_prof([a]), t_prof([b])) for a, b in zip(sub_s, shuffled)])
                if args.size_matched:
                    # Per-side ranking over the same similarity, cut to the
                    # paired variant's exact per-side counts (follow-up 3).
                    if source == "anchor":
                        ranks = (s_rank, t_rank)
                    else:
                        if (source, transform) not in rank_cache:
                            rank_cache[(source, transform)] = class_arms(
                                ev, cls, budgets.get(cls), matrix)[2:4]
                        ranks = rank_cache[(source, transform)]
                    ctl = "sizematch-" + name[len("paired-"):]
                    ctl_s = sorted(ranks[0], key=lambda x: -ranks[0][x])[:len(sub_s)]
                    ctl_t = sorted(ranks[1], key=lambda x: -ranks[1][x])[:len(sub_t)]
                    roles[(cls, ctl)] = cosine_similarity(s_prof(ctl_s), t_prof(ctl_t))
                    picked[ctl] = (ctl_s, ctl_t)
            if args.selection_report:
                # Oracle hits against the chance rate |oracle| / |pool| say
                # whether a ranking finds transferable dependencies at all.
                # Paired: selected dependencies whose image is selected on the
                # other side too -- the property the oracle has by construction.
                orc_s, orc_t = set(s_orc), set(t_orc)
                for name, (sub_s, sub_t) in picked.items():
                    pair_s, pair_t = paired_subsets(cls, ev, sub_s, sub_t)
                    selection_rows.append({
                        "id": rec["id"], "cohort": rec["cohort"], "cls": cls,
                        "selector": name,
                        "source_picked": len(sub_s),
                        "source_hits": sum(d in orc_s for d in sub_s),
                        "source_oracle": len(orc_s), "source_pool": len(src),
                        "target_picked": len(sub_t),
                        "target_hits": sum(d in orc_t for d in sub_t),
                        "target_oracle": len(orc_t), "target_pool": len(tgt),
                        "source_paired": pair_s, "target_paired": pair_t,
                        # Positions (hashed) only for real selectors; dose arms
                        # are random draws.
                        "source_hash": "" if name[:6] in ("noise-", "clean-") else position_hash(sub_s, ev["s_cols"]),
                        "target_hash": "" if name[:6] in ("noise-", "clean-") else position_hash(sub_t, ev["t_cols"])})
            if args.oracle_variants and cls == "FD" and ev["combo_base"] is not None and s_orc and t_orc:
                ov_rows.extend(oracle_variants(rec, ev, s_orc, t_orc, s_prof, t_prof, args.weight))
            if args.oracle_norms and cls == "FD" and ev["combo_base"] is not None and s_orc and t_orc:
                on_rows.extend(oracle_norms(rec, ev, s_orc, t_orc, s_prof, t_prof, args.weight))
            if args.export_subsets and cls == "FD":
                for name, (sub_s, sub_t) in {**arms, **picked}.items():
                    subset_rows.append(json.dumps({
                        "id": rec["id"], "cohort": rec["cohort"], "arm": name,
                        "source": [[sorted(det), dep] for det, dep in sub_s],
                        "target": [[sorted(det), dep] for det, dep in sub_t],
                        "source_all": len(src), "target_all": len(tgt)}, ensure_ascii=False))
            if args.kn_host and cls == "FD" and ev["kn_cols"] is not None:
                # KN_HOST_SELECTION.md: the same subsets, inside the graph match.
                # Reads no random draws, so every other arm is unchanged.
                if "kn_w" not in ev:
                    ev["kn_w"] = (mi_matrix(ev["kn_cols"][0]), mi_matrix(ev["kn_cols"][1]))
                    ev["kn_dense"] = kn_graph_scores(*ev["kn_w"])
                w1, w2 = ev["kn_w"]
                host = [("none", "dense", ev["kn_dense"], 0, 0, 0, 0)]
                for name, (sub_s, sub_t) in {**arms, **picked}.items():
                    e1, e2 = fd_edge_mask(sub_s, ev["s_cols"]), fd_edge_mask(sub_t, ev["t_cols"])
                    for mode in ("restrict", "emphasis"):
                        host.append((name, mode, kn_graph_scores_masked(w1, w2, e1, e2, mode),
                                     len(sub_s), len(sub_t), int(e1.sum() // 2), int(e2.sum() // 2)))
                for name, mode, m, ns, nt, es, et in host:
                    for res in evaluate(ScoreMatrix(scores=m, ground_truth=ev["truth"]),
                                        policies=["hungarian", "mutual_top1"]):
                        kn_rows.append({"id": rec["id"], "cohort": rec["cohort"], "arm": name, "mode": mode,
                                        "extractor": res["extractor"], "f1": res["f1"],
                                        "recall_at_gt": res.get("recall_at_gt_pessimistic", float("nan")),
                                        "mrr": res.get("mrr_pessimistic", float("nan")),
                                        "source_picked": ns, "target_picked": nt,
                                        "source_edges": es, "target_edges": et})
            evaluated = [("none", ev["base"])]
            for arm, (sub_s, sub_t) in arms.items():
                role = cosine_similarity(s_prof(sub_s), t_prof(sub_t))
                roles[(cls, arm)] = role
                evaluated.append((arm, scale_align(ev["base"], role, args.weight)))
            if args.matched_controls and confirmation:
                # A separate stream: drawing from `rng` here would move the
                # `random` arm's samples, which it did -- 979 rows changed.
                control_rng = random.Random(
                    int(hashlib.sha256(
                        f"{rec['id']}|{cls}|{args.seed}".encode()).hexdigest()[:16], 16))
                s_ctl, s_diag = matched_controls(cls, src, s_orc, ev["s_cols"],
                                                 control_rng)
                t_ctl, t_diag = matched_controls(cls, tgt, t_orc, ev["t_cols"],
                                                 control_rng)
                for name in ("arity-matched", "degree-matched", "endpoint-matched"):
                    key = name.split("-")[0]
                    admissible = (s_diag.get(key, {}).get("admissible") and
                                  t_diag.get(key, {}).get("admissible"))
                    control_rows.append({
                        "id": rec["id"], "cls": cls, "control": name,
                        "admissible": bool(admissible),
                        "source_note": json.dumps(s_diag.get(key, {})),
                        "target_note": json.dumps(t_diag.get(key, {})),
                        "oracle_size": len(s_orc), "pool_size": len(src),
                    })
                    if not admissible:
                        continue
                    role = cosine_similarity(s_prof(s_ctl[name]), t_prof(t_ctl[name]))
                    m = scale_align(ev["base"], role, args.weight)
                    for res in evaluate(ScoreMatrix(scores=m, ground_truth=ev["truth"]),
                                        policies=["hungarian", "mutual_top1"]):
                        control_metrics.append({
                            "id": rec["id"], "cls": cls, "arm": name,
                            "extractor": res["extractor"], "f1": res["f1"],
                            "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                    float("nan")),
                            "mrr": res.get("mrr_pessimistic", float("nan"))})

            for arm, m in evaluated:
                if not confirmation:
                    break
                for res in evaluate(ScoreMatrix(scores=m, ground_truth=ev["truth"]),
                                    policies=["hungarian", "mutual_top1"]):
                    rows.append({"id": rec["id"], "cls": cls, "arm": arm,
                                 "extractor": res["extractor"], "f1": res["f1"],
                                 "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                         float("nan")),
                                 "mrr": res.get("mrr_pessimistic", float("nan"))})

        if args.combinations:
            # Only scenarios carrying all three classes, so every subset is
            # scored on identical data and the cells are comparable. Singleton
            # subsets are fused through `fuse` like the rest, which unit-
            # normalises first, so they are not identical to the per-class rows
            # above -- the factorial is internally consistent, not a superset.
            dose = sorted({v for _, v in roles if v[:6] in ("noise-", "clean-")
                           or v.startswith(("paired-", "sizematch-"))})
            for variant in ("oracle", "anchor", *(n for n, _, _ in selectors), *dose):
                have = [c for c in ("FD", "OD", "DC") if (c, variant) in roles]
                if ev["combo_base"] is None or not have:
                    continue
                if len(have) < 3:
                    # Follow-up 5: scenarios with one or two classes fuse the
                    # classes they have; the oracle also carries the base row.
                    if not args.any_classes:
                        continue
                    subsets = ([()] if variant == "oracle" else []) + [tuple(have)]
                else:
                    subsets = ([("FD", "OD", "DC")] if variant in dose else
                               [s for r in range(0, 4)
                                for s in itertools.combinations(("FD", "OD", "DC"), r)])
                for subset in subsets:
                    m = fuse(ev["combo_base"],
                             [roles[(c, variant)] for c in subset], args.weight)
                    for res in evaluate(ScoreMatrix(scores=m,
                                                    ground_truth=ev["truth"]),
                                        policies=["hungarian", "mutual_top1"]):
                        combo_rows.append({
                            "id": rec["id"], "cohort": rec["cohort"],
                            "variant": variant,
                            "subset": "+".join(subset) if subset else "none",
                            "k": len(subset), "extractor": res["extractor"],
                            "f1": res["f1"],
                            "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                    float("nan")),
                            "mrr": res.get("mrr_pessimistic", float("nan"))})
        if args.paired_aggregation and ev["combo_base"] is not None:
            cb = ev["combo_base"]
            for variant, (per_pair, per_shuf) in pair_roles.items():
                if ("FD", variant) not in roles:
                    continue
                zero = np.zeros_like(roles[("FD", variant)])
                pmax = np.max(per_pair, axis=0) if per_pair else zero
                smax = np.max(per_shuf, axis=0) if per_shuf else zero
                fd = {"pooled": roles[("FD", variant)], "pairmax": pmax, "pairmax-shuffled": smax}
                arms = [("base", "none", cb)]
                arms += [(a, "FD", fuse(cb, [c], args.weight)) for a, c in fd.items()]
                arms.append(("pairchannels", "FD", fuse(cb, per_pair or [zero], args.weight)))
                if all((c, variant) in roles for c in ("OD", "DC")):
                    rest = [roles[("OD", variant)], roles[("DC", variant)]]
                    arms += [(a, "FD+OD+DC", fuse(cb, [c] + rest, args.weight)) for a, c in fd.items()]
                for arm, config, m in arms:
                    for res in evaluate(ScoreMatrix(scores=m, ground_truth=ev["truth"]),
                                        policies=["hungarian", "mutual_top1"]):
                        pa_rows.append({"id": rec["id"], "cohort": rec["cohort"], "variant": variant, "arm": arm,
                                        "config": config, "extractor": res["extractor"], "f1": res["f1"],
                                        "recall_at_gt": res.get("recall_at_gt_pessimistic", float("nan")),
                                        "mrr": res.get("mrr_pessimistic", float("nan")),
                                        "pairs": len(per_pair)})

    if args.paired_aggregation and pa_rows:
        with open(args.paired_aggregation, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pa_rows[0]))
            w.writeheader()
            w.writerows(pa_rows)
        print(f"wrote {len(pa_rows)} paired-aggregation rows -> {args.paired_aggregation}")
    if args.matched_controls and control_rows:
        base = Path(args.matched_controls)
        base.parent.mkdir(parents=True, exist_ok=True)
        with base.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(control_rows[0]))
            w.writeheader()
            w.writerows(control_rows)
        metrics_path = base.with_name(base.stem + "_metrics.csv")
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(control_metrics[0]))
            w.writeheader()
            w.writerows(control_metrics)
        print(f"wrote {len(control_rows)} feasibility rows -> {base}")
        print(f"wrote {len(control_metrics)} control metric rows -> {metrics_path}")

    if args.oracle_norms and on_rows:
        with open(args.oracle_norms, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(on_rows[0]))
            w.writeheader()
            w.writerows(on_rows)
        print(f"wrote {len(on_rows)} oracle-norm rows -> {args.oracle_norms}")
    if args.oracle_variants and ov_rows:
        with open(args.oracle_variants, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ov_rows[0]))
            w.writeheader()
            w.writerows(ov_rows)
        print(f"wrote {len(ov_rows)} oracle-variant rows -> {args.oracle_variants}")
    if args.export_subsets and subset_rows:
        Path(args.export_subsets).write_text("\n".join(subset_rows) + "\n", encoding="utf-8")
        print(f"wrote {len(subset_rows)} FD subsets -> {args.export_subsets}")
    if args.kn_host and kn_rows:
        with open(args.kn_host, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kn_rows[0]))
            w.writeheader()
            w.writerows(kn_rows)
        print(f"wrote {len(kn_rows)} Kang-Naughton host rows -> {args.kn_host}")
    if args.selection_report and selection_rows:
        with open(args.selection_report, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(selection_rows[0]))
            w.writeheader()
            w.writerows(selection_rows)
        print(f"wrote {len(selection_rows)} selections -> {args.selection_report}")
    if args.join_report and join_rows:
        with open(args.join_report, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(join_rows[0]))
            w.writeheader()
            w.writerows(join_rows)
        print(f"wrote join counts for {len(join_rows)} scenarios -> {args.join_report}")
    if args.combinations and combo_rows:
        cpath = Path(args.combinations)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with cpath.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(combo_rows[0]))
            w.writeheader()
            w.writerows(combo_rows)
        n = len({r["id"] for r in combo_rows})
        print(f"wrote {len(combo_rows)} combination rows for {n} scenarios "
              f"carrying {'at least one class' if args.any_classes else 'all three classes'}"
              f" -> {cpath}")

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = ["none", "all", "random", "anchor", "oracle"]
    print(f"skipped {dict(skipped)}\n")
    lines = []
    for cls in ("FD", "OD", "DC"):
        sel = [r for r in rows if r["cls"] == cls]
        if not sel:
            continue
        n = len({r["id"] for r in sel})
        print(f"--- {cls}  ({n} scenarios)")
        print(f"  {'arm':<10}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}"
              f"{'vs none':>10}{'W/L':>9}{'p':>9}")
        by = defaultdict(dict)
        for r in sel:
            if r["extractor"] == "hungarian":
                by[r["id"]][r["arm"]] = r
        for arm in order:
            h = [r for r in sel if r["arm"] == arm and r["extractor"] == "hungarian"]
            m = [r for r in sel if r["arm"] == arm and r["extractor"] == "mutual_top1"]
            if not h:
                continue
            pr = [(v[arm], v["none"]) for v in by.values()
                  if arm in v and "none" in v]
            W = sum(1 for x, y in pr if float(x["f1"]) > float(y["f1"]))
            L = sum(1 for x, y in pr if float(x["f1"]) < float(y["f1"]))
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            d = mean(float(x["f1"]) - float(y["f1"]) for x, y in pr) if pr else 0.0
            f1h = mean(float(r["f1"]) for r in h)
            f1m = mean(float(r["f1"]) for r in m)
            rg = mean(float(r["recall_at_gt"]) for r in h)
            mr = mean(float(r["mrr"]) for r in h)
            print(f"  {arm:<10}{f1h:>9.3f}{f1m:>9.3f}{rg:>8.3f}{mr:>8.3f}"
                  f"{d:>+10.3f}{f'{W}/{L}':>9}{p:>9.4f}")
            lines.append((cls, arm, f1h, f1m, rg, mr, d, W, L, p, n))
        print()

    per_class = {cls: n for cls, _, _, _, _, _, _, _, _, _, n in lines}
    tex = Path(args.tex)
    tex.parent.mkdir(parents=True, exist_ok=True)
    with tex.open("w", encoding="utf-8") as fh:
        fh.write("% generated by harness/transferability_table.py\n")
        fh.write("\\begin{table}[t]\n\\centering\\small\n")
        counts = ", ".join(f"{c}~{k}" for c, k in sorted(per_class.items()))
        fh.write("\\caption{Transferability across three dependency classes on "
                 "unseen confirmation scenarios (" + counts + "). One seed, one "
                 "descriptor per class, one fusion weight. \\emph{anchor} uses a "
                 "budget fixed on held-out exploration cases and no ground truth; "
                 "\\emph{oracle} is the ceiling. $\\Delta$ and W/L are against "
                 "\\emph{none}. Order is undefined where a schema yields no order "
                 "dependencies at $\\tau=0.99$, which is why its row rests on fewer "
                 "scenarios.}\n")
        fh.write("\\label{tab:transferability}\n")
        fh.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        fh.write("Class & Arm & F1 & F1$_{\\mathrm{mut}}$ & R@GT & "
                 "$\\Delta$F1 & W/L \\\\\n\\midrule\n")
        last = None
        for cls, arm, f1h, f1m, rg, mr, d, W, L, p, n in lines:
            if last is not None and cls != last:
                fh.write("\\midrule\n")
            star = "$^{*}$" if p < 0.05 and arm != "none" else ""
            label = cls if cls != last else ""
            fh.write(f"{label} & {arm} & {f1h:.3f} & {f1m:.3f} & {rg:.3f} & "
                     f"{d:+.3f}{star} & {W}/{L} \\\\\n")
            last = cls
        fh.write("\\bottomrule\n\\end{tabular}\n")
        fh.write("\\\\[2pt]\\footnotesize $^{*}$ $p<0.05$, exact binomial on "
                 "per-scenario wins.\n\\end{table}\n")
    print(f"wrote {out}\nwrote {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
