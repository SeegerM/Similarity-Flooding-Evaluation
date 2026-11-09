"""The Kang-Naughton un-interpreted structural baseline, in this harness.

Kang and Naughton (SIGMOD 2003) define a matcher as *un-interpreted* when
applying arbitrary independent one-to-one maps to each source's column values
leaves its output unchanged, and obtain it by building a mutual-information
dependency graph per schema and matching the two graphs. The 2008 extension
replaces exact graph matching with an approximation. This record sits directly
in that lineage, so a reviewer will ask how transferability-guided dependency
selection differs from matching interattribute-dependency graphs. This answers
it empirically rather than by argument.

Two implementations, because the paper has two halves and they are worth
separating:

    kn-graph    the dependency-graph match itself: soft quadratic assignment
                over the two MI matrices, iterated with annealing, so a
                *score matrix* comes out and the harness's own extractors and
                ranking metrics apply unchanged
    kn-profile  the cheaper structural summary the same graph supports -- each
                column described by its sorted vector of MI to every other
                column, compared directly

Neither reads a column name or a value in common between the schemas, so both
are un-interpreted in the paper's sense, and both should be exactly invariant
under independent per-column bijective recoding. That invariance is checked
rather than assumed.

MI is normalised by the smaller marginal entropy. Raw MI grows with column
cardinality, and two schemas whose tables differ in size would then have
systematically different edge weights, which the graph match would read as
structural disagreement.

Usage:
    python kang_naughton.py [--cases reports/valentine_full_382_20260903]
    python kang_naughton.py --invariance   # bijective-recoding check only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_paths import resolve  # noqa: E402

from anchor_generalization import bare  # noqa: E402
from class_comparison_conditioned import read_columns  # noqa: E402
from graduated_assignment import sinkhorn  # noqa: E402
from od_role import scale_align  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}
EPS = 1e-12


def codes(column):
    """Integer codes for a column's values; -1 marks null."""
    out, index = np.empty(len(column), dtype=np.int64), {}
    for i, raw in enumerate(column):
        v = raw.strip()
        if v.lower() in NULLS:
            out[i] = -1
            continue
        out[i] = index.setdefault(v, len(index))
    return out, len(index)


def entropy(counts):
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def mi_matrix(columns):
    """Normalised mutual information between every pair of columns.

    Computed on pair-complete rows: a pair where either side is null carries no
    information about their association and including it as a category would
    manufacture some.
    """
    coded = [codes(c) for c in columns]
    n = len(columns)
    out = np.zeros((n, n))
    for i in range(n):
        ci, ki = coded[i]
        for j in range(i + 1, n):
            cj, kj = coded[j]
            ok = (ci >= 0) & (cj >= 0)
            if ok.sum() < 2 or ki < 1 or kj < 1:
                continue
            a, b = ci[ok], cj[ok]
            joint = np.zeros((ki, kj))
            np.add.at(joint, (a, b), 1.0)
            hx = entropy(joint.sum(axis=1))
            hy = entropy(joint.sum(axis=0))
            hxy = entropy(joint.ravel())
            mi = hx + hy - hxy
            denom = min(hx, hy)
            v = 0.0 if denom <= EPS else float(np.clip(mi / denom, 0.0, 1.0))
            out[i, j] = out[j, i] = v
    return out


def kn_profile_scores(w1, w2):
    """Each column described by its sorted MI vector, compared by 1 - mean |d|.

    Sorting makes the descriptor permutation-invariant, which is what lets two
    schemas be compared before any correspondence is known.
    """
    def profile(w, width):
        n = w.shape[0]
        out = np.zeros((n, width))
        for i in range(n):
            row = np.sort(np.delete(w[i], i))[::-1]
            out[i, :min(width, row.size)] = row[:width]
        return out
    width = max(1, min(w1.shape[0], w2.shape[0]) - 1)
    p1, p2 = profile(w1, width), profile(w2, width)
    return np.clip(1.0 - np.abs(p1[:, None, :] - p2[None, :, :]).mean(axis=2), 0, 1)


def kn_graph_scores(w1, w2, betas=(0.5, 1.0, 2.0, 4.0), inner=10):
    """Soft quadratic assignment between two MI dependency graphs.

    The affinity between candidate pairs (i,a) and (j,b) is how closely the two
    edge weights agree. Unlike the sparse FD graphs of FINDINGS 26, an MI graph
    is dense, so agreement about a *missing* edge is not the dominant event and
    the failure mode diagnosed there does not apply here.
    """
    n1, n2 = w1.shape[0], w2.shape[0]
    affinity = 1.0 - np.abs(w1[:, :, None, None] - w2[None, None, :, :])
    affinity = np.clip(affinity, 0.0, 1.0)
    affinity[np.arange(n1), np.arange(n1), :, :] = 0.0
    m = np.full((n1, n2), 1.0 / max(n1, n2))
    for beta in betas:
        for _ in range(inner):
            score = np.einsum("ijab,jb->ia", affinity, m, optimize=True)
            peak = score.max()
            if peak > 0:
                score = score / peak
            m = sinkhorn(np.exp(beta * score))
    return m


def kn_graph_scores_masked(w1, w2, e1, e2, mode, betas=(0.5, 1.0, 2.0, 4.0), inner=10):
    """`kn_graph_scores` with dependency edges selected on each side (KN_HOST_SELECTION.md).

    e1, e2 are 0/1 symmetric edge masks. mode "restrict": an edge pair votes only
    if both edges are selected; mode "emphasis": such pairs vote twice, all others
    once. A full mask under "restrict" reproduces `kn_graph_scores` exactly.
    """
    n1, n2 = w1.shape[0], w2.shape[0]
    affinity = 1.0 - np.abs(w1[:, :, None, None] - w2[None, None, :, :])
    affinity = np.clip(affinity, 0.0, 1.0)
    affinity[np.arange(n1), np.arange(n1), :, :] = 0.0
    both = e1[:, :, None, None] * e2[None, None, :, :]
    if mode == "restrict":
        affinity = affinity * both
    elif mode == "emphasis":
        affinity = affinity * (1.0 + both)
    else:
        raise ValueError(mode)
    m = np.full((n1, n2), 1.0 / max(n1, n2))
    for beta in betas:
        for _ in range(inner):
            score = np.einsum("ijab,jb->ia", affinity, m, optimize=True)
            peak = score.max()
            if peak > 0:
                score = score / peak
            m = sinkhorn(np.exp(beta * score))
    return m


def fd_edge_mask(fds, cols):
    """Undirected determinant-dependent edges of an FD subset, over the columns `cols`."""
    idx = {c: i for i, c in enumerate(cols)}
    e = np.zeros((len(cols), len(cols)))
    for det, dep in fds:
        if dep not in idx:
            continue
        for x in det:
            if x in idx and x != dep:
                e[idx[x], idx[dep]] = e[idx[dep], idx[x]] = 1.0
    return e


def recode(columns, seed):
    """Independent bijective relabelling of each column's values."""
    rng = np.random.default_rng(seed)
    out = []
    for col in columns:
        distinct = sorted({v.strip() for v in col if v.strip().lower() not in NULLS})
        perm = rng.permutation(len(distinct))
        table = {v: f"z{perm[i]:06d}" for i, v in enumerate(distinct)}
        out.append([table.get(v.strip(), v) for v in col])
    return out


def invariance_check(data, limit=6):
    """Both variants must be unchanged by independent per-column recoding."""
    root = Path(data)
    checked, worst_graph, worst_profile = 0, 0.0, 0.0
    for family in sorted(p for p in root.iterdir() if p.is_dir()):
        for scenario in sorted(p for p in family.iterdir() if p.is_dir()):
            src = sorted((scenario / "source").glob("*.csv"))
            tgt = sorted((scenario / "target").glob("*.csv"))
            if not src or not tgt:
                continue
            _, s_cols = read_columns(src[0])
            _, t_cols = read_columns(tgt[0])
            if not s_cols or not t_cols:
                continue
            w1, w2 = mi_matrix(s_cols), mi_matrix(t_cols)
            r1 = mi_matrix(recode(s_cols, 11))
            r2 = mi_matrix(recode(t_cols, 22))
            worst_graph = max(worst_graph, float(np.abs(
                kn_graph_scores(w1, w2) - kn_graph_scores(r1, r2)).max()))
            worst_profile = max(worst_profile, float(np.abs(
                kn_profile_scores(w1, w2) - kn_profile_scores(r1, r2)).max()))
            checked += 1
            if checked >= limit:
                print(f"independent bijective recoding, {checked} scenarios")
                print(f"  max |change| kn-graph  : {worst_graph:.2e}")
                print(f"  max |change| kn-profile: {worst_profile:.2e}")
                return 0
    print("no scenarios checked", file=sys.stderr)
    return 1


def summarise(rows, order, label_pred=None):
    def table(pred, label):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return
        print(f"--- {label}  ({len({r['id'] for r in sel})} scenarios)")
        print(f"  {'arm':<26}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}")
        for arm in order:
            h = [r for r in sel if r["arm"] == arm and r["extractor"] == "hungarian"]
            m = [r for r in sel if r["arm"] == arm and r["extractor"] == "mutual_top1"]
            if not h:
                continue
            print(f"  {arm:<26}{mean(float(r['f1']) for r in h):>9.3f}"
                  f"{mean(float(r['f1']) for r in m):>9.3f}"
                  f"{mean(float(r['recall_at_gt']) for r in h):>8.3f}"
                  f"{mean(float(r['mrr']) for r in h):>8.3f}")
        print()
    return table


def run_schematch(args) -> int:
    """Same two variants on a schematch-layout cohort, e.g. the MD5 condition."""
    root = Path(args.schematch_data) / args.dataset
    base_of = {}
    if args.first_line:
        run = Path(args.first_line)
        for p in sorted(run.rglob("*.csv")):
            if "MatchingStepLine1" in str(p):
                parts = p.relative_to(run).parts
                if len(parts) >= 2 and parts[0] == args.dataset:
                    base_of.setdefault(parts[1], p)

    rows, matrices = [], {}
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        gt_f = sorted((scenario / "ground_truth").glob("*.csv"))
        src = sorted((scenario / "source").glob("*.csv"))
        tgt = sorted((scenario / "target").glob("*.csv"))
        if not gt_f or not src or not tgt:
            continue
        gt, _, _ = read_matrix(gt_f[0])
        s_names, s_cols = read_columns(src[0])
        t_names, t_cols = read_columns(tgt[0])
        if len(s_names) != gt.shape[0] or len(t_names) != gt.shape[1]:
            continue
        w1, w2 = mi_matrix(s_cols), mi_matrix(t_cols)
        graph, profile = kn_graph_scores(w1, w2), kn_profile_scores(w1, w2)
        matrices[scenario.name] = (graph, profile)
        truth = gt > 0.5
        arms = {"kn-graph (standalone)": graph,
                "kn-profile (standalone)": profile}
        base = None
        if scenario.name in base_of:
            base, _, _ = read_matrix(base_of[scenario.name])
            if base.shape == gt.shape:
                arms["seed"] = base
                arms["seed + kn-graph"] = scale_align(base, graph, args.weight)
                arms["seed + kn-profile"] = scale_align(base, profile, args.weight)
        for arm, m in arms.items():
            for res in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                                policies=["hungarian", "mutual_top1"]):
                rows.append({"id": scenario.name, "cohort": args.dataset,
                             "family": args.dataset, "scenario_type": "-",
                             "arm": arm, "extractor": res["extractor"],
                             "f1": res["f1"],
                             "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                     float("nan")),
                             "mrr": res.get("mrr_pessimistic", float("nan"))})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = ["seed", "kn-graph (standalone)", "kn-profile (standalone)",
             "seed + kn-graph", "seed + kn-profile"]
    print(f"{args.dataset}: {len({r['id'] for r in rows})} scenarios\n")
    summarise(rows, order)(lambda r: True, args.dataset)

    by = defaultdict(dict)
    for r in rows:
        if r["extractor"] == "hungarian":
            by[r["id"]][r["arm"]] = r
    for a, b in (("seed + kn-graph", "seed"), ("seed + kn-profile", "seed")):
        pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
        if not pr:
            continue
        W = sum(1 for x, y in pr if float(x["f1"]) > float(y["f1"]))
        L = sum(1 for x, y in pr if float(x["f1"]) < float(y["f1"]))
        p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
        print(f"  {a:<24} vs {b:<8}"
              f"{mean(float(x['f1']) - float(y['f1']) for x, y in pr):>+8.3f}"
              f"  {W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    np.save(Path(str(out) + ".matrices.npy"),
            np.array([(k, m[0], m[1]) for k, m in matrices.items()], dtype=object),
            allow_pickle=True)
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="reports/valentine_full_382_20260903")
    ap.add_argument("--data", default="generated/benchmark-full")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--invariance", action="store_true")
    # Schematch-layout mode, for cohorts the 382 expansion does not cover --
    # the MD5 condition in particular, which lives beside the plaintext data
    # rather than in the prepared benchmark.
    ap.add_argument("--schematch-data", default=None,
                    help="use <dir>/<dataset> scenarios instead of the cached cases")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--first-line", default=None,
                    help="run dir supplying the base matrix in schematch mode")
    ap.add_argument("-o", "--out", default="reports/kang_naughton.csv")
    args = ap.parse_args()

    if args.invariance:
        return invariance_check(args.data)

    if args.schematch_data:
        return run_schematch(args)

    cases_dir, data = Path(args.cases), Path(args.data)
    index = list(csv.DictReader(open(cases_dir / "cases.csv", encoding="utf-8")))
    if args.limit:
        index = index[:args.limit]

    rows, skipped = [], defaultdict(int)
    for rec in index:
        parts = rec["id"].split("/")
        # FINDINGS 67: the legacy rule drops the Magellan and Wikidata
        # families. Default is unchanged; SMX_RESOLVER=index corrects it.
        scenario = resolve(data, rec["id"])
        digest = hashlib.sha256(rec["id"].encode()).hexdigest()[:16]
        seed_path = cases_dir / "cases" / digest / "seed.csv"
        gt_files = sorted((scenario / "ground_truth").glob("*.csv"))
        if not scenario.is_dir() or not seed_path.is_file() or not gt_files:
            skipped["missing artefacts"] += 1
            continue
        base, row_labels, col_labels = read_matrix(seed_path)
        gt, _, _ = read_matrix(gt_files[0])
        if base.shape != gt.shape:
            skipped["shape mismatch"] += 1
            continue
        src = sorted((scenario / "source").glob("*.csv"))
        tgt = sorted((scenario / "target").glob("*.csv"))
        if not src or not tgt:
            skipped["no tables"] += 1
            continue
        s_names, s_cols = read_columns(src[0])
        t_names, t_cols = read_columns(tgt[0])
        if len(s_names) != gt.shape[0] or len(t_names) != gt.shape[1]:
            skipped["width mismatch"] += 1
            continue

        w1, w2 = mi_matrix(s_cols), mi_matrix(t_cols)
        graph = kn_graph_scores(w1, w2)
        profile = kn_profile_scores(w1, w2)
        truth = gt > 0.5
        arms = {
            "seed": base,
            "kn-graph (standalone)": graph,
            "kn-profile (standalone)": profile,
            "seed + kn-graph": scale_align(base, graph, args.weight),
            "seed + kn-profile": scale_align(base, profile, args.weight),
        }
        for arm, m in arms.items():
            for res in evaluate(ScoreMatrix(scores=m, ground_truth=truth),
                                policies=["hungarian", "mutual_top1"]):
                rows.append({"id": rec["id"], "cohort": rec["cohort"],
                             "family": rec["family"],
                             "scenario_type": rec["scenario_type"], "arm": arm,
                             "extractor": res["extractor"], "f1": res["f1"],
                             "recall_at_gt": res.get("recall_at_gt_pessimistic",
                                                     float("nan")),
                             "mrr": res.get("mrr_pessimistic", float("nan"))})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = ["seed", "kn-graph (standalone)", "kn-profile (standalone)",
             "seed + kn-graph", "seed + kn-profile"]
    print(f"scored {len({r['id'] for r in rows})} scenarios; skipped {dict(skipped)}\n")

    def table(pred, label):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return
        print(f"--- {label}  ({len({r['id'] for r in sel})} scenarios)")
        print(f"  {'arm':<26}{'F1 full':>9}{'F1 mut':>9}{'R@GT':>8}{'MRR':>8}")
        for arm in order:
            h = [r for r in sel if r["arm"] == arm and r["extractor"] == "hungarian"]
            m = [r for r in sel if r["arm"] == arm and r["extractor"] == "mutual_top1"]
            if not h:
                continue
            print(f"  {arm:<26}{mean(float(r['f1']) for r in h):>9.3f}"
                  f"{mean(float(r['f1']) for r in m):>9.3f}"
                  f"{mean(float(r['recall_at_gt']) for r in h):>8.3f}"
                  f"{mean(float(r['mrr']) for r in h):>8.3f}")
        print()

    table(lambda r: True, "all scenarios")
    table(lambda r: r["cohort"] == "remaining338", "confirmation cohort only")
    for st in sorted({r["scenario_type"] for r in rows}):
        table(lambda r, s=st: r["scenario_type"] == s, st)

    print("paired, confirmation cohort")
    for ex in ("hungarian", "mutual_top1"):
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex and r["cohort"] == "remaining338":
                by[r["id"]][r["arm"]] = r
        print(f"  {ex}")
        for a, b in (("seed + kn-graph", "seed"), ("seed + kn-profile", "seed"),
                     ("kn-graph (standalone)", "kn-profile (standalone)")):
            pr = [(v[a], v[b]) for v in by.values() if a in v and b in v]
            if not pr:
                continue
            W = sum(1 for x, y in pr if float(x["f1"]) > float(y["f1"]))
            L = sum(1 for x, y in pr if float(x["f1"]) < float(y["f1"]))
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"    {a:<24} vs {b:<22}"
                  f"{mean(float(x['f1']) - float(y['f1']) for x, y in pr):>+8.3f}"
                  f"  {W}W {L}L {len(pr) - W - L}T  p={p:.4f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
