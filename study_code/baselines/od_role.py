"""Order dependencies as a column-role descriptor, tested as a controlled substitution.

Sections 11 and 13 measured four within-table relational roles -- gcov,
containment, Jaccard and the uncertainty coefficient -- as interchangeable
substitutions into a fixed pipeline, and found them nearly equivalent, each
adding between +0.006 and +0.009 F1 on top of char3. Every one of those is
built from value *equality*.

An order dependency is the first descriptor here that is not. It constrains how
values *sort*, so two columns inducing the same partition can still differ in
their OD role. This asks whether that difference carries matching signal, using
the same descriptor shape and the same fusion weight as the equality-based
roles, so the comparison isolates the relation and nothing else.

Each column is described by its outgoing and incoming OD degree, normalised by
the number of columns, plus the mean arity of the dependencies it participates
in. Two columns are scored by the cosine of those profiles, exactly as the
existing role measures are.

Usage:
    python od_role.py <data-dir> --dataset ChEMBL-Unionable_partial050
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smx.adapters import read_matrix  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

OD_RE = re.compile(r"^\[(.*)\]\s*~>\s*(.+)$")
# "[DC:  t0.a <> t1.a  t0.b == t1.b ]" -- one predicate per column reference.
DC_PRED_RE = re.compile(r"t[01]\.([A-Za-z0-9_]+)\s*(==|<>|<=|>=|<|>)")


def parse_ods(path: Path):
    """(lhs columns, rhs columns) pairs, bare column names."""
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = OD_RE.match(line.strip())
        if not m:
            continue
        lhs = [c.strip().split(".")[-1] for c in m.group(1).split(",") if c.strip()]
        rhs = [c.strip().split(".")[-1] for c in m.group(2).split(",") if c.strip()]
        if lhs and rhs:
            out.append((lhs, rhs))
    return out


def parse_dcs(path: Path):
    """Per denial constraint, the columns it mentions and its predicate count.

    A DC is a forbidden combination of predicates over a tuple pair, so it has
    no left/right side to mirror the OD descriptor with. The projection used
    here is deliberately the simplest defensible one -- which columns a
    constraint constrains, and how many predicates it takes to do so -- rather
    than whichever projection scores best.
    """
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cols = DC_PRED_RE.findall(line)
        if not cols:
            continue
        names = [c for c, _ in cols]
        equalities = sum(1 for _, op in cols if op == "==")
        out.append((names, len(cols), equalities))
    return out


def parse_lima_dcs(path: Path):
    """Read the canonical LIMA cache projection, preserving arbitrary labels."""
    if not path.is_file():
        return []
    artifact = json.loads(path.read_text(encoding="utf-8"))
    labels = {column["id"]: column["label"] for column in artifact["spec"]["columns"]}
    out = []
    for record in artifact["records"]:
        names = [labels[predicate["column"]] for predicate in record["predicates"]]
        equalities = sum(predicate["operator"] == "=" for predicate in record["predicates"])
        out.append((names, len(record["predicates"]), equalities))
    return out


def dc_profile(dcs, columns):
    """Degree, mean constraint length, and equality share, per column."""
    n = max(1, len(columns))
    index = {c: i for i, c in enumerate(columns)}
    deg = np.zeros(len(columns))
    lengths = defaultdict(list)
    eq_share = defaultdict(list)
    for names, size, equalities in dcs:
        for c in set(names):
            if c in index:
                i = index[c]
                deg[i] += 1
                lengths[i].append(size)
                eq_share[i].append(equalities / size if size else 0.0)
    prof = np.zeros((len(columns), 3))
    prof[:, 0] = deg / n
    for i in range(len(columns)):
        prof[i, 1] = mean(lengths[i]) / n if lengths[i] else 0.0
        prof[i, 2] = mean(eq_share[i]) if eq_share[i] else 0.0
    return prof


def od_profile(ods, columns):
    """Outgoing degree, incoming degree, mean arity, per column."""
    n = max(1, len(columns))
    index = {c: i for i, c in enumerate(columns)}
    out_deg = np.zeros(len(columns))
    in_deg = np.zeros(len(columns))
    arity = defaultdict(list)
    for lhs, rhs in ods:
        for c in lhs:
            if c in index:
                out_deg[index[c]] += 1
                arity[index[c]].append(len(lhs))
        for c in rhs:
            if c in index:
                in_deg[index[c]] += 1
                arity[index[c]].append(len(rhs))
    prof = np.zeros((len(columns), 3))
    prof[:, 0] = out_deg / n
    prof[:, 1] = in_deg / n
    for i in range(len(columns)):
        prof[i, 2] = mean(arity[i]) / n if arity[i] else 0.0
    return prof


def cosine_similarity(a, b):
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return (a / na) @ (b / nb).T


def scale_align(base, extra, weight):
    """Range-align `extra` to `base` before blending, as the pipeline does."""
    lo, hi = float(base.min()), float(base.max())
    if hi <= lo:
        return base
    e_lo, e_hi = float(extra.min()), float(extra.max())
    scaled = (np.full_like(extra, 0.5) if e_hi <= e_lo
              else (extra - e_lo) / (e_hi - e_lo))
    return (1.0 - weight) * base + weight * (lo + scaled * (hi - lo))


def columns_of(table: Path):
    with table.open(newline="", encoding="utf-8", errors="replace") as fh:
        return next(csv.reader(fh), [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--scenario", action="append", help="optional scenario-name glob")
    ap.add_argument("--first-line", required=True,
                    help="run dir supplying the label matrix as the base")
    ap.add_argument("--relation", default="od", choices=["od", "dc"])
    ap.add_argument("--dc-source", default="dcfinder", choices=["dcfinder", "lima"],
                    help="DC profile to use; LIMA reads canonical dc.scored.json")
    ap.add_argument("--weight", type=float, default=0.35)
    ap.add_argument("-o", "--out", default="reports/od_role.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    root = data_dir / args.dataset

    base_of = {}
    for p in sorted(Path(args.first_line).rglob("*.csv")):
        if "MatchingStepLine1" not in str(p) and p.name != "seed.csv":
            continue
        parts = p.relative_to(args.first_line).parts
        if len(parts) >= 2 and parts[0] == args.dataset:
            base_of.setdefault(parts[1], p)

    rows = []
    counts = []
    for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
        name = scenario.name
        if args.scenario and not any(fnmatch.fnmatchcase(name, pattern) for pattern in args.scenario):
            continue
        if name not in base_of:
            continue
        gt_files = sorted((scenario / "ground_truth").glob("*.csv"))
        src = sorted((scenario / "source").glob("*.csv"))
        tgt = sorted((scenario / "target").glob("*.csv"))
        if not gt_files or not src or not tgt:
            continue
        gt, _, _ = read_matrix(gt_files[0])
        base, _, _ = read_matrix(base_of[name])
        if base.shape != gt.shape:
            continue

        s_cols, t_cols = columns_of(src[0]), columns_of(tgt[0])
        if args.relation == "od":
            fname, parse, profile = "OD_results.txt", parse_ods, od_profile
        else:
            fname, parse, profile = (("dc.scored.json", parse_lima_dcs, dc_profile)
                                     if args.dc_source == "lima" else
                                     ("DC_results.txt", parse_dcs, dc_profile))
        s_ods = parse(scenario / "metadata" / "source" / src[0].stem / fname)
        t_ods = parse(scenario / "metadata" / "target" / tgt[0].stem / fname)
        counts.append((len(s_ods), len(t_ods)))
        if len(s_cols) != base.shape[0] or len(t_cols) != base.shape[1]:
            continue

        role = cosine_similarity(profile(s_ods, s_cols), profile(t_ods, t_cols))
        label = args.relation.upper()
        methods = {
            "base": base,
            f"base + {label} role": scale_align(base, role, args.weight),
            f"{label} role alone": role,
        }
        for method, m in methods.items():
            r = evaluate(ScoreMatrix(scores=m, ground_truth=gt > 0.5),
                         policies=["hungarian", "mutual_top1"])
            for row in r:
                rows.append({"scenario": name, "method": method,
                             "extractor": row["extractor"], "f1": row["f1"],
                             "recall_at_gt": row.get("recall_at_gt_pessimistic",
                                                     float("nan")),
                             "mrr": row.get("mrr_pessimistic", float("nan"))})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{args.dataset}: {len(set(r['scenario'] for r in rows))} scenarios, "
          f"mean {args.relation.upper()}s {mean(a for a, _ in counts):.0f} source / "
          f"{mean(b for _, b in counts):.0f} target\n")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["method"], r["extractor"])]["f1"].append(r["f1"])
        agg[(r["method"], r["extractor"])]["rgt"].append(r["recall_at_gt"])
        agg[(r["method"], r["extractor"])]["mrr"].append(r["mrr"])

    head = f"{'method':<18} {'extractor':<14} {'F1':>7} {'R@GT':>7} {'MRR':>7}"
    print(head)
    print("-" * len(head))
    label = args.relation.upper()
    for method in ("base", f"base + {label} role", f"{label} role alone"):
        for ex in ("hungarian", "mutual_top1"):
            v = agg.get((method, ex))
            if v:
                print(f"{method:<18} {ex:<14} {mean(v['f1']):>7.3f} "
                      f"{mean(v['rgt']):>7.3f} {mean(v['mrr']):>7.3f}")

    from scipy.stats import binomtest
    print()
    for ex in ("hungarian", "mutual_top1"):
        pairs = []
        by = defaultdict(dict)
        for r in rows:
            if r["extractor"] == ex:
                by[r["scenario"]][r["method"]] = r["f1"]
        for v in by.values():
            if f"base + {label} role" in v and "base" in v:
                pairs.append((v[f"base + {label} role"], v["base"]))
        if pairs:
            W = sum(1 for a, b in pairs if a > b)
            L = sum(1 for a, b in pairs if a < b)
            p = binomtest(W, W + L, 0.5).pvalue if W + L else float("nan")
            print(f"{label} role vs base ({ex}): {mean(a - b for a, b in pairs):+.3f} "
                  f"{W}W {L}L {len(pairs) - W - L}T  p={p:.3f}")

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
