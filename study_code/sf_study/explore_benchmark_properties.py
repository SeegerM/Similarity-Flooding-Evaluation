"""EXPLORATORY X77 (2026-10-06): benchmark characterisation, part (a). Per-scenario properties of every
corpus the study used, to show where Valentine is extreme. Written before any property was computed; design and predictions in
EXPLORATORY.md X77. The per-scenario file is the interface to X78 (C).

Matrix order and labels come from each scenario's datatype_nanfix seed (reference/seeds/<corpus>/datatype_nanfix); the truth
matrix is read in that order. Columns are read from the side's CSV tables (first MAX_ROWS rows), labels "table.column".
Value evidence for the true pairs vs a seeded sample of the same number of non-pairs:
  ovl   Jaccard of the value sets (at most MAX_DISTINCT distinct values per column)
  dist  1 − mean |diff| of (distinct ratio, null ratio, min(mean length, 50)/50, numeric share)
  name  difflib ratio of the lower-cased bare column names
Dependency structure from explore_deproles.profiles (corrected FD parse): FDs per column (as right-hand side), share of columns in
a unary UCC, unary INDs per column; side means.

    ../harness/.venv-bdi/Scripts/python.exe explore_benchmark_properties.py [--workers 10]
Output: results/benchmark_properties.csv (per scenario), results/benchmark_profile.md (per corpus)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import math
import re
import sys
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE))
import score_seeds  # noqa: E402
from explore_deproles import profiles  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

csv.field_size_limit(2**31 - 1)
MAX_ROWS, MAX_DISTINCT = 20_000, 2_000
GEN = HERE.parent / "harness" / "generated"
# corpus -> (root, seed folder name, multi-table)
CORPORA = {
    "valentine_plain": (score_seeds.CORPUS["plain"], "plain", False),
    "valentine_md5": (score_seeds.CORPUS["md5"], "md5", False),
    "final_controlled": (GEN / "final-controlled", "final_controlled", False),
    "t2d": (GEN / "t2d-usable", "t2d", False),
    "ctl": (GEN / "controlled-imported-v3", "ctl", False),
    "final_indep": (HERE / "data" / "final_indep", "final_indep", True),
    "indep2": (HERE / "data" / "indep2_plain", "indep2", True),
    "synthea": (HERE / "data" / "synthea_plain", "synthea", True),
    "indep": (HERE / "data" / "indep_plain", "indep", True),
}
FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
NULLS = {"", "null", "none", "nan", "na", "n/a", "\\n"}


def regime_of(corpus: str, sc: str) -> str:
    if corpus == "valentine_md5":
        return "md5"
    if corpus in ("final_controlled", "ctl"):
        return "plain" if sc.endswith("_plain") else "opaque"
    return "plain"


def source_id(corpus: str, ds: str, sc: str) -> str:
    if corpus.startswith("valentine"):
        return f"{ds.split('-')[0]}/{sc.split('_')[0]}"
    if corpus in ("final_controlled", "ctl"):
        return ds.split("-")[0]
    if corpus == "t2d":
        return sc
    if corpus in ("indep2", "indep"):
        return ds
    if corpus == "synthea":
        return "SyntheaOMOP"
    return ds


def column_stats(scen: Path, side: str) -> tuple[dict, int, int]:
    """label -> (value set, (distinct ratio, null ratio, length, numeric share), name); total rows; table count."""
    out, rows_total, tables = {}, 0, 0
    for t in sorted((scen / side).glob("*.csv")):
        tables += 1
        with t.open(encoding="utf-8", errors="replace", newline="") as fh:
            rd = csv.reader(fh)
            try:
                header = next(rd)
            except StopIteration:
                continue
            vals = [[] for _ in header]
            n = 0
            for row in rd:
                for i in range(len(header)):
                    vals[i].append(row[i] if i < len(row) else "")
                n += 1
                if n >= MAX_ROWS:
                    break
        rows_total += n
        for h, v in zip(header, vals):
            nn = [x for x in v if x.strip().lower() not in NULLS]
            k = len(nn)
            cnt = Counter(nn)
            d = len(cnt)
            vs = {x for x, _ in cnt.most_common(MAX_DISTINCT)}
            ml = (sum(len(x) for x in nn) / k) if k else 0.0
            num = (sum(1 for x in nn if FLOAT_RE.match(x.strip())) / k) if k else 0.0
            feat = (d / k if k else 0.0, 1 - k / len(v) if v else 1.0, min(ml, 50) / 50, num)
            out[f"{t.stem}.{h}"] = (vs, feat, h)
    return out, rows_total, tables


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def dep_props(corpus_key: str, ds: str, sc: str, side: str, labels: list[str]) -> tuple[float, float, float]:
    try:
        fds, uccs, inds = profiles(corpus_key, ds, sc, side)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    lab = set(labels)
    n = max(len(labels), 1)
    rhs = sum(1 for _, r in fds if r in lab)
    unary = {u[0] for u in uccs if len(u) == 1}
    return rhs / n, len(unary & lab) / n, sum(1 for d_, _ in inds if d_ in lab) / n


def score_case(job):
    corpus, ds, sc = job
    root, seedname, multi = CORPORA[corpus]
    score_seeds.CORPUS[seedname] = root
    f_d = HERE / "reference" / "seeds" / seedname / "datatype_nanfix" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(root, ds, sc)
    if g is None or not f_d.is_file():
        return None
    _, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != (len(rows), len(cols)):
        return None
    scen = root / ds / sc
    S, src_rows, src_tables = column_stats(scen, "source")
    T, tgt_rows, tgt_tables = column_stats(scen, "target")
    ti = np.argwhere(gt)
    rng = np.random.default_rng(zlib.crc32(f"x77|{corpus}|{ds}|{sc}".encode()))
    non = np.argwhere(~gt)
    ni = non[rng.choice(len(non), size=min(len(ti), len(non)), replace=False)] if len(non) and len(ti) else np.zeros((0, 2), int)

    def ev(idx):
        ovl, dist, name = [], [], []
        for i, j in idx:
            a, b = S.get(rows[i]), T.get(cols[j])
            if a is None or b is None:
                continue
            ovl.append(jacc(a[0], b[0]))
            dist.append(1 - float(np.mean(np.abs(np.array(a[1]) - np.array(b[1])))))
            name.append(difflib.SequenceMatcher(None, a[2].lower(), b[2].lower()).ratio())
        m = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
        return m(ovl), m(dist), m(name), len(ovl)

    ot, dt_, nt, hit_t = ev(ti)
    on, dn, nn_, _ = ev(ni)
    feats = [S[r][1] for r in rows if r in S] + [T[c][1] for c in cols if c in T]
    num_share = float(np.mean([f[3] >= 0.95 for f in feats])) if feats else float("nan")
    kinds = Counter("num" if f[3] >= 0.95 else ("empty" if f[1] >= 1.0 else "text") for f in feats)
    tot = sum(kinds.values())
    dtype_entropy = -sum(c / tot * math.log2(c / tot) for c in kinds.values()) if tot else float("nan")
    seed_key = seedname
    fs = dep_props(seed_key, ds, sc, "source", rows)
    ft = dep_props(seed_key, ds, sc, "target", cols)
    diag = (float(np.mean(ti[:, 0] == ti[:, 1])) if len(ti) else float("nan")) if not multi else float("nan")
    return {"corpus": corpus, "regime": regime_of(corpus, sc), "dataset": ds, "scenario": sc,
            "group": ds.split("-")[0], "source_id": source_id(corpus, ds, sc),
            "src_cols": len(rows), "tgt_cols": len(cols), "size_ratio": min(len(rows), len(cols)) / max(len(rows), len(cols)),
            "src_rows": src_rows, "tgt_rows": tgt_rows, "src_tables": src_tables, "tgt_tables": tgt_tables,
            "n_true": int(gt.sum()), "true_share": float(gt.mean()), "diag_share": diag,
            "num_share": num_share, "dtype_entropy": dtype_entropy,
            "ovl_true": ot, "ovl_non": on, "dist_true": dt_, "dist_non": dn, "name_true": nt, "name_non": nn_,
            "label_hit": hit_t / max(len(ti), 1),
            "fd_per_col": float(np.nanmean([fs[0], ft[0]])), "key_share": float(np.nanmean([fs[1], ft[1]])),
            "ind_per_col": float(np.nanmean([fs[2], ft[2]]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    jobs = []
    for corpus, (root, _, _) in CORPORA.items():
        if a.only and corpus not in a.only.split(","):
            continue
        if not root.is_dir():
            print(f"skip {corpus}: {root} missing")
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for s in sorted(p for p in d.iterdir() if p.is_dir()):
                jobs.append((corpus, d.name, s.name))
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap_unordered(score_case, jobs, chunksize=4) if r]
    rows.sort(key=lambda r: (r["corpus"], r["dataset"], r["scenario"]))
    out = HERE / "results"
    with (out / "benchmark_properties.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["corpus"], r["regime"])].append(r)
    med = lambda rs, k: float(np.nanmedian([float(r[k]) for r in rs]))  # noqa: E731
    lines = ["# X77: benchmark characterisation, per corpus (EXPLORATORY)", "",
             f"{len(rows)} scenarios. Medians over scenarios; ovl/dist/name for true pairs vs sampled non-pairs (gap = true − non).", "",
             "| corpus | regime | scen. | sources | scen./source | tables/side | cols (src×tgt) | true share | diag share | "
             "ovl true / gap | dist true / gap | name true / gap | FD/col | key share | IND/col |",
             "|" + "---|" * 16]
    for (c, rg), rs in sorted(by.items()):
        srcs = len({r["source_id"] for r in rs})
        tb = med(rs, "src_tables"), med(rs, "tgt_tables")
        lines.append(
            f"| {c} | {rg} | {len(rs)} | {srcs} | {len(rs) / srcs:.1f} | {tb[0]:.0f}/{tb[1]:.0f} | "
            f"{med(rs, 'src_cols'):.0f}×{med(rs, 'tgt_cols'):.0f} | {med(rs, 'true_share'):.3f} | "
            f"{med(rs, 'diag_share'):.2f} | "
            f"{med(rs, 'ovl_true'):.2f} / {med(rs, 'ovl_true') - med(rs, 'ovl_non'):+.2f} | "
            f"{med(rs, 'dist_true'):.2f} / {med(rs, 'dist_true') - med(rs, 'dist_non'):+.2f} | "
            f"{med(rs, 'name_true'):.2f} / {med(rs, 'name_true') - med(rs, 'name_non'):+.2f} | "
            f"{med(rs, 'fd_per_col'):.1f} | {med(rs, 'key_share'):.2f} | {med(rs, 'ind_per_col'):.2f} |")
    vt = [r for r in rows if r["corpus"].startswith("valentine")]
    if vt:
        lines += ["", "Valentine by scenario type (dataset suffix), readable:", ""]
        bt = defaultdict(list)
        for r in vt:
            if r["regime"] == "plain":
                bt[r["dataset"].split("-", 1)[1]].append(r)
        for ty, rs in sorted(bt.items()):
            lines.append(f"- {ty} ({len(rs)}): name true {med(rs, 'name_true'):.2f} (non {med(rs, 'name_non'):.2f}); "
                         f"ovl true {med(rs, 'ovl_true'):.2f} (non {med(rs, 'ovl_non'):.2f}); dist true {med(rs, 'dist_true'):.2f}")
    (out / "benchmark_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
