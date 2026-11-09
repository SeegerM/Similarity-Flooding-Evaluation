"""EXPLORATORY X85 (2026-10-07): the new corpora characterised as Valentine was. Design and predictions in
EXPLORATORY.md X85, written before computing. Per-scenario properties are read from results/x81/benchmark_properties_x81.csv
(new corpora) and results/benchmark_properties.csv (Valentine plain, T2D); nothing there is recomputed.

  1. sources: distinct side tables per corpus by SHA-256 of each side's CSV files; scenarios per distinct table
  2. position artefact: the constant matcher's F1_H (index-order ties) vs F1_Hr (50 seeded permutations); share diag >= 0.9
  3. heterogeneity: quartiles of ovl_true and name_true; split-like share (ovl_true >= 0.3 and name_true >= 0.9); name_true = 1 share
  4. by domain: madi use cases

    ../harness/.venv-bdi/Scripts/python.exe explore_x85_characterise.py [--workers 10]
Output: results/x85/{constant.csv, report.md}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "harness", HERE / "tuning", HERE):
    sys.path.insert(0, str(p))
import score_seeds  # noqa: E402

GEN = HERE.parent / "harness" / "generated"
ROOTS = {"valentine_plain": (score_seeds.CORPUS["plain"], "plain"), "t2d": (GEN / "t2d-usable", "t2d"),
         "hdxsm": (GEN / "hdxsm-usable", "hdxsm"), "gdcsm": (GEN / "gdcsm-imported", "gdcsm"),
         "madi": (GEN / "madi-imported", "madi"), "goby": (GEN / "goby-imported", "goby"),
         "mimic": (HERE / "data" / "mimic_omop", "mimic"),
         # added 2026-10-07 for the corpus table: the other corpora of the study
         "final_controlled": (GEN / "final-controlled", "final_controlled"), "ctl": (GEN / "controlled-imported-v3", "ctl"),
         "final_indep": (HERE / "data" / "final_indep", "final_indep"), "synthea": (HERE / "data" / "synthea_plain", "synthea")}
ORDER = ["valentine_plain", "t2d", "gdcsm", "hdxsm", "goby", "madi_indep", "madi_products", "mimic"]
OUT = HERE / "results" / "x85"
# rows of the corpus table (results/x85/corpus_table.csv), in the paper's order: (label, built from one source?)
TABLE = [("valentine_plain", True), ("final_controlled_plain", True), ("ctl_plain", True), ("madi_products", True),
         ("t2d", False), ("synthea", False), ("final_indep", False), ("gdcsm", False), ("hdxsm", False), ("goby", False),
         ("madi_indep", False), ("mimic", False)]


def label(c: str, sc: str) -> str:
    if c == "madi":
        return "madi_products" if sc.startswith("products__") else "madi_indep"
    if c in ("final_controlled", "ctl"):
        return f"{c}_plain" if sc.endswith("_plain") else f"{c}_opaque"
    if c == "synthea":   # the 36 single table pairs; the 3 whole-schema scenarios (scenario = seed only) apart
        return "synthea" if "__" in sc else "synthea_full"
    return c


def components(pairs: list[tuple[str, str]]) -> int:
    """Connected components of scenarios linked by an identical side table (content hash), as X86."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find("S" + a), find("T" + b)
        if ra != rb:
            parent[ra] = rb
    return len({find("S" + a) for a, _ in pairs})


def units(c: str, rs: list[dict], ps: list[dict]) -> str:
    """Independent units: source tables or databases for corpora split from one source (X77's source_id); schema
    domains for the independent schema pairs; table-sharing components elsewhere (GDC-SM: its source studies, which
    share one target)."""
    if c in ("valentine_plain", "final_controlled_plain", "final_controlled_opaque", "ctl_plain", "ctl_opaque", "synthea"):
        return str(len({r["source_id"] for r in ps})) if c != "synthea" else "1"
    if c == "final_indep":
        return str(len({r["dataset"].split("_")[0] for r in ps}))
    if c in ("gdcsm", "hdxsm", "goby", "madi_indep", "madi_products"):   # the clusters of the paper's intervals (X86)
        sys.path.insert(0, str(HERE / "layer3b"))
        from x86_cluster_ci import cluster_map
        m, _ = cluster_map(c, sorted({r["scenario"] for r in ps}))
        return str(len(set(m.values())))
    return str(components([(r["src_hash"], r["tgt_hash"]) for r in rs]))


def side_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.glob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def job(args):
    c, ds, sc = args
    from score_screen_metrics import all_metrics, permutations
    from smx.adapters import read_matrix
    root, seedname = ROOTS[c]
    f_d = HERE / "reference" / "seeds" / seedname / "datatype_nanfix" / ds / sc / "seed.csv"
    f_c = HERE / "reference" / "seeds" / seedname / "constant" / ds / sc / "seed.csv"
    g = score_seeds.ground_truth_file(root, ds, sc)
    if g is None or not f_d.is_file() or not f_c.is_file():
        return None
    _, rows, cols = read_matrix(f_d)
    gt = read_matrix(g)[0] > 0.5
    C, rc, cc = read_matrix(f_c)
    if gt.shape != C.shape or list(rc) != list(rows) or list(cc) != list(cols) or not gt.any():
        return None
    r = all_metrics(np.nan_to_num(C), gt, permutations(gt.shape, f"{ds}/{sc}"))
    scen = root / ds / sc
    hs = side_hash(scen / "source") if (scen / "source").is_dir() else ""
    ht = side_hash(scen / "target") if (scen / "target").is_dir() else ""
    return {"corpus": label(c, sc), "dataset": ds, "scenario": sc, "F1_H": r["F1_H"], "F1_Hr": r["F1_Hr"],
            "src_hash": hs, "tgt_hash": ht}


def q(v, p):
    return float(np.nanpercentile(v, p)) if len(v) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, d.name, s.name) for c, (root, _) in ROOTS.items()
            for d in sorted(p for p in root.iterdir() if p.is_dir()) for s in sorted(p for p in d.iterdir() if p.is_dir())]
    with Pool(a.workers) as pool:
        cons = [r for r in pool.imap_unordered(job, jobs, chunksize=4) if r]
    cons.sort(key=lambda r: (r["corpus"], r["dataset"], r["scenario"]))
    with (OUT / "constant.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(cons[0]))
        w.writeheader()
        w.writerows(cons)
    props = defaultdict(list)
    for f, keep in ((HERE / "results" / "benchmark_properties.csv", {"valentine_plain", "t2d", "final_controlled", "ctl", "final_indep", "synthea"}),
                    (HERE / "results" / "x81" / "benchmark_properties_x81.csv", None)):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if keep is None or r["corpus"] in keep:
                props[label(r["corpus"], r["scenario"])].append(r)
    fl = lambda rs, k: np.array([float(r[k]) for r in rs if r[k] not in ("", "nan")])  # noqa: E731
    lines = ["# X85: the new corpora characterised as Valentine (EXPLORATORY)", "",
             "Properties from X77/X81 (medians and quartiles over scenarios); constant matcher re-scored here. Predictions: EXPLORATORY.md X85.", "",
             "## 1. Sources and shape", "",
             "| corpus | scen. | distinct source / target tables | scen. per distinct table | tables/side | cols src×tgt (median) | rows src/tgt (median) | size ratio |",
             "|---|---|---|---|---|---|---|---|"]
    by_c = defaultdict(list)
    for r in cons:
        by_c[r["corpus"]].append(r)
    for c in ORDER:
        rs, ps = by_c.get(c, []), props.get(c, [])
        if not ps:
            continue
        hs, ht = {r["src_hash"] for r in rs}, {r["tgt_hash"] for r in rs}
        alln = len(hs | ht)
        lines.append(f"| {c} | {len(ps)} | {len(hs)} / {len(ht)} | {2 * len(rs) / alln:.1f} | "
                     f"{np.median(fl(ps, 'src_tables')):.0f}/{np.median(fl(ps, 'tgt_tables')):.0f} | "
                     f"{np.median(fl(ps, 'src_cols')):.0f}×{np.median(fl(ps, 'tgt_cols')):.0f} | "
                     f"{np.median(fl(ps, 'src_rows')):.0f}/{np.median(fl(ps, 'tgt_rows')):.0f} | {np.median(fl(ps, 'size_ratio')):.2f} |")
    lines += ["", "Distinct tables are counted by content (all CSV files of a side). Valentine's 382 scenarios come from 10 source tables "
              "(X77); its sides are distinct fragments, so the content count overstates its independence.", "",
              "## 2. Position artefact (diagonal truth)", "",
              "| corpus | diag share median | scen. with diag ≥ 0.9 | constant F1_H (index order) | constant F1_Hr (random ties) | gap |",
              "|---|---|---|---|---|---|"]
    for c in ORDER:
        rs, ps = by_c.get(c, []), props.get(c, [])
        if not rs:
            continue
        dg = fl(ps, "diag_share")
        fh, fr = np.mean([r["F1_H"] for r in rs]), np.mean([r["F1_Hr"] for r in rs])
        lines.append(f"| {c} | {np.median(dg) if len(dg) else float('nan'):.2f} | "
                     f"{(dg >= 0.9).mean() if len(dg) else float('nan'):.0%} | {fh:.3f} | {fr:.3f} | {fh - fr:+.3f} |")
    lines += ["", "## 3. Heterogeneity within each corpus", "",
              "| corpus | ovl_true q25/q50/q75 | name_true q25/q50/q75 | split-like (ovl ≥ 0.3, name ≥ 0.9) | name_true = 1 | dist_true − dist_non |",
              "|---|---|---|---|---|---|"]
    for c in ORDER:
        ps = props.get(c, [])
        if not ps:
            continue
        o, n = fl(ps, "ovl_true"), fl(ps, "name_true")
        both = [(float(r["ovl_true"]), float(r["name_true"])) for r in ps if r["ovl_true"] not in ("", "nan") and r["name_true"] not in ("", "nan")]
        split = np.mean([x >= 0.3 and y >= 0.9 for x, y in both]) if both else float("nan")
        gap = np.median(fl(ps, "dist_true")) - np.median(fl(ps, "dist_non"))
        lines.append(f"| {c} | {q(o, 25):.2f}/{q(o, 50):.2f}/{q(o, 75):.2f} | {q(n, 25):.2f}/{q(n, 50):.2f}/{q(n, 75):.2f} | "
                     f"{split:.0%} | {(n >= 0.999).mean():.0%} | {gap:+.2f} |")
    lines += ["", "## 4. By domain (MaDI use cases)", "",
              "| domain | scen. | ovl_true | name_true | diag | num share | constant F1_H − F1_Hr |", "|---|---|---|---|---|---|---|"]
    md = [r for c in ("madi_indep", "madi_products") for r in props.get(c, [])]
    cm = {r["scenario"]: r for c in ("madi_indep", "madi_products") for r in by_c.get(c, [])}
    for dom in sorted({r["scenario"].split("__")[0] for r in md}):
        ps = [r for r in md if r["scenario"].startswith(dom + "__")]
        g = [cm[r["scenario"]]["F1_H"] - cm[r["scenario"]]["F1_Hr"] for r in ps if r["scenario"] in cm]
        lines.append(f"| {dom} | {len(ps)} | {np.median(fl(ps, 'ovl_true')):.2f} | {np.median(fl(ps, 'name_true')):.2f} | "
                     f"{np.median(fl(ps, 'diag_share')):.2f} | {np.median(fl(ps, 'num_share')):.2f} | {np.mean(g):+.3f} |")
    hd = props.get("hdxsm", [])
    if hd:
        top = sorted(hd, key=lambda r: -float(r["ovl_true"]) if r["ovl_true"] not in ("", "nan") else 1)[:8]
        lines += ["", "## HDXSM, the most split-like pairs (by ovl_true)", "", "| pair | ovl_true | name_true | diag | cols |", "|---|---|---|---|---|"]
        for r in top:
            lines.append(f"| {r['scenario']} | {float(r['ovl_true']):.2f} | {float(r['name_true']):.2f} | {float(r['diag_share']):.2f} | "
                         f"{r['src_cols']}×{r['tgt_cols']} |")
    # the corpus table for the paper (medians over scenarios; constant-matcher F1 as means, as in section 2)
    tab = []
    for c, split in TABLE:
        rs, ps = by_c.get(c, []), props.get(c, [])
        if not ps:
            continue
        med = lambda k: float(np.nanmedian(fl(ps, k))) if len(fl(ps, k)) else float("nan")  # noqa: E731
        tab.append({"corpus": c, "built_from_one_source": split, "scenarios": len(ps), "units": units(c, rs, ps),
                    "tables_src": med("src_tables"), "tables_tgt": med("tgt_tables"),
                    "cols_src": med("src_cols"), "cols_tgt": med("tgt_cols"),
                    "diag": med("diag_share"),
                    "const_F1_H": float(np.mean([r["F1_H"] for r in rs])) if rs else float("nan"),
                    "const_F1_Hr": float(np.mean([r["F1_Hr"] for r in rs])) if rs else float("nan"),
                    "ovl_true": med("ovl_true"), "ovl_non": med("ovl_non"),
                    "name_true": med("name_true"), "name_non": med("name_non"),
                    "num_share": med("num_share"), "fd_per_col": med("fd_per_col")})
    with (OUT / "corpus_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
