"""EXPLORATORY X92 (2026-10-07): offline checks of two rules suggested by the X88/X91 mechanism. Design and predictions in
EXPLORATORY.md X92, written before computing. No new runs: the stopped outputs of X42, X82 and X88 are read (merged/ only).

Arms per scenario and initial matcher (SF as published = melnik):
  t2, t3        SF stopped after 2 / 3 updates
  conv          SF run to convergence
  avg23         (x_2 + x_3) / 2                         (R1, averaging)
  Ad            q+ + 5D, the frozen additive datatype term (R2, short-path evidence without a graph)
Contrasts in J_tf: C1 avg23 − conv, C2 avg23 − t2, C3 Ad − t2, C4 t2 − conv. Per scenario the mean over the initial matchers
that have every arm on every scenario of the corpus (controls and oracles excluded); macro over groups; 95% bootstrap over
clusters of scenarios sharing a table if there are at least 5 (Valentine: its source tables), else over scenarios within groups.

    ../harness/.venv-bdi/Scripts/python.exe explore_x92_rules.py [--corpora ...]
Output: results/x92/{rows.csv, report.md}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "reference", HERE / "layer3b", HERE.parent / "latex" / "sf-evaluation"):
    sys.path.insert(0, str(p))

RES = HERE / "tuning" / "results"
GEN = HERE.parent / "harness" / "generated"
SEED = 20261092
ARMS = ("t2", "t3", "conv", "avg23", "Ad")


def corpora():
    import score_seeds
    return {   # name: (seed folder, root, iteration stage, converged stage, grouping)
        "Valentine readable": ("plain", score_seeds.CORPUS["plain"], "iter_plain", "layer1_plain", "family"),
        "Valentine recoded": ("md5", score_seeds.CORPUS["md5"], "iter_md5", "layer1_md5", "family"),
        "Indep. multi-table (3 DBs)": ("indep", HERE / "data" / "indep_plain", "iter_indep", "layer1_indep", "dataset"),
        "New databases readable": ("final_controlled", GEN / "final-controlled", "iter_final_controlled", "final_controlled", "db_plain"),
        "New databases recoded": ("final_controlled", GEN / "final-controlled", "iter_final_controlled", "final_controlled", "db_opaque"),
        "Independent schema pairs": ("final_indep", HERE / "data" / "final_indep", "iter_final_indep", "final_indep", "domain"),
        "GDC-SM": ("gdcsm", GEN / "gdcsm-imported", "iter_x81_gdcsm", "x81_gdcsm", "one"),
        "HDXSM": ("hdxsm", GEN / "hdxsm-usable", "iter_x81_hdxsm", "x81_hdxsm", "one"),
        "GOBY": ("goby", GEN / "goby-imported", "iter_x81_goby", "x81_goby", "one"),
        "MaDI use cases": ("madi", GEN / "madi-imported", "iter_x81_madi", "x81_madi", "madi_indep"),
        "MaDI products": ("madi", GEN / "madi-imported", "iter_x81_madi", "x81_madi", "madi_products"),
        "T2D web tables": ("t2d", GEN / "t2d-usable", "iter_e2_t2d", "e2_t2d", "one"),
        "Lahman/MB single-table": ("ctl", GEN / "controlled-imported-v3", "iter_e2_ctl", "e2_ctl", "db_regime"),
        "Synthea-OMOP": ("synthea", HERE / "data" / "synthea_plain", "iter_x54_synthea", "x54_synthea", "dataset"),
        "Lahman/MB multi-table": ("indep2", HERE / "data" / "indep2_plain", "iter_x55_indep2", "x55_indep2", "db"),
    }


def group(rule: str, ds: str, sc: str) -> str | None:
    """the scenario's group, or None if the scenario does not belong to this corpus."""
    import gen_figures as gf
    if rule == "family":
        return gf.family_of(ds, sc)
    if rule in ("db_plain", "db_opaque"):
        plain = sc.endswith("_plain")
        return ds.split("-")[0] if plain == (rule == "db_plain") else None
    if rule == "db_regime":
        return f"{ds.split('-')[0]}|{'plain' if sc.endswith('_plain') else 'recoded'}"
    if rule == "db":
        return ds.split("-")[0]
    if rule == "domain":
        return ds.split("_")[0]
    if rule in ("madi_indep", "madi_products"):
        prod = sc.startswith("products__")
        return sc.split("__")[0] if prod == (rule == "madi_products") else None
    if rule == "dataset":
        return ds
    return "all"


def side_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def job(args):
    corpus, ds, sc, grp, matchers = args
    import score_seeds
    from explore_composite import load, sf_out
    from score_screen_metrics import all_metrics, permutations
    from smx.adapters import read_matrix
    seedname, root, it_stage, conv_stage, _ = corpora()[corpus]
    g = score_seeds.ground_truth_file(root, ds, sc)
    f_d = HERE / "reference" / "seeds" / seedname / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols, D = list(rows), list(cols), np.nan_to_num(D)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != D.shape or not gt.any():
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    scen = root / ds / sc
    hs = side_hash(scen / "source") if (scen / "source").is_dir() else ""
    ht = side_hash(scen / "target") if (scen / "target").is_dir() else ""
    J = lambda M: (lambda r: (r["R_exp"] + r["F1_Hr"]) / 2)(all_metrics(np.clip(M, 0, None), gt, perms))  # noqa: E731
    out = []
    for m in matchers:
        q = load(HERE / "reference" / "seeds" / seedname / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        mats = {}
        for arm, stage, label in (("t2", it_stage, f"{m}__melnik_t2"), ("t3", it_stage, f"{m}__melnik_t3"),
                                  ("conv", conv_stage, f"{m}__melnik")):
            d = RES / stage / label
            rd = d / "merged" if (d / "merged").is_dir() else None
            S = load(sf_out(rd, ds, sc), rows, cols) if rd else None
            if S is not None:
                mats[arm] = S
        if not all(a in mats for a in ("t2", "t3", "conv")):
            continue
        mats["avg23"] = (mats["t2"] + mats["t3"]) / 2
        mats["Ad"] = np.clip(q, 0, None) + 5.0 * D
        row = {"corpus": corpus, "group": grp, "dataset": ds, "scenario": sc, "matcher": m, "src_hash": hs, "tgt_hash": ht}
        row.update({a: J(M) for a, M in mats.items()})
        out.append(row)
    return out


def valentine_sources() -> dict[tuple[str, str], str]:
    out = {}
    with (HERE / "results" / "benchmark_properties.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["corpus"].startswith("valentine"):
                out[(r["dataset"], r["scenario"])] = r["source_id"]
    return out


def clusters(rows: list[dict], vsrc) -> dict[tuple, str]:
    keys = {(r["group"], r["dataset"], r["scenario"]): (r["src_hash"], r["tgt_hash"]) for r in rows}
    if rows and rows[0]["corpus"].startswith("Valentine"):
        return {k: vsrc.get((k[1], k[2]), k[1]) for k in keys}
    c86 = {"GDC-SM": "gdcsm", "HDXSM": "hdxsm", "GOBY": "goby", "MaDI use cases": "madi_indep", "MaDI products": "madi_products"}
    if rows and rows[0]["corpus"] in c86:   # the paper's clusters (X86 cluster_map), so intervals match the text's
        from x86_cluster_ci import cluster_map
        m, _ = cluster_map(c86[rows[0]["corpus"]], sorted({k[2] for k in keys}))
        return {k: m[k[2]] for k in keys}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k, (a, b) in keys.items():
        for side in (a, b):
            ra, rb = find(("K",) + k), find(("T", side))
            if ra != rb:
                parent[ra] = rb
    return {k: find(("K",) + k) for k in keys}


def contrast(rows, x, y, cl, rng):
    cell = defaultdict(list)
    for r in rows:
        cell[(r["group"], r["dataset"], r["scenario"])].append(r[x] - r[y])
    per = {k: float(np.mean(v)) for k, v in cell.items()}
    groups = defaultdict(list)
    for k, v in per.items():
        groups[k[0]].append(v)
    est = float(np.mean([np.mean(v) for v in groups.values()]))
    ncl = len({cl[k] for k in per})
    if ncl >= 5:
        byc = defaultdict(list)
        for k, v in per.items():
            byc[cl[k]].append((k[0], v))
        ks = sorted(byc)
        boots = []
        for _ in range(10_000):
            gs = defaultdict(list)
            for c in (ks[i] for i in rng.integers(0, len(ks), len(ks))):
                for g, v in byc[c]:
                    gs[g].append(v)
            boots.append(float(np.mean([np.mean(v) for v in gs.values()])))
        how = f"{ncl} clusters"
    else:
        arrs = [np.array(v) for v in groups.values()]
        boots = np.mean([[rng.choice(a, size=a.size).mean() for _ in range(10_000)] for a in arrs], axis=0)
        how = f"scenarios ({ncl} clusters)"
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), how


def main() -> int:
    from score_r7 import is_control
    import gen_figures as gf
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="")
    a = ap.parse_args()
    cs = corpora()
    names = [c for c in cs if not a.corpora or c in a.corpora.split(",")]
    jobs = []
    for c in names:
        seedname, root, it_stage, _, rule = cs[c]
        if not (RES / it_stage).is_dir():
            print(f"skip {c}: no {it_stage}")
            continue
        ms = sorted({d.name.split("__")[0] for d in (RES / it_stage).iterdir() if d.name.endswith("__melnik_t2")})
        ms = [m for m in ms if not is_control(m)]
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            for s in sorted(x for x in d.iterdir() if x.is_dir()):
                g = group(rule, d.name, s.name)
                if g is not None:
                    jobs.append((c, d.name, s.name, g, ms))
    with Pool(10) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs, chunksize=2) for r in rr]
    out = HERE / "results" / "x92"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    vsrc = valentine_sources()
    lines = ["# X92: averaging consecutive updates (R1) and short-path evidence only (R2), offline (EXPLORATORY)", "",
             "J_tf contrasts; per scenario the mean over initial matchers, macro over groups; 95% bootstrap over clusters if >= 5 (Valentine: source tables; GDC-SM, HDXSM, GOBY, MaDI: C's X86 clusters; others: tables shared by content).",
             "C1 avg(2,3) − converged; C2 avg(2,3) − t2; C3 A_d − t2; C4 t2 − converged. Predictions: EXPLORATORY.md X92.", "",
             "| corpus | matchers | scenarios | C1 avg23 − conv | C2 avg23 − t2 | C3 A_d − t2 | C4 t2 − conv | interval over |",
             "|---|---|---|---|---|---|---|---|"]
    for c in names:
        rs = [r for r in rows if r["corpus"] == c]
        if not rs:
            continue
        if cs[c][4] == "family":   # the paper's rule: families with at least MIN_FAMILY scenarios
            size = defaultdict(set)
            for r in rs:
                size[r["group"]].add((r["dataset"], r["scenario"]))
            rs = [r for r in rs if len(size[r["group"]]) >= gf.MIN_FAMILY]
        scen = {(r["dataset"], r["scenario"]) for r in rs}
        by = defaultdict(set)
        for r in rs:
            by[r["matcher"]].add((r["dataset"], r["scenario"]))
        inc = {m for m, v in by.items() if v == scen}
        rs = [r for r in rs if r["matcher"] in inc]
        if not rs:
            continue
        cl = clusters(rs, vsrc)
        res = [contrast(rs, x, y, cl, np.random.default_rng(SEED))
               for x, y in (("avg23", "conv"), ("avg23", "t2"), ("Ad", "t2"), ("t2", "conv"))]
        fmt = lambda e: f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}]"  # noqa: E731
        lines.append(f"| {c} | {len(inc)} | {len(scen)} | " + " | ".join(fmt(e) for e in res) + f" | {res[0][3]} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
