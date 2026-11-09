"""EXPLORATORY X88, held-out validation (2026-10-07): "stop SF as published after two updates". Pre-specified rule, test
and predictions in EXPLORATORY.md X88, written before any run; runs (iter_<stage>, collected and gated 12:56).

Statistic per corpus: per scenario the mean over the stage's initial matchers (controls and oracles excluded; a matcher is
included if it has t1, t2, t3, t5, t10 and the converged run on every scenario) of J_tf(t2) − J_tf(converged), macro over
groups. Interval: 95% bootstrap (10,000) over table-sharing clusters if there are at least 5, else over scenarios within
groups. Non-inferior if the lower bound > −0.01; superior if > 0. Validated if non-inferior on all four corpora.
Secondary: the same in expected Recall@GT and tie-free F1; share of matchers with t2 ≥ converged (J); iterations of the
converged runs (sf_stats.tsv); predictions 3 and 4.
Groups: T2D one; Lahman/MB by database × regime; Synthea pairs / whole schemas; indep2 by database.

    ../harness/.venv-bdi/Scripts/python.exe explore_x88_validate.py
Output: results/x88/{validate_rows.csv, validate.md}
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "reference", HERE / "layer3b"):
    sys.path.insert(0, str(p))

RES = HERE / "tuning" / "results"
GEN = HERE.parent / "harness" / "generated"
STAGES = {  # corpus: (stage, seed folder, root)
    "T2D web tables": ("e2_t2d", "t2d", GEN / "t2d-usable"),
    "Lahman/MusicBrainz single-table": ("e2_ctl", "ctl", GEN / "controlled-imported-v3"),
    "Synthea-OMOP": ("x54_synthea", "synthea", HERE / "data" / "synthea_plain"),
    "Lahman/MusicBrainz multi-table": ("x55_indep2", "indep2", HERE / "data" / "indep2_plain"),
}
TS = ("t1", "t2", "t3", "t5", "t10")
SEED = 20261088
MARGIN = -0.01


def group_of(corpus: str, ds: str, sc: str) -> str:
    if corpus.startswith("Lahman/MusicBrainz single"):
        return f"{ds.split('-')[0]}|{'plain' if sc.endswith('_plain') else 'recoded'}"
    if corpus.startswith("Synthea"):
        return ds
    if corpus.startswith("Lahman/MusicBrainz multi"):
        return ds.split("-")[0]
    return "all"


def side_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def job(args):
    corpus, ds, sc, matchers = args
    import score_layer1
    import score_seeds
    from explore_composite import load, sf_out
    from score_screen_metrics import all_metrics, permutations
    from smx.adapters import read_matrix
    stage, seedname, root = STAGES[corpus]
    g = score_seeds.ground_truth_file(root, ds, sc)
    f_d = HERE / "reference" / "seeds" / seedname / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not f_d.is_file():
        return []
    _, rows, cols = read_matrix(f_d)
    rows, cols = list(rows), list(cols)
    gt = read_matrix(g)[0] > 0.5
    if gt.shape != (len(rows), len(cols)) or not gt.any():
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    scen = root / ds / sc
    hs = side_hash(scen / "source") if (scen / "source").is_dir() else ""
    ht = side_hash(scen / "target") if (scen / "target").is_dir() else ""
    out = []
    for m in matchers:
        q = load(HERE / "reference" / "seeds" / seedname / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        row = {"corpus": corpus, "group": group_of(corpus, ds, sc), "dataset": ds, "scenario": sc, "matcher": m,
               "src_hash": hs, "tgt_hash": ht}
        r = all_metrics(np.clip(q, 0, None), gt, perms)
        row.update({"q_J": (r["R_exp"] + r["F1_Hr"]) / 2, "q_R": r["R_exp"], "q_F": r["F1_Hr"]})
        for t in (*TS, "conv"):
            d = RES / (stage if t == "conv" else f"iter_{stage}") / (f"{m}__melnik" if t == "conv" else f"{m}__melnik_{t}")
            rd = d / "merged" if (d / "merged").is_dir() else None
            S = load(sf_out(rd, ds, sc), rows, cols) if rd else None
            if S is None:
                continue
            r = all_metrics(np.clip(S, 0, None), gt, perms)
            row.update({f"{t}_J": (r["R_exp"] + r["F1_Hr"]) / 2, f"{t}_R": r["R_exp"], f"{t}_F": r["F1_Hr"]})
        out.append(row)
    return out


def components(pairs: list[tuple[str, str, str]]) -> dict[str, str]:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for key, a, b in pairs:
        for side in (a, b):
            ra, rb = find("K" + key), find("T" + side)
            if ra != rb:
                parent[ra] = rb
    return {key: find("K" + key) for key, _, _ in pairs}


def contrast(rows: list[dict], metric: str, rng) -> tuple[float, float, float, str]:
    """macro over groups of the per-scenario matcher-mean of metric(t2) − metric(conv); cluster or scenario bootstrap."""
    cell = defaultdict(list)
    meta = {}
    for r in rows:
        k = (r["group"], r["dataset"], r["scenario"])
        cell[k].append(r[f"t2_{metric}"] - r[f"conv_{metric}"])
        meta[k] = (r["src_hash"], r["tgt_hash"])
    per = {k: float(np.mean(v)) for k, v in cell.items()}
    groups = defaultdict(list)
    for k, v in per.items():
        groups[k[0]].append((k, v))
    est = float(np.mean([np.mean([v for _, v in g]) for g in groups.values()]))
    comp = components([("|".join(k), *meta[k]) for k in per])
    ncl = len(set(comp.values()))
    boots = []
    if ncl >= 5:
        # resample clusters; within the resample, macro over groups of the scenario means
        cl = defaultdict(list)
        for k, v in per.items():
            cl[comp["|".join(k)]].append((k[0], v))
        keys = sorted(cl)
        for _ in range(10_000):
            pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            gs = defaultdict(list)
            for c in pick:
                for g, v in cl[c]:
                    gs[g].append(v)
            boots.append(float(np.mean([np.mean(v) for v in gs.values()])))
        how = f"{ncl} table-sharing clusters"
    else:
        arrs = [np.array([v for _, v in g]) for g in groups.values()]
        boots = list(np.mean([[rng.choice(a, size=a.size).mean() for _ in range(10_000)] for a in arrs], axis=0))
        how = f"scenarios within groups ({ncl} clusters)"
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), how


def iterations(stage: str, matchers: list[str]) -> float:
    vals = []
    for m in matchers:
        f = RES / stage / f"{m}__melnik" / "merged" / "sf_stats.tsv"
        if not f.is_file():
            f = RES / stage / f"{m}__melnik" / "sf_stats.tsv"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines()[1:]:
                p = line.split("\t")
                if len(p) > 2:
                    try:
                        vals.append(float(p[2]))
                    except ValueError:
                        pass
    return float(np.median(vals)) if vals else float("nan")


def main() -> int:
    from score_r7 import is_control
    jobs = []
    for corpus, (stage, _, root) in STAGES.items():
        ms = sorted({d.name.split("__")[0] for d in (RES / f"iter_{stage}").iterdir() if d.name.endswith("__melnik_t2")})
        ms = [m for m in ms if not is_control(m)]
        jobs += [(corpus, d.name, s.name, ms) for d in sorted(x for x in root.iterdir() if x.is_dir())
                 for s in sorted(x for x in d.iterdir() if x.is_dir())]
    with Pool(10) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs, chunksize=2) for r in rr]
    out = HERE / "results" / "x88"
    out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "validate_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    need = [f"{t}_{x}" for t in (*TS, "conv") for x in ("J", "R", "F")]
    lines = ["# X88: held-out validation of \"stop SF as published after two updates\" (EXPLORATORY, pre-specified rule)", "",
             "t2 − converged, per scenario the mean over initial matchers, macro over groups; 95% bootstrap (clusters if ≥ 5).",
             "Non-inferior: lower bound > −0.01. Superior: lower bound > 0. Predictions: EXPLORATORY.md X88.", "",
             "| corpus | matchers | scenarios | J_tf t2 − conv [95%] | verdict | R_exp t2 − conv | F1_Hr t2 − conv | "
             "matchers t2 ≥ conv (J) | median iterations to converge | interval over |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    traj = ["", "## Trajectory (gain over q in R_exp / J_tf; macro over groups of scenario means)", "",
            "| corpus | t1 | t2 | t3 | t5 | t10 | conv |", "|---|---|---|---|---|---|---|"]
    verdicts = {}
    for corpus, (stage, _, _) in STAGES.items():
        rs = [r for r in rows if r["corpus"] == corpus]
        scen = {(r["dataset"], r["scenario"]) for r in rs}
        by = defaultdict(list)
        for r in rs:
            by[r["matcher"]].append(r)
        inc = [m for m, v in by.items() if {(r["dataset"], r["scenario"]) for r in v if all(k in r for k in need)} == scen]
        rs = [r for r in rs if r["matcher"] in inc]
        if not rs:
            lines.append(f"| {corpus} | 0 | {len(scen)} | – | – | – | – | – | – | – |")
            continue
        rng = np.random.default_rng(SEED)
        j = contrast(rs, "J", rng)
        rr = contrast(rs, "R", np.random.default_rng(SEED))
        ff = contrast(rs, "F", np.random.default_rng(SEED))
        verdict = "superior" if j[1] > 0 else ("non-inferior" if j[1] > MARGIN else "NOT non-inferior")
        verdicts[corpus] = verdict
        mm = defaultdict(list)
        for r in rs:
            mm[r["matcher"]].append(r["t2_J"] - r["conv_J"])
        share = np.mean([np.mean(v) >= 0 for v in mm.values()])
        lines.append(f"| {corpus} | {len(inc)} | {len(scen)} | {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}] | {verdict} | "
                     f"{rr[0]:+.4f} [{rr[1]:+.4f}, {rr[2]:+.4f}] | {ff[0]:+.4f} [{ff[1]:+.4f}, {ff[2]:+.4f}] | {share:.0%} | "
                     f"{iterations(stage, inc):.0f} | {j[3]} |")
        g = defaultdict(lambda: defaultdict(list))
        for r in rs:
            for t in (*TS, "conv"):
                g[r["group"]][t].append((r[f"{t}_R"] - r["q_R"], r[f"{t}_J"] - r["q_J"]))
        cells = []
        for t in (*TS, "conv"):
            vals = np.array([np.mean(np.array(v[t]), axis=0) for v in g.values()]).mean(axis=0)
            cells.append(f"{vals[0]:+.4f} / {vals[1]:+.4f}")
        traj.append(f"| {corpus} | " + " | ".join(cells) + " |")
    ok = all(v != "NOT non-inferior" for v in verdicts.values()) and len(verdicts) == len(STAGES)
    lines += ["", f"**Rule {'VALIDATED' if ok else 'NOT VALIDATED'}**: non-inferior on "
                  f"{sum(v != 'NOT non-inferior' for v in verdicts.values())} of {len(STAGES)} corpora "
                  f"(superior on {sum(v == 'superior' for v in verdicts.values())})."]
    lines += traj
    (out / "validate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
