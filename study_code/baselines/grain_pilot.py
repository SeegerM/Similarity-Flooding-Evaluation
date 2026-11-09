"""The grain pilot, GRAIN_PILOT.md (registered 1e8be7b, amendment A1).

Subcommands run in registered order and refuse to run out of it:

    python grain_pilot.py manifest   # cohorts, table digests, benefit files, code identity
    python grain_pilot.py infer      # G1/G2 + sensitivity grid on every table (cached)
    python grain_pilot.py check      # P0: inferred vs true grain, reported before accuracy
    python grain_pilot.py k1         # P1: grain against the already-measured benefits

P2-P4 (decomposition, ceiling, SF arms) are added as later subcommands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pickle
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

REPORTS = HERE / "reports"
CACHE = REPORTS / "grain_cache"
MANIFEST = REPORTS / "grain_manifest.json"
TABLES = REPORTS / "grain_tables.json"
GRID_CSV = REPORTS / "grain_grid.csv"
P0_CSV, P0_JSON = REPORTS / "grain_p0.csv", REPORTS / "grain_p0.json"
K1_CSV, K1_JSON = REPORTS / "grain_k1_scenarios.csv", REPORTS / "grain_k1.json"
ANNOTATIONS = HERE / "grain_annotations"

GRID = list(itertools.product((0.25, 0.50, 0.75), (0.80, 0.90, 0.95), (0.80, 0.90, 0.95)))
REGISTERED = (0.50, 0.90, 0.90)
BENEFITS = {
    "B_ORA": REPORTS / "transferability_table_hellinger_r44.csv",
    "B_SF": REPORTS / "sf_constant_zero.csv",
    "B_LAD": REPORTS / "baseline_ladder.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


# ------------------------------------------------------------------ cohorts
def composites_from_profile(profiling: Path, side: str, header: list[str]):
    """Arity-2 LHS sets of the cached FD profile, as column indices."""
    if not profiling.exists():
        return None
    meta = json.loads(profiling.read_text(encoding="utf-8"))
    entry = meta.get(side, {}).get("fd", {})
    path = Path(entry.get("path", ""))
    if entry.get("status") != "complete" or not path.exists():
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    ids = {c["id"]: c["index"] for c in profile["spec"]["columns"]}
    labels = [c["label"] for c in sorted(profile["spec"]["columns"], key=lambda c: c["index"])]
    if labels != header:
        raise ValueError(f"{profiling}: profile columns do not match the {side} header")
    return sorted({tuple(sorted(ids[x] for x in r["lhs"])) for r in profile["records"]
                   if len(r["lhs"]) == 2})


def header_of(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        return [x.strip() for x in next(csv.reader(fh))]


def cohort_cval():
    out = []
    for manifest in sorted((HERE / "generated" / "benchmark-full").glob("*/*/manifest.json")):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if info["family"] == "GDC":
            continue
        folder = manifest.parent
        sides = {}
        for side in ("source", "target"):
            path = folder / info["tables"][side]["path"]
            sides[side] = [{"table": side, "path": rel(path), "sha256": info["tables"][side]["sha256"],
                            "profiling": rel(folder / "metadata" / "profiling.json")}]
        out.append({"cohort": "C-VAL", "scenario": info["id"], "family": info["family"],
                    "scenario_type": info["scenario_type"], "regime": "plain", "sides": sides})
    return out


def cohort_annotated(name: str):
    data = json.loads((ANNOTATIONS / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for key, info in sorted(data["scenarios"].items()):
        folder = ROOT / info["path"]
        sides = {}
        for side in ("source", "target"):
            tables = []
            for path in sorted((folder / side).glob("*.csv")):
                entry = {"table": path.stem, "path": rel(path), "sha256": sha256(path)}
                if name == "c_ext":
                    entry["profiling"] = rel(folder / "metadata" / "profiling.json")
                tables.append(entry)
            sides[side] = tables
        out.append({"cohort": data["cohort"], "scenario": key, "regime": info["regime"],
                    "family": info.get("dataset", key.split("/")[0]),
                    "normalization": info.get("normalization"),
                    "excluded": info.get("excluded"), "sides": sides})
    return out


def cmd_manifest(args):
    if MANIFEST.exists() and not args.force:
        raise SystemExit(f"{MANIFEST.name} exists; the manifest is written once (use --force to rebuild)")
    if STAGE == "conf":
        scenarios = cohort_annotated("c_conf")
    else:
        scenarios = cohort_cval() + cohort_annotated("c_ext") + cohort_annotated("c_sak") \
            + cohort_annotated("c_den")
    for entry in scenarios:
        if entry["cohort"] != "C-VAL":
            continue
        for tables in entry["sides"].values():
            for t in tables:
                if sha256(ROOT / t["path"]) != t["sha256"]:
                    raise SystemExit(f"{t['path']}: digest differs from its manifest")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    manifest = {
        "preregistration": {"file": "harness/GRAIN_PILOT.md", "commit": "1e8be7b", "amendments": ["A1"]},
        "code_commit": commit, "registered_thresholds": dict(zip(("mu_min", "g3n_min", "nearkey_max"), REGISTERED)),
        "grid": GRID,
        "benefit_files": {k: {"path": rel(v), "sha256": sha256(v)} for k, v in BENEFITS.items()},
        "annotation_files": {p.name: sha256(p) for p in sorted(ANNOTATIONS.glob("*.json"))},
        "counts": {c: sum(s["cohort"] == c for s in scenarios)
                   for c in ("C-VAL", "C-EXT", "C-SAK", "C-DEN", "C-CONF")},
        "scenarios": scenarios,
    }
    REPORTS.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.name}: {manifest['counts']}")


# ------------------------------------------------------------------ inference
def _measure(job):
    """Worker: measure one table file, cached by content digest and composite set."""
    from encrypted_feature_ablation import read_table
    from smx.grain import measure_table
    path, digest, composites = job
    tag = hashlib.sha256(json.dumps(composites).encode()).hexdigest()[:12] if composites else "g1"
    target = CACHE / f"{digest}_{tag}.pkl"
    if target.exists():
        return str(target)
    header, columns = read_table(ROOT / path)
    measures = measure_table(columns, composites)
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump({"header": header, "measures": measures}, fh)
    os.replace(tmp, target)
    return str(target)


def cmd_infer(args):
    from smx.grain import grain_from_measures
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = {}
    plan = []
    for entry in manifest["scenarios"]:
        for side, tables in entry["sides"].items():
            for t in tables:
                header = header_of(ROOT / t["path"])
                comps = None
                if t.get("profiling"):
                    comps = composites_from_profile(ROOT / t["profiling"], side, header)
                comps = [list(c) for c in comps] if comps else None
                job = (t["path"], t["sha256"], comps)
                jobs[json.dumps(job)] = job
                plan.append((entry, side, t, json.dumps(job), comps is not None))
    print(f"{len(plan)} tables, {len(jobs)} unique measurement jobs, {args.workers} workers", flush=True)
    done = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (key, result) in enumerate(zip(jobs, pool.map(_measure, jobs.values())), 1):
            done[key] = result
            if i % 50 == 0 or i == len(jobs):
                print(f"  measured {i}/{len(jobs)}", flush=True)

    tables_out, grid_rows = [], []
    for entry, side, t, key, has_g2 in plan:
        with open(done[key], "rb") as fh:
            cached = pickle.load(fh)
        m = cached["measures"]
        record = {"cohort": entry["cohort"], "scenario": entry["scenario"], "side": side,
                  "table": t["table"], "header": cached["header"], "rows": m.rows}
        for rule, use in (("G1", False), ("G2", True)):
            if use and not has_g2:
                continue
            g = grain_from_measures(m, use_composites=use)
            record[rule] = {"homes": [g.class_key(i) for i in range(len(m.distinct))],
                            "summary": g.summary(),
                            "anchors": ["+".join(map(str, a)) for a in g.anchors]}
            for mu, g3, nk in GRID:
                gg = grain_from_measures(m, use_composites=use, mu_min=mu, g3n_min=g3, nearkey_max=nk)
                s = gg.summary()
                grid_rows.append({"cohort": entry["cohort"], "scenario": entry["scenario"],
                                  "side": side, "table": t["table"], "rule": rule, "mu_min": mu,
                                  "g3n_min": g3, "nearkey_max": nk, "G": s["G"], "G_eff": s["G_eff"],
                                  "spread": round(s["spread"], 6), "W": s["W"],
                                  "homes": "|".join(gg.class_key(i) for i in range(len(m.distinct)))})
        tables_out.append(record)
    TABLES.write_text(json.dumps({"manifest_sha256": sha256(MANIFEST), "tables": tables_out},
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    with GRID_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grid_rows[0]))
        w.writeheader()
        w.writerows(grid_rows)
    print(f"wrote {TABLES.name} ({len(tables_out)} tables) and {GRID_CSV.name} ({len(grid_rows)} rows)")


# ------------------------------------------------------------------ P0
_ANNOTATION_CACHE: dict = {}


def truth_for(cohort_file: str, scenario: str, side: str, table: str, header: list[str]):
    if cohort_file not in _ANNOTATION_CACHE:
        _ANNOTATION_CACHE[cohort_file] = json.loads(
            (ANNOTATIONS / f"{cohort_file}.json").read_text(encoding="utf-8"))["scenarios"]
    info = _ANNOTATION_CACHE[cohort_file][scenario]
    labels = info[side]
    table_labels = labels[table] if table in labels else next(iter(labels.values()))
    return [table_labels.get(h) for h in header], info.get("excluded")


def side_ari(records, rule, cohort_file, scenario, side, homes_override=None):
    from smx.grain import adjusted_rand_index
    pred, role, entity = [], [], []
    excluded = None
    for r in records:
        truth, excluded = truth_for(cohort_file, scenario, side, r["table"], r["header"])
        homes = homes_override[r["table"]] if homes_override else r[rule]["homes"]
        for label, home in zip(truth, homes):
            if label is None or label["entity"] == "unknown":
                continue
            pred.append((r["table"], home))
            role.append(tuple(label["role"]))
            entity.append(label["entity"])
    if len(pred) < 2:
        return None, None, len(pred), excluded
    return adjusted_rand_index(role, pred), adjusted_rand_index(entity, pred), len(pred), excluded


def load_tables():
    if not TABLES.exists():
        raise SystemExit("run `infer` first")
    data = json.loads(TABLES.read_text(encoding="utf-8"))
    if data["manifest_sha256"] != sha256(MANIFEST):
        raise SystemExit("grain_tables.json was built from a different manifest")
    grouped = {}
    for r in data["tables"]:
        grouped.setdefault((r["cohort"], r["scenario"], r["side"]), []).append(r)
    return grouped


def cmd_check(args):
    grouped = load_tables()
    files = dict(STEP3_COHORTS)
    rows = []
    grid = {}
    with GRID_CSV.open(encoding="utf-8") as fh:
        for g in csv.DictReader(fh):
            if g["cohort"] in files and g["rule"] == "G1":
                grid.setdefault((g["cohort"], g["scenario"], g["side"], g["mu_min"], g["g3n_min"],
                                 g["nearkey_max"]), {})[g["table"]] = g["homes"].split("|")
    for (cohort, scenario, side), records in sorted(grouped.items()):
        if cohort not in files:
            continue
        for rule in ("G1", "G2"):
            if rule not in records[0]:
                continue
            role_ari, entity_ari, n, excluded = side_ari(records, rule, files[cohort], scenario, side)
            rows.append({"cohort": cohort, "scenario": scenario, "side": side, "rule": rule,
                         "mu_min": REGISTERED[0], "g3n_min": REGISTERED[1], "nearkey_max": REGISTERED[2],
                         "columns": n, "ari_role": role_ari, "ari_entity": entity_ari,
                         "excluded": excluded or ""})
        for mu, g3, nk in GRID:
            homes = grid[(cohort, scenario, side, str(mu), str(g3), str(nk))]
            role_ari, entity_ari, n, excluded = side_ari(records, "G1", files[cohort], scenario, side, homes)
            rows.append({"cohort": cohort, "scenario": scenario, "side": side, "rule": "G1-grid",
                         "mu_min": mu, "g3n_min": g3, "nearkey_max": nk, "columns": n,
                         "ari_role": role_ari, "ari_entity": entity_ari, "excluded": excluded or ""})
    with P0_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def med(select):
        vals = [r["ari_role"] for r in rows if select(r) and r["ari_role"] is not None and not r["excluded"]]
        return {"n_sides": len(vals), "median": statistics.median(vals) if vals else None,
                "min": min(vals) if vals else None, "max": max(vals) if vals else None}

    registered = lambda r: r["rule"] == "G1"  # noqa: E731
    summary = {
        "primary": "ARI of inferred (table, class) vs true (table, entity), G1 registered thresholds",
        "pooled_C-EXT+C-SAK": med(lambda r: registered(r) and r["cohort"] in ("C-EXT", "C-SAK", "C-CONF")),
        "by_cohort": {c: med(lambda r, c=c: registered(r) and r["cohort"] == c) for c in files},
        "by_cohort_regime": {f"{c}/{g}": med(lambda r, c=c, g=g: registered(r) and r["cohort"] == c
                                            and _regime(grouped, r) == g)
                             for c in files if c != "C-DEN" for g in _regimes(grouped, c)},
        "G2_C-EXT": med(lambda r: r["rule"] == "G2" and r["cohort"] == "C-EXT"),
        "entity_level_pooled": _median_entity(rows),
        "grid_pooled_medians (reported, never selected)": {
            f"mu{mu}_g3n{g3}_nk{nk}": med(lambda r, mu=mu, g3=g3, nk=nk: r["rule"] == "G1-grid"
                                         and r["cohort"] in ("C-EXT", "C-SAK")
                                         and (r["mu_min"], r["g3n_min"], r["nearkey_max"]) == (mu, g3, nk))["median"]
            for mu, g3, nk in GRID},
    }
    pooled = summary["pooled_C-EXT+C-SAK"]["median"]
    summary["verdict"] = ("adequate" if pooled is not None and pooled >= 0.5 else
                          "INADEQUATE: A2, A3 and K1 are reported but flagged; no rule revision here")
    P0_JSON.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("pooled_C-EXT+C-SAK", "by_cohort", "by_cohort_regime",
                                               "G2_C-EXT", "entity_level_pooled", "verdict")}, indent=1))


def _regime(grouped, row):
    manifest = _manifest_index()
    return manifest[(row["cohort"], row["scenario"])]["regime"]


def _regimes(grouped, cohort):
    return sorted({v["regime"] for (c, _), v in _manifest_index().items() if c == cohort})


_MANIFEST_INDEX = None


def _manifest_index():
    global _MANIFEST_INDEX
    if _MANIFEST_INDEX is None:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        _MANIFEST_INDEX = {(s["cohort"], s["scenario"]): s for s in data["scenarios"]}
    return _MANIFEST_INDEX


def _median_entity(rows):
    vals = [r["ari_entity"] for r in rows if r["rule"] == "G1" and r["cohort"] in ("C-EXT", "C-SAK")
            and r["ari_entity"] is not None and not r["excluded"]]
    return {"n_sides": len(vals), "median": statistics.median(vals) if vals else None}


# ------------------------------------------------------------------ P1 / K1
def read_benefits():
    ora = {}
    with BENEFITS["B_ORA"].open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["cls"] == "FD" and r["extractor"] == "mutual_top1":
                ora.setdefault(r["id"], {})[r["arm"]] = float(r["f1"])
    sf = {}
    with BENEFITS["B_SF"].open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["regime"] == "plain":
                sf.setdefault(r["id"], {})[r["arm"]] = float(r["f1_mutual"])
    lad = {}
    with BENEFITS["B_LAD"].open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["extractor"] == "mutual_top1":
                lad.setdefault(r["id"], {})[r["rung"][:1]] = float(r["f1"])
    out = {}
    for sid, arms in ora.items():
        if {"oracle", "none"} <= arms.keys():
            out.setdefault(sid, {})["B_ORA"] = arms["oracle"] - arms["none"]
    for sid, arms in sf.items():
        if {"constant_005", "constant_000"} <= arms.keys():
            out.setdefault(sid, {})["B_SF5"] = arms["constant_005"] - arms["constant_000"]
        if {"baseline", "constant_000"} <= arms.keys():
            out.setdefault(sid, {})["B_SF1"] = arms["baseline"] - arms["constant_000"]
    for sid, rungs in lad.items():
        if {"3", "4"} <= rungs.keys():
            out.setdefault(sid, {})["B_LAD"] = rungs["4"] - rungs["3"]
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def cmd_k1(args):
    if not P0_JSON.exists():
        raise SystemExit("P0 (`check`) must be reported before K1")
    grouped = load_tables()
    index = _manifest_index()
    benefits = read_benefits()
    predictors = {}
    for (cohort, scenario, side), records in grouped.items():
        if cohort != "C-VAL":
            continue
        s = records[0]["G1"]["summary"]
        s2 = records[0].get("G2", {}).get("summary")
        slot = predictors.setdefault(scenario, {})
        slot[side] = (s, s2)
    rows = []
    for scenario, sides in sorted(predictors.items()):
        if set(sides) != {"source", "target"}:
            continue
        (ss, ss2), (ts, ts2) = sides["source"], sides["target"]
        row = {"scenario": scenario, "family": index[("C-VAL", scenario)]["family"],
               "G_eff_min": min(ss["G_eff"], ts["G_eff"]), "W_min": min(ss["W"], ts["W"]),
               "spread_min": min(ss["spread"], ts["spread"]),
               "G_min": min(ss["G"], ts["G"]),
               "G2_G_eff_min": min(ss2["G_eff"], ts2["G_eff"]) if ss2 and ts2 else None}
        row.update(benefits.get(scenario, {}))
        rows.append(row)
    with K1_CSV.open("w", newline="", encoding="utf-8") as fh:
        fields = ["scenario", "family", "G_min", "G_eff_min", "W_min", "spread_min", "G2_G_eff_min",
                  "B_ORA", "B_SF5", "B_SF1", "B_LAD"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    outcomes = ("B_ORA", "B_SF5", "B_SF1", "B_LAD")
    result = {"n_scenarios": len(rows),
              "distribution_G_eff_min": _counts(r["G_eff_min"] for r in rows),
              "distribution_G2_G_eff_min": _counts(r["G2_G_eff_min"] for r in rows),
              "by_family_G_eff_min": {f: _counts(r["G_eff_min"] for r in rows if r["family"] == f)
                                      for f in sorted({r["family"] for r in rows})},
              "spearman": {}, "groups": {}, "within_family_spearman": {}}
    for predictor in ("G_eff_min", "W_min", "spread_min", "G2_G_eff_min"):
        result["spearman"][predictor] = {}
        for o in outcomes:
            pairs = [(r[predictor], r[o]) for r in rows if r.get(o) is not None and r[predictor] is not None]
            result["spearman"][predictor][o] = {"n": len(pairs),
                                                "rho": spearman([p for p, _ in pairs], [q for _, q in pairs])}
    for o in outcomes:
        multi = [r[o] for r in rows if r.get(o) is not None and r["G_eff_min"] >= 2]
        single = [r[o] for r in rows if r.get(o) is not None and r["G_eff_min"] < 2]
        result["groups"][o] = {"multi_n": len(multi), "single_n": len(single),
                               "multi_mean": statistics.fmean(multi) if multi else None,
                               "single_mean": statistics.fmean(single) if single else None}
        result["within_family_spearman"][o] = {}
        for f in sorted({r["family"] for r in rows}):
            pairs = [(r["G_eff_min"], r[o]) for r in rows if r["family"] == f and r.get(o) is not None]
            result["within_family_spearman"][o][f] = {"n": len(pairs), "rho": spearman(
                [p for p, _ in pairs], [q for _, q in pairs])}
    g = result["groups"]["B_ORA"]
    evaluable = g["multi_n"] >= 20 and g["single_n"] >= 20
    rhos = [result["spearman"]["G_eff_min"][o]["rho"] for o in ("B_ORA", "B_SF5", "B_LAD")]
    passing = sum(r is not None and r >= 0.15 for r in rhos)
    gap = (g["multi_mean"] - g["single_mean"]) if evaluable else None
    result["K1"] = {"evaluable": evaluable, "rho_ge_0.15_count": passing, "B_ORA_group_gap": gap,
                    "verdict": ("not evaluable (a group has < 20 scenarios)" if not evaluable else
                                "PASS" if passing >= 2 and gap >= 0.01 else "FAIL")}
    p0 = json.loads(P0_JSON.read_text(encoding="utf-8"))
    result["P0_verdict"] = p0["verdict"]
    K1_JSON.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("n_scenarios", "distribution_G_eff_min", "by_family_G_eff_min",
                                             "spearman", "groups", "K1", "P0_verdict")}, indent=1))


# ------------------------------------------------------------------ step 3: matrices, P2 (K2), P3 (K3)
MATRICES = REPORTS / "grain_matrices"
MATRIX_MANIFEST = REPORTS / "grain_matrix_manifest.json"
P2_CSV, P2_JSON = REPORTS / "grain_k2_scenarios.csv", REPORTS / "grain_k2.json"
P3_CSV, P3_JSON = REPORTS / "grain_k3_scenarios.csv", REPORTS / "grain_k3.json"
STEP3_COHORTS = {"C-EXT": "c_ext", "C-SAK": "c_sak", "C-DEN": "c_den"}
CEXT_CASES = REPORTS / "controlled_matchers_v3_20260905"
EPS = 1e-9


def safe(name: str) -> str:
    return name.replace("/", "__")


def axis(tables):
    """Axis labels `table.column` for one side, per amendment A2a."""
    labels = []
    for t in tables:
        labels.extend(f"{t['table']}.{c}" for c in header_of(ROOT / t["path"]))
    return labels


def ground_truth(entry, s_labels, t_labels):
    folder = (ROOT / entry["sides"]["source"][0]["path"]).parent.parent
    si = {l: i for i, l in enumerate(s_labels)}
    ti = {l: j for j, l in enumerate(t_labels)}
    gt = np.zeros((len(s_labels), len(t_labels)), dtype=bool)
    for path in sorted((folder / "ground_truth").glob("*.csv")):
        s_table, t_table = path.stem.split("___")
        s_cols = header_of(ROOT / next(t["path"] for t in entry["sides"]["source"] if t["table"] == s_table))
        t_cols = header_of(ROOT / next(t["path"] for t in entry["sides"]["target"] if t["table"] == t_table))
        with path.open(newline="", encoding="utf-8-sig") as fh:
            grid = [[float(v) for v in row] for row in csv.reader(fh) if row]
        if len(grid) != len(s_cols) or any(len(r) != len(t_cols) for r in grid):
            raise ValueError(f"{path}: grid shape does not match the table headers")
        for a, row in enumerate(grid):
            for b, v in enumerate(row):
                if v > 0.5:
                    gt[si[f"{s_table}.{s_cols[a]}"], ti[f"{t_table}.{t_cols[b]}"]] = True
    return gt


def cached_cext_arms(entry, s_labels, t_labels):
    """Every cached C-EXT matrix, aligned exactly to the header axes (A2a)."""
    from smx.adapters import read_matrix
    with (CEXT_CASES / "cases.csv").open(encoding="utf-8") as fh:
        index = {r["id"]: r["key"] for r in csv.DictReader(fh)}
    scenario = entry["scenario"].split("/")[1]
    sid = next(k for k in index if k.endswith(f"/{scenario}/{scenario}"))
    folder = CEXT_CASES / "cases" / index[sid]
    out = {}
    for path in sorted(folder.glob("*.csv")):
        S, rows, cols = read_matrix(path)
        if rows is None or cols is None:
            raise ValueError(f"{path}: cached matrix has no labels")
        if len(rows) != len(s_labels) or len(cols) != len(t_labels):
            raise ValueError(f"{path}: axis sizes differ")
        rpos = {l: i for i, l in enumerate(rows)}
        cpos = {l: j for j, l in enumerate(cols)}
        try:
            ri = [rpos[l] for l in s_labels]
            ci = [cpos[l] for l in t_labels]
        except KeyError as exc:
            raise ValueError(f"{path}: label {exc} does not align (fail closed)")
        out[path.stem] = S[np.ix_(ri, ci)]
    return out


def computed_arms(entry, s_labels, t_labels):
    from encrypted_feature_ablation import pmf_scores, read_table, scale_aligned_fuse
    from regime_tables import label_matrix
    from best_combination import char3_channel
    cols = {}
    for side in ("source", "target"):
        cols[side] = []
        for t in entry["sides"][side]:
            _, columns = read_table(ROOT / t["path"])
            cols[side].extend(columns)
    out = {"char3": char3_channel(cols["source"], cols["target"], 20260903)}
    if entry["cohort"] != "C-EXT":
        bare = [[l.split(".", 1)[1] for l in labels] for labels in (s_labels, t_labels)]
        lev = np.asarray(label_matrix(*bare), dtype=float)
        pmf = pmf_scores(cols["source"], cols["target"])
        out.update({"levenshtein": lev, "pmf": pmf, "seed": scale_aligned_fuse(lev, pmf, 0.35)})
    return out


def cmd_matrices(args):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [s for s in manifest["scenarios"] if s["cohort"] in STEP3_COHORTS]
    rows = []
    for entry in entries:
        s_labels = axis(entry["sides"]["source"])
        t_labels = axis(entry["sides"]["target"])
        gt = ground_truth(entry, s_labels, t_labels)
        folder = MATRICES / entry["cohort"] / safe(entry["scenario"])
        folder.mkdir(parents=True, exist_ok=True)
        arms = computed_arms(entry, s_labels, t_labels)
        if entry["cohort"] == "C-EXT":
            arms.update(cached_cext_arms(entry, s_labels, t_labels))
        for name, S in arms.items():
            np.savez(folder / f"{name}.npz", S=np.asarray(S, dtype=float),
                     source_labels=np.array(s_labels, dtype=object),
                     target_labels=np.array(t_labels, dtype=object))
        np.save(folder / "gt.npy", gt)
        magneto = folder / "magneto.npz"
        if magneto.exists():
            m = np.load(magneto, allow_pickle=True)
            if list(m["source_labels"]) != s_labels or list(m["target_labels"]) != t_labels:
                raise ValueError(f"{magneto}: Magneto axes differ from the manifest axes")
            arms["magneto"] = m["S"]
        for name in sorted(arms):
            f = folder / f"{name}.npz"
            rows.append({"cohort": entry["cohort"], "scenario": entry["scenario"], "arm": name,
                         "file": rel(f), "sha256": sha256(f), "shape": list(np.asarray(arms[name]).shape),
                         "gt_pairs": int(gt.sum())})
        print(f"{entry['cohort']} {entry['scenario']}: {len(arms)} arms", flush=True)
    counts = {}
    for r in rows:
        counts.setdefault(r["cohort"], {}).setdefault(r["arm"], 0)
        counts[r["cohort"]][r["arm"]] += 1
    MATRIX_MANIFEST.write_text(json.dumps({"grain_manifest_sha256": sha256(MANIFEST),
                                           "arms_per_cohort": counts, "matrices": rows}, indent=1) + "\n",
                               encoding="utf-8")
    print(json.dumps(counts, indent=1))


_TABLE_IDX = None


def _table_index():
    global _TABLE_IDX
    if _TABLE_IDX is None:
        _TABLE_IDX = load_tables()
    return _TABLE_IDX


def _classes(entry, labels, side, rule="G1"):
    """True entity, true role, and inferred (table, home) per axis label."""
    name = STEP3_COHORTS[entry["cohort"]]
    if name not in _ANNOTATION_CACHE:
        _ANNOTATION_CACHE[name] = json.loads(
            (ANNOTATIONS / f"{name}.json").read_text(encoding="utf-8"))["scenarios"]
    info = _ANNOTATION_CACHE[name][entry["scenario"]][side]
    tables = {r["table"]: r for r in _table_index()[(entry["cohort"], entry["scenario"], side)]}
    entity, role, inferred = [], [], []
    for label in labels:
        table, column = label.split(".", 1)
        truth = (info[table] if table in info else next(iter(info.values())))[column]
        entity.append(truth["entity"])
        role.append(tuple(truth["role"]))
        rec = tables[table]
        inferred.append((table, rec[rule]["homes"][rec["header"].index(column)]))
    return entity, role, inferred


def _load_matrix_manifest():
    if not MATRIX_MANIFEST.exists():
        raise SystemExit("run `matrices` first: the matrix manifest precedes any metric")
    data = json.loads(MATRIX_MANIFEST.read_text(encoding="utf-8"))
    if data["grain_manifest_sha256"] != sha256(MANIFEST):
        raise SystemExit("matrix manifest built from a different grain manifest")
    for r in data["matrices"]:
        if sha256(ROOT / r["file"]) != r["sha256"]:
            raise SystemExit(f"{r['file']}: digest changed since the matrix manifest")
    return data


def _regime_group(entry):
    return "plain" if entry["regime"] in ("plain", "ORIG") else "recoded"


def _iter_step3(data):
    index = _manifest_index()
    for r in data["matrices"]:
        entry = index[(r["cohort"], r["scenario"])]
        if entry.get("excluded"):
            continue
        yield entry, r


def _load(r):
    m = np.load(ROOT / r["file"], allow_pickle=True)
    gt = np.load((ROOT / r["file"]).parent / "gt.npy")
    return m["S"], list(m["source_labels"]), list(m["target_labels"]), gt


def cmd_decompose(args):
    data = _load_matrix_manifest()
    rows = []
    for entry, r in _iter_step3(data):
        S, s_labels, t_labels, gt = _load(r)
        t_ent, t_role, _ = _classes(entry, t_labels, "target")
        n_rows = n_err = 0
        cross_ent = cross_role = 0.0
        for i in range(S.shape[0]):
            truth = np.flatnonzero(gt[i])
            if truth.size == 0 or any(t_ent[j] == "unknown" for j in truth):
                continue
            n_rows += 1
            wrong = np.flatnonzero((S[i] >= S[i].max() - EPS) & ~gt[i])
            if wrong.size == 0:
                continue
            n_err += 1
            ents = {t_ent[j] for j in truth}
            roles = {t_role[j] for j in truth}
            cross_ent += float(np.mean([t_ent[j] not in ents for j in wrong]))
            cross_role += float(np.mean([t_role[j] not in roles for j in wrong]))
        rows.append({"cohort": entry["cohort"], "scenario": entry["scenario"], "regime": _regime_group(entry),
                     "family": entry["family"], "arm": r["arm"], "rows": n_rows, "errors": n_err,
                     "cross_entity_sum": cross_ent, "cross_role_sum": cross_role})
    with P2_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for key in sorted({(r["cohort"], r["regime"], r["arm"]) for r in rows}):
        sel = [r for r in rows if (r["cohort"], r["regime"], r["arm"]) == key]
        errors = sum(r["errors"] for r in sel)
        summary["/".join(key)] = {
            "scenarios": len(sel), "rows": sum(r["rows"] for r in sel), "errors": errors,
            "cross_entity_share_pooled": sum(r["cross_entity_sum"] for r in sel) / errors if errors else None,
            "cross_role_share_pooled": sum(r["cross_role_sum"] for r in sel) / errors if errors else None,
            "cross_entity_share_scenario_mean": statistics.fmean(
                [r["cross_entity_sum"] / r["errors"] for r in sel if r["errors"]]) if errors else None}
    P2_JSON.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {P2_CSV.name}, {P2_JSON.name}")


def _class_map(src, tgt, gt, same_label_fallback):
    """Source class -> set of target classes holding most of its GT pairs (A2e)."""
    counts = {}
    for i, j in zip(*np.nonzero(gt)):
        counts.setdefault(src[i], {}).setdefault(tgt[j], 0)
        counts[src[i]][tgt[j]] += 1
    targets = set(tgt)
    out = {}
    for c in set(src):
        if c in counts:
            best = max(counts[c].values())
            out[c] = {d for d, v in counts[c].items() if v == best}
        elif same_label_fallback and c in targets:
            out[c] = {c}
        else:
            out[c] = targets
    return out


def _inferred_map(src, tgt, S):
    """A3: C -> argmax_D mean_{a in C} max_{b in D} of the min-max normalised S. No GT."""
    lo, hi = float(S.min()), float(S.max())
    N = (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)
    s_idx, t_idx = {}, {}
    for i, c in enumerate(src):
        s_idx.setdefault(c, []).append(i)
    for j, d in enumerate(tgt):
        t_idx.setdefault(d, []).append(j)
    out = {}
    for c, rows in s_idx.items():
        k = {d: float(N[np.ix_(rows, cols)].max(axis=1).mean()) for d, cols in t_idx.items()}
        best = max(k.values())
        out[c] = {d for d, v in k.items() if v == best}
    return out


def _vote_map(src, tgt, S, majority=False):
    """CVA / CVA-maj (GRAIN_ALIGNMENT.md s.1), verbatim from grain_alignment_dev.py."""
    from smx.core import mutual_top1
    M = mutual_top1(S)
    votes = {}
    for i, j in zip(*np.nonzero(M)):
        votes.setdefault(src[i], {}).setdefault(tgt[j], 0)
        votes[src[i]][tgt[j]] += 1
    targets = set(tgt)
    out = {}
    for c in set(src):
        if c not in votes:
            out[c] = targets                      # abstain
        elif majority:
            best = max(votes[c].values())
            out[c] = {d for d, v in votes[c].items() if v == best}
        else:
            out[c] = set(votes[c])
    return out


_DIST: dict = {}


def _class_cardinality(entry, labels, side, inferred):
    """Distinct count of each class's anchor; FACT -> the table's row count."""
    from encrypted_feature_ablation import read_table
    out = {}
    tables = {t["table"]: t for t in entry["sides"][side]}
    for cls in inferred:
        if cls in out:
            continue
        table, home = cls
        path = ROOT / tables[table]["path"]
        if path not in _DIST:
            header, cols = read_table(path)
            _DIST[path] = (header, [len({v for v in c if v.strip()}) for c in cols], len(cols[0]))
        header, dist, rows = _DIST[path]
        out[cls] = rows if home == "FACT" else min(dist[int(i)] for i in home.split("+"))
    return out


def _hungarian_map(src, tgt, S, card=None):
    """H / HC (GRAIN_ALIGNMENT.md s.1), verbatim from grain_alignment_dev.py."""
    from scipy.optimize import linear_sum_assignment
    lo, hi = float(S.min()), float(S.max())
    N = (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)
    s_idx, t_idx = {}, {}
    for i, c in enumerate(src):
        s_idx.setdefault(c, []).append(i)
    for j, d in enumerate(tgt):
        t_idx.setdefault(d, []).append(j)
    out = {}
    for c, rows in s_idx.items():
        score = {}
        for d, cols in t_idx.items():
            sub = N[np.ix_(rows, cols)]
            r, k = linear_sum_assignment(sub, maximize=True)
            w = float(sub[r, k].sum()) / len(rows)
            if card is not None:
                cs, ct = max(card[0][c], 1), max(card[1][d], 1)
                w *= float(np.exp(-abs(np.log(cs) - np.log(ct))))
            score[d] = w
        best = max(score.values())
        out[c] = {d for d, v in score.items() if v == best}
    return out


def _metrics(S, gt):
    from smx.core import (ScoreMatrix, hungarian, match_metrics, mutual_top1,
                          recall_at_ground_truth, scoring_diagnostics)
    return {"f1_mutual": match_metrics(mutual_top1(S), gt)["f1"],
            "f1_hungarian": match_metrics(hungarian(S), gt)["f1"],
            "recall_at_gt": recall_at_ground_truth(S, gt)["recall_at_gt_pessimistic"],
            "mrr": scoring_diagnostics(ScoreMatrix(S, gt))["mrr_pessimistic"]}


def cmd_ceiling(args):
    if not P2_JSON.exists():
        raise SystemExit("P2 (`decompose`) must be reported before P3")
    data = _load_matrix_manifest()
    rows = []
    for entry, r in _iter_step3(data):
        S, s_labels, t_labels, gt = _load(r)
        s_ent, s_role, s_inf = _classes(entry, s_labels, "source")
        t_ent, t_role, t_inf = _classes(entry, t_labels, "target")
        maps = {"A1": (s_ent, t_ent, _class_map(s_ent, t_ent, gt, True)),
                "A1_role": (s_role, t_role, _class_map(s_role, t_role, gt, True)),
                "A2": (s_inf, t_inf, _class_map(s_inf, t_inf, gt, False)),
                "A3": (s_inf, t_inf, _inferred_map(s_inf, t_inf, S))}
        if STAGE == "conf":
            s_tab = [(l.split(".", 1)[0],) for l in s_labels]
            t_tab = [(l.split(".", 1)[0],) for l in t_labels]
            card = (_class_cardinality(entry, s_labels, "source", s_inf),
                    _class_cardinality(entry, t_labels, "target", t_inf))
            t1 = _class_map(s_tab, t_tab, gt, False)
            maps.update({"CVA": (s_inf, t_inf, _vote_map(s_inf, t_inf, S)),
                         "CVA_maj": (s_inf, t_inf, _vote_map(s_inf, t_inf, S, True)),
                         "H": (s_inf, t_inf, _hungarian_map(s_inf, t_inf, S)),
                         "HC": (s_inf, t_inf, _hungarian_map(s_inf, t_inf, S, card)),
                         "T1": (s_tab, t_tab, t1)})
            both = [(a, b) for a, b in zip(s_tab, s_ent)], [(a, b) for a, b in zip(t_tab, t_ent)]
            a1 = _class_map(s_ent, t_ent, gt, True)
            maps["T1_and_A1"] = (both[0], both[1],
                                 {c: {d for d in set(both[1]) if d[0] in t1[c[0]] and d[1] in a1[c[1]]}
                                  for c in set(both[0])})
        base = {"cohort": entry["cohort"], "scenario": entry["scenario"], "regime": _regime_group(entry),
                "family": entry["family"], "arm": r["arm"]}
        rows.append({**base, "extraction": "A0", "allowed_frac": 1.0, "gt_disallowed": 0, **_metrics(S, gt)})
        for name, (src, tgt, cmap) in maps.items():
            allowed = np.array([[tgt[j] in cmap[src[i]] for j in range(len(tgt))] for i in range(len(src))])
            masked = np.where(allowed, S, float(S.min()) - 1.0)
            rows.append({**base, "extraction": name, "allowed_frac": float(allowed.mean()),
                         "gt_disallowed": int((gt & ~allowed).sum()), **_metrics(masked, gt)})
    with P3_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for key in sorted({(r["cohort"], r["regime"], r["arm"]) for r in rows}):
        sel = [r for r in rows if (r["cohort"], r["regime"], r["arm"]) == key]
        block = {}
        for ext in sorted({r["extraction"] for r in sel}):
            e = [r for r in sel if r["extraction"] == ext]
            block[ext] = {k: statistics.fmean([x[k] for x in e if x[k] == x[k]]) for k in
                          ("f1_mutual", "f1_hungarian", "recall_at_gt", "mrr", "allowed_frac")}
            block[ext]["gt_disallowed"] = sum(x["gt_disallowed"] for x in e)
            block[ext]["scenarios"] = len(e)
        summary["/".join(key)] = block
    P3_JSON.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {P3_CSV.name}, {P3_JSON.name}")



TABLE_CONTROL = REPORTS / "grain_k3_table_control.csv"


def cmd_table_control(args):
    """EXPLORATORY, added after P3 was read (not registered): C-SAK only.

    T1 allows target columns in the target table(s) holding most GT pairs of the
    source column's table; T1&A1 also requires the true entity. The contrast
    asks whether the grain ceiling is table routing in disguise.
    """
    if not P3_JSON.exists():
        raise SystemExit("run `ceiling` first")
    data = _load_matrix_manifest()
    rows = []
    for entry, r in _iter_step3(data):
        if entry["cohort"] != "C-SAK":
            continue
        S, sl, tl, gt = _load(r)
        s_tab = [l.split(".", 1)[0] for l in sl]
        t_tab = [l.split(".", 1)[0] for l in tl]
        s_ent, _, _ = _classes(entry, sl, "source")
        t_ent, _, _ = _classes(entry, tl, "target")

        def allow(src, tgt, fallback):
            cmap = _class_map(src, tgt, gt, fallback)
            return np.array([[tgt[j] in cmap[src[i]] for j in range(len(tgt))] for i in range(len(src))])

        T, E = allow(s_tab, t_tab, False), allow(s_ent, t_ent, True)
        f1 = lambda A: _metrics(np.where(A, S, float(S.min()) - 1.0), gt)["f1_mutual"]  # noqa: E731
        rows.append({"scenario": entry["scenario"], "normalization": entry["normalization"],
                     "regime": _regime_group(entry), "arm": r["arm"], "A0": _metrics(S, gt)["f1_mutual"],
                     "T1": f1(T), "A1": f1(E), "T1_and_A1": f1(T & E),
                     "allowed_T1": float(T.mean()), "allowed_A1": float(E.mean()),
                     "gt_lost_T1_and_A1": int((gt & ~(T & E)).sum())})
    with TABLE_CONTROL.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {TABLE_CONTROL.name} ({len(rows)} rows)")


def _counts(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


STAGE = "pilot"


def set_stage(stage: str) -> None:
    """`conf` routes every artifact to grain_conf_* files (GRAIN_ALIGNMENT.md)."""
    global STAGE, MANIFEST, TABLES, GRID_CSV, P0_CSV, P0_JSON, MATRICES, MATRIX_MANIFEST
    global P2_CSV, P2_JSON, P3_CSV, P3_JSON, TABLE_CONTROL, STEP3_COHORTS
    STAGE = stage
    if stage == "pilot":
        return
    MANIFEST = REPORTS / "grain_conf_manifest.json"
    TABLES = REPORTS / "grain_conf_tables.json"
    GRID_CSV = REPORTS / "grain_conf_grid.csv"
    P0_CSV, P0_JSON = REPORTS / "grain_conf_p0.csv", REPORTS / "grain_conf_p0.json"
    MATRICES = REPORTS / "grain_conf_matrices"
    MATRIX_MANIFEST = REPORTS / "grain_conf_matrix_manifest.json"
    P2_CSV, P2_JSON = REPORTS / "grain_conf_k2_scenarios.csv", REPORTS / "grain_conf_k2.json"
    P3_CSV, P3_JSON = REPORTS / "grain_conf_k3_scenarios.csv", REPORTS / "grain_conf_k3.json"
    TABLE_CONTROL = REPORTS / "grain_conf_table_control.csv"
    STEP3_COHORTS = {"C-CONF": "c_conf"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("pilot", "conf"), default="pilot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--force", action="store_true")
    i = sub.add_parser("infer")
    i.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    sub.add_parser("check")
    sub.add_parser("k1")
    sub.add_parser("matrices")
    sub.add_parser("decompose")
    sub.add_parser("ceiling")
    sub.add_parser("table-control")
    args = ap.parse_args()
    set_stage(args.stage)
    {"manifest": cmd_manifest, "infer": cmd_infer, "check": cmd_check, "k1": cmd_k1,
     "matrices": cmd_matrices, "decompose": cmd_decompose, "ceiling": cmd_ceiling,
     "table-control": cmd_table_control}[args.cmd](args)


if __name__ == "__main__":
    main()
