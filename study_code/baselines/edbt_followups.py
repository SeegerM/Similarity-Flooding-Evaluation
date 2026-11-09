"""Frozen EDBT follow-ups. See EDBT_FOLLOWUPS_20260928.md.

Run from harness: python edbt_followups.py prepare|grid|grain|groups|summarize|verify
grid accepts --workers; grain accepts --stage pilot|conf. All outputs are additive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "reports/edbt_followups_20260928"
PROTOCOL = HERE / "EDBT_FOLLOWUPS_20260928.md"
AUDIT = HERE / "reports/audited_md5_fused_20260904"
FULL = HERE / "reports/valentine_full_382_20260903"
SEEDS = HERE / "reports/hellinger_seed_20260905"
STRONG = HERE / "reports/strong_base_20260925"
ARMS = {"char3", "levenshtein", "pmf", "seed", "magneto", "pmf_gcov_uncertainty"}
REPORT_INPUTS = ["strong_base_paired.csv", "encrypted_matcher_pmf_gcov_u.csv",
                 "sizematch_plain.csv", "sizematch_md5.csv", "baseline_ladder.csv",
                 "grain_k3_scenarios.csv", "grain_conf_k3_scenarios.csv"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def readcsv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def writecsv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def code_files():
    return sorted([PROTOCOL, *HERE.glob("*.py"), *(HERE / "smx").rglob("*.py"),
                   HERE / "tests/test_edbt_followups.py"])


def check_manifest(inputs=False):
    m = json.loads((OUT / "manifest.json").read_text())
    for label in (["code", "inputs"] if inputs else ["code"]):
        for path, digest in m[label].items():
            if sha(ROOT / path) != digest:
                raise RuntimeError(f"frozen {label} changed: {path}")
    return m


def prepare():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Manifest already exists; no implicit refreeze")
    from anchor_generalization import load_side
    from smx.adapters import read_matrix
    from matrix_health import health
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    cases, excluded, paths = [], [], {AUDIT / "protocol.json", FULL / "cases.csv"}
    for case in protocol["cases"]:
        if case["cohort"] != "remaining338":
            continue
        key = case["key"]
        needed = [STRONG / "cases" / key / "plain/rung3.csv",
                  SEEDS / "cases" / key / "pmf.csv", SEEDS / "cases" / key / "seed.csv",
                  FULL / "cases" / key / "levenshtein.csv",
                  AUDIT / "cases" / key / "plain/pmf.csv"]
        for reg in ("plain", "audited_md5"):
            folder = Path(case[reg])
            needed += [folder / "source/source.csv", folder / "target/target.csv"]
            needed += sorted((folder / "ground_truth").glob("*.csv"))
            needed += sorted((folder / "metadata").rglob("FD_results.txt"))
        missing = [relative(p) for p in needed if not p.is_file()]
        if missing:
            excluded.append({"id": case["id"], "reason": "missing inputs", "files": missing})
            continue
        if health(read_matrix(needed[0])[0])["degenerate"]:
            excluded.append({"id": case["id"], "reason": "degenerate saved direct base"})
            continue
        if not all(load_side(Path(case["plain"]), s)[1] for s in ("source", "target")):
            excluded.append({"id": case["id"], "reason": "FD pool absent on a side"})
            continue
        cases.append({k: case[k] for k in ("id", "key", "cohort", "family", "scenario_type", "plain", "audited_md5")})
        paths.update(needed)
    for prefix in ("grain", "grain_conf"):
        for suffix in ("manifest.json", "matrix_manifest.json", "tables.json"):
            paths.add(HERE / "reports" / f"{prefix}_{suffix}")
        mat = json.loads((HERE / "reports" / f"{prefix}_matrix_manifest.json").read_text())
        for r in mat["matrices"]:
            if r["arm"] in ARMS:
                paths.update([ROOT / r["file"], (ROOT / r["file"]).parent / "gt.npy"])
    paths.update((HERE / "grain_annotations").glob("*.json"))
    paths.update(HERE / "reports" / f for f in REPORT_INPUTS)
    print(f"Freezing {len(cases)} grid cases, {len(paths)} input files", flush=True)
    inputs = {relative(p): sha(p) for p in sorted(paths)}
    dump(OUT / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
         "code": {relative(p): sha(p) for p in code_files()}, "inputs": inputs,
         "grid_cases": cases, "exclusions": excluded,
         "python": sys.version, "numpy": np.__version__})
    print(f"Frozen: {len(cases)} included, {len(excluded)} excluded", flush=True)


def partition_check(a, b):
    if len(a) != len(b):
        raise AssertionError("row count changed")
    forward, backward = {}, {}
    for x, y in zip(a, b):
        if (x == "") != (y == ""):
            raise AssertionError("null mask changed")
        if forward.setdefault(x, y) != y or backward.setdefault(y, x) != x:
            raise AssertionError("equality partition changed")


def exact_recode(column, key):
    mapping = {v: hashlib.sha256((key + "\0" + v).encode("utf-8")).hexdigest() for v in set(column) if v != ""}
    mapping[""] = ""
    out = [mapping[v] for v in column]
    partition_check(column, out)
    return out


def metrics(S, gt):
    from smx.core import ScoreMatrix, evaluate
    vals = list(evaluate(ScoreMatrix(scores=S, ground_truth=gt), policies=["mutual_top1", "hungarian"]))
    a = next(r for r in vals if r["extractor"] == "mutual_top1")
    b = next(r for r in vals if r["extractor"] == "hungarian")
    return {"f1": float(a["f1"]), "f1_h": float(b["f1"]),
            "recall_at_gt": float(a["recall_at_gt_pessimistic"]), "mrr": float(a["mrr_pessimistic"])}


def grid_case(case):
    from anchor_generalization import load_side, anchor_matrix, bare, correspondence
    from baseline_ladder import cardinality_channel, CAP, SEED
    from class_comparison_conditioned import read_columns
    from compose_gate import fuse
    from encrypted_feature_ablation import scale_aligned_fuse
    from regime_tables import label_matrix
    from smx.adapters import read_matrix
    from smx.soft_overlap import exact_matrix, fuzzy_matrix, normalise, sample_values
    from transferability_table import selector_matrix, image_filter, pair_candidates, greedy_pairs, positional
    from transferability_classes import fd_profile
    from od_role import cosine_similarity

    dest = OUT / "grid_cases" / (case["key"] + ".json")
    if dest.exists():
        return case["key"], "cached"
    raw, names, values, fds = {}, {}, {}, {}
    for reg in ("plain", "audited_md5"):
        folder = Path(case[reg])
        pairs = [read_columns(folder / side / f"{side}.csv", max_rows=sys.maxsize) for side in ("source", "target")]
        names[reg] = [[bare(n) for n in p[0]] for p in pairs]
        raw[reg] = [p[1] for p in pairs]
        loaded = [load_side(folder, s) for s in ("source", "target")]
        values[reg], fds[reg] = [x[0] for x in loaded], [x[1] for x in loaded]
        for cols, pool in zip(names[reg], fds[reg]):
            assert len(set(cols)) == len(cols)
            assert all(dep in cols and all(d in cols for d in det) for det, dep in pool), "FD endpoint mismatch"
    historical_violations = []
    for side in range(2):
        assert len(raw["plain"][side]) == len(raw["audited_md5"][side])
        for col, (a, b) in enumerate(zip(raw["plain"][side], raw["audited_md5"][side])):
            try:
                partition_check(a, b)
            except AssertionError as exc:
                historical_violations.append({"side": side, "column": col, "reason": str(exc)})
        assert positional(fds["plain"][side], names["plain"][side]) == positional(fds["audited_md5"][side], names["audited_md5"][side]), "FD profiles differ positionally"
        raw["audited_md5"][side] = [exact_recode(col, f"20260928|{case['id']}|{side}|{j}") for j, col in enumerate(raw["plain"][side])]
        # Exact raw null convention; no plaintext null masks are used downstream.
        values["audited_md5"][side] = {name: {v for v in col if v} for name, col in zip(names["audited_md5"][side], raw["audited_md5"][side])}
    s_cols, t_cols = names["plain"]
    src, tgt = fds["plain"]
    key = case["key"]
    gt = read_matrix(next((Path(case["plain"]) / "ground_truth").glob("*.csv")))[0] > .5
    lex = {1: read_matrix(FULL / "cases" / key / "levenshtein.csv")[0],
           0: label_matrix(*names["audited_md5"]).astype(float)}
    hell = read_matrix(SEEDS / "cases" / key / "pmf.csv")[0]
    saved_seed = read_matrix(SEEDS / "cases" / key / "seed.csv")[0]
    saved_base = read_matrix(STRONG / "cases" / key / "plain/rung3.csv")[0]
    seeds = {1: saved_seed, 0: scale_aligned_fuse(lex[0], hell, .35)}
    pmf_sel = selector_matrix(AUDIT / "cases" / key / "plain/pmf.csv", s_cols, t_cols, "mutual3")
    map_st, map_ts = correspondence(gt, s_cols, t_cols), correspondence(gt.T, t_cols, s_cols)
    oracle = (image_filter("FD", src, tgt, map_st), image_filter("FD", tgt, src, map_ts))
    def paired(mat):
        return greedy_pairs(pair_candidates("FD", src, tgt, mat, s_cols, t_cols), src, tgt,
                            max(1, round(.5 * .282 * len(src))), max(1, round(.5 * .282 * len(tgt))))
    pmf_pick = paired(pmf_sel)
    rows, doses = [], []
    for V, reg in ((1, "plain"), (0, "audited_md5")):
        non_null = [[[v for v in col if v.strip()] for col in side] for side in raw[reg]]
        norm = [[[v for v in sorted({normalise(x) for x in col}) if v] for col in side] for side in non_null]
        samples = [[sample_values(col, CAP, SEED) for col in side] for side in norm]
        direct = [fuzzy_matrix(*samples, kind="char3"), exact_matrix(*non_null, variant="jaccard"), cardinality_channel(raw[reg])]
        actual_anchor = anchor_matrix(*values[reg], *names[reg])
        picks = {"all": (src, tgt), "oracle": oracle, "paired_anchor": paired(actual_anchor), "paired_pmf": pmf_pick}
        roles = {a: cosine_similarity(fd_profile(s, s_cols), fd_profile(t, t_cols)) for a, (s, t) in picks.items()}
        overlap = direct[1]
        if V == 0:
            assert np.max(overlap) == 0, "MD5 cross-side overlap survived"
        for L in (1, 0):
            if L == 0:
                assert not set(names["audited_md5"][0]) & set(names["audited_md5"][1])
            base = fuse(seeds[L], direct, .35)
            if L == V == 1:
                assert np.allclose(base, saved_base, rtol=0, atol=1e-10), "readable base mismatch"
                assert metrics(base, gt) == metrics(saved_base, gt), "readable metrics mismatch"
            matrices = {"base": base, **{a: fuse(base, [r], .35) for a, r in roles.items()}}
            for arm, mat in matrices.items():
                s, t = picks.get(arm, ([], []))
                rows.append({"id": case["id"], "family": case["family"], "L": L, "V": V,
                             "arm": arm, "n_fd_s": len(s), "n_fd_t": len(t), **metrics(mat, gt)})
            doses.append({"L": L, "V": V, "exact_overlap_all_mean": float(overlap.mean()),
                          "exact_overlap_gt_mean": float(overlap[gt].mean()) if gt.any() else 0.,
                          "label_similarity_mean": float(lex[L].mean()),
                          "label_similarity_gt_mean": float(lex[L][gt].mean()) if gt.any() else 0.,
                          "partitions_equal": True, "fd_profiles_equal": True})
    dump(dest, {"rows": rows, "doses": doses, "historical_partition_violations": historical_violations})
    return key, "done"


def grid(workers):
    m = check_manifest(inputs=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(grid_case, c): c for c in m["grid_cases"]}
        for i, f in enumerate(as_completed(pending), 1):
            try:
                key, status = f.result()
            except Exception as exc:
                dump(OUT / "grid_failure.json", {"case": pending[f]["id"], "error": repr(exc)})
                raise
            print(f"grid {i}/{len(pending)} {key} {status}", flush=True)


def allowed_map(src, tgt, mapping):
    return np.asarray([[b in mapping[a] for b in tgt] for a in src], dtype=bool)


def loao_mask(src, tgt, S):
    from smx.core import mutual_top1
    votes = list(zip(*np.nonzero(mutual_top1(S))))
    out = np.ones(S.shape, dtype=bool)
    for i, cls in enumerate(src):
        targets = {tgt[j] for k, j in votes if k != i and src[k] == cls}
        if targets:
            out[i] = [c in targets for c in tgt]
    return out


def permute_classes(classes, key):
    out = list(classes)
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    rng.shuffle(out)
    assert Counter(out) == Counter(classes)
    return out


def grain(stage):
    check_manifest(inputs=True)
    import grain_pilot as gp
    from smx.core import mutual_top1
    gp = importlib.reload(gp)
    gp.set_stage(stage)
    data = gp._load_matrix_manifest()
    old = {(r["scenario"], r["arm"], r["extraction"]): r for r in readcsv(gp.P3_CSV)}
    outrows = []
    for entry, r in gp._iter_step3(data):
        if r["arm"] not in ARMS:
            continue
        S, sl, tl, gt = gp._load(r)
        gt = gt.astype(bool)
        se, _, si = gp._classes(entry, sl, "source")
        te, _, ti = gp._classes(entry, tl, "target")
        M = mutual_top1(S)
        baseline = metrics(S, gt)
        existing = old.get((entry["scenario"], r["arm"], "A0"))
        assert existing is not None and abs(baseline["f1"] - float(existing["f1_mutual"])) < 1e-12
        common = {"stage": stage, "family": entry["family"], "scenario": entry["scenario"],
                  "normalization": entry.get("normalization", "all"), "regime": gp._regime_group(entry),
                  "matcher": r["arm"], "base_f1": baseline["f1"], "base_fp": int((M & ~gt).sum()),
                  "base_tp": int((M & gt).sum()), "gt_count": int(gt.sum())}
        for typ, src, tgt in (("annotated", se, te), ("inferred", si, ti)):
            masks = {
                "GT_map": allowed_map(src, tgt, gp._class_map(src, tgt, gt, typ == "annotated")),
                "CVA": allowed_map(src, tgt, gp._vote_map(src, tgt, S)),
                "H": allowed_map(src, tgt, gp._hungarian_map(src, tgt, S)),
                "LOAO_CVA": loao_mask(src, tgt, S)}
            for rep in range(10):
                seed = f"20260928|{stage}|{entry['scenario']}|{typ}|{rep}"
                rs, rt = permute_classes(src, seed + "s"), permute_classes(tgt, seed + "t")
                masks[f"random_GT_{rep}"] = allowed_map(rs, rt, gp._class_map(rs, rt, gt, typ == "annotated"))
            for align, allowed in masks.items():
                masked = np.where(allowed, S, float(S.min()) - 1)
                selected = mutual_top1(masked)
                met = metrics(masked, gt)
                if align == "CVA":
                    assert np.all(allowed[M]) and np.all(selected[M]), "CVA preservation violated"
                if align == "GT_map":
                    hist = old.get((entry["scenario"], r["arm"], "A1" if typ == "annotated" else "A2"))
                    assert hist is not None and abs(met["f1"] - float(hist["f1_mutual"])) < 1e-12
                outrows.append({**common, "classes": typ, "alignment": align,
                    "n_class_s": len(set(src)), "n_class_t": len(set(tgt)),
                    "singleton_share_s": sum(v == 1 for v in Counter(src).values()) / len(set(src)),
                    "singleton_share_t": sum(v == 1 for v in Counter(tgt).values()) / len(set(tgt)),
                    "allowed_fraction": float(allowed.mean()), "gt_excluded": int((gt & ~allowed).sum()),
                    "fp_removed": int((M & ~gt & ~selected).sum()),
                    "tp_lost": int((M & gt & ~selected).sum()),
                    "tp_added": int((~M & gt & selected).sum()),
                    "fp_added": int((~M & ~gt & selected).sum()),
                    "changed": int(np.any(M != selected)), **met, "gain": met["f1"] - baseline["f1"]})
        print(f"grain {stage} {entry['scenario']} {r['arm']}", flush=True)
    writecsv(OUT / f"grain_{stage}.csv", outrows)


def grouped_effect(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(float(r["effect"]))
    loo = {f: mean(float(r["effect"]) for r in rows if r["family"] != f)
           for f in by if len(by) > 1}
    return {"n": len(rows), "families": len(by), "scenario_mean": mean(float(r["effect"]) for r in rows),
            "family_mean": mean(mean(v) for v in by.values()),
            "by_family": {f: {"n": len(v), "effect": mean(v)} for f, v in sorted(by.items())},
            "leave_one_family_out": loo}


def load_combo(name, metric="f1"):
    out = defaultdict(dict)
    for r in readcsv(HERE / "reports" / name):
        if r["cohort"] == "remaining338" and r["extractor"] == "mutual_top1":
            out[(r["variant"], r["subset"])][r["id"]] = float(r[metric])
    return out


def groups():
    check_manifest(inputs=True)
    rows, summary = [], {}
    families = {r["id"]: r["family"] for r in readcsv(FULL / "cases.csv")}
    def contrast(label, x, y, ids=None):
        ids = sorted(set(x) & set(y) if ids is None else set(ids) & set(x) & set(y))
        batch = [{"contrast": label, "id": i, "family": families[i], "treated": x[i], "base": y[i], "effect": x[i] - y[i]} for i in ids]
        rows.extend(batch)
        summary[label] = grouped_effect(batch)
    for filename, label, pair in (("strong_base_paired.csv", "readable", "paired-anchor-b50"),
                                 ("encrypted_matcher_pmf_gcov_u.csv", "opaque", "paired-pmf-mutual3-b50")):
        d = load_combo(filename)
        base = d[("oracle", "none")]
        # Match the existing full-class result for readable; opaque wider arm uses each case's maximum available class set.
        for arm in (pair, "oracle"):
            vals = {}
            if label == "readable":
                vals = d[(arm, "FD+OD+DC")]
            else:
                for r in readcsv(HERE / "reports" / filename):
                    if r["cohort"] == "remaining338" and r["extractor"] == "mutual_top1" and r["variant"] == arm and r["subset"] != "none":
                        n = int(r["k"])
                        if r["id"] not in vals or n > vals[r["id"]][0]:
                            vals[r["id"]] = (n, float(r["f1"]))
                vals = {i: v for i, (n, v) in vals.items()}
            contrast(f"{label}_{arm}_vs_base", vals, base)
    dp, dm = load_combo("strong_base_paired.csv"), load_combo("encrypted_matcher_pmf_gcov_u.csv")
    ids = set(dp[("oracle", "FD+OD+DC")]) & set(dm[("oracle", "FD+OD+DC")])
    for name, d in (("readable_matched", dp), ("opaque_matched", dm)):
        contrast(name + "_oracle", d[("oracle", "FD+OD+DC")], d[("oracle", "none")], ids)
    for reg, pair in (("plain", "anchor"), ("md5", "pmf-mutual3")):
        for metric in ("f1", "recall_at_gt"):
            d = load_combo(f"sizematch_{reg}.csv", metric)
            contrast(f"{reg}_pair_minus_size_{metric}", d[(f"paired-{pair}-b50", "FD+OD+DC")], d[(f"sizematch-{pair}-b50", "FD+OD+DC")])
    writecsv(OUT / "group_contrasts.csv", rows)
    dump(OUT / "group_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def summarize():
    m = check_manifest()
    gridrows, dose = [], []
    for c in m["grid_cases"]:
        d = json.loads((OUT / "grid_cases" / (c["key"] + ".json")).read_text())
        assert len(d["rows"]) == 20
        gridrows.extend(d["rows"])
        dose.extend({"id": c["id"], **r} for r in d["doses"])
    writecsv(OUT / "grid.csv", gridrows)
    writecsv(OUT / "grid_dose.csv", dose)
    base = {(r["id"], r["L"], r["V"]): r for r in gridrows if r["arm"] == "base"}
    summ = {}
    for arm in ("base", "all", "oracle", "paired_anchor", "paired_pmf"):
        for L in (1, 0):
            for V in (1, 0):
                sel = [r for r in gridrows if (r["arm"], r["L"], r["V"]) == (arm, L, V)]
                for metric in ("f1", "f1_h", "recall_at_gt", "mrr"):
                    vals = [{"family": r["family"], "effect": r[metric] - base[(r["id"], L, V)][metric]} for r in sel]
                    summ[f"{arm}_L{L}V{V}_{metric}"] = {"absolute_mean": mean(r[metric] for r in sel), **grouped_effect(vals)}
    interactions = {}
    for arm in ("all", "oracle", "paired_anchor", "paired_pmf"):
        vals = {(r["id"], r["L"], r["V"]): r["f1"] - base[(r["id"], r["L"], r["V"])]["f1"] for r in gridrows if r["arm"] == arm}
        for label, terms in {"labels_removed_V1": [(0, 1, 1), (1, 1, -1)],
                             "labels_removed_V0": [(0, 0, 1), (1, 0, -1)],
                             "values_removed_L1": [(1, 0, 1), (1, 1, -1)],
                             "values_removed_L0": [(0, 0, 1), (0, 1, -1)],
                             "interaction": [(0, 0, 1), (1, 0, -1), (0, 1, -1), (1, 1, 1)]}.items():
            interactions[f"{arm}_{label}"] = grouped_effect([{"family": c["family"], "effect": sum(sign * vals[(c["id"], L, V)] for L, V, sign in terms)} for c in m["grid_cases"]])
    dump(OUT / "grid_summary.json", summ)
    dump(OUT / "grid_interactions.json", interactions)
    grainrows = readcsv(OUT / "grain_pilot.csv") + readcsv(OUT / "grain_conf.csv")
    grouped = defaultdict(list)
    # Average random replicates within a case first: ten draws do not create ten observations.
    scenario = defaultdict(list)
    for r in grainrows:
        align = "random_GT" if r["alignment"].startswith("random_GT_") else r["alignment"]
        key = tuple(r[k] for k in ("stage", "family", "regime", "normalization", "matcher", "classes")) + (align, r["scenario"])
        scenario[key].append(r)
    fields = ["base_f1", "gain", "f1", "allowed_fraction", "gt_excluded", "gt_count", "base_fp", "base_tp", "fp_removed", "tp_lost", "tp_added", "fp_added", "changed", "n_class_s", "n_class_t", "singleton_share_s", "singleton_share_t"]
    for key, rr in scenario.items():
        grouped[key[:-1]].append({f: mean(float(r[f]) for r in rr) for f in fields})
    gs = []
    for key, rr in sorted(grouped.items()):
        gs.append({**dict(zip(("stage", "family", "regime", "normalization", "matcher", "classes", "alignment"), key)),
                   "n": len(rr), **{f: mean(r[f] for r in rr) for f in fields},
                   "fp_removal_rate": sum(r["fp_removed"] for r in rr) / max(1, sum(r["base_fp"] for r in rr)),
                   "gt_exclusion_rate": sum(r["gt_excluded"] for r in rr) / max(1, sum(r["gt_count"] for r in rr))})
    writecsv(OUT / "grain_summary.csv", gs)
    print(f"Summarized {len(gridrows)} grid and {len(grainrows)} grain rows", flush=True)


def verify():
    m = check_manifest(inputs=True)
    expected = len(m["grid_cases"]) * 20
    assert len(readcsv(OUT / "grid.csv")) == expected
    for stage in ("pilot", "conf"):
        for r in readcsv(OUT / f"grain_{stage}.csv"):
            if r["alignment"] == "CVA":
                assert int(r["fp_removed"]) == int(r["tp_lost"]) == 0
    files = [p for p in OUT.rglob("*") if p.is_file() and p.name not in ("verification.json", "grid_failure.json") and p.suffix != ".tmp"]
    dump(OUT / "verification.json", {"verified_utc": datetime.now(timezone.utc).isoformat(),
         "input_files_unchanged": len(m["inputs"]), "grid_rows": expected,
         "outputs": {relative(p): sha(p) for p in sorted(files)}})
    print(f"Verified input/code identity and {expected} grid rows; CVA correction assertions pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "grid", "grain", "groups", "summarize", "verify"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["pilot", "conf"], default="pilot")
    args = ap.parse_args()
    if args.command == "grid":
        grid(args.workers)
    elif args.command == "grain":
        grain(args.stage)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
