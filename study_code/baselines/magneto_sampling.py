"""Magneto on the independent-sampling cohort (MAGNETO_INDEPENDENT_SAMPLING.md).

    python magneto_sampling.py prepare                  # main interpreter: rebuild tables, check 1, stage frames
    .venv-bdi/Scripts/python.exe magneto_sampling.py magneto   # Magneto matrices (bdi-kit environment)
    python magneto_sampling.py score                    # check 2, arm scoring, summary

Outputs: reports/independent_sampling_magneto/{metrics.csv, summary.json, checks.json}.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPORT = HERE / "reports" / "independent_sampling_20260929"
OUT = HERE / "reports" / "independent_sampling_magneto"
STAGE = HERE / "generated" / "independent_sampling_magneto_stage"
MAX_ROWS = 2000
METHOD = "magneto_zs_bp"


def scenario_dirs():
    return sorted(p for p in STAGE.glob("*/*/*") if (p / "meta.json").is_file())


def cmd_prepare():
    import numpy as np
    import independent_sampling as isx
    import grain_confirm as gc
    checks, n = [], 0
    for case in sorted((REPORT / "cases").iterdir()):
        dbname, seed = case.name.rsplit("_", 1)
        db = isx.DBS[dbname]
        samples = json.loads((case / "samples.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = set(samples["identical"]["source"])
            sr = isx.closure(db, src)
            sv = {v: isx.representation(db, v, sr, "source", tmp / "source") for v in db["variants"]}
            for condition in isx.CONDITIONS:
                tr = sr if condition == "identical" else isx.closure(db, set(samples[condition]["target"]))
                tv = {v: isx.representation(db, v, tr, "target", tmp / condition) for v in db["variants"]}
                for a, b, _ in gc.PAIRINGS[:3]:
                    for regime in ("plain", "opaque"):
                        s, t = isx.flatten(sv[a], regime, "source"), isx.flatten(tv[b], regime, "target")
                        gt = np.asarray([[gc.corresponds(x, y) for y in t["prov"]] for x in s["prov"]], bool)
                        pmf = isx.pmf_scores(s["cols"], t["cols"])
                        stored = np.load(case / condition / f"{a}_{b}_{regime}.npz")
                        checks.append({"case": case.name, "condition": condition, "pair": f"{a}_{b}", "regime": regime,
                                       "gt_equal": bool(np.array_equal(stored["gt"], gt)),
                                       "pmf_equal": bool(np.array_equal(stored["pmf"], pmf))})
                        dest = STAGE / case.name / condition / f"{a}_{b}_{regime}"
                        dest.mkdir(parents=True, exist_ok=True)
                        sides = {}
                        for side, rep in (("source", sv[a]), ("target", tv[b])):
                            files, names = [], []
                            for k, (table, r) in enumerate(rep.items()):
                                cols = r["cols"] if regime == "plain" else r["opaque"]
                                labels = ([f"{table}.{x}" for x in r["labels"]] if regime == "plain" else
                                          isx.flatten({table: r}, regime, side)["labels"])
                                f = dest / f"{side}_{k}.csv"
                                gc.write_table(f, labels, list(zip(*cols))[:MAX_ROWS])
                                files.append(f.name)
                                names.extend(labels)
                            sides[side] = {"files": files, "names": names}
                        np.savez_compressed(dest / "inputs.npz", gt=gt, pmf=pmf)
                        meta = {"database": dbname, "seed": seed, "condition": condition, "pair": f"{a}_{b}",
                                "regime": regime, "sides": sides,
                                "homes": {"source": s["homes"], "target": t["homes"]},
                                "entities": {"source": s["entities"], "target": t["entities"]},
                                "cards": {"source": [[list(k), v] for k, v in s["cards"].items()],
                                          "target": [[list(k), v] for k, v in t["cards"].items()]}}
                        (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                        n += 1
        print(case.name, "prepared", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ok = all(c["gt_equal"] and c["pmf_equal"] for c in checks)
    (OUT / "check_rebuild.json").write_text(json.dumps({"scenarios": n, "all_equal": ok,
                                                         "failures": [c for c in checks if not (c["gt_equal"] and c["pmf_equal"])][:10]},
                                                        indent=1), encoding="utf-8")
    print("scenarios", n, "rebuild check", ok)
    return 0 if ok else 1


def cmd_magneto():
    import numpy as np
    import pandas as pd
    from bdikit.api import rank_schema_matches
    log = OUT / "magneto.log"
    for d in scenario_dirs():
        if (d / "magneto.npz").is_file():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        try:
            frames = {}
            for side in ("source", "target"):
                # as grain_magneto.side_frame: pandas' default parsing, at most MAX_ROWS rows per table
                parts = [pd.read_csv(d / f, nrows=MAX_ROWS, low_memory=False) for f in meta["sides"][side]["files"]]
                frames[side] = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
                assert list(frames[side].columns) == meta["sides"][side]["names"], "column order"
            t0 = time.perf_counter()
            ranked = rank_schema_matches(frames["source"], target=frames["target"],
                                         top_k=max(1, frames["target"].shape[1]), method=METHOD)
            secs = time.perf_counter() - t0
            si = {c: i for i, c in enumerate(meta["sides"]["source"]["names"])}
            ti = {c: j for j, c in enumerate(meta["sides"]["target"]["names"])}
            S = np.zeros((len(si), len(ti)))
            for _, r in ranked.iterrows():
                i, j = si.get(r.iloc[0]), ti.get(r.iloc[1])
                if i is not None and j is not None:
                    S[i, j] = float(r.iloc[2])
            np.savez(d / "magneto.npz", S=S, seconds=secs)
            msg = f"ok {d.relative_to(STAGE)} {S.shape} {secs:.1f}s"
        except Exception as exc:  # noqa: BLE001
            msg = f"FAILED {d.relative_to(STAGE)}: {type(exc).__name__}: {exc}"
        print(msg, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    return 0


def arm_records(S, meta, gt):
    """The arm loop of independent_sampling.score_pair for one matrix, unchanged."""
    import numpy as np
    import independent_sampling as isx
    gp = isx.gp
    si = [tuple(x) for x in meta["homes"]["source"]]
    ti = [tuple(x) for x in meta["homes"]["target"]]
    hashable = lambda x: tuple(hashable(y) for y in x) if isinstance(x, list) else x  # noqa: E731
    se, te = [hashable(x) for x in meta["entities"]["source"]], [hashable(x) for x in meta["entities"]["target"]]
    cards = tuple({tuple(k): v for k, v in meta["cards"][side]} for side in ("source", "target"))
    db, seed, condition, (sv, tv) = meta["database"], meta["seed"], meta["condition"], meta["pair"].split("_")
    masks = {"base": np.ones(S.shape, bool),
             "CVA": isx.allowed_map(si, ti, gp._vote_map(si, ti, S)),
             "H": isx.allowed_map(si, ti, gp._hungarian_map(si, ti, S)),
             "HC": isx.allowed_map(si, ti, gp._hungarian_map(si, ti, S, cards)),
             "LOAO": isx.loao_mask(si, ti, S),
             "inferred_GT": isx.allowed_map(si, ti, gp._class_map(si, ti, gt, False)),
             "annotated_GT": isx.allowed_map(se, te, gp._class_map(se, te, gt, True))}
    for k in range(10):
        rs = isx.permute_classes(se, f"{db}|{seed}|{sv}|{tv}|s|{k}")
        rt = isx.permute_classes(te, f"{db}|{seed}|{sv}|{tv}|t|{k}")
        masks[f"random_GT_{k}"] = isx.allowed_map(rs, rt, gp._class_map(rs, rt, gt, True))
    baseline = gp._metrics(S, gt)
    M = isx.mutual_top1(S)
    out = []
    for arm, mask in masks.items():
        masked = np.where(mask, S, float(S.min()) - 1)
        mets = gp._metrics(masked, gt)
        MM = isx.mutual_top1(masked)
        out.append({"database": db, "seed": seed, "condition": condition, "pair": meta["pair"], "regime": meta["regime"],
                    "arm": arm, "gt": int(gt.sum()), "retained": float(mask.mean()), "lost_gt": int((gt & ~mask).sum()),
                    "removed_fp": int((M & ~gt & ~MM).sum()), "lost_tp": int((M & gt & ~MM).sum()),
                    **mets, **{"base_" + k: v for k, v in baseline.items()},
                    **{"delta_" + k: mets[k] - v for k, v in baseline.items()}})
    return out


def cmd_score():
    import numpy as np
    committed = {}
    for r in csv.DictReader(open(REPORT / "metrics.csv", encoding="utf-8")):
        if r["matcher"] == "pmf":
            committed[(r["database"], r["seed"], r["condition"], r["pair"], r["regime"], r["arm"])] = r
    mismatches, compared, rows, failed = 0, 0, [], []
    for d in scenario_dirs():
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        inputs = np.load(d / "inputs.npz")
        gt = inputs["gt"]
        for r in arm_records(inputs["pmf"], meta, gt):
            c = committed[(r["database"], r["seed"], r["condition"], r["pair"], r["regime"], r["arm"])]
            compared += 1
            mismatches += any(str(v) != c[k] for k, v in r.items() if k in c and k not in ("seed",))
        if not (d / "magneto.npz").is_file():
            failed.append(str(d.relative_to(STAGE)))
            continue
        S = np.load(d / "magneto.npz")["S"]
        if not np.any(S):
            failed.append(str(d.relative_to(STAGE)) + " (all zero)")
            continue
        rows.extend({**r, "matcher": "magneto"} for r in arm_records(S, meta, gt))
    checks = {"scoring reproduces committed pmf rows": {"compared": compared, "mismatches": mismatches, "pass": mismatches == 0},
              "magneto matrices": {"scenarios": len(scenario_dirs()), "failed_or_zero": failed, "pass": len(failed) < len(scenario_dirs())}}
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    if mismatches:
        print("scoring check failed; no Magneto result reported")
        return 1
    with open(OUT / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for regime in ("plain", "opaque"):
        for db in ("TPCH", "AdventureWorks", "Chinook"):
            def val(cond, arm, key):
                xs = [float(r[key]) for r in rows if r["database"] == db and r["regime"] == regime
                      and r["condition"] == cond and r["arm"] == arm]
                return sum(xs) / len(xs) if xs else None
            summary[f"{regime}/{db}"] = {
                "identical": val("identical", "base", "f1_mutual"), "independent": val("independent", "base", "f1_mutual"),
                "HC": val("independent", "HC", "delta_f1_mutual"), "CVA": val("independent", "CVA", "delta_f1_mutual"),
                "annotated_GT": val("independent", "annotated_GT", "delta_f1_mutual")}
            print(regime, db, {k: None if v is None else round(v, 3) for k, v in summary[f"{regime}/{db}"].items()})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit({"prepare": cmd_prepare, "magneto": cmd_magneto, "score": cmd_score}[sys.argv[1]]())
