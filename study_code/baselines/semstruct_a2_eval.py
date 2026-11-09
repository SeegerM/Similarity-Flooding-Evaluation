"""SEMSTRUCT_ABLATION.md, Amendment 2: the gate and the information-matched contrasts of the revised-setup run.

Matrices, copied from <gpu-host> $SMX_REMOTE_ROOT/semstruct-a2/harness/reports/ into reports/semstruct_a2_20261029/<run>/matrices/
<regime>/<arm>__eval<cap>[_native]/<case key>.csv, where <run> is a2_{full,valavg,onehop}_V41[_s<seed>|_nomerge].
Metrics and aggregation as semstruct_ablation_eval.py (the SF paper's tie-free J_tf; macro over families with >= 5
scenarios; 95% bootstrap over scenarios stratified by family, 2,000 resamples). For `full` and `valavg`, the per-scenario
value is the mean over the seeds present (the reading fixed in the queue entry); per-seed contrasts are listed too.
Arms that are not (yet) present are skipped, and so are the contrasts that need them.  EXPLORATORY.

    harness/.venv-isresmat/Scripts/python.exe -I harness/semstruct_a2_eval.py --cap 2000 [--regime plain] [--nomerge]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "layer3b"))
sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "tuning"))
sys.path.insert(0, str(HERE))
from score_r7 import metrics as tf_metrics  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
import semstruct_run as sr  # noqa: E402

ROOT = HERE / "reports" / "semstruct_a2_20261029"
SEEDS = ("", "_s20260909", "_s20260910")
B, SEED, MIN_FAMILY = 2000, 20261029, 5
CONTRASTS = [("full", "lm_only", "gate: structure over the header alone"),
             ("onehop", "full", "information-matched: one hop (own values) minus message passing"),
             ("valavg", "full", "the authors' ValAvg minus message passing"),
             ("onehop", "lm_only", "own values (one hop) over the header alone"),
             ("valavg", "lm_only", "ValAvg over the header alone"),
             ("full@0", "permuted", "row co-occurrence (full, first seed: permuted exists for it only)"),
             # the paper's Fig. 5 ladder, step by step: LM -> MLP -> ValAvg -> SemStruct (second queue, item 5)
             ("mlp", "lm_only", "ladder: MLP on the column embedding over the header alone"),
             ("valavg", "mlp", "ladder: the column's values (ValAvg) over the MLP"),
             ("full", "valavg", "ladder: message passing over ValAvg")]


def sources(nomerge: bool, prefix: str = "a2"):
    """arm -> list of (run folder, arm folder) per seed; the seed mean is taken over those present.
    prefix "goby_a2": the GOBY runs (second queue, item 6), one seed."""
    tag = "_nomerge" if nomerge else ""
    seeds = SEEDS if (not nomerge and prefix == "a2") else ("",)
    return {"full": [(f"{prefix}_full_V41{s}{tag}", "semstruct") for s in seeds],
            "valavg": [(f"{prefix}_valavg_V41{s}{tag}", "semstruct") for s in seeds],
            "onehop": [(f"{prefix}_onehop_V41{tag}", "semstruct")],
            "mlp": [(f"{prefix}_mlp_V41{tag}", "semstruct")],
            "lm_only": [(f"{prefix}_full_V41{tag}", "lm_only")],
            "zero_delta": [(f"{prefix}_full_V41{tag}", "zero_delta")],
            "permuted": [(f"{prefix}_full_V41{tag}", "permuted")]}


def table_clusters(rows):
    """Scenarios that share a source or target table form one cluster (connected components), as the SF paper on GOBY."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for i, r in enumerate(rows):
        # one table is one unit whichever side it is on (the SF paper's rule, sf-matcher-study/layer3b/x86_cluster_ci.py)
        for t in (r["src"], r["tgt"]):
            if t in first:
                parent[find(i)] = find(first[t])
            else:
                first[t] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return [np.array(g) for g in groups.values()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="plain")
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--nomerge", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--goby", action="store_true", help="the GOBY runs (imported corpus, bootstrap over table clusters)")
    ap.add_argument("--cohort", default="remaining338",
                    help="Amendment 2b: remaining505 with SEMSTRUCT_RUN=reports/valentine549_protocol")
    args = ap.parse_args()
    root = Path(args.root)
    raw = {}
    if args.goby:
        sr.CASES_ROOT = HERE / "generated" / "goby-imported"
        # table identity from the raw files, as the SF paper's clustering (generated/goby-raw/convert_manifest.json)
        man = json.loads((HERE / "generated" / "goby-raw" / "convert_manifest.json").read_text(encoding="utf-8"))
        raw = {sc: (v["sha256"][f"{sc}_source.csv"], v["sha256"][f"{sc}_target.csv"])
               for sc, v in man["scenarios"].items()}
    suffix = f"__merge-none__eval{args.cap}" if args.nomerge else f"__eval{args.cap}"

    def folder(run: str, arm: str) -> Path:
        return root / run / "matrices" / args.regime / (arm + suffix)

    src = sources(args.nomerge, "goby_a2" if args.goby else "a2")
    present = {arm: [(r, a) for r, a in runs if folder(r, a).is_dir()] for arm, runs in src.items()}
    present = {arm: runs for arm, runs in present.items() if runs}
    if "full" not in present:
        print(f"no full matrices under {root} for {args.regime}, cap {args.cap}")
        return 1

    rows = []
    for case in sr.load_cases():
        paths = sr.case_paths(case, "plain")
        if not paths:
            continue
        gt = sr.read_ground_truth(paths[2])
        perms = permutations(gt.shape, f"{case['dataset']}/{case['scenario']}")
        rec = {"key": case["key"], "family": case["family"], "cohort": case.get("cohort", "all"),
               "src": raw[case["scenario"]][0] if raw else sr.hashlib.sha256(paths[0].read_bytes()).hexdigest()[:16],
               "tgt": raw[case["scenario"]][1] if raw else sr.hashlib.sha256(paths[1].read_bytes()).hexdigest()[:16]}
        ok = True
        for arm, runs in present.items():
            per_seed = []
            for run, name in runs:
                f = folder(run, name) / f"{case['key']}.csv"
                if not f.is_file():
                    ok = False
                    break
                S = sr.read_written_matrix(f)
                if S.shape != gt.shape or not np.isfinite(S).all():
                    ok = False
                    break
                m = tf_metrics(S, gt, perms)
                per_seed.append({k: float(m[k]) for k in ("J", "R_exp", "F1_Hr")})
            if not ok:
                break
            for k in ("J", "R_exp", "F1_Hr"):
                rec[f"{arm}:{k}"] = float(np.mean([p[k] for p in per_seed]))
                for i, p in enumerate(per_seed):
                    rec[f"{arm}@{i}:{k}"] = p[k]
        if ok:
            rows.append(rec)

    name = f"eval_{'goby_' if args.goby else ''}{args.regime}_cap{args.cap}{'_nomerge' if args.nomerge else ''}"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scenarios.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(SEED)
    lines = [f"# SemStruct, the revised setup (Amendment 2): {args.regime}, evaluation row cap {args.cap}"
             f"{', no merging' if args.nomerge else ''}", "",
             "EXPLORATORY. Paired differences, macro over families with >= 5 scenarios, 95% bootstrap over scenarios "
             f"stratified by family ({B} resamples). Seeds present: "
             + ", ".join(f"{arm} {len(runs)}" for arm, runs in present.items()), ""]
    if "zero_delta" in present and "lm_only" in present:
        zd = max(abs(r["zero_delta:J"] - r["lm_only:J"]) for r in rows)
        lines += [f"Check: zero_delta reproduces lm_only, max |ΔJ| = {zd:.2e} over {len(rows)} scenarios.", ""]

    def contrast(sel, fams, a, b, clusters=None):
        res = {}
        for k in ("J", "R_exp", "F1_Hr"):
            d = np.array([r[f"{a}:{k}"] - r[f"{b}:{k}"] for r in sel])
            est = np.mean([d[ix].mean() for ix in fams.values()])
            if clusters is not None:
                boots = [d[np.concatenate([clusters[c] for c in rng.choice(len(clusters), len(clusters))])].mean()
                         for _ in range(B)]
            else:
                boots = [np.mean([d[rng.choice(ix, len(ix))].mean() for ix in fams.values()]) for _ in range(B)]
            res[k] = (est, np.percentile(boots, 2.5), np.percentile(boots, 97.5))
        return res

    for cohort in (("all",) if args.goby else (args.cohort, "all")):
        sel = [r for r in rows if cohort == "all" or r["cohort"] == cohort]
        fam = defaultdict(list)
        for i, r in enumerate(sel):
            fam[r["family"]].append(i)
        fams = {f: np.array(ix) for f, ix in fam.items() if len(ix) >= MIN_FAMILY}
        clusters = table_clusters(sel) if args.goby else None
        lines += [f"## {cohort}: {len(sel)} scenarios, families {', '.join(f'{f} {len(ix)}' for f, ix in fams.items())}"
                  + (f"; bootstrap over {len(clusters)} clusters of scenarios that share a table" if clusters else ""),
                  "", "| arm | J_tf | R_exp | F1_Hr |", "|---|---|---|---|"]
        for arm in present:
            v = {k: np.mean([np.mean([sel[i][f"{arm}:{k}"] for i in ix]) for ix in fams.values()])
                 for k in ("J", "R_exp", "F1_Hr")}
            lines.append(f"| {arm} | {v['J']:.3f} | {v['R_exp']:.3f} | {v['F1_Hr']:.3f} |")
        lines += ["", "| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |", "|---|---|---|---|---|"]
        for a, b, meaning in CONTRASTS:
            if a.split("@")[0] not in present or b not in present:
                continue
            res = contrast(sel, fams, a, b, clusters)
            j = res["J"]
            lines.append(f"| {a} − {b} | {meaning} | {j[0]:+.3f} [{j[1]:+.3f}, {j[2]:+.3f}] | "
                         f"{res['R_exp'][0]:+.3f} | {res['F1_Hr'][0]:+.3f} |")
        # per-seed contrasts against full's seed mean are not paired by seed; list each seed's own gate and valavg − full
        multi = [arm for arm in ("full", "valavg") if arm in present and len(present[arm]) > 1]
        if multi:
            lines += ["", "Per seed (J_tf, point estimates):", ""]
            n = min(len(present[arm]) for arm in multi)
            for i in range(n):
                parts = []
                if "lm_only" in present:
                    d = np.array([r[f"full@{i}:J"] - r["lm_only:J"] for r in sel])
                    parts.append(f"full − lm_only {np.mean([d[ix].mean() for ix in fams.values()]):+.3f}")
                if "valavg" in multi:
                    d = np.array([r[f"valavg@{i}:J"] - r[f"full@{i}:J"] for r in sel])
                    parts.append(f"valavg − full {np.mean([d[ix].mean() for ix in fams.values()]):+.3f}")
                lines.append(f"- seed {i + 1}: " + "; ".join(parts))
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
