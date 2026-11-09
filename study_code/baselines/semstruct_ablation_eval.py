"""SEMSTRUCT_ABLATION.md: the gate and the information-matched contrasts, with the SF paper's tie-free metrics.

Matrices: reports/semstruct_ablation_20261027/{full,onehop}/matrices/<regime>/<arm>/<case key>.csv (semstruct_run.py
score). Metrics: sf-matcher-study/layer3b/score_r7.metrics (expected Recall@GT, tie-free Hungarian F1 over the same 50
seeded row/column orders as the paper, J_tf = their mean). Aggregation as the paper: per-scenario paired differences,
mean per family, macro over families with >= 5 scenarios; 95% bootstrap over scenarios stratified by family (2,000
resamples). Cohort: remaining338 (the gate) and all 382. EXPLORATORY (Amendment 1).

    harness/.venv-isresmat/Scripts/python.exe -I harness/semstruct_ablation_eval.py [--regime plain] [--cases-root ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "layer3b"))
sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "tuning"))
sys.path.insert(0, str(HERE))
from score_r7 import metrics as tf_metrics  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
import semstruct_run as sr  # noqa: E402

ROOT = HERE / "reports" / "semstruct_ablation_20261027"
ARMS = {"full": ("full", "semstruct"), "lm_only": ("full", "lm_only"), "zero_delta": ("full", "zero_delta"),
        "full_permuted": ("full", "permuted"), "onehop": ("onehop", "semstruct"),
        "onehop_permuted": ("onehop", "permuted")}
CONTRASTS = [("full", "lm_only", "gate: structure over the header alone"),
             ("onehop", "full", "information-matched: one hop (own values) minus message passing"),
             ("onehop", "lm_only", "own values over the header alone"),
             ("full", "full_permuted", "row co-occurrence (full)"),
             ("onehop", "onehop_permuted", "value order (one hop; should be ~0)")]
B, SEED, MIN_FAMILY = 2000, 20261027, 5


def read_matrix(path: Path) -> np.ndarray:
    return sr.read_written_matrix(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="plain")
    ap.add_argument("--cases-root", default=None)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--arm-suffix", default="", help="e.g. __merge-none for the no-merging sensitivity")
    ap.add_argument("--cohort", default="remaining338",
                    help="Amendment 2b: remaining505 with SEMSTRUCT_RUN=reports/valentine549_protocol")
    args = ap.parse_args()
    if args.cases_root:
        sr.CASES_ROOT = Path(args.cases_root)
    root = Path(args.root)
    cases = sr.load_cases()
    rows = []
    for case in cases:
        paths = sr.case_paths(case, args.regime)
        if not paths:
            continue
        gt = sr.read_ground_truth(paths[2])
        perms = permutations(gt.shape, f"{case['dataset']}/{case['scenario']}")
        rec = {"key": case["key"], "family": case["family"], "cohort": case.get("cohort", "all"),
               "src": sr.hashlib.sha256(paths[0].read_bytes()).hexdigest()[:16],
               "tgt": sr.hashlib.sha256(paths[1].read_bytes()).hexdigest()[:16]}
        ok = True
        for arm, (sub, name) in ARMS.items():
            f = root / sub / "matrices" / args.regime / (name + args.arm_suffix) / f"{case['key']}.csv"
            if not f.is_file():
                ok = False
                break
            S = read_matrix(f)
            if S.shape != gt.shape or not np.isfinite(S).all():
                ok = False
                break
            m = tf_metrics(S, gt, perms)
            for k in ("J", "R_exp", "F1_Hr"):
                rec[f"{arm}:{k}"] = float(m[k])
        if ok:
            rows.append(rec)
    out = root / f"eval_{args.regime}{'_' + Path(args.cases_root).name if args.cases_root else ''}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scenarios.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(SEED)
    lines = [f"# SemStruct information-matched ablation ({args.regime}{', ' + str(args.cases_root) if args.cases_root else ''})",
             "", "EXPLORATORY (SEMSTRUCT_ABLATION.md, Amendment 1). Paired differences, macro over families with >= "
             f"{MIN_FAMILY} scenarios, 95% bootstrap over scenarios stratified by family ({B} resamples).", ""]
    zd = max(abs(r["zero_delta:J"] - r["lm_only:J"]) for r in rows)
    lines.append(f"Check: zero_delta reproduces lm_only, max |ΔJ| = {zd:.2e} over {len(rows)} scenarios.")
    lines.append("")
    for cohort in ([args.cohort, "all"] if not args.cases_root else ["all"]):
        sel = [r for r in rows if cohort == "all" or r["cohort"] == cohort]
        fam = defaultdict(list)
        for i, r in enumerate(sel):
            fam[r["family"]].append(i)
        fams = {f: np.array(ix) for f, ix in fam.items() if len(ix) >= MIN_FAMILY}
        # imported corpora: scenarios that share a source or target table form one cluster (connected components), and
        # the bootstrap resamples whole clusters, as the SF paper does on GDC-SM, HDXSM and GOBY
        clusters = None
        if args.cases_root:
            parent = list(range(len(sel)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            first = {}
            for i, r in enumerate(sel):
                for t in ("s" + r["src"], "t" + r["tgt"]):
                    if t in first:
                        parent[find(i)] = find(first[t])
                    else:
                        first[t] = i
            groups = defaultdict(list)
            for i in range(len(sel)):
                groups[find(i)].append(i)
            clusters = [np.array(g) for g in groups.values()]
            lines.append(f"{len(clusters)} clusters of scenarios that share a table; bootstrap over clusters.")
        lines += [f"## {cohort}: {len(sel)} scenarios, families {', '.join(f'{f} {len(ix)}' for f, ix in fams.items())}",
                  "", "| arm | J_tf | R_exp | F1_Hr |", "|---|---|---|---|"]
        for arm in ARMS:
            v = {k: np.mean([np.mean([sel[i][f"{arm}:{k}"] for i in ix]) for ix in fams.values()])
                 for k in ("J", "R_exp", "F1_Hr")}
            lines.append(f"| {arm} | {v['J']:.3f} | {v['R_exp']:.3f} | {v['F1_Hr']:.3f} |")
        lines += ["", "| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |", "|---|---|---|---|---|"]
        for a, b, meaning in CONTRASTS:
            res = {}
            for k in ("J", "R_exp", "F1_Hr"):
                d = np.array([r[f"{a}:{k}"] - r[f"{b}:{k}"] for r in sel])
                est = np.mean([d[ix].mean() for ix in fams.values()])
                if clusters is not None:
                    boots = []
                    for _ in range(B):
                        pick = rng.choice(len(clusters), len(clusters))
                        boots.append(d[np.concatenate([clusters[c] for c in pick])].mean())
                else:
                    boots = [np.mean([d[rng.choice(ix, len(ix))].mean() for ix in fams.values()]) for _ in range(B)]
                res[k] = (est, np.percentile(boots, 2.5), np.percentile(boots, 97.5))
            j = res["J"]
            lines.append(f"| {a} − {b} | {meaning} | {j[0]:+.3f} [{j[1]:+.3f}, {j[2]:+.3f}] | "
                         f"{res['R_exp'][0]:+.3f} | {res['F1_Hr'][0]:+.3f} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
