"""EXPLORATORY X16 (user's request): the constant-damping control for FD weighting, and the best UCC
representation (top-1 UCC) together with weighted FD edges. Follows X14 (explore_graph_variants.py).

Same setting as X14: 4 panel matchers, 60 screen scenarios, plain, layer 1's Melnik settings.

1. Damping or discrimination? QUALITY and TOPK_HYBRID gained most in X14, and they damp most. Their
   actual mean FD-edge weight is measured (-Dsmx.sf.printWeightProfile=true, WEIGHTPROFILE lines,
   mean_retained), and CONSTANT is run on a grid of weights around it:
     fd_quality_p, fd_topk_p      X14's QUALITY / TOPK_HYBRID again, with the weight profile printed
     fd_const_005 ... fd_const_025 CONSTANT 0.05, 0.10, 0.15, 0.20, 0.25
   If CONSTANT at the matched mean weight reaches the weighted arms, the gain is damping; if not, the
   scores select better edges.
2. Top-1 UCC with weighted FD edges. dependencyTopK cuts FDs and UCCs alike, and every weighting also
   weights UCC edges, so this needs two off-by-default switches in an isolated copy of
   DependencyFilter / SchemaGraphBuilder (devjava_b, compiled to devclasses_b, put first on the
   classpath of these runs only; the shared source and the shipped R8 class are untouched):
     -Dsmx.sf.uccTopK=1         keep the best UCC per schema by primary-key score, FDs untouched
     -Dsmx.sf.uccUnweighted=true UCC edges at weight 1 under every weighting
   Arms (dev classes): ucc1 (UCC only; must equal X14's ucc_top1), ucc1_fd (unweighted FDs),
   ucc1_fd_const (0.35), ucc1_fd_quality, ucc1_fd_topk.
   Checks (dev classes, switches off): chk_ucc_quick = X14's ucc_quick and chk_fd_quality = X14's
   fd_quality, cell for cell.

    ../harness/.venv-bdi/Scripts/python.exe explore_gv2.py --prepare | --run --workers 10 | --score
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TUNING = HERE / "tuning"
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(TUNING))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "layer3b"))

import explore_graph_variants as gv  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import latest_run, one  # noqa: E402

SUFFIX = "explore_gv2_plain"
CFG_ROOT = TUNING / "configs" / SUFFIX
RES_ROOT = TUNING / "results" / SUFFIX
DEV_CLASSES = HERE / "devclasses_b"
FD = {"fdFilterThreshold": "0.0"}
UCC = {"uccFilterThreshold": "0.0"}
DEV = {"smx.sf.uccTopK": "1", "smx.sf.uccUnweighted": "true"}
ARMS = {  # arm: (params, props, dev classes?)
    "fd_quality_p": ({**FD, "constraintWeighting": "QUALITY"}, {"smx.sf.printWeightProfile": "true"}, False),
    "fd_topk_p": ({**FD, "constraintWeighting": "TOPK_HYBRID"}, {"smx.sf.topK": "1", "smx.sf.printWeightProfile": "true"}, False),
    **{f"fd_const_{int(w * 100):03d}": ({**FD, "constraintWeighting": "CONSTANT"}, {"smx.sf.constantWeight": f"{w:.2f}"}, False)
       for w in (0.05, 0.10, 0.15, 0.20, 0.25)},
    "chk_ucc_quick": (UCC, {}, True),
    "chk_fd_quality": ({**FD, "constraintWeighting": "QUALITY"}, {}, True),
    "ucc1": (UCC, {"smx.sf.uccTopK": "1"}, True),
    "ucc1_fd": ({**UCC, **FD}, {"smx.sf.uccTopK": "1"}, True),
    "ucc1_fd_const": ({**UCC, **FD, "constraintWeighting": "CONSTANT"}, {**DEV, "smx.sf.constantWeight": "0.35"}, True),
    "ucc1_fd_quality": ({**UCC, **FD, "constraintWeighting": "QUALITY"}, DEV, True),
    "ucc1_fd_topk": ({**UCC, **FD, "constraintWeighting": "TOPK_HYBRID"}, {**DEV, "smx.sf.topK": "1"}, True),
}
CHECKS = {"chk_ucc_quick": "ucc_quick", "chk_fd_quality": "fd_quality", "ucc1": "ucc_top1",
          "fd_quality_p": "fd_quality", "fd_topk_p": "fd_topk_hybrid"}


def prepare():
    if CFG_ROOT.exists():
        shutil.rmtree(CFG_ROOT)
    gv.CFG_ROOT, gv.RES_ROOT = CFG_ROOT, RES_ROOT   # reuse X14's config writer with this suffix
    datasets = sorted(d.name for d in gv.SCREEN_CORPUS.iterdir() if d.is_dir())
    for m in gv.MATCHERS:
        for arm, (params, props, dev) in ARMS.items():
            gv.write_config(f"gv2_{arm}__{m}", datasets, gv.SEEDS / m, params, props, {"seed": m, "arm": arm, "dev": dev})
    print(f"wrote {len(gv.MATCHERS) * len(ARMS)} configurations under {CFG_ROOT}")


def run(workers: int, xmx: str):
    from r8_smoke import run as run_with_classes   # runner: optional classes dir first on the classpath
    from run_seeds import classpath
    configs = sorted(p for p in CFG_ROOT.iterdir() if p.is_dir())
    todo = [c for c in configs if not (RES_ROOT / c.name / "DONE").is_file()]
    print(f"{len(configs)} configurations, {len(todo)} to run, {workers} workers", flush=True)
    deps = classpath()
    with ProcessPoolExecutor(workers) as pool:
        futs = {}
        for c in todo:
            dev = json.loads((c / "meta.json").read_text(encoding="utf-8"))["dev"]
            futs[pool.submit(run_with_classes, c, RES_ROOT / c.name, deps, xmx, str(DEV_CLASSES) if dev else "")] = c
        for f in as_completed(futs):
            label, code, secs = f.result()
            print(f"{label:45s} {'ok' if code == 0 else f'FAILED {code}'} ({secs:.0f}s)", flush=True)


def score_one(label: str) -> list[dict]:
    meta = json.loads((CFG_ROOT / label / "meta.json").read_text(encoding="utf-8"))
    run = latest_run(RES_ROOT / label)
    ref = None
    if meta["arm"] in CHECKS:
        ref = latest_run(gv.RES_ROOT / f"gv_{CHECKS[meta['arm']]}__{meta['seed']}")
    rows = []
    for scen in sorted(run.glob("*/*")):
        ds, sc = scen.parent.name, scen.name
        gt = read_matrix(one(gv.SCREEN_CORPUS / ds / sc / "ground_truth"))[0] > 0.5
        q = read_matrix(Path(meta["seed_dir"]) / ds / sc / "seed.csv")[0]
        f_s = one(scen / "_outputs" / "MatchingStepLine2")
        if f_s is None or q.shape != gt.shape:
            continue
        S = np.clip(read_matrix(f_s)[0], 0.0, None)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": meta["seed"], "arm": meta["arm"], "dataset": ds, "scenario": sc}
        for name, M in (("q", np.clip(q, 0.0, None)), ("sf", S)):
            mm = all_metrics(M, gt, perms)
            for k in gv.METRICS:
                row[f"{name}_{k}"] = mm[k]
        if ref is not None:
            f_r = one(ref / ds / sc / "_outputs" / "MatchingStepLine2")
            row["check_maxdiff"] = float(np.abs(S - np.clip(read_matrix(f_r)[0], 0.0, None)).max()) if f_r else float("nan")
        rows.append(row)
    return rows


def weight_profiles(arm: str) -> float:
    """Mean FD-edge weight of an arm's runs: mean_retained of each WEIGHTPROFILE line (one per schema),
    weighted by its number of retained edges (slots). The JVM prints decimals with the locale's comma."""
    num = den = 0.0
    for d in RES_ROOT.glob(f"gv2_{arm}__*"):
        for line in (d / "run.log").read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"WEIGHTPROFILE .*?slots=(\d+).*?mean_retained=([\d.,]+)", line)
            if m and int(m.group(1)) > 0:
                num += int(m.group(1)) * float(m.group(2).replace(",", "."))
                den += int(m.group(1))
    return num / den if den else float("nan")


def score(workers: int):
    labels = sorted(p.name for p in RES_ROOT.iterdir() if (p / "DONE").is_file())
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(score_one, labels) for r in part]
    keys = sorted({k for r in rows for k in r})
    with (HERE / "results" / "plain" / "gv2_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # reference arms from X14 for the comparisons
    x14 = list(csv.DictReader(open(HERE / "results" / "plain" / "graph_variants_scores.csv", encoding="utf-8")))
    allrows = rows + [{**r, "arm": "x14_" + r["arm"]} for r in x14]
    fam = {(c["dataset"], c["scenario"]): c["family"] for c in json.loads(gv.PROTOCOL.read_text(encoding="utf-8"))["cases"]}
    by = defaultdict(list)
    for r in allrows:
        by[(r["seed"], r["arm"])].append(r)

    def macro(rs, k):
        acc = defaultdict(list)
        for r in rs:
            acc[fam[(r["dataset"], r["scenario"])]].append(float(r[k]))
        big = [v for v in acc.values() if len(v) >= gv.MIN_FAMILY]
        return sum(sum(v) / len(v) for v in big) / len(big)

    seeds = list(gv.MATCHERS)

    def eff(arm, base):
        d = {"R": [], "F": [], "J": []}
        for s in seeds:
            a, b = by[(s, arm)], by[(s, base)]
            r = macro(a, "sf_R_exp") - macro(b, "sf_R_exp")
            f = macro(a, "sf_F1_Hr") - macro(b, "sf_F1_Hr")
            d["R"].append(r), d["F"].append(f), d["J"].append((r + f) / 2)
        return {k: sum(v) / len(v) for k, v in d.items()} | {"up": sum(x > 1e-4 for x in d["J"])}

    lines = ["# X16: damping control and top-1 UCC with weighted FD edges (exploratory, plain, screen)", "",
             "## Checks (max |difference| of SF's output, cell for cell, over all scenarios and matchers)", ""]
    for arm, ref in CHECKS.items():
        diffs = [float(r["check_maxdiff"]) for r in rows if r["arm"] == arm and r.get("check_maxdiff") not in (None, "")]
        lines.append(f"- {arm} vs X14 {ref}: max |diff| {max(diffs) if diffs else float('nan'):.2e} over {len(diffs)} scenario runs")
    wq, wt = weight_profiles("fd_quality_p"), weight_profiles("fd_topk_p")
    lines += ["", f"## Mean FD-edge weight (WEIGHTPROFILE mean_retained): QUALITY {wq:.3f}, TOPK_HYBRID {wt:.3f}", "",
              "| arm | vs unweighted fd: ΔJ_tf | ΔR_exp | ΔF1_Hr | matchers up |", "|---|---|---|---|---|"]
    for arm in ["fd_const_005", "fd_const_010", "fd_const_015", "fd_const_020", "fd_const_025",
                "x14_fd_constant", "x14_fd_quality", "x14_fd_topk_hybrid"]:
        e = eff(arm, "x14_fd")
        lines.append(f"| {arm} | {e['J']:+.4f} | {e['R']:+.4f} | {e['F']:+.4f} | {e['up']} |")
    lines += ["", "## Top-1 UCC with FD edges, against `melnik` (X14) and against the same FD weighting without UCCs", "",
              "| arm | vs melnik: ΔJ_tf | ΔR_exp | ΔF1_Hr | vs same FD arm without UCC: ΔJ_tf |", "|---|---|---|---|---|"]
    pairs = {"ucc1": None, "ucc1_fd": "x14_fd", "ucc1_fd_const": "x14_fd_constant",
             "ucc1_fd_quality": "x14_fd_quality", "ucc1_fd_topk": "x14_fd_topk_hybrid"}
    for arm, nofd in pairs.items():
        e = eff(arm, "x14_melnik")
        e2 = eff(arm, nofd)["J"] if nofd else float("nan")
        lines.append(f"| {arm} | {e['J']:+.4f} | {e['R']:+.4f} | {e['F']:+.4f} | {e2:+.4f} |")
    lines += ["", "## SF − q (SF's output against the initial matcher's), all arms", "",
              "| arm | R_exp | F1_Hr | J_tf |", "|---|---|---|---|"]
    for arm in ["x14_melnik", "x14_fd", *[a for a in ARMS if not a.startswith("chk_")]]:
        r = sum(macro(by[(s, arm)], "sf_R_exp") - macro(by[(s, arm)], "q_R_exp") for s in seeds) / len(seeds)
        f = sum(macro(by[(s, arm)], "sf_F1_Hr") - macro(by[(s, arm)], "q_F1_Hr") for s in seeds) / len(seeds)
        lines.append(f"| {arm} | {r:+.4f} | {f:+.4f} | {(r + f) / 2:+.4f} |")
    (HERE / "results" / "plain" / "gv2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--xmx", default="4g")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.run:
        run(a.workers, a.xmx)
    if a.score:
        score(min(a.workers, 12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
