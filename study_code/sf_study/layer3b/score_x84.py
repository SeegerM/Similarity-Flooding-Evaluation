"""X84 scorer (EXPLORATORY; EXPLORATORY.md X84, written 2026-10-07 before any X84 run): Melnik et al.'s input-noise test
with SF's actual reference configuration, scored against the same perturbed input.

Runs (A): tuning/results/x84_noise_<c>/<m>__<arm>__<variant>, c in valentine, hdxsm, goby; arm in melnik, lit_none
(the corpus's own gated melnik configuration, and that configuration with smx.sf.literalSeed=NONE); variant as
data/x84_noise/manifest.json. Initial matrices q' = data/x84_noise/seeds/<c>/<m>__<variant>/<ds>/<sc>/seed.csv.
Truth: score_seeds.ground_truth_file of the corpus (Valentine: harness/generated/benchmark-full). J_tf etc. with
score_r7.metrics and permutations(shape, "<ds>/<sc>").

Gate (refuse otherwise): for every matcher and case, SF at p0 equals the original arm cell for cell (max |diff| <= 1e-9):
valentine layer1_plain/<m>__{melnik,lit_none}; hdxsm, goby x81_<c>/<m>__melnik (no original lit_none there).

Per corpus, arm, level (p50/p100/p200: the mean over the 3 draws per scenario; flat; p0) and matcher, per scenario:
    delta   = J(SF(q')) − J(q')                 SF against the same perturbed input
    loss_q  = J(q') − J(q);   loss_sf = J(SF(q')) − J(SF(q))
    absorb  = loss_sf − loss_q                   > 0: SF loses less than its input
and the arm contrast delta(melnik) − delta(lit_none). Each is the mean over the 3 matchers, then over the 20 scenarios; 95%
percentile bootstrap over scenarios, 10,000 draws, seed 20261007 (20 scenarios per corpus, no strata). Also per matcher
(point estimates) and in R_exp and F1_Hr.

    python layer3b/score_x84.py [--runs-root DIR --gate-root-valentine DIR --gate-root-x81 DIR]   (the roots: tests only)
Writes tuning/results/x84_noise_scores/{rows.csv, summary.csv, report.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))
from score_r7 import metrics, run_dir  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import CORPUS, ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = STUDY / "tuning" / "results"
NOISE = STUDY / "data" / "x84_noise"
GEN = STUDY.parent / "harness" / "generated"
TRUTH = {"valentine": CORPUS["plain"], "hdxsm": GEN / "hdxsm-usable", "goby": GEN / "goby-imported"}
ARMS = ("melnik", "lit_none")
B, SEED = 10_000, 20261007
F = ("J", "R_exp", "F1_Hr")


def level(v: str) -> str:
    return v.split("_")[0]


def job(args):
    c, m, ds, sc, runs_root, gate_v, gate_x = args
    gt = read_matrix(ground_truth_file(TRUTH[c], ds, sc))[0] > 0.5
    perms = permutations(gt.shape, f"{ds}/{sc}")
    man = json.loads((NOISE / "manifest.json").read_text(encoding="utf-8"))
    rows, gates = [], []
    for v in man["variants"]:
        qv = read_matrix(NOISE / "seeds" / c / f"{m}__{v}" / ds / sc / "seed.csv")[0]
        mq = metrics(qv, gt, perms)
        for arm in ARMS:
            d = run_dir(runs_root / f"x84_noise_{c}" / f"{m}__{arm}__{v}")
            o = one(d / ds / sc / "_outputs" / "MatchingStepLine2") if d else None
            if o is None:
                continue
            S = read_matrix(o)[0]
            if S.shape != gt.shape or not np.isfinite(S).all():
                continue
            ms = metrics(S, gt, perms)
            rows.append({"corpus": c, "matcher": m, "dataset": ds, "scenario": sc, "arm": arm, "variant": v,
                         "level": level(v), **{f"q_{k}": mq[k] for k in F}, **{f"sf_{k}": ms[k] for k in F}})
            if v == "p0":
                ref = (gate_v / f"{m}__{arm}") if c == "valentine" else (gate_x / f"x81_{c}" / f"{m}__{arm}")
                rd = run_dir(ref)
                ro = one(rd / ds / sc / "_outputs" / "MatchingStepLine2") if rd else None
                if ro is not None:
                    gates.append((c, m, arm, ds, sc, float(np.abs(read_matrix(ro)[0] - S).max())))
    return rows, gates


def boot(x: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = x[idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(RES))
    ap.add_argument("--gate-root-valentine", default=str(RES / "layer1_plain"))
    ap.add_argument("--gate-root-x81", default=str(RES))
    ap.add_argument("--out", default=str(RES / "x84_noise_scores"))
    a = ap.parse_args()
    cases = json.loads((NOISE / "cases.json").read_text(encoding="utf-8"))
    man = json.loads((NOISE / "manifest.json").read_text(encoding="utf-8"))
    jobs = [(c, m, ds, sc, Path(a.runs_root), Path(a.gate_root_valentine), Path(a.gate_root_x81))
            for c, cs in cases.items() for m in man["matchers"] for ds, sc in cs]
    rows, gates = [], []
    with Pool(10) as pool:
        for r, g in pool.imap_unordered(job, jobs):
            rows += r
            gates += g
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    gmax = max((g[-1] for g in gates), default=float("nan"))
    expected = sum(len(cs) for cs in cases.values()) * len(man["matchers"]) * (len(ARMS) + 0) - \
        len(cases["hdxsm"]) * len(man["matchers"]) - len(cases["goby"]) * len(man["matchers"])   # no lit_none original
    lines = ["# X84: input noise with SF's actual reference configuration (EXPLORATORY)", "",
             f"- runs scored: {len(df)} of {man['scenario_arm_evaluations']} scenario-arm evaluations",
             f"- gate (SF at p0 = the original arm): {len(gates)} of {expected} cells compared, max |diff| {gmax:.2e}"]
    if not gates or gmax > 1e-9:
        lines.append("- **GATE FAILED or missing: nothing below is interpretable.**")
    # per scenario: draws averaged within a level, then matchers averaged
    k = ["corpus", "arm", "level", "matcher", "dataset", "scenario"]
    per = df.groupby(k)[[f"q_{f}" for f in F] + [f"sf_{f}" for f in F]].mean().reset_index()
    base = per[per.level == "p0"].set_index(["corpus", "arm", "matcher", "dataset", "scenario"])
    rng = np.random.default_rng(SEED)
    summ = []
    for f in F:
        per[f"delta_{f}"] = per[f"sf_{f}"] - per[f"q_{f}"]
        b = base.reindex(per.set_index(["corpus", "arm", "matcher", "dataset", "scenario"]).index)
        per[f"loss_q_{f}"] = per[f"q_{f}"].to_numpy() - b[f"q_{f}"].to_numpy()
        per[f"loss_sf_{f}"] = per[f"sf_{f}"].to_numpy() - b[f"sf_{f}"].to_numpy()
        per[f"absorb_{f}"] = per[f"loss_sf_{f}"] - per[f"loss_q_{f}"]
    stats = [s for f in F for s in (f"delta_{f}", f"loss_q_{f}", f"loss_sf_{f}", f"absorb_{f}")]
    sc_mean = per.groupby(["corpus", "arm", "level", "dataset", "scenario"])[stats].mean().reset_index()
    order = ["p0", "p50", "p100", "p200", "flat"]
    for (c, arm, lv), g in sc_mean.groupby(["corpus", "arm", "level"]):
        for s in stats:
            x = g[s].dropna().to_numpy()
            if len(x):
                lo, hi = boot(x, rng)
                summ.append({"corpus": c, "arm": arm, "level": lv, "stat": s, "estimate": x.mean(), "lo": lo, "hi": hi,
                             "scenarios": len(x)})
    # arm contrast on delta
    w = sc_mean.pivot_table(index=["corpus", "level", "dataset", "scenario"], columns="arm", values=[f"delta_{f}" for f in F])
    for f in F:
        if ("delta_" + f, "melnik") not in w or ("delta_" + f, "lit_none") not in w:
            continue
        dd = (w[("delta_" + f, "melnik")] - w[("delta_" + f, "lit_none")]).dropna()
        for (c, lv), g in dd.groupby(level=["corpus", "level"]):
            x = g.to_numpy()
            lo, hi = boot(x, rng)
            summ.append({"corpus": c, "arm": "melnik−lit_none", "level": lv, "stat": f"delta_{f}", "estimate": x.mean(),
                         "lo": lo, "hi": hi, "scenarios": len(x)})
    sm = pd.DataFrame(summ)
    sm.to_csv(out / "summary.csv", index=False)
    lines += ["", "## J_tf: SF(q') − q' (delta), and absorb = loss of SF − loss of q' (95% CI over scenarios)", "",
              "| corpus | arm | level | delta | absorb | loss q' | loss SF |", "|---|---|---|---|---|---|---|"]
    for c in ("valentine", "hdxsm", "goby"):
        for arm in (*ARMS, "melnik−lit_none"):
            for lv in order:
                q = sm[(sm.corpus == c) & (sm.arm == arm) & (sm.level == lv)].set_index("stat")
                if "delta_J" not in q.index:
                    continue
                cell = lambda s: (f"{q.loc[s, 'estimate']:+.4f} [{q.loc[s, 'lo']:+.4f}, {q.loc[s, 'hi']:+.4f}]"  # noqa: E731
                                  if s in q.index else "–")
                lines.append(f"| {c} | {arm} | {lv} | {cell('delta_J')} | {cell('absorb_J')} | {cell('loss_q_J')} | "
                             f"{cell('loss_sf_J')} |")
    pm = per.groupby(["corpus", "arm", "level", "matcher"])[["delta_J", "absorb_J"]].mean().reset_index()
    lines += ["", "## Per matcher (point estimates, J_tf)", "", "| corpus | arm | level | matcher | delta | absorb |",
              "|---|---|---|---|---|---|"]
    for _, r in pm.iterrows():
        lines.append(f"| {r.corpus} | {r.arm} | {r.level} | {r.matcher} | {r.delta_J:+.4f} | {r.absorb_J:+.4f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
