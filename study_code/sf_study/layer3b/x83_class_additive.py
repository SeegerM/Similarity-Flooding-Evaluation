"""X83 (EXPLORATORY; EXPLORATORY.md X83, design and predictions written 2026-10-07 before computing): additive class
controls on the new corpora, against SF with class nodes (X82 (c)).

Per corpus c, matcher m (the matchers of classes_<c>) and scenario: q = reference/seeds/<c>/<m>, D = datatype_nanfix,
classes from data/x82_classes_<c>/{shape,semtype} (the files the class arms used), terms as score_classfull.py:
    A_d = q+ + 5D;  A_p = A_d + 0.2 eq_shape;  A_s = A_d + 0.1 eq_semtype;  L_p = q + 0.2 eq_shape;  L_s = q + 0.2 eq_semtype;
    B_p / B_s = SF + 0.2 max(SF) eq   (only where <m>__melnik of the base stage is collected)
scored with score_r7.metrics (permutations of "<dataset>/<scenario>"), next to the SF arms melnik_shape / melnik_semtype /
melnik read from the X82 metric rows (x82_metrics/classfull_dev1.csv, so X82's deviations 1 and 2 hold here too).
Contrasts with score_x82.contrast / verdict, unchanged, in J, R_exp and F1_Hr.

    python layer3b/x83_class_additive.py      -> tuning/results/x83_class_additive/{rows.csv, contrasts.csv, report.md}
"""

from __future__ import annotations

import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))
import score_x82 as sx  # noqa: E402
import x82_metrics as xm  # noqa: E402
from score_classfull import edge_classes  # noqa: E402
from score_r7 import metrics, run_dir  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = STUDY / "tuning" / "results"
OUT = RES / "x83_class_additive"
LAM = 0.2
CONTRASTS = [  # (arm, base, Valentine sign; 0 = report only)
    ("A_p", "melnik_shape", +1), ("A_s", "melnik_semtype", +1),
    ("L_p", "melnik_shape", +1), ("L_s", "melnik_semtype", +1),
    ("A_p", "A_d", +1), ("A_s", "A_d", +1),
    ("A_d", "melnik_shape", 0), ("A_d", "melnik_semtype", 0),
    ("B_p", "melnik_shape", +1), ("B_s", "melnik_semtype", +1),
]


def job(args):
    c, ds, sc, matchers, sf_runs = args
    g = ground_truth_file(xm.ROOTS[c], ds, sc)
    seeds = STUDY / "reference" / "seeds" / c
    fd = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not fd.is_file():
        return [], Counter({"no_truth_or_D": 1})
    gt = read_matrix(g)[0] > 0.5
    D = read_matrix(fd)[0]
    perms = permutations(gt.shape, f"{ds}/{sc}")
    edges = STUDY / "data" / f"x82_classes_{c}"
    cls = {f: edge_classes(edges, f, ds, sc) for f in ("shape", "semtype")}
    dataset = ds.split("-")[0] if c == "final_controlled" else (ds.split("_")[0] if c == "final_indep" else ds)
    scen = sc if c == "madi" else f"{ds}/{sc}"            # X82 deviations 2 and 1
    rows, skip = [], Counter()
    for m in matchers:
        f = seeds / m / ds / sc / "seed.csv"
        if not f.is_file():
            continue
        q, rl, cl = read_matrix(f)
        if q.shape != gt.shape or not np.isfinite(q).all() or D.shape != q.shape or not np.isfinite(D).all():
            skip["shape_or_nonfinite"] += 1
            continue
        eq = {k: (np.array([v[r] for r in rl])[:, None] == np.array([v[x] for x in cl])[None, :]).astype(float)
              for k, v in cls.items()}
        Ad = np.clip(q, 0, None) + 5.0 * D
        mats = {"A_d": Ad, "A_p": Ad + 0.2 * eq["shape"], "A_s": Ad + 0.1 * eq["semtype"],
                "L_p": q + LAM * eq["shape"], "L_s": q + LAM * eq["semtype"]}
        r = sf_runs.get(m)
        o = one(r / ds / sc / "_outputs" / "MatchingStepLine2") if r is not None else None
        if o is not None:
            S = read_matrix(o)[0]
            if S.shape == q.shape and np.isfinite(S).all():
                mats["B_p"] = S + LAM * S.max() * eq["shape"]
                mats["B_s"] = S + LAM * S.max() * eq["semtype"]
        for arm, M in mats.items():
            x = metrics(M, gt, perms)
            rows.append({"corpus": c, "dataset": dataset, "scenario": scen, "matcher": m, "arm": arm,
                         "J": x["J"], "R_exp": x["R_exp"], "F1_Hr": x["F1_Hr"], "stage": "x83_offline"})
    return rows, skip


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sfrows = pd.read_csv(RES / "x82_metrics" / "classfull_dev1.csv")
    jobs, notes = [], []
    for c in xm.ROOTS:
        runs = xm.arm_dirs("classfull", c)
        matchers = sorted({m for (m, arm) in runs if arm.startswith("melnik_")})
        sf_runs = {m: r for (m, arm), (_, r) in runs.items() if arm == "melnik"}
        root = xm.ROOTS[c]
        scen = [(d.name, s.name) for d in sorted(root.iterdir()) if d.is_dir() for s in sorted(d.iterdir()) if s.is_dir()]
        jobs += [(c, ds, sc, matchers, sf_runs) for ds, sc in scen]
        notes.append(f"{c}: {len(scen)} scenarios, {len(matchers)} matchers, melnik base for {len(sf_runs)}")
    rows, skip = [], Counter()
    with Pool(10) as pool:
        for r, s in pool.imap_unordered(job, jobs, chunksize=4):
            rows += r
            skip += s
    OUT.mkdir(parents=True, exist_ok=True)
    add = pd.DataFrame(rows)
    add.to_csv(OUT / "rows.csv", index=False)
    df = sx.split_corpus(pd.concat([sfrows, add], ignore_index=True))
    rng = np.random.default_rng(sx.SEED)
    res = []
    for arm, base, sign in CONTRASTS:
        for metric in ("J", "R_exp", "F1_Hr"):
            for corpus in sx.INDEPENDENT + sx.BESIDE:
                d = df[df.corpus == corpus]
                r = sx.contrast(d, arm, base, metric, rng) if not d.empty else None
                if r is None:
                    continue
                res.append({"contrast": f"{arm} − {base}", "metric": metric, "valentine_sign": sign, "corpus": corpus, **r,
                            "verdict": sx.verdict(r, sign) if sign else "report", "independent": corpus in sx.INDEPENDENT})
    out = pd.DataFrame(res)
    out.to_csv(OUT / "contrasts.csv", index=False)
    lines = ["# X83: additive class controls vs SF with class nodes (EXPLORATORY)", "", *[f"- {n}" for n in notes],
             f"- skipped: {dict(skip)}", "", "| contrast | metric | corpus | estimate [95% CI] | scenarios | matchers (> 0) | verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in res:
        ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]" if np.isfinite(r["lo"]) else f"(n = {r['scenarios']}, no CI)"
        lines.append(f"| {r['contrast']} | {r['metric']} | {r['corpus']} | {r['estimate']:+.4f} {ci} | {r['scenarios']} | "
                     f"{r['matchers']} ({r['matchers_positive']}) | {r['verdict']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
