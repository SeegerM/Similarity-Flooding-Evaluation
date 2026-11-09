"""EXPLORATORY X89 (2026-10-07): RQ1 across corpora: coupling checks per corpus, and tests of why the
relation is about zero on some corpora. Design and predictions in EXPLORATORY.md X89, written before computing.

Valentine (readable, recoded): per-scenario rows of layer 1, arm melnik, selected exactly as the paper's RQ1
(latex/sf-evaluation/gen_figures.py: layer1_rows, load_rows). New corpora: results/x82/rows.csv (X82 (f)), controls and
oracles excluded (score_r7.is_control). x = macro mean over groups of q's R_exp, d = macro mean gain in R_exp.

    ../harness/.venv-bdi/Scripts/python.exe explore_x89_rq1.py
Output: results/x89/{report.md, matchers.csv}
"""

from __future__ import annotations

import csv
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LATEX = HERE.parent / "latex" / "sf-evaluation"
for p in (HERE.parent / "harness", HERE / "tuning", HERE, HERE / "layer3b", LATEX):
    sys.path.insert(0, str(p))
import gen_figures as gf  # noqa: E402
from score_r7 import is_control  # noqa: E402

OUT = HERE / "results" / "x89"
SEED = 20261089
MIN_GROUP = {"valentine_plain": gf.MIN_FAMILY, "valentine_md5": gf.MIN_FAMILY}
STRONG = ["valentine_plain", "goby", "hdxsm", "madi_indep"]
ZERO = ["final_controlled_plain", "final_controlled_opaque", "gdcsm"]


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def cells() -> dict[str, list[tuple[str, str, str, float, float]]]:
    """corpus -> [(matcher, group, scenario key, q R_exp, SF R_exp)]."""
    out = defaultdict(list)
    for reg, name in (("plain", "valentine_plain"), ("md5", "valentine_md5")):
        rows = gf.layer1_rows(reg, "melnik")
        keep = {p["seed"] for p in gf.load_rows(rows)[0] if p["group"] != "ctrl"}
        for r in rows:
            if r["seed"] in keep:
                fam = r.get("family") or gf.family_of(r["dataset"], r["scenario"])
                out[name].append((r["seed"], fam, f"{r['dataset']}/{r['scenario']}", float(r["seed_R_exp"]),
                                  float(r["sf_R_exp"])))
    for r in csv.DictReader(open(HERE / "results" / "x82" / "rows.csv", encoding="utf-8")):
        if is_control(r["matcher"]) or not r.get("melnik_R"):
            continue
        out[r["corpus"]].append((r["matcher"], r["group"], f"{r['dataset']}/{r['scenario']}", float(r["q_R"]),
                                 float(r["melnik_R"])))
    return out


def per_matcher(cs, half: int | None = None, corpus: str = "") -> dict[str, tuple[float, float]]:
    """matcher -> (x, d), macro over groups; half = 0/1 keeps one seeded half of the scenarios in every group.
    A matcher is kept only if it has every scenario of the corpus (as the RQ1 inclusion rule)."""
    scen = {c[2] for c in cs}
    sizes = defaultdict(set)
    for c in cs:
        sizes[c[1]].add(c[2])
    sizes = {g: len(v) for g, v in sizes.items()}
    by = defaultdict(list)
    for c in cs:
        by[c[0]].append(c)
    out = {}
    for m, rs in by.items():
        if {r[2] for r in rs} != scen:
            continue
        gx, gd = defaultdict(list), defaultdict(list)
        for _, g, k, q, s in rs:
            if half is not None and (zlib.crc32(f"{SEED}|{k}".encode()) & 1) != half:
                continue
            gx[g].append(q)
            gd[g].append(s - q)
        # the paper's RQ1 rule: macro over groups with at least MIN_FAMILY scenarios (on Valentine this drops Wikidata);
        # applied to the full scenario counts, so both halves use the same groups
        big = [g for g in gx if sizes[g] >= MIN_GROUP.get(corpus, 1)]
        if big:
            out[m] = (float(np.mean([np.mean(gx[g]) for g in big])), float(np.mean([np.mean(gd[g]) for g in big])))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = cells()
    fam = {m: gf.META[m][0] for m in gf.META}
    res, pm_all = {}, {}
    for c, cs in sorted(data.items()):
        pm = per_matcher(cs, corpus=c)
        if len(pm) < 8:
            continue
        pm_all[c] = pm
        ms = sorted(pm)
        x, d = np.array([pm[m][0] for m in ms]), np.array([pm[m][1] for m in ms])
        h0, h1 = per_matcher(cs, 0, c), per_matcher(cs, 1, c)
        common = [m for m in ms if m in h0 and m in h1]
        rel = lambda i: (lambda r: 2 * r / (1 + r))(spearman([h0[m][i] for m in common], [h1[m][i] for m in common]))  # noqa: E731
        rx, rd = rel(0), rel(1)
        rho = spearman(x, d)
        fx, fy = defaultdict(list), defaultdict(list)
        for m in ms:
            fx[fam.get(m, "?")].append(pm[m][0])
            fy[fam.get(m, "?")].append(pm[m][1])
        res[c] = {"n": len(ms), "rho": rho, "oldham": spearman(x + d / 2, d),
                  "fam": spearman([np.mean(v) for v in fx.values()], [np.mean(v) for v in fy.values()]), "nfam": len(fx),
                  "rx": rx, "rd": rd,
                  "disatt": rho / np.sqrt(rx * rd) if rx > 0 and rd > 0 else float("nan"),
                  "sd_x": float(np.std(x)), "iqr_x": float(np.subtract(*np.percentile(x, [75, 25]))),
                  "ceil": float(np.mean(x > 0.9)), "floor": float(np.mean(x < 0.1)),
                  "sd_d": float(np.std(d)), "mean_d": float(np.mean(d)),
                  "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}
    with (OUT / "matchers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "matcher", "family", "x", "d"])
        for c, pm in pm_all.items():
            for m, (x, d) in sorted(pm.items()):
                w.writerow([c, m, fam.get(m, "?"), f"{x:.6f}", f"{d:.6f}"])
    lines = ["# X89: RQ1 across corpora: coupling checks and the near-zero corpora (EXPLORATORY)", "",
             "x = R_exp of the initial matcher, d = R_exp gain of SF as published; per matcher, macro over groups.", "",
             "## Coupling checks and reliability", "",
             "| corpus | matchers | ρ(x,d) | Oldham ρ | families (n) ρ | r_x | r_d | disattenuated ρ |", "|---|---|---|---|---|---|---|---|"]
    for c, r in res.items():
        lines.append(f"| {c} | {r['n']} | {r['rho']:+.2f} | {r['oldham']:+.2f} | ({r['nfam']}) {r['fam']:+.2f} | {r['rx']:.2f} | "
                     f"{r['rd']:.2f} | {r['disatt']:+.2f} |")
    lines += ["", "## Range of matcher quality and of gains", "",
              "| corpus | SD x | IQR x [q25, q75] | ceiling (x > 0.9) | floor (x < 0.1) | mean d | SD d |", "|---|---|---|---|---|---|---|"]
    for c, r in res.items():
        lines.append(f"| {c} | {r['sd_x']:.3f} | {r['iqr_x']:.3f} [{r['q25']:.2f}, {r['q75']:.2f}] | {r['ceil']:.0%} | "
                     f"{r['floor']:.0%} | {r['mean_d']:+.4f} | {r['sd_d']:.4f} |")
    lines += ["", "## Range matching: ρ in a strong corpus, restricted to the matchers whose x lies in a zero corpus's IQR", "",
              "| strong corpus | restricted to the x-range of | matchers kept | ρ restricted | ρ full |", "|---|---|---|---|---|"]
    for s in STRONG:
        if s not in pm_all:
            continue
        pm = pm_all[s]
        for z in ZERO:
            if z not in res:
                continue
            lo, hi = res[z]["q25"], res[z]["q75"]
            keep = [m for m, (x, _) in pm.items() if lo <= x <= hi]
            rr = spearman([pm[m][0] for m in keep], [pm[m][1] for m in keep]) if len(keep) >= 8 else float("nan")
            lines.append(f"| {s} | {z} [{lo:.2f}, {hi:.2f}] | {len(keep)} | {rr:+.2f} | {res[s]['rho']:+.2f} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
