"""Generate LaTeX macros for the four review checks (EXPLORATORY) from their outputs in tuning/results/review_checks/.
Writes output/review_numbers.tex; numbers are never typed by hand.

    python make_review_numbers.py
"""

from __future__ import annotations

import re
from math import comb

import numpy as np
import pandas as pd

from paths import OUT, STUDY
RC = STUDY / "tuning" / "results" / "review_checks"
OUT_FILE = OUT / "review_numbers.tex"
OVL = {"valentine_plain": "prOvlVal", "valentine_md5": None, "madi_products": "prOvlMadiP", "gdcsm": "prOvlGdc",
       "hdxsm": "prOvlHdx", "goby": "prOvlGoby", "madi_indep": "prOvlMadi"}
ONE_SOURCE = {"valentine_plain", "valentine_md5", "madi_products"}


def f2(x: float) -> str:
    return f"{x:+.2f}"


def f3(x: float) -> str:
    return f"{x:+.3f}"


def macro_value(name: str) -> float:
    for f in OUT.glob("*_numbers.tex"):
        m = re.search(rf"\\newcommand{{\\{name}}}{{([^}}]*)}}", f.read_text(encoding="utf-8"))
        if m:
            return float(m.group(1))
    raise KeyError(name)


def main() -> int:
    out = {}
    # 1. RQ1 artefact (readable Valentine): parse the generated table
    t = (RC / "rq1_artefact_plain.md").read_text(encoding="utf-8")
    for key, label in (("Rexp", "expected Recall@GT"), ("FHr", "tie-free F1")):
        row = next(l for l in t.splitlines() if l.startswith(f"| {label} |"))
        c = [x.strip() for x in row.strip("|").split("|")]
        nums = [float(x) for x in re.findall(r"[+-]\d+\.\d+", c[3])]
        out[f"rvSplit{key}"], out[f"rvSplit{key}Lo"], out[f"rvSplit{key}Hi"] = f2(nums[0]), f2(nums[1]), f2(nums[2])
        out[f"rvHead{key}"] = f2(float(c[4]))
        nl = [float(x) for x in re.findall(r"[+-]\d+\.\d+", c[5])]
        out[f"rvNull{key}"], out[f"rvNull{key}Lo"], out[f"rvNull{key}Hi"] = f2(nl[0]), f2(nl[1]), f2(nl[2])
        out[f"rvNullP{key}"] = f"{float(c[6]):.2f}"
        # relative gain under the same ceiling null (second table, added 2026-10-27)
        rel = t[t.index("Relative gain under the same ceiling null"):]
        row = next(l for l in rel.splitlines() if l.startswith(f"| {label} |"))
        c = [x.strip() for x in row.strip("|").split("|")]
        nl = [float(x) for x in re.findall(r"[+-]\d+\.\d+", c[2])]
        out[f"rvRelNull{key}"], out[f"rvRelNull{key}Lo"], out[f"rvRelNull{key}Hi"] = f2(nl[0]), f2(nl[1]), f2(nl[2])
        out[f"rvRelNullP{key}"] = f"{float(c[3]):.3f}"
    # 2. beta sweep
    b = pd.read_csv(RC / "beta_sweep.csv")
    j = b[(b.metric == "J") & (b.beta > 0)]
    shift = j.groupby("corpus").estimate.agg(lambda s: s.max() - s.min()).max()
    signs = j.groupby("corpus").estimate.agg(lambda s: int(np.sign(s).nunique() == 1)).min()
    out["rvBetaMin"], out["rvBetaMax"] = str(int(j.beta.min())), str(int(j.beta.max()))
    out["rvBetaShift"] = f"{shift:.3f}"
    out["rvBetaSignStable"] = "all" if signs == 1 else "not all"
    out["rvBetaCorpora"] = str(j.corpus.nunique())
    # 3. lockbox intervals
    t = (RC / "lockbox_ci.md").read_text(encoding="utf-8")
    keys = {"Tuned, per matcher": "Own", "Tuned, one for all matchers": "Uni", "Standard": "Prod",
            "No propagation (alpha = 0)": "AlphaZero", "SF as published (report only)": "Ref"}
    for line in t.splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if c and c[0] in keys:
            lo, hi = re.findall(r"[+-]\d+\.\d+", c[2])
            clo, chi = re.findall(r"[+-]\d+\.\d+", c[3])
            k = keys[c[0]]
            out[f"rvLbx{k}Lo"], out[f"rvLbx{k}Hi"] = f"{float(lo):+.3f}", f"{float(hi):+.3f}"
            out[f"rvLbx{k}CLo"], out[f"rvLbx{k}CHi"] = f"{float(clo):+.3f}", f"{float(chi):+.3f}"
        if c and c[0] == "SF as published - Tuned, per matcher":
            nums = re.findall(r"[+-]\d+\.\d+", " ".join(c[1:3]))
            out["rvLbxRefVsOwn"], out["rvLbxRefVsOwnLo"], out["rvLbxRefVsOwnHi"] = (f"{float(x):+.3f}" for x in nums[:3])
    out["rvNIMargin"] = "-0.01"
    # 4. corpus associations
    g = pd.read_csv(RC / "corpus_assoc_grid.csv")
    out["rvAssocTested"], out["rvAssocListed"] = str(len(g)), str(int(g.listed.sum()))
    out["rvAssocBHFive"] = str(int((g.listed & g.bh05).sum()))
    out["rvAssocBHTen"] = str(int((g.listed & g.bh10).sum()))
    e5 = b[(b.beta == 5) & (b.metric == "J")].set_index("corpus").estimate
    rows = [(c, (macro_value(m) if m else 0.0), e5[c]) for c, m in OVL.items() if c in e5]
    one = [r for r in rows if r[0] in ONE_SOURCE]
    split = all(r[2] > 0 for r in one) and all(r[2] < 0 for r in rows if r[0] not in ONE_SOURCE)
    out["rvConstrN"] = str(len(rows))
    out["rvConstrSplit"] = f"{len(rows)} of {len(rows)}" if split else "not all"
    out["rvConstrP"] = f"{1 / comb(len(rows), len(one)):.3f}"
    x = pd.Series([r[1] for r in rows]).rank().to_numpy()
    y = pd.Series([r[2] for r in rows]).rank().to_numpy()
    rho = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(20261026)
    perm = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(100_000)])
    out["rvOvlRho"], out["rvOvlP"] = f2(rho), f"{(np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (len(perm) + 1):.2f}"
    lines = ["% Generated by make_review_numbers.py from tuning/results/review_checks/ "
             "(review checks, 2026-10-26); EXPLORATORY; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in out.items()]
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
