"""Review check 1 (EXPLORATORY, after the ICDE review of the compressed draft, 2026-10-26): is answer (A) -- SF's gain
decreases with the initial matcher's quality, Spearman rho over matchers -- an artefact of (i) measuring quality and gain
on the same scenarios, or (ii) the ceiling (a strong matcher has little room left to improve)?

Data and aggregation exactly as the paper's rho (gen_figures.py: layer 1, arm melnik = SF as published, readable or
recoded Valentine, macro over families with >= 5 scenarios, non-control matchers); the script first reproduces the
published rho. Then, per metric (expected Recall@GT; tie-free F1), with quality always the matcher's expected Recall@GT:
  split-half   per family, scenarios split at random into halves A/B; quality from A, gain from B (and B/A); 1,000 splits.
               Independent halves remove the shared-noise coupling (regression to the mean) of (i).
  headroom     rho(quality, gain / (1 - quality)) and rho(quality, gain / mean possible gain), at the matcher level.
  ceiling null per scenario, SF's changes are permuted across matchers and clipped to what is feasible for each matcher
               there ([-q, 1-q] for its own value q); rho over matchers of the permuted gains, 1,000 times. This is the
               rho the bounds alone produce when gain is unrelated to the matcher.
Output: tuning/results/review_checks/rq1_artefact.md

    python review_checks/rq1_artefact.py [plain|md5]
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY.parent / "latex" / "sf-evaluation"))
import gen_figures as gf  # noqa: E402

# rerun 549 (2026-10-31): under SMX_VALENTINE=549 gen_figures reads the assembled 549 inputs and the output goes next to
# the 549 figures, never into review_checks/; SMX_REVIEW_DIR overrides the folder (verification runs)
OUT = (Path(os.environ["SMX_REVIEW_DIR"]) if os.environ.get("SMX_REVIEW_DIR")
       else STUDY.parent / "latex" / "sf-evaluation-549" / "review_checks" if gf.V549
       else STUDY / "tuning" / "results" / "review_checks")
R, SEED = 1000, 20261026


def rank(v):
    return np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)


def rho(x, y) -> float:
    return gf.spearman(list(map(float, x)), list(map(float, y)))


def main() -> int:
    regime = sys.argv[1] if len(sys.argv) > 1 else "plain"
    rows = gf.layer1_rows(regime, "melnik")
    pts, _ = gf.load_rows(rows)
    panel = sorted(p["seed"] for p in pts if p["group"] != "ctrl")
    pub = {"R_exp": rho([p["x"] for p in pts if p["seed"] in panel], [p["d_R_exp"] for p in pts if p["seed"] in panel]),
           "F1_Hr": rho([p["x"] for p in pts if p["seed"] in panel], [p["d_F1_Hr"] for p in pts if p["seed"] in panel])}
    # per (family, scenario) x matcher arrays
    cell = defaultdict(dict)
    for r in rows:
        if r["seed"] in panel:
            fam = r.get("family") or gf.family_of(r["dataset"], r["scenario"])
            cell[(fam, r["dataset"], r["scenario"])][r["seed"]] = r
    fam_n = defaultdict(int)
    for k in cell:
        fam_n[k[0]] += 1
    keys = sorted(k for k in cell if fam_n[k[0]] >= gf.MIN_FAMILY and all(m in cell[k] for m in panel))
    fams = sorted({k[0] for k in keys})
    idx = {f: np.array([i for i, k in enumerate(keys) if k[0] == f]) for f in fams}
    Q = np.array([[float(cell[k][m]["seed_R_exp"]) for k in keys] for m in panel])          # matcher x scenario
    G = {met: np.array([[float(cell[k][m][f"sf_{met}"]) - float(cell[k][m][f"seed_{met}"]) for k in keys] for m in panel])
         for met in ("R_exp", "F1_Hr")}
    QB = {met: np.array([[float(cell[k][m][f"seed_{met}"]) for k in keys] for m in panel]) for met in ("R_exp", "F1_Hr")}

    def macro(E, sel=None):
        return np.mean([E[:, (sel[f] if sel is not None else ix)].mean(axis=1) for f, ix in idx.items()], axis=0)

    q_all = macro(Q)
    full = {met: rho(q_all, macro(G[met])) for met in G}
    rng = np.random.default_rng(SEED)
    res = {}
    for met in G:
        sh = []
        for _ in range(R):
            A, Bh = {}, {}
            for f, ix in idx.items():
                p = rng.permutation(ix)
                A[f], Bh[f] = p[: len(p) // 2], p[len(p) // 2:]
            sh.append(rho(macro(Q, A), macro(G[met], Bh)))
            sh.append(rho(macro(Q, Bh), macro(G[met], A)))
        sh = np.array(sh)
        base = macro(QB[met])
        head = macro(G[met]) / np.clip(1 - base, 1e-9, None)
        nul, nul_rel = [], []
        g, qb = G[met], QB[met]
        for _ in range(R):
            perm = np.array([rng.permutation(len(panel)) for _ in range(g.shape[1])]).T     # per scenario
            gp = np.take_along_axis(g, perm, axis=0)
            gp = np.clip(gp, -qb, 1 - qb)
            nul.append(rho(q_all, macro(gp)))
            # 2026-10-27 (external assessment): the same null for the relative gain; same draws, no extra rng calls,
            # so the raw-null values above are unchanged
            nul_rel.append(rho(q_all, macro(gp) / np.clip(1 - base, 1e-9, None)))
        nul, nul_rel = np.array(nul), np.array(nul_rel)
        head_obs = rho(q_all, head)
        res[met] = dict(full=full[met], pub=pub[met], sh_med=np.median(sh), sh_lo=np.percentile(sh, 2.5),
                        sh_hi=np.percentile(sh, 97.5), sh_neg=(sh < 0).mean(), head=rho(q_all, head),
                        nul_med=np.median(nul), nul_lo=np.percentile(nul, 2.5), nul_hi=np.percentile(nul, 97.5),
                        p_nul=(np.sum(nul <= full[met]) + 1) / (R + 1),
                        rel_med=np.median(nul_rel), rel_lo=np.percentile(nul_rel, 2.5),
                        rel_hi=np.percentile(nul_rel, 97.5), p_rel=(np.sum(nul_rel <= head_obs) + 1) / (R + 1))
    name = {"R_exp": "expected Recall@GT", "F1_Hr": "tie-free F1"}
    lines = [f"# Answer (A): artefact checks, Valentine {regime} (EXPLORATORY review check)", "",
             f"{len(panel)} non-control initial matchers, {len(keys)} scenarios in {len(fams)} families "
             f"({', '.join(f'{f} {len(idx[f])}' for f in fams)}); SF as published (layer 1, arm melnik). Quality = the "
             f"matcher's expected Recall@GT alone. {R} random splits / permutations, seed {SEED}.", "",
             "| gain in | published rho | rho, this panel | split-half rho: median [2.5%, 97.5%] (share < 0) | "
             "rho with gain / headroom | ceiling-null rho: median [2.5%, 97.5%] | P(null <= observed) |",
             "|---|---|---|---|---|---|---|"]
    for met, x in res.items():
        lines.append(f"| {name[met]} | {x['pub']:+.2f} | {x['full']:+.2f} | {x['sh_med']:+.2f} [{x['sh_lo']:+.2f}, "
                     f"{x['sh_hi']:+.2f}] ({x['sh_neg']:.0%}) | {x['head']:+.2f} | {x['nul_med']:+.2f} "
                     f"[{x['nul_lo']:+.2f}, {x['nul_hi']:+.2f}] | {x['p_nul']:.3f} |")
    lines += ["", "Relative gain under the same ceiling null (gain / headroom of each permuted, clipped draw):", "",
              "| gain in | rho with gain / headroom | null: median [2.5%, 97.5%] | P(null <= observed) |", "|---|---|---|---|"]
    for met, x in res.items():
        lines.append(f"| {name[met]} | {x['head']:+.2f} | {x['rel_med']:+.2f} [{x['rel_lo']:+.2f}, {x['rel_hi']:+.2f}] | "
                     f"{x['p_rel']:.4f} |")
    lines += ["", "Reading: if the relation were only shared noise, the split-half rho would shrink towards 0; if it "
              "were only the ceiling, the observed rho would lie inside the ceiling-null band and the headroom-normalised "
              "rho would vanish."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rq1_artefact_{regime}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
