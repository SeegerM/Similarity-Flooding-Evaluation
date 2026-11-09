"""External-review check (EXPLORATORY, 2026-11-03): robustness of the ceiling-only null behind Fig. 3 (answer (A)).

The published null (review_checks/rq1_artefact.py, unchanged; reproduced here first): per scenario, SF's changes are
permuted across the panel's matchers and each permuted change is clipped to what that matcher can still gain or lose on
that scenario, [-q, 1 - q] with q its own initial value of the same metric there; macro over families (>= 5 scenarios);
rho over matchers of (quality = macro expected Recall@GT alone, macro permuted gain). The relative row divides each
matcher's macro gain (observed or permuted) by its macro headroom 1 - q_m. 1,000 draws, seed 20261026.

Variants (same panel, same draws where possible):
  V1 matcher-level null    permute the matchers' macro gains across matchers and clip to [-q_m, 1 - q_m] at the matcher
                           level (instead of per scenario);
  V2 floor-scaled relative relative gain = gain / (1 - q_m) for gains >= 0 and gain / q_m for losses (a loss is bounded
                           by q_m), for the observed and the (published, scenario-level) null gains;
  V3 non-negative gains    only the matchers whose observed macro gain is >= 0; observed rho and the published null on
                           that subset.
For each variant and row (raw / relative x ranking / assignment): observed rho, null median [2.5%, 97.5%], and whether the
observed rho lies below the null band. Output (JSON + MD) in SMX_REVIEW_DIR, default
latex/sf-evaluation-presentation-549/review_checks/.

    SMX_VALENTINE=549 ../harness/.venv-bdi/Scripts/python.exe -B review_checks/xr_rq1_null_variants.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY.parent / "latex" / "sf-evaluation"))
import gen_figures as gf  # noqa: E402

OUT = Path(os.environ.get("SMX_REVIEW_DIR") or STUDY.parent / "latex" / "sf-evaluation-presentation-549" / "review_checks")
R, SEED = 1000, 20261026


def rho(x, y) -> float:
    return gf.spearman(list(map(float, x)), list(map(float, y)))


def band(v) -> tuple[float, float, float]:
    v = np.asarray(v)
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    rows = gf.layer1_rows("plain", "melnik")
    pts, _ = gf.load_rows(rows)
    panel = sorted(p["seed"] for p in pts if p["group"] != "ctrl")
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
    Q = np.array([[float(cell[k][m]["seed_R_exp"]) for k in keys] for m in panel])
    G = {met: np.array([[float(cell[k][m][f"sf_{met}"]) - float(cell[k][m][f"seed_{met}"]) for k in keys] for m in panel])
         for met in ("R_exp", "F1_Hr")}
    QB = {met: np.array([[float(cell[k][m][f"seed_{met}"]) for k in keys] for m in panel]) for met in ("R_exp", "F1_Hr")}

    def macro(E, sel=None):
        return np.mean([E[:, (sel[f] if sel is not None else ix)].mean(axis=1) for f, ix in idx.items()], axis=0)

    q_all = macro(Q)
    rng = np.random.default_rng(SEED)
    out = {"panel": len(panel), "scenarios": len(keys), "draws": R, "seed": SEED, "metrics": {}}
    for met in G:
        # replicate rq1_artefact.py's draws exactly (split-half first, then the null), so the published null reproduces
        for _ in range(R):
            for f, ix in idx.items():
                rng.permutation(ix)
        g, qb = G[met], QB[met]
        base = macro(qb)
        gain = macro(g)
        head = gain / np.clip(1 - base, 1e-9, None)

        def floor_rel(x):
            return np.where(x >= 0, x / np.clip(1 - base, 1e-9, None), x / np.clip(base, 1e-9, None))
        nonneg = gain >= 0
        nul, nul_rel, nul_floor, nul_nn, nul_nn_rel = [], [], [], [], []
        for _ in range(R):
            perm = np.array([rng.permutation(len(panel)) for _ in range(g.shape[1])]).T
            gp = np.clip(np.take_along_axis(g, perm, axis=0), -qb, 1 - qb)
            mg = macro(gp)
            nul.append(rho(q_all, mg))
            nul_rel.append(rho(q_all, mg / np.clip(1 - base, 1e-9, None)))
            nul_floor.append(rho(q_all, floor_rel(mg)))
            nul_nn.append(rho(q_all[nonneg], mg[nonneg]))
            nul_nn_rel.append(rho(q_all[nonneg], (mg / np.clip(1 - base, 1e-9, None))[nonneg]))
        # V1: matcher-level permutation of the macro gains, clipped at the matcher level (separate generator)
        r1 = np.random.default_rng(SEED + 1)
        v1, v1_rel = [], []
        for _ in range(R):
            gp = np.clip(gain[r1.permutation(len(panel))], -base, 1 - base)
            v1.append(rho(q_all, gp))
            v1_rel.append(rho(q_all, gp / np.clip(1 - base, 1e-9, None)))
        obs = {"raw": rho(q_all, gain), "rel": rho(q_all, head), "floor": rho(q_all, floor_rel(gain)),
               "nn_raw": rho(q_all[nonneg], gain[nonneg]), "nn_rel": rho(q_all[nonneg], head[nonneg])}
        res = {"published": {"raw": (obs["raw"], *band(nul)), "rel": (obs["rel"], *band(nul_rel))},
               "V1_matcher_level": {"raw": (obs["raw"], *band(v1)), "rel": (obs["rel"], *band(v1_rel))},
               "V2_floor_scaled": {"raw": (obs["raw"], *band(nul)), "rel": (obs["floor"], *band(nul_floor))},
               "V3_nonnegative": {"raw": (obs["nn_raw"], *band(nul_nn)), "rel": (obs["nn_rel"], *band(nul_nn_rel)),
                                  "matchers": int(nonneg.sum())}}
        for v in res.values():
            for k in ("raw", "rel"):
                o, med, lo, hi = v[k]
                v[k] = {"obs": o, "med": med, "lo": lo, "hi": hi, "below": bool(o < lo), "above": bool(o > hi)}
        out["metrics"][met] = res
    # reproduction of the published macros (review_numbers.tex: rvNull*, rvRelNull*, rvHead*; rq1: rqOneRho*)
    out["check"] = {met: {"raw_null": [round(out["metrics"][met]["published"]["raw"][k], 2) for k in ("med", "lo", "hi")],
                          "rel_null": [round(out["metrics"][met]["published"]["rel"][k], 2) for k in ("med", "lo", "hi")],
                          "obs_raw": round(out["metrics"][met]["published"]["raw"]["obs"], 2),
                          "obs_rel": round(out["metrics"][met]["published"]["rel"]["obs"], 2)} for met in G}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xr_rq1_null_variants.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    name = {"R_exp": "ranking", "F1_Hr": "assignment"}
    md = ["# Ceiling-only null: robustness variants (EXPLORATORY, external review 2026-11-03)", "",
          f"{len(panel)} matchers, {len(keys)} scenarios, {R} draws. 'below' = observed rho under the null's 2.5% bound.", "",
          "| variant | row | observed | null median [2.5%, 97.5%] | below |", "|---|---|---|---|---|"]
    for v in ("published", "V1_matcher_level", "V2_floor_scaled", "V3_nonnegative"):
        for met in G:
            for k in ("raw", "rel"):
                x = out["metrics"][met][v][k]
                md.append(f"| {v} | {k} {name[met]} | {x['obs']:+.2f} | {x['med']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] | "
                          f"{'yes' if x['below'] else 'no'} |")
    md += ["", f"V3 keeps {out['metrics']['R_exp']['V3_nonnegative']['matchers']} (ranking) / "
               f"{out['metrics']['F1_Hr']['V3_nonnegative']['matchers']} (assignment) matchers.", "",
           "Reproduction of the published null: " + json.dumps(out["check"])]
    (OUT / "xr_rq1_null_variants.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
