"""EXPLORATORY X52, stage 2 (2026-10-05): the composite recipes frozen on Valentine, scored unchanged on the held-out
corpora. Written before any of the needed runs existed. Paths changed to x52_<corpus> (13:40)
before any run existed: the registered score_e2.py reads every folder in e2_<corpus>.

Recipes (results/<regime>/composite_recipe.json, chosen on all Valentine families by explore_composite.py):
  plain recipe -> T2D (all usable scenarios) and the readable cell of the controlled corpus (ctl plain)
  md5 recipe   -> the opaque cell of the controlled corpus (ctl opaque)
Composite A = mean_k unit(q_k) over the recipe's base + beta D + g_c eq_class + g_r R, with
  D = reference/seeds/<corpus>/datatype_nanfix; eq from layer3b/f7.column_classes (shape) or f13.column_types
  (semtype) computed on the scenario's source and target tables (MIXED = MIXED counts as equal, as Amendment 8);
  R = the role similarity of explore_deproles from the scenario's profiles (as Amendment 11).
Single matcher = the recipe regime's best single initial matcher on all Valentine families (plain: qgram2_dice;
md5: unary_statistics_invariant), with SF as published (tuning/results/x52_<corpus>/<m>__melnik) and SF with the
top-1 key graph (tuning/results/x52_<corpus>/<m>__ucc_top1) on it.
Fusion arms exactly as stage 1: F_std = fuse(SF_std(coma)) with SF_std from results/<corpus>/coma (the seed study's
own SF line), F_pub = fuse(SF as published seeded by coma, x52_<corpus>/coma__melnik), F_noSF = fuse(coma); channels
char3_soft, cupid, coma_instances; weight 0.35.
Statistic per cell: J_tf per scenario; mean within family (T2D one family; ctl Lahman, MusicBrainz), mean over
families; 95% bootstrap within family, 10,000 resamples, seed 20261021. Scenarios missing any arm of a contrast are
dropped from that contrast and counted. Nothing is refitted on these corpora.

    ../harness/.venv-bdi/Scripts/python.exe explore_composite_heldout.py --corpus t2d|ctl
Output: results/<corpus>/composite_heldout_{scores.csv,summary.md}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))
sys.path.insert(0, str(HERE / "layer3b"))

import score_layer1  # noqa: E402
from explore_composite import CHANNELS, J, fuse, load, sf_out, unit  # noqa: E402
from explore_deproles import profiles, roles  # noqa: E402
from f7 import column_classes  # noqa: E402
from f13 import column_types  # noqa: E402
from score_roles_reg import CORPORA, cases  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import ground_truth_file  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

RES = HERE / "tuning" / "results"
SINGLE = {"plain": "qgram2_dice", "md5": "unary_statistics_invariant"}
RECIPE_OF = {"plain": "plain", "opaque": "md5"}
B, SEED = 10_000, 20261021


def recipe(regime: str) -> dict:
    return json.loads((HERE / "results" / regime / "composite_recipe.json").read_text(encoding="utf-8"))


def table_classes(scen: Path, variant: str) -> dict[str, str]:
    fn = column_classes if variant == "shape" else column_types
    out = {}
    for side in ("source", "target"):
        for t in sorted((scen / side).glob("*.csv")):
            header, cls = fn(t)
            out.update({f"{t.stem}.{h}": c for h, c in zip(header, cls)})
    return out


def run_root(p: Path) -> Path | None:
    return score_layer1.run_dir(p) if p.is_dir() else None


def score_case(job):
    corpus, ds, sc, fam, reg = job
    rec = recipe(RECIPE_OF[reg])
    seeds = HERE / "reference" / "seeds" / corpus
    g = ground_truth_file(CORPORA[corpus], ds, sc)
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not f_d.is_file():
        return None
    D, rows, cols = read_matrix(f_d)
    rows, cols, gt = list(rows), list(cols), read_matrix(g)[0] > 0.5
    D = np.nan_to_num(D)
    if D.shape != gt.shape:
        return None
    perms = permutations(gt.shape, f"{ds}/{sc}")
    row = {"dataset": ds, "scenario": sc, "family": fam, "regime": reg}
    q = {m: load(seeds / m / ds / sc / "seed.csv", rows, cols)
         for m in {*rec["base"], SINGLE[RECIPE_OF[reg]], "coma", *CHANNELS}}
    if all(q[m] is not None for m in rec["base"]):
        base = np.mean([unit(q[m]) for m in rec["base"]], axis=0)
        cl = table_classes(CORPORA[corpus] / ds / sc, rec["class"])
        rc, cc = [cl.get(r) for r in rows], [cl.get(c) for c in cols]
        eq = np.array([[1.0 if a is not None and a == b else 0.0 for b in cc] for a in rc])
        Fs, _ = roles(rows, *profiles(corpus, ds, sc, "source"))
        Ft, _ = roles(cols, *profiles(corpus, ds, sc, "target"))
        R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
        row["composite"] = J(base + rec["beta"] * D + rec["g_c"] * eq + rec["g_r"] * R, gt, perms)
        row["base"] = J(base, gt, perms)
    single = SINGLE[RECIPE_OF[reg]]
    if q[single] is not None:
        row["single"] = J(q[single], gt, perms)
    for arm in ("melnik", "ucc_top1"):
        S = load(sf_out(run_root(RES / f"x52_{corpus}" / f"{single}__{arm}"), ds, sc), rows, cols)
        if S is not None:
            row[f"single_sf_{arm}"] = J(S, gt, perms)
    ch = [q[m] for m in CHANNELS]
    sf_std = load(sf_out(run_root(HERE / "results" / corpus / "coma"), ds, sc), rows, cols)
    sf_pub = load(sf_out(run_root(RES / f"x52_{corpus}" / "coma__melnik"), ds, sc), rows, cols)
    if all(c is not None for c in ch):
        if sf_std is not None:
            row["F_std"], row["SF_std_coma"] = J(fuse(sf_std, ch), gt, perms), J(sf_std, gt, perms)
        if sf_pub is not None:
            row["F_pub"], row["SF_pub_coma"] = J(fuse(sf_pub, ch), gt, perms), J(sf_pub, gt, perms)
        if q["coma"] is not None:
            row["F_noSF"], row["coma"] = J(fuse(q["coma"], ch), gt, perms), J(q["coma"], gt, perms)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("t2d", "ctl"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cs = cases(a.corpus)
    with Pool(a.workers) as pool:
        rows = [r for r in pool.imap_unordered(score_case, [(a.corpus, *c) for c in cs]) if r]
    out = HERE / "results" / a.corpus
    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (out / "composite_heldout_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(SEED)
    arms = [("composite (frozen recipe)", "composite"), ("its base alone", "base"), ("best single matcher", "single"),
            ("SF as published on it", "single_sf_melnik"), ("SF top-1 key graph on it", "single_sf_ucc_top1"),
            ("F_std (old leader)", "F_std"), ("F_pub", "F_pub"), ("fusion without SF", "F_noSF"),
            ("SF_std seeded by COMA", "SF_std_coma"), ("SF as published seeded by COMA", "SF_pub_coma"),
            ("COMA alone", "coma")]
    contrasts = [("composite − best single", "composite", "single"), ("composite − its base", "composite", "base"),
                 ("composite − SF as published on the single", "composite", "single_sf_melnik"),
                 ("composite − SF top-1 key on the single", "composite", "single_sf_ucc_top1"),
                 ("composite − F_std", "composite", "F_std"), ("composite − F_pub", "composite", "F_pub"),
                 ("F_std − fusion without SF", "F_std", "F_noSF"), ("F_pub − fusion without SF", "F_pub", "F_noSF")]
    lines = [f"# X52 stage 2 ({a.corpus}): frozen composite recipes on held-out data (EXPLORATORY)", "",
             f"{len(rows)} of {len(cs)} scenarios read. Recipes: plain {json.dumps(recipe('plain'))}; "
             f"md5 {json.dumps(recipe('md5'))}. J_tf, family macro, 95% CI; n = scenarios with both arms.", ""]
    for reg in sorted({r["regime"] for r in rows}):
        cell = [r for r in rows if r["regime"] == reg]

        def stat(x, y=None):
            use = [r for r in cell if x in r and (y is None or y in r)]
            fams = defaultdict(list)
            for r in use:
                fams[r["family"]].append(r[x] - (r[y] if y else 0.0))
            if not fams:
                return "n/a", 0
            per = [np.array(v) for v in fams.values()]
            est = float(np.mean([v.mean() for v in per]))
            boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per], axis=0)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return (f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}]" if y else f"{est:.4f} [{lo:.4f}, {hi:.4f}]"), len(use)

        lines += [f"## {a.corpus} {reg} (recipe: {RECIPE_OF[reg]})", "", "| arm | J_tf | n |", "|---|---|---|"]
        lines += [f"| {n} | {' | '.join(map(str, stat(k)))} |" for n, k in arms]
        lines += ["", "| contrast | J_tf | n |", "|---|---|---|"]
        lines += [f"| {n} | {' | '.join(map(str, stat(x, y)))} |" for n, x, y in contrasts] + [""]
    (out / "composite_heldout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
