"""EXPLORATORY X54 / X55 scorer (2026-10-05): Synthea -> OMOP and the Lahman/MusicBrainz multi-table corpora.
Written before any run on these corpora existed; predictions in EXPLORATORY.md X54 and X55.

Per (initial matcher, scenario), J_tf = (R_exp + F1_Hr)/2 (score_screen_metrics; 50 seeded tie orders) of:
  q                   the initial matrix (reference/seeds/<corpus>/<m>)
  SF as published     tuning/results/<runs>/<m>__melnik        (<runs> = x54_synthea or x55_indep2)
  SF with FD edges    .../<m>__fd
  SF top-1 key graph  .../<m>__ucc_top1
  A_d    = q+ + 5 D                       (Amendment 5's term; D = datatype_nanfix)
  A_dt   = A_d + 0.5 C                    (Amendment 9's table-context term, C = score_tablectx.table_context;
                                           lambda 0.5 = Amendment 9's choice in every fold, frozen here)
  A_role = A_d + 0.2 R                    (the role term, gamma as Amendment 11 on readable data)
  A_rt   = A_d + 0.2 R + 0.5 C            (both, reported)
  composite = X52's frozen plain recipe (explore_composite_heldout's construction), one value per scenario
SF matrices are clipped at 0, as in score_tablectx. Included matchers per cell: non-controls with q, D and the three SF
arms on every scenario of the cell. Statistic: per scenario the mean over included matchers; mean over the scenarios
of a cell; cells = SyntheaOMOP-Full, SyntheaOMOP-Pair (synthea) and Lahman, MusicBrainz (indep2, with their macro);
95% bootstrap over scenarios within cell, 10,000 resamples, seed 20261023.

    ../harness/.venv-bdi/Scripts/python.exe explore_multitable_ext.py --corpus synthea|indep2 [--workers 10]
Output: results/<corpus>/multitable_{scores.csv,summary.md}
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
import score_seeds  # noqa: E402
from explore_composite import J, load, sf_out, unit  # noqa: E402
from explore_composite_heldout import recipe, table_classes  # noqa: E402
from explore_deproles import CONTROLS, profiles, roles  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_tablectx import table_context  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

CORPORA = {"synthea": (HERE / "data" / "synthea_plain", "x54_synthea"),
           "indep2": (HERE / "data" / "indep2_plain", "x55_indep2")}
score_seeds.CORPUS.update({k: v[0] for k, v in CORPORA.items()})   # explore_deproles.profiles reads CORPUS
RES = HERE / "tuning" / "results"
sys.path.insert(0, str(HERE / "reference"))
import valentine549 as V  # noqa: E402  (HyFD fix: used only under SMX_VALENTINE=549)
# SMX_VALENTINE=549 (fixed HyFD profiles): the fd arm from x54fix_synthea / x55fix_indep2 (pending until collected),
# melnik and ucc_top1 from the old stage, the role vectors of A_role / A_rt from harness/generated_hyfdfix/<corpus>.
# Output results/multitable_fix/<corpus>/.
if V.HYFDFIX:
    score_seeds.CORPUS.update({k: V.FIXED_MIRROR / k for k in CORPORA})   # profiles only; truth from CORPORA
ARMS = ("melnik", "fd", "ucc_top1")
B, SEED = 10_000, 20261023


def cell_of(corpus: str, ds: str) -> str:
    return ds if corpus == "synthea" else ds.replace("Indep-", "")


def cases(corpus: str) -> list[tuple[str, str, str]]:
    root = CORPORA[corpus][0]
    return [(d.name, s.name, cell_of(corpus, d.name)) for d in sorted(p for p in root.iterdir() if p.is_dir())
            for s in sorted(p for p in d.iterdir() if p.is_dir())]


def run_dirs(corpus: str) -> dict[tuple[str, str], Path]:
    out = {}
    stages = [(CORPORA[corpus][1], ARMS)]
    if V.HYFDFIX:
        stage = CORPORA[corpus][1]
        stages = [(stage, tuple(a for a in ARMS if not V.is_fd_arm(a)))]
        if V.fix_ready(stage):
            stages.append((V.fix_stage(stage), tuple(a for a in ARMS if V.is_fd_arm(a))))
    for stage, arms in stages:
        root = RES / stage
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir() and "__" in d.name:
                    m, arm = d.name.split("__", 1)
                    rd = score_layer1.run_dir(d)
                    if arm in arms and rd is not None:
                        out[(m, arm)] = rd
    return out


def score_case(job):
    corpus, ds, sc, cell, matchers, dirs = job
    root = CORPORA[corpus][0]
    seeds = HERE / "reference" / "seeds" / corpus
    g = score_seeds.ground_truth_file(root, ds, sc)
    f_d = seeds / "datatype_nanfix" / ds / sc / "seed.csv"
    if g is None or not f_d.is_file():
        return []
    D, rows, cols = read_matrix(f_d)
    rows, cols, gt = list(rows), list(cols), read_matrix(g)[0] > 0.5
    D = np.nan_to_num(D)
    if D.shape != gt.shape:
        return []
    perms = permutations(gt.shape, f"{ds}/{sc}")
    Fs, _ = roles(rows, *profiles(corpus, ds, sc, "source"))
    Ft, _ = roles(cols, *profiles(corpus, ds, sc, "target"))
    R = 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
    out = []
    q_all = {}
    for m in matchers:
        q = load(seeds / m / ds / sc / "seed.csv", rows, cols)
        if q is None:
            continue
        q_all[m] = q
        row = {"seed": m, "dataset": ds, "scenario": sc, "cell": cell}
        qp = np.clip(q, 0, None)
        C = table_context(q, rows, cols)
        A_d = qp + 5.0 * D
        for name, M in (("q", qp), ("A_d", A_d), ("A_dt", A_d + 0.5 * C), ("A_role", A_d + 0.2 * R),
                        ("A_rt", A_d + 0.2 * R + 0.5 * C)):
            row[name] = J(M, gt, perms)
        for arm in ARMS:
            S = load(sf_out(dirs.get((m, arm)), ds, sc), rows, cols)
            if S is not None:
                row[arm] = J(np.clip(S, 0, None), gt, perms)
        out.append(row)
    rec = recipe("plain")
    if all(m in q_all for m in rec["base"]):
        base = np.mean([unit(q_all[m]) for m in rec["base"]], axis=0)
        cl = table_classes(root / ds / sc, rec["class"])
        rc, cc = [cl.get(r) for r in rows], [cl.get(c) for c in cols]
        eq = np.array([[1.0 if a is not None and a == b else 0.0 for b in cc] for a in rc])
        comp = J(base + rec["beta"] * D + rec["g_c"] * eq + rec["g_r"] * R, gt, perms)
        out.append({"seed": "_composite", "dataset": ds, "scenario": sc, "cell": cell, "composite": comp})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cs = cases(a.corpus)
    seeds_root = HERE / "reference" / "seeds" / a.corpus
    matchers = sorted(p.name for p in seeds_root.iterdir() if p.is_dir() and p.name not in CONTROLS)
    dirs = run_dirs(a.corpus)
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_case, [(a.corpus, d, s, c, matchers, dirs) for d, s, c in cs])
                for r in part]
    out = HERE / "results" / a.corpus
    if V.HYFDFIX:
        if not V.fix_ready(CORPORA[a.corpus][1]):
            print(f"{a.corpus}: pending ({V.fix_stage(CORPORA[a.corpus][1])} not collected); nothing written")
            return 0
        out = HERE / "results" / "multitable_fix" / a.corpus
    out.mkdir(parents=True, exist_ok=True)
    keys = ["seed", "dataset", "scenario", "cell", "q", "A_d", "A_dt", "A_role", "A_rt", *ARMS, "composite"]
    with (out / "multitable_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    val = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    cell_cases = defaultdict(list)
    for d, s, c in cs:
        cell_cases[c].append((d, s))
    rng = np.random.default_rng(SEED)
    lines = [f"# X54/X55 ({a.corpus}): multi-table corpora (EXPLORATORY)", "",
             "J_tf; per scenario the mean over included matchers; 95% bootstrap over scenarios within cell.", ""]
    need = ("q", "A_d", "A_dt", "A_role", "A_rt", *ARMS)
    contrasts = [("SF as published − q", "melnik", "q"), ("SF with FD edges − q", "fd", "q"),
                 ("SF top-1 key − q", "ucc_top1", "q"), ("A_d − SF as published", "A_d", "melnik"),
                 ("A_dt − SF as published", "A_dt", "melnik"), ("A_role − SF as published", "A_role", "melnik"),
                 ("A_role − SF top-1 key", "A_role", "ucc_top1"), ("A_rt − SF as published", "A_rt", "melnik"),
                 ("A_dt − A_d", "A_dt", "A_d")]
    per_cell = {}
    for cell, sc in sorted(cell_cases.items()):
        used = sorted(m for m in matchers if all((m, d, s) in val and all(k in val[(m, d, s)] for k in need)
                                                  for d, s in sc))
        lines += [f"## {cell} ({len(sc)} scenarios, {len(used)} matchers)", "", "| quantity | J_tf |", "|---|---|"]
        if not used:
            lines += ["| (no matcher with every arm on every scenario) | |", ""]
            continue
        for name, x, y in [("q", "q", None), ("SF as published", "melnik", None)] + contrasts:
            arr = np.array([np.mean([val[(m, d, s)][x] - (val[(m, d, s)][y] if y else 0.0) for m in used]) for d, s in sc])
            boots = np.array([rng.choice(arr, size=arr.size).mean() for _ in range(B)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            per_cell[(cell, name)] = (arr, boots)
            lines.append(f"| {name} | {arr.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        comp = [val.get(("_composite", d, s), {}).get("composite") for d, s in sc]
        if all(v is not None for v in comp):
            sfp = np.array([np.mean([val[(m, d, s)]["melnik"] for m in used]) for d, s in sc])
            lines.append(f"| X52 composite (frozen; one matcher) | {np.mean(comp):.4f} |")
            lines.append(f"| composite − SF as published (panel mean) | {np.mean(np.array(comp) - sfp):+.4f} |")
        lines.append("")
    if a.corpus == "indep2" and all((c, "A_dt − SF as published") in per_cell for c in ("Lahman", "MusicBrainz")):
        lines += ["## macro over Lahman and MusicBrainz", "", "| contrast | J_tf |", "|---|---|"]
        for name, *_ in contrasts:
            est = np.mean([per_cell[(c, name)][0].mean() for c in ("Lahman", "MusicBrainz")])
            boots = np.mean([per_cell[(c, name)][1] for c in ("Lahman", "MusicBrainz")], axis=0)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            lines.append(f"| {name} | {est:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    (out / "multitable_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
