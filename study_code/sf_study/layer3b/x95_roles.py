"""X95 step 1 (EXPLORATORY; EXPLORATORY.md X95): the dependency role term R13 (X72) and the value-shape role term V (X73) on any
corpus root, with the definitions of explore_genroles / explore_deproles / explore_matcherroles reused unchanged (only the
corpus root is a parameter instead of score_seeds.CORPUS[regime]).

R13(i, j) = 1 − mean |f(i) − f(j)| over the 13 features: X46's six (explore_deproles.roles) + X72's seven gpdep-weighted ones
(explore_genroles.genuine_features). V_m(i, j) likewise over (twin, typ, share) of measure m's within-table matrix
(explore_matcherroles.role_features), from <self_root>/<m>/<ds>/<sc>__<side>/seed.csv.

    python layer3b/x95_roles.py --coverage        -> results/x95/coverage.md
    python layer3b/x95_roles.py --check-valentine -> results/x95/valentine_check.md
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for p in (STUDY.parent / "harness", STUDY / "tuning", STUDY, STUDY / "reference", STUDY / "servers", HERE):
    sys.path.insert(0, str(p))
import explore_deproles as dep  # noqa: E402
import explore_genroles as gen  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
OUT = STUDY / "results" / "x95"
# corpus: (root with <ds>/<sc>/metadata, seed folder under reference/seeds)
ROOTS = {
    "valentine_plain": (GEN / "benchmark-full", "plain"),
    "valentine_md5": (GEN / "benchmark-full-md5", "md5"),
    "goby": (GEN / "goby-imported", "goby"),
    "hdxsm": (GEN / "hdxsm-usable", "hdxsm"),
    "gdcsm": (GEN / "gdcsm-imported", "gdcsm"),
    "madi": (GEN / "madi-imported", "madi"),
    "t2d": (GEN / "t2d-usable", "t2d"),
    "ctl": (GEN / "controlled-imported-v3", "ctl"),
    "final_controlled": (GEN / "final-controlled", "final_controlled"),
    "final_indep": (STUDY / "data" / "final_indep", "final_indep"),
    "indep2": (STUDY / "data" / "indep2_plain", "indep2"),
    "mimic": (STUDY / "data" / "mimic_omop", "mimic"),
}


def profiles_at(root: Path, ds: str, sc: str, side: str):
    """explore_deproles.profiles with the corpus root as a parameter (same parsing)."""
    meta = root / ds / sc / "metadata" / side
    fds, uccs, inds = [], [], []
    for tdir in sorted(p for p in meta.iterdir() if p.is_dir()) if meta.is_dir() else []:
        f = tdir / "FD_results.txt"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                m = dep.FD_RE.match(line.strip())
                if m:
                    fds.append((tuple(dep.col(x) for x in m.group("lhs").split(",") if x.strip()), dep.col(m.group("rhs"))))
        u = tdir / "UCC_results.txt"
        if u.is_file():
            for line in u.read_text(encoding="utf-8").splitlines():
                m = dep.BR_RE.match(line.strip())
                if m and m.group("cols").strip():
                    uccs.append(tuple(dep.col(x) for x in m.group("cols").split(",")))
    fi = meta / "inds.txt"
    if fi.is_file():
        for line in fi.read_text(encoding="utf-8").splitlines():
            m = dep.IND_RE.match(line.strip())
            if m and "," not in m.group("lhs") and "," not in m.group("rhs"):
                inds.append((dep.col(m.group("lhs")), dep.col(m.group("rhs"))))
    return fds, uccs, inds


def genuine_fds_at(root: Path, ds: str, sc: str, side: str):
    """explore_genroles.genuine_fds with the corpus root as a parameter."""
    meta = root / ds / sc / "metadata" / side
    out = []
    for f in sorted(meta.rglob("FD_results.txt")) if meta.is_dir() else []:
        for line in f.read_text(encoding="utf-8").splitlines():
            m = dep.FD_RE_FIXED.match(line.strip())
            if not m:
                continue
            a = gen.ANN.search(line)
            g = max(float(a.group(2)), 0.0) if a else 0.0
            out.append((tuple(dep.col(x) for x in m.group("lhs").split(",") if x.strip()), dep.col(m.group("rhs")), g))
    return out


def r13_features(root: Path, ds: str, sc: str, side: str, labels: list[str]) -> np.ndarray:
    F6, _ = dep.roles(labels, *profiles_at(root, ds, sc, side))
    Fg7, _ = gen.genuine_features(labels, genuine_fds_at(root, ds, sc, side))
    return np.concatenate([F6, Fg7], axis=1)


def r13(root: Path, ds: str, sc: str, rows: list[str], cols: list[str]) -> np.ndarray:
    return gen.sim(r13_features(root, ds, sc, "source", rows), r13_features(root, ds, sc, "target", cols))


def v_features(self_root: Path, m: str, ds: str, sc: str, side: str) -> dict:
    """explore_matcherroles.role_features with the self-matrix root as a parameter."""
    from smx.adapters import read_matrix
    f = self_root / m / ds / f"{sc}__{side}" / "seed.csv"
    if not f.is_file():
        return {}
    M, rws, _ = read_matrix(f)
    M = np.clip(np.nan_to_num(np.asarray(M, dtype=float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    out, n = {}, M.shape[0]
    for i, r in enumerate(rws):
        other = np.delete(M[i], i) if n > 1 else np.zeros(0)
        out[r.split(".", 1)[1]] = np.array([other.max(), other.mean(), (other >= 0.5).mean()]) if other.size else np.zeros(3)
    return out


def v_term(self_root: Path, m: str, ds: str, sc: str, rows: list[str], cols: list[str]) -> np.ndarray | None:
    fs, ft = v_features(self_root, m, ds, sc, "source"), v_features(self_root, m, ds, sc, "target")
    if not fs or not ft:
        return None
    S = np.array([fs.get(r.split(".", 1)[1], np.zeros(3)) for r in rows])
    T = np.array([ft.get(c.split(".", 1)[1], np.zeros(3)) for c in cols])
    return gen.sim(S, T)


def _cov(args):
    corpus, ds, sc = args
    root, _ = ROOTS[corpus]
    r = {"corpus": corpus, "dataset": ds, "scenario": sc}
    for side in ("source", "target"):
        meta = root / ds / sc / "metadata" / side
        r[f"{side}_fd"] = any(meta.rglob("FD_results.txt")) if meta.is_dir() else False
        r[f"{side}_ucc"] = any(meta.rglob("UCC_results.txt")) if meta.is_dir() else False
        r[f"{side}_ind"] = (meta / "inds.txt").is_file()
        g = genuine_fds_at(root, ds, sc, side)
        r[f"{side}_fds"] = len(g)
        r[f"{side}_gpdep_annotated"] = sum(1 for _, _, x in g if x > 0)
    return r


def scenarios(root: Path):
    return [(d.name, s.name) for d in sorted(p for p in root.iterdir() if p.is_dir())
            for s in sorted(p for p in d.iterdir() if p.is_dir())]


def coverage() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    jobs = [(c, ds, sc) for c, (root, _) in ROOTS.items() if root.is_dir() for ds, sc in scenarios(root)]
    with Pool(10) as pool:
        df = pd.DataFrame(pool.map(_cov, jobs, chunksize=8))
    lines = ["# X95 step 1: inputs for R13 and V per corpus", "",
             "| corpus | scenarios | FD both sides | UCC both | IND both | median FDs/side | FDs with gpdep > 0 | V self-matrices |",
             "|---|---|---|---|---|---|---|---|"]
    for c, g in df.groupby("corpus", sort=False):
        selfroot = STUDY / "reference" / "seeds" / f"self_{ROOTS[c][1]}"
        lines.append(f"| {c} | {len(g)} | {(g.source_fd & g.target_fd).mean():.0%} | {(g.source_ucc & g.target_ucc).mean():.0%} | "
                     f"{(g.source_ind & g.target_ind).mean():.0%} | {pd.concat([g.source_fds, g.target_fds]).median():.0f} | "
                     f"{(g.source_gpdep_annotated.sum() + g.target_gpdep_annotated.sum()) / max(1, g.source_fds.sum() + g.target_fds.sum()):.0%} | "
                     f"{'yes' if selfroot.is_dir() else 'no'} |")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "coverage.csv", index=False)
    (OUT / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def check_valentine() -> int:
    """R13 and V from this module vs explore_genroles / explore_matcherroles on Valentine plain: feature arrays cell for cell."""
    sys.stdout.reconfigure(encoding="utf-8")
    import explore_matcherroles as mr
    from score_seeds import cases_for
    from smx.adapters import read_matrix
    root = ROOTS["valentine_plain"][0]
    selfroot = STUDY / "reference" / "seeds" / "self_plain"
    d13 = dv = 0.0
    n = 0
    for ds, sc, _ in cases_for("plain")[::7]:
        q, rows, cols = read_matrix(STUDY / "reference" / "seeds" / "plain" / "levenshtein" / ds / sc / "seed.csv")
        rows, cols = list(rows), list(cols)
        mine = r13(root, ds, sc, rows, cols)
        ref = gen.sim(gen.side_features("plain", ds, sc, "source", rows)[0]["13"],
                      gen.side_features("plain", ds, sc, "target", cols)[0]["13"])
        d13 = max(d13, float(np.abs(mine - ref).max()))
        vm = v_term(selfroot, "char_distribution", ds, sc, rows, cols)
        fs, ft = mr.role_features("plain", "char_distribution", ds, sc, "source"), mr.role_features("plain", "char_distribution", ds, sc, "target")
        if vm is not None and fs and ft:
            S = np.array([fs[r.split(".", 1)[1]] for r in rows])
            T = np.array([ft[c.split(".", 1)[1]] for c in cols])
            dv = max(dv, float(np.abs(vm - gen.sim(S, T)).max()))
        n += 1
    lines = ["# X95 step 1: Valentine reproduction check", "",
             f"- {n} Valentine plain scenarios (every 7th of the protocol)",
             f"- R13 from this module vs explore_genroles.side_features '13' + sim: max |diff| {d13:.1e}",
             f"- V (char_distribution) vs explore_matcherroles.role_features + sim: max |diff| {dv:.1e}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "valentine_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--check-valentine", action="store_true")
    a = ap.parse_args()
    raise SystemExit(coverage() if a.coverage else check_valentine() if a.check_valentine else 0)
