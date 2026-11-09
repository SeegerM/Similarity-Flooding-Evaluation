"""X60 (EXPLORATORY, 2026-10-05): configurations and fused seeds for "two kinds of evidence in
different parts of SF's graph vs fused beforehand". Design and predictions: EXPLORATORY.md X60.

For every pair (A) and SF graph g in (melnik, fd, ucc_top1) it writes, under tuning/configs/x60_<regime>/:
  two_<A>__<B>__<g>   clone of <A>__<g>; adds smx.sf.idMatrixDir = reference/seeds/<regime>/<B>
                      (A on the column-name pairs, B on the column-ID pairs; overlay_x60)
  two_<B>__<A>__<g>   the swapped placement
  sum_<A>__<B>__<g>   clone of <A>__<g> with initialMatrixDir = reference/seeds/<regime>/x60sum_<A>__<B>  (A + B)
  unit_<A>__<B>__<g>  ... = x60unit_<A>__<B>  (mean of min-max-normalised A and B)
plus the twin gate twin_levenshtein__melnik (layer 1's config with the overlay and the switch off).
Base configurations: layer1_<regime>/<m>__{melnik,fd} and gvfull_<regime>/<m>__ucc_top1. Only resultsDir, the
initial-matrix directory and jvm.properties change. Every task runs with smx.cp.overlay=overlay_x60
(sf-matcher-study/devclasses_x60, built from devjava_x60).

The fused seeds are written in Schematch's seed.csv format (header "Source\\Target" + target labels; first column the
source labels) for every scenario of the regime. A non-finite cell in A or B stops the script: nothing is replaced.

    ../harness/.venv-bdi/Scripts/python.exe servers/make_x60_configs.py [--regime plain|md5|both]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY))
from score_seeds import cases_for  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

PAIRS = {"plain": [("coma", "pmf"), ("levenshtein", "pmf"), ("cupid", "char3_soft"), ("qgram2_dice", "containment"),
                   ("sbert_mpnet", "ks"), ("levenshtein", "jaro_winkler")],
         "md5": [("pmf", "unary_statistics_invariant"), ("role_uncertainty", "distinct_count")]}
GRAPHS = {"melnik": "layer1", "fd": "layer1", "ucc_top1": "gvfull"}
SEEDS = STUDY / "reference" / "seeds"
CONFIGS = STUDY / "tuning" / "configs"
RESULTS = STUDY / "tuning" / "results"
OVERLAY = "overlay_x60"


def unit(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    return np.zeros_like(m) if hi <= lo else (m - lo) / (hi - lo)


def write_seed(path: Path, M: np.ndarray, rows, cols) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("Source\\Target," + ",".join(cols) + "\n")
        for r, row in zip(rows, M):
            fh.write(r + "," + ",".join(repr(float(v)) for v in row) + "\n")


def fused_seeds(regime: str, a: str, b: str) -> tuple[Path, Path]:
    out_sum, out_unit = SEEDS / regime / f"x60sum_{a}__{b}", SEEDS / regime / f"x60unit_{a}__{b}"
    for d in (out_sum, out_unit):
        if d.exists():
            shutil.rmtree(d)
    n = 0
    for ds, sc, _ in cases_for(regime):
        A, rows, cols = read_matrix(SEEDS / regime / a / ds / sc / "seed.csv")
        B, rows_b, cols_b = read_matrix(SEEDS / regime / b / ds / sc / "seed.csv")
        if list(rows) != list(rows_b) or list(cols) != list(cols_b):
            raise SystemExit(f"labels differ between {a} and {b} in {ds}/{sc}")
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            raise SystemExit(f"non-finite seed in {a} or {b} at {ds}/{sc}; X60 does not replace values")
        write_seed(out_sum / ds / sc / "seed.csv", A + B, rows, cols)
        write_seed(out_unit / ds / sc / "seed.csv", 0.5 * (unit(A) + unit(B)), rows, cols)
        n += 1
    print(f"{regime} {a}+{b}: {n} fused seeds")
    return out_sum, out_unit


def clone(regime: str, base_label: str, stage: str, label: str, *, seed_dir: Path | None = None,
          id_dir: Path | None = None) -> None:
    src = CONFIGS / f"{stage}_{regime}" / base_label
    dst = CONFIGS / f"x60_{regime}" / label
    if not src.is_dir():
        raise SystemExit(f"missing base configuration {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    gen = (dst / "general.yaml").read_text(encoding="utf-8").splitlines()
    gen = [f'resultsDir: "{(RESULTS / f"x60_{regime}" / label).as_posix()}"' if g.startswith("resultsDir:") else g
           for g in gen]
    (dst / "general.yaml").write_text("\n".join(gen) + "\n", encoding="utf-8")
    if seed_dir is not None:
        sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8").splitlines()
        sl = [f'  initialMatrixDir: ["{seed_dir.as_posix()}"]' if s.strip().startswith("initialMatrixDir:") else s
              for s in sl]
        (dst / "second_line_matchers.yaml").write_text("\n".join(sl) + "\n", encoding="utf-8")
    jvm = [l.strip() for l in (dst / "jvm.properties").read_text(encoding="utf-8").splitlines()
           if l.strip() and not l.startswith("smx.cp.overlay")]
    jvm.append(f"smx.cp.overlay={OVERLAY}")
    if id_dir is not None:
        jvm.append(f"smx.sf.idMatrixDir={id_dir.as_posix()}")
    (dst / "jvm.properties").write_text("\n".join(jvm) + "\n", encoding="utf-8")
    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8")) if (dst / "meta.json").is_file() else {}
    meta.update({"stage": "x60", "label": label, "base": f"{stage}_{regime}/{base_label}",
                 "overlay": OVERLAY, "idMatrixDir": id_dir.as_posix() if id_dir else ""})
    if seed_dir is not None:  # keep meta.json consistent with the YAML (push ships meta's seed_dir; A)
        meta["seed_dir"] = seed_dir.as_posix()
        meta.setdefault("sf", {})["initialMatrixDir"] = seed_dir.as_posix()
    meta["jvm"] = {l.split("=", 1)[0]: l.split("=", 1)[1] for l in jvm}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5", "both"), default="both")
    a = ap.parse_args()
    for regime in (("plain", "md5") if a.regime == "both" else (a.regime,)):
        root = CONFIGS / f"x60_{regime}"
        if root.exists():
            shutil.rmtree(root)
        clone(regime, "levenshtein__melnik", "layer1", "twin_levenshtein__melnik")
        n = 1
        for ma, mb in PAIRS[regime]:
            s_dir, u_dir = fused_seeds(regime, ma, mb)
            for g, stage in GRAPHS.items():
                clone(regime, f"{ma}__{g}", stage, f"two_{ma}__{mb}__{g}", id_dir=SEEDS / regime / mb)
                clone(regime, f"{mb}__{g}", stage, f"two_{mb}__{ma}__{g}", id_dir=SEEDS / regime / ma)
                clone(regime, f"{ma}__{g}", stage, f"sum_{ma}__{mb}__{g}", seed_dir=s_dir)
                clone(regime, f"{ma}__{g}", stage, f"unit_{ma}__{mb}__{g}", seed_dir=u_dir)
                n += 4
        print(f"{regime}: {n} configurations in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
