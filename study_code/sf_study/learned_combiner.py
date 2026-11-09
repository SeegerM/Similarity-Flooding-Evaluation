"""Learned matcher combination (LSD's meta-learner, YAM, corpus-based matching) as an SF seed.

LSD (Doan, Domingos & Halevy, SIGMOD 2001) combines base learners with a trained meta-learner;
YAM (Duchateau et al., CIKM 2009) learns the combination of matchers from examples; corpus-based
matching (Madhavan et al., ICDE 2005) learns from other, already matched schemas. All three need
labelled matches from *other* schemas, so they cannot be a per-pair Schematch matcher. This
script is that family at harness level:

  features   for each base seed k and pair (i, j): s_k(i,j), s_k(i,j) - max_j' s_k(i,j'),
             s_k(i,j) - max_i' s_k(i',j)   (raw score, and its margins to the row / column best)
  learner    logistic regression (class-balanced), scikit-learn
  protocol   leave-one-FAMILY-out: a scenario's combiner never saw any scenario of its family
             (ChEMBL, OpenData, TPC-DI, Magellan, Wikidata, GDC), because Valentine's scenario
             types within a family share tables. With a single family (a smoke run) it falls back
             to leave-one-scenario-out and says so.
  output     seeds_learned/<suffix>/<name>/<dataset>/<scenario>/seed.csv in Schematch's verbose
             format, which SF reads with initialMatrix: FILE (make_configs.py --file-seed).

The base seeds' first-line matrices are read from results/<suffix>/<seed>/ (latest run), so the
base seeds must have been run first (run_seeds.py). Controls (oracle*, constant, random,
embedding_random) are never used as features.

Usage (needs scikit-learn; the harness's .venv-bdi has it):
    ../harness/.venv-bdi/Scripts/python.exe learned_combiner.py --suffix plain_smoke \
        --bases levenshtein,prefix_suffix,wordnet_wup,containment,pmf,ks --name learned_lofo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))

from smx.adapters import read_matrix  # noqa: E402

CORPUS = {
    "plain": HERE.parent / "harness" / "generated" / "benchmark-full",
    "md5": HERE.parent / "harness" / "generated" / "benchmark-full-md5",
}
PROTOCOL = HERE.parent / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
CONTROLS = ("oracle", "constant", "random", "embedding_random")


def latest(seed_dir: Path) -> Path | None:
    runs = sorted(p for p in seed_dir.iterdir() if p.is_dir()) if seed_dir.is_dir() else []
    return runs[-1] if runs else None


def one(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    return files[0] if len(files) == 1 else None


def features(matrices: list[np.ndarray]) -> np.ndarray:
    cols = []
    for S in matrices:
        cols.append(S)
        cols.append(S - S.max(axis=1, keepdims=True))
        cols.append(S - S.max(axis=0, keepdims=True))
    return np.stack([c.ravel() for c in cols], axis=1)


def write_seed(path: Path, S: np.ndarray, rows: list[str], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Source\\Target"] + cols)
        for label, row in zip(rows, S):
            w.writerow([label] + [repr(float(v)) for v in row])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--bases", required=True, help="comma-separated seed names (run already)")
    ap.add_argument("--name", default="learned_lofo")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength; fixed, not tuned")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    bases = [b for b in args.bases.split(",") if b]
    if any(b.startswith(c) for b in bases for c in CONTROLS):
        raise SystemExit("controls cannot be combiner features")
    regime = args.suffix.split("_")[0]
    corpus = CORPUS[regime]
    root = HERE / "results" / args.suffix
    family_of = {(c["dataset"], c["scenario"]): c["family"]
                 for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]}

    # scenario -> list of base matrices (+ labels from the first base)
    runs = {b: latest(root / b) for b in bases}
    missing = [b for b, r in runs.items() if r is None]
    if missing:
        raise SystemExit(f"bases not run under {root}: {missing}")
    data = {}
    for scenario_dir in sorted(runs[bases[0]].glob("*/*")):
        key = (scenario_dir.parent.name, scenario_dir.name)
        if key not in family_of:
            continue
        mats, labels = [], None
        for b in bases:
            f = one(runs[b] / key[0] / key[1] / "_outputs" / "MatchingStepLine1")
            if f is None:
                break
            S, r, c = read_matrix(f)
            if not np.isfinite(S).all():
                break
            mats.append(np.clip(S, -1.0, 1.0))
            labels = labels or (r, c)
        else:
            gt_file = one(corpus / key[0] / key[1] / "ground_truth")
            if gt_file is None:
                continue
            gt = read_matrix(gt_file)[0] > 0.5
            if all(m.shape == gt.shape for m in mats):
                data[key] = (mats, gt, labels)
    if not data:
        raise SystemExit("no scenario has every base matrix")

    groups = {k: family_of[k] for k in data}
    if len(set(groups.values())) < 2:
        print("only one family present: falling back to leave-one-SCENARIO-out")
        groups = {k: f"{k[0]}/{k[1]}" for k in data}

    out_root = HERE / "seeds_learned" / args.suffix / args.name
    by_group = defaultdict(list)
    for k, g in groups.items():
        by_group[g].append(k)
    weights = {}
    for g, held_out in sorted(by_group.items()):
        train = [k for k in data if groups[k] != g]
        X = np.concatenate([features(data[k][0]) for k in train])
        y = np.concatenate([data[k][1].ravel() for k in train])
        if y.sum() == 0 or y.sum() == len(y):
            raise SystemExit(f"training set without both classes when holding out {g}")
        model = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000)
        model.fit(X, y)
        weights[g] = model.coef_.ravel().round(4).tolist()
        for k in held_out:
            mats, gt, (rows, cols) = data[k]
            P = model.predict_proba(features(mats))[:, 1].reshape(gt.shape)
            write_seed(out_root / k[0] / k[1] / "seed.csv", P, rows, cols)

    (out_root / "provenance.json").write_text(json.dumps({
        "suffix": args.suffix, "bases": bases, "features": "score, row margin, column margin",
        "learner": f"LogisticRegression(C={args.C}, class_weight=balanced)",
        "protocol": "leave-one-family-out" if len(by_group) < len(data) else "leave-one-scenario-out",
        "groups": {g: len(v) for g, v in by_group.items()},
        "coefficients_per_held_out_group": weights,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(data)} learned seeds under {out_root} "
          f"({len(by_group)} held-out groups, {len(bases)} bases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
