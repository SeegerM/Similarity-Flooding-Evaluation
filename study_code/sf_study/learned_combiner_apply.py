"""Apply the frozen leave-one-family-out learned_lofo combiners to new scenarios (Valentine 549 rerun).

learned_combiner.py trains one logistic regression per held-out family on the 382-case protocol
(results/plain/<base>/<run>/, bases and C taken from seeds_learned/plain/learned_lofo/provenance.json)
and writes the held-out family's matrices. This script does NOT retrain on 549 (that would change
the 382 matrices): it rebuilds exactly the same 382 training sets, fits the same models (lbfgs is
deterministic), checks that they reproduce the stored 382 matrices (reference/seeds/plain/learned_lofo),
freezes them (coefficients + intercept, full precision, in seeds_learned/<regime-out>/<name>/frozen_models.json), and applies the model of
family F to every new scenario of family F (the model never saw family F, so no leakage).

    ../harness/.venv-bdi/Scripts/python.exe learned_combiner_apply.py \
        --new-suffix pnew --new-corpus ../harness/generated/valentine-new168 --regime-out pnew
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learned_combiner as lc  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402  (path set by learned_combiner)


def load_case(root_runs, bases, key, corpus):
    mats, labels = [], None
    for b in bases:
        f = lc.one(root_runs[b] / key[0] / key[1] / "_outputs" / "MatchingStepLine1")
        if f is None:
            return None, "missing base " + b
        S, r, c = read_matrix(f)
        if not np.isfinite(S).all():
            return None, "non-finite base " + b
        mats.append(np.clip(S, -1.0, 1.0))
        labels = labels or (r, c)
    gt_file = lc.one(corpus / key[0] / key[1] / "ground_truth")
    if gt_file is None:
        return None, "no ground truth"
    gt = read_matrix(gt_file)[0] > 0.5
    if not all(m.shape == gt.shape for m in mats):
        return None, "shape mismatch"
    return (mats, gt, labels), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-suffix", default="plain")
    ap.add_argument("--name", default="learned_lofo")
    ap.add_argument("--new-suffix", required=True, help="results/<suffix> holding the new base runs")
    ap.add_argument("--new-corpus", required=True, help="scenario tree with manifest.json (family)")
    ap.add_argument("--regime-out", required=True, help="reference/seeds/<regime-out>/<name>/")
    ap.add_argument("--atol", type=float, default=0.0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    import sklearn

    prov = json.loads((HERE / "seeds_learned" / args.train_suffix / args.name / "provenance.json")
                      .read_text(encoding="utf-8"))
    bases = prov["bases"]
    C = float(prov["learner"].split("C=")[1].split(",")[0])
    corpus = lc.CORPUS[args.train_suffix.split("_")[0]]
    family_of = {(c["dataset"], c["scenario"]): c["family"]
                 for c in json.loads(lc.PROTOCOL.read_text(encoding="utf-8"))["cases"]}

    # --- rebuild the 382 training data exactly as learned_combiner.main ---
    root = HERE / "results" / args.train_suffix
    runs = {b: lc.latest(root / b) for b in bases}
    data = {}
    for scenario_dir in sorted(runs[bases[0]].glob("*/*")):
        key = (scenario_dir.parent.name, scenario_dir.name)
        if key not in family_of:
            continue
        case, _ = load_case(runs, bases, key, corpus)
        if case is not None:
            data[key] = case
    groups = {k: family_of[k] for k in data}
    by_group = defaultdict(list)
    for k, g in groups.items():
        by_group[g].append(k)
    assert {g: len(v) for g, v in by_group.items()} == prov["groups"], "training groups differ"

    ref_root = HERE / "reference" / "seeds" / args.train_suffix / args.name
    models, check = {}, {}
    for g, held_out in sorted(by_group.items()):
        train = [k for k in data if groups[k] != g]
        X = np.concatenate([lc.features(data[k][0]) for k in train])
        y = np.concatenate([data[k][1].ravel() for k in train])
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        model.fit(X, y)
        assert model.coef_.ravel().round(4).tolist() == prov["coefficients_per_held_out_group"][g], g
        models[g] = model
        n_ident = n_close = 0
        maxdiff = 0.0
        for k in held_out:
            mats, gt, (rows, cols) = data[k]
            P = model.predict_proba(lc.features(mats))[:, 1].reshape(gt.shape)
            R, rr, rc = read_matrix(ref_root / k[0] / k[1] / "seed.csv")
            assert (rr, rc) == (rows, cols), k
            d = float(np.abs(P - R).max())
            maxdiff = max(maxdiff, d)
            n_ident += int(np.array_equal(P, R))
            n_close += int(d <= args.atol)
        check[g] = {"scenarios": len(held_out), "bit_identical": n_ident, "max_abs_diff": maxdiff}
        print(f"verify {g}: {len(held_out)} scenarios, bit-identical {n_ident}, max|diff| {maxdiff:.3g}",
              flush=True)
    bad = {g: c for g, c in check.items() if c["max_abs_diff"] > args.atol}
    if bad:
        raise SystemExit(f"frozen models do not reproduce the stored 382 matrices: {bad}")

    # --- freeze ---
    out_root = HERE / "reference" / "seeds" / args.regime_out / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    frozen = {
        "trained_on": f"results/{args.train_suffix} (382-case protocol), leave-one-family-out, not retrained",
        "bases": bases, "C": C, "sklearn": sklearn.__version__, "numpy": np.__version__,
        "verification_on_382": check,
        "models": {g: {"coef": m.coef_.ravel().tolist(), "intercept": float(m.intercept_[0]),
                       "classes": m.classes_.tolist()} for g, m in models.items()},
    }

    # --- apply to the new scenarios ---
    new_root = HERE / "results" / args.new_suffix
    new_runs = {b: lc.latest(new_root / b) for b in bases}
    missing = [b for b, r in new_runs.items() if r is None]
    if missing:
        raise SystemExit(f"bases not run under {new_root}: {missing}")
    new_corpus = Path(args.new_corpus).resolve()
    written, skipped = defaultdict(int), {}
    for sc in sorted(p for p in new_corpus.glob("*/*") if (p / "manifest.json").is_file()):
        key = (sc.parent.name, sc.name)
        fam = json.loads((sc / "manifest.json").read_text(encoding="utf-8"))["family"]
        if fam not in models:
            skipped[f"{key[0]}/{key[1]}"] = f"no frozen model for family {fam}"
            continue
        case, why = load_case(new_runs, bases, key, new_corpus)
        if case is None:
            skipped[f"{key[0]}/{key[1]}"] = why
            continue
        mats, gt, (rows, cols) = case
        P = models[fam].predict_proba(lc.features(mats))[:, 1].reshape(gt.shape)
        lc.write_seed(out_root / key[0] / key[1] / "seed.csv", P, rows, cols)
        written[fam] += 1
    frozen.update({"applied_to": str(new_corpus), "base_runs": f"results/{args.new_suffix}",
                   "written_per_family": dict(written), "skipped": skipped,
                   "date": dt.date.today().isoformat()})
    meta_root = HERE / "seeds_learned" / args.regime_out / args.name
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "frozen_models.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"wrote {sum(written.values())} seeds under {out_root}: {dict(written)}; skipped {len(skipped)}")
    for k, v in skipped.items():
        print("  skipped", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
