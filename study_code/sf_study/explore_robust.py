"""EXPLORATORY R1 (ideas from the related-work review): Melnik et al.'s robustness test,
replicated on our corpus.

Melnik, Garcia-Molina, Rahm (ICDE 2002, Sec. 6, Fig. 9) call SF "relatively insensitive to
'errors' in initial similarity values". Their test: "randomization of 50% means that every initial
similarity value was randomly increased or decreased by x percent, x in [-50%, 50%]. Negative
similarity was adjusted to zero", for randomisation levels up to 400%; and a "flatten" variant in
which "each non-zero value in sigma^0 was set to the same number computed as the average of all
positive similarity values". With their string matcher, accuracy fell from about 54% to 44% at
100% and to about 24% at 400%; flatten gave about 30%.

Here, for three initial matchers that differ in sparsity (prefix/suffix, the literal matcher of SF's
reference configuration, about half the cells zero; Levenshtein, dense and heavily tied; PMF,
dense with distinct values), each initial matrix q is perturbed as above, q' = max(0, q (1 + x)),
x ~ U[-p, p] per cell, p in {0, 25, 50, 100, 200, 400}% (three draws each, one for p = 0), or
flattened, and run through SF as a FILE initial matcher in two configurations:

    ref   Melnik's reference: QUICK without FDs, formula C, inverse average, NAME attachment
    std   the standard configuration: QUICK with all FDs, formula C, inverse product, NAME attachment

on the 60 screen scenarios (plain). Scored on the tie-free metrics for q' and for SF(q'): does SF's
output lose less than its input, i.e. is SF robust, or is the matcher's noisy matrix just as good?
Configurations go to tuning/configs/explore_r1_plain, results to tuning/results/explore_r1_plain.

    ../harness/.venv-bdi/Scripts/python.exe explore_robust.py --prepare
    ../harness/.venv-bdi/Scripts/python.exe explore_robust.py --run --workers 10
    ../harness/.venv-bdi/Scripts/python.exe explore_robust.py --score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
TUNING = HERE / "tuning"
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(TUNING))
sys.path.insert(0, str(HERE))

from smx.adapters import read_matrix  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, latest_run, one  # noqa: E402
from explore_penalty import pa  # noqa: E402

SUFFIX = "explore_r1_plain"
CFG_ROOT = TUNING / "configs" / SUFFIX
RES_ROOT = TUNING / "results" / SUFFIX
SEED_ROOT = HERE / "seeds_explore" / "plain" / "r1"
TEMPLATE = TUNING / "configs" / "screen_plain" / "levenshtein__fpC__INV_PROD__NAME__fdfull"
SCREEN_CORPUS = TUNING / "corpus" / "plain" / "screen"
MATCHERS = ("prefix_suffix", "levenshtein", "pmf")
LEVELS = (0, 25, 50, 100, 200, 400)
DRAWS = 3
CONFIGS = {"ref": {"propCoeffPolicy": "INV_AVG", "fdFilterThreshold": ""}, "std": {}}
METRICS = ("F1_Hr", "R_exp", "MRR_exp")


def variants() -> list[str]:
    out = ["p0"]
    for p in LEVELS[1:]:
        out += [f"p{p}_d{d}" for d in range(DRAWS)]
    return out + ["flat"]


def perturb(q: np.ndarray, variant: str, key: str) -> np.ndarray:
    if variant == "p0":
        return q.copy()
    if variant == "flat":
        pos = q > 0
        out = np.zeros_like(q)
        if pos.any():
            out[pos] = q[pos].mean()
        return out
    p = int(variant[1:].split("_")[0]) / 100.0
    rng = np.random.default_rng(int(hashlib.sha256(f"R1/{key}/{variant}".encode()).hexdigest()[:16], 16))
    return np.clip(q * (1.0 + rng.uniform(-p, p, size=q.shape)), 0.0, None)


def write_seeds() -> list[str]:
    cases = sorted((d.name, s.name) for d in SCREEN_CORPUS.iterdir() if d.is_dir() for s in d.iterdir() if s.is_dir())
    for m in MATCHERS:
        for ds, sc in cases:
            src = TUNING / "seeds" / "plain" / m / ds / sc / "seed.csv"
            rows = list(csv.reader(open(src, encoding="utf-8")))
            header, labels = rows[0], [r[0] for r in rows[1:]]
            q = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
            for v in variants():
                out = SEED_ROOT / f"{m}__{v}" / ds / sc
                out.mkdir(parents=True, exist_ok=True)
                qv = perturb(q, v, f"{m}/{ds}/{sc}")
                with open(out / "seed.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(header)
                    for lab, row in zip(labels, qv):
                        w.writerow([lab, *[f"{x:.7g}" for x in row]])
    return sorted({ds for ds, _ in cases})


def write_config(label: str, datasets: list[str], seed_dir: Path, params: dict, meta: dict):
    out = CFG_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    gen = (TEMPLATE / "general.yaml").read_text(encoding="utf-8")
    old_res = (TUNING / "results" / "screen_plain" / TEMPLATE.name).as_posix()
    gen = gen.replace(old_res, (RES_ROOT / label).as_posix())
    assert (RES_ROOT / label).as_posix() in gen and SCREEN_CORPUS.as_posix() in gen, "template paths not found"
    (out / "general.yaml").write_text(gen, encoding="utf-8")
    (out / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets), encoding="utf-8")
    shutil.copy(TEMPLATE / "first_line_matchers.yaml", out / "first_line_matchers.yaml")
    sl = (TEMPLATE / "second_line_matchers.yaml").read_text(encoding="utf-8")
    base = {"graphEncoding": "QUICK", "fdFilterThreshold": "0.0", "propCoeffPolicy": "INV_PROD", "fixpoint": "C",
            "constraintWeighting": "NONE", "initialMatrixDir": (TUNING / "seeds" / "plain" / "levenshtein").as_posix()}
    for k, v in {**base, **params, "initialMatrixDir": seed_dir.as_posix()}.items():
        old = f'  {k}: ["{base[k]}"]'
        assert old in sl, f"{k} not in template"
        sl = sl.replace(old, f'  {k}: ["{v}"]')
    (out / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
    (out / "jvm.properties").write_text("smx.sf.seedAttachment=NAME\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({**meta, "label": label, "seed_dir": seed_dir.as_posix(),
                                               "corpus": SCREEN_CORPUS.as_posix(), "exploratory": True,
                                               "question": "R1"}, indent=1), encoding="utf-8")


def prepare():
    for d in (CFG_ROOT, SEED_ROOT):
        if d.exists():
            shutil.rmtree(d)
    datasets = write_seeds()
    n = 0
    for m in MATCHERS:
        for v in variants():
            for cname, params in CONFIGS.items():
                write_config(f"r1_{cname}__{m}__{v}", datasets, SEED_ROOT / f"{m}__{v}", params,
                             {"seed": m, "variant": v, "config": cname})
                n += 1
    print(f"wrote {n} configurations under {CFG_ROOT}; perturbed initial matrices under {SEED_ROOT}")


def run(workers: int, xmx: str):
    from run_tuning import run_one
    from run_seeds import classpath
    configs = sorted(p for p in CFG_ROOT.iterdir() if p.is_dir())
    # the standard configuration is slower; start those first
    configs.sort(key=lambda c: (not c.name.startswith("r1_std"), c.name))
    todo = [c for c in configs if not (RES_ROOT / c.name / "DONE").is_file()]
    print(f"{len(configs)} configurations, {len(todo)} to run, {workers} workers", flush=True)
    deps = classpath()
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(run_one, c, RES_ROOT / c.name, deps, xmx): c for c in todo}
        for f in as_completed(futures):
            label, code, secs = f.result()
            print(f"{label:45s} {'ok' if code == 0 else f'FAILED {code}'} ({secs:.0f}s)", flush=True)


def score_one(label: str) -> list[dict]:
    meta = json.loads((CFG_ROOT / label / "meta.json").read_text(encoding="utf-8"))
    run = latest_run(RES_ROOT / label)
    rows = []
    for scen in sorted(run.glob("*/*")):
        ds, sc = scen.parent.name, scen.name
        gt = read_matrix(one(SCREEN_CORPUS / ds / sc / "ground_truth"))[0] > 0.5
        q = read_matrix(Path(meta["seed_dir"]) / ds / sc / "seed.csv")[0]
        f_s = one(scen / "_outputs" / "MatchingStepLine2")
        if f_s is None:
            continue
        S = np.clip(read_matrix(f_s)[0], 0.0, None)
        perms = permutations(gt.shape, f"{ds}/{sc}")
        row = {"seed": meta["seed"], "variant": meta["variant"], "config": meta["config"], "dataset": ds, "scenario": sc}
        for name, M in (("q", q), ("sf", S), ("sf_pa", S * pa(S))):
            mm = all_metrics(M, gt, perms)
            for k in METRICS:
                row[f"{name}_{k}"] = mm[k]
        rows.append(row)
    return rows


def score(workers: int):
    labels = sorted(p.name for p in RES_ROOT.iterdir() if (p / "DONE").is_file())
    with Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(score_one, labels) for r in part]
    rows.sort(key=lambda r: (r["seed"], r["config"], r["variant"], r["dataset"], r["scenario"]))
    out = HERE / "results" / "plain" / "r1_robust_scores.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # mean over scenarios per run, then over the draws of a level
    acc = defaultdict(list)
    for r in rows:
        level = r["variant"].split("_")[0]
        acc[(r["seed"], r["config"], level, r["variant"])].append(r)
    per_run = {k: {f"{n}_{m}": mean(x[f"{n}_{m}"] for x in v) for n in ("q", "sf", "sf_pa") for m in METRICS}
               for k, v in acc.items()}
    lvl = defaultdict(list)
    for (s, c, level, _), v in per_run.items():
        lvl[(s, c, level)].append(v)
    agg = {k: {m: mean(x[m] for x in v) for m in v[0]} for k, v in lvl.items()}
    jt = lambda d, n: (d[f"{n}_R_exp"] + d[f"{n}_F1_Hr"]) / 2  # noqa: E731
    levels = [f"p{p}" for p in LEVELS] + ["flat"]
    n_scen = len({(r["dataset"], r["scenario"]) for r in rows})
    lines = ["# R1: Melnik's robustness test on our corpus (exploratory, plain, screen scenarios)", "",
             f"{len(per_run)} runs, {n_scen} scenarios. q' is the perturbed initial matrix, SF(q') SF's output; "
             "J_tf = (R_exp + F1_Hr)/2, mean over scenarios, then over the 3 draws of a level. "
             "'loss' is the drop from p = 0 as a share of the p = 0 value (Melnik report accuracy the same way).", ""]
    for cname in CONFIGS:
        lines += [f"## Configuration `{cname}`", "",
                  "| Initial matcher | Level | J_tf q' | J_tf SF(q') | SF − q' | loss q' | loss SF | J_tf SF×pa | R_exp q' | R_exp SF | F1_Hr q' | F1_Hr SF |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in MATCHERS:
            base = agg.get((s, cname, "p0"))
            if base is None:
                continue
            for level in levels:
                d = agg.get((s, cname, level))
                if d is None:
                    continue
                lq = 1 - jt(d, "q") / jt(base, "q")
                ls = 1 - jt(d, "sf") / jt(base, "sf")
                lines.append(f"| {s} | {level} | {jt(d, 'q'):.3f} | {jt(d, 'sf'):.3f} | {jt(d, 'sf') - jt(d, 'q'):+.3f} | "
                             f"{lq:+.0%} | {ls:+.0%} | {jt(d, 'sf_pa'):.3f} | {d['q_R_exp']:.3f} | {d['sf_R_exp']:.3f} | "
                             f"{d['q_F1_Hr']:.3f} | {d['sf_F1_Hr']:.3f} |")
        lines.append("")
    (HERE / "results" / "plain" / "r1_robust_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


def margins():
    """Why multiplicative noise hurts some matchers more: per true pair, the relative margin
    (q_true - best false q in its row) / q_true. Noise of +-p flips margins below about p."""
    cases = sorted((d.name, s.name) for d in SCREEN_CORPUS.iterdir() if d.is_dir() for s in d.iterdir() if s.is_dir())
    out = []
    for m in MATCHERS:
        rel = []
        for ds, sc in cases:
            q = read_matrix(TUNING / "seeds" / "plain" / m / ds / sc / "seed.csv")[0]
            gt = read_matrix(one(SCREEN_CORPUS / ds / sc / "ground_truth"))[0] > 0.5
            for i in np.flatnonzero(gt.any(axis=1)):
                j = np.flatnonzero(gt[i])[0]
                others = np.delete(q[i], j)
                if q[i, j] > 0 and others.size:
                    rel.append((q[i, j] - others.max()) / q[i, j])
        rel = np.array(rel)
        out.append({"seed": m, "n_true_pairs": len(rel), "median_rel_margin": float(np.median(rel)),
                    "share_margin_0_to_10pct": float(np.mean((rel > 0) & (rel < 0.1))),
                    "share_margin_nonpositive": float(np.mean(rel <= 0))})
    f = HERE / "results" / "plain" / "r1_margins.csv"
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--margins", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--xmx", default="2g")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.run:
        run(a.workers, a.xmx)
    if a.score:
        score(min(a.workers, 12))
    if a.margins:
        margins()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
