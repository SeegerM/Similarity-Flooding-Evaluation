"""Single-machine smoke test (latex/sf-evaluation/REPRODUCE.md §8.3).

On ONE machine, with the local merged Schematch build (code/schematch/target/classes) and the local corpus
(harness/generated/benchmark-full), for the Magellan family (Magellan-Unionable, 7 scenarios):

  stage 1  the initial matchers' seed-study configurations (configs/plain/<m>: first line + the seed study's standard SF
           line), compared with
             - reference/seeds/plain/<m>/.../seed.csv          (the FILE seeds every layer-1 run read)
             - results/plain/<m>/<run>/.../MatchingStepLine2  (the seed study's SF output)
  stage 2  SF as published (melnik) and with FD edges (fd): the registered layer-1 configurations
           (tuning/configs/layer1_plain/<m>__<arm>), compared with
             - tuning/results/layer1_plain/<m>__<arm>/merged/... (the cluster's SF outputs), cell for cell
             - tuning/results/layer1_plain_scores/scores.csv   (the scored rows), metric for metric
Each configuration is copied to smoke/configs/; only the dataset list (Magellan-Unionable), resultsDir (smoke/results/)
and, for stage 2, the corpus path (the hosts' path -> the local corpus) change. Acceptance: every comparison exact
(max |diff| 0, metrics equal). Writes smoke/SMOKE_REPORT.md.

    ../harness/.venv-bdi/Scripts/python.exe smoke/run_smoke.py [--matchers levenshtein,jaro_winkler,pmf,ks] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "reference"))
sys.path.insert(0, str(STUDY.parent / "harness"))
from run_seeds import SCHEMATCH, classpath  # noqa: E402
from score_seeds import CORPUS, latest_run, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
import score_layer1  # noqa: E402

DATASET = "Magellan-Unionable"
ARMS = ("melnik", "fd")
LOCAL_CORPUS = CORPUS["plain"].as_posix()
REMOTE_CORPUS = "$SMX_REMOTE_ROOT/smx/corpus/plain"
METRICS = ("F1_H", "F1_Hr", "F1_M", "R_pess", "R_exp", "MRR_pess", "MRR_exp")


def prepare(src: Path, label: str, remote: bool) -> Path:
    dst = HERE / "configs" / label
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(HERE / "results" / label).as_posix()}"', g)
    if remote:
        assert g2.count(f'defaultDatasetBasePath: "{REMOTE_CORPUS}"') == 1, label
        g2 = g2.replace(f'defaultDatasetBasePath: "{REMOTE_CORPUS}"', f'defaultDatasetBasePath: "{LOCAL_CORPUS}"')
    else:
        assert g2.count(f'defaultDatasetBasePath: "{LOCAL_CORPUS}"') == 1, label
    (dst / "general.yaml").write_text(g2, encoding="utf-8")
    ds = re.findall(r'(?m)^path: "(.*)"$', (dst / "datasets.yaml").read_text(encoding="utf-8"))
    assert DATASET in ds, (label, ds)
    (dst / "datasets.yaml").write_text(f'---\nname: "{DATASET}"\npath: "{DATASET}"\n', encoding="utf-8")
    return dst


def run(config: Path, deps: str) -> tuple[str, int, float]:
    res = HERE / "results" / config.name
    if res.exists():
        shutil.rmtree(res)
    res.mkdir(parents=True)
    props = []
    if (config / "jvm.properties").is_file():
        props = ["-D" + l.strip() for l in (config / "jvm.properties").read_text(encoding="utf-8").splitlines() if "=" in l]
    # the cluster's flags (servers/remote_runner.py): -Xmx and the serial collector; none of them changes the output
    # SMX_EXTRA_CP (optional): a class folder put before target/classes, to test rebuilt classes without installing them
    extra = [os.environ["SMX_EXTRA_CP"]] if os.environ.get("SMX_EXTRA_CP") else []
    cmd = ["java", "-Xmx6g", "-XX:+UseSerialGC", *props, *(f"-D{p}" for p in os.environ.get("SMX_EXTRA_PROPS", "").split() if p),
           "-cp", os.pathsep.join([str(config), *extra, str(SCHEMATCH / "target" / "classes"), deps]),
           "de.uni_marburg.schematch.Main"]
    t0 = time.perf_counter()
    with (res / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=SCHEMATCH, stdout=log, stderr=subprocess.STDOUT).returncode
    return config.name, code, time.perf_counter() - t0


def scenarios() -> list[str]:
    return sorted(p.name for p in (CORPUS["plain"] / DATASET).iterdir() if p.is_dir())


def maxdiff(fa: Path | None, fb: Path | None) -> float | None:
    if fa is None or fb is None:
        return None
    A, B = read_matrix(fa)[0], read_matrix(fb)[0]
    if A.shape != B.shape:
        return float("inf")
    d = np.abs(A - B)
    both = np.isnan(A) & np.isnan(B)
    d[both] = 0.0
    return float(np.nanmax(d)) if not np.isnan(d).any() else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchers", default="levenshtein,jaro_winkler,pmf,ks")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ms = a.matchers.split(",")
    t_start = time.perf_counter()
    deps = classpath()
    jobs = [prepare(STUDY / "configs" / "plain" / m, f"seed__{m}", remote=False) for m in ms]
    jobs += [prepare(STUDY / "tuning" / "configs" / "layer1_plain" / f"{m}__{arm}", f"{m}__{arm}", remote=True)
             for m in ms for arm in ARMS]
    with ThreadPoolExecutor(a.workers) as pool:
        runs = list(pool.map(lambda c: run(c, deps), jobs))
    sc = scenarios()
    L = ["# Single-machine smoke test (REPRODUCE.md §8.3)", "",
         f"Machine: this workstation, one JVM per configuration ({a.workers} in parallel), local merged build "
         f"`code/schematch/target/classes`, local corpus `{LOCAL_CORPUS}`. Family: {DATASET} ({len(sc)} scenarios). "
         f"Initial matchers: {', '.join(ms)}. Wall time: {time.perf_counter() - t_start:.0f} s.", "",
         "## Runs", "", "| configuration | exit | seconds |", "|---|---|---|"]
    L += [f"| {n} | {c} | {s:.0f} |" for n, c, s in runs]
    ok_all = all(c == 0 for _, c, _ in runs)
    # stage 1: first line vs the FILE seeds; second line vs the seed study's SF output
    L += ["", "## Stage 1: initial matrices and the seed study's SF (max |diff| over the 7 scenarios)", "",
          "| matcher | first line vs FILE seed | compared | standard SF vs seed study | compared |", "|---|---|---|---|---|"]
    for m in ms:
        run_dir = latest_run(HERE / "results" / f"seed__{m}")
        ref_sf = latest_run(STUDY / "results" / "plain" / m)
        w1 = w2 = 0.0
        n1 = n2 = 0
        for s in sc:
            d1 = maxdiff(one(run_dir / DATASET / s / "_outputs" / "MatchingStepLine1") if run_dir else None,
                         STUDY / "reference" / "seeds" / "plain" / m / DATASET / s / "seed.csv")
            d2 = maxdiff(one(run_dir / DATASET / s / "_outputs" / "MatchingStepLine2") if run_dir else None,
                         one(ref_sf / DATASET / s / "_outputs" / "MatchingStepLine2") if ref_sf else None)
            if d1 is not None:
                w1, n1 = max(w1, d1), n1 + 1
            if d2 is not None:
                w2, n2 = max(w2, d2), n2 + 1
        ok_all &= (w1 == 0.0 and n1 == len(sc) and w2 == 0.0 and n2 == len(sc))
        L.append(f"| {m} | {w1:g} | {n1}/{len(sc)} | {w2:g} | {n2}/{len(sc)} |")
    # stage 2: SF outputs vs the cluster, and the scored rows
    ref_rows = {}
    with (STUDY / "tuning" / "results" / "layer1_plain_scores" / "scores.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == DATASET and r["seed"] in ms and r["arm"] in ARMS:
                ref_rows[(r["seed"], r["arm"], r["scenario"])] = r
    L += ["", "## Stage 2: layer-1 SF (melnik, fd) against the cluster's runs and the scored rows", "",
          "| configuration | SF output max \\|diff\\| | compared | scored rows equal | metric cells checked |",
          "|---|---|---|---|---|"]
    cases = [(DATASET, s, "Magellan") for s in sc]
    for m in ms:
        for arm in ARMS:
            label = f"{m}__{arm}"
            run_dir = latest_run(HERE / "results" / label)
            ref = STUDY / "tuning" / "results" / "layer1_plain" / label / "merged"
            w, n = 0.0, 0
            for s in sc:
                d = maxdiff(one(run_dir / DATASET / s / "_outputs" / "MatchingStepLine2") if run_dir else None,
                            one(ref / DATASET / s / "_outputs" / "MatchingStepLine2"))
                if d is not None:
                    w, n = max(w, d), n + 1
            rows = score_layer1.score_seed(("plain", m, [str(HERE / "results" / label)], cases)) if run_dir else []
            eq = cells = 0
            for r in rows:
                rr = ref_rows.get((m, arm, r["scenario"]))
                if rr is None:
                    continue
                vals = [(r[f"{p}_{k}"], float(rr[f"{p}_{k}"])) for p in ("seed", "sf") for k in METRICS]
                cells += len(vals)
                eq += all(float(x) == y for x, y in vals)
            ok = w == 0.0 and n == len(sc) and eq == len(sc)
            ok_all &= ok
            L.append(f"| {label} | {w:g} | {n}/{len(sc)} | {eq}/{len(sc)} | {cells} |")
    L += ["", f"**Overall: {'PASSED: every comparison exact' if ok_all else 'NOT PASSED'}.**", ""]
    (HERE / "SMOKE_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
