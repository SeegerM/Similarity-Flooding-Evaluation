"""R8 (restricted propagation scope): local runs and checks on the screen scenarios (plain).

Configurations are the R7 smoke test's `melnik` and `fd` configurations (layer3b/configs/r7_screen) plus
the JVM properties smx.sf.pairScope / smx.sf.pairScopeParam. The Schematch classes can come from a
directory placed before target/classes (--classes), so a new build can be tested before installing it.

    python layer3b/r8_smoke.py check --classes <dir>          # default unchanged + pruning correct
    python layer3b/r8_smoke.py write --seeds ... --scopes ...  # exploratory grid
    python layer3b/r8_smoke.py run --workers 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY / "tuning"))
sys.path.insert(0, str(STUDY))
from smx.adapters import read_matrix  # noqa: E402
from run_tuning import SCHEMATCH, classpath  # noqa: E402
from score_seeds import one  # noqa: E402

R7_CFG = HERE / "configs" / "r7_screen"
R7_RES = HERE / "results" / "r7_screen"
SUFFIX = "r8_screen"
SEEDS = STUDY / "reference" / "seeds" / "plain"


def run(config: Path, results: Path, deps: str, xmx: str, classes: str = "") -> tuple[str, int, float]:
    results.mkdir(parents=True, exist_ok=True)
    props = ["-D" + l.strip() for l in (config / "jvm.properties").read_text(encoding="utf-8").splitlines() if "=" in l]
    cp = [str(config)] + ([classes] if classes else []) + [str(SCHEMATCH / "target" / "classes"), deps]
    t0 = time.perf_counter()
    with (results / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(["java", f"-Xmx{xmx}", *props, "-cp", os.pathsep.join(cp), "de.uni_marburg.schematch.Main"],
                              cwd=SCHEMATCH, stdout=log, stderr=subprocess.STDOUT).returncode
    el = time.perf_counter() - t0
    if code == 0:
        (results / "DONE").write_text(f"{el:.1f}s\n", encoding="utf-8")
    return config.name, code, el


def make(base_label: str, label: str, root: Path, results_root: Path, props: dict[str, str]) -> Path:
    src, dst = R7_CFG / base_label, root / label
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = f'resultsDir: "{(R7_RES / base_label).as_posix()}"'
    assert g.count(old) == 1, old
    (dst / "general.yaml").write_text(g.replace(old, f'resultsDir: "{(results_root / label).as_posix()}"'),
                                      encoding="utf-8")
    jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
    (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
    return dst


def latest(d: Path) -> Path:
    return sorted(p for p in d.iterdir() if p.is_dir())[-1]


def outputs(run_root: Path):
    for ds in sorted(p for p in run_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for sc in sorted(p for p in ds.iterdir() if p.is_dir()):
            f = one(sc / "_outputs" / "MatchingStepLine2")
            if f is not None:
                yield ds.name, sc.name, read_matrix(f)[0]


def keep_rule(q: np.ndarray, scope: str, p: float) -> np.ndarray:
    """Python mirror of SimilarityFlooding.pairScopeKeep (float32 like the Java code)."""
    q = q.astype(np.float32)
    if scope == "QUANTILE":
        a = np.sort(q.ravel())
        return q >= a[min(a.size - 1, int(np.floor(p * a.size)))]
    k = int(p)
    rc = np.array([np.sort(r)[-k] if r.size > k else -np.inf for r in q])
    cc = np.array([np.sort(c)[-k] if c.size > k else -np.inf for c in q.T])
    return (q >= rc[:, None]) | (q >= cc[None, :])


def check(classes: str, reuse: bool = False) -> int:
    root = HERE / "configs" / "r8_check"
    res = HERE / "results" / "r8_check"
    if res.exists() and not reuse:
        shutil.rmtree(res)
    cfgs = [make("levenshtein__fd", "levenshtein__fd__none", root, res, {}),
            make("levenshtein__melnik", "levenshtein__melnik__none", root, res, {}),
            make("levenshtein__melnik", "levenshtein__melnik__q050", root, res,
                 {"smx.sf.pairScope": "QUANTILE", "smx.sf.pairScopeParam": "0.5"}),
            make("levenshtein__fd", "levenshtein__fd__top1", root, res,
                 {"smx.sf.pairScope": "TOPK", "smx.sf.pairScopeParam": "1"})]
    deps = classpath()
    cfgs = [c for c in cfgs if not (reuse and (res / c.name / "DONE").is_file())]
    with ThreadPoolExecutor(4) as pool:
        for label, code, el in pool.map(lambda c: run(c, res / c.name, deps, "6g", classes), cfgs):
            print(f"{label}: exit {code} ({el:.0f}s)")
            if code:
                return 1
    bad = 0
    # 1. defaults unchanged
    for label, ref in (("levenshtein__fd__none", "levenshtein__fd"), ("levenshtein__melnik__none", "levenshtein__melnik")):
        refs = {(d, s): M for d, s, M in outputs(latest(R7_RES / ref))}
        worst = max(float(np.abs(M - refs[(d, s)]).max()) for d, s, M in outputs(latest(res / label)))
        print(f"default unchanged, {label} vs R7 smoke {ref}: max |diff| {worst}")
        bad += worst != 0.0
    # 2. pruning: zeros, kept counts, Proposition 1 among survivors (melnik)
    const = {(d, s): M for d, s, M in outputs(latest(R7_RES / "constant__melnik"))}
    for label, scope, p in (("levenshtein__melnik__q050", "QUANTILE", 0.5), ("levenshtein__fd__top1", "TOPK", 1)):
        log = (res / label / "run.log").read_text(encoding="utf-8", errors="replace")
        java_kept = {(x[1], x[2]): (int(x[5]), int(x[6])) for x in
                     (l.split("\t") for l in log.splitlines() if l.startswith("PAIRSCOPE\t"))}
        n = zero_bad = count_bad = 0
        inv = pairs = 0
        for d, s, S in outputs(latest(res / label)):
            q = read_matrix(SEEDS / "levenshtein" / d / s / "seed.csv")[0]
            k = keep_rule(q, scope, p)
            n += 1
            zero_bad += int((S[~k] != 0).any()) + int((S[k] <= 0).any())
            count_bad += java_kept.get((d, s)) != (int(k.sum()), k.size)
            if scope == "QUANTILE":  # survivors only
                D = const[(d, s)]
                _, ci = np.unique(D.round(7), axis=0, return_inverse=True)
                _, cj = np.unique(D.T.round(7), axis=0, return_inverse=True)
                block = (ci.ravel()[:, None] * (cj.max() + 1) + cj.ravel()[None, :]).ravel()
                kk, ss, qq = k.ravel(), S.ravel(), q.ravel()
                for b in np.unique(block[kk]):
                    idx = np.where((block == b) & kk)[0]
                    dq = qq[idx][:, None] - qq[idx][None, :]
                    dS = ss[idx][:, None] - ss[idx][None, :]
                    inv += int(((dq > 1e-9) & (dS < -1e-6)).sum())
                    pairs += int((dq > 1e-9).sum())
        print(f"{label}: {n} scenarios; pruned≠0 or kept≤0 in {zero_bad}; kept count ≠ Python rule in {count_bad}"
              + (f"; within-block inversions among survivors {inv} of {pairs}" if scope == "QUANTILE" else ""))
        bad += zero_bad + count_bad + inv
    print("OK" if not bad else f"{bad} problems")
    return 1 if bad else 0


def write(seeds: list[str], scopes: list[str]) -> None:
    root = HERE / "configs" / SUFFIX
    n = 0
    for seed in seeds:
        for arm in ("melnik", "fd"):
            for sc in scopes:
                kind, p = sc.split(":")
                label = f"{seed}__{arm}__{kind.lower()}{p}"
                make(f"{seed}__{arm}", label, root, HERE / "results" / SUFFIX,
                     {"smx.sf.pairScope": kind, "smx.sf.pairScopeParam": p})
                n += 1
    print(f"{n} configurations -> {root}")


def run_all(workers: int, xmx: str, classes: str) -> int:
    root = HERE / "configs" / SUFFIX
    todo = [c for c in sorted(root.iterdir()) if not (HERE / "results" / SUFFIX / c.name / "DONE").is_file()]
    print(f"{len(todo)} to run, {workers} workers", flush=True)
    deps, failed, t0 = classpath(), [], time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(run, c, HERE / "results" / SUFFIX / c.name, deps, xmx, classes) for c in todo]
        for k, f in enumerate(as_completed(futs), 1):
            label, code, el = f.result()
            failed += [label] if code else []
            print(f"[{k}/{len(todo)}] {label} {'ok' if code == 0 else 'FAILED'} ({el:.0f}s; {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    print(f"done; {len(failed)} failed {failed}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "write", "run"))
    ap.add_argument("--classes", default="")
    ap.add_argument("--seeds", default="levenshtein,prefix_suffix,pmf,role_gcov")
    ap.add_argument("--scopes", default="QUANTILE:0.5,QUANTILE:0.75,QUANTILE:0.9,TOPK:1,TOPK:3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--xmx", default="6g")
    ap.add_argument("--reuse", action="store_true", help="check: keep finished runs")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "check":
        return check(a.classes, a.reuse)
    if a.cmd == "write":
        write(a.seeds.split(","), a.scopes.split(","))
        return 0
    return run_all(a.workers, a.xmx, a.classes)


if __name__ == "__main__":
    sys.exit(main())
