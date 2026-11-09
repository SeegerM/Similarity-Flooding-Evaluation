"""PRISMA initial matrices on the fixed-HyFD profiles: the old 382 cases and the 168 new ones (Valentine 549 rerun).

The configuration is the one behind the existing PRISMA seeds (configs/audited-prisma-{plain,md5}-full ->
reports/prisma_audited_{plain,md5}_full -> make_file_seeds.py --source prisma -> reference/seeds/{plain,md5}/prisma):
PRISMAMatcher, postprocessing false, thresholdMatches true, GammaStrucAttr 0.35, Distribution features,
gDepFiltering gdep_threshold_0.3; the same Java entry point and Python service as prisma_host.py. Only the dataset
tree changes:

    run            dataset tree                                   results                                   seeds
    verify_old     generated/benchmark-full (old FDs), 3 datasets reports/prisma_fdfix_20261031/verify_old  (compare)
    fdfix_plain    generated_hyfdfix/valentine_plain              reports/prisma_fdfix_20261031/fdfix_plain reference/seeds/plain_fdfix/prisma
    fdfix_md5      generated_hyfdfix/valentine_md5                reports/prisma_fdfix_20261031/fdfix_md5   reference/seeds/md5_fdfix/prisma
    pnew           generated/valentine-new168                     reports/prisma_fdfix_20261031/pnew        reference/seeds/pnew/prisma
    mnew           generated/valentine-new168-md5                 reports/prisma_fdfix_20261031/mnew        reference/seeds/mnew/prisma

The stage-1 mirrors hold the same tables as benchmark-full(-md5) (sha256-checked for all 1,528 sides) with the
fixed-HyFD metadata. Runs are sequential: the service port (5004) is fixed and the Java side writes its graphs to
code/prisma/target/graphs/<dataset>/<scenario>/. If anything already listens on 5004 the run stops instead of killing it.
Export follows make_file_seeds.py: the postprocessing=false MatchingStepLine1 file is copied as seed.csv; an all-zero
matrix is skipped.

    python prisma_fdfix.py run verify_old|fdfix_plain|fdfix_md5|pnew|mnew
    python prisma_fdfix.py compare verify_old            # byte-compare with the old run
    python prisma_fdfix.py export fdfix_plain|...
    python prisma_fdfix.py change                        # old vs new FDs on the 382
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from smx.adapters import read_matrix  # noqa: E402

PRISMA = ROOT / "code" / "prisma"
SERVER_DIR = PRISMA / "src" / "main" / "resources" / "prisma"
JAVA17 = Path("C:/Program Files/Java/jdk-17/bin/java.exe")
OUT = HERE / "reports" / "prisma_fdfix_20261031"
CONFIGS = HERE / "configs" / "prisma-fdfix"
SEEDS = ROOT / "sf-matcher-study" / "reference" / "seeds"
TEMPLATE = {"plain": HERE / "configs" / "audited-prisma-plain-full", "md5": HERE / "configs" / "audited-prisma-md5-full"}
FRESH_T5 = {"plain": HERE / "reports" / "prisma_table5_fresh" / "plain",
            "md5": HERE / "reports" / "prisma_table5_fresh" / "md5"}
OLD_RUN = {"plain": HERE / "reports" / "prisma_audited_plain_full", "md5": HERE / "reports" / "prisma_audited_md5_full"}
OLD13 = ["ChEMBL-Semantically-Joinable", "ChEMBL-Unionable", "ChEMBL-View-Unionable", "Magellan-Unionable",
         "OpenData-Joinable", "OpenData-Semantically-Joinable", "OpenData-Unionable", "TPC-DI-Joinable",
         "TPC-DI-Semantically-Joinable", "TPC-DI-Unionable", "TPC-DI-View-Unionable", "Wikidata-Joinable",
         "Wikidata-Semantically-Joinable"]
NEW7 = ["ChEMBL-Joinable", "ChEMBL-View-Unionable", "OpenData-Semantically-Joinable", "OpenData-Unionable",
        "OpenData-View-Unionable", "TPC-DI-Semantically-Joinable", "TPC-DI-Unionable"]
RUNS = {
    "verify_old": ("plain", HERE / "generated" / "benchmark-full",
                   ["ChEMBL-View-Unionable", "Wikidata-Joinable", "Wikidata-Semantically-Joinable",
                    "TPC-DI-Joinable", "Magellan-Unionable"], None),
    "verify_old_md5": ("md5", HERE / "generated" / "benchmark-full-md5",
                       ["ChEMBL-View-Unionable", "Wikidata-Joinable", "Wikidata-Semantically-Joinable",
                        "TPC-DI-Joinable", "Magellan-Unionable"], None),
    "fdfix_plain": ("plain", HERE / "generated_hyfdfix" / "valentine_plain", OLD13, "plain_fdfix"),
    "fdfix_md5": ("md5", HERE / "generated_hyfdfix" / "valentine_md5", OLD13, "md5_fdfix"),
    "pnew": ("plain", HERE / "generated" / "valentine-new168", NEW7, "pnew"),
    "mnew": ("md5", HERE / "generated" / "valentine-new168-md5", NEW7, "mnew"),
}


def port_open(port: int = 5004) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def tag(name: str) -> str:
    return f"{name}-20261031"


def stage(name: str) -> Path:
    regime, base, datasets, _ = RUNS[name]
    cfg = CONFIGS / name
    cfg.mkdir(parents=True, exist_ok=True)
    general = (TEMPLATE[regime] / "general.yaml").read_text(encoding="utf-8")
    old_base = [l for l in general.splitlines() if l.startswith("defaultDatasetBasePath:")][0]
    old_res = [l for l in general.splitlines() if l.startswith("resultsDir:")][0]
    general = general.replace(old_base, f'defaultDatasetBasePath: "{base.as_posix()}"')
    general = general.replace(old_res, f'resultsDir: "{(OUT / name).as_posix()}"')
    (cfg / "general.yaml").write_text(general, encoding="utf-8")
    shutil.copyfile(TEMPLATE[regime] / "first_line_matchers.yaml", cfg / "first_line_matchers.yaml")
    # unique dataset names per run: PRISMA caches features under target/features/<dataset name>/ (FINDINGS 120)
    (cfg / "datasets.yaml").write_text(
        "".join(f'---\nname: "{d}__{tag(name)}"\npath: "{d}"\n' for d in datasets), encoding="utf-8")
    return cfg


def command_run(args) -> int:
    cfg = stage(args.name)
    if port_open():
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        print("port 5004 already in use; not touching it:\n" +
              "\n".join(l for l in out.splitlines() if ":5004 " in l))
        return 2
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{args.name}_server.log", "w", encoding="utf-8") as server_log:
        server = subprocess.Popen(["py", "-3.11", "prisma_server.py", "-p", "5004"], cwd=SERVER_DIR,
                                  stdout=server_log, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if port_open():
                    break
                if server.poll() is not None:
                    raise RuntimeError("PRISMA service exited")
                time.sleep(0.5)
            else:
                raise RuntimeError("PRISMA service did not start")
            cp = (PRISMA / "cp.txt").read_text(encoding="utf-8").strip()
            cmd = [str(JAVA17), "-Xmx12g", "-cp", f"{cfg};target/classes;{cp}", "de.uni_marburg.schematch.Main"]
            t0 = time.time()
            with open(log_dir / f"{args.name}_java.log", "w", encoding="utf-8") as fh:
                rc = subprocess.run(cmd, cwd=PRISMA, stdout=fh, stderr=subprocess.STDOUT).returncode
            print(f"{args.name}: exit {rc}, {time.time() - t0:.0f}s", flush=True)
            return rc
        finally:
            # only the process tree this script started (the py launcher and its python child)
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
            try:
                server.wait(20)
            except subprocess.TimeoutExpired:
                server.kill()
            for _ in range(50):
                if not port_open():
                    break
                time.sleep(0.2)
            else:
                print("warning: port 5004 still open after stopping our service", flush=True)


def matrix_file(root: Path, dataset: str, scenario: str) -> Path | None:
    runs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for run in runs:
        d = run / dataset / scenario / "_outputs" / "MatchingStepLine1"
        if d.is_dir():
            files = [f for f in d.glob("*.csv") if "postprocessing=false" in f.name]
            return files[0] if len(files) == 1 else None
    return None


def scenarios(name: str):
    _, base, datasets, _ = RUNS[name]
    for d in datasets:
        for sc in sorted(p for p in (base / d).iterdir() if p.is_dir()):
            yield d, sc.name


def command_compare(args) -> int:
    regime = RUNS[args.name][0]
    n = same = same5 = 0
    worst = worst5 = 0.0
    for d, s in scenarios(args.name):
        a = matrix_file(OUT / args.name, f"{d}__{tag(args.name)}", s)
        b = matrix_file(OLD_RUN[regime], d, s)
        c = matrix_file(FRESH_T5[regime], f"{d}__t5-{regime}", s)
        if a is None or b is None or c is None:
            print(f"{d}/{s}: missing new={a is not None} committed={b is not None} t5fresh={c is not None}")
            continue
        n += 1
        same += a.read_bytes() == b.read_bytes()
        same5 += a.read_bytes() == c.read_bytes()
        A = read_matrix(a)[0]
        worst = max(worst, float(np.abs(A - read_matrix(b)[0]).max()))
        worst5 = max(worst5, float(np.abs(A - read_matrix(c)[0]).max()))
    print(f"{args.name}: {same}/{n} byte-identical to the committed September run (max|diff| {worst:.3g}); "
          f"{same5}/{n} byte-identical to the 2026-09-30 fresh rerun prisma_table5_fresh (max|diff| {worst5:.3g})")
    return 0


def command_export(args) -> int:
    target = RUNS[args.name][3]
    written = missing = degenerate = 0
    for d, s in scenarios(args.name):
        src = matrix_file(OUT / args.name, f"{d}__{tag(args.name)}", s)
        if src is None:
            missing += 1
            continue
        if not np.any(read_matrix(src)[0]):
            degenerate += 1
            continue
        dest = SEEDS / target / "prisma" / d / s / "seed.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        written += 1
    print(f"{args.name} -> reference/seeds/{target}/prisma: wrote {written}, missing {missing}, all-zero {degenerate}")
    return 0


def command_change(args) -> int:
    """Old FDs (reference/seeds/{plain,md5}/prisma) vs fixed FDs (*_fdfix) on the old cases."""
    from scipy.optimize import linear_sum_assignment
    report = {}
    # baselines: the old-FD seeds the 382 study used (reference/seeds/<r>/prisma; the md5 ones are the plain-feature
    # artefact of FINDINGS 120), and the 2026-09-30 fresh old-FD rerun with regime-specific names (prisma_table5_fresh)
    baselines = []
    for old, new, regime in (("plain", "plain_fdfix", "plain"), ("md5", "md5_fdfix", "md5")):
        baselines.append((f"{old}:reference_seeds", new,
                          lambda d, s, old=old: SEEDS / old / "prisma" / d / s / "seed.csv"))
        baselines.append((f"{old}:t5fresh", new,
                          lambda d, s, regime=regime: matrix_file(FRESH_T5[regime], f"{d}__t5-{regime}", s)))
    for label, new, base in baselines:
        rows = []
        for d in OLD13:
            for sc in sorted(p for p in (SEEDS / new / "prisma" / d).glob("*") if p.is_dir()) \
                    if (SEEDS / new / "prisma" / d).is_dir() else []:
                a, b = base(d, sc.name), sc / "seed.csv"
                if a is None or not a.is_file() or not np.any(read_matrix(a)[0]):
                    rows.append({"case": f"{d}/{sc.name}", "old_missing": True})
                    continue
                A, B = read_matrix(a)[0], read_matrix(b)[0]
                ha = set(zip(*linear_sum_assignment(-A)))
                hb = set(zip(*linear_sum_assignment(-B)))
                rows.append({"case": f"{d}/{sc.name}", "identical": bool(np.array_equal(A, B)),
                             "max_abs": float(np.abs(A - B).max()), "mean_abs": float(np.abs(A - B).mean()),
                             "argmax_rows_changed": int((A.argmax(1) != B.argmax(1)).sum()), "rows": A.shape[0],
                             "hungarian_pairs_changed": len(ha ^ hb) // 2})
        old_only = []
        old = label.split(":")[0]
        cmp = [r for r in rows if "identical" in r]
        report[label] = {
            "compared": len(cmp), "identical": sum(r["identical"] for r in cmp),
            "changed": sum(not r["identical"] for r in cmp),
            "max_abs_over_cases": max((r["max_abs"] for r in cmp), default=0.0),
            "mean_of_mean_abs": float(np.mean([r["mean_abs"] for r in cmp])) if cmp else 0.0,
            "cases_with_argmax_change": sum(r["argmax_rows_changed"] > 0 for r in cmp),
            "cases_with_hungarian_change": sum(r["hungarian_pairs_changed"] > 0 for r in cmp),
            "new_only": [r["case"] for r in rows if r.get("old_missing")], "old_only": old_only,
            "rows": rows,
        }
        s = report[label]
        print(f"{label}: compared {s['compared']}, identical {s['identical']}, changed {s['changed']}, "
              f"max|diff| {s['max_abs_over_cases']:.3g}, mean|diff| {s['mean_of_mean_abs']:.3g}, "
              f"argmax changed in {s['cases_with_argmax_change']} cases, Hungarian 1:1 changed in "
              f"{s['cases_with_hungarian_change']}; new-only {len(s['new_only'])}, old-only {len(old_only)}")
    (OUT / "change_old_vs_fdfix.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare", "export", "change"])
    ap.add_argument("name", nargs="?", choices=sorted(RUNS))
    args = ap.parse_args()
    return {"run": command_run, "compare": command_compare, "export": command_export,
            "change": command_change}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
