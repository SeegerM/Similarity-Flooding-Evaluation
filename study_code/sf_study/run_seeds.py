"""Run Schematch once per generated seed configuration (see make_configs.py).

For every `configs/<suffix>/<seed>/` it runs

    java -Xmx12g -cp "<config dir>;target/classes;<cp.txt>" de.uni_marburg.schematch.Main

from code/schematch, the same invocation the harness uses (the config directory comes first on
the classpath, so its YAML files shadow the ones in src/main/resources). Seeds that call Python
(`PythonServerMatcher`) are only run when seed_server.py answers on its port; otherwise they are
reported as skipped, never run against a dead server.

A finished run leaves `results/<suffix>/<seed>/DONE`; rerunning skips it unless --force.

Usage:
    python run_seeds.py --suffix plain_smoke
    python run_seeds.py --suffix plain --seeds levenshtein,wordnet_wup --force
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMATCH = HERE.parent / "code" / "schematch"


def classpath() -> str:
    """Schematch's dependency classpath, rebuilt whenever pom.xml is newer.

    Kept here rather than reusing code/schematch/cp.txt, which the harness owns and which
    predates the commons-codec dependency of PhoneticMatcher.
    """
    out = HERE / "cp.txt"
    pom = SCHEMATCH / "pom.xml"
    if not out.is_file() or out.stat().st_mtime < pom.stat().st_mtime:
        subprocess.run(f'mvn -o -q dependency:build-classpath "-Dmdep.outputFile={out}"',
                       cwd=SCHEMATCH, shell=True, check=True)
    return out.read_text(encoding="utf-8").strip()


def server_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True, help="configs/<suffix>, e.g. plain or plain_smoke")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--port", type=int, default=5101)
    ap.add_argument("--xmx", default="12g")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = HERE / "configs" / args.suffix
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist; run make_configs.py first")
    seeds = [s for s in args.seeds.split(",") if s] or sorted(p.name for p in root.iterdir() if p.is_dir())
    deps = classpath()
    sep = os.pathsep
    status = {}
    for seed in seeds:
        config = root / seed
        results = HERE / "results" / args.suffix / seed
        done = results / "DONE"
        if done.is_file() and not args.force:
            status[seed] = "done (skipped)"
            continue
        first_line = (config / "first_line_matchers.yaml").read_text(encoding="utf-8")
        if "PythonServerMatcher" in first_line and not server_up(args.port):
            status[seed] = f"skipped: seed_server.py not reachable on port {args.port}"
            continue
        if "EmbdiMatcher" in first_line and not server_up(5001):
            status[seed] = "skipped: upstream EmbDI server not reachable on port 5001"
            continue
        if "<path to" in first_line or "<module:function>" in first_line:
            status[seed] = "skipped: needs a local resource path in its config (see MATCHERS.md)"
            continue
        marker = config / "seed_file_dir.txt"
        if marker.is_file() and not Path(marker.read_text(encoding="utf-8").strip()).is_dir():
            status[seed] = "skipped: FILE seed directory missing (run learned_combiner.py first)"
            continue
        results.mkdir(parents=True, exist_ok=True)
        cmd = ["java", f"-Xmx{args.xmx}", "-cp",
               sep.join([str(config), str(SCHEMATCH / "target" / "classes"), deps]),
               "de.uni_marburg.schematch.Main"]
        t0 = time.perf_counter()
        with (results / "run.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(cmd, cwd=SCHEMATCH, stdout=log, stderr=subprocess.STDOUT).returncode
        elapsed = time.perf_counter() - t0
        if code == 0:
            done.write_text(f"{elapsed:.1f}s\n", encoding="utf-8")
            status[seed] = f"ok ({elapsed:.0f}s)"
        else:
            status[seed] = f"FAILED exit {code} ({elapsed:.0f}s), see {results / 'run.log'}"
        print(f"{seed:28s} {status[seed]}", flush=True)
    print()
    for seed, s in status.items():
        print(f"{seed:28s} {s}")
    return 0 if all(not s.startswith("FAILED") for s in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
