"""Run a tuning stage: every configuration directory as its own Schematch JVM, in parallel.

Each run gets the configuration's jvm.properties as -D flags (the smx.sf.* switches). A finished
run leaves results/<stage>_<regime>/<label>/DONE; reruns skip it unless --force, so a stopped stage
continues where it stopped.

Usage:
    python run_tuning.py --stage screen --regime plain --workers 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from run_seeds import SCHEMATCH, classpath  # noqa: E402


def run_one(config: Path, results: Path, deps: str, xmx: str) -> tuple[str, int, float]:
    results.mkdir(parents=True, exist_ok=True)
    props = []
    jvm = config / "jvm.properties"
    if jvm.is_file():
        for line in jvm.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                props.append("-D" + line.strip())
    cmd = ["java", f"-Xmx{xmx}", *props, "-cp",
           os.pathsep.join([str(config), str(SCHEMATCH / "target" / "classes"), deps]),
           "de.uni_marburg.schematch.Main"]
    t0 = time.perf_counter()
    with (results / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=SCHEMATCH, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = time.perf_counter() - t0
    if code == 0:
        (results / "DONE").write_text(f"{elapsed:.1f}s\n", encoding="utf-8")
    return config.name, code, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("screen", "dev", "lockbox"), required=True)
    ap.add_argument("--regime", choices=("plain", "md5"), default="plain")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--xmx", default="4g")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N configurations only (a smoke test)")
    ap.add_argument("--only", default="", help="comma-separated labels to run (a smoke test)")
    args = ap.parse_args()

    suffix = f"{args.stage}_{args.regime}"
    root = HERE / "configs" / suffix
    if not root.is_dir():
        raise SystemExit(f"{root} missing; run make_tuning_configs.py first")
    configs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c.name in wanted]
    if args.limit:
        configs = configs[: args.limit]
    todo = [c for c in configs if args.force or not (HERE / "results" / suffix / c.name / "DONE").is_file()]
    print(f"{len(configs)} configurations, {len(todo)} to run, {args.workers} workers", flush=True)
    deps = classpath()
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c, HERE / "results" / suffix / c.name, deps, args.xmx) for c in todo]
        for k, f in enumerate(as_completed(futures), 1):
            label, code, elapsed = f.result()
            if code != 0:
                failed.append(label)
            if code != 0 or k % 20 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] {label} {'ok' if code == 0 else f'FAILED exit {code}'} "
                      f"({elapsed:.0f}s; {(time.time() - t0) / 3600:.2f} h elapsed)", flush=True)
    print(f"done; {len(failed)} failed" + (f": {failed[:10]}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
