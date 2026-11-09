"""Runs on a CPU server: build the split corpora, then run a shard of Schematch configurations.

Copied to $SMX_REMOTE_ROOT/smx/study/servers/ by dispatch.py. Standard library only (the hosts have Python 3.14).

    python3 remote_runner.py prepare --regime plain
        Builds $SMX_REMOTE_ROOT/smx/study/tuning/corpus/<regime>/<split>/<dataset>/<scenario> as symlinks into
        $SMX_REMOTE_ROOT/smx/corpus/<regime>/, from tuning/splits.json and $SMX_REMOTE_ROOT/smx/cases.json.

    python3 remote_runner.py run --suffix screen_plain --list <file> --workers 48 --xmx 6g
        Runs every label in <file> (one per line) as its own JVM, <workers> at a time, with the
        config's jvm.properties as -D flags. A finished run leaves results/<suffix>/<label>/DONE and
        is skipped on restart. The full Schematch log of a successful run is cut to its last 200
        lines (run.tail.log); failed runs keep the full log. Progress goes to
        results/<suffix>/_status_<host>.json.

        A configuration whose first-line matcher is PythonServerMatcher gets its own seed server
        ($SMX_REMOTE_ROOT/smx/pyroot, see CPU_SERVERS.md) on a free port for the length of its JVM, so the
        single-threaded server never serialises the workers; the port goes into a copy of the
        config. The server's log is kept as results/.../seed_server.log.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SMX = Path.home() / "smx"
STUDY = SMX / "study"
JAVA = Path.home() / "opt" / "jdk21" / "bin" / "java"
SCHEMATCH = SMX / "schematch"
PYROOT = SMX / "pyroot"                        # SMX_ROOT layout: code/, harness/, seed_server.py
PYTHON = Path.home() / "smx-py" / "venv" / "bin" / "python"
SEED_SERVER_ENV = {
    "SMX_ROOT": str(PYROOT),
    "HF_HOME": str(PYROOT / "hf"),
    "HF_HUB_OFFLINE": "1",
    "XDG_CACHE_HOME": str(PYROOT / "cache"),   # bdikit's models (Magneto)
    "SMX_DISABLED_METHODS": "isresmat,llm",   # ISResMat runs on the GPU server; LLM needs approval
    "NLTK_DATA": f"{Path.home() / 'nltk_data'}:{PYROOT / 'nltk_data'}",  # local order: ~/nltk_data, then AppData\Roaming
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",    # bdikit imports litellm, which else fetches a price list from GitHub
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
}


def prepare(regime: str) -> int:
    splits = json.loads((STUDY / "tuning" / "splits.json").read_text(encoding="utf-8"))
    cases = {c["key"]: c for c in json.loads((SMX / "cases.json").read_text(encoding="utf-8"))}
    made = 0
    for split in ("lockbox", "dev", "screen"):
        for key in splits[split]:
            c = cases[key]
            target = SMX / "corpus" / regime / c["dataset"] / c["scenario"]
            link = STUDY / "tuning" / "corpus" / regime / split / c["dataset"] / c["scenario"]
            if not target.is_dir():
                print(f"missing corpus scenario {target}", file=sys.stderr)
                return 1
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)
                made += 1
    print(f"{socket.gethostname()}: corpus links for {regime} ready ({made} new)")
    return 0


def classpath_jars() -> list[str]:
    """The dependency jars in Maven's classpath order (servers/classpath.txt, written by dispatch.py push from cp.txt).

    `lib/*` would hand the JVM the jars in directory order, which differs between hosts; with two
    Jackson versions on the classpath that broke YAML parsing on some hosts and not on others.
    """
    order = STUDY / "servers" / "classpath.txt"
    if not order.is_file():
        raise SystemExit(f"{order} missing: the classpath order is required (see dispatch.py push)")
    jars = [SCHEMATCH / "lib" / n.strip() for n in order.read_text(encoding="utf-8").splitlines() if n.strip()]
    missing = [j for j in jars if not j.is_file()]
    if missing:
        raise SystemExit(f"jars missing from {SCHEMATCH / 'lib'}: {missing[:3]}")
    return [str(j) for j in jars]


def shard_config(suffix: str, label: str, dataset: str) -> tuple[Path, Path]:
    """A one-dataset copy of a configuration (task "label::dataset"), and where its results go.

    The copy differs only in datasets.yaml (that dataset alone) and resultsDir
    (results/<suffix>/<label>/_shards/<dataset>); dispatch.py collect merges the shards back.
    """
    src = STUDY / "tuning" / "configs" / suffix / label
    dst = STUDY / "tuning" / "shard_configs" / suffix / label / dataset
    results = STUDY / "tuning" / "results" / suffix / label / "_shards" / dataset
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            text = f.read_text(encoding="utf-8")
            if f.name == "datasets.yaml":
                text = f'---\nname: "{dataset}"\npath: "{dataset}"\n'
            elif f.name == "general.yaml":
                text = "\n".join(f'resultsDir: "{results.as_posix()}"' if l.startswith("resultsDir:") else l
                                 for l in text.splitlines()) + "\n"
            (dst / f.name).write_text(text, encoding="utf-8")
    return dst, results


def uses_seed_server(config: Path) -> bool:
    f = config / "first_line_matchers.yaml"
    return f.is_file() and 'name: "PythonServerMatcher"' in f.read_text(encoding="utf-8")


def start_seed_server(log: Path) -> tuple[subprocess.Popen, int]:
    """A private seed server on a free port; returns once /health answers."""
    env = {**os.environ, **SEED_SERVER_ENV,
           "PATH": f"{JAVA.parent}{os.pathsep}{os.environ.get('PATH', '')}"}  # COMA calls java
    fh = log.open("w", encoding="utf-8")
    proc = subprocess.Popen([str(PYTHON), "-u", str(PYROOT / os.environ.get("SMX_SEED_SERVER", "seed_server.py")), "--port", "0"],
                            cwd=PYROOT, env=env, stdout=subprocess.PIPE, stderr=fh, text=True,
                            start_new_session=True)
    line = proc.stdout.readline()  # "seed_server listening on 127.0.0.1:<port>; methods: ..."
    fh.write(line)
    if "listening on 127.0.0.1:" not in line:
        stop_seed_server(proc)
        raise RuntimeError(f"seed server did not start, see {log}")
    port = int(line.split("127.0.0.1:")[1].split(";")[0])
    import urllib.request
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
            return proc, port
        except OSError:
            time.sleep(1)
    stop_seed_server(proc)
    raise RuntimeError(f"seed server on port {port} never answered /health, see {log}")


def stop_seed_server(proc: subprocess.Popen) -> None:
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)  # its own session: Magneto's spawned workers go too
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def with_port(config: Path, port: int) -> Path:
    """A copy of the config whose PythonServerMatcher talks to the given port."""
    dst = STUDY / "tuning" / "port_configs" / f"{config.parent.name}__{config.name}__{port}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in config.iterdir():
        if f.is_file():
            text = f.read_text(encoding="utf-8")
            if f.name == "first_line_matchers.yaml":
                # unquoted: the field is an Integer, and Matcher.configure casts the YAML value;
                # a port already in the config (e.g. 5102 of a separate local server) is replaced
                text = "".join(l for l in text.splitlines(keepends=True) if not l.lstrip().startswith("serverPort:"))
                text = text.replace("params:\n", f"params:\n  serverPort: [{port}]\n", 1)
            (dst / f.name).write_text(text, encoding="utf-8")
    return dst


def run_one(suffix: str, task: str, xmx: str) -> tuple[str, int, float]:
    label, _, dataset = task.partition("::")
    if dataset:
        config, results = shard_config(suffix, label, dataset)
    else:
        config = STUDY / "tuning" / "configs" / suffix / label
        results = STUDY / "tuning" / "results" / suffix / label
    results.mkdir(parents=True, exist_ok=True)
    server = None
    if uses_seed_server(config):
        try:
            server, port = start_seed_server(results / "seed_server.log")
        except RuntimeError as exc:
            (results / "run.log").write_text(f"{exc}\n", encoding="utf-8")
            return task, 98, 0.0
        config = with_port(config, port)
    try:
        return _run_jvm(task, config, results, xmx)
    finally:
        if server is not None:
            stop_seed_server(server)
            if (results / "DONE").is_file():
                import shutil
                shutil.rmtree(config, ignore_errors=True)  # the per-port config copy


def _run_jvm(task: str, config: Path, results: Path, xmx: str) -> tuple[str, int, float]:
    props = []
    jvm = config / "jvm.properties"
    if jvm.is_file():
        props = ["-D" + line.strip() for line in jvm.read_text(encoding="utf-8").splitlines() if "=" in line]
    # opt-in classpath overlay (e.g. ER10: dev build), a directory under $SMX_REMOTE_ROOT/smx/schematch placed before the
    # shared classes; tasks without smx.cp.overlay run exactly as before
    overlay = [str(SCHEMATCH / p.split("=", 1)[1]) for p in props if p.startswith("-Dsmx.cp.overlay=")]
    if overlay and not Path(overlay[0]).is_dir():
        raise SystemExit(f"classpath overlay {overlay[0]} missing on this host")
    cmd = [str(JAVA), f"-Xmx{xmx}", "-XX:+UseSerialGC", *props, "-cp",
           os.pathsep.join([str(config), *overlay, str(SCHEMATCH / "classes"), *classpath_jars()]),
           "de.uni_marburg.schematch.Main"]
    log = results / "run.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=SCHEMATCH, stdout=fh, stderr=subprocess.STDOUT)
        # wait4 reports this child's own resource use: ru_maxrss is its peak resident set (KiB)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = code = os.waitstatus_to_exitcode(status)
    elapsed = time.time() - t0
    peak_mb = usage.ru_maxrss / 1024
    cpu_s = usage.ru_utime + usage.ru_stime
    text = log.read_text(encoding="utf-8", errors="replace")
    stats = [l for l in text.splitlines() if l.startswith("SFSTATS\t")]
    if stats:  # per scenario: dataset, scenario, iterations, converged, schema/propagation graph sizes, ms
        (results / "sf_stats.tsv").write_text("\n".join(l.split("\t", 1)[1] for l in stats) + "\n", encoding="utf-8")
    if code == 0:
        lines = text.splitlines()[-200:]
        (results / "run.tail.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.unlink()
        # wall time first: merge_shards sums it; peak RSS and CPU time of the JVM after it
        (results / "DONE").write_text(f"{elapsed:.1f}s {socket.gethostname()} peak_rss_mb={peak_mb:.0f} "
                                      f"cpu_s={cpu_s:.1f}\n", encoding="utf-8")
    return task, code, elapsed


def task_done(root: Path, task: str) -> bool:
    label, _, dataset = task.partition("::")
    return ((root / label / "_shards" / dataset / "DONE") if dataset else (root / label / "DONE")).is_file()


def run(suffix: str, list_file: str, workers: int, xmx: str) -> int:
    labels = [l.strip() for l in Path(list_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    root = STUDY / "tuning" / "results" / suffix
    root.mkdir(parents=True, exist_ok=True)
    todo = [t for t in labels if not task_done(root, t)]   # list order is kept: largest tasks first
    host = socket.gethostname()
    status = {"host": host, "total": len(labels), "already_done": len(labels) - len(todo), "done": 0,
              "failed": [], "workers": workers, "xmx": xmx, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def write_status():
        status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (root / f"_status_{host}.json").write_text(json.dumps(status, indent=1), encoding="utf-8")

    write_status()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, suffix, l, xmx) for l in todo]
        for f in as_completed(futures):
            label, code, elapsed = f.result()
            if code == 0:
                status["done"] += 1
            else:
                status["failed"].append(label)
            write_status()
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_status()
    print(f"{host}: {status['done']} done, {len(status['failed'])} failed")
    return 0 if not status["failed"] else 1


def host_status(suffix: str) -> int:
    """This host's runner status JSON plus load and running JVMs, for dispatch.py status."""
    host = socket.gethostname()
    f = STUDY / "tuning" / "results" / suffix / f"_status_{host}.json"
    s = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    jvms = subprocess.run(["pgrep", "-u", os.environ.get("USER", "<remote-user>"), "-f", "[s]chematch.Main"],
                          capture_output=True, text=True).stdout.split()
    s["load"] = open("/proc/loadavg").read().split()[:3]
    s["jvms"] = len(jvms)
    s["host"] = host
    print(json.dumps(s))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status")
    st.add_argument("--suffix", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--regime", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suffix", required=True)
    r.add_argument("--list", required=True)
    r.add_argument("--workers", type=int, required=True)
    r.add_argument("--xmx", default="6g")
    t = sub.add_parser("run-task", help="one task for servers/scheduler.py; leaves a done/failed marker")
    t.add_argument("--suffix", required=True)
    t.add_argument("--task", required=True)
    t.add_argument("--xmx", default="6g")
    args = ap.parse_args()
    if args.cmd == "prepare":
        return prepare(args.regime)
    if args.cmd == "status":
        return host_status(args.suffix)
    if args.cmd == "run-task":
        return run_task(args.suffix, args.task, args.xmx)
    return run(args.suffix, args.list, args.workers, args.xmx)


def marker(suffix: str, task: str, state: str) -> Path:
    """tuning/sched/<suffix>/<task>.<state>: what scheduler.py polls (':' is kept out of file names)."""
    return STUDY / "tuning" / "sched" / suffix / f"{task.replace('::', '@@')}.{state}"


def run_task(suffix: str, task: str, xmx: str) -> int:
    for s in ("done", "failed", "running"):
        marker(suffix, task, s).unlink(missing_ok=True)
    m = marker(suffix, task, "running")
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    root = STUDY / "tuning" / "results" / suffix
    if task_done(root, task):
        code, elapsed = 0, 0.0
    else:
        try:
            _, code, elapsed = run_one(suffix, task, xmx)
        except Exception as exc:  # a crash of the runner itself counts as a failed task
            code, elapsed = 97, 0.0
            marker(suffix, task, "failed").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    m.unlink(missing_ok=True)
    marker(suffix, task, "done" if code == 0 else "failed").write_text(
        f"{code} {elapsed:.1f}s {socket.gethostname()}\n", encoding="utf-8")
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
