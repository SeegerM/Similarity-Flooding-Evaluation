"""Run a tuning stage's Schematch configurations on the CPU servers (servers/CPU_SERVERS.md).

    python dispatch.py push    --suffix screen_plain     # stage + upload + fan out + corpus links
    python dispatch.py start   --suffix screen_plain     # shard by free capacity, start under nohup
    python dispatch.py status  --suffix screen_plain     # per-host progress, load, failures
    python dispatch.py collect --suffix screen_plain     # copy results back into tuning/results/

push: the generated configs (tuning/configs/<suffix>) are rewritten from this workstation's paths
to the servers' (`<repo-root>/sf-matcher-study` -> `$SMX_REMOTE_ROOT/smx/study`), packed together
with the FILE seeds they read, splits.json and remote_runner.py, uploaded once to the hub (<cpu-hub>)
and copied from there to every host over the LAN. Each host then builds its split corpora as
symlinks into $SMX_REMOTE_ROOT/smx/corpus/<regime>.

start: refuses a host that someone else is using (load > 4 or another user logged in) unless
--force. Per host it runs `cores - reserve` JVMs with `-Xmx`, capped so that all of them together
use at most --ram-share of the available RAM. Labels already DONE locally or remotely are skipped;
the rest are split across hosts in proportion to their workers.

collect: copies results/<suffix> back from every host (DONE markers, SF matrices, log tails;
full logs only of failed runs), so score_tuning.py reads them as if they had run locally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
TUNING = STUDY / "tuning"
LOCAL_PREFIX = STUDY.as_posix()                       # <repo-root>/sf-matcher-study
REMOTE_HOME = "$SMX_REMOTE_ROOT"
REMOTE_STUDY = f"{REMOTE_HOME}/smx/study"
HUB = "<cpu-hub>"
HOSTS = ["<cpu-host-00>", "<cpu-host-01>", "<cpu-hub>", "<cpu-host-03>", "<cpu-host-04>", "<cpu-host-05>", "<cpu-host-06>", "<cpu-host-07>", "<cpu-host-08>", "<cpu-host-09>"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]  # 2026-10-05: hung jumphost sessions drop in ~3 min


def ssh(host: str, command: str, timeout: int = 600, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    # no input: stdin from /dev/null, else ssh through the jumphost can hang after the remote command ends (as in collect)
    kw = {"input": stdin} if stdin is not None else {"stdin": subprocess.DEVNULL}
    return subprocess.run(SSH + [host, command], capture_output=True, timeout=timeout, **kw)


def out(p: subprocess.CompletedProcess) -> str:
    return (p.stdout + p.stderr).decode("utf-8", "replace").strip()


def regime_of(suffix: str) -> str:
    return suffix.split("_", 1)[1]


# ------------------------------------------------------------------------------------------ push

def rewrite(text: str) -> str:
    return text.replace(LOCAL_PREFIX, REMOTE_STUDY)


def push(suffix: str, hosts: list[str]) -> int:
    regime = regime_of(suffix)
    cfg_root = TUNING / "configs" / suffix
    if not cfg_root.is_dir():
        raise SystemExit(f"{cfg_root} missing; run make_tuning_configs.py first")
    seed_dirs = set()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "study"
        n = 0
        for cfg in sorted(p for p in cfg_root.iterdir() if p.is_dir()):
            dest = stage / "tuning" / "configs" / suffix / cfg.name
            dest.mkdir(parents=True)
            for f in cfg.iterdir():
                if f.suffix in (".yaml", ".properties"):
                    with open(dest / f.name, "w", encoding="utf-8", newline="\n") as fh:  # LF for Linux
                        fh.write(rewrite(f.read_text(encoding="utf-8")))
            meta = json.loads((cfg / "meta.json").read_text(encoding="utf-8"))
            seed_dirs.add(Path(meta["seed_dir"]))
            # also every seed folder the configuration actually reads (initialMatrixDir in the YAML, smx.sf.idMatrixDir
            # in jvm.properties; 2026-10-05, X60: meta.json's seed_dir may name only the clone's original matcher)
            for f in cfg.iterdir():
                if f.suffix in (".yaml", ".properties"):
                    for ref in re.findall(r'(?:initialMatrixDir: \["|smx\.sf\.idMatrixDir=)([^"\n]+)', f.read_text(encoding="utf-8")):
                        ref_path = Path(ref.strip())
                        if ref_path.is_dir() and STUDY.resolve() in ref_path.resolve().parents:
                            seed_dirs.add(ref_path)
            n += 1
        for sdir in sorted(seed_dirs):
            rel = Path(sdir).resolve().relative_to(STUDY.resolve())
            shutil.copytree(sdir, stage / rel)
        shutil.copy(TUNING / "splits.json", stage / "tuning" / "splits.json")
        (stage / "servers").mkdir()
        shutil.copy(HERE / "remote_runner.py", stage / "servers" / "remote_runner.py")
        # the dependency jars in Maven's classpath order: `lib/*` would use directory order, which
        # differs between hosts (two Jackson versions are on the classpath)
        jars = [Path(j.replace("\\", "/")).name for j in (STUDY / "cp.txt").read_text(encoding="utf-8").strip().split(";") if j]
        with open(stage / "servers" / "classpath.txt", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(jars) + "\n")
        bundle = Path(tmp) / "bundle.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            tar.add(stage, arcname="study")
        size = bundle.stat().st_size / 1e6
        print(f"bundle: {n} configs, {len(seed_dirs)} seed dirs, {size:.1f} MB")
        p = subprocess.run(SSH + [HUB, "mkdir -p $SMX_REMOTE_ROOT/smx && tar xzf - -C $SMX_REMOTE_ROOT/smx && echo hub-ok"],
                           input=bundle.read_bytes(), capture_output=True, timeout=3600)
        print(f"{HUB}: {out(p)}")
        if p.returncode != 0:
            return 1
    others = " ".join(h for h in hosts if h != HUB)
    p = ssh(HUB, f"for h in {others}; do (rsync -a --exclude tuning/results $SMX_REMOTE_ROOT/smx/study $h:smx/ && echo $h synced) & done; wait", 3600)
    print(out(p))
    with ThreadPoolExecutor(min(3, len(hosts))) as pool:
        for r in pool.map(lambda h: out(ssh(h, f"python3 $SMX_REMOTE_ROOT/smx/study/servers/remote_runner.py prepare --regime {regime}")), hosts):
            print(r)
    return 0


# ----------------------------------------------------------------------------------------- start

PROBE = ("echo $(nproc) $(free -g | awk '/Mem:/{print $7}') $(cut -d' ' -f1 /proc/loadavg) "
         "$(who | awk '{print $1}' | grep -v $USER | sort -u | wc -l) "
         "$(pgrep -u $USER -f '[r]emote_runner.py run' | wc -l) "
         "$(pgrep -u $USER -f '[s]chematch.Main' | wc -l)")


def probe(host: str) -> dict:
    p = ssh(host, PROBE, 60)
    try:
        cores, avail, load, others, running, jvms = out(p).split()[:6]
        return {"host": host, "cores": int(cores), "avail_gb": int(avail), "load": float(load),
                "other_users": int(others), "running": int(running), "jvms": int(jvms), "ok": True}
    except Exception:
        return {"host": host, "ok": False, "error": out(p)[:200]}


def dataset_tasks(suffix: str, labels: list[str]) -> tuple[list[str], dict[str, int]]:
    """One task per (configuration, dataset) that is not DONE locally, and each task's size.

    Size is the number of the stage's scenarios in that dataset (from the stage corpus), so the
    largest tasks can be scheduled first.
    """
    stage, regime = suffix.split("_", 1)
    corpus = TUNING / "corpus" / regime / stage
    n_scen = {d.name: sum(1 for _ in d.iterdir()) for d in corpus.iterdir() if d.is_dir()} if corpus.is_dir() else {}
    tasks, size = [], {}
    for label in labels:
        meta = json.loads((TUNING / "configs" / suffix / label / "meta.json").read_text(encoding="utf-8"))
        n_scen = meta.get("dataset_sizes", n_scen)  # seed-study exports (export_seed_study.py) carry their own
        for d in meta["datasets"]:
            if not (TUNING / "results" / suffix / label / "_shards" / d / "DONE").is_file():
                t = f"{label}::{d}"
                tasks.append(t)
                size[t] = n_scen.get(d, 1)
    return tasks, size


def start(suffix: str, hosts: list[str], xmx_gb: int, reserve: int, ram_share: float, force: bool,
          split_datasets: bool = False, share: bool = False, only: str = "") -> int:
    labels = sorted(p.name for p in (TUNING / "configs" / suffix).iterdir() if p.is_dir())
    local_done = {l for l in labels if (TUNING / "results" / suffix / l / "DONE").is_file()}
    todo = [l for l in labels if l not in local_done]
    size = {l: 1 for l in todo}
    if split_datasets:
        todo, size = dataset_tasks(suffix, todo)
    if only:  # restrict to tasks (label or label::dataset) matching a regular expression
        todo = [t for t in todo if re.search(only, t)]
        size = {t: size[t] for t in todo}
        print(f"--only {only!r}: {len(todo)} tasks")
        if not todo:
            return 0
    info = list(ThreadPoolExecutor(min(3, len(hosts))).map(probe, hosts))
    usable = []
    for i in info:
        if not i["ok"]:
            print(f"{i['host']}: unreachable ({i.get('error')})")
            continue
        busy = (i["load"] > 4 and not share) or i["other_users"] > 0
        if i["running"] and not share:
            print(f"{i['host']}: a runner is already active; skipped (--share runs alongside it)")
            continue
        if busy and not force:
            print(f"{i['host']}: in use (load {i['load']}, {i['other_users']} other users); skipped (--force to override)")
            continue
        # with --share the JVMs already running (another stage's runner) keep their cores; free RAM
        # (avail_gb) already excludes their memory
        workers = min(i["cores"] - reserve - (i["jvms"] if share else 0),
                      int(i["avail_gb"] * ram_share // (xmx_gb + 1)))
        if workers < 1:
            continue
        usable.append((i["host"], workers))
        print(f"{i['host']}: {i['cores']} cores, {i['avail_gb']} GB free, load {i['load']} -> {workers} workers x {xmx_gb} GB")
    if not usable:
        return 1
    total = sum(w for _, w in usable)
    shards = {h: [] for h, _ in usable}
    if split_datasets:
        # largest task first, each to the host with the least work per worker so far (LPT); each
        # host's list keeps that order, so its runner also starts with its largest tasks
        load = {h: 0.0 for h, _ in usable}
        workers_of = dict(usable)
        for t in sorted(todo, key=lambda t: (-size[t], t)):
            h = min(load, key=lambda x: load[x] / workers_of[x])
            shards[h].append(t)
            load[h] += size[t]
    else:
        # interleave labels so every host gets a mix of seeds and configurations, split by capacity
        quota = {h: len(todo) * w / total for h, w in usable}
        for label in sorted(todo, key=lambda l: (l.split("__", 1)[1], l)):
            h = max(quota, key=lambda x: quota[x] - len(shards[x]))
            shards[h].append(label)
    for host, workers in usable:
        lst = "\n".join(shards[host]) + "\n"
        lst_path = f"$SMX_REMOTE_ROOT/smx/study/tuning/shards/{suffix}/{host}.txt"
        ssh(host, f"mkdir -p $SMX_REMOTE_ROOT/smx/study/tuning/shards/{suffix} && cat > {lst_path}", 120, lst.encode())
        cmd = (f"cd $SMX_REMOTE_ROOT/smx/study || exit 1; setsid nohup python3 servers/remote_runner.py run --suffix {suffix} "
               f"--list tuning/shards/{suffix}/{host}.txt --workers {workers} --xmx {xmx_gb}g "
               f"> tuning/results/{suffix}.{host}.runner.log 2>&1 < /dev/null & echo started")
        ssh(host, f"mkdir -p $SMX_REMOTE_ROOT/smx/study/tuning/results; {cmd}", 120)
        print(f"{host}: started {len(shards[host])} {'tasks' if split_datasets else 'configurations'}")
    print(f"{len(todo)} {'(configuration, dataset) tasks' if split_datasets else 'configurations'} "
          f"dispatched to {len(usable)} hosts ({total} workers); {len(local_done)} configurations already done locally")
    return 0


# -------------------------------------------------------------------------------------- enqueue

def enqueue(suffix: str, xmx_gb: int, priority: int, split_datasets: bool, only: str, solo: int = 0) -> int:
    """Queue a stage's open tasks (largest first) on the hub's scheduler instead of fixed shares."""
    labels = sorted(p.name for p in (TUNING / "configs" / suffix).iterdir() if p.is_dir())
    todo = [l for l in labels if not (TUNING / "results" / suffix / l / "DONE").is_file()]
    size = {l: 1 for l in todo}
    if split_datasets:
        todo, size = dataset_tasks(suffix, todo)
    if only:
        todo = [t for t in todo if re.search(only, t)]
    todo.sort(key=lambda t: (-size.get(t, 1), t))
    p = ssh(HUB, f"python3 $SMX_REMOTE_ROOT/smx/study/servers/scheduler.py add --suffix {suffix} --xmx-gb {xmx_gb} "
                 f"--priority {priority}" + (f" --solo {solo}" if solo else ""), 300, ("\n".join(todo) + "\n").encode())
    print(out(p))
    return p.returncode


# ---------------------------------------------------------------------------------------- status

def status(suffix: str, hosts: list[str]) -> int:
    """One connection through the jumphost (to the hub); the hub asks the others over the LAN.

    Nine parallel sessions through the jumphost get refused ("Connection closed by UNKNOWN"),
    and a watcher polling that way would also hammer the jumphost.
    """
    probe_cmd = f"python3 $SMX_REMOTE_ROOT/smx/study/servers/remote_runner.py status --suffix {suffix}"
    script = "; ".join(f"echo '=== {h}'; " + (probe_cmd if h == HUB else f"ssh {h} {probe_cmd} 2>&1")
                       for h in hosts)
    text_all = out(ssh(HUB, script, 300))
    blocks = {b.split("\n", 1)[0].strip(): (b.split("\n", 1)[1] if "\n" in b else "")
              for b in text_all.split("=== ")[1:]}
    total = done = failed = 0
    unknown = 0
    for h in hosts:
        text = blocks.get(h, "")
        m = re.search(r"\{.*\}", text, re.S)
        s = json.loads(m.group(0)) if m else {}
        tail = f"LOAD {' '.join(s.get('load', []))} JVMS {s.get('jvms', '?')}"
        if "total" not in s:
            unknown += 1
            print(f"{h}: no runner status  {tail}  {text.strip()[-120:] if not m else ''}")
            continue
        total += s["total"]
        done += s["done"] + s["already_done"]
        failed += len(s["failed"])
        state = "finished" if "finished" in s else "running"
        print(f"{h}: {s['done'] + s['already_done']}/{s['total']} done, {len(s['failed'])} failed, {state}  {tail}")
    print(f"total: {done}/{total} done, {failed} failed, {unknown} hosts without status")
    return 0


# --------------------------------------------------------------------------------------- collect

def collect(suffix: str, hosts: list[str]) -> int:
    dest = TUNING / "results" / suffix
    dest.mkdir(parents=True, exist_ok=True)

    def one(h):
        cmd = (f"cd $SMX_REMOTE_ROOT/smx/study/tuning/results && [ -d {suffix} ] && "
               f"tar czf - --exclude='{suffix}/*/*/*/run.log' {suffix}")
        # stdin=DEVNULL: from a detached follower, ssh otherwise inherits an odd stdin and was seen to hang after the
        # remote command had ended (2026-10-05, hosts without the suffix folder)
        p = subprocess.run(SSH + [h, cmd], capture_output=True, timeout=3600, stdin=subprocess.DEVNULL)
        if p.returncode != 0 or not p.stdout:
            return h, 0, p.stderr.decode("utf-8", "replace")[:200]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as fh:
            fh.write(p.stdout)
        n = 0
        # decided once, before extracting: checking DONE per member would skip the rest of a label
        # as soon as its own DONE (which tar may list before the run folder) had been extracted
        done_before = {p.name for p in dest.iterdir() if (p / "DONE").is_file()}
        with tarfile.open(fh.name) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if member.name.startswith(("/", "\\")) or ".." in parts or not (member.isfile() or member.isdir()):
                    continue  # only plain files and folders inside results/<suffix>
                if len(parts) >= 2 and parts[1] in done_before:
                    continue  # already DONE locally
                tar.extract(member, TUNING / "results")
                n += member.isfile()
        os.unlink(fh.name)
        return h, n, ""
    # 2 at a time: with the followers' scheduler polls this keeps us within 3 jumphost connections
    for h, n, err in ThreadPoolExecutor(min(2, len(hosts))).map(one, hosts):
        print(f"{h}: {n} files" + (f"  ({err.strip()})" if err.strip() else ""))
    merged = merge_shards(suffix)
    done = sum((p / "DONE").is_file() for p in dest.iterdir() if p.is_dir())
    print(f"{dest}: {done} configurations DONE locally" + (f" ({merged} merged from dataset shards)" if merged else ""))
    return 0


def merge_shards(suffix: str) -> int:
    """Turn a configuration whose dataset shards are all DONE into one ordinary run.

    results/<suffix>/<label>/_shards/<dataset>/<timestamp>/<dataset>/<scenario>/... becomes
    results/<suffix>/<label>/merged/<dataset>/<scenario>/..., then DONE is written and _shards is
    removed, so score_tuning.py reads it exactly like a single run. Incomplete configurations stay
    as shards until their last dataset arrives.
    """
    dest = TUNING / "results" / suffix
    merged = 0
    for label_dir in sorted(p for p in dest.iterdir() if (p / "_shards").is_dir()):
        if (label_dir / "DONE").is_file():
            continue
        meta_file = TUNING / "configs" / suffix / label_dir.name / "meta.json"
        if not meta_file.is_file():
            # a label of an earlier export of the same suffix (e.g. E2's stage 1 in seedstudy_ctl), still on the hosts
            # but no longer configured here: leave its shards alone (2026-10-05, X52)
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        shards = label_dir / "_shards"
        if not all((shards / d / "DONE").is_file() for d in meta["datasets"]):
            continue
        target = label_dir / "merged"
        for d in meta["datasets"]:
            runs = sorted(p for p in (shards / d).iterdir() if p.is_dir())
            if not runs or not (runs[-1] / d).is_dir():
                raise SystemExit(f"{label_dir.name}: shard {d} is DONE but has no output for {d}")
            target.mkdir(exist_ok=True)
            shutil.move(str(runs[-1] / d), str(target / d))
        # per-scenario SF statistics (remote_runner.py: SFSTATS lines), one file per configuration
        stats = [shards / d / "sf_stats.tsv" for d in meta["datasets"]]
        if any(f.is_file() for f in stats):
            with (label_dir / "sf_stats.tsv").open("w", encoding="utf-8", newline="\n") as fh:
                for f in stats:
                    if f.is_file():
                        fh.write(f.read_text(encoding="utf-8"))
        seconds = sum(float((shards / d / "DONE").read_text().split("s")[0]) for d in meta["datasets"])
        (label_dir / "DONE").write_text(f"{seconds:.1f}s merged from {len(meta['datasets'])} dataset shards\n",
                                        encoding="utf-8")
        shutil.rmtree(shards)
        merged += 1
    return merged


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("push", "start", "status", "collect", "enqueue"))
    ap.add_argument("--suffix", required=True, help="e.g. screen_plain")
    ap.add_argument("--hosts", default=",".join(HOSTS))
    ap.add_argument("--xmx-gb", type=int, default=6)
    ap.add_argument("--reserve", type=int, default=8, help="cores left free per host")
    ap.add_argument("--ram-share", type=float, default=0.7, help="max share of available RAM for all JVMs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split-datasets", action="store_true",
                    help="start: one task per (configuration, dataset), largest first; collect merges them")
    ap.add_argument("--share", action="store_true",
                    help="start: also use hosts where another stage's runner is active, on the cores it leaves free")
    ap.add_argument("--priority", type=int, default=5, help="enqueue: lower runs first")
    ap.add_argument("--solo", type=int, default=0, help="enqueue: timing pass, at most N per otherwise idle host")
    ap.add_argument("--only", default="", help="start: only tasks whose label (or label::dataset) matches this regex")
    args = ap.parse_args()
    hosts = [h for h in args.hosts.split(",") if h]
    if args.cmd == "push":
        return push(args.suffix, hosts)
    if args.cmd == "start":
        return start(args.suffix, hosts, args.xmx_gb, args.reserve, args.ram_share, args.force,
                     args.split_datasets, args.share, args.only)
    if args.cmd == "status":
        return status(args.suffix, hosts)
    if args.cmd == "enqueue":
        return enqueue(args.suffix, args.xmx_gb, args.priority, args.split_datasets, args.only, args.solo)
    return collect(args.suffix, hosts)


if __name__ == "__main__":
    raise SystemExit(main())
