"""Cluster scheduler: keeps every CPU host filled from one shared, prioritised task queue.

Runs on the hub (<cpu-hub>), standard library only. dispatch.py's `start` splits a stage across the
hosts once, so a host that finishes its share early sits idle; this scheduler instead starts the
next queued task wherever a slot is free. A task is the same unit as before (a configuration, or
"label::dataset" with --split-datasets) and is run by `remote_runner.py run-task`, which writes
its results exactly where `run` would, so `dispatch.py collect` is unchanged.

Free slots on a host = min(cores - reserve - busy, RAM cap), where busy counts every running
Schematch JVM (also those of `run`-mode runners) and 6 cores per ISResMat process, and the RAM
cap is available GB x ram-share / (Xmx + 1). Tasks start in priority order (lower first), then in
the order they were queued (dispatch.py enqueue queues the largest first).

State, on the hub under $SMX_REMOTE_ROOT/smx/sched/:
    queue.jsonl   one task per line: suffix, task, xmx_gb, priority, seq
    state.json    per task: host, status (launched / done / failed), time, tries
    STOP          create it to end the daemon after the current round

    python3 scheduler.py add --suffix layer1_md5 --xmx-gb 6 --priority 1 < tasks.txt
    python3 scheduler.py run            # the daemon (setsid nohup ... &)
    python3 scheduler.py status
    python3 scheduler.py retry-failed --suffix layer1_md5
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOSTS = ["<cpu-host-00>", "<cpu-host-01>", "<cpu-hub>", "<cpu-host-03>", "<cpu-host-04>", "<cpu-host-05>", "<cpu-host-06>", "<cpu-host-07>", "<cpu-host-08>", "<cpu-host-09>"]
HOME = Path.home()
SCHED = HOME / "smx" / "sched"
QUEUE = SCHED / "queue.jsonl"
STATE = SCHED / "state.json"
LOG = SCHED / "scheduler.log"
RESERVE = 2  # 2026-10-04 01:50, the authors: "lower the cores to 2 per host, nobody should be on these machines" (was 8)
RAM_SHARE = 0.7
ISRESMAT_CORES = 6
ROUND_SECONDS = 30
STALE_HOURS = 12

PROBE = ("echo $(nproc) $(free -g | awk '/Mem:/{print $7}') "
         "$(pgrep -u $USER -f 'schematch[.]Main' | wc -l) $(pgrep -u $USER -f -- '-m isresma[t]' | wc -l); "
         "cd $SMX_REMOTE_ROOT/smx/study/tuning/sched 2>/dev/null && find . -name '*.done' -o -name '*.failed'")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def key(suffix: str, task: str) -> str:
    return f"{suffix}|{task}"


def load_queue() -> list[dict]:
    if not QUEUE.is_file():
        return []
    return [json.loads(l) for l in QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(STATE)


def sh(host: str, command: str, timeout: int = 120) -> str:
    """Run on a host: locally on the hub, over the LAN elsewhere."""
    if host == socket.gethostname().split(".")[0]:
        cmd = ["bash", "-c", command]
    else:
        cmd = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def probe(host: str) -> dict | None:
    out = sh(host, PROBE).splitlines()
    try:
        cores, avail, jvms, isr = (int(x) for x in out[0].split())
    except (IndexError, ValueError):
        return None
    markers = {}
    for line in out[1:]:
        name = line.strip().removeprefix("./")  # <suffix>/<task with @@>.<done|failed>
        if "/" in name:
            suffix, rest = name.split("/", 1)
            task, _, status = rest.rpartition(".")
            markers[key(suffix, task.replace("@@", "::"))] = status
    return {"host": host, "cores": cores, "avail": avail, "busy": jvms + ISRESMAT_CORES * isr, "markers": markers}


def add(suffix: str, xmx_gb: int, priority: int, solo: int = 0) -> int:
    SCHED.mkdir(parents=True, exist_ok=True)
    queued = {key(q["suffix"], q["task"]) for q in load_queue()}
    seq = len(queued)
    new = []
    for line in sys.stdin.read().splitlines():
        task = line.strip()
        if task and key(suffix, task) not in queued:
            queued.add(key(suffix, task))
            new.append({"suffix": suffix, "task": task, "xmx_gb": xmx_gb, "priority": priority, "seq": seq,
                        **({"solo": solo} if solo else {})})
            seq += 1
    with QUEUE.open("a", encoding="utf-8") as fh:
        for q in new:
            fh.write(json.dumps(q) + "\n")
    print(f"queued {len(new)} tasks of {suffix} (priority {priority}, -Xmx{xmx_gb}g); "
          f"{len(queued)} in the queue overall")
    return 0


def run() -> int:
    SCHED.mkdir(parents=True, exist_ok=True)
    log(f"scheduler started on {socket.gethostname()}")
    while not (SCHED / "STOP").exists():
        queue = sorted(load_queue(), key=lambda q: (q["priority"], q["seq"]))
        state = load_state()
        # retry requests (written by `retry-failed`): applied here, so the daemon is the only writer of
        # state.json; editing it from outside raced with this loop's save and was silently undone
        for req in sorted(SCHED.glob("RETRY_*")):
            suffix = req.name[len("RETRY_"):]
            again = [k for k, v in state.items() if v["status"] == "failed" and k.startswith(suffix + "|")]
            for k in again:
                del state[k]
            req.unlink()
            save_state(state)
            log(f"retry: {len(again)} failed tasks of {suffix} queued again")
        with ThreadPoolExecutor(len(HOSTS)) as pool:
            infos = [i for i in pool.map(probe, HOSTS) if i]
        changed = False
        for info in infos:  # completions
            for k, status in info["markers"].items():
                s = state.get(k)
                if s and s["status"] == "launched" and s["host"] == info["host"]:
                    s["status"], s["t_end"] = status, time.time()
                    changed = True
                    if status == "failed":
                        log(f"FAILED {k} on {info['host']}")
        pending = [q for q in queue if key(q["suffix"], q["task"]) not in state]
        launched = 0
        for info in sorted(infos, key=lambda i: i["busy"] - i["cores"]):  # emptiest host first
            if not pending:
                break
            free = info["cores"] - RESERVE - info["busy"]
            batch, ram = [], info["avail"] * RAM_SHARE
            # "solo" tasks (timing passes, `add --solo N`): start on a host only while nothing else of ours runs
            # there, at most N of that suffix at once; they are skipped, not blocking, so plain tasks keep order
            on_host = [k for k, v in state.items() if v["status"] == "launched" and v["host"] == info["host"]]
            busy_now = max(info["busy"], len(on_host))
            i = 0
            while i < len(pending) and free > 0 and ram >= pending[i]["xmx_gb"] + 1:
                q = pending[i]
                if q.get("solo"):
                    mine = sum(1 for k in on_host if k.startswith(q["suffix"] + "|"))
                    mine_new = sum(1 for b in batch if b["suffix"] == q["suffix"])
                    others = busy_now - mine + sum(1 for b in batch if not b.get("solo"))
                    if others > 0 or mine + mine_new >= q["solo"]:
                        i += 1
                        continue
                batch.append(pending.pop(i))
                free -= 1
                ram -= q["xmx_gb"] + 1
            if not batch:
                continue
            cmd = "cd $SMX_REMOTE_ROOT/smx/study || exit 1; mkdir -p tuning/sched; " + " ".join(
                f"(setsid nohup python3 servers/remote_runner.py run-task --suffix {shlex.quote(q['suffix'])} "
                f"--task {shlex.quote(q['task'])} --xmx {q['xmx_gb']}g > /dev/null 2>&1 < /dev/null &);"
                for q in batch)
            sh(info["host"], cmd, 300)
            now = time.time()
            for q in batch:
                state[key(q["suffix"], q["task"])] = {"host": info["host"], "status": "launched", "t": now,
                                                      "tries": state.get(key(q["suffix"], q["task"]), {}).get("tries", 0) + 1}
            launched += len(batch)
            changed = True
        if changed:
            save_state(state)
        if launched:
            n = {s: sum(1 for v in state.values() if v["status"] == s) for s in ("launched", "done", "failed")}
            log(f"started {launched}; queue {len(pending)} waiting, {n['launched']} running, {n['done']} done, {n['failed']} failed")
        time.sleep(ROUND_SECONDS)
    log("STOP file found; scheduler ends")
    return 0


def status() -> int:
    queue, state = load_queue(), load_state()
    by = {}
    now = time.time()
    for q in queue:
        s = state.get(key(q["suffix"], q["task"]))
        st = s["status"] if s else "queued"
        if st == "launched" and now - s["t"] > STALE_HOURS * 3600:
            st = "stale?"
        by.setdefault(q["suffix"], {}).setdefault(st, 0)
        by[q["suffix"]][st] += 1
    for suffix, counts in sorted(by.items()):
        print(f"{suffix}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    hosts = {}
    for v in state.values():
        if v["status"] == "launched":
            hosts[v["host"]] = hosts.get(v["host"], 0) + 1
    print("running per host: " + ", ".join(f"{h} {n}" for h, n in sorted(hosts.items())))
    return 0


def retry_failed(suffix: str) -> int:
    """Ask the daemon to re-queue a suffix's failed tasks (it applies the request in its next round)."""
    n = sum(1 for k, v in load_state().items() if v["status"] == "failed" and k.startswith(suffix + "|"))
    (SCHED / f"RETRY_{suffix}").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    print(f"{n} failed tasks of {suffix} will be queued again in the scheduler's next round")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--suffix", required=True)
    a.add_argument("--xmx-gb", type=int, default=6)
    a.add_argument("--priority", type=int, default=5)
    a.add_argument("--solo", type=int, default=0,
                   help="timing pass: start only on hosts with nothing else of ours running, at most N per host")
    sub.add_parser("run")
    sub.add_parser("status")
    r = sub.add_parser("retry-failed")
    r.add_argument("--suffix", required=True)
    args = ap.parse_args()
    if args.cmd == "add":
        return add(args.suffix, args.xmx_gb, args.priority, args.solo)
    if args.cmd == "run":
        return run()
    if args.cmd == "retry-failed":
        return retry_failed(args.suffix)
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
