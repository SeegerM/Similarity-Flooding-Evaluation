"""Add LIMA denial-constraint (DC) profiles to scenarios that profile_worker.py already profiled (FD/UCC/IND).

Same frozen-runtime discipline as profile_worker.py (which it imports unchanged): the DC tool comes from
runtime-dc.json, whose files are hash-checked. Settings are those of the existing scenarios' DC profiles:
approx 0.01, the first 1000 rows, seed 20260903, at most 64 columns, 4 GB heap, 300 s timeout.
Each scenario's existing profiling state is loaded and only the DC entries are added, then the scenario's metadata is
materialised again (FD/UCC/IND files are rewritten from the same cached profiles).

    python3 profile_worker_dc.py --pilot <scenario folder>    # one scenario, one worker
    python3 profile_worker_dc.py --workers 12                 # every job in jobs.json
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import socket
import time

import profile_worker as pw   # host check, resource limits, Java path and options, runtime check

from smx.profiling import Profiler, materialize, stable_hash
from smx.datasets import digest, json_write

ROOT = pw.ROOT
HEAP, TIMEOUT = "4g", 300


class DcProfiler(Profiler):
    def __init__(self):
        super().__init__(ROOT / "profile-cache", java=str(pw.JAVA), timeout=TIMEOUT, heap=HEAP)
        self.runtime = json.loads((ROOT / "runtime-dc.json").read_text())
        self.java_identity = json.loads((ROOT / "outputs/java-identity.json").read_text())

    def tool(self, kind):
        if kind != "dc":
            raise ValueError("this worker only runs DC profiles")
        spec = self.runtime["tools"]["dc"]
        identity = {"snapshot_native_fingerprint": spec["native_fingerprint"],
                    "frozen_source_fingerprint": spec["source_fingerprint"],
                    "java": self.java_identity, "java_tool_options": pw.JAVA_OPTIONS}
        path = ROOT / "outputs/tool-identities/dc.json"
        if not path.exists():
            json_write(path, identity)
        return {"fingerprint": stable_hash(identity), "bridge": spec["bridge"],
                "classpath": os.pathsep.join(str(ROOT / p) for p in spec["classpath"]), "identity_path": str(path)}


def check_runtime_dc():
    runtime = json.loads((ROOT / "runtime-dc.json").read_text())
    for name, expected in runtime["files_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen runtime file changed: " + name)


def run_case(relative):
    path = ROOT / relative / "manifest.json"
    state_path = path.parent / "metadata/profiling.json"
    state = json.loads(state_path.read_text())
    profiler, start = DcProfiler(), time.monotonic()
    for side in ("source", "target"):
        try:
            result = profiler.profile(path, side, "dc")
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
        state[side]["dc"] = result
        json_write(state_path, state)
    ok = all(state[s]["dc"]["status"] in {"complete", "not_applicable"} for s in ("source", "target"))
    materialized, error = False, None
    if ok:
        try:
            materialize(path, state)
            materialized = True
        except Exception as exc:
            error = str(exc)
    status = {"scenario": relative, "dc_complete": ok, "materialized": materialized, "error": error,
              "dc": {s: {k: state[s]["dc"].get(k) for k in ("status", "count", "key", "reason", "error")}
                     for s in ("source", "target")},
              "seconds": round(time.monotonic() - start, 2), "host": socket.gethostname().split(".")[0]}
    json_write(ROOT / "outputs/dc-cases" / (hashlib.sha256(relative.encode()).hexdigest()[:16] + ".json"), status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("Worker cap is 12 per allowed host")
    pw.check_runtime()
    check_runtime_dc()
    if args.pilot:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1, initializer=pw.initialise) as pool:
            print(json.dumps(pool.submit(run_case, args.pilot).result(), indent=1), flush=True)
        return
    jobs = json.loads((ROOT / "jobs.json").read_text())
    todo = []
    for job in jobs:
        p = ROOT / "outputs/dc-cases" / (hashlib.sha256(job.encode()).hexdigest()[:16] + ".json")
        if not p.exists() or not json.loads(p.read_text()).get("materialized"):
            todo.append(job)
    state = {"host": socket.gethostname().split(".")[0], "workers": args.workers, "total": len(jobs),
             "pending_at_start": len(todo), "done": len(jobs) - len(todo), "failed": 0,
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "heap": HEAP, "timeout": TIMEOUT}
    json_write(ROOT / "outputs/dc-status.json", state)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=pw.initialise) as pool:
        futures = {pool.submit(run_case, job): job for job in todo}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                ok = result["materialized"]
            except Exception as exc:
                ok, result = False, {"scenario": futures[future], "error": str(exc)}
            state["done"] += 1
            state["failed"] += not ok
            state["last"] = result
            json_write(ROOT / "outputs/dc-status.json", state)
            print(json.dumps({"done": state["done"], "total": state["total"], "failed": state["failed"]}), flush=True)
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json_write(ROOT / "outputs/dc-status.json", state)


if __name__ == "__main__":
    main()
