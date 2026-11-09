"""Plan the profiling batch of the 168 new Valentine scenarios on <cpu-host-00> and <cpu-host-01>.

Writes, under output/cluster-start-2026-10-30/profiling-plan/:
    jobs-<cpu-host-00>.json, jobs-<cpu-host-01>.json   the new scenarios per host ("data/<dataset>/<scenario>"), balanced by bytes
    files-<host>.txt                       the scenario folders to upload (relative to the import tree)
    validation.json                        one existing scenario, profiled on <cpu-host-00> first and compared with its
                                           local stage-1 profiles (same native tools)

    python plan_profiling.py <workspace>
"""
import json
import sys
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
work = ws / "output/cluster-start-2026-10-30"
tree = ws / "harness/generated/benchmark-official-20261030b"
stage1 = ws / "harness/generated_hyfdfix/valentine_plain"
plan = work / "profiling-plan"
plan.mkdir(exist_ok=True)
cohort = json.loads((work / "cohort-549.json").read_text())


def size(path: str) -> int:
    return sum((tree / path / s / f"{s}.csv").stat().st_size for s in ("source", "target"))


new = sorted(cohort["new_cases"], key=lambda r: -size(r["path"]))
hosts = {"<cpu-host-00>": [], "<cpu-host-01>": []}
load = {h: 0 for h in hosts}
for r in new:   # largest first onto the lighter host
    h = min(load, key=load.get)
    hosts[h].append(r["path"])
    load[h] += size(r["path"])
for h, paths in hosts.items():
    (plan / f"jobs-{h}.json").write_text(json.dumps(["data/" + p for p in sorted(paths)], indent=1), encoding="utf-8")
    (plan / f"files-{h}.txt").write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8", newline="\n")
    print(h, len(paths), "scenarios", round(load[h] / 1e6), "MB")

# validation: an existing scenario of a few MB whose stage-1 profiles are complete for fd, ucc and ind
candidates = []
for r in cohort["prior_cases"]:
    prof = stage1 / r["path"] / "metadata/profiling.json"
    if not prof.is_file():
        continue
    state = json.loads(prof.read_text(encoding="utf-8"))
    if all(state.get(s, {}).get(k, {}).get("status") == "complete" for s in ("source", "target") for k in ("fd", "ucc", "ind")):
        manifest = json.loads((tree / r["path"] / "manifest.json").read_text())
        old_manifest = json.loads((stage1 / r["path"] / "manifest.json").read_text())
        if all(manifest["tables"][s]["sha256"] == old_manifest["tables"][s]["sha256"] for s in ("source", "target")):
            candidates.append((size(r["path"]), r["path"], state))
candidates.sort()
target = 2_000_000
sz, path, state = min(candidates, key=lambda c: abs(c[0] - target))
(plan / "validation.json").write_text(json.dumps({"path": path, "bytes": sz, "stage1_profiling": state}, indent=1),
                                      encoding="utf-8")
print("validation scenario:", path, round(sz / 1e6, 2), "MB;", len(candidates), "candidates")
