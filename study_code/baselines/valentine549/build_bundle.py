"""Build the frozen profiling bundle for <cpu-host-00> / <cpu-host-01> (profile_worker.py's ROOT).

Layout of <bundle>:
    profile_worker.py
    python/smx/{__init__,datasets,profiling,lima}.py   the harness's profiling code (empty package __init__:
                                                       the rest of smx needs numpy and is not used here)
    tools/<kind>/<n>_<name>                            every classpath entry of the local FD/UCC/IND tools,
                                                       including the compiled bridge (with the fixed HyFD Validator)
    runtime.json                                       tool specs and the SHA-256 of every file above

    python build_bundle.py <workspace> <bundle>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

workspace, bundle = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work = workspace / "output/cluster-start-2026-10-30"
if bundle.exists():
    sys.exit(f"{bundle} exists; choose a new folder")
local = json.loads((work / "local-profiler-tools.json").read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


(bundle / "python/smx").mkdir(parents=True)
shutil.copy2(work / "profile_worker.py", bundle / "profile_worker.py")
for name in ("datasets.py", "profiling.py", "lima.py"):
    shutil.copy2(workspace / "harness/smx" / name, bundle / "python/smx" / name)
(bundle / "python/smx/__init__.py").write_text('"""Profiling subset of the smx harness."""\n', encoding="utf-8")

tools = {}
for kind in ("fd", "ucc", "ind"):
    spec = local[kind]
    entries = []
    for i, entry in enumerate(spec["classpath"].split(";")):
        src = Path(entry)
        dst = bundle / "tools" / kind / f"{i:02d}_{src.name}"
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            sys.exit(f"missing classpath entry for {kind}: {src}")
        entries.append(dst.relative_to(bundle).as_posix())
    files = {p.relative_to(bundle).as_posix(): sha(p) for p in sorted((bundle / "tools" / kind).rglob("*")) if p.is_file()}
    tools[kind] = {"native_fingerprint": spec["fingerprint"], "source_fingerprint": stable_hash(files),
                   "bridge": spec["bridge"], "classpath": entries}

files_sha256 = {p.relative_to(bundle).as_posix(): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
runtime = {"built_from": str(workspace), "tools": tools, "files_sha256": files_sha256,
           "settings": {"fd": {"threshold": 0.95, "max_arity": 2}, "ucc": {"threshold": 0.95, "max_arity": 2},
                        "ind": {"threshold": 0.95, "max_arity": 1, "support": "distinct"},
                        "heap": "8g", "timeout_seconds": 7200}}
(bundle / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
print(json.dumps({"files": len(files_sha256), "bytes": sum((bundle / f).stat().st_size for f in files_sha256),
                  "tools": {k: len(v["classpath"]) for k, v in tools.items()}}, indent=2))
