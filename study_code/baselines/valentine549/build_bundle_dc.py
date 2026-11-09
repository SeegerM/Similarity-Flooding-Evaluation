"""Build the DC additions to the profiling bundle: tools/dc (LimaProfile bridge + LIMA jar), profile_worker_dc.py,
runtime-dc.json. They are copied into the same folder as the FD/UCC/IND bundle on each host.

    python build_bundle_dc.py <workspace> <new folder>
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ws, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
work = ws / "output/cluster-start-2026-10-30"
if out.exists():
    sys.exit(f"{out} exists")
BRIDGE = "86a440ff8e5b502e82d201d72184434f4772d87e1921a95a5150cab85de17c1c"   # the existing scenarios' DC tool
bridge_dir = ws / "harness/profile-cache/bridges" / BRIDGE
jar = ws / "code/LIMA/target/LIMA-0.1.jar"
identity = json.loads((bridge_dir / "identity.json").read_text())
assert identity["jar_sha256"] == hashlib.sha256(jar.read_bytes()).hexdigest(), "LIMA jar changed since the existing profiles"

(out / "tools/dc").mkdir(parents=True)
shutil.copytree(bridge_dir, out / "tools/dc" / f"00_{BRIDGE}")
shutil.copy2(jar, out / "tools/dc/01_LIMA-0.1.jar")
shutil.copy2(work / "profile_worker_dc.py", out / "profile_worker_dc.py")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


tool_files = {p.relative_to(out).as_posix(): sha(p) for p in sorted((out / "tools/dc").rglob("*")) if p.is_file()}
runtime = {"tools": {"dc": {"native_fingerprint": BRIDGE,
                            "source_fingerprint": hashlib.sha256(json.dumps(tool_files, sort_keys=True).encode()).hexdigest(),
                            "bridge": "LimaProfile",
                            "classpath": [f"tools/dc/00_{BRIDGE}", "tools/dc/01_LIMA-0.1.jar"]}},
           "files_sha256": {**tool_files, "profile_worker_dc.py": sha(out / "profile_worker_dc.py")},
           "settings": {"dc": {"approx": 0.01, "row_cap": 1000, "seed": 20260903, "max_columns": 64,
                               "heap": "4g", "timeout_seconds": 300}}}
(out / "runtime-dc.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
print(json.dumps({"files": len(runtime["files_sha256"]), "java": identity["java"].splitlines()[0]}))
