"""Merge the profiles of the 168 new scenarios (fetched from <cpu-host-00>/<cpu-host-01> into final-<host>/) into the local tree.

- profile-cache/profiles/<key>/ -> harness/profile-cache/profiles/<key>/ (new keys; an existing key must be identical)
- data/<dataset>/<scenario>/metadata/ -> harness/generated/benchmark-official-20261030b/<dataset>/<scenario>/metadata/,
  with the server's profile paths in profiling.json pointing at the local cache
- tool identities and per-case reports -> profiling-run/<host>/

    python integrate_profiles.py <workspace>
"""
import collections
import filecmp
import json
import shutil
import sys
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
work = ws / "output/cluster-start-2026-10-30"
tree = ws / "harness/generated/benchmark-official-20261030b"
cache = ws / "harness/profile-cache/profiles"
REMOTE = "$SMX_REMOTE_ROOT/smx/profile-549/profile-cache/profiles/"
counts = collections.Counter()

def same_profile(a: Path, b: Path) -> bool:
    """Same cache key from two hosts (a table shared by scenarios on both): same spec and records; run times differ."""
    pa, pb = (json.loads((p / "profile.json").read_text(encoding="utf-8")) for p in (a, b))
    return all(pa.get(k) == pb.get(k) for k in ("status", "spec", "key", "records", "relatedness"))


for host in ("<cpu-host-00>", "<cpu-host-01>"):
    src = work / f"final-{host}"
    referenced = set()
    for job in json.loads((work / "profiling-plan" / f"jobs-{host}.json").read_text()):
        state = json.loads((src / job / "metadata/profiling.json").read_text())
        referenced |= {Path(v["path"]).parent.name for side in state.values() for v in side.values() if "path" in v}
    for key in sorted(referenced):
        prof, dst = src / "profile-cache/profiles" / key, cache / key
        if dst.exists():
            if not same_profile(prof, dst):
                sys.exit(f"cache key {key} exists locally with different content")
            counts["profile already present"] += 1
        else:
            shutil.copytree(prof, dst)
            counts["profile added"] += 1
    shutil.copytree(src / "outputs", work / "profiling-run" / host, dirs_exist_ok=True)
    for job in json.loads((work / "profiling-plan" / f"jobs-{host}.json").read_text()):
        rel = job.removeprefix("data/")
        meta_src, meta_dst = src / job / "metadata", tree / rel / "metadata"
        state = json.loads((meta_src / "profiling.json").read_text())
        for side in ("source", "target"):
            for kind in ("fd", "ucc", "ind", "dc"):
                e = state[side][kind]
                counts[f"{kind}:{e['status']}"] += 1
                if "path" in e:
                    assert e["path"].startswith(REMOTE), e["path"]
                    e["path"] = str(cache / e["path"][len(REMOTE):])
                    assert Path(e["path"]).is_file(), e["path"]
        shutil.copytree(meta_src, meta_dst, dirs_exist_ok=True)
        (meta_dst / "profiling.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        marker = (meta_dst / "profile-status.properties").read_text()
        assert "not_profiled" not in marker and "materializing" not in marker, rel
        counts["scenarios"] += 1
print(json.dumps(dict(sorted(counts.items())), indent=1))
