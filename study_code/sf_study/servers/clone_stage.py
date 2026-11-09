"""Clone every configuration of a base stage into a new stage, changing only the named fields (as make_gvfull_configs.py
and make_iter_configs.py do for layer 1/3 of plain and md5, but for any base stage, e.g. layer1_pnew or l1fd_p382fix).

Each copy differs from its base in resultsDir, the given second_line_matchers.yaml fields (each must occur exactly
once) and the given JVM properties (an existing key is replaced, a new one appended); meta.json records the changes.

    python servers/clone_stage.py --base layer1_pnew --base-arm melnik --suffix gvfull_pnew --arm ucc_top1 \
        --yaml uccFilterThreshold=0.0 dependencyTopK=1
    python servers/clone_stage.py --base layer1_pnew --base-arm melnik --suffix iter_pnew --arm melnik_t2 \
        --props smx.sf.maxIterations=1 --append
    (--seeds a,b and --datasets d1,d2 restrict the clone, e.g. for a twin)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"
EXCLUDED = {"valentine_distribution"}   # 2026-10-30, the authors' decision: no longer run

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--base-arm", required=True)
ap.add_argument("--suffix", required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--yaml", nargs="*", default=[])
ap.add_argument("--props", nargs="*", default=[])
ap.add_argument("--seeds", default="")
ap.add_argument("--datasets", default="")
ap.add_argument("--append", action="store_true", help="add to an existing stage (several arms in one suffix)")
ap.add_argument("--corpus", default="", help="FROM=TO: read the hosts' corpus $SMX_REMOTE_ROOT/smx/corpus/TO instead of FROM")
ap.add_argument("--keep-arm", action="store_true", help="label the clone with the base arm (--arm is then ignored)")
a = ap.parse_args()
yaml = dict(x.split("=", 1) for x in a.yaml)
props = dict(x.split("=", 1) for x in a.props)
dst_root = CFG / a.suffix
if dst_root.exists() and not a.append:
    sys.exit(f"{dst_root} exists (use --append to add an arm)")
seeds = set(a.seeds.split(",")) if a.seeds else None
datasets = a.datasets.split(",") if a.datasets else None
written = 0
for base in sorted((CFG / a.base).glob(f"*__{a.base_arm}")):
    seed = base.name[: -len(a.base_arm) - 2]
    if seed in EXCLUDED or (seeds and seed not in seeds):
        continue
    label = f"{seed}__{a.base_arm if a.keep_arm else a.arm}"
    dst = dst_root / label
    if dst.exists():
        sys.exit(f"{dst} exists")
    shutil.copytree(base, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = f'resultsDir: "{(RES / a.base / base.name).as_posix()}"'
    if g.count(old) != 1:
        sys.exit(f"{base}: expected exactly one {old}")
    g = g.replace(old, f'resultsDir: "{(RES / a.suffix / label).as_posix()}"')
    if a.corpus:
        frm, to = a.corpus.split("=")
        o, n = f'/smx/corpus/{frm}"', f'/smx/corpus/{to}"'
        if g.count(o) != 1:
            sys.exit(f"{base}: corpus {frm} occurs {g.count(o)} times in general.yaml")
        g = g.replace(o, n)
    (dst / "general.yaml").write_text(g, encoding="utf-8")
    s = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    for k, v in yaml.items():
        lines = [l for l in s.splitlines() if l.startswith(f"  {k}: [")]
        if len(lines) != 1:
            sys.exit(f"{base}: {k} occurs {len(lines)} times")
        s = s.replace(lines[0], f'  {k}: ["{v}"]')
    (dst / "second_line_matchers.yaml").write_text(s, encoding="utf-8")
    jvm = dict(l.split("=", 1) for l in (dst / "jvm.properties").read_text(encoding="utf-8").splitlines() if "=" in l)
    jvm.update(props)
    (dst / "jvm.properties").write_text("".join(f"{k}={v}\n" for k, v in jvm.items()), encoding="utf-8")
    m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    if datasets:
        m["datasets"] = [d for d in m["datasets"] if d in datasets]
        m["dataset_sizes"] = {d: n for d, n in m["dataset_sizes"].items() if d in datasets}
        (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in m["datasets"]),
                                           encoding="utf-8")
    m.update({"stage": a.suffix.split("_", 1)[0], "arm": a.arm, "jvm": jvm, "yaml_changes": yaml,
              "cloned_from": f"{a.base}/{base.name}"})
    (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    written += 1
print(f"{a.suffix}: {written} configurations ({a.arm}) cloned from {a.base} ({a.base_arm})")
