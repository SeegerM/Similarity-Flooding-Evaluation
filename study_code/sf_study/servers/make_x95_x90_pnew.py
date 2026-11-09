"""X95 and X90 SF arms on the 168 new Valentine scenarios, readable (2026-11-02; the 382 stages unchanged).

x95sf_valentine_pnew: ii__melnik, ii__melnik_t2, iii__melnik, iii__melnik_t2, built by make_x95_configs.melnik_arm
    exactly as for valentine_plain, from layer1_pnew/levenshtein__melnik and the bases data/x95_bases/<ii|iii>/valentine_pnew.
x90_valentine_pnew: every x90_valentine configuration (4 initial matchers x lit_none, lit_none_t1/2/3/5/10), cloned with
    only initialMatrixDir (reference/seeds/plain/<m> -> pnew/<m>), the corpus (plain -> pnew), resultsDir and datasets.yaml
    (the 7 datasets of the 168) changed; every other file must stay byte-identical.

    python servers/make_x95_x90_pnew.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import make_x95_configs as x95  # noqa: E402
from score_seeds import cases_for  # noqa: E402

CFG, RES, STUDY = x95.CFG, x95.RES, x95.STUDY

# X95
x95.STAGE["valentine_pnew"] = "layer1_pnew"
root = CFG / "x95sf_valentine_pnew"
if root.exists():
    sys.exit(f"{root} exists")
for base in ("ii", "iii"):
    for t2 in (False, True):
        x95.melnik_arm("valentine_pnew", base, t2)
print("x95sf_valentine_pnew:", sorted(p.name for p in root.iterdir()))

# X90
src_root, dst_root = CFG / "x90_valentine", CFG / "x90_valentine_pnew"
if dst_root.exists():
    sys.exit(f"{dst_root} exists")
by_ds: dict[str, int] = {}
for d, s, _ in cases_for("pnew"):
    by_ds[d] = by_ds.get(d, 0) + 1
for src in sorted(p for p in src_root.iterdir() if p.is_dir()):
    seed = src.name.split("__", 1)[0]
    dst = dst_root / src.name
    shutil.copytree(src, dst)
    old_seed = (STUDY / "reference" / "seeds" / "plain" / seed).as_posix()
    new_seed = (STUDY / "reference" / "seeds" / "pnew" / seed).as_posix()
    assert Path(new_seed).is_dir(), new_seed
    sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    assert sl.count(f'["{old_seed}"]') == 1, src.name
    (dst / "second_line_matchers.yaml").write_text(sl.replace(f'["{old_seed}"]', f'["{new_seed}"]'), encoding="utf-8")
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    g, n1 = re.subn(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RES / dst_root.name / src.name).as_posix()}"', g)
    g, n2 = re.subn(r'(?m)^(defaultDatasetBasePath: ".*/smx/corpus/)plain"$', r'\g<1>pnew"', g)
    assert (n1, n2) == (1, 1), src.name
    (dst / "general.yaml").write_text(g, encoding="utf-8")
    (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in sorted(by_ds)), encoding="utf-8")
    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    meta.update({"regime": "pnew", "seed_dir": new_seed, "datasets": sorted(by_ds), "dataset_sizes": by_ds,
                 "cloned_from": f"x90_valentine/{src.name}"})
    meta.setdefault("sf", {})["initialMatrixDir"] = new_seed
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for f in src.iterdir():
        if f.name not in ("general.yaml", "second_line_matchers.yaml", "meta.json", "datasets.yaml"):
            assert (dst / f.name).read_bytes() == f.read_bytes(), (src.name, f.name)
print(f"x90_valentine_pnew: {len(list(dst_root.iterdir()))} configurations, datasets {sorted(by_ds)}")
