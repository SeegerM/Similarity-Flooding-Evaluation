"""Families 7 and 13 on the full panel (2026-10-03 ~23:05): class nodes on all 382 scenarios and every
layer-1 initial matcher. COMPUTED ONLY: no scorer is written and no output is read here, so that hypotheses can be
registered before scoring.

Arms per initial matcher, on layer 1's configuration `<m>__<base>` (base: melnik, and fd where requested):
    <base>_shape, <base>_shape_shuffled       pattern-class nodes (layer3b/f7.column_classes), and permuted within
                                              each table (seeded by dataset/scenario/side/table/family)
    <base>_semtype, <base>_semtype_shuffled   rule-based semantic types (layer3b/f13.column_types), and permuted
Gate per regime: levenshtein__melnik_devnone (dev build, no extra edges) must equal layer 1's levenshtein__melnik.
dev build runs as the classpath overlay $SMX_REMOTE_ROOT/smx/schematch/overlay_er10 (the same snapshot as ER10).

    python servers/make_classfull_configs.py diag  --regime plain|md5     # class statistics only
    python servers/make_classfull_configs.py write --regime plain|md5 [--bases melnik,fd]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import random
import shutil
import sys
from pathlib import Path
from statistics import mean

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "layer3b"))
sys.path.insert(0, str(STUDY))
from score_seeds import CORPUS, cases_for  # noqa: E402

OVERLAY = "overlay_er10"


def families():
    f7, f13 = importlib.import_module("f7"), importlib.import_module("f13")
    return {"shape": ("pattern", f7.column_classes), "semtype": ("semtype", f13.column_types)}


def classes(regime: str, limit: int = 0):
    """Yield (dataset, scenario, side, {variant: lines}, diagnostics); limit > 0: every k-th case only (diag)."""
    fams = families()
    root = CORPUS[regime]
    cases = cases_for(regime)
    if limit:
        cases = cases[::max(1, len(cases) // limit)]
    for ds, sc, _ in cases:
        for side in ("source", "target"):
            lines = {v: [] for f in fams for v in (f, f + "_shuffled")}
            diag = {f: [] for f in fams}
            for table in sorted((root / ds / sc / side).glob("*.csv")):
                for fam, (label, fn) in fams.items():
                    header, cls = fn(table)
                    diag[fam] += cls
                    rng = random.Random(hashlib.sha256(f"{ds}/{sc}/{side}/{table.stem}/{fam}".encode()).hexdigest())
                    shuffled = cls[:]
                    rng.shuffle(shuffled)
                    for v, cl in ((fam, cls), (fam + "_shuffled", shuffled)):
                        lines[v] += [f"{label}\t{table.stem}\t{h}\t@class:{c}\t1.0\n" for h, c in zip(header, cl)]
            yield ds, sc, side, lines, diag


def diag(regime: str) -> None:
    stats = {"shape": [], "semtype": []}
    for ds, sc, side, _, d in classes(regime, limit=40):
        for fam, cls in d.items():
            mixed = sum(c in ("MIXED", "mixed_") for c in cls)
            stats[fam].append((len(cls), len(set(cls)), mixed))
    for fam, s in stats.items():
        print(f"{regime} {fam}: {len(s)} tables; mean columns {mean(x[0] for x in s):.1f}, mean distinct classes "
              f"{mean(x[1] for x in s):.2f}, mean MIXED columns {mean(x[2] for x in s):.1f}, tables with one class "
              f"{sum(x[1] == 1 for x in s)}")


def write(regime: str, bases: list[str]) -> None:
    edges = STUDY / "data" / f"classfull_{regime}"
    if edges.exists():
        shutil.rmtree(edges)
    stats = {"shape": [], "semtype": []}
    for ds, sc, side, lines, d in classes(regime):
        for v, ls in lines.items():
            p = edges / v / ds / sc / f"{side}.tsv"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(ls), encoding="utf-8", newline="\n")
        for fam, cls in d.items():
            stats[fam].append((len(cls), len(set(cls)), sum(c in ("MIXED", "mixed_") for c in cls)))
    with (edges / "class_stats.txt").open("w", encoding="utf-8") as fh:
        for fam, s in stats.items():
            line = (f"{regime} {fam}: {len(s)} tables; mean columns {mean(x[0] for x in s):.1f}, mean distinct classes "
                    f"{mean(x[1] for x in s):.2f}, mean MIXED columns {mean(x[2] for x in s):.1f}, tables with one "
                    f"class {sum(x[1] == 1 for x in s)}")
            fh.write(line + "\n")
            print(line)
    src_root = STUDY / "tuning" / "configs" / f"layer1_{regime}"
    suffix = f"classfull_{regime}"
    dst_root = STUDY / "tuning" / "configs" / suffix
    res_root = STUDY / "tuning" / "results" / suffix
    if dst_root.exists():
        shutil.rmtree(dst_root)

    def clone(base_label: str, label: str, props: dict[str, str]) -> None:
        src, dst = src_root / base_label, dst_root / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        old = (STUDY / "tuning" / "results" / f"layer1_{regime}" / base_label).as_posix()
        assert g.count(f'resultsDir: "{old}"') == 1, base_label
        (dst / "general.yaml").write_text(g.replace(old, (res_root / label).as_posix()), encoding="utf-8")
        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
        (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
        m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        m.update({"stage": "classfull", "arm": label.split("__", 1)[1], "jvm": {**m["jvm"], **props},
                  "note": "families 7/13 on the full panel; computed only, scored after registration or as exploratory"})
        (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")

    seeds = sorted({p.name.split("__")[0] for p in src_root.iterdir() if p.name.endswith("__melnik")})
    n = 0
    for s in seeds:
        for base in bases:
            if not (src_root / f"{s}__{base}").is_dir():
                continue
            for v in ("shape", "shape_shuffled", "semtype", "semtype_shuffled"):
                clone(f"{s}__{base}", f"{s}__{base}_{v}",
                      {"smx.sf.extraEdgesDir": (edges / v).as_posix(), "smx.cp.overlay": OVERLAY})
                n += 1
    clone("levenshtein__melnik", "levenshtein__melnik_devnone", {"smx.cp.overlay": OVERLAY})
    print(f"{suffix}: {n + 1} configurations ({len(seeds)} matchers, bases {bases}) -> {dst_root}; edges -> {edges}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("diag", "write"))
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--bases", default="melnik")
    a = ap.parse_args()
    if a.cmd == "diag":
        diag(a.regime)
    else:
        write(a.regime, [b for b in a.bases.split(",") if b])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
