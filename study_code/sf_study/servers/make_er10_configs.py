"""EXPLORATORY (external review ER10 2026-10-03 ~21:45): SF as published with the class information
of Amendment 5's additive baselines, on the independent corpus, so that A_p / A_s can be compared with SF given
the same information.

Arms (all on layer 1's `melnik` configuration of each of the 74 frozen matchers, reference/additive_indep_matchers.json):
    shape, shape_shuffled         pattern-class nodes (layer3b/f7.column_classes), and the classes permuted within
                                  each table (control)
    semtype, semtype_shuffled     rule-based semantic-type nodes (layer3b/f13.column_types), and the control
plus the gate `levenshtein__melnik__devnone`: dev build with no extra edges, which must equal layer1_indep's
`levenshtein__melnik` cell for cell.

The class nodes need dev build (layer3b/classes_dev: SchemaGraphBuilder with the extra-edges channel). It is
shipped to the hosts as a classpath overlay $SMX_REMOTE_ROOT/smx/schematch/overlay_er10 and selected per task by the JVM property
smx.cp.overlay (servers/remote_runner.py); the shared classes are not touched.

    python servers/make_er10_configs.py            # edge files under data/er10_indep/, configs under tuning/configs/er10_indep/
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "layer3b"))
CORPUS = STUDY / "data" / "indep_plain"
EDGES = STUDY / "data" / "er10_indep"
SRC = STUDY / "tuning" / "configs" / "layer1_indep"
DST = STUDY / "tuning" / "configs" / "er10_indep"
RES = STUDY / "tuning" / "results" / "er10_indep"
OVERLAY = "overlay_er10"


def edges() -> dict[str, int]:
    f7 = importlib.import_module("f7")
    f13 = importlib.import_module("f13")
    fams = {"shape": ("pattern", f7.column_classes), "semtype": ("semtype", f13.column_types)}
    if EDGES.exists():
        shutil.rmtree(EDGES)
    n = {}
    for ds in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        for sc in sorted(p for p in ds.iterdir() if p.is_dir()):
            for side in ("source", "target"):
                lines = {v: [] for f in fams for v in (f, f + "_shuffled")}
                for table in sorted((sc / side).glob("*.csv")):
                    for fam, (label, fn) in fams.items():
                        header, cls = fn(table)
                        rng = random.Random(hashlib.sha256(f"{ds.name}/{sc.name}/{side}/{table.stem}/{fam}".encode()).hexdigest())
                        shuffled = cls[:]
                        rng.shuffle(shuffled)
                        for v, cl in ((fam, cls), (fam + "_shuffled", shuffled)):
                            lines[v] += [f"{label}\t{table.stem}\t{h}\t@class:{c}\t1.0\n" for h, c in zip(header, cl)]
                for v, ls in lines.items():
                    p = EDGES / v / ds.name / sc.name / f"{side}.tsv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("".join(ls), encoding="utf-8", newline="\n")
                    n[v] = n.get(v, 0) + 1
    return n


def clone(base: str, label: str, props: dict[str, str]) -> None:
    src, dst = SRC / base, DST / label
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = (STUDY / "tuning" / "results" / "layer1_indep" / base).as_posix()
    assert g.count(f'resultsDir: "{old}"') == 1, base
    (dst / "general.yaml").write_text(g.replace(old, (RES / label).as_posix()), encoding="utf-8")
    jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
    (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
    m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    m.update({"stage": "er10", "arm": label.split("__", 1)[1], "jvm": {**m["jvm"], **props},
              "note": "EXPLORATORY (ER10): SF(melnik) with class nodes, independent corpus"})
    (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


def main() -> int:
    n = edges()
    print(f"edge files: {n} -> {EDGES}")
    if DST.exists():
        shutil.rmtree(DST)
    matchers = json.loads((STUDY / "reference" / "additive_indep_matchers.json").read_text(encoding="utf-8"))
    k = 0
    for m in matchers:
        for v in ("shape", "shape_shuffled", "semtype", "semtype_shuffled"):
            clone(f"{m}__melnik", f"{m}__melnik_{v}",
                  {"smx.sf.extraEdgesDir": (EDGES / v).as_posix(), "smx.cp.overlay": OVERLAY})
            k += 1
    clone("levenshtein__melnik", "levenshtein__melnik_devnone", {"smx.cp.overlay": OVERLAY})
    print(f"{k + 1} configurations -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
