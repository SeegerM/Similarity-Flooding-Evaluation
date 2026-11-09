"""X82 (pick (c) / preregistration, EXPLORATORY): class nodes on the new corpora.
servers/make_er10_configs.edges() is run UNCHANGED per corpus (its CORPUS and EDGES globals pointed at the corpus);
the clones follow make_er10_configs.clone() with the source being the corpus's own melnik configurations (final_<c> or
x81_<c>; er10's clone hard-codes layer1_indep's resultsDir, so its four lines are repeated here with the stage):
    <m>__melnik_{shape,shape_shuffled,semtype,semtype_shuffled}   + smx.sf.extraEdgesDir=data/x82_classes_<c>/<v>,
                                                                   smx.cp.overlay=overlay_er10
    levenshtein__melnik_devnone                                    + overlay only: gate, must equal <c>'s levenshtein__melnik
Edges data/x82_classes_<c>/ (shipped to the hosts separately); configs tuning/configs/classes_<c>.

    python servers/make_x82_classes.py final_indep|final_controlled|hdxsm|gdcsm|madi|goby|mimic
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "servers"))
import make_er10_configs as er10  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
ROOTS = {"final_indep": STUDY / "data" / "final_indep", "final_controlled": GEN / "final-controlled",
         "hdxsm": GEN / "hdxsm-usable", "gdcsm": GEN / "gdcsm-imported", "madi": GEN / "madi-imported",
         "goby": GEN / "goby-imported", "mimic": STUDY / "data" / "mimic_omop"}
CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"


def main(c: str) -> int:
    stage = c if c.startswith("final_") else f"x81_{c}"
    er10.CORPUS, er10.EDGES = ROOTS[c], STUDY / "data" / f"x82_classes_{c}"
    n = er10.edges()
    src_root, suffix = CFG / stage, f"classes_{c}"
    dst_root = CFG / suffix
    if dst_root.exists():
        shutil.rmtree(dst_root)

    def clone(base: str, label: str, props: dict[str, str]) -> None:
        src, dst = src_root / base, dst_root / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        old = (RES / stage / base).as_posix()
        assert g.count(f'resultsDir: "{old}"') == 1, base
        (dst / "general.yaml").write_text(g.replace(old, (RES / suffix / label).as_posix()), encoding="utf-8")
        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
        assert "overlay" not in jvm and "extraEdges" not in jvm, base
        if jvm and not jvm.endswith("\n"):
            jvm += "\n"
        (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
        m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        m.update({"stage": "classes", "arm": label.split("__", 1)[1], "jvm": {**m.get("jvm", {}), **props},
                  "source": f"{stage}/{base}", "note": "X82 EXPLORATORY: class nodes (er10 arms) on the new corpora"})
        (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")

    matchers = sorted(p.name[: -len("__melnik")] for p in src_root.iterdir() if p.name.endswith("__melnik"))
    for m in matchers:
        for v in ("shape", "shape_shuffled", "semtype", "semtype_shuffled"):
            clone(f"{m}__melnik", f"{m}__melnik_{v}", {"smx.sf.extraEdgesDir": (er10.EDGES / v).as_posix(),
                                                         "smx.cp.overlay": er10.OVERLAY})
    clone("levenshtein__melnik", "levenshtein__melnik_devnone", {"smx.cp.overlay": er10.OVERLAY})
    print(f"{suffix}: edge files {n}; {len(matchers) * 4 + 1} configurations ({len(matchers)} matchers x 4 + gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
