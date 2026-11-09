"""X82 (pick 2, EXPLORATORY): the parallel-edge repair on the fd arm, on the new corpora.
The same clones as layer3b/make_parallel_configs.py (its ARMS and clone rules), with two differences: the source is the
corpus's own SF configurations (final_<c> or x81_<c>, not layer1_<regime>), and only the `split` arm is built (C:
"fd with parallelEdges SPLIT"). Per corpus:
    <m>__fd_split                 <c>'s <m>__fd + smx.sf.parallelEdges=SPLIT
    levenshtein__fd_droptwin      <c>'s levenshtein__fd + smx.sf.parallelEdgeRule=DROP (the default): gate, must equal it
    levenshtein__melnik_split, pmf__melnik_split   no FD edges: expected about equal to melnik (report only)
Output tuning/configs/parallel_<c>, runs tuning/results/parallel_<c>. Scored with layer3b/score_parallel.py (C).

    python servers/make_x82_parallel.py final_indep|final_controlled|hdxsm|gdcsm|madi|goby|mimic
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "layer3b"))
from make_parallel_configs import ARMS  # noqa: E402

CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"


def main(c: str) -> int:
    stage = c if c.startswith("final_") else f"x81_{c}"
    src_root, suffix = CFG / stage, f"parallel_{c}"
    dst_root, res_root = CFG / suffix, RES / suffix
    if dst_root.exists():
        shutil.rmtree(dst_root)

    def clone(base_label: str, label: str, props: dict[str, str]) -> None:
        src, dst = src_root / base_label, dst_root / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        old = (RES / stage / base_label).as_posix()
        assert g.count(f'resultsDir: "{old}"') == 1, base_label
        (dst / "general.yaml").write_text(g.replace(old, (res_root / label).as_posix()), encoding="utf-8")
        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
        assert "parallelEdge" not in jvm and "overlay" not in jvm, base_label
        if jvm and not jvm.endswith("\n"):
            jvm += "\n"
        (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
        m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        m.update({"stage": "parallel", "arm": label.split("__", 1)[1], "jvm": {**m.get("jvm", {}), **props},
                  "source": f"{stage}/{base_label}", "note": "X82 EXPLORATORY: parallel-edge rule for FD edges"})
        (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")

    seeds = sorted(p.name[: -len("__fd")] for p in src_root.iterdir() if p.name.endswith("__fd"))
    for s in seeds:
        clone(f"{s}__fd", f"{s}__fd_split", ARMS["split"])
    clone("levenshtein__fd", "levenshtein__fd_droptwin", {"smx.sf.parallelEdgeRule": "DROP"})
    for s in ("levenshtein", "pmf"):
        clone(f"{s}__melnik", f"{s}__melnik_split", ARMS["split"])
    print(f"{suffix}: {len(list(dst_root.iterdir()))} configurations ({len(seeds)} matchers split + gate + 2 checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
