"""parallel (EXPLORATORY, 2026-10-05): the parallel-edge rule for FD edges on the full panel.

The default container (COLLAPSE, rule DROP) keeps one propagation edge per ordered pair. Wherever two columns determine
each other, the forward and the reverse `determines` messages compete for one slot and one is silently refused: about
21% of forward FD messages, 58% on TPC-DI (harness/SF_PARALLEL_EDGE_DEFECT.md; 40-scenario pmf pilot only). The paper's
FD results run on that operator, and §III-B claims the rules are compared. The switches are in the shared build:
    smx.sf.parallelEdges=SPLIT          one edge per (source, target, channel): nothing is refused (the repair)
    smx.sf.parallelEdgeRule=AVG | MAX   one slot, the competing coefficients combined
(SUM is left out: it clamps on most contested slots, "a rule nobody would choose".)

Arms per layer-1 initial matcher, cloned from layer 1's `<m>__fd`: fd_split, fd_pavg, fd_pmax. Checks:
    levenshtein__fd_droptwin  (rule DROP set explicitly, the default) must equal layer 1 levenshtein__fd cell for cell;
    levenshtein__melnik_split, pmf__melnik_split  (no FD edges: expected about equal to layer 1 melnik; report only).
No overlay: everything is in the shared classes.

    python layer3b/make_parallel_configs.py --regime plain|md5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
ARMS = {"split": {"smx.sf.parallelEdges": "SPLIT"}, "pavg": {"smx.sf.parallelEdgeRule": "AVG"},
        "pmax": {"smx.sf.parallelEdgeRule": "MAX"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    r = ap.parse_args().regime
    src_root = STUDY / "tuning" / "configs" / f"layer1_{r}"
    suffix = f"parallel_{r}"
    dst_root = STUDY / "tuning" / "configs" / suffix
    res_root = STUDY / "tuning" / "results" / suffix
    if dst_root.exists():
        shutil.rmtree(dst_root)

    def clone(base_label: str, label: str, props: dict[str, str]) -> None:
        src, dst = src_root / base_label, dst_root / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        old = (STUDY / "tuning" / "results" / f"layer1_{r}" / base_label).as_posix()
        assert g.count(f'resultsDir: "{old}"') == 1, base_label
        (dst / "general.yaml").write_text(g.replace(old, (res_root / label).as_posix()), encoding="utf-8")
        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
        assert "parallelEdge" not in jvm and "overlay" not in jvm, base_label
        (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
        m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        m.update({"stage": "parallel", "arm": label.split("__", 1)[1], "jvm": {**m["jvm"], **props},
                  "note": "EXPLORATORY: parallel-edge rule for FD edges"})
        (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")

    seeds = sorted(p.name[: -len("__fd")] for p in src_root.iterdir() if p.name.endswith("__fd"))
    for s in seeds:
        for arm, props in ARMS.items():
            clone(f"{s}__fd", f"{s}__fd_{arm}", props)
    clone("levenshtein__fd", "levenshtein__fd_droptwin", {"smx.sf.parallelEdgeRule": "DROP"})
    for s in ("levenshtein", "pmf"):
        clone(f"{s}__melnik", f"{s}__melnik_split", ARMS["split"])
    n = len(list(dst_root.iterdir()))
    print(f"{suffix}: {n} configurations ({len(seeds)} matchers x {len(ARMS)} arms + gate + 2 melnik checks) -> {dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
