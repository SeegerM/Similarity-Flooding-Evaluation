"""Amendment 11 (registered replication of the dependency-role term), §4: two extra SF arms per corpus, built as X33
(gvfull) built them, from the corpus's own E2 configurations, plus one melnik twin.

    ucc_top1       clone of e2_<corpus>/<m>__ucc  + dependencyTopK "1"                         (shared build)
    ucc1_fd_topk   clone of e2_<corpus>/<m>__fd   + uccFilterThreshold "0.0", constraintWeighting "TOPK_HYBRID";
                   smx.sf.topK=1, smx.sf.uccTopK=1, smx.sf.uccUnweighted=true, smx.cp.overlay=overlay_gvb
    levenshtein__melnik_twin   clone of e2_<corpus>/levenshtein__melnik, unchanged (checked cell for cell)
Only these settings and resultsDir change (asserted: each replaced YAML line occurs exactly once).
Output: tuning/configs/e2x_<corpus>/, results under tuning/results/e2x_<corpus>/.

    python servers/make_e2x_configs.py ctl|t2d
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG = STUDY / "tuning" / "configs"
ARMS = {
    "ucc_top1": ("ucc", {"dependencyTopK": "1"}, {}),
    "ucc1_fd_topk": ("fd", {"uccFilterThreshold": "0.0", "constraintWeighting": "TOPK_HYBRID"},
                     {"smx.sf.topK": "1", "smx.sf.uccTopK": "1", "smx.sf.uccUnweighted": "true",
                      "smx.cp.overlay": "overlay_gvb"}),
}


def clone(corpus: str, base: str, label: str, yaml: dict, props: dict) -> None:
    suffix = f"e2x_{corpus}"
    src, dst = CFG / f"e2_{corpus}" / base, CFG / suffix / label
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = (STUDY / "tuning" / "results" / f"e2_{corpus}" / base).as_posix()
    assert g.count(f'resultsDir: "{old}"') == 1, base
    (dst / "general.yaml").write_text(g.replace(old, (STUDY / "tuning" / "results" / suffix / label).as_posix()),
                                      encoding="utf-8")
    s = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    for k, v in yaml.items():
        lines = [l for l in s.splitlines() if l.startswith(f"  {k}: [")]
        assert len(lines) == 1, (label, k)
        s = s.replace(lines[0], f'  {k}: ["{v}"]')
    (dst / "second_line_matchers.yaml").write_text(s, encoding="utf-8")
    if props:
        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
        (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
    m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    m.update({"stage": "e2x", "arm": label.split("__", 1)[1], "yaml_changes": yaml,
              "note": "Amendment 11 §4 (registered replication of the role term)"})
    (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


def main() -> int:
    corpus = sys.argv[1]
    root = CFG / f"e2x_{corpus}"
    if root.exists():
        shutil.rmtree(root)
    seeds = sorted({p.name.split("__")[0] for p in (CFG / f"e2_{corpus}").iterdir() if p.is_dir()})
    n = 0
    for m in seeds:
        for arm, (base_arm, yaml, props) in ARMS.items():
            clone(corpus, f"{m}__{base_arm}", f"{m}__{arm}", yaml, props)
            n += 1
    clone(corpus, "levenshtein__melnik", "levenshtein__melnik_twin", {}, {})
    print(f"e2x_{corpus}: {n + 1} configurations ({len(seeds)} matchers x {len(ARMS)} arms + twin)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
