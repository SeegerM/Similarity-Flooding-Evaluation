"""X82 P12 (EXPLORATORY; arm names fixed 23:00; TOPK as R8 registered it):
E2's remaining arms and R8's primary scope contrast on the new corpora, cloned from the corpus's own configurations
(final_<c> or x81_<c>: corpus path, seeds and datasets already set) with exactly the registered one-line changes:
    e2rest_<c>/<m>__coef_prod_fd   <c>/<m>__fd with propCoeffPolicy INV_AVG -> INV_PROD   (= layer1_plain coef_prod_fd)
    e2rest_<c>/<m>__ucc            <c>/<m>__fd with uccFilterThreshold "0.0", fdFilterThreshold ""   (= layer3_plain ucc)
    e2rest_<c>/<m>__fd_top1        <c>/<m>__fd + smx.sf.pairScope=TOPK, pairScopeParam=1        (= e2 fd_top1, r8 fd_top1)
    e2rest_<c>/levenshtein__fd_twin  <c>/levenshtein__fd unchanged: gate (must equal it)
    r8_<c>/<m>__melnik_top1        <c>/<m>__melnik + smx.sf.pairScope=TOPK, pairScopeParam=1   (= r8 melnik_top1)
For each matcher the layer-1/3 source is checked to differ from layer1_plain <m>__fd in exactly that YAML line.

    python servers/make_x82_p12.py final_indep|final_controlled|hdxsm|gdcsm|madi|goby|mimic
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"
TOPK = {"smx.sf.pairScope": "TOPK", "smx.sf.pairScopeParam": "1"}
YAML = {"coef_prod_fd": ("layer1_plain", "coef_prod_fd",
                         [('  propCoeffPolicy: ["INV_AVG"]', '  propCoeffPolicy: ["INV_PROD"]')]),
        "ucc": ("layer3_plain", "ucc", [('  uccFilterThreshold: [""]', '  uccFilterThreshold: ["0.0"]'),
                                        ('  fdFilterThreshold: ["0.0"]', '  fdFilterThreshold: [""]')])}


def main(c: str) -> int:
    stage = c if c.startswith("final_") else f"x81_{c}"
    out = {s: CFG / f"{s}_{c}" for s in ("e2rest", "r8")}
    for d in out.values():
        if d.exists():
            shutil.rmtree(d)

    def clone(base: str, suffix: str, label: str, edits=(), props=None) -> None:
        src, dst = CFG / stage / base, CFG / f"{suffix}_{c}" / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        old = (RES / stage / base).as_posix()
        assert g.count(f'resultsDir: "{old}"') == 1, base
        (dst / "general.yaml").write_text(g.replace(old, (RES / f"{suffix}_{c}" / label).as_posix()), encoding="utf-8")
        sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
        for a, b in edits:
            assert sl.count(a) == 1, (label, a)
            sl = sl.replace(a, b)
        (dst / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
        if props:
            jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
            assert "pairScope" not in jvm, label
            if jvm and not jvm.endswith("\n"):
                jvm += "\n"
            (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
        m = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        m.update({"stage": suffix, "arm": label.split("__", 1)[1], "source": f"{stage}/{base}",
                  "note": "X82 P12 EXPLORATORY (arm names fixed in SYNC 2026-10-06 23:00)"})
        (dst / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")

    matchers = sorted(p.name[: -len("__fd")] for p in (CFG / stage).iterdir() if p.name.endswith("__fd"))
    for m in matchers:
        ref = (CFG / "layer1_plain" / f"{m}__fd" / "second_line_matchers.yaml").read_text(encoding="utf-8")
        for arm, (lstage, larm, edits) in YAML.items():
            expect = ref
            for a, b in edits:
                expect = expect.replace(a, b)
            got = (CFG / lstage / f"{m}__{larm}" / "second_line_matchers.yaml").read_text(encoding="utf-8")
            assert got == expect, f"{lstage}/{m}__{larm} differs from layer1_plain/{m}__fd beyond {edits}"
            clone(f"{m}__fd", "e2rest", f"{m}__{arm}", edits=edits)
        clone(f"{m}__fd", "e2rest", f"{m}__fd_top1", props=TOPK)
        clone(f"{m}__melnik", "r8", f"{m}__melnik_top1", props=TOPK)
    clone("levenshtein__fd", "e2rest", "levenshtein__fd_twin")
    print(f"e2rest_{c}: {len(matchers) * 3 + 1} configurations; r8_{c}: {len(matchers)} ({len(matchers)} matchers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
