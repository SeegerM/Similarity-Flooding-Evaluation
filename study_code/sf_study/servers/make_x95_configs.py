"""X95 stage 2 inputs (the decision on X94; pre-registration layer3b/X95_PREREGISTRATION.md): SF configurations.
Suffix x95sf_<c>, c in valentine_plain, t2d, ctl, final_controlled, goby, hdxsm, gdcsm, madi (2026-10-08):
    ii__melnik, iii__melnik          clones of the corpus's own levenshtein__melnik (layer1_plain, e2_t2d, e2_ctl,
    ii__melnik_t2, iii__melnik_t2    final_controlled, x81_<c>); only initialMatrixDir (-> data/x95_bases/<ii|iii>/<c>),
                                     resultsDir and, for _t2, smx.sf.maxIterations=1 (two updates, as make_iter_configs) change
    coma__std                        goby, hdxsm, gdcsm, madi, final_controlled only (C: plain, t2d, ctl already have it):
                                     SF in the standard configuration seeded by COMA = configs/<c>/coma with
                                     runSecondLineMatchers true, rewritten for the hosts as export_seed_study.export does
                                     (corpus path, resultsDir, code path); hdxsm on the FD-capped hdxsm_fd (193)
Runs: tuning/results/x95sf_<c>/<label>.

    python servers/make_x95_configs.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "servers"))
from export_seed_study import REMOTE_CORPUS, REMOTE_PYROOT  # noqa: E402

CFG, RES = STUDY / "tuning" / "configs", STUDY / "tuning" / "results"
STAGE = {"valentine_plain": "layer1_plain", "t2d": "e2_t2d", "ctl": "e2_ctl", "final_controlled": "final_controlled",
         "goby": "x81_goby", "hdxsm": "x81_hdxsm", "gdcsm": "x81_gdcsm", "madi": "x81_madi"}
STD = {"goby": "goby", "hdxsm": "hdxsm_fd", "gdcsm": "gdcsm", "madi": "madi", "final_controlled": "final_controlled"}


def melnik_arm(c: str, base: str, t2: bool) -> None:
    suffix, label = f"x95sf_{c}", f"{base}__melnik" + ("_t2" if t2 else "")
    src, dst = CFG / STAGE[c] / "levenshtein__melnik", CFG / f"x95sf_{c}" / label
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RES / suffix / label).as_posix()}"', g)
    assert g2.count(f'{suffix}/{label}"') == 1, label
    (dst / "general.yaml").write_text(g2, encoding="utf-8")
    seeds = STUDY / "data" / "x95_bases" / base / c
    assert seeds.is_dir(), seeds
    sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    lines = [l for l in sl.splitlines() if l.startswith("  initialMatrixDir: [")]
    assert len(lines) == 1, label
    (dst / "second_line_matchers.yaml").write_text(sl.replace(lines[0], f'  initialMatrixDir: ["{seeds.as_posix()}"]'),
                                                   encoding="utf-8")
    if t2:
        jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
        assert "maxIterations" not in jvm, label
        if jvm and not jvm.endswith("\n"):
            jvm += "\n"
        (dst / "jvm.properties").write_text(jvm + "smx.sf.maxIterations=1\n", encoding="utf-8")
    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    meta.update({"seed": f"base_{base}", "arm": label.split("__", 1)[1], "stage": "x95sf", "seed_dir": seeds.as_posix(),
                 "source": f"{STAGE[c]}/levenshtein__melnik", "note": "X95 EXPLORATORY: SF seeded by C's averaged base"})
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    for f in src.iterdir():
        if f.name not in ("general.yaml", "second_line_matchers.yaml", "meta.json", "jvm.properties"):
            assert (dst / f.name).read_bytes() == f.read_bytes(), (label, f.name)


def coma_std(c: str, suffix_prefix: str = "x95sf") -> None:
    suffix, label = f"{suffix_prefix}_{c}", "coma__std"
    src, dst = STUDY / "configs" / c / "coma", CFG / suffix / label
    dst.mkdir(parents=True)
    for f in src.iterdir():
        text = f.read_text(encoding="utf-8")
        if f.name == "general.yaml":
            text = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{STD[c]}"', text)
            text = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RES / suffix / label).as_posix()}"', text)
            assert text.count("runSecondLineMatchers: false") == 1, c
            text = text.replace("runSecondLineMatchers: false", "runSecondLineMatchers: true")
            # and SAVE SF's output (MatchingStepLine2): the --no-sf seed configs also set this false (fixed 2026-10-08 22:50,
            # after C found Line1 only; then the only difference from configs/plain/coma, X52's F_std)
            assert text.count("saveOutputSecondLineMatchers: false") == 1, c
            text = text.replace("saveOutputSecondLineMatchers: false", "saveOutputSecondLineMatchers: true")
        text = text.replace((STUDY.parent / "code").as_posix(), f"{REMOTE_PYROOT}/code")
        (dst / f.name).write_text(text, encoding="utf-8")
    sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    for want in ('initialMatrix: ["RAW_FIRST_LINE"]', 'propCoeffPolicy: ["INV_PROD"]', 'graphEncoding: ["QUICK"]', 'fixpoint: ["C"]'):
        assert want in sl, (c, want)
    datasets = re.findall(r'(?m)^path: "(.*)"$', (dst / "datasets.yaml").read_text(encoding="utf-8"))
    # seed_dir as export_seed_study writes it (dispatch.py push reads it); COMA runs through the seed server, no seed files
    meta = {"seed": "coma", "arm": "std", "regime": c, "stage": "x95sf", "seed_dir": (STUDY / "resources").as_posix(),
            "datasets": datasets, "corpus": STD[c],
            "note": "X95 EXPLORATORY: SF standard configuration seeded by COMA (F_std), as configs/<c>/coma with SF on"}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def std_rerun() -> int:
    """The coma__std rerun with SF output saved, under x95sf2_<c> (the scheduler holds x95sf_<c>|coma__std as done); after
    collection A copies x95sf2_<c>/coma__std over x95sf_<c>/coma__std, the path hashed scorer reads."""
    for c in STD:
        out = CFG / f"x95sf2_{c}"
        if out.exists():
            shutil.rmtree(out)
        coma_std(c, "x95sf2")
        print(f"x95sf2_{c}: {sorted(p.name for p in out.iterdir())}")
    return 0


def main() -> int:
    if sys.argv[1:] == ["std-rerun"]:
        return std_rerun()
    for c in STAGE:
        out = CFG / f"x95sf_{c}"
        if out.exists():
            shutil.rmtree(out)
        for base in ("ii", "iii"):
            for t2 in (False, True):
                melnik_arm(c, base, t2)
        if c in STD:
            coma_std(c)
        print(f"x95sf_{c}: {sorted(p.name for p in out.iterdir())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
