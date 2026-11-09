"""X54 (Synthea -> OMOP) and X55 (Lahman / MusicBrainz in the independent corpus's design), corpora, EXPLORATORY,
. Corpora: data/synthea_plain (39 scenarios), data/indep2_plain (54), with profiles.

Stage 1 (`seeds <corpus>`): the 83 initial matchers of layer1_indep (64 Java + 19 seed server) plus datatype_nanfix
(-Dsmx.datatype.nanAsZero=true, as E2), --no-sf, via make_configs.py in-process (corpus table extended in this process
only) and export_seed_study.export -> seedstudy_<corpus>, seedpy_<corpus>.
Stage 2 (`arms <corpus>`): the first-line matrices -> reference/seeds/<corpus>/<m>/ (a matcher is kept for a dataset
only if every scenario of it has one); SF arms under x54_synthea / x55_indep2 (never a registered folder):
    <m>__melnik, <m>__fd          clones of layer1_plain/<m>__<arm>
    <m>__ucc_top1                 clone of layer3_plain/<m>__ucc + dependencyTopK "1" (Amendment 11's ucc_top1)
    levenshtein__melnik_twin      levenshtein__melnik unchanged except resultsDir: the gate (run twice, equal cell for
                                  cell)
Only the corpus path, datasets, seed directory, resultsDir and (ucc_top1) dependencyTopK change (asserted). No overlay:
melnik, fd and ucc_top1 run on the shared classes (the merge twins: identical outputs either way).

    python servers/make_x5455_configs.py seeds synthea|indep2
    python servers/make_x5455_configs.py arms synthea|indep2
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "servers"))
import make_configs  # noqa: E402
import score_seeds  # noqa: E402

CORPORA = {"synthea": (STUDY / "data" / "synthea_plain", "x54_synthea"),
           "indep2": (STUDY / "data" / "indep2_plain", "x55_indep2")}
for _c, (_p, _) in CORPORA.items():
    make_configs.CORPUS[_c] = _p
    make_configs.EXTERNAL.add(_c)
    score_seeds.CORPUS[_c] = _p
    score_seeds.EXTERNAL.add(_c)
REMOTE_CORPUS = "$SMX_REMOTE_ROOT/smx/corpus"
CFG = STUDY / "tuning" / "configs"
MATCHERS = sorted({p.name.split("__")[0] for p in (CFG / "layer1_indep").iterdir() if p.is_dir()})
assert len(MATCHERS) == 83, len(MATCHERS)


def is_py(m: str) -> bool:
    return make_configs.SEEDS[m][0] == "PythonServerMatcher"


def seeds(corpus: str) -> int:
    argv = sys.argv
    try:
        sys.argv = ["make_configs.py", "--regime", corpus, "--no-sf", "--seeds", ",".join(MATCHERS)]
        make_configs.main()
    finally:
        sys.argv = argv
    cfg = STUDY / "configs" / corpus
    src, dst = cfg / "datatype", cfg / "datatype_nanfix"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = (STUDY / "results" / corpus / "datatype").as_posix()
    assert g.count(old) == 1
    (dst / "general.yaml").write_text(g.replace(old, (STUDY / "results" / corpus / "datatype_nanfix").as_posix()),
                                      encoding="utf-8")
    (dst / "jvm.properties").write_text("smx.datatype.nanAsZero=true\nsmx.cp.overlay=overlay_gvb\n", encoding="utf-8")
    for m in MATCHERS + ["datatype_nanfix"]:
        assert "runSecondLineMatchers: false" in (cfg / m / "general.yaml").read_text(encoding="utf-8"), m
    import export_seed_study as ess
    ess.java_seeds = (lambda f: (lambda: f() + ["datatype_nanfix"]))(ess.java_seeds)
    java = {m for m in MATCHERS if not is_py(m)} | {"datatype_nanfix"}
    py = {m for m in MATCHERS if is_py(m)}
    ess.export(corpus, include=java, python=False, only=java)
    ess.export(corpus, include=py, python=True, only=py)
    print(f"stage 1 {corpus}: {len(java)} java, {len(py)} seed server")
    return 0


def arms(corpus: str) -> int:
    import export_seed_study as ess
    from score_seeds import latest_run, one
    root, suffix = CORPORA[corpus]
    for py in (False, True):
        (STUDY / "tuning" / "results" / ess.suffix_of(corpus, py)).mkdir(parents=True, exist_ok=True)
        ess.import_cluster_results(corpus, python=py)
    ds_all = sorted(p.name for p in root.iterdir() if p.is_dir())
    seed_root = STUDY / "reference" / "seeds" / corpus
    usable = {}
    for m in MATCHERS + ["datatype_nanfix"]:
        run = latest_run(STUDY / "results" / corpus / m)
        if run is None:
            print(f"{m}: no collected stage-1 run; left out")
            continue
        ok = []
        for d in ds_all:
            scen = sorted(p.name for p in (root / d).iterdir() if p.is_dir())
            files = {s: one(run / d / s / "_outputs" / "MatchingStepLine1") for s in scen}
            if all(files.values()):
                for s, f in files.items():
                    dst = seed_root / m / d / s / "seed.csv"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dst)
                ok.append(d)
        usable[m] = ok
        if ok != ds_all:
            print(f"{m}: datasets with a seed for every scenario: {ok} of {ds_all}")
    out_root = CFG / suffix
    if out_root.exists():
        shutil.rmtree(out_root)
    n = 0
    plan = [(m, arm) for m in MATCHERS if usable.get(m) for arm in ("melnik", "fd", "ucc_top1")]
    plan.append(("levenshtein", "melnik_twin"))
    for m, arm in plan:
        stage, src_arm = {"melnik": ("layer1_plain", "melnik"), "fd": ("layer1_plain", "fd"),
                          "ucc_top1": ("layer3_plain", "ucc"), "melnik_twin": ("layer1_plain", "melnik")}[arm]
        src = CFG / stage / f"{m}__{src_arm}"
        label = f"{m}__{arm}"
        ds = usable[m]
        dst = out_root / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(STUDY / "tuning" / "results" / suffix / label).as_posix()}"', g)
        g2 = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{corpus}"', g2)
        assert g2.count(f"{suffix}/{label}\"") == 1 and g2.count(f"{REMOTE_CORPUS}/{corpus}\"") == 1, label
        (dst / "general.yaml").write_text(g2, encoding="utf-8")
        sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
        old_seed = (STUDY / "reference" / "seeds" / "plain" / m).as_posix()
        assert sl.count(old_seed) == 1, label
        sl = sl.replace(old_seed, (seed_root / m).as_posix())
        if arm == "ucc_top1":
            lines = [l for l in sl.splitlines() if l.startswith("  dependencyTopK: [")]
            assert len(lines) == 1, label
            sl = sl.replace(lines[0], '  dependencyTopK: ["1"]')
        (dst / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
        (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in ds), encoding="utf-8")
        meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        meta.update({"seed": m, "arm": arm, "regime": corpus, "stage": suffix.split("_")[0], "seed_dir": (seed_root / m).as_posix(),
                     "datasets": ds, "dataset_sizes": {d: sum(1 for p in (root / d).iterdir() if p.is_dir()) for d in ds},
                     "source": f"{stage}/{m}__{src_arm}", "note": f"{suffix} (B, EXPLORATORY)"})
        meta.pop("sf", None)
        (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        n += 1
    print(f"{suffix}: {n} configurations ({len(plan) - 1} arms + twin), datasets {ds_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"seeds": seeds, "arms": arms}[sys.argv[1]](sys.argv[2]))
