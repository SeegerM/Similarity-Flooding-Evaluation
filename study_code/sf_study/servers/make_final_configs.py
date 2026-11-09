"""Amendment 12 (final validation 2026-10-06): configurations for both parts. Nothing of this runs
before the amendment is appended.

Corpora (study names = the paths the frozen scorer reference/score_final.py expects):
  final_indep       sf-matcher-study/data/final_indep          (Part A, 9 multi-table pairs; data/convert_final_indep.py)
  final_controlled  harness/generated/final-controlled         (Part B, 96 single-table scenarios; data/build_final_controlled.py)
On the hosts: $SMX_REMOTE_ROOT/smx/corpus/final_indep, $SMX_REMOTE_ROOT/smx/corpus/final_controlled.

Stage 1 (`seeds <corpus>`): E2's initial-matcher panel (servers/make_e2_configs.py MATCHERS: levenshtein, prefix_suffix,
jaro_winkler, sbert_mpnet, pmf, ks, jsd_values, kang_naughton, role_gcov, datatype) plus datatype_nanfix (D of the
additive terms; -Dsmx.datatype.nanAsZero=true, overlay_gvb, as E2), --no-sf, via make_configs.py in-process ->
seedstudy_<corpus> (Java) and seedpy_<corpus> (seed server).
Stage 2 (`arms <corpus>`): first-line matrices -> reference/seeds/<corpus>/<m>/; SF arms under tuning/configs/<corpus>:
    <m>__melnik, <m>__fd                  clones of layer1_plain/<m>__<arm>
    <m>__ucc_top1, <m>__ucc1_fd_topk      clones of gvfull_plain/<m>__<arm>   (E2x's arms; unchanged otherwise)
    levenshtein__melnik_twin              levenshtein__melnik again: the gate (two runs equal cell for cell)
Only the corpus path, datasets, seed directory and resultsDir change (asserted). A matcher is kept for a dataset only if
every scenario of it has a first-line matrix (gaps listed).

    python servers/make_final_configs.py seeds final_indep|final_controlled
    python servers/make_final_configs.py arms  final_indep|final_controlled
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
import make_e2_configs as e2  # noqa: E402

CORPORA = {"final_indep": STUDY / "data" / "final_indep",
           "final_controlled": STUDY.parent / "harness" / "generated" / "final-controlled"}
for _c, _p in CORPORA.items():
    make_configs.CORPUS[_c] = _p
    make_configs.EXTERNAL.add(_c)
    score_seeds.CORPUS[_c] = _p
    score_seeds.EXTERNAL.add(_c)
REMOTE_CORPUS = "$SMX_REMOTE_ROOT/smx/corpus"
CFG = STUDY / "tuning" / "configs"
MATCHERS = list(e2.MATCHERS)
ARMS = {"melnik": ("layer1_plain", "melnik"), "fd": ("layer1_plain", "fd"),
        "ucc_top1": ("gvfull_plain", "ucc_top1"), "ucc1_fd_topk": ("gvfull_plain", "ucc1_fd_topk")}


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
    print(f"stage 1 {corpus}: java {sorted(java)}; seed server {sorted(py)}")
    return 0


def arms(corpus: str) -> int:
    import export_seed_study as ess
    from score_seeds import latest_run, one
    root = CORPORA[corpus]
    for py in (False, True):
        (STUDY / "tuning" / "results" / ess.suffix_of(corpus, py)).mkdir(parents=True, exist_ok=True)
        ess.import_cluster_results(corpus, python=py)
    ds_all = sorted(p.name for p in root.iterdir() if p.is_dir())
    seed_root = STUDY / "reference" / "seeds" / corpus
    usable, gaps = {}, {}
    for m in MATCHERS + ["datatype_nanfix"]:
        d_run = STUDY / "results" / corpus / m
        run = latest_run(d_run) if d_run.is_dir() else None
        if run is None:
            gaps[m] = "no collected run"
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
            else:
                gaps.setdefault(m, []).append(d)
        usable[m] = ok
    out_root = CFG / corpus
    if out_root.exists():
        shutil.rmtree(out_root)
    plan = [(m, arm) for m in MATCHERS if usable.get(m) for arm in ARMS] + [("levenshtein", "melnik_twin")]
    for m, arm in plan:
        stage, src_arm = ARMS.get(arm, ("layer1_plain", "melnik"))
        src = CFG / stage / f"{m}__{src_arm}"
        label = f"{m}__{arm}"
        dst = out_root / label
        shutil.copytree(src, dst)
        g = (dst / "general.yaml").read_text(encoding="utf-8")
        g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(STUDY / "tuning" / "results" / corpus / label).as_posix()}"', g)
        g2 = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{corpus}"', g2)
        assert g2.count(f"{corpus}/{label}\"") == 1 and g2.count(f"{REMOTE_CORPUS}/{corpus}\"") == 1, label
        (dst / "general.yaml").write_text(g2, encoding="utf-8")
        sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
        old_seed = (STUDY / "reference" / "seeds" / "plain" / m).as_posix()
        assert sl.count(old_seed) == 1, label
        (dst / "second_line_matchers.yaml").write_text(sl.replace(old_seed, (seed_root / m).as_posix()), encoding="utf-8")
        ds = usable[m]
        (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in ds), encoding="utf-8")
        meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
        meta.update({"seed": m, "arm": arm, "regime": corpus, "stage": "final", "seed_dir": (seed_root / m).as_posix(),
                     "datasets": ds, "dataset_sizes": {d: sum(1 for p in (root / d).iterdir() if p.is_dir()) for d in ds},
                     "source": f"{stage}/{m}__{src_arm}", "note": "Amendment 12 (final validation)"})
        meta.pop("sf", None)
        (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        for f in src.iterdir():
            if f.name not in ("general.yaml", "second_line_matchers.yaml", "datasets.yaml", "meta.json"):
                assert (dst / f.name).read_bytes() == f.read_bytes(), (label, f.name)
    print(f"{corpus}: {len(plan)} configurations ({len(plan) - 1} arms + twin); gaps {gaps or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"seeds": seeds, "arms": arms}[sys.argv[1]](sys.argv[2]))
