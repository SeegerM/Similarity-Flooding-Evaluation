"""E2 (Amendment 7 of PREREGISTRATION_LAYERS.md): external replication on the controlled corpus (`ctl` =
harness/generated/controlled-imported-v3; Lahman and MusicBrainz, 4 relationship types, plain and opaque, profiled).
The registered configuration generator; T2D (stage 2) is not covered here (it needs profiling first).

    python servers/make_e2_configs.py seeds   # stage 1: the initial matchers on ctl (seed study, no SF), for the cluster
    python servers/make_e2_configs.py arms    # stage 2, after stage 1 is collected and imported: seed files + SF arms

Stage 1: make_configs.py --regime ctl --no-sf for the frozen matchers and controls, plus `datatype_nanfix` (the datatype
matcher with -Dsmx.datatype.nanAsZero=true, overlay `overlay_gvb`); exported with export_seed_study.py as
seedstudy_ctl (Java) and seedpy_ctl (Python, seed server).
Stage 2: each matcher's first-line matrices become reference/seeds/ctl/<m>/<dataset>/<scenario>/seed.csv; the arms are
clones of the registered configurations with only the corpus, the seed directory, the datasets and resultsDir changed:
    melnik, fd, coef_prod_fd   layer1_plain/<m>__<arm>
    ucc                        layer3_plain/<m>__ucc
    fd_top1                    layer1_plain/<m>__fd + smx.sf.pairScope=TOPK, smx.sf.pairScopeParam=1 (R8's TOPK 1)
written to tuning/configs/e2_ctl/<m>__<arm>; `datatype_nanfix` gets no arm (it is the D of the additive term).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY / "servers"))
import make_configs  # noqa: E402

MATCHERS = ["levenshtein", "prefix_suffix", "jaro_winkler", "sbert_mpnet", "pmf", "ks", "jsd_values", "kang_naughton",
            "role_gcov", "datatype"]
CONTROLS = ["constant", "oracle_noise050"]
REGIME = "ctl"
CORPUS = make_configs.CORPUS[REGIME]
REMOTE_CORPUS = "$SMX_REMOTE_ROOT/smx/corpus"
ARMS = {"melnik": ("layer1_plain", "melnik", {}), "fd": ("layer1_plain", "fd", {}),
        "coef_prod_fd": ("layer1_plain", "coef_prod_fd", {}), "ucc": ("layer3_plain", "ucc", {}),
        "fd_top1": ("layer1_plain", "fd", {"smx.sf.pairScope": "TOPK", "smx.sf.pairScopeParam": "1"})}


def datasets() -> list[str]:
    return sorted(p.name for p in CORPUS.iterdir() if p.is_dir())


def seeds() -> int:
    py = sys.executable
    subprocess.run([py, str(STUDY / "make_configs.py"), "--regime", REGIME, "--no-sf",
                    "--seeds", ",".join(MATCHERS + CONTROLS)], check=True, cwd=STUDY)
    src, dst = STUDY / "configs" / REGIME / "datatype", STUDY / "configs" / REGIME / "datatype_nanfix"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    old = (STUDY / "results" / REGIME / "datatype").as_posix()
    assert g.count(old) == 1
    (dst / "general.yaml").write_text(g.replace(old, (STUDY / "results" / REGIME / "datatype_nanfix").as_posix()),
                                      encoding="utf-8")
    (dst / "jvm.properties").write_text("smx.datatype.nanAsZero=true\nsmx.cp.overlay=overlay_gvb\n", encoding="utf-8")
    import export_seed_study as ess
    java = [m for m in MATCHERS + CONTROLS + ["datatype_nanfix"] if make_configs.SEEDS.get(m, ("",))[0] != "PythonServerMatcher"]
    python = [m for m in MATCHERS + CONTROLS if make_configs.SEEDS.get(m, ("",))[0] == "PythonServerMatcher"]
    # datatype_nanfix is not in make_configs.SEEDS; export_seed_study picks seeds from its own pools, so add it there
    ess.java_seeds = (lambda f: (lambda: f() + ["datatype_nanfix"]))(ess.java_seeds)
    ess.export(REGIME, include=set(java), python=False, only=set(java))
    ess.export(REGIME, include=set(python), python=True, only=set(python))
    print(f"stage 1: java {java}; python {python}")
    return 0


def arms() -> int:
    import export_seed_study as ess
    ess.import_cluster_results(REGIME, python=False)
    ess.import_cluster_results(REGIME, python=True)
    seed_root = STUDY / "reference" / "seeds" / REGIME
    cases = [(d, s.name) for d in datasets() for s in sorted((CORPUS / d).iterdir()) if s.is_dir()]
    from score_seeds import latest_run, one
    for m in MATCHERS + CONTROLS + ["datatype_nanfix"]:
        run = latest_run(STUDY / "results" / REGIME / m)
        if run is None:
            raise SystemExit(f"{m}: no collected stage-1 run in results/{REGIME}/{m}")
        for d, s in cases:
            f = one(run / d / s / "_outputs" / "MatchingStepLine1")
            if f is None:
                raise SystemExit(f"{m}: no first-line matrix for {d}/{s}")
            dst = seed_root / m / d / s / "seed.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
    out_root = STUDY / "tuning" / "configs" / f"e2_{REGIME}"
    if out_root.exists():
        shutil.rmtree(out_root)
    ds = datasets()
    n = 0
    for m in MATCHERS + CONTROLS:
        for arm, (stage, src_arm, props) in ARMS.items():
            src = STUDY / "tuning" / "configs" / stage / f"{m}__{src_arm}"
            label = f"{m}__{arm}"
            dst = out_root / label
            shutil.copytree(src, dst)
            g = (dst / "general.yaml").read_text(encoding="utf-8")
            g2 = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(STUDY / "tuning" / "results" / f"e2_{REGIME}" / label).as_posix()}"', g)
            g2 = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{REMOTE_CORPUS}/{REGIME}"', g2)
            assert g2.count(f"e2_{REGIME}/{label}") == 1 and g2.count(f"{REMOTE_CORPUS}/{REGIME}") == 1, label
            (dst / "general.yaml").write_text(g2, encoding="utf-8")
            sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
            old_seed = (STUDY / "reference" / "seeds" / "plain" / m).as_posix()
            assert sl.count(old_seed) == 1, label
            (dst / "second_line_matchers.yaml").write_text(sl.replace(old_seed, (seed_root / m).as_posix()), encoding="utf-8")
            (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in ds), encoding="utf-8")
            if props:
                jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
                (dst / "jvm.properties").write_text(jvm + "".join(f"{k}={v}\n" for k, v in props.items()), encoding="utf-8")
            meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
            meta.update({"seed": m, "arm": arm, "regime": REGIME, "stage": "e2", "seed_dir": (seed_root / m).as_posix(),
                         "datasets": ds, "dataset_sizes": {d: sum(1 for p in (CORPUS / d).iterdir() if p.is_dir()) for d in ds},
                         "source": f"{stage}/{m}__{src_arm}", "note": "E2 (Amendment 7)"})
            meta.pop("sf", None)
            (dst / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            n += 1
    print(f"e2_{REGIME}: {n} configurations ({len(MATCHERS + CONTROLS)} matchers x {len(ARMS)} arms), "
          f"{len(cases)} scenarios in {len(ds)} datasets; seed files under {seed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"seeds": seeds, "arms": arms}[sys.argv[1]]())
