"""Graph and trajectory export panel (Proposition 1 graph-ID check + X88 mechanism 2026-10-07).
Runs SF locally with -Dsmx.sf.export (SfExport.java; off by default, outputs byte-identical with it off: smoke test PASSED with
it off and on) on 14-scenario panel:
    Valentine plain (benchmark-full): 8 TPC-DI cases + the smallest OpenData and ChEMBL scenarios
    GOBY: goby_6017__goby_9524, goby_4588__goby_7681      HDXSM: pair_165, pair_166
initial matchers levenshtein, pmf, prefix_suffix (their FILE seeds from reference/seeds/{plain,goby,hdxsm}), 12 settings
    seedAttachment {NAME, ID, SPLIT} x propCoeffPolicy {INV_AVG, INV_PROD} x literalSeed {PREFIX_SUFFIX, NONE}
on layer1_plain/<m>__melnik (formula C, QUICK, unpruned; only the setting lines, the corpus, the seed dir and resultsDir change).
Writes results/sf_export/<m>__<seed>_<policy>_<lit>/<ds>/<sc>/{nodes,edges,init,sigma_t,trace}.tsv + meta.json and the
ordinary run under results/sf_export_runs/<label>; then checks, per scenario, that MatchingStepLine2 equals (float) the last
sigma_t on the scored nodes (output link) -> results/sf_export/OUTPUT_LINK_CHECK.tsv.

    python servers/sf_export_panel.py [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))
from run_seeds import SCHEMATCH, classpath  # noqa: E402
from score_seeds import one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
VAL = GEN / "benchmark-full"
OUT, RUNS = STUDY / "results" / "sf_export", STUDY / "results" / "sf_export_runs"
WORK = STUDY / "results" / "sf_export_work"
C_CASES = ["TPC-DI-View-Unionable/prospect_both_0_1_ac3_ev", "TPC-DI-Joinable/prospect_vertical_1_ac3_ev",
           "TPC-DI-Semantically-Joinable/prospect_both_50_1_ac5_av", "TPC-DI-Semantically-Joinable/prospect_vertical_1_ac5_av",
           "TPC-DI-Semantically-Joinable/prospect_both_50_30_ac1_av", "TPC-DI-Joinable/prospect_vertical_30_ac1_ev",
           "TPC-DI-Semantically-Joinable/prospect_vertical_30_ac1_av", "TPC-DI-Joinable/prospect_both_50_50_ac1_ev"]
OTHER = [("goby", GEN / "goby-imported", "GOBY-Unionable/goby_6017__goby_9524"),
         ("goby", GEN / "goby-imported", "GOBY-Unionable/goby_4588__goby_7681"),
         ("hdxsm", GEN / "hdxsm-usable", "HDXSM-Unionable/pair_165"),
         ("hdxsm", GEN / "hdxsm-usable", "HDXSM-Unionable/pair_166")]
MATCHERS = ["levenshtein", "pmf", "prefix_suffix"]
SETTINGS = [(a, p, l) for a in ("NAME", "ID", "SPLIT") for p in ("INV_AVG", "INV_PROD") for l in ("PREFIX_SUFFIX", "NONE")]


def smallest(family: str) -> str:
    cands = [(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()), f"{d.parent.name}/{d.name}")
             for ds in VAL.glob(f"{family}-*") for d in ds.iterdir() if d.is_dir()]
    return min(cands)[1]


def panel() -> list[tuple[str, Path, str]]:
    return ([("plain", VAL, c) for c in C_CASES] + [("plain", VAL, smallest("OpenData")), ("plain", VAL, smallest("ChEMBL"))]
            + OTHER)


def build(cases) -> None:
    """A local corpus with just the panel and, per matcher, a seed folder with just their seeds (copies)."""
    if WORK.exists():
        shutil.rmtree(WORK)
    for corpus, root, case in cases:
        shutil.copytree(root / case, WORK / "corpus" / case)
        for m in MATCHERS:
            seed = STUDY / "reference" / "seeds" / corpus / m / case / "seed.csv"
            assert seed.is_file(), seed
            dst = WORK / "seeds" / m / case / "seed.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(seed, dst)


def config(m: str, at: str, pol: str, lit: str, datasets: list[str]) -> Path:
    label = f"{m}__{at}_{pol}_{lit}"
    src, dst = STUDY / "tuning" / "configs" / "layer1_plain" / f"{m}__melnik", WORK / "configs" / label
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    g = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RUNS / label).as_posix()}"', g)
    g = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{(WORK / "corpus").as_posix()}"', g)
    (dst / "general.yaml").write_text(g, encoding="utf-8")
    sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    sl = re.sub(r'(?m)^  initialMatrixDir: \[.*\]$', f'  initialMatrixDir: ["{(WORK / "seeds" / m).as_posix()}"]', sl)
    assert sl.count('propCoeffPolicy: ["INV_AVG"]') == 1, label
    sl = sl.replace('propCoeffPolicy: ["INV_AVG"]', f'propCoeffPolicy: ["{pol}"]')
    (dst / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
    jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
    assert "smx.sf.seedAttachment=NAME" in jvm and "smx.sf.literalSeed=PREFIX_SUFFIX" in jvm, label
    jvm = jvm.replace("smx.sf.seedAttachment=NAME", f"smx.sf.seedAttachment={at}")
    jvm = jvm.replace("smx.sf.literalSeed=PREFIX_SUFFIX", f"smx.sf.literalSeed={lit}")
    jvm += f"smx.sf.export={OUT.as_posix()}\nsmx.sf.exportLabel={label}\n"
    (dst / "jvm.properties").write_text(jvm, encoding="utf-8")
    (dst / "datasets.yaml").write_text("".join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets), encoding="utf-8")
    return dst


def run(cfg: Path, deps: str) -> tuple[str, int]:
    props = ["-D" + l.strip() for l in (cfg / "jvm.properties").read_text(encoding="utf-8").splitlines() if "=" in l]
    (RUNS / cfg.name).mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-Xmx6g", "-XX:+UseSerialGC", *props, "-cp",
           os.pathsep.join([str(cfg), str(SCHEMATCH / "target" / "classes"), deps]), "de.uni_marburg.schematch.Main"]
    with (RUNS / cfg.name / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=SCHEMATCH, stdout=log, stderr=subprocess.STDOUT).returncode
    return cfg.name, code


def check(cases) -> list[list]:
    """MatchingStepLine2 == (float) last sigma_t on the scored nodes, for every label and scenario."""
    import numpy as np
    rows = []
    for label_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        runs = sorted(p for p in (RUNS / label_dir.name).iterdir() if p.is_dir())
        run = runs[-1] if runs else None
        for _, _, case in cases:
            ex = label_dir / case
            f = one(run / case / "_outputs" / "MatchingStepLine2") if run else None
            if not (ex / "meta.json").is_file() or f is None:
                rows.append([label_dir.name, case, "missing", "", "", ""])
                continue
            M = read_matrix(f)[0]
            with (ex / "nodes.tsv").open(encoding="utf-8") as fh:
                scored = {int(r["idx"]): (int(r["row"]), int(r["col"])) for r in csv.DictReader(fh, delimiter="\t")
                          if r["scored"] == "true"}
            last, x = None, {}
            with (ex / "sigma_t.tsv").open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh, delimiter="\t"):
                    t = int(r["t"])
                    if last is None or t > last:
                        last, x = t, {}
                    if t == last:
                        x[int(r["idx"])] = float(r["value"])
            diff, n, nomap = 0.0, 0, 0
            for i, (rr, cc) in scored.items():
                if rr < 0:
                    nomap += 1
                    continue
                # the matrix file holds float32 values as shortest decimal strings: compare as float32
                diff = max(diff, abs(float(np.float32(x[i])) - float(np.float32(M[rr, cc]))))
                n += 1
            rows.append([label_dir.name, case, "ok" if diff == 0.0 and nomap == 0 and n == M.size else "DIFF", diff, n, M.size])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    cases = panel()
    print("panel:", [c for _, _, c in cases])
    if a.check_only:
        cfgs = []
    else:
        for d in (OUT, RUNS):
            if d.exists():
                shutil.rmtree(d)
        build(cases)
    if not a.check_only:
        datasets = sorted({c.split("/")[0] for _, _, c in cases})
        cfgs = [config(m, *s, datasets) for m in MATCHERS for s in SETTINGS]
        deps = classpath()
        with ThreadPoolExecutor(a.workers) as pool:
            for name, code in pool.map(lambda c: run(c, deps), cfgs):
                print(f"{name}: exit {code}", flush=True)
    rows = check(cases)
    with (OUT / "OUTPUT_LINK_CHECK.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["label", "scenario", "status", "max_abs_diff", "scored_cells", "matrix_cells"])
        w.writerows(rows)
    bad = [r for r in rows if r[2] != "ok"]
    print(f"{len(rows)} exports; output link exact on {len(rows) - len(bad)}; not exact or missing: {len(bad)} {bad[:5]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
