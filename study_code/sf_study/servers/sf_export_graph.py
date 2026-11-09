"""X93 (EXPLORATORY): graph-only SF export on every scenario of the study's corpora.
Each corpus's own melnik configuration (NAME, literalSeed PREFIX_SUFFIX, INV_AVG, QUICK, unpruned formula C) with levenshtein as
the initial matcher, run locally with the SfExport flag (code/schematch SfExport.java; off by default, smoke-tested). The only other
change is smx.sf.maxIterations=0 (one update): the cap acts after the graph and the initial vector are built, so nodes, edges and
init are those of the published run (checked below); sigma_t / trace then hold t = 0, 1 only.
Literal-seed entries of init.tsv do not depend on the initial matcher: buildInitialMappingFromMatrix keys every column-name pair
whatever its value, and addLiteralSeeds gives every other non-ID pair PrefixSuffix(value_l, value_r).
    results/sf_export_graph/<corpus>/<ds>/<sc>/{nodes,edges,init,sigma_t,trace}.tsv + meta.json
Check: for the first scenario of each corpus, a full export (default cap) gives byte-identical nodes/edges/init
    -> results/sf_export_graph/GRAPH_ONLY_CHECK.tsv

    python servers/sf_export_graph.py [--workers 4]
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
import make_configs  # noqa: E402
from run_seeds import SCHEMATCH, classpath  # noqa: E402

GEN = STUDY.parent / "harness" / "generated"
OUT, RUNS, WORK = STUDY / "results" / "sf_export_graph", STUDY / "results" / "sf_export_graph_runs", STUDY / "results" / "sf_export_graph_work"
# corpus -> (stage holding its melnik configuration, local corpus root, seed folder name)
CORPORA = {"valentine_plain": ("layer1_plain", make_configs.CORPUS["plain"], "plain"),
           "valentine_md5": ("layer1_md5", make_configs.CORPUS["md5"], "md5"),
           "final_controlled": ("final_controlled", GEN / "final-controlled", "final_controlled"),
           "goby": ("x81_goby", GEN / "goby-imported", "goby"),
           "hdxsm": ("x81_hdxsm", GEN / "hdxsm-usable", "hdxsm"),
           "gdcsm": ("x81_gdcsm", GEN / "gdcsm-imported", "gdcsm"),
           "madi": ("x81_madi", GEN / "madi-imported", "madi"),
           "final_indep": ("final_indep", STUDY / "data" / "final_indep", "final_indep"),
           "mimic": ("x81_mimic", STUDY / "data" / "mimic_omop", "mimic"),
           # X95 (2026-10-08): T2D and ctl, added with --corpora t2d ctl
           "t2d": ("e2_t2d", GEN / "t2d-usable", "t2d"),
           "ctl": ("e2_ctl", GEN / "controlled-imported-v3", "ctl"),
           # X96 (2026-10-09): the multi-table lockbox corpora, added with --corpora synthea indep2
           "synthea": ("x54_synthea", STUDY / "data" / "synthea_plain", "synthea"),
           "indep2": ("x55_indep2", STUDY / "data" / "indep2_plain", "indep2"),
           # 2026-11-01: the 168 new Valentine scenarios (549 cohort), added with --corpora valentine_pnew valentine_mnew
           "valentine_pnew": ("layer1_pnew", make_configs.CORPUS["pnew"], "pnew"),
           "valentine_mnew": ("layer1_mnew", make_configs.CORPUS["mnew"], "mnew")}
M = "levenshtein"


def config(corpus: str, dataset: str, export_root: Path, label: str, graph_only: bool) -> Path:
    stage, root, seeds = CORPORA[corpus]
    src = STUDY / "tuning" / "configs" / stage / f"{M}__melnik"
    name = f"{corpus}__{dataset}" + ("" if graph_only else "__full")
    dst = WORK / "configs" / name
    shutil.copytree(src, dst)
    g = (dst / "general.yaml").read_text(encoding="utf-8")
    g = re.sub(r'(?m)^resultsDir: ".*"$', f'resultsDir: "{(RUNS / name).as_posix()}"', g)
    g = re.sub(r'(?m)^defaultDatasetBasePath: ".*"$', f'defaultDatasetBasePath: "{Path(root).as_posix()}"', g)
    (dst / "general.yaml").write_text(g, encoding="utf-8")
    sl = (dst / "second_line_matchers.yaml").read_text(encoding="utf-8")
    sl = re.sub(r'(?m)^  initialMatrixDir: \[.*\]$',
                f'  initialMatrixDir: ["{(STUDY / "reference" / "seeds" / seeds / M).as_posix()}"]', sl)
    (dst / "second_line_matchers.yaml").write_text(sl, encoding="utf-8")
    jvm = (dst / "jvm.properties").read_text(encoding="utf-8")
    if jvm and not jvm.endswith("\n"):
        jvm += "\n"
    jvm += f"smx.sf.export={export_root.as_posix()}\nsmx.sf.exportLabel={label}\n" + ("smx.sf.maxIterations=0\n" if graph_only else "")
    (dst / "jvm.properties").write_text(jvm, encoding="utf-8")
    (dst / "datasets.yaml").write_text(f'---\nname: "{dataset}"\npath: "{dataset}"\n', encoding="utf-8")
    return dst


def run(cfg: Path, deps: str) -> tuple[str, int]:
    props = ["-D" + l.strip() for l in (cfg / "jvm.properties").read_text(encoding="utf-8").splitlines() if "=" in l]
    (RUNS / cfg.name).mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-Xmx8g", "-XX:+UseSerialGC", *props, "-cp",
           os.pathsep.join([str(cfg), str(SCHEMATCH / "target" / "classes"), deps]), "de.uni_marburg.schematch.Main"]
    with (RUNS / cfg.name / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=SCHEMATCH, stdout=log, stderr=subprocess.STDOUT).returncode
    return cfg.name, code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpora", nargs="*", help="only these corpora; the other corpora's exports are left untouched")
    a = ap.parse_args()
    corpora = a.corpora or list(CORPORA)
    if a.corpora:
        for c in corpora:
            for d in (OUT / c, WORK / "full_export" / c):
                if d.exists():
                    shutil.rmtree(d)
            for d in [*RUNS.glob(f"{c}__*"), *(WORK / "configs").glob(f"{c}__*")]:   # only this corpus's runs and configs
                shutil.rmtree(d)
    else:
        for d in (OUT, RUNS, WORK):
            if d.exists():
                shutil.rmtree(d)
    check_root = WORK / "full_export"
    jobs, firsts = [], {}
    for corpus in corpora:
        root = CORPORA[corpus][1]
        seeds = STUDY / "reference" / "seeds" / CORPORA[corpus][2] / M
        # only datasets of the study protocol (the ones with seeds): benchmark-full also holds a GDC folder outside it
        datasets = sorted(p.name for p in Path(root).iterdir() if p.is_dir() and (seeds / p.name).is_dir())
        for d in datasets:
            jobs.append(config(corpus, d, OUT, corpus, True))
        firsts[corpus] = datasets[0]
        jobs.append(config(corpus, datasets[0], check_root, corpus, False))   # full export of the first dataset: the check
    deps = classpath()
    failed = []
    with ThreadPoolExecutor(a.workers) as pool:
        for name, code in pool.map(lambda c: run(c, deps), jobs):
            print(f"{name}: exit {code}", flush=True)
            if code:
                failed.append(name)
    rows = []
    for corpus, d in firsts.items():
        full = sorted((check_root / corpus / d).iterdir())
        for sc in full:
            same = all((OUT / corpus / d / sc.name / f).read_bytes() == (sc / f).read_bytes()
                       for f in ("nodes.tsv", "edges.tsv", "init.tsv"))
            rows.append([corpus, f"{d}/{sc.name}", "identical" if same else "DIFFERENT"])
    check_file = OUT / ("GRAPH_ONLY_CHECK.tsv" if not a.corpora else f"GRAPH_ONLY_CHECK_{'_'.join(corpora)}.tsv")
    with check_file.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["corpus", "scenario", "nodes_edges_init_vs_full_export"])
        w.writerows(rows)
    counts = {c: sum(1 for _ in (OUT / c).glob("*/*/meta.json")) for c in corpora}
    print("exports per corpus:", counts, "| failed configs:", failed,
          "| check:", sum(r[2] == "identical" for r in rows), "of", len(rows), "identical")
    return 0 if not failed and all(r[2] == "identical" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
