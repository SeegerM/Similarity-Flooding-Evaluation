"""Rerun 549 (2026-10-31): one command that assembles the 549-scenario inputs, regenerates the paper's macros on them and
writes the macro diff 382 -> 549. Rerun it whenever a new batch is scored (reference/score_batch.py).

    ../harness/.venv-bdi/Scripts/python.exe -B reference/run_549.py            # verify 382, assemble, generate, diff
    ../harness/.venv-bdi/Scripts/python.exe -B reference/run_549.py --no-verify

Steps
  0. verify: every modified generator, run WITHOUT the switch into a temporary folder, reproduces its current 382 output
     byte for byte (gen_figures.py vs latex/sf-evaluation/figures, rq1_artefact.py vs tuning/results/review_checks,
     make_review_numbers / make_abs_numbers / make_ea_numbers / the rq1_relgain data vs the active manuscript's figures).
  1. reference/assemble_549.py (pending families are reported, never filled with old numbers).
  2. the generators with SMX_VALENTINE=549 -> latex/sf-evaluation-549/figures/ (+ review_checks/): gen_figures.py,
     review_checks/rq1_artefact.py plain, review_checks/make_review_numbers.py, make_abs_numbers.py, make_ea_numbers.py,
     and the rq1_relgain plot data (as make_rq1_relgain.py computes it; the active figure is hand-edited, so only its data).
  3. same-panel decomposition: the rq1 / rq1_std macros on the 382 scenarios restricted to the 549 panel's matchers.
  4. the diff: harness/logs_rerun549/macro_diff_382_549.md and latex/sf-evaluation-549/MACRO_STATUS.tsv. Old values are the
     active manuscript's (latex/sf-evaluation-presentation-proposal/figures), new values the 549 folder's. Status per macro:
     changed / unchanged (input assembled on 549), pending (input batch or rerun step missing; the 549 folder may hold a
     value computed from 382 inputs, which is not a 549 number), fixed (registered or not Valentine: must be identical).
Nothing in the active manuscript, the long version's figures or the old result folders is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
PY = ROOT / "harness" / ".venv-bdi" / "Scripts" / "python.exe"
LONG = ROOT / "latex" / "sf-evaluation"
PRES = ROOT / "latex" / "sf-evaluation-presentation-proposal"
OUT549 = ROOT / "latex" / "sf-evaluation-549"
FIG549 = OUT549 / "figures"
LOGDIR = ROOT / "harness" / "logs_rerun549"
ASSEMBLY = STUDY / "results" / "valentine549" / "assembly.json"
GROUP_FILES = {"label": 0, "inst": 1, "dep": 2, "comb": 3}   # group -> the index in the per-group files' first column


def run(cmd: list, cwd: Path, env: dict | None = None, quiet: bool = True) -> str:
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    e.pop("SMX_VALENTINE", None), e.pop("SMX_FIGURES_DIR", None), e.pop("SMX_REVIEW_DIR", None)
    e.update(env or {})
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(map(str, cmd))} (in {cwd})")
    if not quiet:
        print(r.stdout[-2000:])
    return r.stdout


# ------------------------------------------------------------------------------------------------ rq1_relgain data
def relgain(gain_csv: Path, out: Path) -> None:
    """make_rq1_relgain.py's data step (rel = d / (1 - x) over the non-control matchers), plus the per-group files the
    active figure plots (group column 0, same rows)."""
    rows = [r for r in csv.DictReader(gain_csv.open(encoding="utf-8")) if r["group"] != "ctrl"]
    for r in rows:
        r["rel"] = float(r["d_R_exp"]) / (1 - float(r["x"]))
    with (out / "rq1_relgain_data.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "x", "d", "rel"])
        for r in rows:
            w.writerow([r["group"], f"{float(r['x']):.5f}", f"{float(r['d_R_exp']):.5f}", f"{r['rel']:.5f}"])
    for g in GROUP_FILES:
        with (out / f"rq1_relgain_data_{g}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["group", "x", "d", "rel"])
            for r in rows:
                if r["group"] == g:
                    w.writerow([GROUP_FILES[g], f"{float(r['x']):.5f}", f"{float(r['d_R_exp']):.5f}", f"{r['rel']:.5f}"])


def deproles_549(regime: str, env: dict) -> None:
    """explore_deproles.py under SMX_VALENTINE=549 (tens of minutes per regime): rerun only when its output is missing or
    the content of one of its assembled inputs (layer1, layer1nan, gvfull of that regime) changed since its last run
    (sha256 recorded in results/<r>549/deproles_inputs.sha256)."""
    import hashlib
    name = {"plain": "plain549", "md5": "md549"}[regime]
    out = STUDY / "results" / name / "deproles_summary.md"
    stamp = STUDY / "results" / name / "deproles_inputs.sha256"
    ins = [STUDY / "tuning" / "results" / f"{f}_{name}_scores" / "scores.csv" for f in ("layer1", "layer1nan", "gvfull")]
    digest = "\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else '-'}  {p.parent.name}"
                       for p in ins) + "\n"
    if out.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == digest:
        print(f"deproles {name}: up to date")
        return
    print(run([PY, "-B", "explore_deproles.py", "--regime", regime, "--workers", "14"], STUDY, env)[-400:])
    if out.is_file() and out.stat().st_mtime > max(p.stat().st_mtime for p in ins if p.is_file()):
        stamp.write_text(digest, encoding="utf-8")


FAILED: set[str] = set()   # 549 steps that failed in this run: their macros are reported pending, whatever file exists


def try_run(cmd: list, cwd: Path, env: dict) -> None:
    """A 549 step whose failure must not stop the run (e.g. a report part not computed yet); recorded in FAILED."""
    try:
        run(cmd, cwd, env)
    except SystemExit as exc:
        FAILED.add(Path(str(cmd[2])).name)
        print(f"step failed (its macros stay pending): {exc}")


def cached(name: str, cmd: list, cwd: Path, env: dict, inputs: list[Path], outputs: list[Path]) -> None:
    """Run an expensive 549 step only when its outputs are missing or the content of its inputs (assembled files and the
    script itself) changed since its last successful run; stamps in latex/sf-evaluation-549/.stamps/<name>.sha256."""
    import hashlib
    stamp = OUT549 / ".stamps" / f"{name}.sha256"
    digest = "\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else '-'}  {p}" for p in inputs) + "\n"
    if all(o.is_file() for o in outputs) and stamp.is_file() and stamp.read_text(encoding="utf-8") == digest:
        print(f"{name}: up to date")
        return
    print(run(cmd, cwd, env)[-400:])
    if all(o.is_file() for o in outputs):
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(digest, encoding="utf-8")


def asm_file(folder: str) -> Path:
    return STUDY / "tuning" / "results" / folder / "scores.csv"


# ------------------------------------------------------------------------------------------------ verification (382)
def verify_382(tmp: Path) -> list[str]:
    msgs = []

    def cmp(a: Path, b: Path, label: str) -> None:
        ok = a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()
        msgs.append(f"{'ok' if ok else 'DIFFERENT'}: {label}")
        if not ok:
            raise SystemExit(f"382 behaviour changed: {label} ({a} vs {b})")

    g = tmp / "gen_figures"
    run([PY, "-B", "gen_figures.py"], LONG, {"SMX_FIGURES_DIR": str(g)})
    files = sorted(p for p in g.iterdir() if p.is_file())
    for p in files:
        cmp(p, LONG / "figures" / p.name, f"gen_figures.py -> figures/{p.name}")
    msgs.append(f"gen_figures.py, 382 mode: {len(files)} files byte-identical to latex/sf-evaluation/figures/")
    r = tmp / "review"
    run([PY, "-B", "review_checks/rq1_artefact.py", "plain"], STUDY, {"SMX_REVIEW_DIR": str(r)})
    cmp(r / "rq1_artefact_plain.md", STUDY / "tuning" / "results" / "review_checks" / "rq1_artefact_plain.md",
        "rq1_artefact.py plain")
    f = tmp / "pres"
    f.mkdir(parents=True, exist_ok=True)
    run([PY, "-B", "review_checks/make_review_numbers.py"], STUDY, {"SMX_FIGURES_DIR": str(f)})
    run([PY, "-B", "make_abs_numbers.py"], PRES, {"SMX_FIGURES_DIR": str(f)})
    run([PY, "-B", "make_ea_numbers.py"], PRES, {"SMX_FIGURES_DIR": str(f)})
    for s in ("make_ss_numbers.py", "make_ssa_numbers.py", "make_om_numbers.py"):
        run([PY, "-B", s], PRES, {"SMX_FIGURES_DIR": str(f)})
    relgain(LONG / "figures" / "rq1_gain_data.csv", f)
    for name in ["review_numbers.tex", "abs_numbers.tex", "abs_numbers_report.md", "ea_numbers.tex",
                 "ea_numbers_report.md", "ss_numbers.tex", "ssa_numbers.tex", "om_numbers.tex", "rq1_relgain_data.csv"] + [f"rq1_relgain_data_{x}.csv" for x in GROUP_FILES]:
        cmp(f / name, PRES / "figures" / name, f"{name} (active manuscript)")
    msgs = [m for m in msgs if not m.startswith("ok: gen_figures.py ->")]
    # the report scripts refactored for the 549 run (explore_iterations, explore_cluster_batches, score_parallel)
    msgs += run([PY, "-B", "reference/verify_549_reports.py"], STUDY).strip().splitlines()
    t = tmp / "tu"
    t.mkdir(parents=True, exist_ok=True)
    run([PY, "-B", "review_checks/two_update_offset_check.py"], STUDY, {"SMX_REVIEW_DIR": str(t)})
    run([PY, "-B", "make_tu_numbers.py"], PRES, {"SMX_REVIEW_DIR": str(t), "SMX_FIGURES_DIR": str(t)})
    for a, b, label in ((t / "two_update_offset.csv", STUDY / "tuning" / "results" / "review_checks" / "two_update_offset.csv",
                         "two_update_offset_check.py"), (t / "tu_numbers.tex", PRES / "figures" / "tu_numbers.tex",
                                                         "tu_numbers.tex (active manuscript)")):
        cmp(a, b, label)
    return msgs


# ------------------------------------------------------------------------------------------------ same-panel helper
def same_panel_child(panel_file: str, out_file: str) -> int:
    """Run in a subprocess. 549 mode: write the panels and their macros; 382 mode: the rq1 / rq1_std macros on the
    382 scenarios for the matchers of the 549 panels."""
    sys.path.insert(0, str(LONG))
    import gen_figures as gf  # noqa: E402
    rows = gf.layer1_rows("plain", "melnik")
    ref, _ = gf.load_rows(rows)
    std, _ = gf.load("plain")
    if gf.V549:
        out = {"rq1": sorted(p["seed"] for p in ref if p["group"] != "ctrl"),
               "std": sorted(p["seed"] for p in std if p["group"] != "ctrl")}
        # the 549 inputs on the 381 old scenarios only (same 549 panel): separates the new scenarios from the FD fix
        old381 = {(c["dataset"], c["scenario"]) for c in csv.DictReader(
            open(STUDY / "results" / "valentine549" / "cases.csv", encoding="utf-8")) if c["cohort"] == "old381"}
        std_rows = list(csv.DictReader(open(STUDY / "results" / "plain549" / "scores_tiefree.csv", encoding="utf-8")))
        for key, src, panel in (("rqOne381", rows, out["rq1"]), ("rqStd381", std_rows, out["std"])):
            sub = [r for r in src if (r["dataset"], r["scenario"]) in old381]
            pts, _ = gf.load_rows(sub, 381)
            out[key] = gf.rq_numbers("plain", [p for p in pts if p["seed"] in set(panel) or p["group"] == "ctrl"], sub)
    else:
        panel = json.loads(Path(panel_file).read_text(encoding="utf-8"))
        full = {"rq1": sorted(p["seed"] for p in ref if p["group"] != "ctrl"),
                "std": sorted(p["seed"] for p in std if p["group"] != "ctrl")}
        out = {"panel382": full, "rqOne": gf.rq_numbers("plain", [p for p in ref if p["seed"] in set(panel["rq1"])
                                                                  or p["group"] == "ctrl"], rows),
               "rqStd": gf.rq_numbers("plain", [p for p in std if p["seed"] in set(panel["std"])
                                                or p["group"] == "ctrl"], None)}
    Path(out_file).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------------ diff
def parse_macros(text: str) -> dict[str, str]:
    out, i = {}, 0
    pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{")
    while True:
        m = pat.search(text, i)
        if not m:
            return out
        j, depth = m.end(), 1
        while depth and j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out[m.group(1)] = text[m.end():j - 1]
        i = j


def manuscript_files() -> tuple[list[str], set[str]]:
    """The macro files main.tex inputs, and every token \\name used in the manuscript outside the macro files."""
    seen, numbers, used = set(), [], set()

    def walk(rel: str) -> None:
        p = PRES / (rel if rel.endswith(".tex") else rel + ".tex")
        if p in seen or not p.is_file():
            return
        seen.add(p)
        text = p.read_text(encoding="utf-8")
        if p.name.endswith("_numbers.tex"):
            numbers.append(p.name)
            return
        body = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%") else "" for line in text.splitlines())
        used.update(re.findall(r"\\([A-Za-z]+)", body))
        for inc in re.findall(r"\\input\{([^}]+)\}", body):
            walk(inc)
    walk("main.tex")
    return numbers, used


# Status rules, first match wins: (file, macro regex, needs [family/arm in assembly.json], extra rerun step or None,
# fixed reason or None). A rule is "assembled" when all needs are assembled and no extra step is named.
L1P, L1M = "layer1_plain/melnik", "layer1_md5/melnik"
RULES = [
    ("rq1_numbers.tex", r".*", [L1P], None, None),
    ("rq1_std_numbers.tex", r".*", ["seed_plain"], None, None),
    ("rq4_numbers.tex", r".*", ["seed_plain", "seed_md5"], None, None),
    ("rqc_numbers.tex", r"rqcVPl.*", [L1P], None, None),
    ("rqc_numbers.tex", r"rqcVMd.*", [L1M], None, None),
    ("layer1_plain_numbers.tex", r"lp(N|Ref.*)", [L1P], None, None),
    ("layer1_plain_numbers.tex", r"lpFd.*", ["layer1_plain/fd"], None, None),
    ("layer1_plain_numbers.tex", r".*", [], "not in the 549 rerun scope (layer-1 arms attach_id, coef_prod, "
                                             "coef_prod_fd, lit_none: long version only)", None),
    ("layer1_md5_numbers.tex", r"lm(N|Ref.*)", [L1M], None, None),
    ("layer1_md5_numbers.tex", r"lmFd.*", ["layer1_md5/fd"], None, None),
    ("layer1_md5_numbers.tex", r".*", [], "not in the 549 rerun scope (long-version layer-1 arms)", None),
    ("review_numbers.tex", r"rv(Split|Head|Null|RelNull).*", [L1P], None, None),
    ("review_numbers.tex", r"rvBeta(Min|Max)", [], None, "design constant (the beta grid of the sweep)"),
    ("review_numbers.tex", r"rvBeta.*", [L1P, L1M, "file:../../../latex/sf-evaluation-549/review_checks/beta_sweep.csv"],
     None, None),
    ("review_numbers.tex", r"rv(Lbx.*|NIMargin)", [], None, "registered lockbox / constant"),
    ("review_numbers.tex", r"rv(Constr|Ovl)\w*", ["file:../../../latex/sf-evaluation-549/review_checks/beta_sweep.csv",
                                                  "file:../../results/x81_549/benchmark_profile_x81.md"], None, None),
    ("review_numbers.tex", r".*", ["file:../../../latex/sf-evaluation-549/review_checks/corpus_assoc_grid.csv"], None, None),
    ("abs_numbers.tex", r"abs(Q|Sf)Val", [L1P], None, None),
    ("abs_numbers.tex", r"abs(Q|Sf)Syn", ["mtfix:synthea"], None, None),
    ("abs_numbers.tex", r"abs(Q|Sf)LmM", ["mtfix:indep2"], None, None),
    ("abs_numbers.tex", r".*", [], None, "not Valentine"),
    ("ea_numbers.tex", r"eaPos(Scenarios|Pairs|DiagPairs|OffDiagPairs)", ["file:../../results/valentine549/positional_survey.json"],
     None, None),
    ("ea_numbers.tex", r"eaPosSima\w*", [], "SiMa has no matrices for the 168 (held-out SiMa model not rerun)", None),
    ("ea_numbers.tex", r"eaPos(Old\w*|Coma\w*|Cupid\w*|PermRepeats|Const\w*)", ["file:../../../harness/reports/positional_subsidy_549.json"],
     None, None),
    ("ea_numbers.tex", r"eaPosPermComaIdentical", ["file:../../../harness/reports/positional_subsidy_549new168_rerun_coma.json"], None, None),
    ("ea_numbers.tex", r"eaPos.*", [], "not recomputed on 549 (positional_subsidy rerun for Cupid / COMA+instances, md5)", None),
    ("ea_numbers.tex", r".*", [], None, "not Valentine (independent sampling)"),
    ("robust_numbers.tex", r"robLit.*", [], None, "registered X84 sample"),
    ("robust_numbers.tex", r".*", [], "rerun explore_robust.py on 549", None),
    ("dr_numbers.tex", r"dr(Pl|Md)((Ucc|FdUccInd)Vs\w+|\w+Vs(Ucc|FdUccInd)(Est|Lo|Hi|Pos|N))", [],
     "layer 3 (ucc, fd_ucc_ind) is not rerun on 549 (long version only)", None),
    ("dr_numbers.tex", r"drMd(UccFdTopkVs\w+|\w+VsUccFdTopk\w*)", [], "recoded ucc1_fd_topk is not rerun on 549", None),
    ("dr_numbers.tex", r"drPl\w*", ["layer1_plain/melnik", "layer1_plain/fd", "gvfull_plain/ucc_top1",
                                    "gvfull_plain/ucc1_fd_topk", "layer1nan_plain/melnik", "deproles:plain549"], None, None),
    ("dr_numbers.tex", r"drMd\w*", [L1M, "layer1_md5/fd", "gvfull_md5/ucc_top1", "layer1nan_md5/melnik",
                                    "deproles:md549"], None, None),
    ("drm_numbers.tex", r"drmPl\w*", ["deproles:plain549", "file:../../results/plain549/x46_fullmetrics_summary.csv"], None, None),
    ("drm_numbers.tex", r"drmMd\w*", ["deproles:md549", "file:../../results/md549/x46_fullmetrics_summary.csv"], None, None),
    ("fam_numbers.tex", r"famPl\w*", ["layer1_plain/fd", "file:x58_plain549/report.md"], None, None),
    ("fam_numbers.tex", r"famMd\w*", ["layer1_md5/fd", "file:x58_md549/report.md"], None, None),
    ("it_numbers.tex", r"itInd\w*", [], None, "not Valentine (independent corpus)"),
    ("it_numbers.tex", r"it(Pl|Md)Fd\w*", [], "fd_t* are not rerun on 549 (long version only)", None),
    ("it_numbers.tex", r"it(Pl|Md)InvMax", [], "the inversion check needs the melnik_t* matrices of the 168", None),
    ("it_numbers.tex", r"itPl\w*", [f"iter_plain/{x}" for x in ("melnik", "melnik_t1", "melnik_t10")], None, None),
    ("it_numbers.tex", r"itMd\w*", [f"iter_md5/{x}" for x in ("melnik", "melnik_t1", "melnik_t10")], None, None),
    ("itr_numbers.tex", r"itrInd\w*", [], None, "not Valentine (independent corpus)"),
    ("itr_numbers.tex", r"itrPl\w*", ["iter_plain/melnik", "iter_plain/melnik_t1"], None, None),
    ("itr_numbers.tex", r"itrMd\w*", ["iter_md5/melnik", "iter_md5/melnik_t1"], None, None),
    ("ld_numbers.tex", r"ldRho(Val|ShareVal)", ["file:x93_549_scores/mechanism.csv"], None, None),
    ("ld_numbers.tex", r"ldRho(VMd|ShareVMd)", [], "X93 P1 covers readable Valentine only", None),
    ("ld_numbers.tex", r"ld\w*Val\w*", ["layer1_plain/melnik", "iter_plain/melnik_t2", "file:x93_549_scores/contrasts.csv"],
     None, None),
    ("ld_numbers.tex", r"ld\w*VMd\w*", ["layer1_md5/melnik", "iter_md5/melnik_t2", "file:x93_549_scores/contrasts.csv"],
     None, None),
    ("ld_numbers.tex", r"ld(LamL|LamDL)", [], None, "frozen lambda (fit on the 382 screen)"),
    ("ld_numbers.tex", r".*", [], None, "not Valentine; melnik arms only (no FD edges), unchanged by the HyFD fix"),
    ("rlx_numbers.tex", r"rlx\w*(Val|VMd)", [], "rerun explore_x92_rules.py on 549 (not wired; unused in the manuscript)",
     None),
    ("rlx_numbers.tex", r".*", [], None, "not Valentine; melnik arms only (no FD edges), unchanged by the HyFD fix"),
    ("pe_numbers.tex", r"pe(Pl|Md)(AvgFd|MaxFd)\w*", [], "fd_pavg / fd_pmax are not rerun on 549", None),
    ("pe_numbers.tex", r"pePl\w*", ["parallel_plain/fd_split", "layer1_plain/fd"], None, None),
    ("pe_numbers.tex", r"peMd\w*", ["parallel_md5/fd_split", "layer1_md5/fd"], None, None),
    # external FD stages (fixed HyFD profiles; explore_x81.py / explore_multitable_ext.py under SMX_VALENTINE=549)
    ("xt_numbers.tex", r"xtGdc\w*", ["x81fix:gdcsm"], None, None),
    ("xt_numbers.tex", r"xtHdx\w*", ["x81fix:hdxsm"], None, None),
    ("xt_numbers.tex", r"xtGoby\w*", ["x81fix:goby"], None, None),
    ("xt_numbers.tex", r"xtMadiP\w*", ["x81fix:madi_products"], None, None),
    ("xt_numbers.tex", r"xtMadi\w*", ["x81fix:madi_indep"], None, None),
    ("xt_numbers.tex", r"xtMimic\w*", ["x81fix:mimic"], None, None),
    ("xt_numbers.tex", r".*", ["x81fix:gdcsm", "x81fix:hdxsm", "x81fix:goby", "x81fix:madi_indep"], None, None),
    ("wp_numbers.tex", r".*", [], None, "X82 (f) and the melnik trajectory: melnik arms only (no FD edges), unchanged by "
                                      "the HyFD fix"),
    ("mt_numbers.tex", r"mtSyn\w*", ["mtfix:synthea"], None, None),
    ("mt_numbers.tex", r".*", ["mtfix:indep2"], None, None),
    ("cx_numbers.tex", r"cxSplit\w*", [], "X82 parallel FD edges on the dev split (x82_parallel_scores_dev1): FD arms, not "
     "rerun on the fixed FDs", None),
    ("cx_numbers.tex", r".*", [], None, "class nodes and additive class terms on the external corpora (X82/X83/X86): "
     "no FD input, unchanged by the HyFD fix"),
    ("xe_numbers.tex", r"xePl\w*", [L1P, "file:../../results/plain549/extraction_summary.md"], None, None),
    ("xe_numbers.tex", r"xeMd\w*", [L1M, "file:../../results/md549/extraction_summary.md"], None, None),
    ("xe_numbers.tex", r".*", [], None, "not Valentine (independent corpus)"),
    ("xc_numbers.tex", r"xcConst(H|R)Val", ["seed_plain"], None, None),
    ("xc_numbers.tex", r"xcConst(H|R)\w+", [], None, "constant matcher on a non-Valentine corpus (no FD input)"),
    ("xc_numbers.tex", r"xcOrdKey\w*", [], None, "X87 permuted columns on HDXSM/MaDI, single-key (UCC) graph: no FD "
     "input, UCC profiles unchanged"),
    ("xc_numbers.tex", r".*", [L1P], "rerun explore_x85_*.py / X87 on 549 (not wired)", None),
    ("dt_numbers.tex", r"dtUniBeta", [], None, "frozen weight, chosen on the 60 screen scenarios (no FD input)"),
    ("dt_numbers.tex", r".*", [], "rerun explore_dtype_baseline.py on 549 (not wired)", None),
    ("cp_numbers.tex", r"cp\w*(Val|VMd)\w*", [L1P], "rerun explore_x89_rq1.py on 549 (not wired)", None),
    ("cp_numbers.tex", r".*", [], None, "not Valentine; melnik arms only (no FD edges)"),
    ("pr_numbers.tex", r"pr\w*Val", ["file:../../results/x81_549/benchmark_profile_x81.md"], None, None),
    ("pr_numbers.tex", r"pr\w*VMd", [], "recoded Valentine properties not recomputed", None),
    ("pr_numbers.tex", r".*", [], None, "benchmark properties of a non-Valentine corpus (value overlap, names; no FD input)"),
    ("ce_numbers.tex", r".*", ["file:../../results/x78_549/benchmark_effects_report.md"], None, None),
    ("st_numbers.tex", r".*", [], "rerun explore_scenario_types.py on 549", None),
    ("tp_numbers.tex", r".*", [], "x60 seeds and explore_x60.py on 549", None),
    ("xso_numbers.tex", r".*", [], "x61 seeds and explore_x61.py on 549", None),
    ("hy_numbers.tex", r".*", [], "X70 (screen FDs; long version)", None),
    ("eg_numbers.tex", r".*", [], "x62_plain on 549 (X62)", None),
    ("mr_numbers.tex", r"mrPl(N|LevBestName|LevBestEst|LevEdgeSfMean|LevRoleEdgeMean)",
     ["file:../../results/plain549/matcherroles_summary.md"], None, None),
    ("mr_numbers.tex", r".*", [], "X73 on 549 is run for readable levenshtein only (cluster: seedstudy_self_pnew, "
     "x62_pnew); the other matchers / the recoded regime are not rerun", None),
    ("rn_numbers.tex", r".*", [], "X97 on 549 (long version)", None),
    ("fw_numbers.tex", r"fwPh\w*", [], "X95 post-hoc matched comparison (x95_posthoc_matched.py) not rerun on 549", None),
    ("fw_numbers.tex", r"fwPrim(Val|VMd)", ["file:x95_549/contrasts.csv"], None, None),
    ("fw_numbers.tex", r"fwPrim\w*", [], "full − SF converged uses the role term R13 (FD profiles); the other corpora's "
     "X95 rows are still on the old FD profiles (X95 not rerun on their fixed FDs)", None),
    ("fw_numbers.tex", r"fwLTwo\w*", ["file:x95_549/contrasts.csv"], None, None),
    ("fw_numbers.tex", r".*", ["file:../../results/x94_549/stage1.md", "file:x94_549/cells.csv.gz"], None, None),
    ("lt_numbers.tex", r"ltVal\w*", ["file:x90_549_scores/report.md"], None, None),
    ("lt_numbers.tex", r".*", [], None, "not Valentine; melnik and lit_none arms (no FD edges)"),
    ("tpass_numbers.tex", r".*", [], None, "timing pass measured on the 382 (descriptive cost measurement; the paper "
     "states it; the user's decision 2026-11-02)"),
    ("kr_numbers.tex", r".*", [], "knnrnd (long version)", None),
    ("fsr_numbers.tex", r".*", [], "f6rnd (long version)", None),
    ("ss_numbers.tex", r".*", ["file:../../../harness/reports/semstruct_ablation_549", "step:make_ss_numbers.py"], None, None),
    ("ssa_numbers.tex", r".*", ["file:../../../harness/reports/semstruct_a2_549", "step:make_ssa_numbers.py"], None, None),
    ("om_numbers.tex", r".*", ["file:../../../harness/reports/omnimatch_ablation_549", "step:make_om_numbers.py"], None, None),
    ("tu_numbers.tex", r".*", ["file:../../../latex/sf-evaluation-549/review_checks/two_update_offset.csv"], None, None),
    ("src_numbers.tex", r".*", [L1P, "deproles:plain549", "file:../../../latex/sf-evaluation-549/review_checks/source_sensitivity.json"], None, None),
    ("lbx_numbers.tex", r".*", [], None, "registered lockbox"),
    ("fin_numbers.tex", r".*", [], None, "registered final test (A12)"),
    ("cf_numbers.tex", r".*", [], None, "registered class nodes (A8)"),
    ("etwo_numbers.tex", r".*", [], None, "registered A7 (not Valentine)"),
    ("rr_numbers.tex", r".*", [], None, "registered A11 (not Valentine)"),
    ("tc_numbers.tex", r".*", [], None, "registered A9 (not Valentine)"),
    ("amf_numbers.tex", r".*", [], None, "registered A5 (not Valentine)"),
    ("xl_numbers.tex", r".*", [], None, "fixed X38 sample"),
    ("sg_numbers.tex", r".*", [], None, "fixed screen sample (X63)"),
    ("kn_numbers.tex", r".*", [], None, "fixed X79 knob bases"),
    ("gx_numbers.tex", r".*", [], None, "fixed X91 export check"),
    ("ttd_numbers.tex", r".*", [], None, "not Valentine (T2D)"),
    ("erten_numbers.tex", r".*", [], None, "not Valentine (ER10, indep)"),
    ("sw_numbers.tex", r".*", [], None, "not Valentine (X51 scaling sweep)"),
    ("sv_numbers.tex", r".*", [], None, "not Valentine: X88 held-out corpora, melnik arms only (no FD edges)"),
]
HEADLINE = re.compile(r"rqOne(N|Rho.*)|rqStd(N|Mean.*|Rho.*)|rqc.*|rv(Split|Head|Null|RelNull).*|lpN|lpRef.*|abs(Q|Sf)Val")


def classify(file: str, macro: str, asm: dict) -> tuple[str, str]:
    for f, rx, needs, extra, fixed in RULES:
        if f == file and re.fullmatch(rx, macro):
            if fixed:
                return "fixed", fixed
            miss = [n for n in needs if status_of(asm, n) != "assembled"]
            if miss:
                return "pending", "input batch not assembled: " + ", ".join(miss) + (f"; then {extra}" if extra else "")
            if extra:
                return "pending", extra
            return "assembled", ", ".join(needs)
    return "pending", "unclassified (no rule): check"


def status_of(asm: dict, need: str) -> str:
    if need.startswith("step:"):   # a 549 step that must have succeeded in this run
        return "missing" if need[5:] in FAILED else "assembled"
    if need.startswith("file:"):   # an output of a 549 report step (file or folder), relative to tuning/results
        return "assembled" if (STUDY / "tuning" / "results" / need[5:]).exists() else "missing"
    if need.startswith("x81fix:"):   # an external corpus computed on the fixed FDs (results/x81fix/report.md section)
        rep = need.split(":")[1]
        f = STUDY / "results" / "x81fix" / "report.md"
        ok = f.is_file() and any(line.startswith(f"## {rep}") and "pending" not in line
                                 for line in f.read_text(encoding="utf-8").splitlines())
        return "assembled" if ok else "missing"
    if need.startswith("mtfix:"):
        f = STUDY / "results" / "multitable_fix" / need.split(":")[1] / "multitable_summary.md"
        return "assembled" if f.is_file() else "missing"
    if need.startswith("deproles:"):   # explore_deproles.py's 549 summary (written by step 2 when its inputs are assembled)
        return "assembled" if (STUDY / "results" / need.split(":")[1] / "deproles_summary.md").is_file() else "missing"
    fam, _, arm = need.partition("/")
    e = asm["families"].get(fam, {})
    if not arm:
        return e.get("status", "missing")
    return e.get("arms", {}).get(arm, {}).get("status", "missing")


def diff(asm: dict, same: dict, verify: list[str], panel549: dict) -> None:
    numbers, used = manuscript_files()
    ladder = set(re.findall(r"\\([A-Za-z]+)", (PRES / "figures" / "info_ladder.tex").read_text(encoding="utf-8")))
    ladder |= {f"{b}{s}" for b in ("drPlRefVsAd", "drMdRefVsAd", "tcHOne") for s in ("Est", "Lo", "Hi")}   # \vA \vB \vL
    table2 = set(re.findall(r"\\([A-Za-z]+)", (PRES / "tables" / "evaluation_controls.tex").read_text(encoding="utf-8")))
    rows, per_file = [], {}
    for name in sorted(numbers):
        old = parse_macros((PRES / "figures" / name).read_text(encoding="utf-8"))
        nf = FIG549 / name
        new = parse_macros(nf.read_text(encoding="utf-8")) if nf.is_file() else {}
        counts = {}
        for m, ov in old.items():
            kind, why = classify(name, m, asm)
            nv = new.get(m)
            if kind == "assembled":
                st = "missing in 549 output" if nv is None else ("changed" if nv != ov else "unchanged")
                if nv is None and re.fullmatch(r"rq(One|Std)D[A-Z]\w*", m):
                    st, why = "pending", "this matcher is not on all 549 scenarios (left the panel; see Panels)"
                elif nv is None:
                    st, why = "pending", "not produced by the 549 run (that part of the input is not computed yet)"
            elif kind == "fixed":
                st = "fixed" if nv in (None, ov) else "FIXED BUT DIFFERENT"
                why += "" if nv is not None else " (not regenerated)"
            else:
                st = "pending"
                if nv is not None and nv != ov:
                    why += " [549 folder value differs: computed from 382 inputs or a mixed panel, not a 549 number]"
            tag = ("ladder " if m in ladder else "") + ("TableII " if m in table2 else "") + \
                  ("headline " if HEADLINE.fullmatch(m) else "")
            pre = "rqOne" if m.startswith("rqOne") else "rqStd" if m.startswith("rqStd") else None
            sp = same.get(pre, {}).get(m[5:]) if pre else None
            s381 = panel549.get(pre + "381", {}).get(m[5:]) if pre else None
            rows.append({"file": name, "macro": m, "old": ov, "new": nv if nv is not None else "",
                         "old_on_549_panel": sp or "", "new_on_old381": s381 or "", "status": st, "used": "yes" if m in used else "no",
                         "tags": tag.strip(), "reason": why})
            counts[st] = counts.get(st, 0) + 1
        for m in sorted(set(new) - set(old)):
            rows.append({"file": name, "macro": m, "old": "", "new": new[m], "old_on_549_panel": "",
                         "new_on_old381": "", "status": "new in 549 output", "used": "no", "tags": "", "reason": ""})
            counts["new in 549 output"] = counts.get("new in 549 output", 0) + 1
        per_file[name] = counts
    FIG549.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "macro", "old", "new", "old_on_549_panel", "new_on_old381", "status", "used", "tags", "reason"]
    for p in (OUT549 / "MACRO_STATUS.tsv", LOGDIR / "macro_status_382_549.tsv"):
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    tot = {}
    for r in rows:
        tot[r["status"]] = tot.get(r["status"], 0) + 1
    fams = asm["families"]
    md = [f"# Macro diff, Valentine 382 -> 549 ({datetime.now():%Y-%m-%d %H:%M}, sf-matcher-study/reference/run_549.py)", "",
          "Old: the active manuscript's macro files (latex/sf-evaluation-presentation-proposal/figures). New: "
          "latex/sf-evaluation-549/figures, written by the generators under SMX_VALENTINE=549 from the assembled inputs "
          "(sf-matcher-study/results/valentine549/ASSEMBLY.md). No paper .tex was edited. Full table also as "
          "latex/sf-evaluation-549/MACRO_STATUS.tsv.", "",
          "Status: **changed / unchanged** = input assembled on 549; **pending** = an input batch or a rerun step is "
          "missing (any value in the 549 folder is NOT a 549 number); **fixed** = registered or not Valentine, must stay "
          "identical. `old on 549 panel` = the 382 value recomputed for the matchers of the 549 panel (separates the "
          "panel change from the rest); `549 on old 381` = the 549 inputs restricted to the 381 old scenarios, same 549 "
          "panel (for rq1_std this isolates the FD fix of the standard configuration; SF as published (rq1) has no FD "
          "edges, so there it differs from `old on 549 panel` only by the dropped duplicate scenario).", "",
          "## Inputs", ""]
    for n, e in fams.items():
        arms = ", ".join(f"{k} {v['status']}" for k, v in e.get("arms", {}).items())
        md.append(f"- {n}: **{e['status']}**" + (f" ({arms})" if arms else "") +
                  (f"; configurations on all 549: {e['full']} of {e['configs']}" if "full" in e else ""))
    md += ["", "Verification of the 382 behaviour (generators run without the switch into a temporary folder):", ""]
    md += [f"- {m}" for m in verify] or ["- skipped (--no-verify)"]
    md += ["", "## Summary", "", "| status | macros |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in sorted(tot.items())]
    p382, p549 = same.get("panel382", {}), panel549
    md += ["", "## Panels (non-control matchers)", ""]
    for k, label in (("rq1", "RQ1 / Fig. (layer 1, arm melnik)"), ("std", "seed study, standard configuration")):
        a, b = set(p382.get(k, [])), set(p549.get(k, []))
        md.append(f"- {label}: 382 panel {len(a)}, 549 panel {len(b)}; left: {', '.join(sorted(a - b)) or '-'}; "
                  f"joined: {', '.join(sorted(b - a)) or '-'}")
    md += ["  (a matcher leaves when it lacks the 168 new scenarios or its replacement rows; see ASSEMBLY.md; "
           "valentine_distribution is dropped everywhere)"]
    for famname, arm, label in (("layer1_plain", "melnik", "RQ1"), ("seed_plain", None, "seed study")):
        part = fams.get(famname, {}).get("partial", {})
        items = [(k.split("|")[0], v) for k, v in part.items() if arm is None or k.endswith("|" + arm)]
        if arm and status_of(asm, "layer1nan_plain/melnik") == "assembled":   # X30: datatype's rows are datatype_nanfix's
            items = [(s, v) for s, v in items if s != "datatype"]
        if items:
            md.append(f"- not on all 549 yet ({label}; n = old381 + new168; joins once its batches land, unless a "
                      f"control): " + "; ".join(f"{s} {v['n']}={v['old381']}+{v['new168']}" for s, v in sorted(items)))
    md.append("")
    md += ["## Headline macros", "",
           "| macro | file | old (382) | old on 549 panel | 549 on old 381 | new (549) | status | tags |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["tags"] and (r["status"] in ("changed", "unchanged", "missing in 549 output") or "TableII" in r["tags"]
                          or "ladder" in r["tags"]):
            md.append(f"| `{r['macro']}` | {r['file']} | {r['old']} | {r['old_on_549_panel']} | {r['new_on_old381']} | "
                      f"{r['new'] if r['status'] != 'pending' else '—'} | "
                      f"{r['status']} | {r['tags']} |")
    md += ["", "## Per file", "", "| file | status counts | rule reason(s) |", "|---|---|---|"]
    for name, c in per_file.items():
        reasons = sorted({r["reason"] for r in rows if r["file"] == name and r["status"] != "new in 549 output"})
        md.append(f"| {name} | {', '.join(f'{k} {v}' for k, v in sorted(c.items()))} | {'; '.join(reasons)[:400]} |")
    md += ["", "## Every macro", ""]
    for name in per_file:
        md += [f"### {name}", "", "| macro | old | new | status | used | note |", "|---|---|---|---|---|---|"]
        for r in rows:
            if r["file"] == name:
                note = r["reason"] if r["status"] not in ("changed", "unchanged") else \
                    (f"old on 549 panel: {r['old_on_549_panel']}" if r["old_on_549_panel"] else "")
                md.append(f"| `{r['macro']}` | {r['old']} | {r['new'] if r['status'] != 'pending' else '—'} | "
                          f"{r['status']} | {r['used']} | {note} |")
        md.append("")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / "macro_diff_382_549.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[: md.index("## Per file")]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--same-panel", nargs=2, metavar=("PANEL_JSON", "OUT_JSON"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.same_panel:
        return same_panel_child(*a.same_panel)
    sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="run549_") as td:
        td = Path(td)
        verify = [] if a.no_verify else verify_382(td / "v382")
        print("\n".join(verify))
        print(run([PY, "-B", "reference/assemble_549.py"], STUDY)[-1500:])
        asm = json.loads(ASSEMBLY.read_text(encoding="utf-8"))
        env = {"SMX_VALENTINE": "549"}
        FIG549.mkdir(parents=True, exist_ok=True)
        # report scripts on the assembled rows (each prints "pending" and writes nothing while its inputs are missing)
        for r in ("plain", "md5"):
            print(run([PY, "-B", "explore_iterations.py", "--regime", r], STUDY, env)[:200])
            print(run([PY, "-B", "explore_cluster_batches.py", "--regime", r, "--batch", "gvfull"], STUDY, env)[-300:])
            print(run([PY, "-B", "layer3b/score_parallel.py", "--regime", r], STUDY, env)[-300:])
            deproles_549(r, env)
            n = {"plain": "plain549", "md5": "md549"}[r]
            l1 = asm_file(f"layer1_{n}_scores")
            res = STUDY / "results" / n
            # X46 rescoring (Table II's as-published column, drm), X50 extraction (xe), X58 family (fam)
            cached(f"x46_{r}", [PY, "-B", "explore_x46_fullmetrics.py", "--regime", r, "--workers", "14"], STUDY, env,
                   [l1, res / "deproles_summary.md", STUDY / "explore_x46_fullmetrics.py"],
                   [res / "x46_fullmetrics_summary.csv"])
            cached(f"x50_{r}", [PY, "-B", "explore_extraction.py", "--regime", r, "--workers", "14"], STUDY, env,
                   [l1, STUDY / "explore_extraction.py"], [res / "extraction_summary.md"])
            cached(f"x58_{r}", [PY, "-B", "layer3b/x58_family.py", "--regime", r], STUDY, env,
                   [l1, STUDY / "layer3b" / "x58_family.py"], [STUDY / "tuning" / "results" / f"x58_{n}" / "report.md"])
        # positional ground-truth survey (eaPos counts) and Valentine's benchmark properties (pr, rvConstr/rvOvl)
        cases = STUDY / "results" / "valentine549" / "cases.csv"
        cached("positional_survey", [PY, "-B", "reference/positional_survey_549.py"], STUDY, env,
               [cases, HERE / "positional_survey_549.py"], [STUDY / "results" / "valentine549" / "positional_survey.json"])
        cached("x81_props", [PY, "-B", "explore_x81_props.py"], STUDY, env, [cases, STUDY / "explore_x81_props.py"],
               [STUDY / "results" / "x81_549" / "benchmark_profile_x81.md"])
        # positional survey inputs for the 168 (local): Valentine COMA / Cupid matrices, the COMA rerun on permuted columns,
        # then the extractor-mode survey over the 549 (harness/positional_subsidy.py); X73 for readable levenshtein
        harn = ROOT / "harness"
        run168 = {**env, "SMX_POSITIONAL_RUN": "reports/valentine549_new168_run"}
        p168 = harn / "reports" / "valentine549_new168_run" / "protocol.json"
        vmdir = harn / "reports" / "valentine_matchers_549new168"
        for m in ("coma", "cupid"):
            cached(f"vm_{m}_168", [PY, "-B", "valentine_matchers_corpus.py", "--matcher", m, "--regime", "plain",
                                   "--matrices", f"reports/valentine_matchers_549new168/matrices/{m}_plain",
                                   "-o", f"reports/valentine_matchers_549new168/{m}_corpus_plain.csv"], harn, run168,
                   [p168, harn / "valentine_matchers_corpus.py"], [vmdir / f"{m}_corpus_plain.csv"])
        cached("positional_rerun_coma_168", [PY, "-B", "positional_subsidy.py", "--mode", "rerun", "--matcher", "coma",
                                             "--regime", "plain", "--json", "reports/positional_subsidy_549new168_rerun_coma.json"],
               harn, run168, [p168, harn / "positional_subsidy.py"],
               [harn / "reports" / "positional_subsidy_549new168_rerun_coma.json"])
        cached("positional_extractor_549", [PY, "-B", "positional_subsidy.py", "--mode", "extractor",
                                            "--json", "reports/positional_subsidy_549.json"], harn,
               {**env, "SMX_POSITIONAL_RUN": "reports/valentine549_run",
                "SMX_POSITIONAL_VM_EXTRA": "reports/valentine_matchers_549new168/matrices"},
               [harn / "reports" / "valentine549_run" / "protocol.json", vmdir / "coma_corpus_plain.csv",
                vmdir / "cupid_corpus_plain.csv", harn / "positional_subsidy.py"],
               [harn / "reports" / "positional_subsidy_549.json"])
        cached("x73_plain", [PY, "-B", "explore_matcherroles.py", "--regime", "plain", "--workers", "8"], STUDY, env,
               [asm_file("layer1_plain549_scores"), STUDY / "tuning" / "results" / "x62_pnew" / "COLLECTED",
                STUDY / "explore_matcherroles.py"], [STUDY / "results" / "plain549" / "matcherroles_summary.md"])
        # X90 (lit_none trajectories), X94 stage 1 (cells of the 168 + analysis), X95 (Valentine on the 549)
        tr = STUDY / "tuning" / "results"
        cached("x90", [PY, "-B", "layer3b/score_x90.py"], STUDY, env,
               [tr / "x90_valentine_pnew" / "COLLECTED", STUDY / "layer3b" / "score_x90.py"], [tr / "x90_549_scores" / "report.md"])
        cached("x94_cells", [PY, "-B", "layer3b/x94_stage1.py", "--compute"], STUDY, env,
               [cases, STUDY / "layer3b" / "x94_stage1.py"], [tr / "x94_549" / "cells.csv.gz"])
        cached("x94_analyse", [PY, "-B", "layer3b/x94_analyse.py"], STUDY, env,
               [tr / "x94_549" / "cells.csv.gz", STUDY / "layer3b" / "x94_analyse.py"], [STUDY / "results" / "x94_549" / "stage1.md"])
        cached("x95", [PY, "-B", "layer3b/score_x95.py", "--score"], STUDY, {**env, "PYTHONHASHSEED": "0"},
               [tr / "x95sf_valentine_pnew" / "COLLECTED", asm_file("layer1_plain549_scores"), asm_file("layer1_md549_scores"),
                STUDY / "layer3b" / "score_x95.py"], [tr / "x95_549" / "contrasts.csv"])
        # X78 effects on 549 / fixed FDs (Valentine, indep2, synthea; registered corpora stored) and the corpus associations
        x78 = STUDY / "results" / "x78_549"
        cached("benchmark_properties_549", [PY, "-B", "reference/benchmark_properties_549.py"], STUDY, env,
               [cases, HERE / "benchmark_properties_549.py"], [x78 / "benchmark_properties.csv"])
        x78in = [asm_file(f"{f}_{n}_scores") for f in ("layer1", "gvfull") for n in ("plain549", "md549")] + \
            [STUDY / "results" / n / "deproles_scores.csv" for n in ("plain549", "md549")] + \
            [STUDY / "results" / "multitable_fix" / c / "multitable_scores.csv" for c in ("indep2", "synthea")]
        cached("x78_effects", [PY, "-B", "layer3b/x78_effects.py"], STUDY, env, x78in + [STUDY / "layer3b" / "x78_effects.py"],
               [x78 / "benchmark_effects.csv"])
        cached("x78_analysis", [PY, "-B", "layer3b/x78_analysis.py"], STUDY, env,
               [x78 / "benchmark_effects.csv", x78 / "benchmark_properties.csv", STUDY / "layer3b" / "x78_analysis.py"],
               [x78 / "benchmark_effects_report.md"])
        cached("corpus_assoc", [PY, "-B", "review_checks/corpus_assoc.py"], STUDY, env,
               [x78 / "benchmark_effects.csv", x78 / "benchmark_properties.csv", STUDY / "review_checks" / "corpus_assoc.py"],
               [OUT549 / "review_checks" / "corpus_assoc_grid.csv"])
        # beta sweep (rvBeta*) on the Valentine part
        cached("beta_sweep", [PY, "-B", "review_checks/beta_sweep.py"], STUDY, env,
               [asm_file("layer1_plain549_scores"), asm_file("layer1_md549_scores"),
                STUDY / "review_checks" / "beta_sweep.py"], [OUT549 / "review_checks" / "beta_sweep.csv"])
        # external FD stages on the fixed HyFD profiles (FD arms from the *fix reruns; corpora without one stay pending)
        tres = STUDY / "tuning" / "results"
        if any((p / "COLLECTED").is_file() for p in tres.glob("x81fix_*")):
            cap = json.loads((STUDY / "results" / "hdxsm_fd_cap.json").read_text(encoding="utf-8"))["excluded"]
            print(run([PY, "-B", "explore_x81.py", "--corpora", "gdcsm,madi,goby,mimic,hdxsm",
                       "--exclude-fd", ",".join(f"hdxsm:{x}" for x in cap)], STUDY, env)[-300:])
        if any((p / "COLLECTED").is_file() for p in tres.glob("iterx81fix_*")):   # X82 (b): the fd trajectory
            print(run([PY, "-B", "explore_x82_bf.py"], STUDY, env)[-300:])
        for c in ("synthea", "indep2"):
            print(run([PY, "-B", "explore_multitable_ext.py", "--corpus", c], STUDY, env)[-300:])
        # literal term (X93: frozen lambdas, Valentine rows on 549; fixed hash seed so the cluster intervals reproduce)
        # and the two-update offset check over the graph exports (incl. valentine_pnew/mnew)
        x93in = [asm_file(f"{f}_{n}_scores") for f in ("layer1", "iter") for n in ("plain549", "md549")]
        cached("x93", [PY, "-B", "layer3b/score_x93.py", "--score"], STUDY, {**env, "PYTHONHASHSEED": "0"},
               x93in + [STUDY / "layer3b" / "score_x93.py"], [STUDY / "tuning" / "results" / "x93_549_scores" / "contrasts.csv"])
        cached("x93_mechanism", [PY, "-B", "layer3b/x93_mechanism.py"], STUDY, env,
               x93in + [STUDY / "tuning" / "results" / "x93_549_scores" / "rows.csv", STUDY / "layer3b" / "x93_mechanism.py"],
               [STUDY / "tuning" / "results" / "x93_549_scores" / "mechanism.md"])
        print(run([PY, "-B", "review_checks/two_update_offset_check.py"], STUDY, env)[-300:])
        run([PY, "-B", "make_tu_numbers.py"], PRES, env)
        run([PY, "-B", "gen_figures.py"], LONG, env, quiet=False)
        run([PY, "-B", "review_checks/rq1_artefact.py", "plain"], STUDY, env)
        run([PY, "-B", "review_checks/make_review_numbers.py"], STUDY, env)
        run([PY, "-B", "make_abs_numbers.py"], PRES, env)
        run([PY, "-B", "make_ea_numbers.py"], PRES, env)
        # source sensitivity (checks itself against the 549 rq1/review/dr macros written above)
        print(run([PY, "-B", "review_checks/source_sensitivity.py"], STUDY, env)[-200:])
        run([PY, "-B", "make_src_numbers.py"], PRES, env)
        # SemStruct / OmniMatch ablations on remaining505 (the learned-system agent's 549 reports; missing parts stay pending)
        for s in ("make_ss_numbers.py", "make_ssa_numbers.py", "make_om_numbers.py"):
            try_run([PY, "-B", s], PRES, env)
        relgain(FIG549 / "rq1_gain_data.csv", FIG549)
        pj, sj = td / "panel549.json", td / "same382.json"
        run([PY, "-B", Path(__file__), "--same-panel", pj, pj], STUDY, env)
        run([PY, "-B", Path(__file__), "--same-panel", pj, sj], STUDY)
        panel549 = json.loads(pj.read_text(encoding="utf-8"))
        same = json.loads(sj.read_text(encoding="utf-8"))
        (OUT549 / "README.md").write_text(
            "# Valentine 549 macros (generated; do not edit)\n\nWritten by sf-matcher-study/reference/run_549.py "
            f"({datetime.now():%Y-%m-%d %H:%M}). figures/ holds the generators' output under SMX_VALENTINE=549; only "
            "macros with status changed/unchanged in MACRO_STATUS.tsv are 549 numbers (pending ones may hold values "
            "computed from 382 inputs). The diff: harness/logs_rerun549/macro_diff_382_549.md.\n", encoding="utf-8")
        diff(asm, same, verify, panel549)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
