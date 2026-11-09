"""Rerun 549 (2026-10-31): assemble the Valentine 549-scenario score files from the old 382-case results and the new
batches, per the rules table of harness/RERUN_549_LOG.md. Table-driven: every family lists its sources; a family (or
one arm of it) whose source batch is not scored yet is reported as PENDING and not written, never filled with old numbers.

Scenario base: the 382-case protocol has 381 distinct scenarios (TPC-DI prospect_horizontal_50_ec_av is listed under
Joinable and Unionable with identical tables; the Joinable copy is dropped), plus the 168 new scenarios of the official
release (harness/generated/valentine-new168, family from each manifest): 549. valentine_distribution is dropped
everywhere (the user's decision).

Outputs (new names only; the old folders are read, never written):
  results/plain549/, results/md549/           seed study: scores_tiefree.csv + summary_tiefree.csv (score_seeds_tiefree's
                                              summary code, verified byte-identical on the old 382 files)
  tuning/results/layer1_plain549_scores/ ...  layer-format families: scores.csv (+ summary.csv and report.md for layer 1,
                                              score_layer1's summary/report code, verified byte-identical on the old files)
  results/valentine549/                       cases.csv (549 scenarios, family, cohort), assembly.json (machine-readable
                                              status per family and arm, panel exclusions, checks), ASSEMBLY.md

Checks (any failure stops before anything is written): (1) the seed columns of every assembled row equal the old row
of the same (seed, scenario) exactly; (2) per configuration: 549 scenarios, else listed (configurations without the new
168 drop out of full-coverage panels, as under the 382 rule); (3) no scenario twice; families from the protocol (old)
and the manifests (new).

    ../harness/.venv-bdi/Scripts/python.exe -B reference/assemble_549.py [--selftest-only]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent
RES = STUDY / "results"
TRES = STUDY / "tuning" / "results"
OUTMETA = RES / "valentine549"
PROTOCOL = ROOT / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
NEW_TREE = {"plain": ROOT / "harness" / "generated" / "valentine-new168",
            "md5": ROOT / "harness" / "generated" / "valentine-new168-md5"}
COHORT = ROOT / "output" / "cluster-start-2026-10-30" / "cohort-549.json"

METRICS = ("F1_H", "F1_Hr", "F1_M", "R_pess", "R_exp", "MRR_pess", "MRR_exp")   # score_screen_metrics.METRICS
DUP = ("TPC-DI-Joinable", "prospect_horizontal_50_ec_av")   # the Joinable copy of the duplicate; Unionable stays
DROP_SEEDS = {"valentine_distribution"}
BASE = 549
NAME = {"plain": "plain549", "md5": "md549"}
NEWREG = {"plain": "pnew", "md5": "mnew"}
FIXREG = {"plain": "p382fix", "md5": "m382fix"}

# ------------------------------------------------------------------------------------------------ the rules table
# kind "seed": results/<r>/scores_tiefree.csv format. 381 part: seed columns from the old seed study, SF columns from
#              `sf381` (std_<x>382fix, the standard configuration rerun with the fixed FD profiles); seeds in `sf381`
#              without an old seed-study row (datatype_nanfix) take both from `sf381`, their seed columns checked against
#              `seedref` (melnik rows) instead. 168 part: `new` (seed-study format) plus the layer-format `new_extra`.
# kind "layer": score_layer1 format (seed, arm, dataset, scenario, family, seed_*, sf_*, stats). Per arm a list of
#              sources S(folder, arm in that folder, part "381"|"168", seeds=only these, exclude=not these,
#              optional=True: a missing optional batch does not make the arm pending; its seeds just lack those
#              scenarios and drop out of full-coverage panels). folder None = no batch named yet (arm pending).
#              `ref` is the old 382 folder whose rows the 381 seed columns are checked against.
# Routing (collecting session, 2026-10-31): datatype_nanfix lives in layer1nan_*, isresmat* in layer1isr_*, the FILE
# seeds computed locally for the 168 in the *fs_<new> batches; on the 381 every seed's FD arms come from the *fix
# batches and its non-FD arms keep the old 382 rows.


def S(folder, arm, part, seeds=None, exclude=None, optional=False) -> dict:
    return dict(folder=folder, arm=arm, part=part, seeds=seeds, exclude=exclude, optional=optional)


NAN, ISR = {"datatype_nanfix"}, {"isresmat", "isresmat_hybrid"}
FS_SEEDS = "learned_lofo, rema, rema_shuffled (possibly prisma, semstruct, semstruct_lm_only, sima_tpcdi)"
# PRISMA reads FD profiles: its old 382 initial matrices were computed on the old (unfixed) FDs
OLD_FD_INITIAL = {"prisma": "old-FD initial matrix: PRISMA's matrices on the 381 were computed from the old (unfixed) "
                            "FD profiles; no new 382 PRISMA matrices exist"}
NOT_REPRODUCIBLE = {s: "not reproducible: GPU training without a fixed seed (matrices differ by up to 0.033 per cell "
                       "between runs)" for s in ISR}
SIX = ("melnik", "fd", "attach_id", "coef_prod", "coef_prod_fd", "lit_none")
FAMILIES: list[dict] = []
for _r in ("plain", "md5"):
    _p, _n, _f = NAME[_r], NEWREG[_r], FIXREG[_r]
    _x = NAN | ISR   # never in the main layer-1 file (they have their own families, as in the 382 layout)
    FAMILIES += [
        dict(name=f"seed_{_r}", kind="seed", regime=_r, out=f"results/{_p}",
             old=f"results/{_r}/scores_tiefree.csv", sf381=(f"std_{_f}_scores", "std"), seedref=f"layer1nan_{_r}_scores",
             new=f"results/{_n}/scores_tiefree.csv",
             new_extra=[S(f"stdfs_{_n}_scores", "std", "168", optional=True),
                        S(f"stdnan_{_n}_scores", "std", "168", seeds=NAN, optional=True)],
             macros="rq1_std (Table II standard column), rq4, posthoc/penalty inputs",
             note=f"FILE seeds on the 168 ({FS_SEEDS}) from stdfs_{_n}; datatype_nanfix: 381 from std_{_f} (seed "
                  f"columns checked against layer1nan_{_r}), 168 from stdnan_{_n}; datatype_nanfix is not in "
                  "gen_tables.META, so no paper panel uses its seed-study rows"),
        dict(name=f"layer1_{_r}", kind="layer", regime=_r, out=f"layer1_{_p}_scores", ref=f"layer1_{_r}_scores",
             summary=True,
             arms={"melnik": [S(f"layer1_{_r}_scores", "melnik", "381", exclude=_x),
                              S(f"layer1_{_n}_scores", "melnik", "168", exclude=_x),
                              S(f"layer1fs_{_n}_scores", "melnik", "168", exclude=_x, optional=True)],
                   "fd": [S(f"l1fd_{_f}_scores", "fd", "381", exclude=_x),
                          S(f"layer1_{_n}_scores", "fd", "168", exclude=_x),
                          S(f"layer1fs_{_n}_scores", "fd", "168", exclude=_x, optional=True)]},
             macros="rq1, rqc, layer1_<r> (Ref, Fd rows), rv{Split,Head,Null,RelNull}, abs (Val), fam/lpFdVsRexp (fd)",
             note="arms attach_id, coef_prod, coef_prod_fd, lit_none are not rerun on 549 (long version only); "
                  f"FILE seeds on the 168 from layer1fs_{_n}"),
        dict(name=f"layer1nan_{_r}", kind="layer", regime=_r, out=f"layer1nan_{_p}_scores", ref=f"layer1nan_{_r}_scores",
             summary=True,
             arms={a: [S(f"l1fd_{_f}_scores" if a == "fd" else f"l1nanfix_{_f}_scores" if a == "coef_prod_fd"
                         else f"layer1nan_{_r}_scores", a, "381", seeds=NAN),
                       S(f"layer1nan_{_n}_scores", a, "168", seeds=NAN)] for a in SIX},
             macros="the datatype matcher's layer-1 rows (X30 replacement in gen_figures.layer1_rows / arm_rows)",
             note=f"datatype_nanfix: 381 fd from l1fd_{_f} (confirmed), coef_prod_fd from l1nanfix_{_f}, melnik, "
                  f"attach_id, coef_prod, lit_none keep the old rows; 168 from layer1nan_{_n}"),
        dict(name=f"layer1isr_{_r}", kind="layer", regime=_r, out=f"layer1isr_{_p}_scores", ref=f"layer1isr_{_r}_scores",
             summary=True,
             arms={a: [S(f"l1fdisr_{_f}_scores" if a == "fd" else f"l1cpfdisr_{_f}_scores" if a == "coef_prod_fd"
                         else f"layer1isr_{_r}_scores", a, "381", seeds=ISR),
                       S(f"layer1isr_{_n}_scores", a, "168", seeds=ISR)] for a in SIX},
             macros="none in the paper path (gen_figures does not read layer1isr); kept for completeness",
             note=f"isresmat/isresmat_hybrid: 381 fd from l1fdisr_{_f}, coef_prod_fd from l1cpfdisr_{_f} (log "
                  f"2026-10-31 18:20), the non-FD arms keep the old rows; 168 from layer1isr_{_n} (GPU host)"),
        dict(name=f"gvfull_{_r}", kind="layer", regime=_r, out=f"gvfull_{_p}_scores", ref=f"gvfull_{_r}_scores",
             summary=False,
             arms={"ucc_top1": [S(f"gvfull_{_r}_scores", "ucc_top1", "381"),
                                S(f"gvfull_{_n}_scores", "ucc_top1", "168"),
                                S(f"gvfullfs_{_n}_scores", "ucc_top1", "168", optional=True)],
                   **({"ucc1_fd_topk": [S(f"gvfull_{_f}_scores", "ucc1_fd_topk", "381"),
                                        S(f"gvfull_{_n}_scores", "ucc1_fd_topk", "168"),
                                        S(f"gvfullfs_{_n}_scores", "ucc1_fd_topk", "168", optional=True)]}
                      if _r == "plain" else {})},
             macros="dr (explore_deproles.py), rvBeta* (beta_sweep.py), drPlRoleVsUccFdTopk",
             note="report.md (explore_cluster_batches.py) and deproles must be regenerated from the assembled rows"),
        dict(name=f"iter_{_r}", kind="layer", regime=_r, out=f"iter_{_p}_scores", ref=f"iter_{_r}_scores", summary=False,
             # the converged reference "melnik": iter_<r>'s melnik rows are layer-1 melnik's (identical on all 33,560
             # plain rows, metrics and iterations, checked 2026-11-01), and iter_<new> runs only melnik_t*, so its 168
             # come from layer1_<new> (+ layer1fs_<new>; prismafs_<new> is added below)
             arms={a: [S(f"iter_{_r}_scores", a, "381"),
                       *([S(f"layer1_{_n}_scores", a, "168", exclude=NAN | ISR),
                          S(f"layer1fs_{_n}_scores", a, "168", exclude=NAN | ISR, optional=True)] if a == "melnik" else
                         [S(f"iter_{_n}_scores", a, "168"), S(f"iterfs_{_n}_scores", a, "168", optional=True)])]
                   for a in ("melnik", "melnik_t1", "melnik_t2", "melnik_t3", "melnik_t5", "melnik_t10")},
             macros="it, itr, ld, rlx",
             note="fd_t* are not rerun (long version only); report.md (explore_iterations.py) to be regenerated"),
        dict(name=f"parallel_{_r}", kind="layer", regime=_r, out=f"parallel_{_p}_scores", ref=f"parallel_{_r}_scores",
             summary=False,
             arms={"fd_split": [S(f"parallelfix_{_f}_scores", "fd_split", "381"),
                                S(f"parallel_{_n}_scores", "fd_split", "168"),
                                S(f"parallelfs_{_n}_scores", "fd_split", "168", optional=True)]},
             macros="pe", note="contrasts/report (score_parallel.py) to be regenerated"),
    ]
# PRISMA and the rema fill (2026-11-01). PRISMA's initial matrices were recomputed on the fixed-HyFD profiles (it reads
# FDs), so its 381 rows come ENTIRELY from prismafix_<fix> (seed and SF columns; the documented exception to check 1,
# replaced by an internal-consistency check: one seed matrix per scenario across all arms of the batch); its old rows
# (old FDs) are never used, and its 168 come from prismafs_<new>. rema / rema_shuffled: the old rows (non-FD arms) or the
# *fix batches (FD arms) for the 253 scenarios the old copy had, plus remafill_<fix> for the 129 it lacked (no old row
# exists for these, so they too get the internal-consistency check); their 168 come from the *fs_<new> batches.
PRISMA, REMA = {"prisma"}, {"rema", "rema_shuffled"}
INTERNAL_CHECK_BATCHES = ("prismafix_", "remafill_")
for _fam in FAMILIES:
    _n, _f = NEWREG[_fam["regime"]], FIXREG[_fam["regime"]]
    if _fam["kind"] == "seed":
        _fam["exclude"] = PRISMA
        _fam["extra381"] = [S(f"prismafix_{_f}_scores", "std", "381", seeds=PRISMA, optional=True),
                            S(f"remafill_{_f}_scores", "std", "381", seeds=REMA, optional=True)]
        for _s in _fam["new_extra"]:
            _s["exclude"] = (_s["exclude"] or set()) | PRISMA
        _fam["new_extra"].append(S(f"prismafs_{_n}_scores", "std", "168", seeds=PRISMA, optional=True))
    elif _fam["name"].split("_")[0] in ("layer1", "gvfull", "iter", "parallel"):
        for _arm, _srcs in _fam["arms"].items():
            for _s in _srcs:
                _s["exclude"] = (_s["exclude"] or set()) | PRISMA
            _srcs += [S(f"prismafix_{_f}_scores", _arm, "381", seeds=PRISMA, optional=True),
                      S(f"remafill_{_f}_scores", _arm, "381", seeds=REMA, optional=True),
                      S(f"prismafs_{_n}_scores", _arm, "168", seeds=PRISMA, optional=True)]
OLD_FD_INITIAL = {}   # PRISMA's old-FD rows are no longer used anywhere (see above)

FIXED_ON_382 = [
    ("tpass (timing pass)", "measured on the 382 scenarios as a separate descriptive cost measurement; not rerun on 549 "
                            "(the user's decision 2026-11-02; the paper states that timings were measured on the 382)"),
]
EXTERNAL_PENDING = [
    ("xt, wp, mt, cx", "x81_<c>, wide_<c>, iter_x81_<c>, x54/x55 FD arms", "*fix_<corpus> batches (non-Valentine corpora; "
     "they replace the FD arms in place and are not assembled here)"),
]


# ------------------------------------------------------------------------------------------------ helpers
def read(p: Path) -> list[dict]:
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def header(p: Path) -> list[str]:
    with open(p, encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def cases(regime: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """(dataset, scenario, family) of the 381 old and the 168 new scenarios."""
    old = [(c["dataset"], c["scenario"], c["family"])
           for c in json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"] if (c["dataset"], c["scenario"]) != DUP]
    root = NEW_TREE[regime]
    new = [(d.name, s.name, json.loads((s / "manifest.json").read_text(encoding="utf-8"))["family"])
           for d in sorted(root.iterdir()) if d.is_dir() for s in sorted(d.iterdir()) if s.is_dir()]
    return old, new


def key(r: dict) -> tuple[str, str]:
    return r["dataset"], r["scenario"]


def same_seed_cols(a: dict, b: dict) -> bool:
    return all(float(a[f"seed_{m}"]) == float(b[f"seed_{m}"]) for m in METRICS if f"seed_{m}" in a and f"seed_{m}" in b)


def write_csv(p: Path, rows: list[dict], fields: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")   # drops the _part/_src bookkeeping
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------- summaries: the original code paths, copied verbatim (verified)
def seed_summary(rows: list[dict]) -> list[dict]:
    """score_seeds_tiefree.main's summary loop (scenario means per initial matcher), unchanged."""
    summary = []
    for seed in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == seed]
        row = {"seed": seed, "n": len(rs)}
        for m in METRICS:
            row[f"seed_{m}"] = mean(r[f"seed_{m}"] for r in rs)
            row[f"sf_{m}"] = mean(r[f"sf_{m}"] for r in rs)
            row[f"delta_{m}"] = row[f"sf_{m}"] - row[f"seed_{m}"]
        summary.append(row)
    return summary


def _macro(rows: list[dict], k: str, MIN_FAMILY: int = 5) -> float:
    """score_layer1.macro, unchanged."""
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r[k])
    fams = [f for f, v in by.items() if len(v) >= MIN_FAMILY]
    return mean(mean(by[f]) for f in fams) if fams else float("nan")


def layer_summary_report(rows: list[dict], layer: int, regime: str, out_dir: Path) -> None:
    """score_layer1.main's paired summaries and report, unchanged (rows: metric columns as floats)."""
    sys.path.insert(0, str(STUDY))
    from score_seeds import CONTROLS  # noqa: E402
    ORACLES, MIN_FAMILY = ("oracle",), 5
    macro = _macro
    cell = defaultdict(dict)  # (seed, arm) -> {(ds, sc): row}
    for r in rows:
        cell[(r["seed"], r["arm"])][(r["dataset"], r["scenario"])] = r
    summary = []
    for (seed, arm), scen in sorted(cell.items()):
        ref = cell.get((seed, "melnik"), {})
        rs = list(scen.values())
        s = {"seed": seed, "arm": arm, "scenarios": len(rs)}
        for m in METRICS:
            for r in rs:
                r[f"delta_{m}"] = r[f"sf_{m}"] - r[f"seed_{m}"]
            s[f"seed_{m}"] = macro(rs, f"seed_{m}")
            s[f"sf_{m}"] = macro(rs, f"sf_{m}")
            s[f"delta_{m}"] = macro(rs, f"delta_{m}")
            paired = [dict(r, vs=r[f"sf_{m}"] - ref[k][f"sf_{m}"]) for k, r in scen.items() if k in ref]
            s[f"vs_melnik_{m}"] = macro(paired, "vs") if paired and arm != "melnik" else float("nan")
        its = [int(r["iterations"]) for r in rs if str(r["iterations"]).lstrip("-").isdigit()]
        s["mean_iterations"] = mean(its) if its else float("nan")
        s["converged_share"] = mean(r["converged"] == "1" for r in rs if r["converged"] != "") if any(r["converged"] != "" for r in rs) else float("nan")
        ms = [int(r["build_ms"]) + int(r["iterate_ms"]) for r in rs if str(r["build_ms"]).isdigit()]
        s["mean_sf_ms"] = mean(ms) if ms else float("nan")
        summary.append(s)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    n_full = max(s["scenarios"] for s in summary)
    arms = sorted({s["arm"] for s in summary}, key=lambda x: (x != "melnik", x))
    seeds = sorted({s["seed"] for s in summary if s["seed"] not in CONTROLS and not s["seed"].startswith(ORACLES)})
    full = [x for x in seeds if all(any(s["seed"] == x and s["arm"] == arm and s["scenarios"] == n_full for s in summary)
                                    for arm in arms)]
    lines = [f"# Layer {layer} ({regime}): {len(rows)} scored (seed, arm, scenario) triples", "",
             f"Means over {len(full)} non-control initial matchers with every arm on all {n_full} scenarios; "
             f"each value is the macro mean over families with >= {MIN_FAMILY} scenarios.", "",
             "| arm | " + " | ".join(f"Δ{m} (SF−seed)" for m in ("F1_M", "F1_Hr", "R_exp", "MRR_exp")) +
             " | " + " | ".join(f"vs melnik {m}" for m in ("F1_M", "F1_Hr", "R_exp")) + " | iterations | converged | SF ms |",
             "|---|" + "---|" * 10]
    for arm in arms:
        ss = [s for s in summary if s["arm"] == arm and s["seed"] in full]
        if not ss:
            continue
        f = lambda k: mean(s[k] for s in ss)  # noqa: E731
        lines.append(f"| {arm} | " + " | ".join(f"{f('delta_' + m):+.3f}" for m in ("F1_M", "F1_Hr", "R_exp", "MRR_exp")) +
                     " | " + " | ".join("—" if arm == "melnik" else f"{f('vs_melnik_' + m):+.3f}" for m in ("F1_M", "F1_Hr", "R_exp")) +
                     f" | {f('mean_iterations'):.1f} | {f('converged_share'):.0%} | {f('mean_sf_ms'):.0f} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def floats(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        for m in METRICS:
            for p in ("seed", "sf"):
                if f"{p}_{m}" in r:
                    r[f"{p}_{m}"] = float(r[f"{p}_{m}"])
        out.append(r)
    return out


def selftest(tmp: Path) -> list[str]:
    """The copied summary/report code reproduces the old files byte for byte."""
    msgs = []
    tmp.mkdir(parents=True, exist_ok=True)
    for r in ("plain", "md5"):
        rows = floats(read(RES / r / "scores_tiefree.csv"))
        write_csv(tmp / f"summary_tiefree_{r}.csv", (s := seed_summary(rows)), list(s[0]))
        ok = (tmp / f"summary_tiefree_{r}.csv").read_bytes() == (RES / r / "summary_tiefree.csv").read_bytes()
        msgs.append(f"seed-study summary_tiefree.csv ({r}) reproduced byte-identically: {ok}")
        if not ok:
            raise SystemExit(msgs[-1])
    for folder in ("layer1_plain_scores", "layer1nan_plain_scores"):
        d = tmp / folder
        d.mkdir(exist_ok=True)
        rows = floats(read(TRES / folder / "scores.csv"))
        layer_summary_report(rows, 1, folder.split("_")[1], d)
        for f in ("summary.csv", "report.md"):
            ok = (d / f).read_bytes() == (TRES / folder / f).read_bytes()
            msgs.append(f"{folder}/{f} reproduced byte-identically: {ok}")
            if not ok:
                raise SystemExit(msgs[-1])
    return msgs


# ------------------------------------------------------------------------------------------------ assembly
class Fail(Exception):
    pass


def coverage(rows: list[dict], by: tuple[str, ...]) -> dict:
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        c[tuple(r[k] for k in by)][0 if r["_part"] == "381" else 1] += 1
    return c


_CACHE: dict[Path, list[dict]] = {}


def cached(p: Path) -> list[dict]:
    if p not in _CACHE:
        _CACHE[p] = read(p)
    return _CACHE[p]


INTERNAL: dict[str, str] = {}


def internal_check(folder: str) -> None:
    """Batches whose 381 seed columns have no old row to be checked against (PRISMA on fixed FDs, the rema fill): every
    (seed, scenario) must carry one and the same initial matrix's metrics across all arms of the batch."""
    if folder in INTERNAL:
        return
    by = defaultdict(list)
    for r in cached(TRES / folder / "scores.csv"):
        by[(r["seed"],) + key(r)].append(r)
    bad = [k for k, rs in by.items() if any(not same_seed_cols(rs[0], x) for x in rs[1:])]
    if bad:
        raise Fail(f"{folder}: seed columns differ across arms for {len(bad)} (seed, scenario), e.g. {bad[:3]}")
    arms = {r["arm"] for rs in by.values() for r in rs}
    INTERNAL[folder] = f"{len(by)} (seed, scenario) with identical seed columns across {len(arms)} arms"


def take(src: dict, target_arm: str, allowed: dict, famname: str) -> list[dict]:
    """Rows of one layer-format source: its arm, its seed filter, the scenarios of its part (the duplicate's Joinable
    copy and anything outside the part are skipped), families checked against the protocol / manifests."""
    if src["folder"].startswith(INTERNAL_CHECK_BATCHES):
        internal_check(src["folder"])
    out = []
    for r in cached(TRES / src["folder"] / "scores.csv"):
        s = r["seed"]
        if r.get("arm") != src["arm"] or s in DROP_SEEDS or (src["seeds"] and s not in src["seeds"]) \
                or (src["exclude"] and s in src["exclude"]):
            continue
        fam_of = allowed[src["part"]].get(key(r))
        if fam_of is None:
            continue
        if r.get("family") not in (None, "", fam_of):
            raise Fail(f"{famname}/{target_arm}: family {r.get('family')} != {fam_of} for {key(r)} in {src['folder']}")
        out.append({**r, "arm": target_arm, "_part": src["part"], "_src": src["folder"]})
    return out


def flags_for(seeds: set[str]) -> dict[str, str]:
    return {s: n for s, n in {**OLD_FD_INITIAL, **NOT_REPRODUCIBLE}.items() if s in seeds}


def assemble_seed(fam: dict, old_cases, new_cases) -> dict:
    old_f, new_f = STUDY / fam["old"], STUDY / fam["new"]
    sf_f = TRES / fam["sf381"][0] / "scores.csv"
    missing = [str(p.relative_to(ROOT)) for p in (old_f, new_f, sf_f) if not p.is_file()]
    if missing:
        return {"status": "pending", "missing": missing}
    allowed = {"381": {(d, s): f for d, s, f in old_cases}, "168": {(d, s): f for d, s, f in new_cases}}
    fields = header(old_f)
    skip = DROP_SEEDS | fam.get("exclude", set())   # PRISMA: its rows come only from prismafix / prismafs
    old = [r for r in read(old_f) if key(r) in allowed["381"] and r["seed"] not in skip]
    std = {(r["seed"],) + key(r): r for r in read(sf_f)
           if r["arm"] == fam["sf381"][1] and key(r) in allowed["381"] and r["seed"] not in skip}
    new = [r for r in read(new_f) if r["seed"] not in skip]
    bad_new = [key(r) for r in new if key(r) not in allowed["168"]]
    if bad_new:
        raise Fail(f"{fam['name']}: {len(bad_new)} new rows outside the 168 cases, e.g. {bad_new[:3]}")
    rows, no_std, seed_mismatch = [], defaultdict(int), []
    for r in old:
        s = std.get((r["seed"],) + key(r))
        if s is None:
            no_std[r["seed"]] += 1
            continue
        if not same_seed_cols(r, s):
            seed_mismatch.append((r["seed"],) + key(r))
            continue
        rows.append({**{k: r[k] for k in fields if not k.startswith("sf_")},
                     **{k: s[k] for k in fields if k.startswith("sf_")}, "_part": "381"})
    # seeds of the std batch without an old seed-study row (datatype_nanfix): both column sets from the std batch,
    # the seed columns checked against the old layer-1 rows of `seedref`
    # only datatype_nanfix is admitted this way; any other seed missing from the old seed study (fuzzy_jaccard on md5:
    # its seed-study scoring predates the matcher) was not in the 382 panel and stays out of the 549 one as well
    old_seeds = {r["seed"] for r in old}
    std_only = sorted(({k[0] for k in std} - old_seeds) & NAN)
    not_in_382 = sorted({k[0] for k in std} - old_seeds - NAN)
    if not_in_382:
        print(f"{fam['name']}: not in the 382 seed-study panel, left out: {', '.join(not_in_382)}")
        rows_before = len(rows)
        rows = [r for r in rows if r["seed"] not in not_in_382]
        assert len(rows) == rows_before
        new = [r for r in new if r["seed"] not in not_in_382]
    ref_f = TRES / fam["seedref"] / "scores.csv"
    ref = {(r["seed"],) + key(r): r for r in cached(ref_f) if r["arm"] == "melnik"} if ref_f.is_file() else {}
    std_only_checked = 0
    for k, s in std.items():
        if k[0] in std_only:
            o = ref.get(k)
            if o is None:
                raise Fail(f"{fam['name']}: {k} has no old seed-study row and no {fam['seedref']} row to check against")
            if not same_seed_cols(s, o):
                seed_mismatch.append(k)
                continue
            std_only_checked += 1
            rows.append({**{f: s[f] for f in fields}, "_part": "381"})
    if seed_mismatch:
        raise Fail(f"{fam['name']}: check 1 failed, {len(seed_mismatch)} rows whose seed columns differ from the old "
                   f"rows, e.g. {seed_mismatch[:3]}")
    rows += [{**{k: r[k] for k in fields}, "_part": "168"} for r in new]
    extra = {}
    for src in fam.get("extra381", []):   # both column sets from the batch (internal-consistency check in take)
        f = TRES / src["folder"] / "scores.csv"
        if not f.is_file():
            extra[src["folder"]] = "pending (optional: its seeds lack these 381 scenarios until it lands)"
            continue
        got = take(src, "std", allowed, fam["name"])
        rows += [{**{k: r[k] for k in fields}, "_part": "381"} for r in got]
        extra[src["folder"]] = f"assembled ({len(got)} rows, seeds {', '.join(sorted({r['seed'] for r in got}))})"
    for src in fam["new_extra"]:
        f = TRES / src["folder"] / "scores.csv"
        if not f.is_file():
            extra[src["folder"]] = "pending (optional: its seeds lack the 168 until it lands)"
            continue
        got = take(src, "std", allowed, fam["name"])
        rows += [{**{k: r[k] for k in fields}, "_part": "168"} for r in got]
        extra[src["folder"]] = f"assembled ({len(got)} rows, seeds {', '.join(sorted({r['seed'] for r in got}))})"
    seen = set()
    for r in rows:
        k = (r["seed"],) + key(r)
        if k in seen:
            raise Fail(f"{fam['name']}: check 3 failed, scenario twice: {k}")
        seen.add(k)
    rows.sort(key=lambda r: (r["seed"], r["dataset"], r["scenario"]))
    cov = coverage(rows, ("seed",))
    partial = {s[0]: {"n": a + b, "old381": a, "new168": b} for s, (a, b) in cov.items() if a + b != BASE}
    return {"status": "assembled", "rows": rows, "fields": fields, "out": STUDY / fam["out"],
            "configs": len(cov), "full": sum(1 for v in cov.values() if sum(v) == BASE), "partial": partial,
            "old_without_std": dict(no_std), "std_only_seeds": std_only, "optional_sources": extra,
            "flags": flags_for({r["seed"] for r in rows}),
            "checks": {"seed_columns_equal_old": sum(1 for r in rows if r["_part"] == "381") - std_only_checked,
                       "std_only_seed_columns_equal_seedref": std_only_checked, "seed_column_mismatches": 0,
                       "duplicates": 0}}


def assemble_layer(fam: dict, old_cases, new_cases) -> dict:
    allowed = {"381": {(d, s): f for d, s, f in old_cases}, "168": {(d, s): f for d, s, f in new_cases}}
    ref_f = TRES / fam["ref"] / "scores.csv"
    arm_status, rows, fields = {}, [], None
    ref_idx = None
    for arm, srcs in fam["arms"].items():
        req = [s for s in srcs if not s["optional"]]
        miss = [f"{s['part']}: no batch named yet" if s["folder"] is None
                else str((TRES / s["folder"] / "scores.csv").relative_to(ROOT))
                for s in req if s["folder"] is None or not (TRES / s["folder"] / "scores.csv").is_file()]
        if miss:
            arm_status[arm] = {"status": "pending", "missing": miss}
            continue
        part_rows, empty, optional = [], [], {}
        for s in srcs:
            if s["optional"] and not (TRES / s["folder"] / "scores.csv").is_file():
                optional[s["folder"]] = "pending (optional: its seeds lack these scenarios until it lands)"
                continue
            got = take(s, arm, allowed, fam["name"])
            if s["optional"]:
                optional[s["folder"]] = f"assembled ({len(got)} rows)"
            elif not got:
                empty.append(f"{s['folder']}: no rows of arm {s['arm']}")
            part_rows += got
        if empty:
            arm_status[arm] = {"status": "pending", "missing": empty}
            continue
        fields = fields or header(TRES / req[0]["folder"] / "scores.csv")
        # check 1: seed columns of every 381 row equal the old 382 row (same arm, else melnik) of that (seed, scenario)
        if ref_idx is None and ref_f.is_file():
            ref_idx = defaultdict(dict)
            for r in cached(ref_f):
                ref_idx[(r["seed"],) + key(r)][r["arm"]] = r
        mism, checked = [], 0
        for r in part_rows:
            if r["_part"] != "381" or ref_idx is None or not any(k.startswith("seed_") for k in r) \
                    or r["_src"].startswith(INTERNAL_CHECK_BATCHES):   # documented exception: internal check instead
                continue
            o = ref_idx.get((r["seed"],) + key(r), {})
            o = o.get(arm) or o.get("melnik") or next(iter(o.values()), None)
            if o is None:
                continue
            checked += 1
            if not same_seed_cols(r, o):
                mism.append((r["seed"],) + key(r))
        if mism:
            raise Fail(f"{fam['name']}/{arm}: check 1 failed, {len(mism)} rows whose seed columns differ from "
                       f"{fam['ref']}, e.g. {mism[:3]}")
        arm_status[arm] = {"status": "assembled", "sources": [s["folder"] for s in req], "seed_cols_checked": checked,
                           **({"optional": optional} if optional else {})}
        rows += part_rows
    if not rows:
        return {"status": "pending", "arms": arm_status}
    seen = set()
    for r in rows:
        k = (r["seed"], r["arm"]) + key(r)
        if k in seen:
            raise Fail(f"{fam['name']}: check 3 failed, scenario twice: {k}")
        seen.add(k)
    rows.sort(key=lambda r: (r["seed"], r["arm"], r["dataset"], r["scenario"]))
    cov = coverage(rows, ("seed", "arm"))
    partial = {f"{s}|{a}": {"n": x + y, "old381": x, "new168": y} for (s, a), (x, y) in cov.items() if x + y != BASE}
    status = "assembled" if all(v["status"] == "assembled" for v in arm_status.values()) else "partial"
    return {"status": status, "arms": arm_status, "rows": rows, "fields": fields, "out": TRES / fam["out"],
            "configs": len(cov), "full": sum(1 for v in cov.values() if sum(v) == BASE), "partial": partial,
            "flags": flags_for({r["seed"] for r in rows})}


def cross_check_new(results: dict) -> list[str]:
    """Information: seed columns of the 168 layer-1 rows against the seed study's new rows (same matrices expected)."""
    out = []
    for r in ("plain", "md5"):
        a, b = results.get(f"seed_{r}"), results.get(f"layer1_{r}")
        if not a or not b or "rows" not in a or "rows" not in b:
            continue
        idx = {(x["seed"],) + key(x): x for x in a["rows"] if x["_part"] == "168"}
        n = bad = 0
        for x in b["rows"]:
            if x["_part"] == "168" and x["arm"] == "melnik":
                y = idx.get((x["seed"],) + key(x))
                if y is not None:
                    n += 1
                    bad += not same_seed_cols(x, y)
        out.append(f"{r}: seed columns of layer-1 melnik rows on the 168 vs the seed study's new rows: {n} compared, "
                   f"{bad} differ")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--tmp", default=str(ROOT / "harness" / "logs_rerun549" / "assemble_selftest"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    log = selftest(Path(a.tmp))
    print("\n".join(log))
    if a.selftest_only:
        return 0
    old_cases, new_cases = cases("plain")
    old_m, new_m = cases("md5")
    assert [c[:2] for c in old_cases] == [c[:2] for c in old_m] and len(old_cases) == 381, len(old_cases)
    assert {c[:2] for c in new_cases} == {c[:2] for c in new_m} and len(new_cases) == 168, len(new_cases)
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    assert cohort["valid"] == BASE == len(old_cases) + len(new_cases)
    assert {tuple(c["path"].split("/")) for c in cohort["new_cases"]} == {c[:2] for c in new_cases}
    assert {tuple(c["path"].split("/")) for c in cohort["prior_cases"]} == {c[:2] for c in old_cases}
    results, errors = {}, []
    for fam in FAMILIES:
        oc, nc = (old_cases, new_cases) if fam["regime"] == "plain" else (old_m, new_m)
        try:
            results[fam["name"]] = (assemble_seed if fam["kind"] == "seed" else assemble_layer)(fam, oc, nc)
        except Fail as e:
            errors.append(str(e))
    if errors:
        print("\n".join(errors))
        raise SystemExit("assembly checks failed; nothing written")
    xcheck = cross_check_new(results)

    # write
    OUTMETA.mkdir(parents=True, exist_ok=True)
    write_csv(OUTMETA / "cases.csv",
              [{"dataset": d, "scenario": s, "family": f, "cohort": "old381"} for d, s, f in old_cases] +
              [{"dataset": d, "scenario": s, "family": f, "cohort": "new168"} for d, s, f in new_cases],
              ["dataset", "scenario", "family", "cohort"])
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "base": BASE, "dropped_duplicate": list(DUP),
                "dropped_seeds": sorted(DROP_SEEDS), "selftest": log, "cross_check_new168": xcheck,
                "external_pending": EXTERNAL_PENDING, "families": {}}
    for fam in FAMILIES:
        res = results[fam["name"]]
        entry = {k: v for k, v in res.items() if k not in ("rows", "fields", "out")}
        entry.update({"kind": fam["kind"], "regime": fam["regime"], "out": fam["out"], "macros": fam["macros"],
                      "note": fam["note"]})
        if "rows" in res:
            rows, out = res["rows"], res["out"]
            out.mkdir(parents=True, exist_ok=True)
            if fam["kind"] == "seed":
                write_csv(out / "scores_tiefree.csv", rows, res["fields"])
                s = seed_summary(floats(rows))
                write_csv(out / "summary_tiefree.csv", s, list(s[0]))
                entry["written"] = [f"{fam['out']}/scores_tiefree.csv", f"{fam['out']}/summary_tiefree.csv"]
            else:
                write_csv(out / "scores.csv", rows, res["fields"])
                entry["written"] = [f"tuning/results/{fam['out']}/scores.csv"]
                if fam["summary"]:
                    layer_summary_report(floats(rows), 1, NAME[fam["regime"]], out)
                    entry["written"] += [f"tuning/results/{fam['out']}/{f}" for f in ("summary.csv", "report.md")]
            (out / "ASSEMBLED_549").write_text(json.dumps({k: entry[k] for k in ("status", "out") if k in entry}
                                                          | {"arms": {k: v["status"] for k, v in entry.get("arms", {}).items()},
                                                             "by": "reference/assemble_549.py",
                                                             "created": manifest["created"]}, indent=1),
                                               encoding="utf-8")
        else:
            for stale in ([STUDY / fam["out"]] if fam["kind"] == "seed" else [TRES / fam["out"]]):
                if (stale / "ASSEMBLED_549").is_file():   # an earlier assembly that is no longer valid
                    raise SystemExit(f"{stale} holds an earlier assembly but its sources are now missing; check")
        manifest["families"][fam["name"]] = entry
    manifest["internal_checks"] = INTERNAL
    (OUTMETA / "assembly.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")

    md = [f"# Valentine 549 assembly ({manifest['created']}, reference/assemble_549.py)", "",
          f"Base {BASE} = 381 old (the Joinable copy of {DUP[1]} dropped) + 168 new; dropped seeds: "
          f"{', '.join(sorted(DROP_SEEDS))}. Self-test: " + "; ".join(log), "",
          "| family | status | arms | configs (full 549) | not on 549 (n = old381 + new168) | output |", "|---|---|---|---|---|---|"]
    for name, e in manifest["families"].items():
        arms = ", ".join(f"{k}: {v['status']}" for k, v in e.get("arms", {}).items()) or "-"
        part = "; ".join(f"{k} {v['n']}={v['old381']}+{v['new168']}" for k, v in sorted(e.get("partial", {}).items()))
        md.append(f"| {name} | {e['status']} | {arms} | {e.get('configs', '-')} ({e.get('full', '-')}) | {part or '-'} | "
                  f"{e['out'] if 'written' in e else '-'} |")
    md += ["", "Pending sources:", ""]
    for name, e in manifest["families"].items():
        for arm, v in (e.get("arms", {}) or {"": e}).items():
            if v.get("status") == "pending":
                md.append(f"- {name}{'/' + arm if arm else ''}: missing {', '.join(v['missing'])}")
            for folder, st in (v.get("optional") or {}).items():
                if st.startswith("pending"):
                    md.append(f"- {name}{'/' + arm if arm else ''}: optional {folder} {st}")
        for folder, st in (e.get("optional_sources") or {}).items():
            if st.startswith("pending"):
                md.append(f"- {name}: optional {folder} {st}")
    for macros, runs, src in EXTERNAL_PENDING:
        md.append(f"- {macros} ({runs}): {src}")
    md += ["", "Flags on assembled rows (by seed):", ""]
    flagged = {(s, n) for e in manifest["families"].values() for s, n in (e.get("flags") or {}).items()}
    md += [f"- {s}: {n}" for s, n in sorted(flagged)] or ["- none in the assembled families"]
    md += [f"- {s} (when its rows land): {n}" for s, n in sorted({**OLD_FD_INITIAL, **NOT_REPRODUCIBLE}.items())
           if s not in {x for x, _ in flagged}]
    md += ["", "Check 1 exception (PRISMA on fixed FDs, rema fill: no old row to compare with), internal consistency:", ""]
    md += [f"- {k}: {v}" for k, v in sorted(INTERNAL.items())] or ["- no such batch used yet"]
    md += ["- PRISMA: rows only from prismafix_* (381) and prismafs_* (168); its old rows (old FD profiles; md5 copies "
           "equal to plain by a feature-cache bug) are never used. rema/rema_shuffled: old or *fix rows for the 253 old "
           "scenarios, remafill_* for the 129 the old copy lacked, *fs_* on the 168."]
    md += ["", "Kept on the 382 by decision:", ""] + [f"- {k}: {v}" for k, v in FIXED_ON_382]
    md += ["", "Notes:", ""] + [f"- {n}: {e['note']}" for n, e in manifest["families"].items() if e["note"]]
    md += [f"- {x}" for x in xcheck]
    for name, e in manifest["families"].items():
        if e.get("old_without_std"):
            md.append(f"- {name}: old seed-study rows without a std_*382fix row (dropped): "
                      + ", ".join(f"{k} {v}" for k, v in sorted(e['old_without_std'].items())))
        if e.get("std_only_seeds"):
            md.append(f"- {name}: seeds only in the std batch (no old seed-study row; both column sets from the std "
                      "batch, seed columns checked against the old layer-1 rows): " + ", ".join(e["std_only_seeds"]))
        for folder, st in (e.get("optional_sources") or {}).items():
            if not st.startswith("pending"):
                md.append(f"- {name}: {folder} {st}")
    (OUTMETA / "ASSEMBLY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
