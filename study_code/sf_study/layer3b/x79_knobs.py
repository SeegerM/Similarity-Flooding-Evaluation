"""X79 (b) (EXPLORATORY; design and predictions in EXPLORATORY.md X79, written before any knob output): dose-response
of E1–E4 under controlled knobs (K1 column order, K2 values recoded, K3 names opaque, K4 FDs thinned), target side only.

Per variant scenario and matcher: J_tf (R_exp for E3) of q, SF melnik / fd / ucc_top1 (tuning/results/x79_knobs/<m>__<arm>),
and offline A_d = q⁺ + 5D, A_role = A_d + 0.2·R (explore_deproles' role similarity on the variant's own profiles).
Effects E1 = A_d − melnik, E2 = A_role − ucc_top1, E3 = fd − melnik, E4 = melnik − q; per variant the mean over
non-control matchers complete at base and variant; ΔE = variant − base per base scenario; dose-response = macro over
families (Valentine) or databases (final-controlled) of mean ΔE, 95% bootstrap over bases stratified by family.
Check first: on Valentine bases, A_d / A_role equal results/plain/deproles_scores and SF melnik equals layer 1.

    python layer3b/x79_knobs.py [--reuse] [--check-only]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STUDY))
import explore_deproles as dp  # noqa: E402
from score_r7 import is_control, metrics, run_dir  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from score_seeds import ground_truth_file, one  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

KN = STUDY / "data" / "knobs"
SEEDS = STUDY / "reference" / "seeds" / "knobs"
RUNS = STUDY / "tuning" / "results" / "x79_knobs"          # A, 20:55
OUT = STUDY / "tuning" / "results" / "x79_knobs_scores"   # kept apart from the run folders
ARMS = ("melnik", "fd", "ucc_top1")
GAMMA = 0.2
FC_DBS = {"AdventureWorks", "IMDB", "SCOPe", "Sakila"}
B_, SEED = 2000, 20261006


def profiles(ds, scv, side):
    """explore_deproles.profiles for an arbitrary scenario folder (same parsing, FD_RE is the fixed one)."""
    meta = KN / ds / scv / "metadata" / side
    fds, uccs, inds = [], [], []
    for tdir in sorted(p for p in meta.iterdir() if p.is_dir()) if meta.is_dir() else []:
        f = tdir / "FD_results.txt"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                m = dp.FD_RE.match(line.strip())
                if m:
                    fds.append((tuple(dp.col(x) for x in m.group("lhs").split(",") if x.strip()), dp.col(m.group("rhs"))))
        u = tdir / "UCC_results.txt"
        if u.is_file():
            for line in u.read_text(encoding="utf-8").splitlines():
                m = dp.BR_RE.match(line.strip())
                if m and m.group("cols").strip():
                    uccs.append(tuple(dp.col(x) for x in m.group("cols").split(",")))
    fi = meta / "inds.txt"
    if fi.is_file():
        for line in fi.read_text(encoding="utf-8").splitlines():
            m = dp.IND_RE.match(line.strip())
            if m and "," not in m.group("lhs") and "," not in m.group("rhs"):
                inds.append((dp.col(m.group("lhs")), dp.col(m.group("rhs"))))
    return fds, uccs, inds


def job(args):
    ds, scv, base, knob, matchers = args
    g = ground_truth_file(KN, ds, scv)
    if g is None:
        return []
    gt = read_matrix(g)[0] > 0.5
    # K1 permutes the target columns: map every matrix back to the base's column order by label, so the base's tie
    # orders below apply to the same columns (review). Other knobs keep positions (K3 renames without moving).
    # The truth files carry no labels (0/1 only), so the column labels come from the scenario's seed matrices, whose
    # rows and columns follow the same file order as the truth (fix 2026-10-07, before any X79 output was read).
    def seed_cols(v):
        for m in [*matchers, "datatype_nanfix"]:
            f = SEEDS / m / ds / v / "seed.csv"
            if f.is_file():
                return list(read_matrix(f)[2])
        return None
    gcols, bcols = seed_cols(scv), seed_cols(f"{base}__base")
    if gcols is None or len(gcols) != gt.shape[1]:
        return []

    def align(M, cols):
        """Reorder M's columns to the base's label order when its labels are a permutation of the base's."""
        cols = list(cols)
        if bcols is None or cols == bcols or sorted(cols) != sorted(bcols):
            return M, cols
        idx = [cols.index(c) for c in bcols]
        return M[:, idx], bcols
    gt, gcols = align(gt, gcols)
    perms = permutations(gt.shape, f"{ds}/{base}")         # the base's tie orders for every variant of it
    fs = profiles(ds, scv, "source"); ft = profiles(ds, scv, "target")
    Df = SEEDS / "datatype_nanfix" / ds / scv / "seed.csv"
    D = align(*read_matrix(Df)[::2])[0] if Df.is_file() else None
    rows = []
    for m in matchers:
        f = SEEDS / m / ds / scv / "seed.csv"
        if not f.is_file():
            continue
        q, rl, cl = read_matrix(f)
        q, cl = align(q, cl)
        if q.shape != gt.shape or not np.isfinite(q).all():
            continue
        rec = {"dataset": ds, "variant": scv, "base": base, "knob": knob, "seed": m}
        r = metrics(q, gt, perms); rec["J_q"], rec["R_q"] = r["J"], r["R_exp"]
        for arm in ARMS:
            d = run_dir(RUNS / f"{m}__{arm}")
            o = one(d / ds / scv / "_outputs" / "MatchingStepLine2") if d else None
            if o is None:
                continue
            S = align(*read_matrix(o)[::2])[0]
            if S.shape == gt.shape and np.isfinite(S).all():
                r = metrics(S, gt, perms); rec[f"J_{arm}"], rec[f"R_{arm}"] = r["J"], r["R_exp"]
        if D is not None and D.shape == q.shape and np.isfinite(D).all():
            Ad = np.clip(q, 0, None) + 5.0 * D
            rec["J_Ad"] = metrics(Ad, gt, perms)["J"]
            Fs, _ = dp.roles(rl, *fs); Ft, _ = dp.roles(cl, *ft)
            R = 1 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)
            rec["J_Arole"] = metrics(Ad + GAMMA * R, gt, perms)["J"]
        rows.append(rec)
    return rows


def family(ds):
    db = ds.split("-")[0]
    return ("final_controlled", db) if db in FC_DBS else ("valentine", "TPC-DI" if ds.startswith("TPC-DI") else db)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    matchers = sorted(p.name for p in SEEDS.iterdir() if p.is_dir() and p.name != "datatype_nanfix"
                      and not is_control(p.name) and p.name not in dp.CONTROLS)
    variants = []
    for d in sorted(p for p in KN.iterdir() if p.is_dir()):   # data/knobs also holds knobs_manifest.json
        for v in sorted(x for x in d.iterdir() if x.is_dir() and "__" in x.name):
            base, knob = v.name.rsplit("__", 1)
            variants.append((d.name, v.name, base, knob))
    OUT.mkdir(parents=True, exist_ok=True)
    if a.reuse:
        df = pd.read_csv(OUT / "scores.csv")
    else:
        todo = [(d, v, b, k, matchers) for d, v, b, k in variants if not a.check_only or k == "base"]
        with Pool(10) as pool:
            df = pd.DataFrame([r for part in pool.imap_unordered(job, todo) for r in part])
        if not a.check_only:
            df.to_csv(OUT / "scores.csv", index=False)
    # check: Valentine bases against deproles and layer 1
    base = df[df.knob == "base"].copy()
    base["corpus"] = [family(d)[0] for d in base.dataset]
    vb = base[base.corpus == "valentine"]
    dpr = pd.read_csv(STUDY / "results" / "plain" / "deproles_scores.csv").set_index(["seed", "dataset", "scenario"])
    l1 = pd.read_csv(STUDY / "tuning" / "results" / "layer1_plain_scores" / "scores.csv",
                     usecols=["seed", "arm", "dataset", "scenario", "sf_R_exp", "sf_F1_Hr"])
    l1 = l1[l1.arm == "melnik"].assign(J=lambda t: (t.sf_R_exp + t.sf_F1_Hr) / 2).set_index(["seed", "dataset", "scenario"]).J
    diffs = defaultdict(list)
    for _, r in vb.iterrows():
        k = (r.seed, r.dataset, r.base)
        if k in dpr.index:
            if pd.notna(r.get("J_Ad")): diffs["A_d vs deproles"].append(abs(r.J_Ad - dpr.loc[k, "A_0.0_0.0"]))
            if pd.notna(r.get("J_Arole")): diffs["A_role vs deproles"].append(abs(r.J_Arole - dpr.loc[k, "A_0.2_0.0"]))
        if k in l1.index and pd.notna(r.get("J_melnik")):
            diffs["SF melnik vs layer 1"].append(abs(r.J_melnik - l1.loc[k]))
    checks = [f"{k}: {len(v)} cells, max |diff| {max(v):.2e}" for k, v in diffs.items() if v]
    print("\n".join(checks))
    if a.check_only:
        return 0
    # effects per variant
    df["E1"] = df.J_Ad - df.J_melnik
    df["E2"] = df.J_Arole - df.J_ucc_top1
    df["E3R"] = df.R_fd - df.R_melnik
    df["E3J"] = df.J_fd - df.J_melnik
    df["E4"] = df.J_melnik - df.J_q
    effs = ["E1", "E2", "E3R", "E3J", "E4"]
    names = {"E1": "E1 datatype − SF", "E2": "E2 role − key", "E3R": "E3 fd − melnik (R_exp)", "E3J": "E3 fd − melnik (J)",
             "E4": "E4 SF − q"}
    b0 = df[df.knob == "base"].set_index(["dataset", "base", "seed"])[effs]
    rng = np.random.default_rng(SEED)
    lines = ["# X79 (b): controlled knobs, dose-response (EXPLORATORY)", "", *[f"- check {c}" for c in checks],
             f"- {df.base.nunique()} base scenarios, {len(matchers)} matchers; ΔE = variant − base, per base the mean over "
             "matchers complete at both; macro over families / databases; 95% bootstrap over bases within family.", ""]
    rows_out = []
    for corpus in ("valentine", "final_controlled"):
        sub = df[[family(d)[0] == corpus for d in df.dataset]]
        if sub.empty:
            continue
        lines += [f"## {corpus} bases ({sub.base.nunique()})", "", "| knob | " + " | ".join(names[e] for e in effs) + " |",
                  "|---|" + "---|" * len(effs)]
        lines.append("| base level | " + " | ".join(f"{sub[sub.knob == 'base'][e].mean():+.3f}" for e in effs) + " |")
        for knob in ["K1", "K2p25", "K2p50", "K2p75", "K2p100", "K3p25", "K3p50", "K3p75", "K3p100", "K4f50", "K4f25"]:
            v = sub[sub.knob == knob].set_index(["dataset", "base", "seed"])[effs]
            if v.empty:
                continue
            d = (v - b0.reindex(v.index)).dropna(how="all")
            per = d.groupby(level=["dataset", "base"]).mean().reset_index()
            per["fam"] = [family(x)[1] for x in per.dataset]
            cells = []
            for e in effs:
                x = per[["fam", e]].dropna()
                if x.empty:
                    cells.append("–"); continue
                groups = {f: g[e].to_numpy() for f, g in x.groupby("fam")}
                est = np.mean([g.mean() for g in groups.values()])
                bs = [np.mean([rng.choice(g, len(g)).mean() for g in groups.values()]) for _ in range(B_)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cells.append(f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                rows_out.append({"corpus": corpus, "knob": knob, "effect": e, "delta": est, "lo": lo, "hi": hi,
                                 "bases": len(x)})
            lines.append(f"| {knob} | " + " | ".join(cells) + " |")
        lines.append("")
    pd.DataFrame(rows_out).to_csv(OUT / "dose_response.csv", index=False)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
