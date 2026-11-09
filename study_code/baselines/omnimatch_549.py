"""OMNIMATCH_ABLATION.md, Amendment 549: drive omnimatch_ablation.run_all on the 549-scenario case list.

    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_549.py check            # 5 old scenarios x seeds 1-3,
                                                                                    # byte-compared with the stored ones
    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_549.py reuse            # copy the stored 381 x 3 x 4
    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_549.py run --shards 4 --shard k [--all]
    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_ablation.py score --out reports/omnimatch_ablation_549
        (with SEMSTRUCT_RUN=reports/valentine549_protocol; report: omnimatch_ablation.py report ... --cohort remaining505)

Options, arms and seeds exactly as the 382 run (omnimatch_ablation.OPTIONS, arms F N A RF, seeds 1 2 3).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SEMSTRUCT_RUN", str(HERE / "reports" / "valentine549_protocol"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import omnimatch_ablation as oa  # noqa: E402

OLD = HERE / "reports" / "omnimatch_ablation_20261027"
OUT = HERE / "reports" / "omnimatch_ablation_549"
ARMS, SEEDS = ["F", "N", "A", "RF"], [1, 2, 3]


def cases() -> list[dict]:
    cs = oa.sr.load_cases()
    assert len(cs) == 549 and Path(oa.sr.RUN).name == "valentine549_protocol", oa.sr.RUN
    return cs


def run_cases(selected: list[dict], out: Path, force: bool, log) -> None:
    for case in selected:
        ctx = oa.context(case)
        if ctx is None:
            log(f"MISSING {case['id']}")
            continue
        for seed in SEEDS:
            todo = [a for a in ARMS if force or not (out / "matrices" / f"{a}__s{seed}" / f"{case['key']}.csv").exists()]
            if not todo:
                continue
            t0 = time.perf_counter()
            res = oa.run_all({k: ctx[k] for k in ("rt", "ds", "sc", "rows", "cols")}, dict(oa.OPTIONS, seed=seed), todo)
            for arm in todo:
                oa.write_matrix(out / "matrices" / f"{arm}__s{seed}" / f"{case['key']}.csv", res[arm])
            log(f"{case['id']}  {'/'.join(todo)} s{seed}  {time.perf_counter() - t0:.1f}s  gamma {res.get('A_gamma')}")


def command_check(args) -> int:
    old = [c for c in cases() if c["cohort382"] != "new168"]
    rng = np.random.default_rng(20261031)
    pick = [old[i] for i in sorted(rng.choice(len(old), 5, replace=False))]
    scratch = Path(args.scratch)
    run_cases(pick, scratch, True, lambda s: print(s, flush=True))
    same = total = 0
    for c in pick:
        for seed in SEEDS:
            for arm in ARMS:
                a = scratch / "matrices" / f"{arm}__s{seed}" / f"{c['key']}.csv"
                b = OLD / "matrices" / f"{arm}__s{seed}" / f"{c['key']}.csv"
                total += 1
                ok = a.read_bytes() == b.read_bytes()
                same += ok
                if not ok:
                    print("DIFF", c["id"], arm, seed)
    print(f"CHECK {same}/{total} byte-identical", "OK" if same == total else "MISMATCH")
    return 0 if same == total else 1


def command_reuse(args) -> int:
    n = 0
    for c in cases():
        if c["cohort382"] == "new168":
            continue
        for seed in SEEDS:
            for arm in ARMS:
                src = OLD / "matrices" / f"{arm}__s{seed}" / f"{c['key']}.csv"
                dst = OUT / "matrices" / f"{arm}__s{seed}" / f"{c['key']}.csv"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copyfile(src, dst)
                n += 1
    print(f"{n} stored matrices in place (381 x 3 seeds x 4 arms = {381 * 12})")
    return 0


def command_run(args) -> int:
    cs = [c for c in cases() if args.all or c["cohort382"] == "new168"]
    cs = [c for i, c in enumerate(cs) if i % args.shards == args.shard]
    run_cases(cs, OUT, False, lambda s: print(s, flush=True))
    print("SHARD DONE", args.shard, flush=True)
    return 0


REL_OLD = HERE / "reports" / "omnimatch_release_20261027"
REL_NEW = HERE / "reports" / "omnimatch_release_549"
NEW9_SEED, NEW9_N = 20261027, 9                       # Amendment 549: round(20 x 168 / 381) = 9


def new9() -> list[str]:
    new = [c for c in cases() if c["cohort382"] == "new168"]
    assert len(new) == 168
    return [new[i]["key"] for i in sorted(np.random.default_rng(NEW9_SEED).choice(168, NEW9_N, replace=False))]


def command_release_compare(args) -> int:
    """omnimatch_release_run.command_compare over the 20 step-0 scenarios + the 9 drawn from the 168 (same contrasts,
    same bootstrap), and over the 9 alone; writes reports/omnimatch_release_549/{comparison.md, per_scenario_J.csv,
    report.md}. report.md carries the lines make_om_numbers.py parses."""
    import json
    import pandas as pd
    old_cases = json.loads((HERE / "reports" / "audited_md5_fused_20260904" / "protocol.json").read_text(
        encoding="utf-8"))["cases"]
    rng0 = np.random.default_rng(20261027)
    old20 = [old_cases[i]["key"] for i in sorted(rng0.choice(len(old_cases), 20, replace=False))]
    keys9 = new9()
    rel = pd.concat([pd.read_csv(REL_OLD / "scores.csv"), pd.read_csv(REL_NEW / "scores.csv")])
    rel["arm"] = "release:" + rel["arm"]
    port = pd.read_csv(OUT / "scores.csv")
    port = port[port["seed"].isin([1, 2, 3]) & port["arm"].isin(["F", "N", "RF"])]
    port["arm"] = "port:" + port["arm"]
    d = pd.concat([rel, port])
    contrasts = [("release:rgcn_margin", "release:mlp_cross_entropy"), ("release:rgcn_margin", "port:F"),
                 ("release:rf", "port:RF"), ("release:rgcn_margin", "release:rgcn_margin_noedges"),
                 ("release:rgcn_margin_noedges", "port:N"), ("release:rgcn_margin_rowmajor", "port:F"),
                 ("release:rgcn_margin", "release:rgcn_margin_rowmajor"), ("release:rf", "release:rgcn_margin"),
                 ("port:F", "port:N")]
    out_lines, results = [], {}
    for label, keys in (("29 = 20 step-0 + 9 new", old20 + keys9), ("the 9 new only", keys9)):
        dd = d[d["key"].isin(keys)]
        per = dd.groupby(["arm", "key"])[["J", "R_exp", "F1_Hr"]].mean()
        nseeds = dd.groupby(["arm", "key"])["seed"].nunique()
        arms = sorted(per.index.get_level_values(0).unique())
        lines = [f"## {label}", "", "| arm | scenarios | seeds/scenario | J_tf | R_exp | F1_Hr |", "|---|---|---|---|---|---|"]
        for a in arms:
            x = per.loc[a]
            lines.append(f"| {a} | {len(x)} | {nseeds.loc[a].min()}-{nseeds.loc[a].max()} | {x['J'].mean():.3f} | "
                         f"{x['R_exp'].mean():.3f} | {x['F1_Hr'].mean():.3f} |")
        lines += ["", "| contrast | scenarios | J_tf diff [95% bootstrap] | R_exp | F1_Hr |", "|---|---|---|---|---|"]
        rng = np.random.default_rng(20261027)
        for a, b in contrasts:
            if a not in arms or b not in arms:
                continue
            common = sorted(set(per.loc[a].index) & set(per.loc[b].index))
            diff = per.loc[a].loc[common] - per.loc[b].loc[common]
            J = diff["J"].to_numpy()
            boots = [J[rng.integers(0, len(J), len(J))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            results[(label, a, b)] = (J.mean(), lo, hi, len(common))
            lines.append(f"| {a} - {b} | {len(common)} | {J.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                         f"{diff['R_exp'].mean():+.3f} | {diff['F1_Hr'].mean():+.3f} |")
        out_lines += lines + [""]
        if label.startswith("29"):
            per["J"].unstack(0).to_csv(REL_NEW / "per_scenario_J.csv")
            n29 = len(per.loc["release:rgcn_margin"])
            j = {a: per.loc[a]["J"].mean() for a in arms}
    (REL_NEW / "comparison.md").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    lab = "29 = 20 step-0 + 9 new"

    def fmt(a, b):
        e, lo, hi, _ = results[(lab, a, b)]
        return f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "−")
    rep = [f"# OmniMatch release vs our port on {n29} readable Valentine scenarios (Amendment 549, EXPLORATORY)", "",
           f"The {n29} step-0 scenarios (the 20 of 2026-10-27 + 9 drawn from the 168 new ones, "
           "OMNIMATCH_ABLATION.md Amendment 549) × seeds 1–3; release runs of the 20 from reports/omnimatch_release_20261027, "
           "of the 9 from reports/omnimatch_release_549 (same driver and environment); port matrices from "
           "reports/omnimatch_ablation_549. Generated by omnimatch_549.py release-compare; details: comparison.md.", "",
           "| model | release | port |", "|---|---|---|",
           f"| rgcn_margin / F | {j['release:rgcn_margin']:.3f} | {j['port:F']:.3f} |",
           f"| no-edge variant / N | {j['release:rgcn_margin_noedges']:.3f} | {j['port:N']:.3f} |",
           f"| rf / RF | {j['release:rf']:.3f} | {j['port:RF']:.3f} |", "",
           "| paired difference | J_tf [95% bootstrap over scenarios] |", "|---|---|",
           f"| release rgcn − release mlp | {fmt('release:rgcn_margin', 'release:mlp_cross_entropy')} |",
           f"| release rgcn − port F | {fmt('release:rgcn_margin', 'port:F')} |",
           f"| release rf − port RF | {fmt('release:rf', 'port:RF')} |",
           f"| release rgcn − release no-edge | {fmt('release:rgcn_margin', 'release:rgcn_margin_noedges')} |",
           f"| port F − port N | {fmt('port:F', 'port:N')} |",
           f"| release rf − release rgcn | {fmt('release:rf', 'release:rgcn_margin')} |", ""]
    (REL_NEW / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(out_lines))
    print("\n".join(rep))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("release-compare")
    sub.add_parser("new9")
    c = sub.add_parser("check")
    c.add_argument("--scratch", required=True)
    sub.add_parser("reuse")
    r = sub.add_parser("run")
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--all", action="store_true")
    a = ap.parse_args()
    return {"check": command_check, "reuse": command_reuse, "run": command_run,
            "release-compare": command_release_compare,
            "new9": lambda _a: print(" ".join(new9())) or 0}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
