"""External-review check (EXPLORATORY, 2026-11-03): per-seed values of the SemStruct reconstruction (Amendment 2b, readable
Valentine, cohort remaining505, evaluation row cap 2000), from the stored per-scenario scores only
(harness/reports/semstruct_a2_549/eval_plain_cap2000/scenarios.csv; columns <arm>@<seed>:J).

Per arm and training seed: J_tf as the evaluation report aggregates it (mean within family, families with >= 5
scenarios, mean over families); per seed the contrasts value-average − full and full − header-only; the one-hop and
header-only arms exist for the first seed only. Checks: the seed means and the per-seed contrasts reproduce the
report's numbers (report.md of the same folder) to rounding. Output (JSON + MD) in SMX_REVIEW_DIR, default
latex/sf-evaluation-presentation-549/review_checks/.

    ../harness/.venv-bdi/Scripts/python.exe -B review_checks/xr_semstruct_seeds.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent.parent
SRC = STUDY.parent / "harness" / "reports" / "semstruct_a2_549" / "eval_plain_cap2000"
OUT = Path(os.environ.get("SMX_REVIEW_DIR") or STUDY.parent / "latex" / "sf-evaluation-presentation-549" / "review_checks")


def main() -> int:
    d = pd.read_csv(SRC / "scenarios.csv")
    d = d[d.cohort == "remaining505"]
    fams = [f for f, n in d.family.value_counts().items() if n >= 5]
    d = d[d.family.isin(fams)]

    def macro(col: str) -> float:
        return float(d.groupby("family")[col].mean().mean())
    arms = {}
    for arm in ("full", "valavg", "onehop", "lm_only", "mlp"):
        seeds = sorted(int(m.group(1)) for c in d.columns if (m := re.fullmatch(rf"{re.escape(arm)}@(\d+):J", c)))
        arms[arm] = {s: macro(f"{arm}@{s}:J") for s in seeds}
        arms[arm]["mean"] = macro(f"{arm}:J")
    per_seed = {s: {"valavg_minus_full": arms["valavg"][s] - arms["full"][s],
                    "full_minus_lm_only": arms["full"][s] - arms["lm_only"][0]} for s in (0, 1, 2)}
    rep = (SRC / "report.md").read_text(encoding="utf-8")
    for s in (0, 1, 2):
        m = re.search(rf"seed {s + 1}: full − lm_only ([+-][0-9.]+); valavg − full ([+-][0-9.]+)", rep)
        # the report's per-seed gate uses each seed's own lm_only; lm_only is trained once, so it is the same column
        assert m and abs(per_seed[s]["valavg_minus_full"] - float(m.group(2))) < 6e-4, (s, per_seed[s], m and m.groups())
        assert abs(per_seed[s]["full_minus_lm_only"] - float(m.group(1))) < 6e-4, (s, per_seed[s], m.groups())
    for arm, val in (("full", "0.604"), ("valavg", "0.654"), ("onehop", "0.605"), ("lm_only", "0.391")):
        assert f"{arms[arm]['mean']:.3f}" == val, (arm, arms[arm]["mean"])
    vf = [per_seed[s]["valavg_minus_full"] for s in (0, 1, 2)]
    fl = [arms["full"][s] for s in (0, 1, 2)]
    va = [arms["valavg"][s] for s in (0, 1, 2)]
    out = {"families": fams, "scenarios": int(len(d)), "arms": {a: {str(k): v for k, v in x.items()} for a, x in arms.items()},
           "per_seed": {str(k): v for k, v in per_seed.items()},
           "spread": {"full": [min(fl), max(fl)], "valavg": [min(va), max(va)], "valavg_minus_full": [min(vf), max(vf)]},
           "onehop_minus_full_seed0": arms["onehop"][0] - arms["full"][0]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xr_semstruct_seeds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# SemStruct reconstruction: per-seed values (EXPLORATORY, external review 2026-11-03)", "",
          f"remaining505, plain, cap 2000; macro over {', '.join(fams)} ({len(d)} scenarios).", "",
          "| arm | seed 1 | seed 2 | seed 3 | mean |", "|---|---|---|---|---|"]
    for a, x in arms.items():
        md.append(f"| {a} | " + " | ".join(f"{x[s]:.3f}" if s in x else "–" for s in (0, 1, 2)) + f" | {x['mean']:.3f} |")
    md += ["", "| per seed | seed 1 | seed 2 | seed 3 |", "|---|---|---|---|",
           "| value-average − full | " + " | ".join(f"{v:+.3f}" for v in vf) + " |",
           "| full − header-only | " + " | ".join(f"{per_seed[s]['full_minus_lm_only']:+.3f}" for s in (0, 1, 2)) + " |",
           "", f"one-hop − full, first seed only: {out['onehop_minus_full_seed0']:+.3f}"]
    (OUT / "xr_semstruct_seeds.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
