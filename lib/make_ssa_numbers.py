"""Write output/ssa_numbers.tex: SemStruct under the revised training setup (the SemStruct ablation plan,
Amendment 2b: Amendment 2 retrained and scored on the 549 Valentine scenarios, EXPLORATORY). Sources, parsed, nothing typed:
- results/baselines/reports/semstruct_a2_549/eval_plain_cap2000/report.md (semstruct_a2_eval.py; cohort remaining505 =
  the 337 remaining scenarios of the earlier copy + the 168 new; full and valavg averaged over the training seeds present,
  onehop one seed; row co-occurrence paired within the first seed);
- .../recall_check.md (semstruct_recall_check.py --a2; the paper's own scoring: per source column Recall@GT, ties by
  index, softmax + bipartite reranking; median per family; first seed).
- GOBY was not rerun (Amendment 2b): its macros read results/baselines/reports/semstruct_a2_20261029/eval_goby_plain_cap2000/.

    python make_ssa_numbers.py
"""

from __future__ import annotations

import csv
import re

from paths import BASELINES, OUT
COHORT = "remaining505"
R382 = BASELINES / "reports" / "semstruct_a2_20261029"
R = BASELINES / "reports" / "semstruct_a2_549"
text = (R / "eval_plain_cap2000" / "report.md").read_text(encoding="utf-8")
assert "zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00" in text, "zero_delta control failed"
block = text[text.index(f"## {COHORT}"):text.index("## all")]


def contrast(a: str, b: str) -> tuple[str, str, str]:
    row = next(line for line in block.splitlines() if line.startswith(f"| {a} − {b} |"))
    est, lo, hi = re.findall(r"[+-]\d+\.\d+", row.split("|")[3])
    return est, lo, hi


def arm(name: str) -> str:
    row = next(line for line in block.splitlines() if line.startswith(f"| {name} |"))
    return row.split("|")[2].strip()


out = {"ssaN": re.search(rf"## {COHORT}: (\d+) scenarios", block).group(1),
       "ssaSeeds": re.search(r"Seeds present: full (\d+)", text).group(1)}
out["ssaGate"], out["ssaGateLo"], out["ssaGateHi"] = contrast("full", "lm_only")
out["ssaOneFull"], out["ssaOneFullLo"], out["ssaOneFullHi"] = contrast("onehop", "full")
out["ssaValFull"], out["ssaValFullLo"], out["ssaValFullHi"] = contrast("valavg", "full")
out["ssaPerm"], out["ssaPermLo"], out["ssaPermHi"] = contrast("full@0", "permuted")
out["ssaJFull"], out["ssaJOne"], out["ssaJVal"], out["ssaJLm"] = (arm("full"), arm("onehop"), arm("valavg"),
                                                                  arm("lm_only"))
# the paper's Fig. 5 ladder (MLP rung, first seed; second queue item 5)
if "| mlp |" in block:
    out["ssaJMlp"] = arm("mlp")
    out["ssaLadMlp"], out["ssaLadMlpLo"], out["ssaLadMlpHi"] = contrast("mlp", "lm_only")
    out["ssaLadVal"], out["ssaLadValLo"], out["ssaLadValHi"] = contrast("valavg", "mlp")
seeds = [float(x) for x in re.findall(r"^- seed \d+: .*valavg − full ([+-]\d+\.\d+)", block, re.M)]
out["ssaSeedMin"], out["ssaSeedMax"] = f"{min(seeds):+.3f}", f"{max(seeds):+.3f}"

# checks: evaluation row cap 400 (seed mean) and the recoded regime (first seed), same script
def other(name: str) -> str:
    t = (R / name / "report.md").read_text(encoding="utf-8")
    return t[t.index(f"## {COHORT}"):t.index("## all")]


for prefix, name in (("ssaFh", "eval_plain_cap400"), ("ssaMd", "eval_audited_md5_cap2000")):
    b = other(name)
    for key, (a, c) in {"OneFull": ("onehop", "full"), "ValFull": ("valavg", "full"), "Gate": ("full", "lm_only")}.items():
        row = next(line for line in b.splitlines() if line.startswith(f"| {a} − {c} |"))
        out[prefix + key], out[prefix + key + "Lo"], out[prefix + key + "Hi"] = re.findall(r"[+-]\d+\.\d+",
                                                                                          row.split("|")[3])
    if prefix == "ssaMd":
        js = [float(line.split("|")[2]) for line in b.splitlines()
              if re.match(r"\| (full|valavg|onehop|lm_only) \|", line)]
        out["ssaMdJLo"], out["ssaMdJHi"] = f"{min(js):.2f}", f"{max(js):.2f}"

# no-merging sensitivity (first seed; queue item 4): full and valavg trained and scored without value merging
nm = (R / "eval_plain_cap2000_nomerge" / "report.md").read_text(encoding="utf-8")
assert "zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00" in nm, "zero_delta control failed (no merging)"
nb = nm[nm.index(f"## {COHORT}"):nm.index("## all")]
for key, (a, c) in {"NmValFull": ("valavg", "full"), "NmGate": ("full", "lm_only")}.items():
    row = next(line for line in nb.splitlines() if line.startswith(f"| {a} − {c} |"))
    out["ssa" + key], out["ssa" + key + "Lo"], out["ssa" + key + "Hi"] = re.findall(r"[+-]\d+\.\d+", row.split("|")[3])
for name in ("full", "valavg"):
    row = next(line for line in nb.splitlines() if line.startswith(f"| {name} |"))
    out["ssaNmJ" + name.capitalize()] = row.split("|")[2].strip()
# the merged first-seed values the no-merging run is compared with (same report as above, per-seed rows)
seed_one = re.search(r"^- seed 1: (.*)$", block, re.M).group(1)
out["ssaSeedOneValFull"] = re.search(r"valavg − full ([+-]\d+\.\d+)", seed_one).group(1)
sc = [row for row in csv.DictReader((R / "eval_plain_cap2000" / "scenarios.csv").open(encoding="utf-8"))
      if row["cohort"] == COHORT]
fams = {}
for row in sc:
    fams.setdefault(row["family"], []).append(float(row["full@0:J"]))
out["ssaSeedOneJFull"] = f"{sum(sum(v) / len(v) for v in fams.values() if len(v) >= 5) / sum(1 for v in fams.values() if len(v) >= 5):.3f}"

# GOBY (independently produced; first seed; bootstrap over the SF paper's table-sharing clusters; second queue item 6)
gp = R382 / "eval_goby_plain_cap2000" / "report.md"
if gp.is_file():
    gt = gp.read_text(encoding="utf-8")
    assert "zero_delta reproduces lm_only, max |ΔJ| = 0.00e+00" in gt, "zero_delta control failed (GOBY)"
    gb = gt[gt.index("## all"):]
    out["ssaGobyN"] = re.search(r"## all: (\d+) scenarios", gb).group(1)
    out["ssaGobyClusters"] = re.search(r"bootstrap over (\d+) clusters", gb).group(1)
    for key, (a, c) in {"Gate": ("full", "lm_only"), "OneFull": ("onehop", "full"), "ValFull": ("valavg", "full"),
                        "Perm": ("full@0", "permuted"), "LadMlp": ("mlp", "lm_only"),
                        "LadVal": ("valavg", "mlp")}.items():
        row = next((line for line in gb.splitlines() if line.startswith(f"| {a} − {c} |")), None)
        if row:
            out["ssaGoby" + key], out["ssaGoby" + key + "Lo"], out["ssaGoby" + key + "Hi"] = re.findall(
                r"[+-]\d+\.\d+", row.split("|")[3])
    for name in ("full", "valavg", "onehop", "lm_only", "mlp"):
        row = next((line for line in gb.splitlines() if line.startswith(f"| {name} |")), None)
        if row:
            out["ssaGobyJ" + name.replace("_", "").capitalize()] = row.split("|")[2].strip()

# reproduction under the paper's own scoring
rc = (R / "recall_check.md").read_text(encoding="utf-8")
sec = rc[rc.index("## Ties: per source column"):]
header = next(line for line in rc.splitlines() if line.startswith("| arm |")).split("|")[2:-2]
families = [h.strip() for h in header]
row = next(line for line in sec.splitlines() if line.startswith("| full (revised setup), reranked |"))
medians = {f: float(c.split()[0]) for f, c in zip(families, row.split("|")[2:]) if f in
           ("ChEMBL", "OpenData", "TPC-DI", "Magellan")}
reached = [f for f, v in medians.items() if v >= 0.9995]
others = [v for f, v in medians.items() if v < 0.9995]
words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four"}
out["ssaRepReached"] = words[len(reached)]
out["ssaRepFamilies"] = words[len(medians)]
out["ssaRepLo"], out["ssaRepHi"] = f"{min(others):.2f}", f"{max(others):.2f}"
for f, v in medians.items():
    out["ssaRep" + f.replace("-", "")] = f"{v:.2f}"
lm = next(line for line in sec.splitlines() if line.startswith("| LM only (e5-small-v2, header), reranked |"))
lm_medians = [float(c.split()[0]) for f, c in zip(families, lm.split("|")[2:]) if f in ("ChEMBL", "OpenData", "TPC-DI")]
out["ssaRepLmLo"], out["ssaRepLmHi"] = f"{min(lm_medians):.2f}", f"{max(lm_medians):.2f}"

lines = [f"% Generated by make_ssa_numbers.py from results/baselines/reports/{R.name}/{{eval_plain_cap2000/report.md,"
         f"recall_check.md}} (SEMSTRUCT_ABLATION.md, Amendment 2b, 549; GOBY from {R382.name}); EXPLORATORY; do not edit."]
lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in out.items()]
(OUT / "ssa_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
