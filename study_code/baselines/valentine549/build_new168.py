"""Step 1: the readable and recoded corpora of the 168 new Valentine scenarios.

    harness/generated/valentine-new168        the 168 scenario folders of benchmark-official-20261030b, profiled
    harness/generated/valentine-new168-md5    make_md5_audited.py on them (same master seed as the 382), cohort new168
    output/cluster-start-2026-10-30/new168-cases/cases.csv   the cohort labels make_md5_audited.py records

    python build_new168.py <workspace>
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
work = ws / "output/cluster-start-2026-10-30"
tree = ws / "harness/generated/benchmark-official-20261030b"
plain = ws / "harness/generated/valentine-new168"
md5 = ws / "harness/generated/valentine-new168-md5"
for p in (plain, md5):
    if p.exists():
        sys.exit(f"{p} exists")
cohort = json.loads((work / "cohort-549.json").read_text())
cases_dir = work / "new168-cases"
cases_dir.mkdir(exist_ok=True)
with (cases_dir / "cases.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "cohort"])
    for r in cohort["new_cases"]:
        w.writerow([r["id"], "new168"])
for r in cohort["new_cases"]:
    shutil.copytree(tree / r["path"], plain / r["path"])
print("plain:", sum(1 for _ in plain.glob("*/*/manifest.json")), "scenarios")
py = sys.executable
for args in (["--data", str(plain), "--out", str(md5), "--cases", str(cases_dir)],
             ["--data", str(plain), "--out", str(md5), "--verify"]):
    r = subprocess.run([py, str(ws / "harness/make_md5_audited.py"), *args], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        sys.exit(r.stderr)
cohorts = {json.loads(p.read_text())["cohort"] for p in md5.glob("*/*/transform-manifest.json")}
print("md5:", sum(1 for _ in md5.glob("*/*/manifest.json")), "scenarios; cohort labels", cohorts)
