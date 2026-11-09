"""Run every experiment script and report which ones reproduce the paper's numbers.

    python run_all.py              # all experiments (about a minute)
    python run_all.py E03 E06      # only these

Each experiment writes its report to reports/<id>.md and exits with status 1 if a value differs from checks/<id>.json.
Afterwards this script also compares every intermediate value of the shared analysis code (4,600+ values) with
checks/all_values.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))

wanted = sys.argv[1:]
scripts = sorted(p for p in (ROOT / "experiments").glob("E*.py") if not wanted or p.name[:3] in wanted)
results = []
for s in scripts:
    t = time.time()
    r = subprocess.run([sys.executable, str(s)], cwd=ROOT / "experiments", stdout=subprocess.DEVNULL)
    results.append((s.name, r.returncode, time.time() - t))
    print(f"{'ok      ' if r.returncode == 0 else 'MISMATCH'} {time.time() - t:6.1f} s  experiments/{s.name}")

import values  # noqa: E402
v = values.values(quiet=True)
exp = json.loads((ROOT / "checks" / "all_values.json").read_text(encoding="utf-8"))
bad = [k for k in exp if v.get(k) != exp[k]]
print(f"\nshared analysis values: {len(exp) - len(bad)} of {len(exp)} identical to checks/all_values.json")
for k in bad[:20]:
    print(f"  {k}: expected {exp[k]}, got {v.get(k)}")
failed = [n for n, rc, _ in results if rc] + (["all_values"] if bad else [])
print(f"{len(results) - sum(1 for _, rc, _ in results if rc)} of {len(results)} experiments reproduce the paper"
      + ("" if not failed else f"; check: {', '.join(failed)}"))
sys.exit(1 if failed else 0)
