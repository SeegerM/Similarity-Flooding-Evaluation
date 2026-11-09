"""Import ISResMat's GPU outputs for the 168 new Valentine scenarios (2026-10-31; <gpu-host> $SMX_REMOTE_ROOT/smx/out/<regime>/<variant>/
<case key>.csv, cases_<regime>.json) as FILE seeds reference/seeds/<regime>/{isresmat,isresmat_hybrid}/, with
exactly the checks of import_isresmat.py (row/column order of the levenshtein seed, only empty columns may be absent and
get 0, every value finite). Configurations are written separately (servers/reseed_stage.py).

    python servers/import_isresmat_new.py <dir with out/ and cases.json> pnew|mnew
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from import_isresmat import SEEDS, VARIANTS, empty_columns, read, strip  # noqa: E402
from score_seeds import CORPUS  # noqa: E402

src, regime = Path(sys.argv[1]), sys.argv[2]
cases = json.loads((src / "cases.json").read_text(encoding="utf-8"))
for variant, seed in VARIANTS.items():
    root = SEEDS / regime / seed
    if root.exists():
        sys.exit(f"{root} exists")
    zero = n = 0
    for c in cases:
        ds, sc = c["dataset"], c["scenario"]
        tcols, trows, _ = read(SEEDS / regime / "levenshtein" / ds / sc / "seed.csv")
        cols, rows, vals = read(src / "out" / regime / variant / f"{c['key']}.csv")
        ci = {strip(x): i for i, x in enumerate(cols)}
        ri = {strip(x): i for i, x in enumerate(rows)}
        miss_c = [x for x in map(strip, tcols) if x not in ci]
        miss_r = [x for x in map(strip, trows) if x not in ri]
        if miss_c or miss_r:
            scen = CORPUS[regime] / ds / sc
            ok_c = empty_columns(next((scen / "target").glob("*.csv")))
            ok_r = empty_columns(next((scen / "source").glob("*.csv")))
            bad = [x for x in miss_c if x not in ok_c] + [x for x in miss_r if x not in ok_r]
            if bad:
                sys.exit(f"{regime}/{variant}/{ds}/{sc}: non-empty columns missing from ISResMat: {bad[:3]}")
        if (len(ci) + len(miss_c) - len(tcols), len(ri) + len(miss_r) - len(trows)) != (0, 0):
            sys.exit(f"{regime}/{variant}/{ds}/{sc}: ISResMat has names outside the seed template")
        out = root / ds / sc / "seed.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Source\\Target"] + tcols)
            for r in trows:
                row = [("0.0" if strip(r) in miss_r or strip(t) in miss_c else vals[ri[strip(r)]][ci[strip(t)]])
                       for t in tcols]
                if not all(math.isfinite(float(v)) for v in row):
                    sys.exit(f"{regime}/{variant}/{ds}/{sc}: non-finite value")
                w.writerow([r] + row)
        zero += len(miss_c) + len(miss_r)
        n += 1
    (root / "COMPLETE").write_text(f"{regime}/{seed}: {n} scenarios from ISResMat {variant} (<gpu-host>, A10 GPU; "
                                   f"nondeterministic on GPU), imported by servers/import_isresmat_new.py; "
                                   f"{zero} empty columns filled with 0\n", encoding="utf-8")
    print(f"{regime}/{seed}: {n} matrices, {zero} empty columns filled with 0")
