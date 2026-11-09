"""Import ISResMat's GPU outputs (<gpu-host>, $SMX_REMOTE_ROOT/smx/out/<regime>/<variant>/<case key>.csv; cases.json maps keys to
scenarios) as FILE seeds reference/seeds/<regime>/<seed>/<dataset>/<scenario>/seed.csv, and write layer-1/3
configurations for them (2026-10-04 ~22:55).

    variant instance -> seed isresmat, variant hybrid -> seed isresmat_hybrid

Each matrix is written in exactly the row and column order of the levenshtein seed of the same scenario (ISResMat
uses bare column names; the seeds carry the table prefix). Every name must match and every value be finite, else the
import stops. The configurations are clones of the registered `levenshtein__<arm>` configurations of
layer1_<regime> / layer3_<regime> with only the seed directory, the label and resultsDir changed (asserted), under the
separate suffixes layer1isr_<regime> / layer3isr_<regime> (score with score_layer1.py --variant isr).

    python servers/import_isresmat.py <dir with out/ and cases.json>
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY))
SEEDS = STUDY / "reference" / "seeds"
CFG = STUDY / "tuning" / "configs"
VARIANTS = {"instance": "isresmat", "hybrid": "isresmat_hybrid"}


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        r = list(csv.reader(fh))
    return r[0][1:], [row[0] for row in r[1:]], [row[1:] for row in r[1:]]


def strip(name: str) -> str:
    return name.split(".", 1)[1] if name.startswith(("source.", "target.")) else name


def empty_columns(table: Path) -> set[str]:
    """Columns of a corpus table without any non-blank value."""
    csv.field_size_limit(1 << 30)
    with table.open(encoding="utf-8", newline="", errors="replace") as fh:
        r = csv.reader(fh)
        header = next(r)
        seen = [False] * len(header)
        for row in r:
            for i, v in enumerate(row[: len(header)]):
                if not seen[i] and v.strip():
                    seen[i] = True
    return {h for h, s in zip(header, seen) if not s}


def import_seeds(src: Path) -> int:
    from score_seeds import CORPUS
    cases = json.loads((src / "cases.json").read_text(encoding="utf-8"))
    n, filled = 0, {}
    for regime in ("plain", "md5"):
        for variant, seed in VARIANTS.items():
            zero = 0
            for c in cases:
                ds, sc = c["dataset"], c["scenario"]
                tcols, trows, _ = read(SEEDS / regime / "levenshtein" / ds / sc / "seed.csv")
                cols, rows, vals = read(src / "out" / regime / variant / f"{c['key']}.csv")
                ci = {strip(x): i for i, x in enumerate(cols)}
                ri = {strip(x): i for i, x in enumerate(rows)}
                # ISResMat leaves out columns without any value; they get 0 (no evidence), and only those
                miss_c = [x for x in map(strip, tcols) if x not in ci]
                miss_r = [x for x in map(strip, trows) if x not in ri]
                if miss_c or miss_r:
                    scen = CORPUS[regime] / ds / sc
                    ok_c = empty_columns(next((scen / "target").glob("*.csv")))
                    ok_r = empty_columns(next((scen / "source").glob("*.csv")))
                    bad = [x for x in miss_c if x not in ok_c] + [x for x in miss_r if x not in ok_r]
                    if bad:
                        raise SystemExit(f"{regime}/{variant}/{ds}/{sc}: non-empty columns missing from ISResMat: {bad[:3]}")
                extra = len(ci) + len(miss_c) - len(tcols), len(ri) + len(miss_r) - len(trows)
                if extra != (0, 0):
                    raise SystemExit(f"{regime}/{variant}/{ds}/{sc}: ISResMat has names outside the seed template")
                out = SEEDS / regime / seed / ds / sc / "seed.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["Source\Target"] + tcols)
                    for r in trows:
                        row = [("0.0" if strip(r) in miss_r or strip(t) in miss_c else vals[ri[strip(r)]][ci[strip(t)]])
                               for t in tcols]
                        if not all(math.isfinite(float(v)) for v in row):
                            raise SystemExit(f"{regime}/{variant}/{ds}/{sc}: non-finite value")
                        w.writerow([r] + row)
                zero += len(miss_c) + len(miss_r)
                n += 1
            filled[f"{regime}/{seed}"] = zero
    print(f"empty columns filled with 0 (absent from ISResMat's output): {filled}")
    return n


def configs() -> int:
    n = 0
    for regime in ("plain", "md5"):
        for layer in (1, 3):
            src_root, suffix = CFG / f"layer{layer}_{regime}", f"layer{layer}isr_{regime}"
            dst_root = CFG / suffix
            if dst_root.exists():
                shutil.rmtree(dst_root)
            for src in sorted(src_root.glob("levenshtein__*")):
                arm = src.name.split("__", 1)[1]
                for seed in VARIANTS.values():
                    label = f"{seed}__{arm}"
                    dst = dst_root / label
                    shutil.copytree(src, dst)
                    for f in dst.iterdir():
                        old = f.read_text(encoding="utf-8")
                        new = (old.replace(f"/reference/seeds/{regime}/levenshtein\"", f"/reference/seeds/{regime}/{seed}\"")
                                  .replace(f"/results/layer{layer}_{regime}/levenshtein__{arm}\"",
                                           f"/results/{suffix}/{label}\""))
                        if f.name == "meta.json":
                            m = json.loads(new)
                            m.update({"seed": seed, "stage": f"layer{layer}isr", "note": "ISResMat (GPU) imported as FILE seed"})
                            new = json.dumps(m, indent=1)
                        if f.name == "second_line_matchers.yaml":
                            assert f"/reference/seeds/{regime}/{seed}\"" in new, label
                        if f.name == "general.yaml":
                            assert f"/results/{suffix}/{label}\"" in new, label
                        f.write_text(new, encoding="utf-8")
                    n += 1
            print(f"{suffix}: {len(list(dst_root.iterdir()))} configurations")
    return n


def main() -> int:
    n = import_seeds(Path(sys.argv[1]))
    print(f"{n} seed matrices imported")
    print(f"{configs()} configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
