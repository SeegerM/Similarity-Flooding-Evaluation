"""X95 (layer3b/X95_PREREGISTRATION.md §4): the averaged bases as FILE initial matrices.
  ii  = mean of the min-max-normalised qgram2_dice, composite_dominants, composite_average (X52's frozen plain base)
  iii = mean of the min-max-normalised coma, char3_soft, cupid (the authors' base)
Min-max over the whole matrix (a constant matrix becomes 0). Labels must agree across the components (else the scenario is
skipped and counted). Values written %.7g as the study's seed files.

    python layer3b/x95_bases.py   -> data/x95_bases/<ii|iii>/<corpus>/<ds>/<sc>/seed.csv + manifest.json
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent.parent
SEEDS = STUDY / "reference" / "seeds"
OUT = STUDY / "data" / "x95_bases"
BASES = {"ii": ("qgram2_dice", "composite_dominants", "composite_average"), "iii": ("coma", "char3_soft", "cupid")}
CORPORA = {"valentine_plain": "plain", "t2d": "t2d", "ctl": "ctl", "final_controlled": "final_controlled", "goby": "goby",
           "hdxsm": "hdxsm", "gdcsm": "gdcsm", "madi": "madi"}


# rerun 549 (2026-11-02): SMX_VALENTINE=549 builds only the bases of the 168 new readable Valentine scenarios
# (reference/seeds/pnew -> data/x95_bases/<ii|iii>/valentine_pnew/) with manifest_valentine_pnew.json; the existing bases
# and manifest.json are not touched.
import os  # noqa: E402
MANIFEST = "manifest.json"
if os.environ.get("SMX_VALENTINE", "382") == "549":
    CORPORA, MANIFEST = {"valentine_pnew": "pnew"}, "manifest_valentine_pnew.json"


def read(p: Path):
    rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8"), newline="")))
    head, body = rows[0], rows[1:]
    return head, [r[0] for r in body], np.array([[float(x) if x not in ("", "NaN", "nan") else np.nan for x in r[1:]] for r in body])


def minmax(M):
    M = np.nan_to_num(M, nan=0.0)
    lo, hi = M.min(), M.max()
    return (M - lo) / (hi - lo) if hi > lo else np.zeros_like(M)


def main() -> int:
    man = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "bases": BASES, "counts": {}, "skipped": []}
    for base, comps in BASES.items():
        for corpus, seed in CORPORA.items():
            n = 0
            first = SEEDS / seed / comps[0]
            for f in sorted(first.glob("*/*/seed.csv")):
                ds, sc = f.parent.parent.name, f.parent.name
                parts = []
                for m in comps:
                    g = SEEDS / seed / m / ds / sc / "seed.csv"
                    if not g.is_file():
                        parts = None
                        break
                    parts.append(read(g))
                if parts is None or any(p[0] != parts[0][0] or p[1] != parts[0][1] for p in parts):
                    man["skipped"].append(f"{base}/{corpus}/{ds}/{sc}")
                    continue
                M = np.mean([minmax(p[2]) for p in parts], axis=0)
                out = OUT / base / corpus / ds / sc / "seed.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(parts[0][0])
                    for lab, row in zip(parts[0][1], M):
                        w.writerow([lab, *[f"{x:.7g}" for x in row]])
                n += 1
            man["counts"][f"{base}/{corpus}"] = n
    (OUT / MANIFEST).write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man["counts"], indent=1), "skipped", len(man["skipped"]), man["skipped"][:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
