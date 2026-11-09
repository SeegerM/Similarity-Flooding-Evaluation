"""X84 (EXPLORATORY; EXPLORATORY.md X84, design and predictions written 2026-10-07 before any run): Melnik et al.'s input-noise
test with SF's actual reference configuration (X7's `ref` lacked the literal similarities and used the default epsilon;
EXPLORATORY X7 correction 2026-10-03). This script only writes the perturbed initial matrices and the case list; A builds
the configurations.

Cases: 20 scenarios per corpus, drawn once with seed 20261007 (random.Random(f"X84/<corpus>").sample, sorted):
    valentine  the 382 protocol cases (score_seeds.cases_for("plain")), seeds reference/seeds/plain/<m>
    hdxsm      the 199 usable pairs, seeds reference/seeds/hdxsm/<m>
    goby       the 200 pairs, seeds reference/seeds/goby/<m>
Matchers prefix_suffix, levenshtein, pmf (as X7). Variants p0; p50, p100, p200 with draws d0..d2; flat. The perturbation is
explore_robust.perturb (X7's, imported unchanged): q' = max(0, q (1 + x)), x ~ U[-p, p] per cell, the generator seeded by
SHA-256 of "R1/<key>/<variant>" with key "<corpus>/<m>/<ds>/<sc>"; flat = every positive cell set to the mean of the positive
cells. p0 is a byte copy of the source seed file, so the p0 runs are the gate (SF from the FILE seed = the corpus's original arm, cell for cell).

Writes data/x84_noise/seeds/<corpus>/<m>__<variant>/<ds>/<sc>/seed.csv (labels and header as the source seed, values %.7g
as X7), data/x84_noise/cases.json and data/x84_noise/manifest.json (script hash, counts, SHA-256 per written seed).

    python layer3b/x84_noise_prepare.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(STUDY.parent / "harness"))
from explore_robust import perturb  # noqa: E402
from score_seeds import cases_for  # noqa: E402

OUT = STUDY / "data" / "x84_noise"
SEEDS = STUDY / "reference" / "seeds"
GEN = STUDY.parent / "harness" / "generated"
MATCHERS = ("prefix_suffix", "levenshtein", "pmf")
LEVELS = (50, 100, 200)
DRAWS = 3
N_CASES, SEED = 20, 20261007


def variants() -> list[str]:
    return ["p0"] + [f"p{p}_d{d}" for p in LEVELS for d in range(DRAWS)] + ["flat"]


def corpus_cases(c: str) -> list[tuple[str, str]]:
    if c == "valentine":
        pool = sorted({(ds, sc) for ds, sc, _ in cases_for("plain")})
    else:
        root = GEN / ("hdxsm-usable" if c == "hdxsm" else "goby-imported")
        pool = sorted((d.name, s.name) for d in root.iterdir() if d.is_dir() for s in d.iterdir() if s.is_dir())
    rng = random.Random(f"X84/{c}/{SEED}")
    return sorted(rng.sample(pool, N_CASES))


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    cases = {c: corpus_cases(c) for c in ("valentine", "hdxsm", "goby")}
    hashes, n = {}, 0
    for c, cs in cases.items():
        src_root = SEEDS / ("plain" if c == "valentine" else c)
        for m in MATCHERS:
            for ds, sc in cs:
                src = src_root / m / ds / sc / "seed.csv"
                rows = list(csv.reader(open(src, encoding="utf-8", newline="")))
                header, labels = rows[0], [r[0] for r in rows[1:]]
                q = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
                if not np.isfinite(q).all():
                    raise SystemExit(f"{src}: non-finite initial matrix")
                for v in variants():
                    out = OUT / "seeds" / c / f"{m}__{v}" / ds / sc
                    out.mkdir(parents=True, exist_ok=True)
                    f = out / "seed.csv"
                    if v == "p0":                 # byte copy, so the gate (FILE seed = original arm) is exact
                        shutil.copyfile(src, f)
                        hashes[f.relative_to(OUT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
                        n += 1
                        continue
                    qv = perturb(q, v, f"{c}/{m}/{ds}/{sc}")
                    with open(f, "w", newline="", encoding="utf-8") as fh:
                        w = csv.writer(fh)
                        w.writerow(header)
                        for lab, row in zip(labels, qv):
                            w.writerow([lab, *[f"{x:.7g}" for x in row]])
                    hashes[f.relative_to(OUT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
                    n += 1
    (OUT / "cases.json").write_text(json.dumps({c: [list(x) for x in cs] for c, cs in cases.items()}, indent=1),
                                    encoding="utf-8")
    man = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "matchers": MATCHERS,
           "variants": variants(), "cases_per_corpus": N_CASES, "seed_files": n,
           "runs_per_arm": n, "arms": ["melnik", "lit_none"], "scenario_arm_evaluations": 2 * n, "sha256": hashes}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"{n} seed files ({len(MATCHERS)} matchers x 60 scenarios x {len(variants())} variants); "
          f"{2 * n} scenario-arm evaluations with melnik and lit_none -> {OUT}")
    for c, cs in cases.items():
        print(c, cs[:3], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
