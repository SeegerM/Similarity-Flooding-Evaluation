"""EXPLORATORY: import ISResMat's GPU outputs for the E2 corpora (<gpu-host>, run_isresmat_ext.py:
$SMX_REMOTE_ROOT/smx/out/<ctl|t2d>/<variant>/<case key>.csv, cases_<corpus>.json) as FILE seeds
reference/seeds/<corpus>/<isresmat|isresmat_hybrid>/<dataset>/<scenario>/seed.csv, with the same checks as
servers/import_isresmat.py (names must match the levenshtein seed template; only empty columns may be absent and get
0; every value finite), and write SF configurations: clones of e2_<corpus>/levenshtein__<arm> with only the seed
directory, the label and resultsDir changed (asserted), under tuning/configs/e2isr_<corpus>.

    python servers/import_isresmat_ext.py <dir with out/ and cases_<corpus>.json> ctl|t2d [--complete-only]

--complete-only: ISResMat itself fails on
4 t2d scenarios in every attempt. With this flag only the scenarios with an output for BOTH variants are imported; the
others are listed in <corpus>_isr_skipped.json, and the configurations read the corpus `<corpus>isr` (on each host:
$SMX_REMOTE_ROOT/smx/corpus/<corpus>isr/<dataset>/<scenario> = symlinks to $SMX_REMOTE_ROOT/smx/corpus/<corpus>/..., made by
`servers/link_isr_corpus.py`), so that every scenario they run has a seed.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "servers"))
from import_isresmat import VARIANTS, empty_columns, read, strip  # noqa: E402

SEEDS = STUDY / "reference" / "seeds"
CFG = STUDY / "tuning" / "configs"
GEN = STUDY.parent / "harness" / "generated"
CORPUS = {"ctl": GEN / "controlled-imported-v3", "t2d": GEN / "t2d-usable"}


def import_seeds(src: Path, corpus: str, complete_only: bool = False) -> int:
    cases = json.loads((src / f"cases_{corpus}.json").read_text(encoding="utf-8"))
    if complete_only:
        ok = [c for c in cases if all((src / "out" / corpus / v / f"{c['key']}.csv").is_file() for v in VARIANTS)]
        skipped = [c for c in cases if c not in ok]
        (STUDY / "tuning" / "results" / "isr_ext" / f"{corpus}_isr_skipped.json").write_text(
            json.dumps(skipped, indent=1), encoding="utf-8")
        print(f"{corpus}: {len(ok)} of {len(cases)} scenarios with both variants; skipped "
              f"{[c['scenario'] for c in skipped]}")
        cases = ok
    n, filled = 0, {}
    for variant, seed in VARIANTS.items():
        zero = 0
        for c in cases:
            ds, sc = c["dataset"], c["scenario"]
            tcols, trows, _ = read(SEEDS / corpus / "levenshtein" / ds / sc / "seed.csv")
            cols, rows, vals = read(src / "out" / corpus / variant / f"{c['key']}.csv")
            ci = {strip(x): i for i, x in enumerate(cols)}
            ri = {strip(x): i for i, x in enumerate(rows)}
            miss_c = [x for x in map(strip, tcols) if x not in ci]
            miss_r = [x for x in map(strip, trows) if x not in ri]
            if miss_c or miss_r:
                scen = CORPUS[corpus] / ds / sc
                ok_c = empty_columns(next((scen / "target").glob("*.csv")))
                ok_r = empty_columns(next((scen / "source").glob("*.csv")))
                bad = [x for x in miss_c if x not in ok_c] + [x for x in miss_r if x not in ok_r]
                if bad:
                    raise SystemExit(f"{corpus}/{variant}/{ds}/{sc}: non-empty columns missing from ISResMat: {bad[:3]}")
            extra = len(ci) + len(miss_c) - len(tcols), len(ri) + len(miss_r) - len(trows)
            if extra != (0, 0):
                raise SystemExit(f"{corpus}/{variant}/{ds}/{sc}: ISResMat has names outside the seed template")
            out = SEEDS / corpus / seed / ds / sc / "seed.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Source\\Target"] + tcols)
                for r in trows:
                    row = [("0.0" if strip(r) in miss_r or strip(t) in miss_c else vals[ri[strip(r)]][ci[strip(t)]])
                           for t in tcols]
                    if not all(math.isfinite(float(v)) for v in row):
                        raise SystemExit(f"{corpus}/{variant}/{ds}/{sc}: non-finite value")
                    w.writerow([r] + row)
            zero += len(miss_c) + len(miss_r)
            n += 1
        filled[f"{corpus}/{seed}"] = zero
    print(f"empty columns filled with 0 (absent from ISResMat's output): {filled}")
    return n


def configs(corpus: str, complete_only: bool = False) -> int:
    suffix = f"e2isr_{corpus}"
    dst_root = CFG / suffix
    if dst_root.exists():
        shutil.rmtree(dst_root)
    n = 0
    for src in sorted((CFG / f"e2_{corpus}").glob("levenshtein__*")):
        arm = src.name.split("__", 1)[1]
        for seed in VARIANTS.values():
            label = f"{seed}__{arm}"
            dst = dst_root / label
            shutil.copytree(src, dst)
            for f in dst.iterdir():
                old = f.read_text(encoding="utf-8")
                new = (old.replace(f"/reference/seeds/{corpus}/levenshtein\"", f"/reference/seeds/{corpus}/{seed}\"")
                          .replace(f"/results/e2_{corpus}/levenshtein__{arm}\"", f"/results/{suffix}/{label}\""))
                if complete_only and f.name == "general.yaml":
                    base = f'defaultDatasetBasePath: "$SMX_REMOTE_ROOT/smx/corpus/{corpus}"'
                    assert new.count(base) == 1, label
                    new = new.replace(base, base[:-1] + 'isr"')
                if f.name == "meta.json":
                    m = json.loads(new)
                    m.update({"seed": seed, "stage": "e2isr",
                              "note": "EXPLORATORY: ISResMat (GPU) imported as FILE seed on the E2 corpus"})
                    new = json.dumps(m, indent=1)
                if f.name == "second_line_matchers.yaml":
                    assert f"/reference/seeds/{corpus}/{seed}\"" in new, label
                if f.name == "general.yaml":
                    assert f"/results/{suffix}/{label}\"" in new, label
                f.write_text(new, encoding="utf-8")
            n += 1
    print(f"{suffix}: {n} configurations")
    return n


def main() -> int:
    corpus, complete_only = sys.argv[2], "--complete-only" in sys.argv[3:]
    print(f"{import_seeds(Path(sys.argv[1]), corpus, complete_only)} seed matrices imported")
    configs(corpus, complete_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
