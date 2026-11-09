"""X79 (b) (2026-10-06): controlled construction knobs on existing scenarios. Written before any knob run.
Design and predictions: EXPLORATORY.md X79.

Base: a stratified sample of Valentine readable scenarios (ChEMBL, OpenData, TPC-DI; up to PER_STRATUM per family × split ×
name noise × value noise) plus the 48 readable final-controlled scenarios. Every knob changes the TARGET side only:
  base     unchanged copy (the gate: must reproduce the original runs)
  K1       target columns in a seeded random order (removes the diagonal); truth columns permuted accordingly
  K2p<p>   a fraction p of the target's distinct values recoded to fresh tokens. One mapping per target side (the same value gets
           the same token in every column), so FDs, UCCs and within-side INDs still hold and the profiles carry over unchanged
           (pdep/gpdep depend only on value equality). Cross-side equality drops. Nested: a value recoded at p is recoded at
           every larger p.
  K3p<p>   a fraction p of the target's column names replaced by opaque tokens; the profiles' labels are rewritten. Nested as K2.
  K4f<f>   each side's FD file thinned to a seeded fraction f of its lines (role term and FD edges see fewer FDs).
p in (0.25, 0.5, 0.75, 1.0); f in (0.5, 0.25). File hashes in metadata/profile-status.properties are recomputed for every
changed table and profile file (Schematch verifies them).
Output: data/knobs/<dataset>/<scenario>__<knob>/ (one corpus "knobs"), manifest data/knobs/knobs_manifest.json.

    ../harness/.venv-bdi/Scripts/python.exe data/make_knobs.py [--per-stratum 4] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parent / "harness"))
sys.path.insert(0, str(STUDY))
from score_seeds import CORPUS, cases_for  # noqa: E402

csv.field_size_limit(2**31 - 1)
OUT = HERE / "knobs"
FINAL = STUDY.parent / "harness" / "generated" / "final-controlled"
PS = (0.25, 0.5, 0.75, 1.0)
FS = (0.5, 0.25)
PAT = re.compile(r"_(?P<split>both|horizontal|vertical)_.*_(?P<name>ac\d+|ec)_(?P<val>av|ev)$")
PROFILE_FILES = {"FD_results.txt": "fd", "UCC_results.txt": "ucc", "inds.txt": "ind", "LIMA_DC_results.txt": "dc"}


def frac(key: str) -> float:
    return (zlib.crc32(key.encode("utf-8")) % 100000) / 100000.0


def knobs() -> list[str]:
    return ["base", "K1"] + [f"K2p{int(p * 100)}" for p in PS] + [f"K3p{int(p * 100)}" for p in PS] + \
        [f"K4f{int(f * 100)}" for f in FS]


def base_cases(per_stratum: int) -> list[tuple[Path, str, str]]:
    strata = defaultdict(list)
    for ds, sc, fam in cases_for("plain"):
        m = PAT.search(sc)
        if m:
            strata[(fam, m["split"], "ec" if m["name"] == "ec" else "ac", m["val"])].append((ds, sc))
    out = []
    for k, v in sorted(strata.items()):
        v = sorted(v, key=lambda x: frac(f"x79base|{x[0]}|{x[1]}"))
        out += [(CORPUS["plain"], ds, sc) for ds, sc in v[:per_stratum]]
    for d in sorted(p for p in FINAL.iterdir() if p.is_dir()):
        for s in sorted(p for p in d.iterdir() if p.is_dir() and p.name.endswith("_plain")):
            out.append((FINAL, d.name, s.name))
    return out


def read_table(p: Path):
    with p.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def write_table(p: Path, header, rows) -> None:
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rehash(scen: Path) -> None:
    """Recompute <side>.table.sha256 and <side>.<kind>.sha256 in metadata/profile-status.properties."""
    st = scen / "metadata" / "profile-status.properties"
    if not st.is_file():
        return
    lines = st.read_text(encoding="utf-8").splitlines()
    new = {}
    for side in ("source", "target"):
        t = next((scen / side).glob("*.csv"), None)
        if t is not None:
            new[f"{side}.table.sha256"] = sha(t)
        for fname, kind in PROFILE_FILES.items():
            files = sorted((scen / "metadata" / side).rglob(fname))
            if files:
                new[f"{side}.{kind}.sha256"] = sha(files[0]) if len(files) == 1 else \
                    hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    out = []
    for l in lines:
        k = l.split("=", 1)[0]
        out.append(f"{k}={new[k]}" if k in new and "=" in l else l)
    st.write_text("\n".join(out) + "\n", encoding="utf-8")


def rename_labels(scen: Path, mapping: dict[str, str], all_names: list[str]) -> None:
    """Rewrite target column labels (target.csv.<name>) in every metadata text/json file. The pattern lists every target column
    name, longest first, so that a name that is a prefix of another (e.g. "Project" / "Project number") is never matched inside it;
    only names in `mapping` are replaced."""
    if not mapping:
        return
    alt = "|".join(re.escape(k) for k in sorted(all_names, key=len, reverse=True))
    pat = re.compile(r"target\.csv\.(" + alt + r")(?=[\],)\"]|\s\(|\s*$|,|\s*\])")
    lab = re.compile(r'("label":\s*")(' + alt + r')(")')
    for f in (scen / "metadata").rglob("*"):
        if f.is_file() and f.suffix in (".txt", ".json"):
            t = f.read_text(encoding="utf-8")
            t2 = pat.sub(lambda m: "target.csv." + mapping.get(m.group(1), m.group(1)), t)
            if f.suffix == ".json" and f.parent.name == "target":
                t2 = lab.sub(lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)) + m.group(3), t2)
            if t2 != t:
                f.write_text(t2, encoding="utf-8")


def build(job):
    root, ds, sc = job
    src = root / ds / sc
    res = {}
    for knob in knobs():
        dst = OUT / ds / f"{sc}__{knob}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        tgt = next((dst / "target").glob("*.csv"))
        gt_file = next((dst / "ground_truth").glob("*.csv"))
        key = f"x79|{ds}|{sc}"
        if knob == "K1":
            header, rows = read_table(tgt)
            order = sorted(range(len(header)), key=lambda j: frac(f"{key}|K1|{j}"))
            write_table(tgt, [header[j] for j in order], [[r[j] if j < len(r) else "" for j in order] for r in rows])
            g = list(csv.reader(gt_file.open(encoding="utf-8", newline="")))
            with gt_file.open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh, lineterminator="\n").writerows([[r[j] for j in order] for r in g])
        elif knob.startswith("K2"):
            p = int(knob[3:]) / 100
            header, rows = read_table(tgt)
            code = {}

            def rec(v):
                if v == "" or frac(f"{key}|K2|{v}") >= p:
                    return v
                if v not in code:
                    code[v] = "v" + hashlib.sha1(f"{key}|K2tok|{v}".encode()).hexdigest()[:14]
                return code[v]
            write_table(tgt, header, [[rec(v) for v in r] for r in rows])
        elif knob.startswith("K3"):
            p = int(knob[3:]) / 100
            header, rows = read_table(tgt)
            mapping = {h: "c" + hashlib.sha1(f"{key}|K3tok|{h}".encode()).hexdigest()[:10]
                       for h in header if frac(f"{key}|K3|{h}") < p}
            write_table(tgt, [mapping.get(h, h) for h in header], rows)
            rename_labels(dst, mapping, header)
        elif knob.startswith("K4"):
            f = int(knob[3:]) / 100
            for fd in (dst / "metadata").rglob("FD_results.txt"):
                lines = [l for l in fd.read_text(encoding="utf-8").splitlines() if l.strip()]
                kept = [l for l in lines if frac(f"{key}|K4|{fd.parent.name}|{l}") < f]
                fd.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        rehash(dst)
        res[knob] = sha(tgt)
    return ds, sc, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="test: only the first N base scenarios")
    a = ap.parse_args()
    cases = base_cases(a.per_stratum)
    if a.limit:
        cases = cases[:a.limit] + cases[-1:]
    OUT.mkdir(exist_ok=True)
    with Pool(a.workers) as pool:
        res = list(pool.imap_unordered(build, cases))
    manifest = {"knobs": knobs(), "n_base": len(cases), "n_scenarios": len(cases) * len(knobs()),
                "cases": sorted([{"dataset": d, "scenario": s, "target_sha256": r} for d, s, r in res],
                                key=lambda x: (x["dataset"], x["scenario"]))}
    (OUT / "knobs_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(cases)} base scenarios x {len(knobs())} variants = {manifest['n_scenarios']} scenarios in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
