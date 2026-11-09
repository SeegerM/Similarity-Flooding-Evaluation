"""R7 (layer 3b): saturated and reduced FD encodings, as parallel corpus copies.

The corpus's FDs are approximate (relaxed HyFD, threshold 0.95 on a tuple-violation measure,
determinants of at most 2 columns; see each table's profile spec). Approximate FDs are not closed
under Armstrong's axioms, so the profiled set is neither saturated nor minimal. Two variants of the
FD set that reaches SF's graph are written:

  closure  G plus every FD X -> A implied by G (Armstrong, FDs treated as exact) with
           1 <= |X| <= MAX_LHS, X not a UCC and X minimal for A under G. Falleri et al.'s
           "Saturated" encoding, transferred to dependencies.
  cover    a non-redundant subset of G: an FD is dropped if the remaining ones imply it.
           Larger determinants are dropped first, then by text. Falleri's "Minimal" analogue.
           No left-reduction, so every FD kept was found by the profiler.

G is what Schematch's graph uses (SchemaGraphBuilder via DatabaseMetadata.
getMeaningfulFunctionalDependencies and DependencyFilter.filterFunctionalDependencies): FDs whose
determinant has 1 to 3 columns and is not a UCC. Key FDs (determinant is a UCC) are written back
unchanged; Schematch drops them anyway. Under constraintWeighting NONE every FD edge has weight 1,
so the variants differ from the registered `fd` / `fd_reified` arms only in which FDs exist.

Each variant is a corpus copy <out>/<variant>/<dataset>/<scenario>/: the CSVs and the ground truth
are hard links to the original files, metadata/ is copied, FD_results.txt is rewritten and
profile-status.properties gets the new FD checksum (ManagedProfiles verifies it). The original
corpus is not touched. A third variant, `same`, rewrites FD_results.txt from the parsed G without
changing it, as a control for the parser and writer: SF on `same` must reproduce SF on the
original corpus exactly. A fourth, `thin` (added 2026-10-03 after the first smoke-test results), is a
size-matched null for `cover`: a random subset of G with as many FDs as the cover, per table, seeded
by SHA-256 of dataset/scenario/side. It separates "which FDs" from "how many".

    python layer3b/r7_build.py --cases screen            # the 60 screen scenarios
    python layer3b/r7_build.py --cases screen --dry-run  # diagnostics only

Writes layer3b/r7_<cases>_diagnostics.csv (one row per scenario side and variant).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SCREEN = STUDY / "tuning" / "corpus" / "plain" / "screen"
FULL = STUDY.parent / "harness" / "generated" / "benchmark-full"
MAX_LHS = 2          # the profiler's max_arity; Schematch's own cap is 3
VARIANTS = ("same", "closure", "cover", "thin", "thin1", "thin2", "thin3")
CLUSTER_VARIANTS = ("same", "closure", "cover", "thin1", "thin2", "thin3")


# ---------------------------------------------------------------- parsing (mirrors InputReader)

def strip_annotation(value: str) -> str:
    i = value.rfind(" (pdep ")
    return (value[:i] if i >= 0 else value).strip()


def parse_lhs(text: str) -> list[str]:
    """InputReader.extractColumnsFromString: items between '[' / ',' and ',' / ']'."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"unexpected determinant {text!r}")
    inner = text[1:-1]
    return [x.strip() for x in inner.split(",")] if inner.strip() else []


def read_fds(path: Path) -> list[tuple[frozenset, str, str]]:
    """(determinant, dependant, original line) per FD; one FD per right-hand item."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split(" --> ", 1)
        lhs = parse_lhs(left)
        if not lhs:
            raise ValueError(f"empty determinant in {path}: {line!r}")  # none in the corpus (checked)
        rhs = [x.strip() for x in strip_annotation(right).split(",")]
        annotation = right[len(strip_annotation(right)):] if " (pdep " in right else ""
        for a in rhs:
            out.append((frozenset(lhs), a, f"{left} --> {a}{annotation}" if len(rhs) > 1 else line))
    return out


def read_uccs(path: Path) -> set[frozenset]:
    if not path.is_file():
        return set()
    return {frozenset(parse_lhs(l)) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip() and l.strip() != "[]"}


def fd_line(lhs: frozenset, a: str) -> str:
    return f"[{', '.join(sorted(lhs))}] --> {a}"


# ---------------------------------------------------------------- FD logic

def closure(x: frozenset, fds: list[tuple[frozenset, str]]) -> set:
    c = set(x)
    changed = True
    while changed:
        changed = False
        for lhs, a in fds:
            if a not in c and lhs <= c:
                c.add(a)
                changed = True
    return c


def saturate(g: list[tuple[frozenset, str]], columns: list[str], uccs: set[frozenset]) -> list[tuple[frozenset, str]]:
    """G plus implied FDs X -> A, 1 <= |X| <= MAX_LHS, X no UCC, X minimal for A under G."""
    have = set(g)
    clos = {}
    for k in range(1, MAX_LHS + 1):
        for x in combinations(sorted(columns), k):
            clos[frozenset(x)] = closure(frozenset(x), g)
    added = []
    for x, cx in clos.items():
        if x in uccs:
            continue
        for a in sorted(cx - x):
            if (x, a) in have:
                continue
            if any(a in clos[frozenset(y)] for k in range(1, len(x)) for y in combinations(sorted(x), k)):
                continue  # not minimal
            added.append((x, a))
    return list(g) + added


def nonredundant(g: list[tuple[frozenset, str]], key=lambda f: fd_line(*f)) -> list[tuple[frozenset, str]]:
    order = sorted(set(g), key=lambda f: (-len(f[0]), key(f)))
    keep = set(order)
    for f in order:
        rest = [h for h in keep if h != f]
        if f[1] in closure(f[0], rest):
            keep.discard(f)
    return [f for f in g if f in keep]


def left_reducible(g: list[tuple[frozenset, str]]) -> int:
    """Diagnostic only: FDs with an extraneous determinant column under G."""
    return sum(1 for lhs, a in g if len(lhs) > 1 and any(a in closure(lhs - {b}, g) for b in lhs))


# ---------------------------------------------------------------- diagnostics

def projected_edges(g):
    return {(b, a) for lhs, a in g for b in lhs if b != a}


def wl_classes(columns, edges, rounds=10) -> int:
    """Colour refinement on the directed projected FD graph, uniform start. Columns in one class
    are structurally interchangeable for the FD channel; a complete graph collapses to one class."""
    succ, pred = defaultdict(list), defaultdict(list)
    for b, a in edges:
        succ[b].append(a)
        pred[a].append(b)
    colour = {c: 0 for c in columns}
    for _ in range(rounds):
        sig = {c: (colour[c], tuple(sorted(Counter(colour[x] for x in succ[c]).items())),
                   tuple(sorted(Counter(colour[x] for x in pred[c]).items()))) for c in columns}
        ids = {s: i for i, s in enumerate(sorted(set(sig.values())))}
        new = {c: ids[sig[c]] for c in columns}
        if len(set(new.values())) == len(set(colour.values())):
            colour = new
            break
        colour = new
    return len(set(colour.values()))


# ---------------------------------------------------------------- corpus

def cases(which: str, regime: str = "plain") -> list[tuple[str, str, Path]]:
    if which == "screen":
        screen = STUDY / "tuning" / "corpus" / regime / "screen"
        return [(d.name, s.name, s.resolve()) for d in sorted(screen.iterdir()) if d.is_dir()
                for s in sorted(d.iterdir())]
    if which == "all":  # the 382 protocol cases
        sys.path.insert(0, str(STUDY))
        from score_seeds import CORPUS, cases_for
        return [(d, s, CORPUS[regime] / d / s) for d, s, _ in cases_for(regime)]
    raise SystemExit(which)


def header(csv_path: Path) -> list[str]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def link_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        q = dst / p.name
        if p.is_dir():
            link_tree(p, q)
        elif not q.exists():
            try:
                os.link(p, q)
            except OSError:
                shutil.copyfile(p, q)


def side_variants(scen: Path, side: str):
    meta = scen / "metadata" / side / side
    table = scen / side / f"{side}.csv"
    cols = [f"{side}.csv.{c}" for c in header(table)]
    raw = read_fds(meta / "FD_results.txt")
    uccs = read_uccs(meta / "UCC_results.txt")
    colset = set(cols)
    unknown = {c for lhs, a, _ in raw for c in (*lhs, a)} - colset
    if unknown:
        raise SystemExit(f"{scen}/{side}: FD names that are not '<side>.csv.<label>': {sorted(unknown)[:3]}")
    graph = [(lhs, a) for lhs, a, _ in raw if 1 <= len(lhs) <= 3 and lhs not in uccs]
    keyfds = [line for lhs, a, line in raw if not (1 <= len(lhs) <= 3 and lhs not in uccs)]
    original = {(lhs, a): line for lhs, a, line in raw}
    spec = json.loads((scen / "metadata" / "profiling.json").read_text(encoding="utf-8"))[side]["fd"]
    prof = Path(spec["path"])
    pspec = json.loads(prof.read_text(encoding="utf-8"))["spec"] if prof.is_file() else {}
    # Ties in the cover's order and the random draws are broken by the FD's text under the *plain*
    # column names, so the MD5 regime gets the same FD sets as plain (its labels are salted tokens;
    # transform-manifest.json maps plain label -> token). On plain this is the identity.
    plain_name = {}
    tm = scen / "transform-manifest.json"
    if tm.is_file():
        columns = json.loads(tm.read_text(encoding="utf-8"))["sides"][side]["tables"][0]["columns"]
        plain_name = {f"{side}.csv.{c['token']}": f"{side}.csv.{label}" for label, c in columns.items()}
        if set(plain_name) != colset:
            raise SystemExit(f"{scen}/{side}: transform manifest does not cover the table's columns")
    canon = lambda f: fd_line(frozenset(plain_name.get(c, c) for c in f[0]), plain_name.get(f[1], f[1]))
    cover = nonredundant(graph, canon)
    out = {"same": graph, "closure": saturate(graph, cols, uccs), "cover": cover}
    # size-matched random nulls: `thin` is the smoke test's single draw (seed without a draw index,
    # kept for EXPLORATORY X8); thin1..thin3 are the registered draws (Amendment R7, §7)
    for v, key in [("thin", f"{scen.parent.name}/{scen.name}/{side}")] + \
            [(f"thin{d}", f"{scen.parent.name}/{scen.name}/{side}/{d}") for d in (1, 2, 3)]:
        rng = random.Random(hashlib.sha256(key.encode()).hexdigest())
        keep = set(rng.sample(sorted(set(graph), key=canon), len(set(cover))))
        out[v] = [f for f in graph if f in keep]
    rows, files = [], {}
    for v, g in out.items():
        lines = keyfds + [original.get(f, fd_line(*f)) for f in g]
        files[v] = "\n".join(lines) + ("\n" if lines else "")
        e = projected_edges(g)
        rows.append({"side": side, "variant": v, "columns": len(cols), "fds_file": len(raw),
                     "key_fds": len(keyfds), "fds_graph": len(g),
                     "fds_lhs2": sum(1 for lhs, _ in g if len(lhs) == 2),
                     "edges_quick": len(e), "density_quick": round(len(e) / max(1, len(cols) * (len(cols) - 1)), 4),
                     "wl_classes": wl_classes(cols, e),
                     "left_reducible": left_reducible(graph) if v == "same" else "",
                     "prof_threshold": pspec.get("threshold", ""), "prof_max_arity": pspec.get("max_arity", "")})
    return rows, files


def write_variant(dataset, name, scen: Path, out: Path, variant: str, files: dict[str, str]):
    dst = out / variant / dataset / name
    for sub in ("source", "target", "ground_truth"):
        link_tree(scen / sub, dst / sub)
    for p in scen.iterdir():
        if p.is_file() and not (dst / p.name).exists():
            shutil.copyfile(p, dst / p.name)
    if (dst / "metadata").exists():
        shutil.rmtree(dst / "metadata")
    shutil.copytree(scen / "metadata", dst / "metadata")
    status = (dst / "metadata" / "profile-status.properties").read_text(encoding="utf-8").splitlines()
    for side, text in files.items():
        path = dst / "metadata" / side / side / "FD_results.txt"
        path.write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        status = [f"{side}.fd.sha256={digest}" if l.startswith(f"{side}.fd.sha256=") else l for l in status]
    (dst / "metadata" / "profile-status.properties").write_text("\n".join(status) + "\n", encoding="utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_status(scen: Path, files: dict[str, str]) -> str:
    status = (scen / "metadata" / "profile-status.properties").read_text(encoding="utf-8").splitlines()
    for side, text in files.items():
        digest = sha(text.encode("utf-8"))
        status = [f"{side}.fd.sha256={digest}" if l.startswith(f"{side}.fd.sha256=") else l for l in status]
    return "\n".join(status) + "\n"


def bundle(path: Path, regime: str, variants: list[str], built: list[tuple[str, str, Path, dict]]) -> None:
    """Only the changed files, for r7_apply_remote.py: per variant and scenario the two FD files and
    profile-status.properties, plus manifest.json with every file's SHA-256 and the SHA-256 of the
    original FD files and status file (the cluster's corpus must match them)."""
    import io
    import tarfile
    manifest = {"regime": regime, "root": f"r7_{regime}", "variants": variants, "scenarios": []}
    with tarfile.open(path, "w:gz") as tar:
        def add(name: str, data: bytes):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
        for dataset, name, scen, per_side in built:
            base = {"metadata/profile-status.properties":
                    sha((scen / "metadata" / "profile-status.properties").read_bytes())}
            for side in ("source", "target"):
                rel = f"metadata/{side}/{side}/FD_results.txt"
                base[rel] = sha((scen / rel).read_bytes())
            entry = {"dataset": dataset, "scenario": name, "base": base, "files": {}}
            for v in variants:
                files = {s: per_side[s][v] for s in per_side}
                payload = {f"metadata/{s}/{s}/FD_results.txt": t.encode("utf-8") for s, t in files.items()}
                payload["metadata/profile-status.properties"] = new_status(scen, files).encode("utf-8")
                for rel, data in payload.items():
                    add(f"r7_{regime}/{v}/{dataset}/{name}/{rel}", data)
                    entry["files"].setdefault(v, {})[rel] = sha(data)
            manifest["scenarios"].append(entry)
        add("manifest.json", json.dumps(manifest, indent=1).encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="screen", choices=("screen", "all"))
    ap.add_argument("--regime", default="plain", choices=("plain", "md5"))
    ap.add_argument("--out", default="", help="default: data/r7_<regime>")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variants", default="same,closure,cover,thin", help="corpus copies to (re)write")
    ap.add_argument("--bundle", default="", help="write only the changed files of CLUSTER_VARIANTS to this "
                                                 ".tar.gz (no local corpus copies)")
    a = ap.parse_args()
    out = Path(a.out or STUDY / "data" / f"r7_{a.regime}") / a.cases
    rows, built = [], []
    for dataset, name, scen in cases(a.cases, a.regime):
        per_side = {}
        for side in ("source", "target"):
            r, files = side_variants(scen, side)
            for x in r:
                rows.append({"dataset": dataset, "scenario": name, **x})
            per_side[side] = files
        if a.bundle:
            built.append((dataset, name, scen, per_side))
        elif not a.dry_run:
            for v in a.variants.split(","):
                write_variant(dataset, name, scen, out, v, {s: per_side[s][v] for s in per_side})
    if a.bundle:
        bundle(Path(a.bundle), a.regime, list(CLUSTER_VARIANTS), built)
        print(f"bundle -> {a.bundle} ({Path(a.bundle).stat().st_size / 1e6:.1f} MB, {len(built)} scenarios)")
    diag = HERE / (f"r7_{a.cases}_diagnostics.csv" if a.regime == "plain" else f"r7_{a.regime}_{a.cases}_diagnostics.csv")
    with diag.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["variant"]]["fds_graph"].append(r["fds_graph"])
        by[r["variant"]]["edges_quick"].append(r["edges_quick"])
        by[r["variant"]]["density_quick"].append(r["density_quick"])
        by[r["variant"]]["wl_share"].append(r["wl_classes"] / r["columns"])
    print(f"{len(rows) // len(VARIANTS)} tables; diagnostics -> {diag}")
    for v, d in by.items():
        print(f"  {v:8s} " + "  ".join(f"{k} mean {sum(x) / len(x):.3f} max {max(x):.3f}" for k, x in d.items()))
    specs = Counter((r["prof_threshold"], r["prof_max_arity"]) for r in rows if r["variant"] == "same")
    print(f"  profile specs (threshold, max_arity): {dict(specs)}")
    if not a.dry_run and not a.bundle:
        print(f"corpus copies -> {out}/<variant>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
