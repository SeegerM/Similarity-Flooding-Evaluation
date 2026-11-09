"""Two more databases for the multi-table corpus: Lahman and MusicBrainz (EXPLORATORY; prepared 2026-10-05, the authors'
go). Same construction as the independent corpus (harness/INDEPENDENT_SAMPLING_20260929.md, data/convert_indep.py):
sample root rows, close over foreign keys, build a normalised schema N and two denormalisations D1 and D2, and pair them
(N-D1, N-D2, D1-D2) under three sampling seeds and three overlap conditions (identical, half, disjoint root sets).
The ground truth is schema provenance (grain_confirm.corresponds), so it is many-to-many where a denormalisation copies
parent attributes into several tables.

Reused unchanged from the harness: grain_confirm.build_variant, labels_for, corresponds, read_rows (CSV), and the
sampling rules of independent_sampling.choose/closure/canonical, copied here verbatim so the frozen study module is
not imported. Raw data: $DATA_ROOT (read-only), the same releases the controlled corpus used.

Database definitions, fixed before any matcher ran:
  Lahman (CSV): root People (playerID), 1,500 players per side. Children Batting, Salaries, AwardsPlayers,
    CollegePlaying, HallOfFame (by playerID); parent Schools (CollegePlaying.schoolID). Teams is left out: its key is
    composite (yearID, teamID), which the closure does not support.
  MusicBrainz (headerless PostgreSQL TSV dumps; \\N = null): root release (id), 1,500 per side. Child medium; parents
    release_group, artist_credit, release_status, language, medium_format, release_group_primary_type.
Target labels: Lahman's camelCase becomes snake_case (grain_confirm.target_label, as for AdventureWorks and Chinook);
MusicBrainz's snake_case becomes camelCase (the same transformation reversed), so the two sides never share a label
by construction.

Output: data/indep2_plain/Indep-<db>/s<seed>_<condition>_<a>-<b>/ in convert_indep.py's layout.

    ../harness/.venv-bdi/Scripts/python.exe data/make_multitable_ext.py [--db Lahman|MusicBrainz]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent.parent / "harness"
sys.path.insert(0, str(HARNESS))
import grain_confirm as gc  # noqa: E402

RAW = Path(r"$DATA_ROOT")
OUT = HERE / "indep2_plain"
SEEDS = (20260929, 20260930, 20260931)
CONDITIONS = ("identical", "half", "disjoint")
N_ROOTS = 1500
NULLS = {"", "null", "none", "nan", "n/a", "na", "\\n"}

LAHMAN = {
    "name": "Lahman", "root": RAW / "Lahman", "delimiter": ",", "base": "People",
    "tables": {"People": ("People.csv", "playerID"), "Batting": ("Batting.csv", None),
               "Salaries": ("Salaries.csv", None), "AwardsPlayers": ("AwardsPlayers.csv", None),
               "CollegePlaying": ("CollegePlaying.csv", None), "HallOfFame": ("HallOfFame.csv", None),
               "Schools": ("Schools.csv", "schoolID")},
    "fks": {("Batting", "playerID"): "People", ("Salaries", "playerID"): "People",
            ("AwardsPlayers", "playerID"): "People", ("CollegePlaying", "playerID"): "People",
            ("HallOfFame", "playerID"): "People", ("CollegePlaying", "schoolID"): "Schools"},
    "variants": {
        "N": {t: (t, []) for t in ("People", "Batting", "Salaries", "AwardsPlayers", "CollegePlaying", "HallOfFame",
                                   "Schools")},
        "D1": {"batting": ("Batting", [("playerID", None)]), "salaries": ("Salaries", []),
               "awards": ("AwardsPlayers", []), "college": ("CollegePlaying", [("schoolID", None)])},
        "D2": {"people": ("People", []), "salaries": ("Salaries", [("playerID", None)]),
               "hall_of_fame": ("HallOfFame", [("playerID", None)]), "batting": ("Batting", []),
               "college": ("CollegePlaying", [("schoolID", None)])}}}

MB_HEADERS = {
    "release": ["id", "gid", "name", "artist_credit", "release_group", "status", "packaging", "language", "script",
                "barcode", "comment", "edits_pending", "quality", "last_updated"],
    "medium": ["id", "release", "position", "format", "name", "edits_pending", "last_updated", "track_count"],
    "release_group": ["id", "gid", "name", "artist_credit", "type", "comment", "edits_pending", "last_updated"],
    "artist_credit": ["id", "name", "artist_count", "ref_count", "created", "edits_pending", "gid"],
    "release_status": ["id", "name", "parent", "child_order", "description", "gid"],
    "language": ["id", "iso_code_2t", "iso_code_2b", "iso_code_1", "name", "frequency", "iso_code_3"],
    "medium_format": ["id", "name", "parent", "child_order", "year", "has_discids", "description", "gid"],
    "release_group_primary_type": ["id", "name", "parent", "child_order", "description", "gid"]}
MUSICBRAINZ = {
    "name": "MusicBrainz", "root": RAW / "Musicbrainz", "headers": MB_HEADERS, "base": "release",
    "tables": {t: (t, "id") for t in MB_HEADERS},
    "fks": {("medium", "release"): "release", ("medium", "format"): "medium_format",
            ("release", "release_group"): "release_group", ("release", "artist_credit"): "artist_credit",
            ("release", "status"): "release_status", ("release", "language"): "language",
            ("release_group", "type"): "release_group_primary_type",
            ("release_group", "artist_credit"): "artist_credit"},
    "variants": {
        "N": {t: (t, []) for t in MB_HEADERS},
        "D1": {"release": ("release", [("release_group", None), ("status", None), ("language", None)]),
               "medium": ("medium", [("format", None)]), "artist_credit": ("artist_credit", [])},
        "D2": {"medium": ("medium", [("release", None)]),
               "release_group": ("release_group", [("type", None), ("artist_credit", None)]),
               "release_status": ("release_status", []), "language": ("language", [])}}}
DBS = {d["name"]: d for d in (LAHMAN, MUSICBRAINZ)}

_csv_read_rows = gc.read_rows
_target_label = gc.target_label


def read_rows(db, table, keep=None):
    """CSV through the harness reader; MusicBrainz's headerless dumps here (\\N -> empty)."""
    if "headers" not in db:
        yield from _csv_read_rows(db, table, keep)
        return
    header = db["headers"][table]
    with (db["root"] / db["tables"][table][0]).open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) != len(header):
                raise ValueError(f"{table}: ragged row")
            rec = {h: ("" if v == "\\N" else v) for h, v in zip(header, row)}
            if keep is None or keep(rec):
                yield rec


def camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def target_label(db_name, raw_label):
    return camel(raw_label) if db_name == "MusicBrainz" else _target_label(db_name, raw_label)


gc.read_rows = read_rows            # build_variant and labels_for look these up in their module
gc.target_label = target_label


# --- independent_sampling.choose / closure / canonical, verbatim except gc.read_rows (patched above) ---
def choose(keys, n, seed, condition, source=None):
    rng = random.Random(seed)
    if source is None:
        return set(rng.sample(keys, n))
    if condition == "identical":
        return set(source)
    if condition == "independent":
        return set(rng.sample(keys, n))
    shared = n // 2 if condition == "half" else 0
    outside = [x for x in keys if x not in source]
    return set(rng.sample(sorted(source), shared) + rng.sample(outside, n - shared))


def closure(db, selected):
    base = db["base"]
    bk = db["tables"][base][1]
    rows = {base: list(gc.read_rows(db, base, lambda r: r[bk] in selected))}
    keys = {base: set(selected)}
    for (t, c), ref in db["fks"].items():
        if ref == base and t not in rows:
            rows[t] = list(gc.read_rows(db, t, lambda r, c=c: r[c] in selected))
    changed = True
    while changed:
        changed = False
        for (t, c), ref in db["fks"].items():
            if t not in rows or ref == base:
                continue
            needed = {r[c] for r in rows[t] if r[c]}
            have = keys.get(ref, set())
            if needed - have:
                k = db["tables"][ref][1]
                want = needed | have
                new = list(gc.read_rows(db, ref, lambda r, k=k: r[k] in want))
                got = {r[k] for r in new}
                assert needed <= got, (db["name"], t, c, "missing FK", len(needed - got))
                rows[ref] = new
                keys[ref] = got
                changed = True
    for t in db["tables"]:
        assert rows.get(t), (db["name"], t, "empty table")
    return rows


def canonical(v):
    return "" if v.strip().lower() in NULLS else v.strip()


def representation(db, variant, rows, side):
    out = {}
    for table, (prov, trows, raw) in sorted(gc.build_variant(db, rows, variant).items()):
        names = gc.labels_for(db, prov, raw, side, "plain", variant, table)
        out[table] = (names, prov, [[canonical(v) for v in r] for r in trows])
    return out


def write_scenario(folder: Path, src: dict, tgt: dict, meta: dict) -> int:
    for side, rep in (("source", src), ("target", tgt)):
        (folder / side).mkdir(parents=True, exist_ok=True)
        for t, (names, _, body) in rep.items():
            with (folder / side / f"{t}.csv").open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(names)
                w.writerows(body)
    sc = [(t, n, p) for t in sorted(src) for n, p in zip(src[t][0], src[t][1])]
    tc = [(t, n, p) for t in sorted(tgt) for n, p in zip(tgt[t][0], tgt[t][1])]
    gt = [[1 if gc.corresponds(sp, tp) else 0 for _, _, tp in tc] for _, _, sp in sc]
    with (folder / "ground_truth_full.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(gt)
    (folder / "ground_truth").mkdir(exist_ok=True)
    for st in sorted(src):
        for tt in sorted(tgt):
            rows_ = [[1 if gc.corresponds(sp, tp) else 0 for tp in tgt[tt][1]] for sp in src[st][1]]
            with (folder / "ground_truth" / f"{st}___{tt}.csv").open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerows(rows_)
    pos = sum(map(sum, gt))
    meta.update({"source_columns": [f"{t}.{n}" for t, n, _ in sc], "target_columns": [f"{t}.{n}" for t, n, _ in tc],
                 "positives": pos, "source_rows": {t: len(v[2]) for t, v in src.items()},
                 "target_rows": {t: len(v[2]) for t, v in tgt.items()}})
    (folder / "provenance.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return pos


def run(dbname: str) -> None:
    db = DBS[dbname]
    key = db["tables"][db["base"]][1]
    population = [r[key] for r in read_rows(db, db["base"])]
    assert len(set(population)) == len(population) and len(population) >= 2 * N_ROOTS
    for seed in SEEDS:
        src = choose(population, N_ROOTS, seed, "source")
        sr = closure(db, src)
        sv = {v: representation(db, v, sr, "source") for v in db["variants"]}
        for condition in CONDITIONS:
            sel = choose(population, N_ROOTS, seed + 100000, condition, src)
            tr = sr if condition == "identical" else closure(db, sel)
            tv = {v: representation(db, v, tr, "target") for v in db["variants"]}
            for a, b, _ in gc.PAIRINGS[:3]:
                folder = OUT / f"Indep-{dbname}" / f"s{seed}_{condition}_{a}-{b}"
                if folder.exists():
                    shutil.rmtree(folder)
                pos = write_scenario(folder, sv[a], tv[b], {"database": dbname, "seed": seed, "condition": condition,
                                                            "pairing": f"{a}-{b}", "roots_shared": len(src & sel)})
                print(dbname, seed, condition, f"{a}-{b}", "positives", pos, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", choices=sorted(DBS), action="append")
    a = ap.parse_args()
    for name in a.db or sorted(DBS):
        run(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
