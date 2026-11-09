"""Amendment 12 (final validation, draft reference/AMENDMENT_FINAL_DRAFT.md), Part A: the independently
designed schema pairs from $DATA_ROOT (read-only), imported into the study's multi-table layout (as
data/convert_synthea.py writes it):

    data/final_indep/<pair>/full/source/<table>.csv, target/<table>.csv     (tables copied row for row, all rows)
                                ground_truth/<s>___<t>.csv               (0/1, rows = source columns of s)
                                ground_truth_full.csv                    (all source x all target columns; tables by
                                                                          name, columns in file order)
                                provenance.json                          ("table.column" labels, positives, rows)

<pair> = the truth file stem in data/final_gt/ (bib_s1a_s2b, ..., music_dis2_fdb1); source = the first schema of the stem.
Tables: every <table>.csv of the schema folder, except mb1, which is restricted to the 26 tables of the draft §3.
Truth: the rows of data/final_gt/<pair>.csv with confidence C or L (primary truth); every named table and column must
exist in the imported tables, else the import stops (no silent drop). Columns absent from the truth file are
non-matches. Writes data/final_indep/import_manifest.json (importer hash, per pair: sizes, positives, SHA-256 of
ground_truth_full.csv and of every table file).

    python data/convert_final_indep.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path("$DATA_ROOT")
GT = HERE / "final_gt"
OUT = HERE / "final_indep"
MB1_TABLES = ["artist", "artist_alias", "artist_credit", "artist_credit_name", "artist_tag", "artist_type", "gender", "label",
              "label_alias", "label_type", "release", "release_group", "release_group_primary_type", "release_group_tag",
              "release_country", "release_label", "release_status", "area", "medium", "medium_format", "medium_cdtoc",
              "cdtoc", "track", "recording", "isrc", "tag"]
PAIRS = ["bib_s1a_s2b", "bib_s1a_s3b", "bib_s1a_s4b", "bib_s2b_s3a", "bib_s2b_s4b", "bib_s3a_s4b",
         "music_mb1_dis2", "music_mb1_fdb1", "music_dis2_fdb1"]
csv.field_size_limit(1 << 30)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    rows = list(csv.reader(text.splitlines()))
    hdr = rows[0] if rows else []
    if len(set(hdr)) != len(hdr):
        raise SystemExit(f"{path}: duplicate column names {hdr}")
    body = [(r + [""] * (len(hdr) - len(r)))[: len(hdr)] for r in rows[1:] if r]
    return hdr, body


def tables_of(schema: str) -> dict[str, Path]:
    files = {p.stem: p for p in sorted((RAW / schema).glob("*.csv"))}
    if schema == "mb1":
        missing = [t for t in MB1_TABLES if t not in files]
        if missing:
            raise SystemExit(f"mb1: tables of the draft not found: {missing}")
        files = {t: files[t] for t in MB1_TABLES}
    return files


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = {"importer_sha256": sha(Path(__file__)), "source": str(RAW), "pairs": {}}
    for pair in PAIRS:
        _, s_schema, t_schema = pair.split("_")
        data = {}
        for side, schema in (("source", s_schema), ("target", t_schema)):
            for t, path in tables_of(schema).items():
                data[(side, t)] = read_table(path) + (path,)
        with (GT / f"{pair}.csv").open(encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r["confidence"].strip() in ("C", "L")]
        truth = set()
        for r in rows:
            s = (r["source_table"].strip(), r["source_column"].strip())
            t = (r["target_table"].strip(), r["target_column"].strip())
            for side, (tab, col) in (("source", s), ("target", t)):
                if (side, tab) not in data or col not in data[(side, tab)][0]:
                    raise SystemExit(f"{pair}: truth names {side} column {tab}.{col}, which is not in the imported tables")
            truth.add((s, t))
        folder = OUT / pair / "full"
        src = sorted(t for side, t in data if side == "source")
        tgt = sorted(t for side, t in data if side == "target")
        for side, tabs in (("source", src), ("target", tgt)):
            (folder / side).mkdir(parents=True, exist_ok=True)
            for t in tabs:
                hdr, body, _ = data[(side, t)]
                with (folder / side / f"{t}.csv").open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(hdr)
                    w.writerows(body)
        sc = [(t, c) for t in src for c in data[("source", t)][0]]
        tc = [(t, c) for t in tgt for c in data[("target", t)][0]]
        (folder / "ground_truth").mkdir()
        with (folder / "ground_truth_full.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            for s in sc:
                w.writerow([1 if (s, t) in truth else 0 for t in tc])
        for st in src:
            for tt in tgt:
                with (folder / "ground_truth" / f"{st}___{tt}.csv").open("w", encoding="utf-8", newline="") as fh:
                    w = csv.writer(fh)
                    for c in data[("source", st)][0]:
                        w.writerow([1 if ((st, c), (tt, d)) in truth else 0 for d in data[("target", tt)][0]])
        n_pos = sum(1 for s in sc for t in tc if (s, t) in truth)
        assert n_pos == len(truth), (pair, n_pos, len(truth))
        prov = {"kind": "full", "pair": pair, "truth_file": f"data/final_gt/{pair}.csv", "truth_rows": "C+L",
                "source_columns": [f"{t}.{c}" for t, c in sc], "target_columns": [f"{t}.{c}" for t, c in tc],
                "positives": n_pos,
                "rows": {f"{s}:{t}": len(data[(s, t)][1]) for s, t in data}}
        (folder / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
        manifest["pairs"][pair] = {
            "source_tables": len(src), "target_tables": len(tgt), "source_columns": len(sc), "target_columns": len(tc),
            "positives_C_plus_L": n_pos, "positives_C": sum(1 for r in rows if r["confidence"].strip() == "C"),
            "truth_csv_sha256": sha(GT / f"{pair}.csv"),
            "ground_truth_full_sha256": sha(folder / "ground_truth_full.csv"),
            "raw_tables_sha256": {f"{side}:{t}": sha(v[2]) for (side, t), v in sorted(data.items())},
            "rows": prov["rows"]}
        print(f"{pair}: {len(src)} x {len(tgt)} tables, {len(sc)} x {len(tc)} columns, {n_pos} positives")
    (OUT / "import_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
