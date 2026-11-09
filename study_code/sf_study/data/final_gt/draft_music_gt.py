"""FINAL VALIDATION, part A (2026-10-06; DRAFT for validation): ground truth for three independently designed
music schemas in $DATA_ROOT: MusicBrainz (mb1), Discogs (dis2), FreeDB (fdb1).

Drafted from schema, table and column names and a look at sample values only; no matcher output was read. Same rules as
draft_bib_gt.py:
  - each listed column gets (entity, attribute, confidence); C = certain, L = likely (the same attribute in a different
    representation, e.g. length in ms vs "10:12", or a judgement call);
  - unlisted columns have no counterpart: system columns (gid/UUID, edits_pending, last_updated, data_quality, created),
    images, URLs, videos, relationship tables without a counterpart, and MB lookup ids;
  - two columns correspond if attribute and entity agree; the pair's confidence is the lower of the two;
  - keys: key <-> key and link FK <-> link FK, not FK <-> referenced key. Names map to names; an FK id never maps to a name
    (e.g. MB release_label.label (id) is excluded; label.name maps to Discogs releases_labels.name).
Judgement calls, flagged L:
  - Discogs releases.status is a submission status ("Accepted"), not MB's release status ("Official"): excluded.
  - MB artist_alias ~ Discogs artists_namevariations (spelling variants) C; Discogs artists_aliases (pseudonyms) L.
  - FreeDB discs.dtitle is "artist / title": maps to the release title, L.
  - Genres: MB tag names vs Discogs genres/styles and FreeDB dgenre, L (tags are a folksonomy).
  - MB track.number ("A1", "4") ~ Discogs tracklist.position ("A", "AA") C; MB track.position (1, 2, ...) and FreeDB tracks.n
    (0, 1, ...) are ordinal positions: L to the printed positions, C to each other.

Decisions after the authors' review (2026-10-06):
  - Dropped comment <-> profile/notes (MB comments are disambiguation notes, Discogs profiles/notes are free text).
  - MB track.medium <-> FreeDB tracks.discid is C (disc <-> disc); either <-> Discogs tracklist.release_id is L.
  - MB alias names <-> Discogs name variations are L (localised names vs spelling variants); the link keys stay C.
  - FreeDB dgenre (CDDB category) <-> Discogs styles is excluded; dgenre <-> Discogs genres and MB tags <-> genres/styles L.

    python data/final_gt/draft_music_gt.py   -> data/final_gt/music_<s>_<t>.csv + music_concepts.csv
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent

MB = {
    "artist": {"id": ("ARTIST", "id", "C"), "name": ("ARTIST", "name", "C")},
    "artist_alias": {"artist": ("ALIAS", "artist_fk", "C"), "name": ("ALIAS", "name", "L")},
    "artist_credit_name": {"artist": ("CREDIT", "artist_fk", "L"), "name": ("CREDIT", "name", "L")},
    "label": {"id": ("LABEL", "id", "C"), "name": ("LABEL", "name", "C")},
    "release": {"id": ("RELEASE", "id", "C"), "name": ("RELEASE", "title", "C"),
                "release_group": ("RELEASE", "group_fk", "C"), "barcode": ("RELEASE", "barcode", "L")},
    "release_group": {"id": ("GROUP", "id", "C"), "name": ("GROUP", "title", "C")},
    "release_country": {"date_year": ("RELEASE", "year", "L")},
    "area": {"name": ("RELEASE", "country", "L")},
    "release_label": {"catalog_number": ("RELEASE", "catno", "C")},
    "medium_format": {"name": ("FORMAT", "name", "C")},
    "track": {"id": ("TRACK", "id", "C"), "name": ("TRACK", "title", "C"), "number": ("TRACK", "printed_position", "C"),
              "position": ("TRACK", "ordinal", "C"), "length": ("TRACK", "duration", "L"),
              "medium": ("TRACK", "disc_fk", "C")},
    "recording": {"name": ("TRACK", "title", "L"), "length": ("TRACK", "duration", "L")},
    "cdtoc": {"freedb_id": ("DISC", "freedb_id", "C")},
    "tag": {"name": ("GENRE", "tag", "L")},
}
DIS = {
    "artists": {"id": ("ARTIST", "id", "C"), "name": ("ARTIST", "name", "C")},
    "artists_namevariations": {"artist_id": ("ALIAS", "artist_fk", "C"), "name": ("ALIAS", "name", "C")},
    "artists_aliases": {"artist_id": ("ALIAS", "artist_fk", "L"), "name": ("ALIAS", "name", "L")},
    "labels": {"id": ("LABEL", "id", "C"), "name": ("LABEL", "name", "C")},
    "releases": {"id": ("RELEASE", "id", "C"), "title": ("RELEASE", "title", "C"), "master_id": ("RELEASE", "group_fk", "C"),
                 "country": ("RELEASE", "country", "L"),
                 "released": ("RELEASE", "year", "L")},
    "masters": {"id": ("GROUP", "id", "C"), "title": ("GROUP", "title", "C"), "year": ("RELEASE", "year", "L")},
    "releases_artists": {"id": ("CREDIT", "artist_fk", "L"), "name": ("CREDIT", "name", "L")},
    "masters_artists": {"id": ("CREDIT", "artist_fk", "L"), "name": ("CREDIT", "name", "L")},
    "releases_identifiers": {"value": ("RELEASE", "barcode", "L")},
    "releases_labels": {"name": ("LABEL", "name", "L"), "catno": ("RELEASE", "catno", "C")},
    "releases_formats": {"name": ("FORMAT", "name", "C")},
    "releases_tracklist": {"track_id": ("TRACK", "id", "C"), "title": ("TRACK", "title", "C"),
                           "position": ("TRACK", "printed_position", "C"), "duration": ("TRACK", "duration", "L"),
                           "release_id": ("TRACK", "release_fk", "L")},
    "releases_genres": {"genre": ("GENRE", "genre", "L")}, "masters_genres": {"genre": ("GENRE", "genre", "L")},
    "releases_styles": {"style": ("GENRE", "style", "L")}, "masters_styles": {"style": ("GENRE", "style", "L")},
}
FDB = {
    "discs": {"discid": ("DISC", "freedb_id", "C"), "dtitle": ("RELEASE", "title", "L"), "dyear": ("RELEASE", "year", "L"),
              "dgenre": ("GENRE", "genre", "L")},
    "tracks": {"discid": ("TRACK", "disc_fk", "C"), "n": ("TRACK", "ordinal", "C"), "ttitle": ("TRACK", "title", "C")},
}
SCHEMAS = {"mb1": MB, "dis2": DIS, "fdb1": FDB}
PAIRS = [("mb1", "dis2"), ("mb1", "fdb1"), ("dis2", "fdb1")]
# cross-attribute pairs that are the same information in two attribute names (ordinal vs printed position)
EQUIV = {frozenset({"ordinal", "printed_position"}),      # the same position, printed vs ordinal
         frozenset({"disc_fk", "release_fk"}),             # a track's container: one disc vs the whole release
         frozenset({"tag", "genre"}), frozenset({"tag", "style"})}   # MB tags vs Discogs genres / styles


def main() -> None:
    with (HERE / "music_concepts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["schema", "table", "column", "entity", "attribute", "confidence"])
        for s, sch in SCHEMAS.items():
            for t, cols in sch.items():
                for c, (e, a, conf) in cols.items():
                    w.writerow([s, t, c, e, a, conf])
    for s, t in PAIRS:
        rows = []
        for (ts, cs), (tt, ct) in itertools.product(
                [(tb, c) for tb, cols in SCHEMAS[s].items() for c in cols],
                [(tb, c) for tb, cols in SCHEMAS[t].items() for c in cols]):
            es, as_, cfs = SCHEMAS[s][ts][cs]
            et, at, cft = SCHEMAS[t][tt][ct]
            if es != et:
                continue
            if as_ == at:
                conf = "C" if cfs == cft == "C" else "L"
            elif frozenset({as_, at}) in EQUIV:
                conf = "L"
            else:
                continue
            rows.append([ts, cs, tt, ct, as_ if as_ == at else f"{as_}~{at}", conf])
        with (HERE / f"music_{s}_{t}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source_table", "source_column", "target_table", "target_column", "attribute", "confidence"])
            w.writerows(rows)
        print(f"{s} -> {t}: {len(rows)} correspondences ({sum(r[5] == 'C' for r in rows)} certain)")


if __name__ == "__main__":
    main()
