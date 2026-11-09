"""FINAL VALIDATION, part A (2026-10-06; DRAFT for validation): ground truth for the four independently
designed bibliographic schemas in $DATA_ROOT (s1a, s2b, s3a/s3b, s4b).

Drafted from schema, table and column names and a look at sample values only. No matcher output was read.
Each column gets a concept (entity, attribute) and a confidence:
  C = certain     the same attribute of the same entity by name and content
  L = likely      plausible from names and sample values, but a judgement call (the authors decide)
  -               no counterpart concept (excluded)
Entities:
  PUB = any publication (generic schemas s2, s4)
  ARTICLE, BOOK, INCOLL, INPROC, MANUAL, MISC, TECH, UNPUB = typed publications (s1, s3)
  AUTHOR; AUTHLINK = the author-publication relationship
Rule: two columns correspond if they share the attribute and their entities are compatible: equal, or one side is the
generic PUB (a typed column maps to the generic one). The pair's confidence is the lower of the two.
Keys: publication and author identifiers map to identifiers. The FK columns of an author-publication link map to the
other schema's link columns (AUTHLINK). Attribute-table FKs that exist only because a schema is vertically
partitioned (s2's citkey in titles, years, ...) have no counterpart and are excluded.

Decisions after the authors' review (2026-10-06):
  - s1a misc.confloc (conference location; one row, empty) has no counterpart and is excluded.
  - s1a loc (e.g. "Brunel Un. DCS (Uxbridge Middlesex UK)") is the authors' organisation and place, not a BibTeX address.
    It maps to s4b author.affiliations ("CASC, 4th Academy, Xian, China") as L. s2b and s3 have no affiliation field.
  - Keys: link <-> link and key <-> key. A link-table FK does not also map to the key it references; that would reward
    linking any two id columns of one entity and inflate s1a's seven link tables.

    python data/final_gt/draft_bib_gt.py      -> data/final_gt/bib_<s>_<t>.csv (one row per correspondence) + concepts.csv
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
TYPED = {"ARTICLE", "BOOK", "INCOLL", "INPROC", "MANUAL", "MISC", "TECH", "UNPUB"}

# schema -> table -> column -> (entity, attribute, confidence)
S1_TYPES = {"article": ("ARTICLE", "articleid", "journal", "journal"), "book": ("BOOK", "bookid", "publisher", "publisher"),
            "incollection": ("INCOLL", "collid", "bktitle", "booktitle"),
            "inproceedings": ("INPROC", "inprocid", "bktitle", "booktitle"),
            "manual": ("MANUAL", "manid", "org", "organization"), "techreport": ("TECH", "techid", "inst", "institution")}
S1 = {}
for tab, (ent, pk, venue_col, venue_attr) in S1_TYPES.items():
    S1[tab] = {pk: (ent, "id", "C"), "title": (ent, "title", "C"), venue_col: (ent, venue_attr, "C"),
               "year": (ent, "year", "C"), "month": (ent, "month", "C"), "pages": (ent, "pages", "C"),
               "vol": (ent, "volume", "C"), "num": (ent, "number", "C"), "loc": (ent, "affiliation", "L"),
               "class": (ent, "keyword", "L"), "note": (ent, "note", "C"), "annote": (ent, "annotation", "L")}
S1["misc"] = {"miscid": ("MISC", "id", "C"), "title": ("MISC", "title", "C"), "howpub": ("MISC", "howpublished", "C"),
              "year": ("MISC", "year", "C"), "month": ("MISC", "month", "C"),
              "pages": ("MISC", "pages", "C"), "vol": ("MISC", "volume", "C"), "num": ("MISC", "number", "C"),
              "loc": ("MISC", "affiliation", "L"), "class": ("MISC", "keyword", "L"), "note": ("MISC", "note", "C"),
              "annote": ("MISC", "annotation", "L")}
for tab, (ent, pk, *_rest) in list(S1_TYPES.items()) + [("misc", ("MISC", "miscid"))]:
    link = {"article": "articlepublished", "book": "bookpublished", "incollection": "incollpublished",
            "inproceedings": "inprocpublished", "manual": "manualpublished", "techreport": "techpublished",
            "misc": "miscpublished"}[tab]
    S1[link] = {pk: (ent, "authlink_pub", "C"), "authid": ("AUTHOR", "authlink_author", "C")}
S1["author"] = {"authid": ("AUTHOR", "id", "C"), "name": ("AUTHOR", "name", "C")}

S2 = {"allbibs": {"citkey": ("PUB", "id", "C")}, "titles": {"title": ("PUB", "title", "C")},
      "years": {"yr": ("PUB", "year", "C")}, "months": {"mon": ("PUB", "month", "C")},
      "pages": {"pgrange": ("PUB", "pages", "C")}, "volumes": {"volnum": ("PUB", "volume", "C")},
      "numbers": {"num": ("PUB", "number", "C")}, "notes": {"note": ("PUB", "note", "C")},
      "authors": {"citkey": ("PUB", "authlink_pub", "C"), "autnm": ("AUTHOR", "name", "C")},
      "editors": {"ednm": ("PUB", "editor", "C")}, "abstracts": {"txt": ("PUB", "abstract", "C")},
      "addresses": {"address": ("PUB", "address", "C")}, "institutions": {"institnm": ("PUB", "institution", "C")},
      "schools": {"schoolnm": ("PUB", "school", "C")}, "isbn": {"isbnnum": ("PUB", "isbn", "C")},
      "types": {"type": ("PUB", "reporttype", "L")}, "citform": {"form": ("PUB", "entrytype", "C")},
      "journal": {"jrnlnm": ("PUB", "journal", "C")}, "publisher": {"pubnm": ("PUB", "publisher", "C")},
      "booktitle": {"bktitlenm": ("PUB", "booktitle", "C")}, "series": {"seriesnm": ("PUB", "series", "C")},
      "keyword": {"word": ("PUB", "keyword", "C")}}

S3 = {"article": {"articleid": ("ARTICLE", "id", "C"), "title": ("ARTICLE", "title", "C"),
                  "volume": ("ARTICLE", "volume", "C"), "number": ("ARTICLE", "number", "C"),
                  "pages": ("ARTICLE", "pages", "C"), "month": ("ARTICLE", "month", "C"), "year": ("ARTICLE", "year", "C"),
                  "refkey": ("ARTICLE", "refkey", "L"), "note": ("ARTICLE", "note", "C"),
                  "remarks": ("ARTICLE", "annotation", "L"), "references_": ("ARTICLE", "references", "C"),
                  "abstract": ("ARTICLE", "abstract", "C")},
      "articleauthor": {"articleid": ("ARTICLE", "authlink_pub", "C"), "authorid": ("AUTHOR", "authlink_author", "C")},
      "author": {"authorid": ("AUTHOR", "id", "C"), "name": ("AUTHOR", "name", "C")},
      "unpublished": {"unpubid": ("UNPUB", "id", "C"), "title": ("UNPUB", "title", "C"), "refkey": ("UNPUB", "refkey", "L"),
                      "note": ("UNPUB", "note", "C"), "references_": ("UNPUB", "references", "C")},
      "unpubauthor": {"unpubid": ("UNPUB", "authlink_pub", "C"), "authorid": ("AUTHOR", "authlink_author", "C")}}

S4 = {"publication": {"pid": ("PUB", "id", "C"), "title": ("PUB", "title", "C"), "abstract": ("PUB", "abstract", "C"),
                      "journal": ("PUB", "journal", "C"), "confinfo": ("PUB", "booktitle", "L"),
                      "book": ("PUB", "booktitle", "L"), "category": ("PUB", "entrytype", "L"),
                      "language": ("PUB", "language", "C")},
      "author": {"aid": ("AUTHOR", "id", "C"), "name": ("AUTHOR", "name", "C"),
                 "affiliations": ("PUB", "affiliation", "L")},
      "written": {"aid": ("AUTHOR", "authlink_author", "C"), "pid": ("PUB", "authlink_pub", "C"),
                  "pubyear": ("PUB", "year", "C"), "publisher": ("PUB", "publisher", "C")},
      "descriptor": {"descriptor": ("PUB", "keyword", "C")},
      "record": {"isbn": ("PUB", "isbn", "C"), "series": ("PUB", "series", "C"), "notes": ("PUB", "note", "L")}}

SCHEMAS = {"s1a": S1, "s2b": S2, "s3a": S3, "s3b": S3, "s4b": S4}
PAIRS = [("s1a", "s2b"), ("s1a", "s3b"), ("s1a", "s4b"), ("s2b", "s3a"), ("s2b", "s4b"), ("s3a", "s4b")]


def compatible(e1: str, e2: str) -> bool:
    if e1 == e2:
        return True
    if "PUB" in (e1, e2) and (e1 in TYPED or e2 in TYPED or e1 == e2):
        return True
    return False


def main() -> None:
    with (HERE / "concepts.csv").open("w", newline="", encoding="utf-8") as fh:
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
            if as_ == at and compatible(es, et):
                conf = "C" if cfs == cft == "C" else "L"
                rows.append([ts, cs, tt, ct, as_, conf])
        with (HERE / f"bib_{s}_{t}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source_table", "source_column", "target_table", "target_column", "attribute", "confidence"])
            w.writerows(rows)
        print(f"{s} -> {t}: {len(rows)} correspondences ({sum(r[5] == 'C' for r in rows)} certain)")


if __name__ == "__main__":
    main()
