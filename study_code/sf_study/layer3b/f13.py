"""Family 13 (semantic types) as graph information, with a rule-based typer: a training-free proxy.

No semantic-type model (Sherlock, Sato, DoDuo, Pythagoras) is installed; the authors chose "rules now, Sherlock
later" (2026-10-03). Each value gets the first matching type of TYPES (regexes and small dictionaries); a
column's type is the majority type of its non-empty values if it covers at least 60% of them, else `mixed`.
Type names are written with a trailing underscore ("country_"): Schematch's literal string match splits labels
into words and drops it, while the node value can no longer equal a column named "country" (Node.equals compares
values only for nodes without a table).

Edge files go to data/f13_plain/screen/{semtype,shuffled}/; runs and scoring use layer3b/f7.py with
SMX_L3B_FAMILY=f13:

    python layer3b/f13.py
    SMX_L3B_FAMILY=f13 python layer3b/f7.py write | run --workers 4 | score
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
from r7_build import cases  # noqa: E402

csv.field_size_limit(1 << 30)
EDGES = STUDY / "data" / "f13_plain" / "screen"
SHARE = 0.6

COUNTRIES = {c.lower() for c in """Afghanistan Albania Algeria Argentina Armenia Australia Austria Azerbaijan Bahamas
Bahrain Bangladesh Belarus Belgium Bolivia Bosnia Brazil Bulgaria Cambodia Cameroon Canada Chile China Colombia
Croatia Cuba Cyprus Czechia Denmark Ecuador Egypt Estonia Ethiopia Finland France Georgia Germany Ghana Greece
Guatemala Honduras Hungary Iceland India Indonesia Iran Iraq Ireland Israel Italy Jamaica Japan Jordan Kazakhstan
Kenya Korea Kuwait Latvia Lebanon Libya Lithuania Luxembourg Madagascar Malaysia Mali Malta Mexico Moldova Monaco
Mongolia Morocco Mozambique Namibia Nepal Netherlands Nicaragua Niger Nigeria Norway Oman Pakistan Panama Paraguay
Peru Philippines Poland Portugal Qatar Romania Russia Rwanda Senegal Serbia Singapore Slovakia Slovenia Somalia
Spain Sudan Sweden Switzerland Syria Taiwan Tanzania Thailand Tunisia Turkey Uganda Ukraine Uruguay Uzbekistan
Venezuela Vietnam Yemen Zambia Zimbabwe USA US UK UAE""".split()} | {"united states", "united kingdom",
                                                                    "united states of america", "south africa",
                                                                    "new zealand", "saudi arabia", "costa rica"}
US_STATES = set("""AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC
ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC""".split())
NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

TYPES: list[tuple[str, callable]] = [
    ("boolean", lambda v: v.lower() in {"true", "false", "yes", "no", "y", "n", "t", "f"}),
    ("gender", lambda v: v.lower() in {"m", "f", "male", "female", "man", "woman", "w"}),
    ("email", lambda v: re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", v) is not None),
    ("url", lambda v: re.match(r"(?i)(https?://|www\.)\S+$", v) is not None),
    ("datetime", lambda v: re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?Z?", v) is not None),
    ("date", lambda v: re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|\d{8}", v) is not None
     and not v.isdigit() or re.fullmatch(r"(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", v) is not None),
    ("time", lambda v: re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?( ?[AaPp][Mm])?", v) is not None),
    ("year", lambda v: re.fullmatch(r"(18|19|20)\d{2}", v) is not None),
    ("percent", lambda v: re.fullmatch(NUM + r" ?%", v) is not None),
    ("currency", lambda v: re.fullmatch(r"[$€£¥] ?" + NUM + r"|" + NUM + r" ?(USD|EUR|GBP|[$€£])", v) is not None),
    ("integer", lambda v: re.fullmatch(r"[-+]?\d+", v) is not None),
    ("float", lambda v: re.fullmatch(NUM, v) is not None or re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", v) is not None),
    ("phone", lambda v: re.fullmatch(r"\+?[\d\s().-]{7,20}", v) is not None and 7 <= sum(c.isdigit() for c in v) <= 15
     and any(c in v for c in "+-(). ")),
    ("postal", lambda v: re.fullmatch(r"\d{5}-\d{4}|[A-Z]\d[A-Z] ?\d[A-Z]\d|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}", v) is not None),
    ("country", lambda v: v.lower() in COUNTRIES),
    ("us_state", lambda v: v in US_STATES),
    ("code", lambda v: re.fullmatch(r"[A-Z0-9_-]{1,8}", v) is not None),
    ("name_like", lambda v: re.fullmatch(r"[A-Z][a-z'’-]+( [A-Z][a-z'’.-]*){0,3}", v) is not None),
    ("text", lambda v: len(v.split()) >= 4),
]


def value_type(v: str) -> str:
    v = v.strip()
    for name, test in TYPES:
        try:
            if test(v):
                return name
        except re.error:
            pass
    return "other"


def column_types(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        counts = [Counter() for _ in header]
        for row in r:
            for i, v in enumerate(row[: len(header)]):
                if v.strip():
                    counts[i][value_type(v)] += 1
    out = []
    for c in counts:
        total = sum(c.values())
        top, n = c.most_common(1)[0] if c else ("", 0)
        out.append(f"{top}_" if total and n / total >= SHARE else "mixed_")
    return header, out


def main() -> int:
    diag, dist = [], Counter()
    for dataset, name, scen in cases("screen", "plain"):
        for side in ("source", "target"):
            header, types = column_types(scen / side / f"{side}.csv")
            dist.update(types)
            rng = random.Random(hashlib.sha256(f"{dataset}/{name}/{side}".encode()).hexdigest())
            shuffled = types[:]
            rng.shuffle(shuffled)
            for v, ts in (("semtype", types), ("shuffled", shuffled)):
                p = EDGES / v / dataset / name / f"{side}.tsv"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("".join(f"semtype\t{side}\t{h}\t@class:{t}\t1.0\n" for h, t in zip(header, ts)),
                             encoding="utf-8", newline="\n")
            diag.append({"dataset": dataset, "scenario": name, "side": side, "columns": len(header),
                         "types": len(set(types)), "mixed": types.count("mixed_")})
    with (HERE / "f13_plain_screen_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diag[0]))
        w.writeheader()
        w.writerows(diag)
    print(f"{len(diag)} tables; mean columns {mean(d['columns'] for d in diag):.1f}, mean types "
          f"{mean(d['types'] for d in diag):.1f}, mean mixed {mean(d['mixed'] for d in diag):.1f}")
    print("type distribution over all columns:", ", ".join(f"{k} {v}" for k, v in dist.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
