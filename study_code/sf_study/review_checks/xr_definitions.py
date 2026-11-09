"""Cache the term and graph definitions behind the external-review macros (xrDef*, xrSyn*) that live outside this
repository, so that analysis/make_xr_numbers.py runs offline:

  - Schematch's Java sources (a checkout of https://github.com/UMR-Big-Data-Analytics/schematch): the datatype
    matcher's similarities (DatatypeMatcher.java, Datatype.java), the FD and UCC size limits of the dependency filter
    (DependencyFilter.java), the key score's length (UniqueColumnCombination.java) and the TOPK_HYBRID weights
    (GraphOptions.java);
  - the converted Synthea->OMOP corpus (experiments/sf_study/data/convert_synthea.py's output, not redistributed): the
    number of table-pair scenarios and whole-schema scenarios.

Each value is found with the same pattern analysis/make_xr_numbers.py's counterpart in the paper's sources used; the
file records the pattern, the source file's SHA-256 and the Schematch commit. The Python sources and reports these
macros also read are part of this repository and are parsed by make_xr_numbers.py directly.

    python xr_definitions.py --schematch <checkout> --synthea <synthea_plain dir> --out xr_definitions.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

JAVA = Path("src") / "main" / "java" / "de" / "uni_marburg" / "schematch"
FACTS = [  # macro, file under JAVA, pattern (group 1 is the value)
    ("xrDefIntFloat", "matching/metadata/DatatypeMatcher.java", r"Datatype\.FLOAT\)\)\s*\{.*?return\s+([0-9.]+)f;"),
    ("xrDefStrText", "matching/metadata/DatatypeMatcher.java", r"Datatype\.TEXT\)\)\s*\{.*?return\s+([0-9.]+)f;"),
    ("xrDefTypeShare", "data/metadata/Datatype.java", r"percentage < ([0-9.]+)\) return STRING"),
    ("xrDefMaxLhs", "matching/similarityFlooding/DependencyFilter.java", r"getDeterminant\(\)\.size\(\) <= (\d+)"),
    ("xrDefMaxUcc", "matching/similarityFlooding/DependencyFilter.java", r"getColumnCombination\(\)\.size\(\) <= (\d+)"),
    ("xrDefKeyLen", "data/metadata/dependency/UniqueColumnCombination.java", r"int n = (\d+);"),
    ("xrDefTopkHigh", "matching/similarityFlooding/GraphOptions.java", r'"topkHigh", "smx\.sf\.topkHigh", "([0-9.]+)"'),
    ("xrDefTopkLow", "matching/similarityFlooding/GraphOptions.java", r'"topkLow", "smx\.sf\.topkLow", "([0-9.]+)"'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schematch", type=Path, required=True)
    ap.add_argument("--synthea", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(a.schematch), "rev-parse", "HEAD"], capture_output=True, text=True)
    out = {"schematch_commit": commit.stdout.strip() or None, "facts": {}}
    for macro, rel, pattern in FACTS:
        f = a.schematch / JAVA / rel
        text = f.read_text(encoding="utf-8")
        m = re.search(pattern, text, re.S)
        assert m, (rel, pattern)
        out["facts"][macro] = {"value": m.group(1), "file": (JAVA / rel).as_posix(), "pattern": pattern,
                               "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    pairs = {re.sub(r"^s\d+_", "", p.name) for p in (a.synthea / "SyntheaOMOP-Pair").iterdir() if p.is_dir()}
    full = [p for p in (a.synthea / "SyntheaOMOP-Full").iterdir() if p.is_dir()]
    out["synthea"] = {"pairs": len(pairs), "full_schemas": len(full),
                      "note": "distinct table pairs (seed prefix s<k>_ removed) and whole-schema scenarios"}
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {a.out}: {len(out['facts'])} facts, Schematch {out['schematch_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
