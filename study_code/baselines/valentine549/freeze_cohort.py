"""Freeze the 549-scenario Valentine cohort from the import in harness/generated/benchmark-official-20261030b.

The import (import_official_v2.py) expected 382 known + 167 new scenarios. The old 382-case protocol lists
TPC-DI prospect_horizontal_50_ec_av twice, under Joinable and under Unionable, with byte-identical tables and ground
truth; the official release has it only under Unionable. So the old cohort has 381 distinct scenarios, and the release
adds 168. This script records that mapping instead of asserting the old counts.

    python freeze_cohort.py <workspace>
"""
import hashlib
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
work = workspace / "output/cluster-start-2026-10-30"
sys.path.insert(0, str(workspace / "harness"))
from smx.datasets import digest, json_write  # noqa: E402

output = workspace / "harness/generated/benchmark-official-20261030b"
report = json.loads((work / "official-release-audit-corrected.json").read_text())
valid = [r for r in report["scenarios"] if r["status"] == "valid"]
invalid = [r for r in report["scenarios"] if r["status"] != "valid"]
assert len(valid) == 549 and len(invalid) == 2 and all(r["family"] == "Wikidata" for r in invalid)
old = json.loads((workspace / "harness/reports/audited_md5_fused_20260904/protocol.json").read_text())["cases"]
old_ids = {r["id"]: r for r in old}

DUP_OLD = "valentine/TPC-DI/Joinable/prospect_horizontal_50_ec_av/prospect_horizontal_50_ec_av"
DUP_NEW = "valentine/TPC-DI/Unionable/prospect_horizontal_50_ec_av/prospect_horizontal_50_ec_av"
existing, added, corrections = [], [], []
for record in valid:
    manifest_path = output / record["dataset"] / record["scenario"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for side in ("source", "target"):
        assert digest(manifest_path.parent / manifest["tables"][side]["path"]) == manifest["tables"][side]["sha256"]
    if manifest.get("repeated_header_removed"):
        corrections.append(record["id"])
    row = {"id": record["id"], "path": manifest_path.parent.relative_to(output).as_posix(),
           "manifest_sha256": digest(manifest_path),
           "tables": {side: manifest["tables"][side]["sha256"] for side in ("source", "target")}}
    match = old_ids.get(record["id"])
    if match is not None:
        row["prior_id"], row["prior_key"] = match["id"], match["key"]
        if record["id"] == DUP_NEW:
            row["prior_duplicate"] = {"id": DUP_OLD, "key": old_ids[DUP_OLD]["key"]}
        existing.append(row)
    else:
        row["key"] = hashlib.sha256(record["id"].encode()).hexdigest()[:16]
        added.append(row)

assert DUP_OLD in old_ids and DUP_OLD not in {r["id"] for r in valid}
unmatched = sorted(set(old_ids) - {r["id"] for r in valid})
assert unmatched == [DUP_OLD], unmatched
assert len(existing) == 381 and len(added) == 168, (len(existing), len(added))
assert len(corrections) == 7, corrections

cohort = {
    "input": "official Valentine v1, archive md5 98da6d6701806538e5c685f4426b95f6",
    "tree": "harness/generated/benchmark-official-20261030b",
    "total_release": 551, "valid": 549, "excluded": invalid,
    "prior_protocol_cases": 382, "prior_distinct_scenarios": 381,
    "prior_duplicate_note": "The old protocol lists TPC-DI prospect_horizontal_50_ec_av under Joinable and Unionable "
                            "with identical tables and ground truth; the release has it under Unionable only.",
    "new_cohort": len(added), "prior_cases": existing, "new_cases": added,
    "magellan_header_corrections": corrections,
    "allowed_compute_hosts": ["<cpu-host-00>", "<cpu-host-01>"],
}
json_write(work / "cohort-549.json", cohort)
print(json.dumps({"valid": 549, "prior": len(existing), "new": len(added),
                  "magellan_headers_removed": len(corrections),
                  "new_table_bytes": sum((output / r["path"] / s / (s + ".csv")).stat().st_size
                                         for r in added for s in ("source", "target"))}, indent=2))
