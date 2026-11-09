"""Correct the 48 OpenData View-Unionable scenarios that the importer filed as Unionable.

The release names the folder "View -Unionable" (with a space). smx.datasets only recognised "View-Unionable", fell back
to guessing from the scenario name and labelled them Unionable / dataset OpenData-Unionable. Tables, ground truth and
profiles are unaffected (keyed by content); only the folder, the manifest's scenario_type and the audit record change.

- official-release-audit-corrected.json: the audit with dataset/scenario_type corrected for these 48 (original kept)
- harness/generated/benchmark-official-20261030b and valentine-new168: folder OpenData-Unionable/<s> moved to
  OpenData-View-Unionable/<s>, manifest scenario_type set to View-Unionable

    python fix_view_unionable.py <workspace>
"""
import json
import sys
from pathlib import Path

ws = Path(sys.argv[1]).resolve()
work = ws / "output/cluster-start-2026-10-30"
audit = json.loads((work / "official-release-audit.json").read_text())
moved = []
for r in audit["scenarios"]:
    if r["family"] == "OpenData" and "/View -Unionable/" in r["id"]:
        assert r["dataset"] == "OpenData-Unionable" and r["scenario_type"] == "Unionable", r["id"]
        r["dataset"], r["scenario_type"] = "OpenData-View-Unionable", "View-Unionable"
        r["corrections"] = r.get("corrections", []) + [
            "scenario type from the release folder 'View -Unionable' (importer did not recognise the space)"]
        moved.append(r["scenario"])
assert len(moved) == 48, len(moved)
(work / "official-release-audit-corrected.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")

for tree in ("benchmark-official-20261030b", "valentine-new168"):
    root = ws / "harness/generated" / tree
    for s in moved:
        src, dst = root / "OpenData-Unionable" / s, root / "OpenData-View-Unionable" / s
        if dst.exists():
            assert not src.exists()
            continue
        dst.parent.mkdir(exist_ok=True)
        src.rename(dst)
        m = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
        assert "/View -Unionable/" in m["id"] and m["scenario_type"] == "Unionable"
        m["scenario_type"] = "View-Unionable"
        m["corrections"] = m.get("corrections", []) + ["scenario_type corrected from Unionable (release folder 'View -Unionable')"]
        (dst / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(tree, ": OpenData-Unionable", len(list((root / "OpenData-Unionable").iterdir())),
          "| OpenData-View-Unionable", len(list((root / "OpenData-View-Unionable").iterdir())))
