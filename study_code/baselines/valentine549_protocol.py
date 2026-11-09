"""The 549-scenario Valentine case list for the learned-system case studies (SEMSTRUCT_ABLATION.md Amendment 2b,
OMNIMATCH_ABLATION.md Amendment 549; RERUN_549_LOG.md).

549 = the 382-case protocol (reports/audited_md5_fused_20260904/protocol.json) without the Joinable copy of TPC-DI
prospect_horizontal_50_ec_av (the release files it as Unionable; identical tables and ground truth), plus the 168 new
scenarios (generated/valentine-new168, -md5; keys sha256(id)[:16] as the 382 protocol and semstruct_new168.py).

Cohorts: inspected44 stays inspected44 (excluded from every reading, as before); remaining338 (337 after the dedup) and
the 168 new form `remaining505`; `cohort382` keeps the old label (new168 for the new ones).

    harness/.venv-bdi/Scripts/python.exe -I harness/valentine549_protocol.py
        -> reports/valentine549_protocol/protocol.json (local paths; semstruct_run.py reads it via SEMSTRUCT_RUN)
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "reports" / "audited_md5_fused_20260904" / "protocol.json"
NEW = HERE / "generated" / "valentine-new168"
NEW_MD5 = HERE / "generated" / "valentine-new168-md5"
OUT = HERE / "reports" / "valentine549_protocol"
DROP_KEY = "8bd722b8b259659e"          # TPC-DI-Joinable/prospect_horizontal_50_ec_av (the duplicate's Joinable copy)
MAIN = "remaining505"


def build() -> list[dict]:
    old = json.loads(OLD.read_text(encoding="utf-8"))["cases"]
    assert len(old) == 382
    cases = []
    for c in old:
        if c["key"] == DROP_KEY:
            assert c["dataset"] == "TPC-DI-Joinable" and c["scenario"] == "prospect_horizontal_50_ec_av"
            continue
        keep = {k: c[k] for k in ("id", "key", "family", "scenario_type", "scenario", "dataset", "plain", "audited_md5")}
        keep["cohort382"] = c["cohort"]
        keep["cohort"] = "inspected44" if c["cohort"] == "inspected44" else MAIN
        cases.append(keep)
    for folder in sorted(p for p in NEW.glob("*/*") if (p / "manifest.json").is_file()):
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        cases.append({"id": man["id"], "key": hashlib.sha256(man["id"].encode()).hexdigest()[:16],
                      "family": man["family"], "scenario_type": man["scenario_type"], "scenario": folder.name,
                      "dataset": folder.parent.name, "plain": str(folder),
                      "audited_md5": str(NEW_MD5 / folder.parent.name / folder.name),
                      "cohort382": "new168", "cohort": MAIN})
    keys = [c["key"] for c in cases]
    rels = [f"{c['dataset']}/{c['scenario']}" for c in cases]
    assert len(set(keys)) == len(keys) == 549, len(keys)
    assert len(set(rels)) == 549, "a dataset/scenario folder name occurs twice"
    return cases


def main() -> int:
    cases = build()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps({"cases": cases, "note": __doc__}, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases -> {OUT / 'protocol.json'}")
    print(collections.Counter(c["cohort"] for c in cases))
    print(collections.Counter((c["cohort382"], c["family"]) for c in cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
