"""Fix the lockbox / dev / screen splits and build their corpora (PREREGISTRATION.md, Splits).

Splits are stratified by (family, scenario type) and ordered by SHA-256 of the case id, so they
are a pure function of the protocol. Each split gets a corpus directory per regime whose scenario
folders are directory junctions to the real ones (no copying, no admin rights needed on Windows):

    tuning/corpus/<regime>/<split>/<dataset>/<scenario>  ->  harness/generated/benchmark-full[-md5]/...

Usage:
    python make_splits.py                  # writes splits.json and both regimes' corpora
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROTOCOL = ROOT / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"
CORPUS = {"plain": ROOT / "harness" / "generated" / "benchmark-full",
          "md5": ROOT / "harness" / "generated" / "benchmark-full-md5"}
LOCKBOX_SHARE = 0.25
SCREEN_SIZE = 60


def order(case: dict) -> str:
    return hashlib.sha256(case["id"].encode()).hexdigest()


def stratified(cases: list[dict], share: float) -> tuple[list[dict], list[dict]]:
    """Take round(share * n) of every stratum (at least 1 if the stratum has 2+ cases)."""
    strata = defaultdict(list)
    for c in cases:
        strata[(c["family"], c["scenario_type"])].append(c)
    picked, rest = [], []
    for key in sorted(strata):
        members = sorted(strata[key], key=order)
        k = round(share * len(members))
        if k == 0 and len(members) >= 2:
            k = 1
        picked += members[:k]
        rest += members[k:]
    return picked, rest


def junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True,
                       capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def main() -> int:
    cases = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    lockbox, dev = stratified(cases, LOCKBOX_SHARE)
    screen, _ = stratified(dev, SCREEN_SIZE / len(dev))
    splits = {
        "lockbox": sorted(c["key"] for c in lockbox),
        "dev": sorted(c["key"] for c in dev),
        "screen": sorted(c["key"] for c in screen),
        "rule": f"stratified by (family, scenario_type), order sha256(id); lockbox {LOCKBOX_SHARE}, "
                f"screen ~{SCREEN_SIZE} of dev",
    }
    (HERE / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    by_key = {c["key"]: c for c in cases}
    for regime, base in CORPUS.items():
        for split in ("lockbox", "dev", "screen"):
            for key in splits[split]:
                c = by_key[key]
                junction(HERE / "corpus" / regime / split / c["dataset"] / c["scenario"],
                         base / c["dataset"] / c["scenario"])
    for split in ("lockbox", "dev", "screen"):
        counts = defaultdict(int)
        for key in splits[split]:
            counts[by_key[key]["family"]] += 1
        print(f"{split:8s} {len(splits[split]):4d}  {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
