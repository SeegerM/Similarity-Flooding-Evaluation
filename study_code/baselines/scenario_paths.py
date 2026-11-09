"""Resolve a Valentine case id to its scenario directory.

Eleven scripts derived the directory from the case id with the same two lines:

    dataset = "GDC" if parts[1] == "GDC" else f"{parts[1]}-{parts[2]}"
    scenario = data / dataset / parts[3]

That is right for ChEMBL, OpenData and TPC-DI, whose ids carry
`family/scenario_type/scenario`. It is wrong for the other two families:

    valentine/Magellan/amazon_google_exp/amazon_google_exp
        -> Magellan-amazon_google_exp/amazon_google_exp   (does not exist)
        actually Magellan-Unionable/amazon_google_exp

    valentine/Wikidata/Musicians/Musicians_joinable/musicians_joinable
        -> Wikidata-Musicians/Musicians_joinable          (does not exist)
        actually Wikidata-Joinable/musicians_joinable

Magellan ids have no scenario-type segment, so `parts[2]` is the scenario name;
Wikidata ids have one segment too many and differ in case. Nine of 382 cases --
all seven Magellan and both Wikidata -- therefore fail to resolve, and every
script counts them as "missing artefacts" although their tables, profiles and
ground truth are present. FINDINGS 67.

The corrected rule tries the legacy path first and falls back to a lookup of the
id's **last** segment against an index of scenario directories. Legacy-first is
what makes the correction safe: every case that resolves today keeps exactly the
path it uses today, so the fix can only add cases, never move one. It also
settles the single ambiguous name -- `prospect_horizontal_50_ec_av` exists under
both TPC-DI-Joinable and TPC-DI-Unionable -- because those two ids carry a
scenario-type segment and so resolve on the legacy path.

Which rule is used is a **switch, defaulting to the old behaviour**, so no
existing artifact changes unless a run asks for the correction:

    SMX_RESOLVER=legacy   (default) the historical rule, 373 of 382
    SMX_RESOLVER=index    the corrected rule, 382 of 382

This mirrors how the corrected PMF distance was introduced in the Java
`DistributionMatcher`: new behaviour behind a flag, default path untouched and
still reproducing every cached artifact.
"""

from __future__ import annotations

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

RESOLVER = os.environ.get("SMX_RESOLVER", "legacy").strip().lower()


def legacy_path(data: Path, case_id: str) -> Path:
    """The historical rule, reproduced exactly."""
    parts = case_id.split("/")
    dataset = "GDC" if parts[1] == "GDC" else f"{parts[1]}-{parts[2]}"
    return Path(data) / dataset / parts[3]


@lru_cache(maxsize=8)
def _index(data: str) -> dict:
    """scenario directory name -> the dataset directories holding that name."""
    root = Path(data)
    found = defaultdict(list)
    if not root.is_dir():
        return {}
    for dataset in sorted(p for p in root.iterdir() if p.is_dir()):
        for scenario in sorted(p for p in dataset.iterdir() if p.is_dir()):
            found[scenario.name].append(scenario)
    return dict(found)


def resolve(data, case_id: str, resolver: str | None = None) -> Path:
    """Scenario directory for a case id, or the legacy path if none is found.

    Returns a Path either way; callers keep their own `is_dir()` check, so a
    genuinely absent scenario still reports as absent rather than raising.
    """
    legacy = legacy_path(data, case_id)
    if (resolver or RESOLVER) != "index":
        return legacy
    if legacy.is_dir():
        return legacy
    hits = _index(str(data)).get(case_id.split("/")[-1], [])
    return hits[0] if len(hits) == 1 else legacy
