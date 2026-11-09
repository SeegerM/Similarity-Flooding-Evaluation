"""Refresh the plaintext / recoded comparison tables on the audited cohort.

The two tables in the paper draft were computed on the 80-scenario benchmark
shipped with PRISMA, and their recoded half used the **shipped**
`_md5_encrypted` variants. FINDINGS 33 showed that transform is not a bijective
recoding: it hashes nulls, so 52,008 empty cells in one table collapse into a
single shared digest, columns that are 40% null agree on 40% of their rows for
no reason, and maximum |delta MI| between the regimes is 0.626 instead of zero.
Every arm there that reads equality partitions -- SF, the IND/gcov roles, PDED,
PRISMA -- is confounded by manufactured agreement, so those numbers cannot be
updated in place; they have to be replaced by the audited cohort.

This regenerates both halves from the frozen audited rerun
(`reports/audited_md5_fused_20260904`), which scores every arm in both regimes
on identical scenarios, and adds the two label-based baselines the rerun does
not carry:

    Levenshtein   the production label matcher. In the plaintext regime the
                  cached production matrix is read; only the recoded regime,
                  which has no cached matrix, uses the Python port. The port is
                  verified against all 382 cached matrices to 2.9e-08, but that
                  is float32 resolution and is NOT sufficient for per-scenario
                  equality: a 1-ULP difference flips Hungarian tie-breaks on
                  33 of 382 scenarios (median |dF1| 0.061, max 0.100) while the
                  cohort mean moves by 0.001. See FINDINGS 70.
    seed          the frozen 65% Levenshtein / 35% PMF blend, built with the
                  production `scale_aligned_fuse`.

Scoring is `smx.core.evaluate` against the same ground truth the rerun uses, so
the added rows and the rerun's rows are commensurable by construction.

PDED and PRISMA were run on the audited corpus for this table, in both regimes:

    PDED      `PdedSchematchRunner` at the documented distribution weight 0.65,
              over `generated/benchmark-full` and `generated/benchmark-full-md5`.
    PRISMA    the raw soft matrix before PRISMA's own postprocessing, from its
              Flask service under JDK 17, config overlays
              `configs/audited-prisma-{plain,md5}`.

PRISMA's matrix is permuted into our column order **by bare label**, never by
position: FINDINGS 41b records a comparison that read 0.052 F1 instead of 0.626
because two matrices were aligned positionally. A scenario whose label sets do
not correspond is dropped rather than scored.

Still absent: vanilla SF and label-seeded SF+FD, which were not part of the
audited protocol and would need their own runs.

Usage:
    python regime_tables.py [--cohort old80|all382] [--tex out.tex]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from encrypted_feature_ablation import read_matrix as read_truth  # noqa: E402
from encrypted_feature_ablation import scale_aligned_fuse  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from smx.adapters import read_matrix as read_verbose  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

RUN = HERE / "reports" / "audited_md5_fused_20260904"
CACHED = HERE / "reports" / "valentine_full_382_20260903"
PDED = HERE / "pded_audited"
# The whole-corpus runs supersede the three-family ones; keep both so the table
# can still be built if only the smaller run is present.
PRISMA = {"plain": [HERE / "reports" / "prisma_audited_plain_full",
                    HERE / "reports" / "prisma_audited_plain"],
          "audited_md5": [HERE / "reports" / "prisma_audited_md5_full",
                          HERE / "reports" / "prisma_audited_md5"]}
# Same matcher, no gdep filtering. Reported beside the historical threshold
# rather than replacing it: at 0.3 every Magellan scenario has no surviving FD.
PRISMA_GDEP0 = {"plain": [HERE / "reports" / "prisma_audited_plain_gdep0"],
                "audited_md5": [HERE / "reports" / "prisma_audited_md5_gdep0"]}
# Similarity Flooding started from another system's matrix, via
# initialMatrix=FILE. Its QUICK encoding drops columns, so these matrices are
# often smaller than the ground truth and must be aligned by label.
SF_SEEDED = {
    ("kn", "plain"): HERE / "reports" / "sf_seeded_kn_plain",
    ("kn", "audited_md5"): HERE / "reports" / "sf_seeded_kn_md5",
    ("prisma", "plain"): HERE / "reports" / "sf_seeded_prisma_plain",
    ("prisma", "audited_md5"): HERE / "reports" / "sf_seeded_prisma_md5",
    # Provisional, with the classical block: SF flooded from COMA's matrix,
    # which is a much stronger plaintext seed (0.746 mutual) than the frozen
    # Levenshtein/PMF blend (0.607). Tests whether the standing result that
    # PRISMA is the only seed flooding never damages survives a better seed.
    ("coma", "plain"): HERE / "reports" / "sf_seeded_coma_plain",
}
SF_SEED_NAME = {"kn": "SF seeded by Kang-Naughton",
                "prisma": "SF seeded by PRISMA",
                "coma": "SF seeded by COMA"}
# The factorial's full set, one CSV per regime. The recoded half reads the
# `_fixed` run: `class_combinations_audited_md5.csv` joined plaintext seed
# labels against hashed profiles, so no dependency attached and every subset
# collapsed to the base (FINDINGS 113 addendum). The fixed run uses
# `--labels tables --strict-join`; every FD and DC joins.
COMBINATIONS = {"plain": HERE / "reports" / "class_combinations_audited_plain.csv",
                "audited_md5": HERE / "reports" / "class_combinations_audited_md5_fixed.csv"}
COMBO_ARM = "SF + FD + OD + DC"

# The external MusicBrainz/Lahman corpus, prepared independently of matcher
# outcomes and carrying its own recoding regimes.
EXTERNAL = HERE / "reports" / "controlled_matchers_v3_20260905" / "metrics.csv"
# Only constructions verified identical across the two runs. v3's SF arms are
# seeded by 65% Levenshtein + 35% PMF (its frozen protocol), not by PMF alone,
# and its PDED weight and PRISMA gdep threshold are unrecorded; mapping any of
# those onto this table's rows would put two different arms on one line.
EXTERNAL_ARM = {
    "levenshtein": "Levenshtein",
    "pmf": "PMF",
    "pmf_gcov_uncertainty": "PMF+gcov+uncertainty",
}
# Each external family has exactly one recoded regime, so both map to the
# table's recoded block without colliding.
# Standalone channels, both regimes, all 382 scenarios.
MAGNETO_PLAIN = HERE / "reports" / "magneto_corpus_plain.csv"
MAGNETO_MD5 = HERE / "reports" / "magneto_corpus_md5.csv"
# Only the ranked arms enter the table; "native" is a decision,
# not a score matrix, and is reported separately in FINDINGS 91.
MAGNETO_ARM = {"ranked_hungarian": "hungarian",
               "ranked_mutual_top1": "mutual_top1"}

# --- classical Valentine baselines -----------------------------------------
# COMA (Do & Rahm, VLDB'02) and Cupid (Madhavan et al., VLDB'01), through the
# Valentine 1.0.0 pure-Python reimplementations. Provisional: added so they can
# be compared against the arms above, and self-contained so they can be removed
# by deleting this block, CLASSICAL_RUN/CLASSICAL, and the BLOCKS entry.
#
# Read the mutual-top1 column, not Hungarian. Two cautions specific to these:
#
#  1. On flat relational input Valentine's COMA builds NAME_CM (+ InstancesCM),
#     not the paper's Path/Leaves/Parents matchers. "COMA (schema)" is a name
#     matcher and belongs beside Levenshtein, not beside a structural system.
#  2. Under the audited recoding the instance matcher scores exactly 0 on every
#     pair, and COMA's equal-weight mean halves the name score rather than
#     deferring to it. So the two COMA rows are identical there by construction
#     -- a uniform rescale is invariant under both extractors -- not by mistake.
#
# Full account, including the --verify contract and the metric cross-check
# against smx.core.evaluate, in VALENTINE_CLASSICAL_MATCHERS.md.
CLASSICAL_RUN = HERE / "reports" / "valentine_matchers_20260909"
CLASSICAL = {"coma": "COMA (schema)",
             "coma_instances": "COMA (schema+instances)",
             "cupid": "Cupid"}
# The same three matchers on the controlled MusicBrainz/Lahman corpus, whose
# regimes are per-case rather than parallel copies. `EXTERNAL_REGIME` maps its
# two recoding regimes onto the table's recoded column, exactly as the other
# external rows are mapped: an independent recode for MusicBrainz, and for
# Lahman a monotone numeric map, which preserves order while destroying value
# identity and so answers a different question in that column.
CLASSICAL_EXTERNAL = "_external.csv"
# Same rule as Magneto: only the ranked arms are matrices. Both matchers' native
# arms are their own published selections (COMA's DIR_BOTH at delta=0.15,
# Cupid's th_accept=0.7) and are decisions, not scores.
CLASSICAL_ARM = MAGNETO_ARM

# SemStruct (KDD'26 reconstruction). These two rows were in the committed
# `latex/table_regimes_all.tex` but had NO wiring here, so regenerating the
# table silently deleted them -- a complete, plausible-looking table missing an
# arm. Wired in from the frozen run and verified to reproduce the committed
# numbers exactly (ChEMBL plaintext 0.414/0.403/0.351/0.649 for the matcher and
# 0.411/0.404/0.353/0.650 for its control) before being trusted.
SEMSTRUCT = HERE / "reports" / "semstruct_20260908" / "report.csv"
SEMSTRUCT_ARM = {"semstruct": "SemStruct-reimpl",
                 "permuted": "SemStruct row-permuted control"}

# --- the follow-up experiments -------------------------------------------
# All provisional, all self-contained, all removable by deleting their constant
# and their BLOCKS entry. Full accounts in SF_CLASSICAL_FUSION.md,
# SEED_PRECONDITION.md and CONSTRAINT_GATE_ABLATION.md.

# COMA/Cupid fused onto the SF base as extra channels, at the frozen weight
# 0.35, built exactly as sf_channel_fusions.py builds its arms.
SF_FUSION_CLASSICAL = HERE / "reports" / "sf_classical_fusion.csv"
SF_FUSION_CLASSICAL_ARMS = ["SF + cupid", "SF + coma", "SF + char3 + cupid",
                            "SF + char3 + coma", "SF + char3 + cupid + coma"]

# The dependency-edge weighting ablation: four SF runs from the identical PMF
# seed differing only in how a dependency's quality score becomes an edge
# weight. Plaintext only, so the recoded half is blank rather than zero.
GATE_SWEEP = HERE / "reports" / "sf_gate_sweep.csv"
GATE_ARM = {"NONE": "SF+FD, weighting: none",
            "QUALITY": "SF+FD, weighting: quality",
            "GATED": "SF+FD, weighting: gated",
            "GATED-CALIB": "SF+FD, weighting: gated (calibrated)"}

# First-line seed combiners. `pmf` and the fixed blend are already table rows
# under their own names, so only the two new combiners are folded in.
SEED_AUDIT = HERE / "reports" / "seed_fusion_audit.csv"
SEED_AUDIT_ARM = {"adaptive-row": "seed, decisiveness-weighted",
                  "precondition": "seed, label precondition"}

# REMA-ARC, our reconstruction: there is no released implementation, so this is
# built from the four-page paper and REMA_REPRODUCTION.md records which lines
# are the paper's and which are our declared defaults. Complete-run files are
# preferred; the `_partial` fallback is a run that was stopped, and its reduced
# coverage is marked in the table rather than averaged away.
REMA_DIR = HERE / "reports" / "rema_20260908"


def _rema_metrics(regime: str) -> Path:
    complete = REMA_DIR / f"metrics_{regime}.csv"
    return complete if complete.is_file() else REMA_DIR / f"metrics_{regime}_partial.csv"


# Only the `rema` arm. `shuffled` was built as a null control and is not one:
# REMA-ARC is exactly invariant under per-side injective value recoding, so on
# the recoded regime the two arms are bit-identical (REMA_REPRODUCTION.md
# amendment 2). Only the two matrix extractors enter; REMA's native greedy is a
# selection policy, reported separately like Magneto's native output.
REMA_ARM = "rema"
REMA_EXTRACTORS = ("hungarian", "mutual_top1")

# Informationless references. Not matchers -- they are the level that column
# order alone reaches on this corpus, and after FINDINGS 90 no absolute F1 here
# is interpretable without them. Full 382 coverage: they need only the ground
# truth, not the tables.
CHANCE = REMA_DIR / "metrics_chance.csv"
CHANCE_ARM = {"constant": "Constant matrix (no information)",
              "random_matrix": "Random matrix (chance)"}
CHANNELS = HERE / "reports" / "regime_matrix_r44.csv"
CHANNEL_ARM = {
    "values exact": "Value overlap (exact)",
    "values char3": "Value overlap (char 3-gram)",
    "gcov role": "gcov role",
    "uncertainty role": "Uncertainty role",
    "order role": "Order role",
}
CHANNEL_REGIME = {"plain": "plain", "md5": "audited_md5"}

EXTERNAL_REGIME = {"plain": "plain",
                   "independent-recode": "audited_md5",
                   "monotone-numeric": "audited_md5"}
# The remaining arms, recomputed on v3 with the audited constructions by
# v3_audited_arms.py, so they belong on the same rows as the Valentine families.
# Its arm names are already the table's method names.
V3_AUDITED = HERE / "reports" / "v3_audited_arms.csv"
# The value/role channels, the frozen seed, and the SF channel fusions,
# computed on v3 by v3_channels_and_fusions.py. Same shape as V3_AUDITED.
V3_CHANNELS = HERE / "reports" / "v3_channels_and_fusions.csv"
# The factorial on the external corpus, at the frozen Valentine budgets.
V3_COMBINATIONS = HERE / "reports" / "class_combinations_v3.csv"
V3_SHIM = HERE / "reports" / "v3_oracle_table_shim" / "cases.csv"
# char3 and the uncertainty role fused onto the same SF base, per regime.
SF_FUSIONS = HERE / "reports" / "sf_channel_fusions.csv"
# The combinations from the declared search (FINDINGS 73), Valentine and
# external. Same six arms, same constructions, so the rows are comparable.
COMBO_ARMS = HERE / "reports" / "sf_combination_arms.csv"
V3_COMBO_ARMS = HERE / "reports" / "v3_combination_arms.csv"

# --- the grand-combination experiment (GRAND_COMBINATION.md, FINDINGS 92) ---
# The three separate leaders -- the classical fusion, the COMA seed, and the
# dependency triple -- combined, plus a rule that chooses base and channels
# from the declared label precondition. Same base, same `compose_gate.fuse`,
# same weight 0.35 as every other fused row here; its `SF + char3` control
# reproduces `sf_channel_fusions.csv` to max |dF1| = 0 over 1,528 cells.
#
# Only the *new* arms are folded in. The run also rebuilds `SF + char3`,
# `SF + char3 + cupid + coma`, `SF + char3 + FD + DC + OD` and the bare
# COMA-seeded SF, which are already rows here from their own sources; taking
# them from both would double-count the same matrices.
GRAND = HERE / "reports" / "sf_grand_combination_valentine.csv"
V3_GRAND = HERE / "reports" / "sf_grand_combination_external.csv"
GRAND_ARM = {
    "N1 SF + char3 + cupid + coma + FD + DC + OD":
        "SF + char3 + cupid + coma + FD + DC + OD",
    "N2 SF(COMA-seeded) + char3 + cupid + coma":
        "SF(COMA) + char3 + cupid + coma",
    "N3 SF(COMA-seeded) + char3 + FD + DC + OD":
        "SF(COMA) + char3 + FD + DC + OD",
    "N4 SF(COMA-seeded) + char3 + cupid + coma + FD + DC + OD":
        "SF(COMA) + char3 + cupid + coma + FD + DC + OD",
    "G5 gated, available channels (N4 | R3)": "gated, char3 branch",
    "G6 gated, available channels (N4 | roles)": "gated, roles branch",
}
# The external run additionally supplies the COMA-seeded SF base for
# MusicBrainz and Lahman over all three of their regimes -- a Schematch run
# made for this comparison. That row was blank for both families and for the
# whole recoded block, so it is filled here rather than left dashed.
GRAND_SEED_ARM = {"R4 SF(COMA-seeded)": "SF seeded by COMA"}
CORPUS = {"plain": "benchmark-full", "audited_md5": "benchmark-full-md5"}
REGIMES = ("plain", "audited_md5")
POLICIES = ("hungarian", "mutual_top1")
WEIGHT = 0.35

# Order and display names for the emitted table.
BLOCKS = [
    ("Label evidence", ["Levenshtein"]),
    ("Distribution evidence", ["PMF", "seed (0.65 Lev + 0.35 PMF)",
                               "seed, decisiveness-weighted",
                               "seed, label precondition"]),
    ("Value-overlap evidence", ["Value overlap (exact)",
                                "Value overlap (char 3-gram)"]),
    ("Equality-partition evidence", ["Kang-Naughton graph", "Kang-Naughton profile",
                                     "gcov role", "Uncertainty role", "Order role",
                                     "PMF+gcov+uncertainty"]),
    ("Similarity Flooding on invariant seeds", ["QUICK SF+FD(PMF)",
                                                "QUICK SF+FD+PMF+gcov",
                                                "QUICK SF+FD+PMF+Jaccard"]),
    ("Other structural architectures",
     ["PDED", "PRISMA (raw)", "PRISMA (raw, no gdep filter)"]),
    # A different family from the structural systems above: a small language
    # model retriever with a bipartite reranker, run zero-shot. Its own block
    # so it is not read as another structural architecture.
    ("Contemporary semantic matcher",
     ["Magneto (zero-shot)", "SemStruct-reimpl",
      "SemStruct row-permuted control"]),
    # Provisional block; see CLASSICAL above. The two original Valentine
    # baselines, older than everything else here. Kept in their own block
    # rather than merged into the evidence blocks, because "COMA (schema)" is
    # a label matcher and "COMA (schema+instances)" adds a value channel --
    # they straddle two of the categories above and belong to neither.
    ("Classical baselines", ["COMA (schema)", "COMA (schema+instances)",
                             "Cupid"]),
    # A third family again: random walks over an attribute-row-cell graph, with
    # skip-gram embeddings and cosine between mean cell-value vectors. It reads
    # no labels at all, so it is neither structural nor semantic in the senses
    # above; its only cross-table channel is value identity.
    ("Graph embeddings over instance values", ["REMA-ARC (reconstruction)"]),
    # Not matchers. FINDINGS 90: 6,422 of 6,432 true pairs lie on the diagonal,
    # and Hungarian resolves a constant matrix onto it, so these are the level
    # reachable with no information whatsoever. Every F1 above is read against
    # them or not read at all.
    ("Informationless references (FINDINGS 90)",
     ["Constant matrix (no information)", "Random matrix (chance)"]),
    ("Similarity Flooding started elsewhere",
     ["SF seeded by Kang-Naughton", "SF seeded by PRISMA",
      "SF seeded by COMA"]),
    # Provisional follow-ups; see SF_CLASSICAL_FUSION.md and
    # CONSTRAINT_GATE_ABLATION.md.
    ("Classical matchers fused into SF",
     ["SF + cupid", "SF + coma", "SF + char3 + cupid", "SF + char3 + coma",
      "SF + char3 + cupid + coma"]),
    ("Dependency-edge weighting (identical PMF seed)",
     ["SF+FD, weighting: none", "SF+FD, weighting: quality",
      "SF+FD, weighting: gated", "SF+FD, weighting: gated (calibrated)"]),
    ("Similarity Flooding with fused channels",
     ["SF + char3", "SF + uncertainty", "SF + gcov", "SF + KN",
      "SF + char3 + uncertainty", "SF + gcov + uncertainty", COMBO_ARM,
      "SF + char3 + FD + DC + OD",
      "SF + DC + OD + gcov + uncertainty + KN",
      "SF + all seven channels"]),
    # Provisional; see GRAND_COMBINATION.md. The two gated rows are one rule
    # each -- COMA-seeded SF with every channel where the labels are natural
    # identifiers, and a PMF-seeded SF without the label-reading channels where
    # they are not -- so they are the only rows here that are defined on every
    # scenario of both regimes.
    ("Combining the leaders, and gating on the label precondition",
     ["SF + char3 + cupid + coma + FD + DC + OD",
      "SF(COMA) + char3 + FD + DC + OD",
      "SF(COMA) + char3 + cupid + coma",
      "SF(COMA) + char3 + cupid + coma + FD + DC + OD",
      "gated, char3 branch", "gated, roles branch"]),
]
TEX_NAME = {
    "Levenshtein": "Levenshtein (labels only)",
    "PMF": "PMF (distribution)",
    "Value overlap (exact)": "Value overlap, exact",
    "Value overlap (char 3-gram)": "Value overlap, char 3-gram",
    "gcov role": "gcov role",
    "Uncertainty role": "Uncertainty role",
    "Order role": "Order role",
    "seed (0.65 Lev + 0.35 PMF)": "Seed ($0.65$~Lev $+$ $0.35$~PMF)",
    "Kang-Naughton graph": "Kang--Naughton graph",
    "Kang-Naughton profile": "Kang--Naughton profile",
    "PMF+gcov+uncertainty": "PMF $+$ gcov $+$ uncertainty",
    "QUICK SF+FD(PMF)": "SF+FD, PMF-seeded",
    "QUICK SF+FD+PMF+gcov": "SF+FD+PMF $+$ gcov",
    "QUICK SF+FD+PMF+Jaccard": "SF+FD+PMF $+$ Jaccard",
    "PDED": "PDED ($w=0.65$)",
    "Magneto (zero-shot)": "\\textsc{Magneto} (zero-shot)",
    # Provisional, with the classical block; see CLASSICAL.
    "COMA (schema)": "\\textsc{Coma} (names)",
    "COMA (schema+instances)": "\\textsc{Coma} (names $+$ instances)",
    "Cupid": "\\textsc{Cupid}",
    "REMA-ARC (reconstruction)": "\\textsc{Rema}-ARC (our reconstruction)",
    "Constant matrix (no information)": "Constant matrix (no information)",
    "Random matrix (chance)": "Random matrix (chance)",
    "PRISMA (raw)": "\\textsc{Prisma} (raw, gdep $\\geq 0.3$)",
    "PRISMA (raw, no gdep filter)": "\\textsc{Prisma} (raw, no gdep filter)",
    "SF seeded by Kang-Naughton": "SF seeded by Kang--Naughton",
    "SF seeded by PRISMA": "SF seeded by \\textsc{Prisma}",
    "SF seeded by COMA": "SF seeded by \\textsc{Coma}",
    "SemStruct-reimpl": "\\textsc{SemStruct}-reimpl",
    "SemStruct row-permuted control": "\\quad row-permuted control",
    "seed, decisiveness-weighted": "Seed, decisiveness-weighted",
    "seed, label precondition": "Seed, label precondition",
    "SF + cupid": "SF $+$ \\textsc{Cupid}",
    "SF + coma": "SF $+$ \\textsc{Coma}",
    "SF + char3 + cupid": "SF $+$ char 3-gram $+$ \\textsc{Cupid}",
    "SF + char3 + coma": "SF $+$ char 3-gram $+$ \\textsc{Coma}",
    "SF + char3 + cupid + coma": "SF $+$ char 3-gram $+$ \\textsc{Cupid} $+$ \\textsc{Coma}",
    "SF+FD, weighting: none": "SF+FD, dep. weighting: none",
    "SF+FD, weighting: quality": "SF+FD, dep. weighting: quality",
    "SF+FD, weighting: gated": "SF+FD, dep. weighting: gated",
    "SF+FD, weighting: gated (calibrated)": "SF+FD, dep. weighting: gated (calib.)",
    "SF + char3": "SF $+$ char 3-gram",
    "SF + uncertainty": "SF $+$ uncertainty role",
    "SF + char3 + uncertainty": "SF $+$ char 3-gram $+$ uncertainty",
    "SF + FD + OD + DC": "SF $+$ FD $+$ OD $+$ DC (GT-free)",
    "SF + gcov": "SF $+$ gcov role",
    "SF + KN": "SF $+$ Kang--Naughton",
    "SF + gcov + uncertainty": "SF $+$ gcov $+$ uncertainty",
    "SF + char3 + FD + DC + OD": "SF $+$ char3 $+$ FD $+$ DC $+$ OD",
    "SF + DC + OD + gcov + uncertainty + KN":
        "SF $+$ DC $+$ OD $+$ gcov $+$ unc. $+$ KN",
    "SF + all seven channels": "SF $+$ all seven channels",
    "SF + char3 + cupid + coma + FD + DC + OD":
        "SF $+$ char3 $+$ \\textsc{Cupid} $+$ \\textsc{Coma} $+$ FD $+$ DC $+$ OD",
    "SF(COMA) + char3 + FD + DC + OD":
        "SF(\\textsc{Coma}-seeded) $+$ char3 $+$ FD $+$ DC $+$ OD",
    "SF(COMA) + char3 + cupid + coma":
        "SF(\\textsc{Coma}-seeded) $+$ char3 $+$ \\textsc{Cupid} $+$ \\textsc{Coma}",
    "SF(COMA) + char3 + cupid + coma + FD + DC + OD":
        "SF(\\textsc{Coma}-seeded) $+$ char3 $+$ \\textsc{Cupid} $+$ "
        "\\textsc{Coma} $+$ FD $+$ DC $+$ OD",
    "gated, char3 branch":
        "Gated by label precondition, char3 fallback",
    "gated, roles branch":
        "Gated by label precondition, role fallback",
}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def label_matrix(source, target) -> np.ndarray:
    """Java `Levenshtein.compare`: raw labels, 1 - dist/max(len), float32."""
    out = np.zeros((len(source), len(target)), dtype=np.float32)
    for i, a in enumerate(source):
        for j, b in enumerate(target):
            denom = max(len(a), len(b))
            out[i, j] = (np.float32(1.0) - np.float32(levenshtein(a, b)) / np.float32(denom)
                         if denom else np.float32(1.0))
    return out


def headers(scenario: Path, side: str) -> list[str]:
    files = sorted((scenario / side).glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one CSV in {scenario / side}")
    with files[0].open(newline="", encoding="utf-8", errors="ignore") as fh:
        return [h.strip() for h in next(csv.reader(fh), [])]


def align(matrix, rows, cols, want_rows, want_cols):
    """Permute a foreign matrix into our column order by bare label.

    Returns None when the label sets do not correspond, rather than silently
    scoring a mismatched matrix (FINDINGS 41b).
    """
    if not rows or not cols:
        return None
    r_index = {r.split(".")[-1]: i for i, r in enumerate(rows)}
    c_index = {c.split(".")[-1]: j for j, c in enumerate(cols)}
    if any(w not in r_index for w in want_rows) or \
            any(w not in c_index for w in want_cols):
        return None
    return matrix[np.ix_([r_index[w] for w in want_rows],
                         [c_index[w] for w in want_cols])]


DEGENERATE = defaultdict(int)


def foreign_matrices(case: dict, regime: str, want_rows, want_cols) -> dict:
    """PDED and PRISMA matrices for one case and regime, aligned by label.

    An all-zero matrix is dropped rather than scored: it means the matcher
    produced no output (for PRISMA, an empty dependency graph after gdep
    filtering), and scoring it would report a confident 0.000 (FINDINGS 71).
    """
    out = {}
    pded = PDED / CORPUS[regime] / case["dataset"] / case["scenario"] / "pded.csv"
    if pded.is_file():
        m, rows, cols = read_matrix(pded)
        aligned = align(m, rows, cols, want_rows, want_cols)
        if aligned is not None:
            if np.any(aligned):
                out["PDED"] = aligned
            else:
                DEGENERATE[(regime, "PDED", case["family"])] += 1

    for label, roots in (("PRISMA (raw)", PRISMA[regime]),
                         ("PRISMA (raw, no gdep filter)", PRISMA_GDEP0[regime])):
        for root in roots:
            if not root.is_dir():
                continue
            found = False
            for run in sorted(root.iterdir()):
                d = (run / case["dataset"] / case["scenario"]
                     / "_outputs" / "MatchingStepLine1")
                if not d.is_dir():
                    continue
                files = [f for f in d.glob("*.csv") if "postprocessing=false" in f.name]
                if len(files) != 1:
                    continue
                m, rows, cols = read_verbose(files[0])
                aligned = align(m, rows, cols, want_rows, want_cols)
                if aligned is not None:
                    if np.any(aligned):
                        out[label] = aligned
                    else:
                        DEGENERATE[(regime, label, case["family"])] += 1
                found = True
                break
            if found:
                break

    for source, label in SF_SEED_NAME.items():
        root = SF_SEEDED.get((source, regime))
        if root is None or not root.is_dir():
            continue
        for run in sorted(root.iterdir()):
            d = (run / case["dataset"] / case["scenario"]
                 / "_outputs" / "MatchingStepLine2")
            if not d.is_dir():
                continue
            files = sorted(d.glob("*.csv"))
            if len(files) != 1:
                continue
            m, rows, cols = read_verbose(files[0])
            aligned = align(m, rows, cols, want_rows, want_cols)
            if aligned is not None:
                if np.any(aligned):
                    out[label] = aligned
                else:
                    DEGENERATE[(regime, label, case["family"])] += 1
            break
    return out


def added_rows(case: dict) -> list[dict]:
    """Levenshtein and the blended seed, in both regimes, for one case."""
    gt_files = sorted((Path(case["plain"]) / "ground_truth").glob("*.csv"))
    if not gt_files:
        return []
    gt = read_truth(gt_files[0])[0] > 0.5
    out = []
    for regime in REGIMES:
        scenario = Path(case[regime])
        pmf_path = RUN / "cases" / case["key"] / regime / "pmf.csv"
        if not scenario.is_dir() or not pmf_path.is_file():
            return []
        # Plaintext: read the production matrix rather than recompute it, so
        # this table cannot disagree with the historical one over tie-breaks
        # (FINDINGS 70). Recoded: no cached matrix exists, so the port is used.
        cached_lex = CACHED / "cases" / case["key"] / "levenshtein.csv"
        if regime == "plain" and cached_lex.is_file():
            lex = read_matrix(cached_lex)[0]
        else:
            lex = label_matrix(headers(scenario, "source"), headers(scenario, "target"))
        if lex.shape != gt.shape:
            return []
        pmf = read_matrix(pmf_path)[0]
        if pmf.shape != gt.shape:
            return []
        want_rows = [r.split(".")[-1] for r in headers(scenario, "source")]
        want_cols = [c.split(".")[-1] for c in headers(scenario, "target")]
        extra = foreign_matrices(case, regime, want_rows, want_cols)
        candidates = [("Levenshtein", lex),
                      ("seed (0.65 Lev + 0.35 PMF)", scale_aligned_fuse(lex, pmf, WEIGHT))]
        candidates += [(k, v) for k, v in extra.items() if v.shape == gt.shape]
        for name, matrix in candidates:
            for result in evaluate(ScoreMatrix(scores=matrix, ground_truth=gt),
                                   policies=list(POLICIES)):
                out.append({
                    "id": case["id"], "family": case["family"],
                    "scenario_type": case["scenario_type"],
                    "old80": str(case["old80"]), "regime": regime, "method": name,
                    "extractor": result["extractor"], "f1": result["f1"],
                    "recall_at_gt": result["recall_at_gt_pessimistic"],
                    "mrr": result["mrr_pessimistic"],
                })
    return out


def aggregate(rows, families):
    """(regime, method, family) -> (F1_H, F1_M, R@GT, MRR, n)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[(r["regime"], r["method"], r["family"])][r["extractor"]].append(
            (float(r["f1"]), float(r["recall_at_gt"]), float(r["mrr"])))
    table = {}
    for key, by_extractor in acc.items():
        h = by_extractor.get("hungarian", [])
        m = by_extractor.get("mutual_top1", [])
        if not h:
            continue
        table[key] = (mean(x[0] for x in h), mean(x[0] for x in m) if m else float("nan"),
                      mean(x[1] for x in h), mean(x[2] for x in h), len(h))
    return table


SHORT_COVERAGE = []


def emit_tex(table, families, counts, cohort_label,
             regime_counts=None) -> str:
    cols = "l" + "r" * (4 * len(families))
    # 28 numeric columns do not fit a rotated letter page at \small with the
    # default column glue; both are reduced rather than dropping a metric.
    # Twenty-eight numeric columns exceed a rotated letter page. Scaling to
    # \textheight -- the width available after rotation -- keeps every arm and
    # every family in one table instead of dropping a metric or splitting it.
    head = ["\\begin{table*}[t]", "\\centering",
            # Shrink to fit only when too wide; \resizebox{\textheight}{!}
            # alone would enlarge a narrower table until it overflowed.
            "\\rotatebox{90}{\\resizebox{\\ifdim\\width>\\textheight"
            "\\textheight\\else\\width\\fi}{!}{",
            "\\setlength{\\tabcolsep}{2.2pt}",
            f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
            "\\textbf{Approach} & "
            + " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{f}}} ({counts[f]})}}"
                         for f in families) + " \\\\"]
    head.append(" ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}"
                         for i in range(len(families))))
    head.append(" & " + " & ".join(
        "F1$_H$ & F1$_M$ & R@GT & MRR" for _ in families) + " \\\\")
    head.append("\\midrule")

    body = []
    for regime, regime_label in (("plain", "Plaintext"),
                                 ("audited_md5", "Audited per-side recoding")):
        body.append(f"\\multicolumn{{{1 + 4 * len(families)}}}{{l}}"
                    f"{{\\textbf{{{regime_label}}}}} \\\\")
        body.append("\\addlinespace[2pt]")
        for block, methods in BLOCKS:
            body.append(f"\\multicolumn{{{1 + 4 * len(families)}}}{{l}}"
                        f"{{\\emph{{{block}}}}} \\\\")
            for method in methods:
                cells, short = [], []
                for fam in families:
                    v = table.get((regime, method, fam))
                    if v is None:
                        cells += ["--"] * 4
                        continue
                    # A cell averaged over fewer scenarios than the family
                    # holds is not on the same cohort as the rows beside it.
                    # Mark it; do not let it pass as a full-cohort mean.
                    # Per *regime*, not per family: the external corpora use
                    # different scenario ids for plain and recoded, so a
                    # per-family count double-counts them and would mark every
                    # row in the table short.
                    total = (regime_counts or {}).get((regime, fam))
                    if total is not None and v[4] < total:
                        short.append((fam, v[4], total))
                    cells += [f"{v[i]:.3f}" for i in range(4)]
                if short:
                    SHORT_COVERAGE.append((regime, method, short))
                name = TEX_NAME[method] + ("$^{\\dagger}$" if short else "")
                body.append(f"{name} & " + " & ".join(cells) + " \\\\")
            body.append("\\addlinespace[3pt]")
        body.append("\\midrule")
    body = body[:-1]

    tail = ["\\bottomrule", "\\end{tabular}", "}}",
            "\\caption{" + cohort_label + " F1$_H$: full rectangular Hungarian assignment; "
            "F1$_M$: mutual-top1 with ties; R@GT: pessimistic global Recall@GT; "
            "MRR: pessimistic row mean reciprocal rank. Both regimes are scored on "
            "identical scenarios and identical ground truth. The recoding is the "
            "\\emph{audited} per-side salted transform, which reserves nulls; the "
            "shipped \\texttt{md5\\_encrypted} variants used previously hash nulls "
            "and are not information-preserving. "
            "$^{\\dagger}$~marks a row averaged over fewer scenarios than its "
            "column holds, so it is not on the same cohort as the rows beside "
            "it. The informationless references are not matchers: they are the "
            "level reached by column order alone, and after "
            "Finding~90---6{,}422 of 6{,}432 true pairs lie on the "
            "diagonal---no absolute F1$_H$ on this corpus is interpretable "
            "without them. The two gated rows are one rule each: on natural "
            "labels, the \\textsc{Coma}-seeded matrix with every channel; where "
            "the labels fail the precondition---which reads label shape only, "
            "no scores and no ground truth---a PMF-seeded matrix without the "
            "label-reading channels, falling back on char3 in one row and on "
            "the gcov/uncertainty roles in the other. It fires on no plaintext "
            "scenario, so the two rows coincide above and differ only below; "
            "they are the only combination rows defined on every scenario of "
            "both regimes.}",
            "\\label{tab:regime-" + ("old80" if "80" in cohort_label else "all") + "}",
            "\\end{table*}"]
    return "\n".join(head + body + tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="old80", choices=["old80", "all382"])
    ap.add_argument("--cache", default="reports/regime_tables_added.csv",
                    help="cache for the recomputed Levenshtein and seed rows")
    ap.add_argument("--tex", default=None)
    ap.add_argument("--dump-rows", default=None,
                    help="write the per-scenario rows behind the table to this "
                         "CSV. The table itself reports means, which cannot "
                         "support a paired comparison against a newly added "
                         "matcher; this exports the rows those means are "
                         "computed from. Read-only with respect to the table.")
    args = ap.parse_args()

    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cases = protocol["cases"]

    cache = Path(args.cache)
    if cache.is_file():
        added = list(csv.DictReader(cache.open(encoding="utf-8")))
        print(f"reusing {len(added)} cached baseline rows from {cache}")
    else:
        added = []
        for index, case in enumerate(cases, 1):
            added.extend(added_rows(case))
            if index % 50 == 0:
                print(f"  baselines {index}/{len(cases)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(added[0]))
            writer.writeheader()
            writer.writerows(added)
        print(f"wrote {len(added)} baseline rows to {cache}")
        if DEGENERATE:
            print("degenerate (all-zero) matrices dropped, not scored as 0.000:")
            for (regime, method, family), n in sorted(DEGENERATE.items()):
                print(f"    {regime:<12} {method:<14} {family:<10} {n}")

    scored = list(csv.DictReader((RUN / "scenarios.csv").open(encoding="utf-8")))

    # The factorial's full set, as one more arm. Its rows carry no family or
    # old80 marker, so they are joined to the protocol on the case id.
    meta = {c["id"]: c for c in protocol["cases"]}
    combo = []
    for regime, path in COMBINATIONS.items():
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r["variant"] != "anchor" or r["subset"] != "FD+OD+DC":
                continue
            case = meta.get(r["id"])
            if case is None:
                continue
            combo.append({"id": r["id"], "family": case["family"],
                          "scenario_type": case["scenario_type"],
                          "old80": str(case["old80"]), "regime": regime,
                          "method": COMBO_ARM, "extractor": r["extractor"],
                          "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                          "mrr": r["mrr"]})
    if combo:
        print(f"folded in {len(combo)} rows for {COMBO_ARM} "
              f"({len({r['id'] for r in combo})} scenarios)")

    external = []
    for source in (V3_AUDITED, V3_CHANNELS):
        if not source.is_file():
            continue
        for r in csv.DictReader(source.open(encoding="utf-8")):
            regime = EXTERNAL_REGIME.get(r["evidence_regime"])
            if regime is None:
                continue
            external.append({"id": r["id"], "family": r["dataset"],
                             "scenario_type": r["scenario_type"],
                             "old80": "False", "regime": regime,
                             "method": r["arm"], "extractor": r["extractor"],
                             "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                             "mrr": r["mrr"]})
    if EXTERNAL.is_file():
        for r in csv.DictReader(EXTERNAL.open(encoding="utf-8")):
            method = EXTERNAL_ARM.get(r["arm"])
            regime = EXTERNAL_REGIME.get(r["evidence_regime"])
            if method is None or regime is None:
                continue
            external.append({"id": r["id"], "family": r["dataset"],
                             "scenario_type": r["scenario_type"],
                             "old80": "False", "regime": regime,
                             "method": method, "extractor": r["extractor"],
                             "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                             "mrr": r["mrr"]})
        if external:
            fams = sorted({r["family"] for r in external})
            print(f"folded in {len(external)} external rows "
                  f"({len({r['id'] for r in external})} scenarios, {', '.join(fams)})")

    # Magneto (bdi-kit, zero-shot), the one contemporary matcher here. Its
    # ranked scores go through the same extractors as everything else; its
    # native discrete output is deliberately NOT folded in, because it is a
    # decision rather than a score matrix and the table's columns assume a
    # matrix (FINDINGS 91 reports both separately).
    magneto = []
    for path, regime in ((MAGNETO_PLAIN, "plain"),
                         (MAGNETO_MD5, "audited_md5")):
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            extractor = MAGNETO_ARM.get(r["arm"])
            if extractor is None:
                continue
            magneto.append({"id": r["id"], "family": r["family"],
                            "scenario_type": r["scenario_type"],
                            "old80": r["old80"], "regime": regime,
                            "method": "Magneto (zero-shot)",
                            "extractor": extractor, "f1": r["f1"],
                            "recall_at_gt": r["recall_at_gt"],
                            "mrr": r["mrr"]})
    if magneto:
        print(f"folded in {len(magneto)} Magneto rows "
              f"({len({r['id'] for r in magneto})} scenarios, "
              f"{len({r['regime'] for r in magneto})} regimes)")

    # COMA and Cupid (provisional; see CLASSICAL). Same contract as Magneto:
    # ranked arms only, and a missing file is an arm that was not run, which
    # the table marks as a blank cell rather than a zero.
    classical = []
    for method, label in CLASSICAL.items():
        for tag, regime in (("plain", "plain"), ("md5", "audited_md5")):
            path = CLASSICAL_RUN / f"{method}_corpus_{tag}.csv"
            if not path.is_file():
                continue
            for r in csv.DictReader(path.open(encoding="utf-8")):
                extractor = CLASSICAL_ARM.get(r["arm"])
                if extractor is None:
                    continue
                classical.append({"id": r["id"], "family": r["family"],
                                  "scenario_type": r["scenario_type"],
                                  "old80": r["old80"], "regime": regime,
                                  "method": label, "extractor": extractor,
                                  "f1": r["f1"],
                                  "recall_at_gt": r["recall_at_gt"],
                                  "mrr": r["mrr"]})
    # ... and the same three on MusicBrainz/Lahman, whose per-case regimes map
    # through EXTERNAL_REGIME rather than being named "plain"/"audited_md5".
    for method, label in CLASSICAL.items():
        path = CLASSICAL_RUN / f"{method}{CLASSICAL_EXTERNAL}"
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            extractor = CLASSICAL_ARM.get(r["arm"])
            regime = EXTERNAL_REGIME.get(r["regime"])
            if extractor is None or regime is None:
                continue
            classical.append({"id": r["id"], "family": r["family"],
                              "scenario_type": r["scenario_type"],
                              "old80": r["old80"], "regime": regime,
                              "method": label, "extractor": extractor,
                              "f1": r["f1"],
                              "recall_at_gt": r["recall_at_gt"],
                              "mrr": r["mrr"]})

    if classical:
        arms = sorted({r["method"] for r in classical})
        print(f"folded in {len(classical)} classical-baseline rows "
              f"({len({r['id'] for r in classical})} scenarios, "
              f"{len({r['regime'] for r in classical})} regimes): "
              f"{', '.join(arms)}")

    # SemStruct, keyed by case rather than by scenario id.
    semstruct = []
    if SEMSTRUCT.is_file():
        by_key = {c["key"]: c for c in protocol["cases"]}
        for r in csv.DictReader(SEMSTRUCT.open(encoding="utf-8")):
            label = SEMSTRUCT_ARM.get(r["arm"])
            case = by_key.get(r["key"])
            if label is None or case is None or r["policy"] not in POLICIES:
                continue
            semstruct.append({"id": case["id"], "family": case["family"],
                              "scenario_type": case["scenario_type"],
                              "old80": str(case["old80"]),
                              "regime": r["regime"], "method": label,
                              "extractor": r["policy"], "f1": r["f1"],
                              "recall_at_gt": r["recall_at_gt_pessimistic"],
                              "mrr": r["mrr_pessimistic"]})
        if semstruct:
            print(f"folded in {len(semstruct)} SemStruct rows "
                  f"({len({r['id'] for r in semstruct})} scenarios)")

    # The follow-up experiments. All three share the same tidy shape already:
    # id/family/scenario_type/regime/method/extractor/f1/recall_at_gt/mrr.
    followups = []
    for path, keep, rename in ((SF_FUSION_CLASSICAL,
                                set(SF_FUSION_CLASSICAL_ARMS), None),
                               (GATE_SWEEP, set(GATE_ARM), GATE_ARM),
                               (SEED_AUDIT, set(SEED_AUDIT_ARM), SEED_AUDIT_ARM)):
        if not path.is_file():
            continue
        n = 0
        by_id = {c["id"]: c for c in protocol["cases"]}
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r["method"] not in keep or r["extractor"] not in POLICIES:
                continue
            case = by_id.get(r["id"])
            if case is None:
                continue
            followups.append({
                "id": r["id"], "family": case["family"],
                "scenario_type": case["scenario_type"],
                "old80": str(case["old80"]),
                # the gate sweep is plaintext only and carries no regime column
                "regime": r.get("regime") or "plain",
                "method": (rename or {}).get(r["method"], r["method"]),
                "extractor": r["extractor"], "f1": r["f1"],
                "recall_at_gt": r["recall_at_gt"], "mrr": r["mrr"]})
            n += 1
        if n:
            print(f"folded in {n} rows from {path.name}")

    # REMA-ARC (our reconstruction) and the informationless references.
    rema = []
    for regime in REGIMES:
        path = _rema_metrics(regime)
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r["arm"] != REMA_ARM or r["extractor"] not in REMA_EXTRACTORS:
                continue
            case = meta.get(r["id"])
            if case is None:
                continue
            rema.append({"id": r["id"], "family": case["family"],
                         "scenario_type": case["scenario_type"],
                         "old80": str(case["old80"]), "regime": regime,
                         "method": "REMA-ARC (reconstruction)",
                         "extractor": r["extractor"], "f1": r["f1"],
                         "recall_at_gt": r["recall_at_gt"], "mrr": r["mrr"]})
    if rema:
        covered = len({r["id"] for r in rema})
        print(f"folded in {len(rema)} REMA-ARC rows ({covered} scenarios, "
              f"{len({r['regime'] for r in rema})} regimes)"
              + ("" if covered == len(meta)
                 else f"  -- PARTIAL: {covered} of {len(meta)}, marked in the table"))

    chance = []
    if CHANCE.is_file():
        for r in csv.DictReader(CHANCE.open(encoding="utf-8")):
            method = CHANCE_ARM.get(r["arm"])
            case = meta.get(r["id"])
            if method is None or case is None or r["extractor"] not in POLICIES:
                continue
            chance.append({"id": r["id"], "family": case["family"],
                           "scenario_type": case["scenario_type"],
                           "old80": str(case["old80"]), "regime": r["regime"],
                           "method": method, "extractor": r["extractor"],
                           "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                           "mrr": r["mrr"]})
    if chance:
        print(f"folded in {len(chance)} informationless-reference rows "
              f"({len({r['id'] for r in chance})} scenarios)")

    channels = []
    if CHANNELS.is_file():
        for r in csv.DictReader(CHANNELS.open(encoding="utf-8")):
            method = CHANNEL_ARM.get(r["channel"])
            regime = CHANNEL_REGIME.get(r["regime"])
            case = meta.get(r["id"])
            if method is None or regime is None or case is None:
                continue
            channels.append({"id": r["id"], "family": case["family"],
                             "scenario_type": case["scenario_type"],
                             "old80": str(case["old80"]), "regime": regime,
                             "method": method, "extractor": r["extractor"],
                             "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                             "mrr": r["mrr"]})
        if channels:
            print(f"folded in {len(channels)} channel rows "
                  f"({len({r['id'] for r in channels})} scenarios, "
                  f"{len(CHANNEL_ARM)} channels)")

    fusions = []
    if SF_FUSIONS.is_file():
        fusions = list(csv.DictReader(SF_FUSIONS.open(encoding="utf-8")))
        if fusions:
            print(f"folded in {len(fusions)} SF channel-fusion rows "
                  f"({len({r['id'] for r in fusions})} scenarios)")

    if V3_COMBINATIONS.is_file() and V3_SHIM.is_file():
        shim = {r["id"]: r for r in csv.DictReader(V3_SHIM.open(encoding="utf-8"))}
        added_combo = 0
        for r in csv.DictReader(V3_COMBINATIONS.open(encoding="utf-8")):
            if r["variant"] != "anchor" or r["subset"] != "FD+OD+DC":
                continue
            case = shim.get(r["id"])
            if case is None:
                continue
            regime = EXTERNAL_REGIME.get(case["evidence_regime"])
            if regime is None:
                continue
            external.append({"id": r["id"], "family": case["family"],
                             "scenario_type": case["scenario_type"],
                             "old80": "False", "regime": regime,
                             "method": COMBO_ARM, "extractor": r["extractor"],
                             "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                             "mrr": r["mrr"]})
            added_combo += 1
        if added_combo:
            print(f"folded in {added_combo} external rows for {COMBO_ARM}")

    if COMBO_ARMS.is_file():
        extra = list(csv.DictReader(COMBO_ARMS.open(encoding="utf-8")))
        fusions = fusions + extra
        if extra:
            print(f"folded in {len(extra)} combination-arm rows "
                  f"({len({r['method'] for r in extra})} arms)")

    if V3_COMBO_ARMS.is_file():
        n = 0
        for r in csv.DictReader(V3_COMBO_ARMS.open(encoding="utf-8")):
            regime = EXTERNAL_REGIME.get(r["evidence_regime"])
            if regime is None:
                continue
            external.append({"id": r["id"], "family": r["dataset"],
                             "scenario_type": r["scenario_type"],
                             "old80": "False", "regime": regime,
                             "method": r["arm"], "extractor": r["extractor"],
                             "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                             "mrr": r["mrr"]})
            n += 1
        if n:
            print(f"folded in {n} external combination-arm rows")

    # The grand-combination arms. Valentine rows join `fusions` (they are keyed
    # by the protocol's own ids); the external rows join `external` after the
    # usual evidence-regime mapping. Only the arms in GRAND_ARM and
    # GRAND_SEED_ARM are taken, so the arms this run rebuilds as controls do
    # not land on rows that already have their own source.
    if GRAND.is_file():
        n = 0
        for r in csv.DictReader(GRAND.open(encoding="utf-8")):
            method = GRAND_ARM.get(r["method"])
            case = meta.get(r["id"])
            if method is None or case is None or r["extractor"] not in POLICIES:
                continue
            fusions.append({"id": r["id"], "family": case["family"],
                            "scenario_type": case["scenario_type"],
                            "old80": str(case["old80"]), "regime": r["regime"],
                            "method": method, "extractor": r["extractor"],
                            "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                            "mrr": r["mrr"]})
            n += 1
        if n:
            print(f"folded in {n} grand-combination rows "
                  f"({len(GRAND_ARM)} arms)")

    if V3_GRAND.is_file():
        n = 0
        names = {**GRAND_ARM, **GRAND_SEED_ARM}
        for r in csv.DictReader(V3_GRAND.open(encoding="utf-8")):
            method = names.get(r["method"])
            regime = EXTERNAL_REGIME.get(r["regime"])
            if method is None or regime is None or r["extractor"] not in POLICIES:
                continue
            external.append({"id": r["id"], "family": r["group"],
                             "scenario_type": r["scenario_type"],
                             "old80": "False", "regime": regime,
                             "method": method, "extractor": r["extractor"],
                             "f1": r["f1"], "recall_at_gt": r["recall_at_gt"],
                             "mrr": r["mrr"]})
            n += 1
        if n:
            print(f"folded in {n} external grand-combination rows")

    rows = (scored + added + combo + external + channels + fusions
            + magneto + classical + semstruct + followups + rema + chance)
    if args.dump_rows:
        dump = Path(args.dump_rows)
        dump.parent.mkdir(parents=True, exist_ok=True)
        fields = ["id", "family", "scenario_type", "old80", "regime",
                  "method", "extractor", "f1", "recall_at_gt", "mrr"]
        with dump.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"dumped {len(rows)} per-scenario rows -> {dump}")
    if args.cohort == "old80":
        rows = [r for r in rows if r["old80"] == "True"]
        label = "The 80-scenario benchmark shipped with \\textsc{Prisma} (76 resolvable)."
    else:
        label = ("All 382 Valentine scenarios, plus the external "
                 "\\textsc{MusicBrainz} and \\textsc{Lahman} corpus. The recoded "
                 "block is the audited per-side transform for the Valentine "
                 "families and for \\textsc{MusicBrainz}, which is the same class "
                 "of transformation; for \\textsc{Lahman} it is a monotone "
                 "numeric map, which preserves order while destroying value "
                 "identity, so that column answers a different question. Blank "
                 "cells are arms not run on that corpus, not failures.")

    listed = [m for _, methods in BLOCKS for m in methods]
    duplicates = sorted({m for m in listed if listed.count(m) > 1})
    if duplicates:
        raise SystemExit(f"method listed in BLOCKS more than once, which "
                         f"renders it as two separate arms: {duplicates}")

    families = sorted({r["family"] for r in rows})
    counts = {f: len({r["id"] for r in rows if r["family"] == f}) for f in families}
    table = aggregate(rows, families)

    for regime in REGIMES:
        print(f"\n=== {regime} ({args.cohort}) ===")
        print(f"{'method':<30}" + "".join(f"{f:>28}" for f in families))
        for _, methods in BLOCKS:
            for method in methods:
                line = f"{method:<30}"
                for fam in families:
                    v = table.get((regime, method, fam))
                    line += ("{:>28}".format("--") if v is None
                             else f"{v[0]:>7.3f}{v[1]:>7.3f}{v[2]:>7.3f}{v[3]:>7.3f}")
                print(line)

    regime_counts = {}
    for r in rows:
        regime_counts.setdefault((r["regime"], r["family"]), set()).add(r["id"])
    regime_counts = {k: len(v) for k, v in regime_counts.items()}
    tex = emit_tex(table, families, counts, label, regime_counts)
    if SHORT_COVERAGE:
        print("\ncoverage below the column's scenario count (marked with a "
              "dagger in the table):")
        for regime, method, short in SHORT_COVERAGE:
            detail = ", ".join(f"{f} {n}/{total}" for f, n, total in short)
            print(f"  {regime:<12}{method:<34}{detail}")
    if args.tex:
        Path(args.tex).write_text(tex + "\n", encoding="utf-8")
        print(f"\nwrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
