"""Matched PMF/gcov/uncertainty/char3 comparison on the paper datasets.

This closes a small factorial left open by the encrypted-feature experiments:
the uncertainty role was tested with PMF on ChEMBL, while within-table IND
``gcov`` was tested with SF+FD+PMF and PRISMA on all four paper families.  Here
the two role channels are compared and combined on exactly the same 80
scenarios, in plaintext and in the shipped MD5 variants. The plaintext arm
also includes the frozen character-trigram channel and fusion order from
``best_combination.py``.

There is no weight search.  Added evidence always receives 35%.  When gcov and
uncertainty are both present they are independently min-max normalised and
share that evidence budget equally.  Thus the combined arm does not win merely
by receiving twice the role weight.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / ".deps"))
import numpy as np
from scipy.stats import wilcoxon

from encrypted_feature_ablation import (
    find_ground_truth,
    fuse,
    non_null_cipher_columns,
    non_null_plain_columns,
    read_matrix,
    read_table,
    scale_aligned_fuse,
    vector_similarity,
    within_ind_evidence,
    within_ind_role_profile,
)
from best_combination import char3_channel
from od_role import cosine_similarity
from order_role import role_profile
from plaintext_encrypted_feature_transfer import matcher_matrix, only_csv
from smx.adapters import read_matrix as read_verbose_matrix
from smx.core import ScoreMatrix, evaluate
from smx.relational_roles import relation_matrices


PLAIN_DATASETS = (
    "TPC-DI-Unionable",
    "ChEMBL-Unionable",
    "Valentine-Wikidata-Unionable",
    "Magellan-Unionable",
)
SHORT_NAMES = {
    "TPC-DI-Unionable": "TPC-DI",
    "ChEMBL-Unionable": "ChEMBL",
    "Valentine-Wikidata-Unionable": "Wikidata",
    "Magellan-Unionable": "Magellan",
}
ROLE_WEIGHT = 0.35


def normalised(matrix: np.ndarray) -> np.ndarray | None:
    span = float(np.ptp(matrix))
    if span <= 1e-12:
        return None
    return (matrix - float(matrix.min())) / span


def equal_role_block(*matrices: np.ndarray) -> np.ndarray:
    """Average informative role matrices after independent normalisation."""
    parts = [part for matrix in matrices if (part := normalised(matrix)) is not None]
    if not parts:
        return np.zeros_like(matrices[0])
    return np.mean(parts, axis=0)


def prisma_matrix(run: Path, dataset: str, scenario: str) -> np.ndarray:
    directory = run / dataset / scenario / "_outputs" / "MatchingStepLine1"
    files = [path for path in directory.glob("*.csv") if "postprocessing=false" in path.name]
    if len(files) != 1:
        raise ValueError(f"expected one raw PRISMA matrix in {directory}, found {len(files)}")
    return read_verbose_matrix(files[0])[0]


def role_matrices(
    columns_s: list[list[str]],
    columns_t: list[list[str]],
    gcov_s: list[list[str]],
    gcov_t: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_gcov, _ = within_ind_evidence(gcov_s)
    t_gcov, _ = within_ind_evidence(gcov_t)
    gcov = vector_similarity(
        within_ind_role_profile(s_gcov), within_ind_role_profile(t_gcov)
    )

    # Match the uncertainty-role experiments exactly: their common column
    # reader profiles the first 5,000 rows.  gcov intentionally remains the
    # prior full-column calculation above.
    uncertainty_s = [column[:5000] for column in columns_s]
    uncertainty_t = [column[:5000] for column in columns_t]
    s_rel, _, _ = relation_matrices(uncertainty_s)
    t_rel, _, _ = relation_matrices(uncertainty_t)
    uncertainty = cosine_similarity(
        role_profile(s_rel["uncertainty"], len(columns_s)),
        role_profile(t_rel["uncertainty"], len(columns_t)),
    )
    return gcov, uncertainty, equal_role_block(gcov, uncertainty)


def add_metrics(
    rows: list[dict[str, object]],
    cohort: str,
    dataset: str,
    scenario: str,
    method: str,
    matrix: np.ndarray,
    truth: np.ndarray,
) -> None:
    for result in evaluate(
        ScoreMatrix(scores=matrix, ground_truth=truth),
        policies=("hungarian", "mutual_top1"),
    ):
        rows.append(
            {
                "cohort": cohort,
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "extractor": result["extractor"],
                "f1": result["f1"],
                "recall_at_gt": result.get("recall_at_gt_pessimistic", float("nan")),
                "mrr": result.get("mrr_pessimistic", float("nan")),
            }
        )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../code/schematch/data")
    parser.add_argument(
        "--plain-distribution-run",
        default="../code/schematch/results/09-02-2026_17-51-42",
    )
    parser.add_argument(
        "--md5-distribution-run",
        default="../code/schematch/results/09-02-2026_10-22-31",
    )
    parser.add_argument(
        "--lexical-run", default="../code/schematch/results/09-02-2026_01-16-15"
    )
    parser.add_argument(
        "--md5-lexical-run",
        default="../code/schematch/results/09-02-2026_09-40-55",
    )
    parser.add_argument(
        "--plain-prisma-run",
        default="../code/prisma/results/09-02-2026_01-22-25 paper-benchmark",
    )
    parser.add_argument(
        "--md5-prisma-run",
        default="../code/prisma/results/09-02-2026_09-47-53 encrypted-benchmark",
    )
    parser.add_argument("--out", default="reports/pmf_uncertainty_combination_scenarios.csv")
    parser.add_argument("--summary", default="reports/pmf_uncertainty_combination_summary.csv")
    parser.add_argument(
        "--comparisons", default="reports/pmf_uncertainty_combination_comparisons.csv"
    )
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    plain_distribution_run = Path(args.plain_distribution_run)
    md5_distribution_run = Path(args.md5_distribution_run)
    lexical_run = Path(args.lexical_run)
    md5_lexical_run = Path(args.md5_lexical_run)
    plain_prisma_run = Path(args.plain_prisma_run)
    md5_prisma_run = Path(args.md5_prisma_run)
    rows: list[dict[str, object]] = []

    for plain_dataset in PLAIN_DATASETS:
        plain_root = data_dir / plain_dataset
        for plain_scenario in sorted(path for path in plain_root.iterdir() if path.is_dir()):
            if find_ground_truth(plain_scenario) is None:
                continue
            scenario_name = plain_scenario.name
            plain_s_headers, plain_s_columns = read_table(only_csv(plain_scenario / "source"))
            plain_t_headers, plain_t_columns = read_table(only_csv(plain_scenario / "target"))
            del plain_s_headers, plain_t_headers

            for cohort in ("plain", "md5"):
                dataset = plain_dataset if cohort == "plain" else f"{plain_dataset}_md5_encrypted"
                scenario = data_dir / dataset / scenario_name
                if cohort == "plain":
                    source_columns, target_columns = plain_s_columns, plain_t_columns
                    gcov_source = non_null_plain_columns(source_columns)
                    gcov_target = non_null_plain_columns(target_columns)
                    distribution_run = plain_distribution_run
                    prisma_run = plain_prisma_run
                else:
                    _, source_columns = read_table(only_csv(scenario / "source"))
                    _, target_columns = read_table(only_csv(scenario / "target"))
                    # The shipped transform hashes NULL as an ordinary value.
                    # Use the aligned owner mask, matching the prior gcov test.
                    gcov_source = non_null_cipher_columns(source_columns, plain_s_columns)
                    gcov_target = non_null_cipher_columns(target_columns, plain_t_columns)
                    distribution_run = md5_distribution_run
                    prisma_run = md5_prisma_run

                gcov, uncertainty, both_roles = role_matrices(
                    source_columns, target_columns, gcov_source, gcov_target
                )
                pmf = matcher_matrix(distribution_run, dataset, scenario_name, 1, "DistributionMatcher")
                sf = matcher_matrix(distribution_run, dataset, scenario_name, 2, "SimilarityFlooding")
                prisma = prisma_matrix(prisma_run, dataset, scenario_name)
                gt = read_matrix(find_ground_truth(scenario))[0]
                truth = gt > 0.5

                pmf_gcov = fuse(pmf, gcov, ROLE_WEIGHT)
                pmf_uncertainty = fuse(pmf, uncertainty, ROLE_WEIGHT)
                pmf_both = fuse(pmf, both_roles, ROLE_WEIGHT)
                methods = {
                    "PMF": pmf,
                    "PMF + within-IND gcov": pmf_gcov,
                    "PMF + uncertainty": pmf_uncertainty,
                    "PMF + gcov + uncertainty": pmf_both,
                    "SF+FD+PMF": sf,
                    "SF+FD+PMF + within-IND gcov": scale_aligned_fuse(sf, gcov, ROLE_WEIGHT),
                    "SF+FD+PMF + uncertainty": scale_aligned_fuse(sf, uncertainty, ROLE_WEIGHT),
                    "SF+FD+PMF + gcov + uncertainty": scale_aligned_fuse(
                        sf, both_roles, ROLE_WEIGHT
                    ),
                    "PRISMA": prisma,
                    "PRISMA + within-IND gcov": scale_aligned_fuse(prisma, gcov, ROLE_WEIGHT),
                    "PRISMA + uncertainty": scale_aligned_fuse(prisma, uncertainty, ROLE_WEIGHT),
                    "PRISMA + gcov + uncertainty": scale_aligned_fuse(
                        prisma, both_roles, ROLE_WEIGHT
                    ),
                }

                active_lexical_run = lexical_run if cohort == "plain" else md5_lexical_run
                levenshtein = matcher_matrix(
                    active_lexical_run, dataset, scenario_name, 1, "LevenshteinMatcher"
                )
                sampled_source = [column[:5000] for column in source_columns]
                sampled_target = [column[:5000] for column in target_columns]
                char3 = char3_channel(sampled_source, sampled_target, args.seed)
                levenshtein_char3 = scale_aligned_fuse(
                    levenshtein, char3, ROLE_WEIGHT
                )
                methods.update(
                    {
                        "Levenshtein": levenshtein,
                        "Levenshtein + uncertainty": scale_aligned_fuse(
                            levenshtein, uncertainty, ROLE_WEIGHT
                        ),
                        "Levenshtein + char3": levenshtein_char3,
                        "Levenshtein + char3 + uncertainty": scale_aligned_fuse(
                            levenshtein_char3, uncertainty, ROLE_WEIGHT
                        ),
                    }
                )
                if cohort == "plain":
                    lexical_sf = matcher_matrix(
                        lexical_run, dataset, scenario_name, 2, "fdFilterThreshold=0.0"
                    )
                    methods.update(
                        {
                            "Label-aware SF+FD + PMF + within-IND gcov": scale_aligned_fuse(
                                lexical_sf, pmf_gcov, ROLE_WEIGHT
                            ),
                            "Label-aware SF+FD + PMF + uncertainty": scale_aligned_fuse(
                                lexical_sf, pmf_uncertainty, ROLE_WEIGHT
                            ),
                            "Label-aware SF+FD + PMF + gcov + uncertainty": scale_aligned_fuse(
                                lexical_sf, pmf_both, ROLE_WEIGHT
                            ),
                        }
                    )

                for method, matrix in methods.items():
                    if matrix.shape != truth.shape:
                        raise ValueError(
                            f"shape mismatch for {cohort}/{dataset}/{scenario_name}/{method}: "
                            f"{matrix.shape} != {truth.shape}"
                        )
                    add_metrics(
                        rows,
                        cohort,
                        SHORT_NAMES[plain_dataset],
                        scenario_name,
                        method,
                        matrix,
                        truth,
                    )

    write_rows(Path(args.out), rows)

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cohort"]), str(row["dataset"]), str(row["method"]), str(row["extractor"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    for key, selected in grouped.items():
        cohort, dataset, method, extractor = key
        summary_rows.append(
            {
                "cohort": cohort,
                "dataset": dataset,
                "method": method,
                "extractor": extractor,
                "scenarios": len(selected),
                "mean_f1": float(np.mean([float(row["f1"]) for row in selected])),
                "mean_recall_at_gt": float(
                    np.mean([float(row["recall_at_gt"]) for row in selected])
                ),
                "mean_mrr": float(np.mean([float(row["mrr"]) for row in selected])),
            }
        )
    summary_rows.sort(key=lambda row: (str(row["cohort"]), str(row["dataset"]), str(row["extractor"]), str(row["method"])))
    write_rows(Path(args.summary), summary_rows)

    indexed: dict[tuple[str, str], dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)
    for row in rows:
        arm = (str(row["cohort"]), str(row["method"]))
        key = (str(row["dataset"]), str(row["scenario"]))
        indexed[arm].setdefault(key, {})[str(row["extractor"])] = float(row["f1"])

    pairs = []
    for base in ("PMF", "SF+FD+PMF", "PRISMA"):
        pairs.extend(
            (
                (f"{base} + uncertainty", base),
                (f"{base} + gcov + uncertainty", f"{base} + within-IND gcov"),
                (f"{base} + gcov + uncertainty", f"{base} + uncertainty"),
            )
        )
    pairs.extend(
        (
            ("Levenshtein + uncertainty", "Levenshtein"),
            ("Levenshtein + char3", "Levenshtein"),
            ("Levenshtein + char3 + uncertainty", "Levenshtein + char3"),
            (
                "Levenshtein + char3 + uncertainty",
                "Label-aware SF+FD + PMF + within-IND gcov",
            ),
            (
                "Label-aware SF+FD + PMF + gcov + uncertainty",
                "Label-aware SF+FD + PMF + within-IND gcov",
            ),
            (
                "Label-aware SF+FD + PMF + gcov + uncertainty",
                "Label-aware SF+FD + PMF + uncertainty",
            ),
        )
    )
    comparison_rows: list[dict[str, object]] = []
    for cohort in ("plain", "md5"):
        for treatment, baseline in pairs:
            treatment_values = indexed.get((cohort, treatment))
            baseline_values = indexed.get((cohort, baseline))
            if not treatment_values or not baseline_values:
                continue
            keys = sorted(set(treatment_values) & set(baseline_values))
            for extractor in ("hungarian", "mutual_top1"):
                differences = np.asarray(
                    [
                        treatment_values[key][extractor] - baseline_values[key][extractor]
                        for key in keys
                    ]
                )
                nonzero = differences[np.abs(differences) > 1e-12]
                p_value = (
                    float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                    if nonzero.size
                    else float("nan")
                )
                comparison_rows.append(
                    {
                        "cohort": cohort,
                        "treatment": treatment,
                        "baseline": baseline,
                        "extractor": extractor,
                        "scenarios": len(keys),
                        "mean_f1_delta": float(differences.mean()),
                        "wins": int((differences > 1e-12).sum()),
                        "losses": int((differences < -1e-12).sum()),
                        "ties": int((np.abs(differences) <= 1e-12).sum()),
                        "wilcoxon_p_nonzero": p_value,
                    }
                )
    write_rows(Path(args.comparisons), comparison_rows)

    for cohort in ("plain", "md5"):
        selected = [row for row in summary_rows if row["cohort"] == cohort and row["extractor"] == "hungarian"]
        print(f"\n{cohort.upper()} -- Hungarian macro F1 by dataset")
        for dataset in SHORT_NAMES.values():
            ranked = sorted(
                (row for row in selected if row["dataset"] == dataset),
                key=lambda row: float(row["mean_f1"]),
                reverse=True,
            )
            best = ranked[0]
            print(f"{dataset:<9} {best['mean_f1']:.3f}  {best['method']}")
    print(f"\nwrote {args.out}, {args.summary}, and {args.comparisons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
