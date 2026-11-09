"""Test whether hash-invariant encrypted features transfer to plaintext data.

This is deliberately a fixed-weight transfer test, not a tuning run.  It
compares distribution-seeded SF+FD+PMF with the ordinary label-aware SF+FD
baseline and adds the same within-table IND gcov role profile used in the
encrypted ablation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from encrypted_feature_ablation import (
    evaluate_matrix,
    find_ground_truth,
    fuse,
    non_null_plain_columns,
    read_matrix,
    read_table,
    scale_aligned_fuse,
    vector_similarity,
    within_ind_evidence,
    within_ind_role_profile,
    wilcoxon,
)
from smx.adapters import read_matrix as read_verbose_matrix


DATASETS = (
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


def only_csv(path: Path) -> Path:
    files = sorted(path.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one CSV in {path}, found {len(files)}")
    return files[0]


def matcher_matrix(run: Path, dataset: str, scenario: str, line: int, token: str) -> np.ndarray:
    directory = run / dataset / scenario / "_outputs" / f"MatchingStepLine{line}"
    files = [path for path in directory.glob("*.csv") if token in path.name]
    if len(files) != 1:
        raise ValueError(f"expected one {token!r} matrix in {directory}, found {len(files)}")
    return read_matrix(files[0])[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../code/schematch/data")
    parser.add_argument("--distribution-run", required=True)
    parser.add_argument("--lexical-run", default="../code/schematch/results/09-02-2026_01-16-15")
    parser.add_argument("--encrypted-run", default="../code/schematch/results/09-02-2026_10-22-31")
    parser.add_argument("--pded-run", default="pded_distribution_plain/w065")
    parser.add_argument("--encrypted-pded-run", default="pded_distribution_out/w065")
    parser.add_argument(
        "--prisma-run", default="../code/prisma/results/09-02-2026_01-22-25 paper-benchmark"
    )
    parser.add_argument(
        "--encrypted-prisma-run",
        default="../code/prisma/results/09-02-2026_09-47-53 encrypted-benchmark",
    )
    parser.add_argument("--out", default="reports/plaintext_encrypted_feature_transfer_scenarios.csv")
    parser.add_argument("--summary", default="reports/plaintext_encrypted_feature_transfer_summary.csv")
    parser.add_argument("--invariance", default="reports/plaintext_encrypted_feature_transfer_invariance.csv")
    parser.add_argument("--comparisons", default="reports/plaintext_encrypted_feature_transfer_comparisons.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    distribution_run = Path(args.distribution_run)
    lexical_run = Path(args.lexical_run)
    encrypted_run = Path(args.encrypted_run)
    pded_run = Path(args.pded_run)
    encrypted_pded_run = Path(args.encrypted_pded_run)
    prisma_run = Path(args.prisma_run)
    encrypted_prisma_run = Path(args.encrypted_prisma_run)
    rows: list[dict[str, object]] = []
    invariant_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        for scenario in sorted((data_dir / dataset).iterdir()):
            if not scenario.is_dir() or find_ground_truth(scenario) is None:
                continue
            source_headers, source_columns = read_table(only_csv(scenario / "source"))
            target_headers, target_columns = read_table(only_csv(scenario / "target"))
            source_columns = non_null_plain_columns(source_columns)
            target_columns = non_null_plain_columns(target_columns)

            source_gcov, _ = within_ind_evidence(source_columns)
            target_gcov, _ = within_ind_evidence(target_columns)
            ind_scores = vector_similarity(
                within_ind_role_profile(source_gcov),
                within_ind_role_profile(target_gcov),
            )

            pmf = matcher_matrix(distribution_run, dataset, scenario.name, 1, "DistributionMatcher")
            distribution_sf = matcher_matrix(distribution_run, dataset, scenario.name, 2, "SimilarityFlooding")
            lexical_sf = matcher_matrix(
                lexical_run, dataset, scenario.name, 2, "fdFilterThreshold=0.0"
            )
            pded = read_matrix(pded_run / dataset / scenario.name / "pded.csv")[0]
            prisma_path = next(
                path
                for path in (
                    prisma_run / dataset / scenario.name / "_outputs" / "MatchingStepLine1"
                ).glob("*.csv")
                if "postprocessing=false" in path.name
            )
            prisma = read_verbose_matrix(prisma_path)[0]
            gt = read_matrix(find_ground_truth(scenario))[0]
            if any(
                matrix.shape != gt.shape
                for matrix in (pmf, distribution_sf, lexical_sf, pded, prisma, ind_scores)
            ):
                raise ValueError(
                    f"shape mismatch in {dataset}/{scenario.name}: gt={gt.shape}, "
                    f"pmf={pmf.shape}, dist_sf={distribution_sf.shape}, "
                    f"lex_sf={lexical_sf.shape}, pded={pded.shape}, "
                    f"prisma={prisma.shape}, ind={ind_scores.shape}"
                )

            pmf_ind = fuse(pmf, ind_scores, 0.35)
            distribution_sf_ind = scale_aligned_fuse(distribution_sf, ind_scores, 0.35)
            lexical_sf_pmf = scale_aligned_fuse(lexical_sf, pmf, 0.35)
            lexical_sf_ind = scale_aligned_fuse(lexical_sf, ind_scores, 0.35)
            # 65% label-aware SF+FD plus a fixed evidence block containing
            # 65% PMF and 35% within-IND gcov (22.75%/12.25% overall).
            lexical_sf_pmf_ind = scale_aligned_fuse(lexical_sf, pmf_ind, 0.35)
            pded_ind = scale_aligned_fuse(pded, ind_scores, 0.35)
            prisma_ind = scale_aligned_fuse(prisma, ind_scores, 0.35)

            methods = {
                "PMF": pmf,
                "PMF + within-IND gcov": pmf_ind,
                "Distribution-seeded SF+FD+PMF": distribution_sf,
                "Distribution-seeded SF+FD+PMF + within-IND gcov": distribution_sf_ind,
                "Label-aware SF+FD": lexical_sf,
                "Label-aware SF+FD + PMF": lexical_sf_pmf,
                "Label-aware SF+FD + within-IND gcov": lexical_sf_ind,
                "Label-aware SF+FD + PMF + within-IND gcov": lexical_sf_pmf_ind,
                "PDED+PMF": pded,
                "PDED+PMF + within-IND gcov": pded_ind,
                "PRISMA": prisma,
                "PRISMA + within-IND gcov": prisma_ind,
            }
            for method, matrix in methods.items():
                rows.append(
                    {
                        "dataset": SHORT_NAMES[dataset],
                        "scenario": scenario.name,
                        "method": method,
                        "f1": evaluate_matrix(matrix, gt),
                    }
                )

            encrypted_dataset = f"{dataset}_md5_encrypted"
            encrypted_pmf = matcher_matrix(
                encrypted_run, encrypted_dataset, scenario.name, 1, "DistributionMatcher"
            )
            encrypted_sf = matcher_matrix(
                encrypted_run, encrypted_dataset, scenario.name, 2, "SimilarityFlooding"
            )
            encrypted_pded = read_matrix(
                encrypted_pded_run / encrypted_dataset / scenario.name / "pded.csv"
            )[0]
            encrypted_prisma_path = next(
                path
                for path in (
                    encrypted_prisma_run
                    / encrypted_dataset
                    / scenario.name
                    / "_outputs"
                    / "MatchingStepLine1"
                ).glob("*.csv")
                if "postprocessing=false" in path.name
            )
            encrypted_prisma = read_verbose_matrix(encrypted_prisma_path)[0]
            invariant_rows.extend(
                [
                    {
                        "dataset": SHORT_NAMES[dataset],
                        "scenario": scenario.name,
                        "matrix": "PMF",
                        "max_abs_plain_encrypted_difference": float(np.max(np.abs(pmf - encrypted_pmf))),
                    },
                    {
                        "dataset": SHORT_NAMES[dataset],
                        "scenario": scenario.name,
                        "matrix": "Distribution-seeded SF+FD+PMF",
                        "max_abs_plain_encrypted_difference": float(
                            np.max(np.abs(distribution_sf - encrypted_sf))
                        ),
                    },
                    {
                        "dataset": SHORT_NAMES[dataset],
                        "scenario": scenario.name,
                        "matrix": "PRISMA",
                        "max_abs_plain_encrypted_difference": float(
                            np.max(np.abs(prisma - encrypted_prisma))
                        ),
                    },
                    {
                        "dataset": SHORT_NAMES[dataset],
                        "scenario": scenario.name,
                        "matrix": "PDED+PMF",
                        "max_abs_plain_encrypted_difference": float(
                            np.max(np.abs(pded - encrypted_pded))
                        ),
                    },
                ]
            )

    write_rows(Path(args.out), rows)
    write_rows(Path(args.invariance), invariant_rows)

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["dataset"]))].append(float(row["f1"]))
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    summary_rows = []
    for method in methods:
        for dataset in (SHORT_NAMES[name] for name in DATASETS):
            values = grouped[(method, dataset)]
            summary_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "scenarios": len(values),
                    "mean_f1": float(np.mean(values)),
                }
            )
    write_rows(Path(args.summary), summary_rows)

    by_method: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        by_method[str(row["method"])][(str(row["dataset"]), str(row["scenario"]))] = float(row["f1"])
    comparisons = (
        ("PMF + within-IND gcov", "PMF"),
        ("Distribution-seeded SF+FD+PMF + within-IND gcov", "Distribution-seeded SF+FD+PMF"),
        ("Label-aware SF+FD + PMF", "Label-aware SF+FD"),
        ("Label-aware SF+FD + within-IND gcov", "Label-aware SF+FD"),
        ("Label-aware SF+FD + PMF + within-IND gcov", "Label-aware SF+FD"),
        ("Label-aware SF+FD + PMF + within-IND gcov", "Label-aware SF+FD + PMF"),
        ("PDED+PMF + within-IND gcov", "PDED+PMF"),
        ("PRISMA + within-IND gcov", "PRISMA"),
    )
    comparison_rows: list[dict[str, object]] = []
    for treatment, baseline in comparisons:
        keys = sorted(by_method[treatment])
        differences = np.asarray(
            [by_method[treatment][key] - by_method[baseline][key] for key in keys]
        )
        nonzero = differences[np.abs(differences) > 1e-12]
        p_value = (
            float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            if wilcoxon is not None and nonzero.size
            else float("nan")
        )
        comparison_rows.append(
            {
                "treatment": treatment,
                "baseline": baseline,
                "scenarios": len(keys),
                "mean_f1_delta": float(differences.mean()),
                "wins": int((differences > 1e-12).sum()),
                "losses": int((differences < -1e-12).sum()),
                "ties": int((np.abs(differences) <= 1e-12).sum()),
                "wilcoxon_p_nonzero": p_value,
            }
        )
    write_rows(Path(args.comparisons), comparison_rows)

    max_pmf = max(
        float(row["max_abs_plain_encrypted_difference"])
        for row in invariant_rows
        if row["matrix"] == "PMF"
    )
    max_sf = max(
        float(row["max_abs_plain_encrypted_difference"])
        for row in invariant_rows
        if row["matrix"] == "Distribution-seeded SF+FD+PMF"
    )
    max_pded = max(
        float(row["max_abs_plain_encrypted_difference"])
        for row in invariant_rows
        if row["matrix"] == "PDED+PMF"
    )
    max_prisma = max(
        float(row["max_abs_plain_encrypted_difference"])
        for row in invariant_rows
        if row["matrix"] == "PRISMA"
    )
    print(f"evaluated {len(rows) // len(methods)} scenarios")
    print(f"max plaintext/encrypted PMF matrix difference: {max_pmf:.3g}")
    print(f"max plaintext/encrypted distribution-SF matrix difference: {max_sf:.3g}")
    print(f"max plaintext/encrypted PDED+PMF matrix difference: {max_pded:.3g}")
    print(f"max plaintext/encrypted PRISMA matrix difference: {max_prisma:.3g}")
    for row in summary_rows:
        print(f"{row['method']:<58} {row['dataset']:<9} {row['mean_f1']:.6f}")
    for row in comparison_rows:
        print(
            f"delta {row['treatment']} vs {row['baseline']}: "
            f"{row['mean_f1_delta']:+.6f}, "
            f"W/L/T={row['wins']}/{row['losses']}/{row['ties']}, "
            f"p={row['wilcoxon_p_nonzero']:.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
