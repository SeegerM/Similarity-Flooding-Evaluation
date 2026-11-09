"""Prepare fixed SF seeds/null masks and evaluate the representation ablation.

Run prepare once, the two frozen Java overlays, then evaluate with their result
directories. No ground truth is read when preparing seeds or null validity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from encrypted_feature_ablation import (
    NULLS, evaluate_matrix, non_null_plain_columns, read_table,
    non_null_cipher_columns, scale_aligned_fuse, vector_similarity,
    within_ind_evidence, within_ind_role_profile, wilcoxon, write_rows,
)
from smx.adapters import read_matrix
from smx.core import ScoreMatrix, scoring_diagnostics

DATASETS = {
    "TPC-DI-Unionable": "TPC-DI",
    "ChEMBL-Unionable": "ChEMBL",
    "Valentine-Wikidata-Unionable": "Wikidata",
    "Magellan-Unionable": "Magellan",
}
ENCODINGS = ("QUICK", "COLUMN_ONLY", "RELATIONAL", "RELATIONAL_IND")
SUFFIX = "_md5_encrypted"
EXPECTED_COUNTS = {"TPC-DI": 36, "ChEMBL": 36, "Wikidata": 1, "Magellan": 7}


def top1_target_concentration(scores: np.ndarray) -> float:
    """Largest target's share of row-top choices, splitting exact ties fairly."""
    if not scores.size:
        return 0.0
    best = np.abs(scores - scores.max(axis=1, keepdims=True)) <= 1e-9
    mass = best / best.sum(axis=1, keepdims=True)
    return float(mass.mean(axis=0).max())


def one_csv(directory: Path) -> Path:
    files = list(directory.glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one CSV under {directory}, got {len(files)}")
    return files[0]


def matrix_file(run: Path, dataset: str, scenario: str, line: int, token: str) -> Path:
    files = [p for p in (run / dataset / scenario / "_outputs" / f"MatchingStepLine{line}").glob("*.csv") if token in p.name]
    if len(files) != 1:
        raise ValueError(f"expected one {token} matrix for {dataset}/{scenario}, got {len(files)}")
    return files[0]


def write_matrix(path: Path, matrix: np.ndarray, rows: list[str], columns: list[str]) -> None:
    if matrix.shape != (len(rows), len(columns)) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid seed for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Source\\Target", *columns])
        for label, values in zip(rows, matrix):
            writer.writerow([label, *values])


def prepare(args: argparse.Namespace) -> None:
    data, output = Path(args.data), Path(args.inputs)
    manifest = []
    for dataset in DATASETS:
        for scenario in sorted((data / dataset).iterdir()):
            if not (scenario / "source").is_dir():
                continue
            plain_tables = {side: read_table(one_csv(scenario / side)) for side in ("source", "target")}
            for encrypted in (False, True):
                name = dataset + SUFFIX if encrypted else dataset
                instance = data / name / scenario.name
                labels = {}
                for side in ("source", "target"):
                    table_path = one_csv(instance / side)
                    header, values = read_table(table_path)
                    _, plain_values = plain_tables[side]
                    # Validates dimensions and row alignment length before any mask is emitted.
                    non_null_cipher_columns(values, plain_values)
                    labels[side] = [f"{table_path.stem}.{column}" for column in header]
                    mask = output / "m" / name / scenario.name / side / table_path.name
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    with mask.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(header)
                        for row in zip(*plain_values):
                            writer.writerow([int(value.strip().lower() not in NULLS) for value in row])
                run = Path(args.encrypted_distribution_run if encrypted else args.distribution_run)
                pmf, rows, cols = read_matrix(matrix_file(run, name, scenario.name, 1, "DistributionMatcher"))
                if rows != labels["source"] or cols != labels["target"]:
                    raise ValueError(f"seed label order differs: {name}/{scenario.name}")
                write_matrix(output / "p" / name / scenario.name / "seed.csv", pmf, rows, cols)
                manifest.append({"dataset": name, "scenario": scenario.name, "seed": "PMF", "rows": pmf.shape[0], "columns": pmf.shape[1]})
                if not encrypted:
                    lexical, lr, lc = read_matrix(matrix_file(Path(args.lexical_run), dataset, scenario.name, 1, "LevenshteinMatcher"))
                    if lr != rows or lc != cols:
                        raise ValueError("lexical and PMF seed orders differ")
                    combined = scale_aligned_fuse(lexical, pmf, 0.35)
                    write_matrix(output / "l" / name / scenario.name / "seed.csv", combined, rows, cols)
                    manifest.append({"dataset": name, "scenario": scenario.name, "seed": "LEX_PMF", "rows": pmf.shape[0], "columns": pmf.shape[1]})
    write_rows(output / "manifest.csv", manifest)
    print(f"Prepared {len(manifest)} fixed seeds and owner-supplied validity masks under {output}")


def ind_scores(data: Path, dataset: str, scenario: str) -> np.ndarray:
    profiles = []
    for side in ("source", "target"):
        _, values = read_table(one_csv(data / dataset / scenario / side))
        plain_name = dataset.removesuffix(SUFFIX)
        if dataset.endswith(SUFFIX):
            _, plain = read_table(one_csv(data / plain_name / scenario / side))
            cleaned = non_null_cipher_columns(values, plain)
        else:
            cleaned = non_null_plain_columns(values)
        gcov, _ = within_ind_evidence(cleaned)
        profiles.append(within_ind_role_profile(gcov))
    return vector_similarity(*profiles)


def evaluate(args: argparse.Namespace) -> None:
    data, inputs, reports = Path(args.data), Path(args.inputs), Path(args.reports)
    rows, invariance = [], []
    matrices_for_invariance = {}
    profile_cache = {}
    for seed_name, short, run in (("PMF", "p", Path(args.pmf_run)), ("LEX_PMF", "l", Path(args.lexpmf_run))):
        for dataset in DATASETS:
            for encrypted in ((False, True) if seed_name == "PMF" else (False,)):
                name = dataset + SUFFIX if encrypted else dataset
                regime = "encrypted" if encrypted else "plaintext"
                for scenario in sorted((data / name).iterdir()):
                    if not (scenario / "ground_truth").is_dir():
                        continue
                    gt = read_matrix(one_csv(scenario / "ground_truth"))[0] > 0.5
                    seed = read_matrix(inputs / short / name / scenario.name / "seed.csv")[0]
                    profile_key = (name, scenario.name)
                    if profile_key not in profile_cache:
                        profile_cache[profile_key] = ind_scores(data, name, scenario.name)
                    gcov = profile_cache[profile_key]
                    matrices = {"Seed only": seed,
                                "Seed + gcov profile": scale_aligned_fuse(seed, gcov, 0.35)}
                    for encoding in ENCODINGS:
                        path = matrix_file(run, name, scenario.name, 2, f"graphEncoding={encoding} &")
                        matrices[encoding] = read_matrix(path)[0]
                    matrices["QUICK + gcov profile"] = scale_aligned_fuse(matrices["QUICK"], gcov, 0.35)
                    # Secondary, explicitly separate from the representation-only
                    # test: can its output help as auxiliary evidence? Reuse the
                    # pre-existing 35% fusion weight, without a sweep or selection.
                    for encoding in ENCODINGS[1:]:
                        matrices[f"Seed + {encoding} (35%)"] = scale_aligned_fuse(seed, matrices[encoding], 0.35)
                    for method, scores in matrices.items():
                        if scores.shape != gt.shape or not np.isfinite(scores).all():
                            raise ValueError(f"invalid score matrix: {name}/{scenario.name}/{method}")
                        diagnostics = scoring_diagnostics(ScoreMatrix(scores, gt))
                        rows.append({"regime": regime, "seed": seed_name, "dataset": DATASETS[dataset],
                                     "scenario": scenario.name, "method": method,
                                     "f1": evaluate_matrix(scores, gt),
                                     "recall_at_gt_pessimistic": diagnostics["recall_at_gt_pessimistic"],
                                     "recall_at_gt_optimistic": diagnostics["recall_at_gt_optimistic"],
                                     "mean_row_entropy": diagnostics["mean_row_entropy"],
                                     "mean_margin": diagnostics["mean_margin"],
                                     "top1_target_concentration": top1_target_concentration(scores)})
                        if seed_name == "PMF":
                            matrices_for_invariance[(dataset, scenario.name, regime, method)] = scores
    for dataset, scenario, regime, method in matrices_for_invariance:
        if regime != "plaintext":
            continue
        plain = matrices_for_invariance[(dataset, scenario, regime, method)]
        encrypted = matrices_for_invariance[(dataset, scenario, "encrypted", method)]
        invariance.append({"dataset": DATASETS[dataset], "scenario": scenario, "method": method,
                           "max_abs_difference": float(np.max(np.abs(plain - encrypted)))})
    write_rows(reports / "sf_representation_scenarios.csv", rows)
    write_rows(reports / "sf_representation_invariance.csv", invariance)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["seed"], row["dataset"], row["method"])].append(row)
    summary = []
    for (regime, seed, dataset, method), values in sorted(groups.items()):
        if len(values) != EXPECTED_COUNTS[dataset]:
            raise ValueError(f"incomplete benchmark group: {regime}/{seed}/{dataset}/{method}: {len(values)}")
        summary.append({"regime": regime, "seed": seed, "dataset": dataset, "method": method,
                        "scenarios": len(values), **{key: float(np.mean([r[key] for r in values]))
                        for key in ("f1", "recall_at_gt_pessimistic", "recall_at_gt_optimistic", "mean_row_entropy", "mean_margin", "top1_target_concentration")}})
    write_rows(reports / "sf_representation_summary.csv", summary)
    family_groups = defaultdict(list)
    for row in summary:
        family_groups[(row["regime"], row["seed"], row["method"])].append(row)
    macro = []
    for (regime, seed, method), families in sorted(family_groups.items()):
        if len(families) != 4:
            raise ValueError(f"incomplete family coverage: {regime}/{seed}/{method}")
        count = sum(row["scenarios"] for row in families)
        macro.append({"regime": regime, "seed": seed, "method": method, "scenarios": count,
                      "family_macro_f1": float(np.mean([row["f1"] for row in families])),
                      "scenario_weighted_f1": sum(row["f1"] * row["scenarios"] for row in families) / count})
    write_rows(reports / "sf_representation_macro.csv", macro)
    comparisons = []
    comparisons_to_run = [(method, baseline) for method in ENCODINGS for baseline in ("Seed only", "QUICK + gcov profile")]
    comparisons_to_run += [("RELATIONAL_IND", "RELATIONAL"), ("RELATIONAL", "COLUMN_ONLY"), ("COLUMN_ONLY", "QUICK")]
    comparisons_to_run += [("QUICK + gcov profile", "Seed + gcov profile"),
                           ("QUICK + gcov profile", "QUICK"),
                           ("Seed + gcov profile", "Seed only")]
    comparisons_to_run += [(f"Seed + {method} (35%)", baseline)
                           for method in ENCODINGS[1:]
                           for baseline in ("Seed only", "Seed + gcov profile")]
    for regime, seed in (("plaintext", "PMF"), ("encrypted", "PMF"), ("plaintext", "LEX_PMF")):
        lookup = {(r["dataset"], r["scenario"], r["method"]): r["f1"] for r in rows if r["regime"] == regime and r["seed"] == seed}
        keys = sorted({(dataset, scenario) for dataset, scenario, _ in lookup})
        for method, baseline in comparisons_to_run:
            differences = np.asarray([lookup[(d, s, method)] - lookup[(d, s, baseline)] for d, s in keys])
            nonzero = differences[np.abs(differences) > 1e-12]
            family_deltas = [np.mean([differences[i] for i, (d, _) in enumerate(keys) if d == family]) for family in DATASETS.values()]
            comparisons.append({"regime": regime, "seed": seed, "method": method, "baseline": baseline,
                                "scenario_weighted_delta": float(differences.mean()),
                                "family_macro_delta": float(np.mean(family_deltas)),
                                "wins": int((differences > 1e-12).sum()), "losses": int((differences < -1e-12).sum()),
                                "ties": int((np.abs(differences) <= 1e-12).sum()),
                                "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if nonzero.size and wilcoxon else 1.0})
    write_rows(reports / "sf_representation_comparisons.csv", comparisons)
    harness = Path(__file__).resolve().parent
    source_root = harness.parent / "code" / "schematch" / "src" / "main" / "java" / "de" / "uni_marburg" / "schematch"
    sources = [source_root / "matching" / "similarityFlooding" / filename for filename in
               ("ColumnRelationFlooding.java", "SimilarityFlooding.java", "SchemaGraphBuilder.java", "FixpointFormula.java")]
    sources.append(source_root / "data" / "metadata" / "dependency" / "FunctionalDependency.java")
    for config in ("sf-relational-pmf", "sf-relational-lexpmf", "paper-benchmark-sf"):
        sources.extend(sorted((harness / "configs" / config).glob("*.yaml")))
    provenance = {"pmf_run": str(Path(args.pmf_run).resolve()), "lexpmf_run": str(Path(args.lexpmf_run).resolve()),
                  "inputs": str(inputs.resolve()), "data": str(data.resolve()), "method_scenario_rows": len(rows),
                  "config": {"formula": "B", "coefficient_policy": "INV_PROD", "fusion_weight": 0.35,
                             "seed_policy": "FILE", "extractor": "Hungarian"},
                  "sha256": {str(path.relative_to(harness.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (reports / "sf_representation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for regime, seed in (("plaintext", "PMF"), ("encrypted", "PMF"), ("plaintext", "LEX_PMF")):
        print(f"\n{regime} / {seed}")
        for method in ("Seed only", "Seed + gcov profile", *ENCODINGS, "QUICK + gcov profile",
                       *(f"Seed + {encoding} (35%)" for encoding in ENCODINGS[1:])):
            numbers = [groups[(regime, seed, family, method)] for family in DATASETS.values()]
            print(f"{method:26} " + " ".join(f"{np.mean([r['f1'] for r in g]):.4f}" for g in numbers))
    print(f"\nEvaluated {len(rows)} method/scenario results. Reports: {reports}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--data", default="../code/schematch/data")
    parser.add_argument("--inputs", default="../code/schematch/rel_exp")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--distribution-run", default="../code/schematch/results/09-02-2026_17-51-42")
    parser.add_argument("--encrypted-distribution-run", default="../code/schematch/results/09-02-2026_10-22-31")
    parser.add_argument("--lexical-run", default="../code/schematch/results/09-02-2026_01-16-15")
    parser.add_argument("--pmf-run")
    parser.add_argument("--lexpmf-run")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        if not args.pmf_run or not args.lexpmf_run:
            parser.error("evaluate needs --pmf-run and --lexpmf-run")
        evaluate(args)


if __name__ == "__main__":
    main()
