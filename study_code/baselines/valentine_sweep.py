"""Frozen stratified plaintext sweep over already imported/profiled Valentine.

No discovery, row sampling, parameter selection, or GT-informed seed generation.
Each Java phase has an independent timeout; failures remain in the denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from encrypted_feature_ablation import (
    non_null_plain_columns, scale_aligned_fuse, vector_similarity,
    within_ind_evidence, within_ind_role_profile,
)
from sf_representation_ablation import write_matrix
from smx.adapters import read_matrix
from smx.core import ScoreMatrix, hungarian, match_metrics, mutual_top1, scoring_diagnostics

HERE = Path(__file__).resolve().parent
SF = HERE.parent / "code" / "schematch"
CURRENT = "SF+FD+PMF+gcov"
METHODS = ["Levenshtein", "PMF", "Value Jaccard", "LEX+PMF seed",
           "Seed+gcov", "SF (LEX+PMF)", "SF+FD (LEX+PMF)", CURRENT]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_sample(candidates, per_stratum, seed):
    """Order-independent deterministic selection using IDs only, not outcomes."""
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["family"], row["scenario_type"])].append(row)
    selected = []
    for key, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda row: (
            hashlib.sha256(f'{seed}:{row["id"]}'.encode()).hexdigest(), row["id"]))
        selected.extend(ranked[:per_stratum])
    return selected


def source_fingerprint():
    files = [Path(__file__), HERE / "java" / "ValentineSweep.java",
             HERE / "encrypted_feature_ablation.py", HERE / "sf_representation_ablation.py",
             HERE / "smx" / "core.py", HERE / "smx" / "adapters.py"]
    files += sorted((SF / "src" / "main").rglob("*.java"))
    files += sorted((SF / "target" / "classes").rglob("*.class"))
    files += sorted((SF / "target" / "classes").glob("*.yaml"))
    files += sorted((HERE / "configs" / "paper-benchmark-sf").glob("*.yaml"))
    return {str(p.relative_to(HERE.parent)): sha(p) for p in files}


def prepare(args):
    root, out = args.data.resolve(), args.out.resolve()
    if (out / "sample.json").exists():
        raise FileExistsError(f"Sample already frozen: {out}; use run/report or a new output directory")
    candidates = []
    for path in sorted(root.glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not manifest["id"].startswith("valentine/"):
            continue
        candidates.append({"id": manifest["id"], "family": manifest["family"],
                           "scenario_type": manifest["scenario_type"],
                           "path": str(path.parent), "manifest_sha256": sha(path)})
    selected = select_sample(candidates, args.per_stratum, args.seed)
    if not selected:
        raise ValueError("No imported Valentine scenarios")
    strata = defaultdict(lambda: {"available": 0, "sampled": 0})
    for row in candidates:
        strata[(row["family"], row["scenario_type"])]["available"] += 1
    for row in selected:
        row["key"] = hashlib.sha256(row["id"].encode()).hexdigest()[:16]
        path = Path(row["path"])
        row["metadata_sha256"] = {str(p.relative_to(path)): sha(p)
                                   for p in sorted((path / "metadata").rglob("*")) if p.is_file()}
        strata[(row["family"], row["scenario_type"])]["sampled"] += 1
    coverage = [{"family": f, "scenario_type": t, **v} for (f, t), v in sorted(strata.items())]
    json_write(out / "sample.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "data": str(root),
        "seed": args.seed, "per_stratum": args.per_stratum, "available": len(candidates),
        "sampling": "smallest SHA256(seed:scenario-id) per family/type; no replacement; all rows",
        "settings": {"seed": "65% raw Levenshtein + 35% range-aligned PMF",
                     "graph": "QUICK", "formula": "B", "coefficient": "INV_PROD",
                     "fd_filter": 0.0, "weighting": "NONE", "seed_input": "FILE",
                     "gcov": "35% range-aligned within-table gcov role-profile",
                     "primary_extractor": "full Hungarian, including nonpositive selected edges",
                     "methods": METHODS},
        "source_sha256": source_fingerprint(), "strata": coverage, "scenarios": selected})
    csv_write(out / "coverage.csv", coverage)
    print(f"Frozen {len(selected)}/{len(candidates)} scenarios in {len(coverage)} strata at {out}", flush=True)


def validate_scenario(row):
    path = Path(row["path"])
    if sha(path / "manifest.json") != row["manifest_sha256"]:
        raise ValueError("Imported manifest changed since sampling")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for entry in [*manifest["tables"].values(), manifest["ground_truth"]]:
        if sha(path / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Data/GT checksum mismatch: {entry['path']}")
    for relative, expected in row["metadata_sha256"].items():
        if sha(path / relative) != expected:
            raise ValueError(f"Profile changed since sampling: {relative}")
    return manifest


def expected_labels(manifest, side):
    return [f'{side}.{column["label"]}'.strip() for column in manifest["tables"][side]["columns"]]


def matrix(path, manifest):
    values, rows, columns = read_matrix(path)
    if rows != expected_labels(manifest, "source") or columns != expected_labels(manifest, "target"):
        raise ValueError(f"Matrix orientation/label mismatch: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite scores: {path}")
    return values


def strict_columns(path, entry):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != [c["label"] for c in entry["columns"]]:
            raise ValueError(f"Header mismatch: {path}")
        columns = [[] for _ in header]
        count = 0
        for values in reader:
            if len(values) != len(header):
                raise ValueError(f"Ragged table: {path}")
            for column, value in zip(columns, values):
                column.append(value)
            count += 1
        if count != entry["rows"]:
            raise ValueError(f"Row count mismatch: {path}")
    return columns


def jaccard(left, right):
    a, b = [set(c) for c in left], [set(c) for c in right]
    return np.array([[len(x & y) / len(x | y) if x or y else 0.0 for y in b] for x in a])


def run_one(row, args, cp):
    start = time.monotonic()
    dest = args.out / "cases" / row["key"]
    dest.mkdir(parents=True, exist_ok=True)
    status_path = dest / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["status"] == "complete":
            validate_scenario(row)
            for relative, expected in status["outputs_sha256"].items():
                if sha(dest / relative) != expected:
                    raise ValueError(f"Cached output checksum mismatch: {dest / relative}")
            return status
        if not args.retry_failed:
            return status
    status = {"id": row["id"], "key": row["key"], "status": "running", "phase": "validate"}
    json_write(status_path, status)
    try:
        manifest = validate_scenario(row)
        phase_seconds = {}
        for phase in ("first", "flood"):
            status["phase"] = phase
            json_write(status_path, status)
            command = [str(args.java), "-Xmx4g", "-Dfile.encoding=UTF-8", "-cp", cp,
                       "ValentineSweep", phase, row["path"], str(dest), str(args.out / "seeds")]
            phase_start = time.monotonic()
            with (dest / f"{phase}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=SF, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=True)
            phase_seconds[phase] = time.monotonic() - phase_start
            if phase == "first":
                lex, pmf = matrix(dest / "levenshtein.csv", manifest), matrix(dest / "pmf.csv", manifest)
                seed = scale_aligned_fuse(lex, pmf, 0.35)
                original = Path(row["path"])
                write_matrix(args.out / "seeds" / original.parent.name / original.name / "seed.csv",
                             seed, expected_labels(manifest, "source"), expected_labels(manifest, "target"))
                write_matrix(dest / "seed.csv", seed, expected_labels(manifest, "source"), expected_labels(manifest, "target"))
        status["phase"] = "features"
        columns = [non_null_plain_columns(strict_columns(Path(row["path"]) / manifest["tables"][s]["path"],
                                                       manifest["tables"][s])) for s in ("source", "target")]
        profiles = [within_ind_role_profile(within_ind_evidence(c)[0]) for c in columns]
        for name, scores in (("gcov", vector_similarity(*profiles)), ("jaccard", jaccard(*columns))):
            write_matrix(dest / f"{name}.csv", scores, expected_labels(manifest, "source"), expected_labels(manifest, "target"))
        # Verify all production matrices before marking the scenario complete.
        for name in ("sf", "sf_fd"):
            matrix(dest / f"{name}.csv", manifest)
        status.update(status="complete", phase_seconds=phase_seconds,
                      outputs_sha256={p.name: sha(p) for p in sorted(dest.glob("*.csv"))})
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    status["seconds"] = time.monotonic() - start
    json_write(status_path, status)
    return status


def run(args):
    sample = json.loads((args.out / "sample.json").read_text())
    if sample["source_sha256"] != source_fingerprint():
        raise ValueError("Code/config/classes changed since freeze; prepare a new sweep")
    classes = args.out / "classes"
    classes.mkdir(exist_ok=True)
    subprocess.run([str(args.javac), "-proc:none", "-cp", str(SF / "target" / "classes"),
                    "-d", str(classes), str(HERE / "java" / "ValentineSweep.java")], check=True)
    cp = os.pathsep.join([str(classes), str(HERE / "configs" / "paper-benchmark-sf"),
                         str(SF / "target" / "classes"), (SF / "target" / "cp.txt").read_text().strip()])
    json_write(args.out / "runtime.json", {
        "java": subprocess.run([str(args.java), "-version"], capture_output=True, text=True).stderr,
        "javac": str(args.javac), "python": os.sys.version, "numpy": np.__version__,
        "workers": args.workers, "java_heap": "4g per process", "phase_timeout_seconds": args.timeout,
        "classpath": cp})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, row, args, cp): row for row in sample["scenarios"]}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f'{index}/{len(futures)} {result["status"]} {result["id"]} ({result.get("seconds", 0):.1f}s)', flush=True)


def report(args):
    sample = json.loads((args.out / "sample.json").read_text())
    rows, failures, cases = [], [], []
    for row in sample["scenarios"]:
        dest = args.out / "cases" / row["key"]
        status = json.loads((dest / "status.json").read_text()) if (dest / "status.json").exists() else {"status": "not_run"}
        if status["status"] != "complete":
            failures.append({**row, "status": status})
            continue
        manifest = validate_scenario(row)
        for relative, expected in status["outputs_sha256"].items():
            if sha(dest / relative) != expected:
                raise ValueError(f"Modified output: {dest / relative}")
        scores = {name: matrix(dest / f"{name}.csv", manifest) for name in
                  ("levenshtein", "pmf", "jaccard", "seed", "gcov", "sf", "sf_fd")}
        gt = read_matrix(Path(row["path"]) / manifest["ground_truth"]["path"])[0] > 0.5
        expected = np.zeros(gt.shape, dtype=bool)
        for i, j in manifest["ground_truth"]["pairs"]:
            expected[i, j] = True
        if not np.array_equal(gt, expected):
            raise ValueError("GT export differs from positional manifest")
        matrices = dict(zip(METHODS, [scores["levenshtein"], scores["pmf"], scores["jaccard"], scores["seed"],
                        scale_aligned_fuse(scores["seed"], scores["gcov"], .35), scores["sf"], scores["sf_fd"],
                        scale_aligned_fuse(scores["sf_fd"], scores["gcov"], .35)]))
        # Upper bound for *full* 1:1 assignment under the supplied partial/multi-match GT.
        ceiling = match_metrics(hungarian(gt.astype(float), drop_nonpositive=False), gt)["f1"]
        case = {"key": row["key"], "id": row["id"], "family": row["family"], "scenario_type": row["scenario_type"],
                "n_source": gt.shape[0], "n_target": gt.shape[1], "n_gt": int(gt.sum()),
                "full_assignment_f1_ceiling": ceiling, "seconds": status["seconds"],
                "source_rows": manifest["tables"]["source"]["rows"], "target_rows": manifest["tables"]["target"]["rows"],
                "one_to_many_sources": manifest["ground_truth"]["one_to_many_sources"],
                "many_to_one_targets": manifest["ground_truth"]["many_to_one_targets"]}
        cases.append(case)
        for method, values in matrices.items():
            if values.shape != gt.shape:
                raise ValueError("Score/GT shape mismatch")
            diagnostics = scoring_diagnostics(ScoreMatrix(values, gt))
            metrics = match_metrics(hungarian(values, drop_nonpositive=False), gt)
            mutual = match_metrics(mutual_top1(values), gt)
            rows.append({**case, "method": method, **metrics,
                         "mutual_top1_f1": mutual["f1"], "mutual_top1_n_match": mutual["n_match"],
                         **{k: diagnostics[k] for k in ("recall_at_gt_pessimistic", "recall_at_gt_optimistic",
                            "mrr_pessimistic", "mrr_optimistic", "mean_row_entropy", "frac_rows_tied_at_max")}})
    json_write(args.out / "failures.json", failures)
    csv_write(args.out / "scenarios.csv", rows)
    csv_write(args.out / "cases.csv", cases)
    if not rows:
        raise ValueError("No completed scenarios to report")
    summary = []
    metrics = ("f1", "precision", "recall", "recall_at_gt_pessimistic", "recall_at_gt_optimistic",
               "mrr_pessimistic", "mrr_optimistic", "mutual_top1_f1", "full_assignment_f1_ceiling")
    for dimension in ("overall", "family", "scenario_type", "stratum"):
        groups = defaultdict(list)
        for row in rows:
            group = ("all" if dimension == "overall" else row["family"] + "/" + row["scenario_type"]
                     if dimension == "stratum" else row[dimension])
            groups[(group, row["method"])].append(row)
        for (group, method), values in sorted(groups.items()):
            summary.append({"dimension": dimension, "group": group, "method": method, "n": len(values),
                            **{k: float(np.mean([v[k] for v in values])) for k in metrics}})
    csv_write(args.out / "summary.csv", summary)
    lookup = {(r["id"], r["method"]): r for r in rows}
    comparisons = []
    for baseline in METHODS[:-1]:
        deltas = np.array([lookup[(c["id"], CURRENT)]["f1"] - lookup[(c["id"], baseline)]["f1"] for c in cases])
        comparisons.append({"method": CURRENT, "baseline": baseline, "mean_f1_delta": float(deltas.mean()),
                            "wins": int((deltas > 1e-9).sum()), "ties": int((abs(deltas) <= 1e-9).sum()),
                            "losses": int((deltas < -1e-9).sum())})
    csv_write(args.out / "comparisons.csv", comparisons)
    text = ["# Valentine sampled sweep", "", f"{len(cases)}/{len(sample['scenarios'])} completed; {len(failures)} failed/not run.",
            f"Fixed seed {sample['seed']}; up to {sample['per_stratum']} per family/type; {sample['available']} eligible scenarios.",
            "Plaintext; full tables; frozen existing fusion weights; cached FD profiles; no parameter tuning.", "",
            "## Overall (scenario means)", "",
            "| Approach | F1 | Recall@GT (pess.–opt.) | MRR (pess.–opt.) | Mutual-top1 F1 |",
            "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        r = next(r for r in summary if r["dimension"] == "overall" and r["method"] == method)
        text.append(f'| {method} | {r["f1"]:.3f} | {r["recall_at_gt_pessimistic"]:.3f}–{r["recall_at_gt_optimistic"]:.3f} | '
                    f'{r["mrr_pessimistic"]:.3f}–{r["mrr_optimistic"]:.3f} | {r["mutual_top1_f1"]:.3f} |')
    for dimension, title in (("family", "By family"), ("scenario_type", "By scenario type")):
        text += ["", "## " + title, "", "| Group | n | Levenshtein | PMF | Jaccard | Seed | Seed+gcov | SF | SF+FD | Current |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in sorted({r["group"] for r in summary if r["dimension"] == dimension}):
            values = [next(r for r in summary if r["dimension"] == dimension and r["group"] == group and r["method"] == m) for m in METHODS]
            text.append(f'| {group} | {values[0]["n"]} | ' + " | ".join(f'{v["f1"]:.3f}' for v in values) + " |")
    text += ["", "## Current pipeline versus controls", "", "| Baseline | Mean F1 difference | Wins / ties / losses |", "|---|---:|---:|"]
    for c in comparisons:
        text.append(f'| {c["baseline"]} | {c["mean_f1_delta"]:+.3f} | {c["wins"]} / {c["ties"]} / {c["losses"]} |')
    text += ["", "## Lowest current-pipeline F1", "", "| Scenario | F1 | Full-assignment ceiling | Recall@GT (pess.) |", "|---|---:|---:|---:|"]
    for r in sorted((r for r in rows if r["method"] == CURRENT), key=lambda r: (r["f1"], r["id"]))[:10]:
        text.append(f'| {r["id"]} | {r["f1"]:.3f} | {r["full_assignment_f1_ceiling"]:.3f} | {r["recall_at_gt_pessimistic"]:.3f} |')
    text += ["", "## Interpretation limits", "",
             "- F1 uses full rectangular Hungarian assignment for every method, as in the earlier representation ablation. This forces min(source,target) matches; partial GT imposes an F1 ceiling even for an oracle. See cases.csv and the alternative mutual-top1 diagnostic (ties included).",
             "- Recall@GT ranks all matrix cells at k=|GT|; MRR ranks the first relevant target per matched source row. Both report adversarial/favourable ties. Unmatched source rows are excluded from MRR.",
             "- Sample means are not population-weighted estimates. Families/types have different sample sizes; related synthetic variants are not independent trials. Excluded malformed/unresolvable imports are outside the sampling frame; GDC is not Valentine and is excluded.",
             "- Current = 65% raw Levenshtein + 35% range-aligned PMF, then QUICK SF with cached FDs (filter 0, unweighted, B/INV_PROD), then 35% range-aligned gcov role similarity. SF and SF+FD controls use that identical FILE seed. No first-line threshold boosting is applied.",
             "- FD profiles were discovered at relaxed threshold 0.95 and max arity 2; these are newly profiled imports, not the paper's supplied metadata. gcov role descriptors are computed on demand once and retained in this run; they are not the thresholded SPIND legacy export.",
             "- PMF uses the production DistributionMatcher, including empty-string categories. Jaccard and gcov reuse the earlier ablation's null-token removal and value trimming. Jaccard is cross-table plaintext value overlap, not an encrypted baseline.",
             "- This sweep does not compare PRISMA/PDED, encrypted variants, or every historical SF tuning variant. It is a transfer check of the fixed current representation pipeline.", ""]
    (args.out / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    print(f"Report: {args.out / 'REPORT.md'}; {len(rows)} method/scenario rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "all"))
    parser.add_argument("--data", type=Path, default=HERE / "generated" / "benchmark-full")
    parser.add_argument("--out", type=Path, default=HERE / "reports" / "valentine_sample_20260903")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--java", type=Path, required=False, default=Path("java"))
    parser.add_argument("--javac", type=Path, required=False, default=Path("javac"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.per_stratum < 1 or args.workers < 1 or args.timeout < 1:
        parser.error("per-stratum, workers and timeout must be positive")
    if args.command in ("prepare", "all"):
        prepare(args)
    if args.command in ("run", "all"):
        run(args)
    if args.command in ("report", "all"):
        report(args)


if __name__ == "__main__":
    main()
