"""TODO 40a: fixed fused rerun on the audited MD5 Valentine cohort.

The protocol is frozen before scores are inspected. PMF is the same production
matrix used by the all-382 plaintext expansion. QUICK SF+FD is rerun with PMF
as its only seed in both regimes. Equality-partition roles and Kang-Naughton
are computed once from plaintext, then reused numerically after verified
audited recoding; no labels or cross-schema value equality enter those channels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / ".deps"))
import numpy as np
from scipy.stats import wilcoxon

from encrypted_feature_ablation import read_matrix as read_truth, scale_aligned_fuse
from kang_naughton import kn_graph_scores, kn_profile_scores, mi_matrix
from pmf_uncertainty_combination import equal_role_block
from sf_representation_ablation import write_matrix
from smx.adapters import read_matrix
from smx.core import ScoreMatrix, evaluate
from smx.relational_roles import relation_matrices, role_scores
from valentine_sweep import csv_write, json_write, sha, strict_columns

SF = HERE.parent / "code" / "schematch"
WEIGHT = 0.35
METHODS = {
    "pmf": "PMF",
    "sf_pmf": "QUICK SF+FD(PMF)",
    "sf_pmf_gcov": "QUICK SF+FD+PMF+gcov",
    "sf_pmf_jaccard": "QUICK SF+FD+PMF+Jaccard",
    "pmf_gcov_u": "PMF+gcov+uncertainty",
    "kn_graph": "Kang-Naughton graph",
    "kn_profile": "Kang-Naughton profile",
}
REGIMES = ("plain", "audited_md5")
POLICIES = ("hungarian", "mutual_top1")


def fingerprints():
    paths = [Path(__file__), HERE / "java" / "AuditedPmfSf.java",
             HERE / "make_md5_audited.py",
             HERE / "kang_naughton.py", HERE / "pmf_uncertainty_combination.py",
             HERE / "smx" / "core.py", HERE / "smx" / "adapters.py",
             HERE / "smx" / "relational_roles.py",
             HERE / "encrypted_feature_ablation.py"]
    paths += sorted((SF / "src" / "main").rglob("*.java"))
    return {str(path.resolve()): sha(path) for path in paths}


def table_csv(scenario: Path, side: str) -> Path:
    files = sorted((scenario / side).glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one {side} CSV under {scenario}, got {len(files)}")
    return files[0]


def labels(scenario: Path, side: str) -> list[str]:
    with table_csv(scenario, side).open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    return [f"{side}.{name}" for name in header]


def metadata_hashes(folder: Path):
    return {str(path.relative_to(folder)): sha(path)
            for path in sorted((folder / "metadata").rglob("*")) if path.is_file()}


def prepare(args):
    if (args.out / "protocol.json").exists():
        raise FileExistsError("protocol exists; use run/report or choose another --out")
    prior_path = args.prior / "protocol.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    old80 = set()
    paper_families = {
        "TPC-DI-Unionable": "TPC-DI",
        "ChEMBL-Unionable": "ChEMBL",
        "Valentine-Wikidata-Unionable": "Wikidata",
        "Magellan-Unionable": "Magellan",
    }
    for dataset, family in paper_families.items():
        root = args.paper_data / dataset
        if root.is_dir():
            old80.update((family.casefold(), path.name.casefold())
                         for path in root.iterdir() if path.is_dir())
    cases = []
    for row in prior["cases"]:
        plain = Path(row["path"])
        md5 = args.md5 / plain.parent.name / plain.name
        transform = md5 / "transform-manifest.json"
        if not md5.is_dir() or not transform.is_file():
            raise ValueError(f"missing audited scenario {md5}")
        prior_case = args.prior / "cases" / row["key"]
        pmf = prior_case / "pmf.csv"
        status = prior_case / "status.json"
        if not pmf.is_file() or not status.is_file():
            raise ValueError(f"missing prior PMF {row['id']}")
        cases.append({
            "id": row["id"], "key": row["key"], "cohort": row["cohort"],
            "family": row["family"], "scenario_type": row["scenario_type"],
            "scenario": plain.name, "dataset": plain.parent.name,
            "plain": str(plain.resolve()), "audited_md5": str(md5.resolve()),
            "plain_manifest_sha256": sha(plain / "manifest.json"),
            "md5_transform_sha256": sha(transform),
            "plain_metadata_sha256": metadata_hashes(plain),
            "md5_metadata_sha256": metadata_hashes(md5),
            "prior_pmf": str(pmf.resolve()), "prior_pmf_sha256": sha(pmf),
            "prior_status_sha256": sha(status),
            "old80": (row["scenario_type"] == "Unionable" and
                      (row["family"].casefold(), plain.name.casefold()) in old80),
        })
    counts = Counter(row["cohort"] for row in cases)
    if len(cases) != 382 or counts != Counter(inspected44=44, remaining338=338):
        raise ValueError(f"wrong case counts: {len(cases)}, {counts}")
    matched_old80 = {(row["family"].casefold(), row["scenario"].casefold())
                     for row in cases if row["old80"]}
    missing_old80 = sorted(f"{family}/{scenario}" for family, scenario
                           in old80 - matched_old80)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": fingerprints(),
        "prior_protocol": str(prior_path.resolve()),
        "prior_protocol_sha256": sha(prior_path),
        "audited_root": str(args.md5.resolve()),
        "cases": cases, "methods": METHODS, "regimes": REGIMES,
        "policies": POLICIES, "weight": WEIGHT,
        "freeze": "Fixed before audited fused outcomes: no family router, new role, threshold, or weight search.",
        "questions": [
            "Jaccard versus gcov for full-Hungarian F1",
            "PMF+gcov+U versus the assignment candidate for mutual F1, Recall@GT and MRR",
            "QUICK SF+FD(PMF) versus the identical PMF seed",
            "Kang-Naughton plaintext/audited invariance",
        ],
        "reuse": "PMF from frozen production all-382 run; equality roles and KN computed on plaintext and reused numerically; SF production-run in both regimes.",
        "old80_expected_cases": len(old80),
        "old80_cases": sum(row["old80"] for row in cases),
        "old80_missing": missing_old80,
        "python": sys.version,
    }
    json_write(args.out / "protocol.json", protocol)
    print(f"frozen {len(cases)} cases: {dict(counts)}; "
          f"paper intersection={protocol['old80_cases']}/{protocol['old80_expected_cases']}; "
          f"missing={missing_old80}")


def verify(args):
    protocol = json.loads((args.out / "protocol.json").read_text(encoding="utf-8"))
    if protocol["source_sha256"] != fingerprints():
        raise ValueError("implementation changed after freeze")
    if sha(Path(protocol["prior_protocol"])) != protocol["prior_protocol_sha256"]:
        raise ValueError("prior protocol changed")
    for row in protocol["cases"]:
        plain, audited = Path(row["plain"]), Path(row["audited_md5"])
        checks = (
            (sha(plain / "manifest.json"), row["plain_manifest_sha256"], "plain manifest"),
            (sha(audited / "transform-manifest.json"), row["md5_transform_sha256"], "MD5 transform"),
            (metadata_hashes(plain), row["plain_metadata_sha256"], "plain metadata"),
            (metadata_hashes(audited), row["md5_metadata_sha256"], "MD5 metadata"),
            (sha(Path(row["prior_pmf"])), row["prior_pmf_sha256"], "prior PMF"),
            (sha(Path(row["prior_pmf"]).parent / "status.json"),
             row["prior_status_sha256"], "prior status"),
        )
        for actual, expected, name in checks:
            if actual != expected:
                raise ValueError(f"{name} changed after freeze: {row['id']}")
    return protocol


def compile_bridge(args):
    classes = args.out / "classes"
    classes.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.javac), "-proc:none", "-cp", str(SF / "target" / "classes"),
                    "-d", str(classes), str(HERE / "java" / "AuditedPmfSf.java")], check=True)
    return os.pathsep.join([str(classes.resolve()), str((HERE / "configs" / "paper-benchmark-sf").resolve()),
                            str(SF / "target" / "classes"),
                            (SF / "cp.txt").read_text().strip()])


def checked(path: Path, shape=None):
    matrix, rows, columns = read_matrix(path)
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"shape mismatch {path}: {matrix.shape} != {shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"non-finite matrix {path}")
    return matrix, rows, columns


def stage_seed(args, row, regime, pmf):
    scenario = Path(row[regime])
    source_labels, target_labels = labels(scenario, "source"), labels(scenario, "target")
    if pmf.shape != (len(source_labels), len(target_labels)):
        raise ValueError(f"PMF/header shape mismatch: {row['id']}/{regime}")
    path = args.out / "seeds" / regime / row["dataset"] / row["scenario"] / "seed.csv"
    write_matrix(path, pmf, source_labels, target_labels)
    return source_labels, target_labels


def run_sf(args, row, regime, cp, dest, pmf):
    source_labels, target_labels = stage_seed(args, row, regime, pmf)
    output = dest / regime
    output.mkdir(parents=True, exist_ok=True)
    command = [str(args.java), "-Xmx4g", "-Dfile.encoding=UTF-8", "-cp", cp,
        "AuditedPmfSf", row[regime], str(output.resolve()),
               str((args.out / "seeds" / regime).resolve())]
    with (output / "sf.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=SF, stdout=log, stderr=subprocess.STDOUT,
                       timeout=args.timeout, check=True)
    matrix, actual_rows, actual_columns = checked(output / "sf_fd_pmf.csv", pmf.shape)
    if actual_rows != source_labels or actual_columns != target_labels:
        raise ValueError(f"SF labels differ: {row['id']}/{regime}")
    return matrix


def save(path, matrix, source_labels, target_labels):
    write_matrix(path, matrix, source_labels, target_labels)


def run_case(args, row, cp):
    dest = args.out / "cases" / row["key"]
    dest.mkdir(parents=True, exist_ok=True)
    status_path = dest / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("status") == "complete":
            for name, digest in status["outputs_sha256"].items():
                if sha(dest / name) != digest:
                    raise ValueError(f"cached output changed: {dest/name}")
            return status
        if not args.retry_failed:
            return status
    started = time.monotonic()
    status = {"id": row["id"], "cohort": row["cohort"], "status": "running"}
    json_write(status_path, status)
    try:
        plain = Path(row["plain"])
        manifest = json.loads((plain / "manifest.json").read_text(encoding="utf-8"))
        raw = [strict_columns(table_csv(plain, side), manifest["tables"][side])
               for side in ("source", "target")]
        pmf, _, _ = checked(Path(row["prior_pmf"]))
        expected_shape = (len(raw[0]), len(raw[1]))
        if pmf.shape != expected_shape:
            raise ValueError(f"PMF shape mismatch {row['id']}")
        relations = [relation_matrices(columns)[0] for columns in raw]
        role = {name: role_scores(relations[0][name], relations[1][name])
                for name in ("gcov", "jaccard", "uncertainty")}
        mi = [mi_matrix(columns) for columns in raw]
        kn = {"kn_graph": kn_graph_scores(*mi), "kn_profile": kn_profile_scores(*mi)}
        plain_labels = labels(plain, "source"), labels(plain, "target")
        for name, matrix in {"pmf": pmf, **role, **kn}.items():
            save(dest / f"{name}.csv", matrix, *plain_labels)

        sf = {regime: run_sf(args, row, regime, cp, dest, pmf) for regime in REGIMES}
        methods = {}
        for regime in REGIMES:
            base = sf[regime]
            methods[regime] = {
                "pmf": pmf,
                "sf_pmf": base,
                "sf_pmf_gcov": scale_aligned_fuse(base, role["gcov"], WEIGHT),
                "sf_pmf_jaccard": scale_aligned_fuse(base, role["jaccard"], WEIGHT),
                "pmf_gcov_u": scale_aligned_fuse(
                    pmf, equal_role_block(role["gcov"], role["uncertainty"]), WEIGHT),
                **kn,
            }
            regime_labels = labels(Path(row[regime]), "source"), labels(Path(row[regime]), "target")
            for name, matrix in methods[regime].items():
                save(dest / regime / f"{name}.csv", matrix, *regime_labels)
        max_sf_change = float(np.max(np.abs(sf["plain"] - sf["audited_md5"])))
        json_write(dest / "details.json", {
            "max_sf_plain_md5_change": max_sf_change,
            "role_spans": {name: float(np.ptp(matrix)) for name, matrix in role.items()},
            "method_count": len(METHODS),
        })
        outputs = [path for path in dest.rglob("*")
                   if path.is_file() and path.name not in ("status.json", "sf.log")]
        status.update(status="complete", seconds=time.monotonic() - started,
                      max_sf_plain_md5_change=max_sf_change,
                      outputs_sha256={str(path.relative_to(dest)): sha(path) for path in outputs})
    except Exception as exc:
        status.update(status="failed", seconds=time.monotonic() - started,
                      error=f"{type(exc).__name__}: {exc}")
    json_write(status_path, status)
    return status


def run(args):
    protocol = verify(args)
    cp = compile_bridge(args)
    cases = protocol["cases"][:args.limit or None]
    json_write(args.out / "runtime.json", {
        "java": subprocess.run([str(args.java), "-version"], capture_output=True,
                               text=True).stderr,
        "workers": args.workers, "timeout": args.timeout,
    })
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, args, row, cp) for row in cases]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"{index}/{len(cases)} {result['status']} {result['id']} "
                  f"({result.get('seconds', 0):.1f}s)", flush=True)


def score_case(args, row):
    dest = args.out / "cases" / row["key"]
    status = json.loads((dest / "status.json").read_text())
    if status.get("status") != "complete":
        return [], {"id": row["id"], "status": status.get("status", "missing"),
                    "error": status.get("error", "")}
    gt = read_truth(sorted((Path(row["plain"]) / "ground_truth").glob("*.csv"))[0])[0] > .5
    rows = []
    for regime in REGIMES:
        for key, method in METHODS.items():
            matrix = checked(dest / regime / f"{key}.csv", gt.shape)[0]
            for result in evaluate(ScoreMatrix(scores=matrix, ground_truth=gt), policies=POLICIES):
                rows.append({
                    "id": row["id"], "key": row["key"], "cohort": row["cohort"],
                    "family": row["family"], "scenario_type": row["scenario_type"],
                    "scenario": row["scenario"], "old80": row["old80"],
                    "regime": regime, "method_key": key, "method": method,
                    "extractor": result["extractor"], "f1": result["f1"],
                    "recall_at_gt": result["recall_at_gt_pessimistic"],
                    "mrr": result["mrr_pessimistic"],
                })
    return rows, None


def aggregate(rows):
    output = []
    for cohort in ("all382", "inspected44", "remaining338"):
        for regime in REGIMES:
            for key, method in METHODS.items():
                for extractor in POLICIES:
                    selected = [row for row in rows if row["regime"] == regime
                                and row["method_key"] == key and row["extractor"] == extractor
                                and (cohort == "all382" or row["cohort"] == cohort)]
                    if selected:
                        output.append({"cohort": cohort, "regime": regime,
                                       "method_key": key, "method": method,
                                       "extractor": extractor, "n": len(selected),
                                       "f1": float(np.mean([float(x["f1"]) for x in selected])),
                                       "recall_at_gt": float(np.mean([float(x["recall_at_gt"]) for x in selected])),
                                       "mrr": float(np.mean([float(x["mrr"]) for x in selected]))})
    return output


def paired(rows, treatment, baseline, extractor, regime="audited_md5", subset="all382"):
    selected = [row for row in rows if row["regime"] == regime and row["extractor"] == extractor
                and (subset == "all382" or (subset == "old80" and row["old80"]))]
    index = {(row["method_key"], row["id"]): float(row["f1"]) for row in selected}
    ids = sorted({row["id"] for row in selected})
    delta = np.asarray([index[(treatment, item)] - index[(baseline, item)] for item in ids])
    nonzero = delta[np.abs(delta) > 1e-12]
    return {"subset": subset, "regime": regime, "treatment": treatment,
            "baseline": baseline, "extractor": extractor, "n": len(ids),
            "mean_f1_delta": float(delta.mean()), "wins": int((delta > 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
            "wilcoxon_p_nonzero": float(wilcoxon(nonzero).pvalue) if nonzero.size else float("nan")}


def report(args):
    protocol = verify(args)
    rows, failures = [], []
    for row in protocol["cases"]:
        scored, failure = score_case(args, row)
        rows.extend(scored)
        if failure:
            failures.append(failure)
    if not rows:
        raise ValueError("no completed cases")
    csv_write(args.out / "scenarios.csv", rows)
    csv_write(args.out / "summary.csv", aggregate(rows))
    comparisons = []
    for subset in ("all382", "old80"):
        for extractor in POLICIES:
            for treatment, baseline in (
                ("sf_pmf_jaccard", "sf_pmf_gcov"),
                ("pmf_gcov_u", "sf_pmf_jaccard"),
                ("sf_pmf", "pmf"),
            ):
                comparisons.append(paired(rows, treatment, baseline, extractor,
                                          subset=subset))
    csv_write(args.out / "comparisons.csv", comparisons)
    csv_write(args.out / "failures.csv", failures or [{"id": "", "status": "", "error": ""}])
    sf_changes = [json.loads((args.out / "cases" / row["key"] / "details.json").read_text())
                  ["max_sf_plain_md5_change"] for row in protocol["cases"]
                  if (args.out / "cases" / row["key"] / "details.json").is_file()]
    json_write(args.out / "audit.json", {
        "completed_cases": len({row["id"] for row in rows}),
        "failed_cases": len(failures),
        "metric_rows": len(rows),
        "max_sf_plain_md5_change": max(sf_changes, default=float("nan")),
    })
    print(json.dumps(json.loads((args.out / "audit.json").read_text()), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--prior", type=Path, default=Path("reports/valentine_full_382_20260903"))
    parser.add_argument("--md5", type=Path, default=Path("generated/benchmark-full-md5"))
    parser.add_argument("--paper-data", type=Path, default=Path("../code/schematch/data"))
    parser.add_argument("--out", type=Path, default=Path("reports/audited_md5_fused_20260904"))
    parser.add_argument("--java", type=Path, default=Path("java"))
    parser.add_argument("--javac", type=Path, default=Path("javac"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "run": run, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
