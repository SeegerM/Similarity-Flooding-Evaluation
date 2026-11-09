"""Frozen full matcher experiment for the controlled MusicBrainz/Lahman corpus.

The data and matcher choices are pre-registered in
``configs/controlled-benchmark-protocol.json``.  This runner is deliberately
resumable: production matcher matrices are materialised per scenario before
the deterministic feature arms and common evaluation are computed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from anchor_generalization import anchor_matrix, bare, load_side
from class_comparison_conditioned import read_columns
from compose_gate import fuse
from dc_gate import anchor_ranking as dc_anchor_ranking
from dc_gate import build_similarity, load_dcs
from encrypted_feature_ablation import scale_aligned_fuse
from gated_method import dc_channel, fd_channel, scenario_evidence
from od_role import cosine_similarity, od_profile
from order_role import role_profile
from pmf_uncertainty_combination import equal_role_block
from sf_representation_ablation import write_matrix
from smx.adapters import read_matrix
from smx.core import ScoreMatrix, evaluate
from smx.relational_roles import relation_matrices, role_scores
from transferability_table import TAU, od_anchor_ranking, order_sets
from valentine_sweep import expected_labels, matrix, sha, strict_columns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "generated" / "controlled-imported-v2"
OUT = HERE / "reports" / "controlled_matchers_20260904"
PROTOCOL = HERE / "configs" / "controlled-benchmark-protocol.json"
SF = ROOT / "code" / "schematch"
PDED = ROOT / "code" / "PDED"
WEIGHT = 0.35
FD_BUDGET = 0.232
DC_BUDGET = 0.541
OD_BUDGET = 0.455
SEED = 20260904
ARMS = (
    "levenshtein", "pmf", "prisma", "pded", "sf_fd_pmf",
    "sf_fd_pmf_jaccard", "pmf_gcov_uncertainty",
    "seed_kn_fd_anchor_dc_anchor",
    "seed_kn_fd_anchor_dc_anchor_od_anchor",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        fieldnames.extend(k for row in rows for k in row if k not in fieldnames)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cases(data: Path) -> list[dict]:
    out = []
    for mp in sorted(data.glob("*/*/manifest.json")):
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        design = json.loads((mp.parent / manifest["design"]["path"]).read_text(encoding="utf-8"))
        cid = manifest["id"]
        out.append({
            "id": cid,
            "key": hashlib.sha256(cid.encode()).hexdigest()[:16],
            "dataset": manifest["family"],
            "scenario_type": manifest["scenario_type"],
            "evidence_regime": design["evidence_regime"],
            "path": str(mp.parent.resolve()),
            "group": mp.parent.parent.name,
            "scenario": mp.parent.name,
            "manifest_sha256": sha(mp),
        })
    if len(out) != 16:
        raise ValueError(f"Frozen protocol requires 16 cases, found {len(out)}")
    return out


def validate_case(row: dict) -> dict:
    folder = Path(row["path"])
    mp = folder / "manifest.json"
    if sha(mp) != row["manifest_sha256"]:
        raise ValueError(f"Manifest changed: {row['id']}")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for entry in [*manifest["tables"].values(), manifest["ground_truth"], manifest["design"]]:
        if sha(folder / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen artefact changed: {row['id']} / {entry['path']}")
    return manifest


def prepare(args) -> list[dict]:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if tuple(a["id"] for a in protocol["arms"]) != ARMS:
        raise ValueError("Runner arms do not match frozen protocol")
    rows = cases(args.data)
    for row in rows:
        validate_case(row)
    runtime = {
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha(args.protocol),
        "data": str(args.data.resolve()),
        "created_unix": time.time(),
        "cases": rows,
        "arms": list(ARMS),
        "settings": protocol["fixed_settings"],
        "runtime_addendum": {
            "pded_distribution_weight": WEIGHT,
            "prisma_postprocessing": False,
            "prisma_gamma_structural_attribute": WEIGHT,
            "prisma_feature": "Distribution",
            "prisma_gdep_filtering": "gdep_threshold_0.3",
            "multi_channel_fusion": "independent min-max; channels split one 0.35 evidence budget equally",
            "od_threshold_tau": TAU,
        },
    }
    write_json(args.out / "runtime_protocol.json", runtime)
    write_csv(args.out / "cases.csv", rows)
    return rows


def load_runtime(args) -> list[dict]:
    path = args.out / "runtime_protocol.json"
    if not path.exists():
        return prepare(args)
    runtime = json.loads(path.read_text(encoding="utf-8"))
    if runtime["protocol_sha256"] != sha(args.protocol):
        raise ValueError("Frozen protocol changed since run preparation")
    rows = runtime["cases"]
    for row in rows:
        validate_case(row)
    return rows


def sf_classpath(args) -> str:
    classes = args.out / "classes"
    classes.mkdir(parents=True, exist_ok=True)
    target = classes / "ValentineSweep.class"
    source = HERE / "java" / "ValentineSweep.java"
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        subprocess.run([str(args.javac), "-proc:none", "-cp", str(SF / "target" / "classes"),
                        "-d", str(classes), str(source)], check=True)
    return os.pathsep.join([str(classes), str(HERE / "configs" / "paper-benchmark-sf"),
                            str(SF / "target" / "classes"),
                            (SF / "target" / "cp.txt").read_text().strip()])


def run_sf(args, rows: list[dict]) -> None:
    cp = sf_classpath(args)
    for index, row in enumerate(rows, 1):
        manifest = validate_case(row)
        dest = args.out / "cases" / row["key"]
        dest.mkdir(parents=True, exist_ok=True)
        complete = all((dest / f"{name}.csv").exists()
                       for name in ("levenshtein", "pmf", "sf", "sf_fd"))
        if complete:
            for name in ("levenshtein", "pmf", "sf", "sf_fd"):
                matrix(dest / f"{name}.csv", manifest)
            print(f"SF {index:02d}/16 cached {row['scenario']}", flush=True)
            continue
        for phase in ("first", "flood"):
            command = [str(args.java), "-Xmx4g", "-Dfile.encoding=UTF-8", "-cp", cp,
                       "ValentineSweep", phase, row["path"], str(dest), str(args.out / "seeds")]
            with (dest / f"{phase}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=SF, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=True)
            if phase == "first":
                lex = matrix(dest / "levenshtein.csv", manifest)
                pmf = matrix(dest / "pmf.csv", manifest)
                seed = scale_aligned_fuse(lex, pmf, WEIGHT)
                seed_path = args.out / "seeds" / row["group"] / row["scenario"] / "seed.csv"
                write_matrix(seed_path, seed, expected_labels(manifest, "source"), expected_labels(manifest, "target"))
                write_matrix(dest / "seed.csv", seed, expected_labels(manifest, "source"), expected_labels(manifest, "target"))
        for name in ("levenshtein", "pmf", "sf", "sf_fd"):
            matrix(dest / f"{name}.csv", manifest)
        print(f"SF {index:02d}/16 complete {row['scenario']}", flush=True)


def run_pded(args, rows: list[dict]) -> None:
    cp = os.pathsep.join([str(PDED / "target" / "classes"), (PDED / "cp.txt").read_text().strip()])
    for index, group in enumerate(sorted({r["group"] for r in rows}), 1):
        destination = args.out / "pded" / group
        expected = [destination / r["scenario"] / "pded.csv" for r in rows if r["group"] == group]
        if all(p.exists() for p in expected):
            print(f"PDED {index:02d}/8 cached {group}", flush=True)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        command = [str(args.java), "-Xmx4g", "-cp", cp,
                   "de.cleaning.integration.PdedSchematchRunner",
                   str(args.data / group), str(destination), str(WEIGHT)]
        with (destination / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=PDED, stdout=log, stderr=subprocess.STDOUT,
                           timeout=args.timeout, check=True)
        if not all(p.exists() for p in expected):
            raise ValueError(f"PDED did not materialise both cases for {group}")
        print(f"PDED {index:02d}/8 complete {group}", flush=True)


def od_channel(s_raw, t_raw, s_cols, t_cols, anchors) -> tuple[np.ndarray | None, dict]:
    _, s_ods = order_sets(s_raw, s_cols, SEED)
    _, t_ods = order_sets(t_raw, t_cols, SEED)
    diagnostics = {"source_ods": len(s_ods), "target_ods": len(t_ods)}
    if not s_ods or not t_ods:
        return None, diagnostics
    sim = build_similarity(anchors, s_cols, t_cols)
    s_rank = od_anchor_ranking(s_ods, t_ods, sim, "st")
    t_rank = od_anchor_ranking(t_ods, s_ods, sim, "ts")
    s_keep = sorted(s_rank, key=lambda x: -s_rank[x])[:max(1, round(OD_BUDGET * len(s_ods)))]
    t_keep = sorted(t_rank, key=lambda x: -t_rank[x])[:max(1, round(OD_BUDGET * len(t_ods)))]
    diagnostics.update(source_selected=len(s_keep), target_selected=len(t_keep))
    return cosine_similarity(od_profile(s_keep, s_cols), od_profile(t_keep, t_cols)), diagnostics


def load_pded(path: Path, manifest: dict) -> np.ndarray:
    values, rows, cols = read_matrix(path)
    expected_r = [c["label"] for c in manifest["tables"]["source"]["columns"]]
    expected_c = [c["label"] for c in manifest["tables"]["target"]["columns"]]
    clean = lambda xs: [x.split(".", 1)[-1] for x in xs]
    if clean(rows) != expected_r or clean(cols) != expected_c:
        raise ValueError(f"PDED matrix label/orientation mismatch: {path}")
    return values


def derive(args, rows: list[dict]) -> None:
    for index, row in enumerate(rows, 1):
        manifest = validate_case(row)
        folder = Path(row["path"])
        dest = args.out / "cases" / row["key"]
        lex = matrix(dest / "levenshtein.csv", manifest)
        pmf = matrix(dest / "pmf.csv", manifest)
        sf_fd = matrix(dest / "sf_fd.csv", manifest)
        seed = matrix(dest / "seed.csv", manifest)
        raw = [strict_columns(folder / manifest["tables"][side]["path"], manifest["tables"][side])
               for side in ("source", "target")]
        rel = [relation_matrices(side)[0] for side in raw]
        gcov = role_scores(rel[0]["gcov"], rel[1]["gcov"])
        uncertainty = cosine_similarity(
            role_profile(rel[0]["uncertainty"], len(raw[0])),
            role_profile(rel[1]["uncertainty"], len(raw[1])),
        )
        jaccard = role_scores(rel[0]["jaccard"], rel[1]["jaccard"])
        pmf_gcov_u = scale_aligned_fuse(pmf, equal_role_block(gcov, uncertainty), WEIGHT)
        sf_jaccard = scale_aligned_fuse(sf_fd, jaccard, WEIGHT)

        ev = scenario_evidence(folder, dest / "seed.csv", None, SEED,
                               use_ground_truth=False)
        if ev is None:
            raise ValueError(f"Missing FD evidence for frozen arm: {row['id']}")
        fd = fd_channel(ev, FD_BUDGET)
        dc = dc_channel(ev, DC_BUDGET)
        channels = [ev["kn"], fd] + ([dc] if dc is not None else [])
        deployable = fuse(seed, channels, WEIGHT)

        s_names, s_raw = read_columns(folder / manifest["tables"]["source"]["path"])
        t_names, t_raw = read_columns(folder / manifest["tables"]["target"]["path"])
        s_cols = [bare(c) for c in expected_labels(manifest, "source")]
        t_cols = [bare(c) for c in expected_labels(manifest, "target")]
        s_values, _ = load_side(folder, "source")
        t_values, _ = load_side(folder, "target")
        anchors = anchor_matrix(s_values, t_values, s_cols, t_cols)
        od, od_diag = od_channel(s_raw, t_raw, s_cols, t_cols, anchors)
        deployable_od = fuse(seed, channels + ([od] if od is not None else []), WEIGHT)

        matrices = {
            "sf_fd_pmf": sf_fd,
            "sf_fd_pmf_jaccard": sf_jaccard,
            "pmf_gcov_uncertainty": pmf_gcov_u,
            "seed_kn_fd_anchor_dc_anchor": deployable,
            "seed_kn_fd_anchor_dc_anchor_od_anchor": deployable_od,
        }
        for name, values in matrices.items():
            write_matrix(dest / f"{name}.csv", values,
                         expected_labels(manifest, "source"), expected_labels(manifest, "target"))
        write_json(dest / "derived.json", {
            "id": row["id"], "od": od_diag, "dc_available": dc is not None,
            "matrix_sha256": {name: sha(dest / f"{name}.csv") for name in matrices},
        })
        print(f"features {index:02d}/16 complete {row['scenario']}", flush=True)


def prisma_source(args, row: dict) -> Path:
    local = args.out / "cases" / row["key"] / "prisma.csv"
    if local.exists():
        return local
    if args.prisma_root:
        directory = args.prisma_root / row["group"] / row["scenario"] / "_outputs" / "MatchingStepLine1"
        hits = [p for p in directory.glob("*.csv") if "postprocessing=false" in p.name]
        if len(hits) == 1:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hits[0], local)
            write_json(local.with_suffix(".provenance.json"), {
                "source": str(hits[0].resolve()),
                "source_sha256": sha(hits[0]),
                "materialized_sha256": sha(local),
            })
            return local
    raise FileNotFoundError(f"Missing PRISMA matrix for {row['id']}")


def report(args, rows: list[dict]) -> None:
    metrics = []
    label_checks = []
    od_coverage = []
    for row in rows:
        manifest = validate_case(row)
        folder = Path(row["path"])
        gt, _, _ = read_matrix(folder / manifest["ground_truth"]["path"])
        truth = gt > 0.5
        dest = args.out / "cases" / row["key"]
        derived_meta = json.loads((dest / "derived.json").read_text(encoding="utf-8"))
        od_coverage.append({**{k: row[k] for k in ("id", "dataset", "scenario_type", "evidence_regime")},
                            **derived_meta["od"],
                            "available_both_sides": bool(derived_meta["od"]["source_ods"]
                                                         and derived_meta["od"]["target_ods"])})
        source_names = [c["label"] for c in manifest["tables"]["source"]["columns"]]
        target_names = [c["label"] for c in manifest["tables"]["target"]["columns"]]
        gt_pairs = np.argwhere(truth)
        suffix_hits = sum(source_names[i].split("_", 1)[-1] == target_names[j].split("_", 1)[-1]
                          for i, j in gt_pairs)
        label_checks.append({
            **{k: row[k] for k in ("id", "dataset", "scenario_type", "evidence_regime")},
            "ground_truth_pairs": len(gt_pairs),
            "gt_pairs_with_equal_opaque_suffix": suffix_hits,
            "fraction": suffix_hits / len(gt_pairs) if len(gt_pairs) else 0.0,
        })
        sources = {
            "levenshtein": dest / "levenshtein.csv",
            "pmf": dest / "pmf.csv",
            "prisma": prisma_source(args, row),
            "pded": args.out / "pded" / row["group"] / row["scenario"] / "pded.csv",
            "sf_fd_pmf": dest / "sf_fd_pmf.csv",
            "sf_fd_pmf_jaccard": dest / "sf_fd_pmf_jaccard.csv",
            "pmf_gcov_uncertainty": dest / "pmf_gcov_uncertainty.csv",
            "seed_kn_fd_anchor_dc_anchor": dest / "seed_kn_fd_anchor_dc_anchor.csv",
            "seed_kn_fd_anchor_dc_anchor_od_anchor": dest / "seed_kn_fd_anchor_dc_anchor_od_anchor.csv",
        }
        for arm, path in sources.items():
            scores = load_pded(path, manifest) if arm == "pded" else matrix(path, manifest)
            for result in evaluate(ScoreMatrix(scores=scores, ground_truth=truth),
                                   policies=("hungarian", "mutual_top1")):
                metrics.append({
                    **{k: row[k] for k in ("id", "dataset", "scenario_type", "evidence_regime")},
                    "arm": arm, "extractor": result["extractor"], "f1": result["f1"],
                    "recall_at_gt": result["recall_at_gt_pessimistic"],
                    "mrr": result["mrr_pessimistic"], "predicted": result["n_match"],
                    "ground_truth": result["n_ground_truth"],
                })
    write_csv(args.out / "metrics.csv", metrics)
    summaries = []
    strata = (("overall", ()), ("dataset", ("dataset",)),
              ("scenario_type", ("scenario_type",)),
              ("regime", ("evidence_regime",)),
              ("dataset_regime", ("dataset", "evidence_regime")))
    for stratum, keys in strata:
        grouped = defaultdict(list)
        for rec in metrics:
            grouped[tuple(rec[k] for k in keys) + (rec["arm"], rec["extractor"])].append(rec)
        for key, group in sorted(grouped.items()):
            dimensions = key[:len(keys)]
            summaries.append({
                "stratum": stratum,
                "value": "/".join(dimensions) if dimensions else "all",
                "arm": key[-2], "extractor": key[-1], "cases": len(group),
                "f1": mean(x["f1"] for x in group),
                "recall_at_gt": mean(x["recall_at_gt"] for x in group),
                "mrr": mean(x["mrr"] for x in group),
            })
    write_csv(args.out / "summary.csv", summaries)
    write_csv(args.out / "od_coverage.csv", od_coverage)
    write_csv(args.out / "label_signal_audit.csv", label_checks)

    paired = []
    by_pair = defaultdict(dict)
    for rec in metrics:
        key = (rec["dataset"], rec["scenario_type"], rec["arm"], rec["extractor"])
        regime = "plain" if rec["evidence_regime"] == "plain" else "structural"
        by_pair[key][regime] = rec
    for key, pair in sorted(by_pair.items()):
        if set(pair) != {"plain", "structural"}:
            continue
        paired.append({
            "dataset": key[0], "scenario_type": key[1], "arm": key[2], "extractor": key[3],
            "plain_f1": pair["plain"]["f1"], "structural_f1": pair["structural"]["f1"],
            "f1_delta": pair["structural"]["f1"] - pair["plain"]["f1"],
            "plain_recall_at_gt": pair["plain"]["recall_at_gt"],
            "structural_recall_at_gt": pair["structural"]["recall_at_gt"],
            "recall_at_gt_delta": pair["structural"]["recall_at_gt"] - pair["plain"]["recall_at_gt"],
            "plain_mrr": pair["plain"]["mrr"], "structural_mrr": pair["structural"]["mrr"],
            "mrr_delta": pair["structural"]["mrr"] - pair["plain"]["mrr"],
        })
    write_csv(args.out / "paired_regime_deltas.csv", paired)
    output_files = [args.out / name for name in ("metrics.csv", "summary.csv", "od_coverage.csv",
                                                  "label_signal_audit.csv", "paired_regime_deltas.csv")]
    write_json(args.out / "results_manifest.json", {
        "status": "complete", "cases": len(rows), "arms": len(ARMS),
        "metric_rows": len(metrics), "summary_rows": len(summaries),
        "all_profile_integrity_rechecked": (args.out / "profile_integrity_postrun.json").exists(),
        "outputs_sha256": {p.name: sha(p) for p in output_files},
    })
    print(f"wrote {len(metrics)} metric rows and {len(summaries)} summaries", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare", "sf", "pded", "derive", "report", "all"))
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--prisma-root", type=Path)
    ap.add_argument("--java", default="java")
    ap.add_argument("--javac", default="javac")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    # Java matchers run from their own project directories.  Canonicalise all
    # caller-supplied paths before changing cwd so relative CLI overrides stay
    # anchored to the harness invocation directory.
    args.data = args.data.resolve()
    args.out = args.out.resolve()
    args.protocol = args.protocol.resolve()
    if args.prisma_root is not None:
        args.prisma_root = args.prisma_root.resolve()
    rows = prepare(args) if args.command == "prepare" else load_runtime(args)
    if args.command in ("sf", "all"):
        run_sf(args, rows)
    if args.command in ("pded", "all"):
        run_pded(args, rows)
    if args.command in ("derive", "all"):
        derive(args, rows)
    if args.command in ("report", "all"):
        report(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
