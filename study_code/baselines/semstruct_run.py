"""SemStruct-reimpl against this project's cohort, under the common extractors.

The algorithm lives in `code/semstruct`; nothing here chooses a parameter, a
variant, a row cap, an extractor or a threshold.  Every such choice is frozen
in `SEMSTRUCT_REIMPLEMENTATION.md`, written before any evaluation scenario was
scored, and mirrored in `semstruct/config.py` with a provenance marker.

Subcommands, in the order they are meant to be run:

    cost      measure build and encode time at several row caps; this is what
              the S4 cost rule reads, and it must be run before `train`
    train     fit the structural encoder on synthetic views of *source* tables
              only, and write a checkpoint plus its manifest
    score     write one score matrix per (case, regime, arm)
    report    score every matrix through smx's extractors and metrics

The arms, all on the same matrices pipeline (S5/S6):

    semstruct     the reconstruction
    lm_only       ablation and validation check 1: the same graphs, the same
                  serialisation, the GNN and MLP skipped entirely
    permuted      validation check 2: cells permuted independently within each
                  column, so column marginals survive and row co-occurrence
                  does not.  The structural gain must shrink here
    zero_delta    validation check 3: MLP(.) forced to zero.  Must reproduce
                  `lm_only` bit for bit; if it does not, something outside the
                  residual is moving the scores

Runs in the embedding virtual environment:

    .venv-embed/Scripts/python.exe semstruct_run.py cost
    .venv-embed/Scripts/python.exe semstruct_run.py train
    .venv-embed/Scripts/python.exe semstruct_run.py score --regime plain
    .venv-embed/Scripts/python.exe semstruct_run.py report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code" / "semstruct"))

from matrix_health import health  # noqa: E402
from smx.core import ScoreMatrix, evaluate  # noqa: E402

from semstruct import (AMENDMENT2_DEFAULTS, AUTHORS, DEFAULT, Encoder, GraphStore, SemStructConfig,  # noqa: E402
                       build_groups, build_model, native_matrix,
                       permute_within_columns, read_table, score_matrix, train)
from semstruct.tables import table_digest  # noqa: E402

# SEMSTRUCT_RUN: a folder with a protocol.json whose case paths point elsewhere (the GPU server's copy, Amendment 2)
RUN = Path(os.environ.get("SEMSTRUCT_RUN", HERE / "reports" / "audited_md5_fused_20260904"))
OUT = HERE / "reports" / "semstruct_20260908"
CACHE = HERE / "profile-cache" / "semstruct"

REGIMES = ("plain", "audited_md5")
ARMS = ("semstruct", "lm_only", "permuted", "zero_delta")
# S4.2: what tau = 0.9 is a threshold *on*.  `raw` is the literal reading and
# the primary arm; the other two are declared sensitivity arms, fixed before
# any scenario was scored.
MERGES = {"raw": ("raw", 0.9), "centered": ("centered", 0.9), "none": ("raw", 1.01)}
POLICIES = ("hungarian", "mutual_top1")
EXPLORATION = "inspected44"
CONFIRMATION = "remaining338"
PERMUTATION_SEED = 20260908


# ---------------------------------------------------------------- corpus I/O

CASES_ROOT: Optional[Path] = None      # SEMSTRUCT_ABLATION.md: an imported corpus instead of Valentine's protocol


def load_cases(limit: int = 0) -> List[dict]:
    if CASES_ROOT is not None:
        # <root>/<dataset>/<scenario>/{source,target,ground_truth}; one readable regime only. The key is the
        # scenario path's hash, the family the dataset folder, the cohort a single "all".
        cases = []
        for manifest in sorted(CASES_ROOT.glob("*/*/manifest.json")):
            folder = manifest.parent
            rel = folder.relative_to(CASES_ROOT).as_posix()
            cases.append({"id": rel, "key": hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16],
                          "cohort": "all", "family": folder.parent.name, "dataset": folder.parent.name,
                          "scenario": folder.name, "plain": str(folder), "audited_md5": str(folder)})
        return cases[:limit] if limit else cases
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    cases = protocol["cases"]
    return cases[:limit] if limit else cases


def case_paths(case: dict, regime: str) -> Optional[Tuple[Path, Path, Path]]:
    folder = Path(case[regime])
    ground_truth = sorted((Path(case["plain"]) / "ground_truth").glob("*.csv"))
    source = sorted((folder / "source").glob("*.csv"))
    target = sorted((folder / "target").glob("*.csv"))
    if not (ground_truth and source and target):
        return None
    return source[0], target[0], ground_truth[0]


def read_ground_truth(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if row and any(cell.strip() for cell in row):
                rows.append([float(cell) if cell.strip() else 0.0 for cell in row])
    return np.array(rows) > 0.5


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in matrix:
            writer.writerow([f"{value:.10g}" for value in row])
    temporary.replace(path)


def read_written_matrix(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.array([[float(c) for c in row]
                         for row in csv.reader(handle) if row])


# ------------------------------------------------------------------ helpers

def make_config(args) -> SemStructConfig:
    # SEMSTRUCT_ABLATION.md Amendment 2: --authors starts from the revised training setup
    cfg = AUTHORS if getattr(args, "authors", False) else DEFAULT
    if getattr(args, "plm", None):
        cfg = replace(cfg, plm=args.plm)
    if getattr(args, "views", 0):
        cfg = replace(cfg, views_per_table=args.views)
    if getattr(args, "seed", 0):
        cfg = replace(cfg, seed=args.seed)               # Amendment 2 follow-up: training seeds
    if getattr(args, "valavg", False):
        cfg = replace(cfg, valavg=True, row_value_edges=False)
    if getattr(args, "mlp_only", False):
        cfg = replace(cfg, mlp_only=True)
    if getattr(args, "row_cap", 0):
        cfg = replace(cfg, row_cap=args.row_cap)
    if getattr(args, "epochs", 0):
        cfg = replace(cfg, epochs=args.epochs)
    merge = getattr(args, "merge", "raw")
    similarity, threshold = MERGES[merge]
    cfg = replace(cfg, merge_similarity=similarity, merge_threshold=threshold)
    # SEMSTRUCT_ABLATION.md (2026-10-27): repair R1 and the one-hop information-matched arm
    if getattr(args, "delta_init", None):
        cfg = replace(cfg, delta_init=args.delta_init)
    if getattr(args, "onehop", False):
        cfg = replace(cfg, n_layers=1, row_value_edges=False)
    if getattr(args, "center", False):
        cfg = replace(cfg, center_features=True)
    if getattr(args, "mining", None):
        cfg = replace(cfg, mining=args.mining)
    return cfg


def suffix(merge: str) -> str:
    """A non-primary merge arm never shares a directory or a tag with the
    primary one -- a checkpoint trained under one graph construction is not
    valid under another, and `load_model` refuses the mismatch anyway."""
    return "" if merge == "raw" else f"__merge-{merge}"


DEVICE = "cpu"      # --device; the encoder only (the structural model is small and stays on the CPU)


def make_encoder(cfg: SemStructConfig, threads: Optional[int]) -> Encoder:
    return Encoder(cfg, cache_dir=CACHE / "text", threads=threads, device=DEVICE)


def eval_suffix(args) -> str:
    """Amendment 2: matrices scored at an evaluation row cap other than the training cap get their own folder."""
    cap = getattr(args, "eval_row_cap", 0)
    return f"__eval{cap}" if cap else ""


def config_fingerprint(cfg: SemStructConfig) -> str:
    fields = cfg.to_dict()
    # fields added for SEMSTRUCT_ABLATION.md enter the fingerprint only when changed, so the
    # 2026-09-08 checkpoints keep their recorded fingerprints
    for name, default in (("delta_init", "default"), ("row_value_edges", True), ("center_features", False)):
        if fields.get(name) == default:
            fields.pop(name, None)
    for name, default in AMENDMENT2_DEFAULTS.items():             # Amendment 2, same rule
        if fields.get(name) == default:
            fields.pop(name, None)
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


# --------------------------------------------------------------------- cost

def command_cost(args) -> int:
    """S4: measure, then let the declared rule pick the cap.

    Reports per-table graph size and wall time at each candidate cap, on a
    sample that spans every family and scenario type in the cohort plus the
    widest scenario, in both regimes -- the recoded regime is the expensive one
    because hashing removes every repeated value.
    """
    cases = load_cases()
    by_class: Dict[Tuple[str, str], dict] = {}
    for case in cases:
        by_class.setdefault((case["family"], case["scenario_type"]), case)
    sample = list(by_class.values())
    print(f"cost pilot over {len(sample)} (family, type) classes x "
          f"{len(REGIMES)} regimes x {len(args.caps)} caps")

    rows = []
    for cap in args.caps:
        cfg = replace(DEFAULT, row_cap=cap)
        encoder = make_encoder(cfg, args.threads)
        store = GraphStore(cfg, encoder, cache_dir=None)   # measure cold cost
        for regime in REGIMES:
            for case in sample:
                paths = case_paths(case, regime)
                if not paths:
                    continue
                source_path, target_path, _ = paths
                started = time.perf_counter()
                sizes = []
                for path in (source_path, target_path):
                    headers, columns = read_table(path, cap)
                    if not headers:
                        continue
                    graph = store.get(headers, columns)
                    sizes.append((graph.n_cols, graph.n_rows, graph.n_values,
                                  graph.edge_index.shape[1]))
                elapsed = time.perf_counter() - started
                encoder.clear_memo()
                if not sizes:
                    continue
                rows.append({
                    "cap": cap, "regime": regime, "family": case["family"],
                    "scenario_type": case["scenario_type"], "seconds": elapsed,
                    "n_cols": sum(s[0] for s in sizes),
                    "n_values": sum(s[2] for s in sizes),
                    "n_edges": sum(s[3] for s in sizes)})
                print(f"  cap {cap:>5}  {regime:<11} {case['family']:<9} "
                      f"{case['scenario_type']:<22} {elapsed:6.1f}s  "
                      f"values {rows[-1]['n_values']:>6}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cost.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n_cases = len(load_cases())
    print(f"\n{'cap':>6}{'mean s/pair':>13}{'median':>9}{'slowest':>9}"
          f"{'projected hours (382 x 2 regimes)':>36}")
    projection = {}
    for cap in args.caps:
        selected = [r["seconds"] for r in rows if r["cap"] == cap]
        if not selected:
            continue
        hours = mean(selected) * n_cases * len(REGIMES) / 3600.0
        projection[cap] = hours
        print(f"{cap:>6}{mean(selected):>13.1f}{median(selected):>9.1f}"
              f"{max(selected):>9.1f}{hours:>36.1f}")
    json_write(OUT / "cost.json", {"rows": rows, "projected_hours": projection,
                                   "n_cases": n_cases, "regimes": list(REGIMES)})
    print(f"\nwrote {OUT / 'cost.csv'} and cost.json")
    return 0


# -------------------------------------------------------------------- train

def training_tables(cases: Sequence[dict], cfg: SemStructConfig,
                    hold_out_family: Optional[str], limit: int,
                    regime: str = "plain"
                    ) -> List[Tuple[str, List[str], List[List[str]]]]:
    """Distinct **source** tables only (PAPER S3.4: targets are never seen).

    De-duplicated by content, because the corpus reuses one source table across
    many scenarios and training on it 30 times would silently reweight the
    objective toward whichever family fabricates the most scenarios.
    """
    seen = set()
    tables = []
    for case in cases:
        if hold_out_family and case["family"] == hold_out_family:
            continue
        paths = case_paths(case, regime)
        if not paths:
            continue
        headers, columns = read_table(paths[0], cfg.row_cap)
        if not headers or not columns or not columns[0]:
            continue
        digest = table_digest(headers, columns)
        if digest in seen:
            continue
        seen.add(digest)
        tables.append((digest, headers, columns))
        if limit and len(tables) >= limit:
            break
    return tables


def command_train(args) -> int:
    import torch
    cfg = make_config(args)
    cases = load_cases(args.limit_cases)
    tables = training_tables(cases, cfg, args.hold_out_family, args.limit_tables,
                             args.train_regime)
    if not tables:
        print("no training tables found")
        return 1

    rng = np.random.default_rng(cfg.seed)
    groups = build_groups(tables, cfg, rng)
    order = rng.permutation(len(groups))
    n_validation = max(1, int(round(len(groups) * args.validation_frac)))
    validation = [groups[i] for i in order[:n_validation]]
    training = [groups[i] for i in order[n_validation:]]
    print(f"{len(training)} training tables, {len(validation)} held back for "
          f"early stopping, {cfg.views_per_table} views each")

    encoder = make_encoder(cfg, args.threads)
    store = GraphStore(cfg, encoder, cache_dir=CACHE / "graphs")
    started = time.perf_counter()
    model, history = train(training, store, cfg, validation=validation,
                           log=lambda line: print("  " + line, flush=True))
    elapsed = time.perf_counter() - started

    tag = (args.tag or (f"holdout-{args.hold_out_family}"
                        if args.hold_out_family else "full"))
    if args.train_regime != "plain":
        tag += f"__train-{args.train_regime}"
    tag += suffix(args.merge)
    checkpoint = OUT / "checkpoints" / f"{tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    json_write(checkpoint.with_suffix(".json"), {
        "tag": tag,
        "config": cfg.to_dict(),
        "config_sha256_16": config_fingerprint(cfg),
        "hold_out_family": args.hold_out_family,
        "merge_arm": args.merge,
        "train_regime": args.train_regime,
        "n_training_tables": len(training),
        "n_validation_tables": len(validation),
        "training_table_digests": [g.table_id for g in training],
        "history": history,
        "seconds": elapsed,
        "graph_cache": {"hits": store.hits, "misses": store.misses},
        "torch": torch.__version__,
    })
    print(f"\ntrained in {elapsed / 60:.1f} min -> {checkpoint}")
    return 0


# -------------------------------------------------------------------- score

def load_model(cfg: SemStructConfig, tag: str):
    import torch
    checkpoint = OUT / "checkpoints" / f"{tag}.pt"
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    manifest = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest["config_sha256_16"] != config_fingerprint(cfg):
        raise SystemExit(
            f"checkpoint {tag} was trained under a different configuration "
            f"({manifest['config_sha256_16']} != {config_fingerprint(cfg)}); "
            "retrain or pass the configuration it was trained with")
    return model, manifest


def command_score(args) -> int:
    cfg = make_config(args)
    tag = args.tag + suffix(args.merge)
    model, manifest = load_model(cfg, tag)
    zero_model, _ = load_model(cfg, tag)
    zero_model.zero_structural_delta()

    encoder = make_encoder(cfg, args.threads)
    store = GraphStore(cfg, encoder, cache_dir=CACHE / "graphs")
    cases = load_cases(args.limit)
    arms = args.arms or list(ARMS)

    written, failures = 0, []
    for index, case in enumerate(cases, 1):
        paths = case_paths(case, args.regime)
        if not paths:
            failures.append((case["id"], "missing files"))
            continue
        source_path, target_path, ground_truth_path = paths
        try:
            ground_truth = read_ground_truth(ground_truth_path)
            # Amendment 2: --eval-row-cap reads the evaluation tables at another cap than training used, without
            # changing the configuration (and so the checkpoint fingerprint)
            read_cap = args.eval_row_cap or cfg.row_cap
            source_headers, source_columns = read_table(source_path, read_cap)
            target_headers, target_columns = read_table(target_path, read_cap)
            if ground_truth.shape != (len(source_headers), len(target_headers)):
                failures.append((case["id"], "shape mismatch"))
                continue

            for arm in arms:
                destination = (OUT / "matrices" / args.regime /
                               (arm + suffix(args.merge) + eval_suffix(args)) / f"{case['key']}.csv")
                if destination.exists() and not args.force:
                    continue
                if arm == "permuted":
                    left = permute_within_columns(source_columns, PERMUTATION_SEED)
                    right = permute_within_columns(target_columns, PERMUTATION_SEED + 1)
                else:
                    left, right = source_columns, target_columns
                source_graph = store.get(source_headers, left)
                target_graph = store.get(target_headers, right)
                active = zero_model if arm == "zero_delta" else model
                structural = arm != "lm_only"
                S = score_matrix(active, source_graph, target_graph,
                                 structural=structural)
                write_matrix(destination, S)
                if args.native:
                    write_matrix(OUT / "matrices" / args.regime /
                                 (arm + suffix(args.merge) + eval_suffix(args) + "_native") /
                                 f"{case['key']}.csv", native_matrix(S))
                written += 1
        except Exception as exception:            # recorded, never swallowed
            failures.append((case["id"], f"{type(exception).__name__}: {exception}"))
            continue
        if index % 20 == 0:
            print(f"  {index}/{len(cases)}  {written} matrices, "
                  f"{len(failures)} failures", flush=True)

    json_write(OUT / f"score_manifest_{args.regime}_{tag}{eval_suffix(args)}.json", {
        "regime": args.regime, "tag": tag, "arms": arms,
        "merge_arm": args.merge,
        "row_cap_training": cfg.row_cap, "row_cap_evaluation": args.eval_row_cap or cfg.row_cap,
        "device": DEVICE,
        # The training regime is a property of the checkpoint, not of this
        # command: scoring the recoded regime with a plaintext-trained model is
        # the primary arm, so the two must be recorded separately.
        "train_regime": manifest.get("train_regime"),
        "config_sha256_16": config_fingerprint(cfg),
        "checkpoint": manifest.get("tag"),
        "n_matrices": written, "failures": failures,
        "permutation_seed": PERMUTATION_SEED,
    })
    print(f"\nwrote {written} matrices, {len(failures)} failures")
    for case_id, why in failures[:10]:
        print(f"   {case_id.split('/')[-1]}: {why[:100]}")
    return 0


# ------------------------------------------------------------------- report

def command_report(args) -> int:
    cases = {case["key"]: case for case in load_cases()}
    rows = []
    for regime in (args.regime,) if args.regime else REGIMES:
        root = OUT / "matrices" / regime
        wanted = tuple(args.arms) if args.arms else ARMS
        for folder in sorted(p for p in root.glob("*") if p.is_dir()):
            arm = folder.name
            if not arm.split("__")[0].replace("_native", "") in wanted:
                continue
            for path in sorted(folder.glob("*.csv")):
                case = cases.get(path.stem)
                if case is None:
                    continue
                paths = case_paths(case, "plain")
                if not paths:
                    continue
                ground_truth = read_ground_truth(paths[2])
                S = read_written_matrix(path)
                if S.shape != ground_truth.shape:
                    continue
                report = health(S, expect_shape=ground_truth.shape)
                sm = ScoreMatrix(scores=S, ground_truth=ground_truth)
                for record in evaluate(sm, policies=POLICIES):
                    rows.append({
                        "regime": regime, "arm": arm,
                        "policy": record["extractor"],
                        "key": path.stem, "family": case["family"],
                        "cohort": case["cohort"],
                        "scenario_type": case["scenario_type"],
                        "degenerate": bool(report["all_zero"] or report["constant"]),
                        **{k: v for k, v in record.items()
                           if isinstance(v, (int, float, bool))},
                    })
    if not rows:
        print("no matrices to report; run `score` first")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with (OUT / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Cohorts are kept apart: this project reports the 44 exploration and 338
    # confirmation scenarios separately, and single-correspondence scenarios
    # separately again (FINDINGS 89) -- at |GT| = 1 the true pair sits at (0,0)
    # in every one of them, so an uninformative matrix scores through
    # tie-breaking alone.  `report.csv` carries n_ground_truth for that split.
    print(f"{'regime':<12}{'arm':<22}{'policy':<14}{'cohort':<14}{'n':>5}"
          f"{'F1':>8}{'MRR':>8}{'R@GT':>8}{'|M|':>8}")
    for regime in sorted({r["regime"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows}):
            for policy in POLICIES:
                for cohort in (EXPLORATION, CONFIRMATION):
                    selected = [r for r in rows if r["regime"] == regime
                                and r["arm"] == arm and r["policy"] == policy
                                and r["cohort"] == cohort
                                and not r["degenerate"]]
                    if not selected:
                        continue
                    print(f"{regime:<12}{arm:<22}{policy:<14}{cohort:<14}"
                          f"{len(selected):>5}"
                          f"{mean(r.get('f1', 0.0) for r in selected):>8.4f}"
                          f"{mean(r.get('mrr_pessimistic', 0.0) for r in selected):>8.4f}"
                          f"{mean(r.get('recall_at_gt_pessimistic', 0.0) for r in selected):>8.4f}"
                          f"{mean(r.get('n_match', 0.0) for r in selected):>8.1f}")
    print(f"\nwrote {OUT / 'report.csv'} ({len(rows)} rows)")
    return 0


# -------------------------------------------------------------------- verify

def command_verify(args) -> int:
    """The mechanical half of the S6 validation ladder, over written matrices.

    Check 3 is exact and blocking: `zero_delta` forces MLP(.) = 0, so by Eq. 2
    it must reproduce the frozen semantic initialisation, which is what
    `lm_only` computes by a different route (the GNN is skipped entirely).  If
    those two matrices differ anywhere, something outside the residual is
    moving the scores and no other number here may be reported.

    Check 1 is reported, not asserted: `lm_only` is the frozen-E5 column
    baseline, so `semstruct` differing from it is the *minimum* evidence that
    the structural encoder does anything at all.
    """
    failures, compared, moved = [], 0, []
    for regime in (args.regime,) if args.regime else REGIMES:
        root = OUT / "matrices" / regime
        if not root.is_dir():
            continue
        for suffix_name in sorted({p.name.split("__", 1)[1] if "__" in p.name else ""
                                   for p in root.glob("*") if p.is_dir()}):
            tail = f"__{suffix_name}" if suffix_name else ""
            lm = root / f"lm_only{tail}"
            zero = root / f"zero_delta{tail}"
            full = root / f"semstruct{tail}"
            if not (lm.is_dir() and zero.is_dir()):
                continue
            for path in sorted(lm.glob("*.csv")):
                other = zero / path.name
                if not other.exists():
                    continue
                a, b = read_written_matrix(path), read_written_matrix(other)
                compared += 1
                if a.shape != b.shape or not np.array_equal(a, b):
                    failures.append(f"{regime}{tail}/{path.stem}")
                if (full / path.name).exists():
                    c = read_written_matrix(full / path.name)
                    if c.shape == a.shape:
                        moved.append(float(np.abs(c - a).max()))

    print(f"check 3 (zero structural delta): {compared} matrices compared, "
          f"{len(failures)} mismatches")
    for name in failures[:10]:
        print(f"   MISMATCH {name}")
    if moved:
        print(f"check 1 (structure moves the scores): max |semstruct - lm_only| "
              f"median {float(np.median(moved)):.4f}, "
              f"min {min(moved):.4f}, max {max(moved):.4f}")
        if max(moved) == 0.0:
            print("   WARNING: the structural encoder changed nothing anywhere")
    return 1 if failures else 0


# ---------------------------------------------------------------------- main

def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    # SEMSTRUCT_ABLATION.md: each ablation arm writes into its own folder (checkpoints, matrices, reports), so the
    # 2026-09-08 results are never overwritten; the graph cache is shared.
    parser.add_argument("--out", default=None, help="output folder (default reports/semstruct_20260908)")
    parser.add_argument("--delta-init", choices=("default", "zero"), default=None)
    parser.add_argument("--onehop", action="store_true", help="1 layer, no row-value edges")
    parser.add_argument("--center", action="store_true", help="repair R2: centre column/value features per graph")
    parser.add_argument("--mining", choices=("hardest", "semihard"), default=None, help="Amendment 1: semihard")
    parser.add_argument("--authors", action="store_true", help="Amendment 2: the revised training setup")
    parser.add_argument("--plm", default=None, help="override the checkpoint (timing test: the v1 stand-in)")
    parser.add_argument("--views", type=int, default=0, help="views per source table")
    parser.add_argument("--valavg", action="store_true", help="Amendment 2: the ValAvg arm")
    parser.add_argument("--mlp-only", action="store_true", help="the MLP rung of the paper's Fig. 5 ladder")
    parser.add_argument("--seed", type=int, default=0, help="training seed (default: the configuration's)")
    parser.add_argument("--device", default="cpu", help="encoder device, e.g. cuda (Amendment 2, GPU server)")
    parser.add_argument("--cases-root", default=None,
                        help="an imported corpus tree (e.g. generated/goby-imported) instead of Valentine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cost = subparsers.add_parser("cost", help="S4 cost pilot")
    cost.add_argument("--caps", type=int, nargs="+", default=[100, 200, 400])
    cost.add_argument("--threads", type=int, default=None)
    cost.set_defaults(func=command_cost)

    trainer = subparsers.add_parser("train", help="fit the structural encoder")
    trainer.add_argument("--row-cap", type=int, default=0)
    trainer.add_argument("--epochs", type=int, default=0)
    trainer.add_argument("--limit-cases", type=int, default=0)
    trainer.add_argument("--limit-tables", type=int, default=0)
    trainer.add_argument("--hold-out-family", default=None)
    trainer.add_argument("--validation-frac", type=float, default=0.1)
    trainer.add_argument("--threads", type=int, default=None)
    trainer.add_argument("--tag", default=None)
    trainer.add_argument("--merge", choices=sorted(MERGES), default="raw")
    trainer.add_argument("--train-regime", choices=REGIMES, default="plain",
                         help="which copy of the source tables to train on. The "
                              "paper trains once; the primary checkpoint is "
                              "trained on plaintext and applied to both regimes, "
                              "and a recoded-regime checkpoint exists so that "
                              "reading is available too")
    trainer.set_defaults(func=command_train)

    scorer = subparsers.add_parser("score", help="write score matrices")
    scorer.add_argument("--regime", choices=REGIMES, default="plain")
    scorer.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    scorer.add_argument("--row-cap", type=int, default=0)
    scorer.add_argument("--epochs", type=int, default=0)
    scorer.add_argument("--limit", type=int, default=0)
    scorer.add_argument("--tag", default="full")
    scorer.add_argument("--merge", choices=sorted(MERGES), default="raw")
    scorer.add_argument("--threads", type=int, default=None)
    scorer.add_argument("--native", action="store_true",
                        help="also write the paper's softmax + bipartite matrix")
    scorer.add_argument("--force", action="store_true")
    scorer.add_argument("--eval-row-cap", type=int, default=0,
                        help="Amendment 2: rows read per evaluation table (default: the training row cap)")
    scorer.set_defaults(func=command_score)

    reporter = subparsers.add_parser("report", help="metrics over written matrices")
    reporter.add_argument("--regime", choices=REGIMES, default=None)
    reporter.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    reporter.set_defaults(func=command_report)

    verifier = subparsers.add_parser("verify", help="S6 controls over written matrices")
    verifier.add_argument("--regime", choices=REGIMES, default=None)
    verifier.set_defaults(func=command_verify)

    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    if args.out:
        OUT = Path(args.out) if Path(args.out).is_absolute() else HERE / args.out
    if args.cases_root:
        global CASES_ROOT
        CASES_ROOT = Path(args.cases_root) if Path(args.cases_root).is_absolute() else HERE / args.cases_root
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
