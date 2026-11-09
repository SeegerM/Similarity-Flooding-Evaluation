"""Run SiMa end to end on a silo benchmark built by ``sima_silos.py``.

Follows the pipeline of the authors' ``notebooks/matching_data_silos.ipynb``
using the vendored code in ``code/SiMa/src`` unmodified: profile every column,
build one relatedness graph per silo from the intra-silo matches, sample
negatives with strategy NS3, train GraphSAGE + the MLP link predictor
incrementally, then score every inter-silo column pair with the predictor.

Two deviations from the notebook, both verified rather than assumed:

1. ``tools.run_multithread`` re-pickles both full embedding matrices *per
   column pair*, which is unusable under Windows' spawn start method. Inter-silo
   scoring is therefore batched on tensors instead. ``--verify-scoring``
   checks the batched scores against SiMa's own ``tools.prediction_score`` on a
   random sample and reports the maximum absolute deviation.
2. Current scipy returns NaN for the kurtosis/skewness of a constant column
   where the pinned scipy 1.5.x returned -3.0 and 0.0. Those two constants are
   put back, since a NaN would spread across a whole feature row in
   ``F.normalize`` and silently destroy training. Counts are reported per
   feature kind so the substitution is never silent.

Beyond the score matrix over all inter-silo pairs, per-scenario matrices are
exported in the ``smx`` interchange format, so SiMa's output can be evaluated
by the same extractors and metrics as every other system in the harness.

Usage
-----
    python harness/sima_run.py harness/generated/sima-silos/dev-tpcdi \
        --out harness/reports/sima_dev --epochs 10 --embed 256
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIMA_SRC = REPO / "code" / "SiMa" / "src"
if str(SIMA_SRC) not in sys.path:
    sys.path.insert(0, str(SIMA_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# SiMa, vendored and unmodified apart from two compatibility fixes.
from tools import (  # noqa: E402
    generate_ids_paths,
    columns_to_profiles,
    create_profiles_tensor,
    prediction_score,
)
from graph import create_relatedness_graphs_from_silos  # noqa: E402
from train import negative_sampling_3, train_model, compute_probabilities_labels  # noqa: E402


# ---------------------------------------------------------------- utilities

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_ground_truth(path: Path) -> set:
    """Read SiMa's matches CSV into the set of unordered (table, column) pairs.

    Table names carry a ``.csv`` suffix in the file and are stripped exactly as
    the authors' notebook does.
    """
    df = pd.read_csv(path)
    gt = set()
    for _, row in df.iterrows():
        gt.add(((row["Dataset-1"][:-4], row["Column-1"]),
                (row["Dataset-2"][:-4], row["Column-2"])))
    return gt


def load_silos(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): set(v) for k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))}


# ---------------------------------------------------------------- profiling

def profile_columns(files_to_paths: dict, cache: Path | None) -> tuple[dict, dict]:
    if cache is not None and cache.exists():
        with open(cache, "rb") as fh:
            profiles = pickle.load(fh)
        tables = {k[0] for k in profiles}
        if tables == set(files_to_paths):
            return profiles, {"source": "cache", "path": str(cache)}
        print(f"profile cache {cache} does not cover this benchmark; recomputing", file=sys.stderr)

    t0 = time.perf_counter()
    profiles = columns_to_profiles(files_to_paths)
    elapsed = time.perf_counter() - t0
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump(profiles, fh)
    return profiles, {"source": "computed", "seconds": round(elapsed, 3),
                      "path": str(cache) if cache else None}


def feature_names(width: int) -> list[str]:
    """Recover the ordered feature names produced by ``tools.get_features``.

    ``columns_to_profiles`` returns bare lists, so the names have to be
    reconstructed from the profiler itself rather than hard-coded.
    """
    from tools import get_features

    probe = pd.DataFrame({"a": ["x1", "y2"], "b": ["1", "2"]})
    names = list(get_features(probe).columns)
    if len(names) != width:
        raise SystemExit(f"profiler returned {width} features but the probe yielded {len(names)}")
    return names


def sanitize(tensor: torch.Tensor, names: list[str]) -> tuple[torch.Tensor, dict]:
    """Restore the constant-column semantics SiMa's pinned scipy 1.5.x had.

    On a zero-variance column scipy <=1.8 returned ``0`` from the moment ratio,
    so ``kurtosis`` came back as ``-3.0`` (Fisher) and ``skew`` as ``0.0``.
    Current scipy returns NaN instead. Feeding NaN into ``F.normalize`` would
    make a whole feature row NaN and silently destroy training, so the original
    constants are put back rather than zeroing the feature.
    """
    replacement = torch.zeros(len(names), dtype=tensor.dtype)
    kinds = {"kurtosis": 0, "skewness": 0, "other": 0}
    for i, name in enumerate(names):
        if name.endswith("-agg-kurtosis"):
            replacement[i] = -3.0
        elif name.endswith("-agg-skewness"):
            replacement[i] = 0.0

    bad = ~torch.isfinite(tensor)
    n_bad = int(bad.sum())
    if n_bad:
        cols = bad.any(dim=0).nonzero().flatten().tolist()
        for i in cols:
            n = int(bad[:, i].sum())
            if names[i].endswith("-agg-kurtosis"):
                kinds["kurtosis"] += n
            elif names[i].endswith("-agg-skewness"):
                kinds["skewness"] += n
            else:
                kinds["other"] += n
        tensor = torch.where(bad, replacement.expand_as(tensor), tensor)
    if not torch.isfinite(tensor).all():
        raise SystemExit("feature tensor still contains non-finite values after substitution")
    return tensor, {"non_finite_features": n_bad,
                    "non_finite_by_kind": kinds,
                    "non_finite_columns": int(bad.any(dim=1).sum()),
                    "total_features": int(tensor.numel())}


# ---------------------------------------------------------------- scoring

PAIR_BUDGET_BYTES = 256 << 20


@torch.no_grad()
def batched_scores(pred, h_a: torch.Tensor, h_b: torch.Tensor, chunk: int | None = None) -> np.ndarray:
    """sigmoid(W2(relu(W1([h_a ; h_b])))) for every (a, b) pair, as SiMa's predictor.

    Returns an ``len(h_a) x len(h_b)`` matrix. The row block is sized so the
    materialized pair tensor stays inside PAIR_BUDGET_BYTES, since a silo pair
    can be hundreds of thousands of column pairs wide.
    """
    if chunk is None:
        per_row = max(1, h_b.shape[0] * 2 * h_a.shape[1] * 4)
        chunk = max(1, min(h_a.shape[0], PAIR_BUDGET_BYTES // per_row))
    out = torch.empty((h_a.shape[0], h_b.shape[0]), dtype=torch.float32)
    for start in range(0, h_a.shape[0], chunk):
        block = h_a[start:start + chunk]
        left = block.unsqueeze(1).expand(-1, h_b.shape[0], -1)
        right = h_b.unsqueeze(0).expand(block.shape[0], -1, -1)
        pair = torch.cat([left, right], dim=2)
        out[start:start + chunk] = torch.sigmoid(pred.W2(F.relu(pred.W1(pair)))).squeeze(-1)
    return out.numpy()


def verify_scoring(pred, embeddings, all_columns, all_cols_ids, samples: int, rng) -> dict:
    """Compare the batched scores against SiMa's own per-pair prediction_score."""
    silos = [i for i in embeddings if len(all_columns[i])]
    worst = 0.0
    checked = 0
    for _ in range(samples):
        a, b = rng.sample(silos, 2)
        ca = rng.choice(all_columns[a])
        cb = rng.choice(all_columns[b])
        reference = prediction_score(embeddings[a], embeddings[b], ca, cb,
                                     all_cols_ids[a], all_cols_ids[b], pred)
        ia, ib = all_cols_ids[a][ca], all_cols_ids[b][cb]
        mine = float(batched_scores(pred, embeddings[a][ia:ia + 1], embeddings[b][ib:ib + 1])[0, 0])
        worst = max(worst, abs(reference - mine))
        checked += 1
    return {"pairs_checked": checked, "max_abs_deviation": worst}


# ---------------------------------------------------------------- metrics

PR_BINS = 1 << 20


class PrHistogram:
    """Streaming precision-recall accumulator for score/label pairs.

    At full corpus scale there are ~2.6e8 inter-silo pairs, and materialising
    them to hand to ``sklearn.precision_recall_curve`` costs several GB and a
    sort of the same length. Scores here are sigmoid outputs in (0, 1), so
    binning them at 2^-20 resolution -- finer than float32 itself carries in
    that range -- gives the same curve from O(bins) memory. Validated against
    the exact path in ``validate_pr_histogram``.
    """

    def __init__(self, bins: int = PR_BINS):
        self.bins = bins
        self.pos = np.zeros(bins + 1, dtype=np.int64)
        self.neg = np.zeros(bins + 1, dtype=np.int64)

    def add(self, labels: np.ndarray, scores: np.ndarray) -> None:
        idx = np.clip((np.asarray(scores, dtype=np.float64) * self.bins).astype(np.int64),
                      0, self.bins)
        lab = np.asarray(labels).astype(bool)
        self.pos += np.bincount(idx[lab], minlength=self.bins + 1)
        self.neg += np.bincount(idx[~lab], minlength=self.bins + 1)

    def summary(self) -> dict:
        # Sweep the threshold downward: cumulative counts from the top bin.
        tp = np.cumsum(self.pos[::-1])[::-1]
        fp = np.cumsum(self.neg[::-1])[::-1]
        total_pos = int(tp[0])
        n_pairs = total_pos + int(fp[0])
        if not total_pos or total_pos == n_pairs:
            return {"pairs": n_pairs, "positives": total_pos,
                     "skipped": "one class only; a PR curve is undefined"}

        keep = (tp + fp) > 0
        precision = tp[keep] / (tp[keep] + fp[keep])
        recall = tp[keep] / total_pos
        thresholds = np.nonzero(keep)[0] / self.bins

        denom = precision + recall
        f1 = np.zeros_like(precision)
        np.divide(2 * precision * recall, denom, out=f1, where=denom > 0)
        best = int(np.argmax(f1))

        # AP as sum over the recall steps, matching average_precision_score's
        # step-wise definition (recall increases as the threshold falls).
        order = np.argsort(thresholds)[::-1]
        r_sorted, p_sorted = recall[order], precision[order]
        ap = float(np.sum(np.diff(np.concatenate(([0.0], r_sorted))) * p_sorted))

        half = int(np.searchsorted(thresholds, 0.5))
        half = min(half, precision.size - 1)
        return {
            "pairs": n_pairs,
            "positives": total_pos,
            "positive_rate": total_pos / n_pairs,
            "average_precision": ap,
            "best_f1": {"f1": float(f1[best]), "precision": float(precision[best]),
                        "recall": float(recall[best]), "threshold": float(thresholds[best])},
            "at_threshold_0.5": {"threshold": 0.5, "precision": float(precision[half]),
                                  "recall": float(recall[half]), "f1": float(f1[half])},
            "estimator": f"binned histogram, {self.bins} bins",
        }


def validate_pr_histogram(labels: np.ndarray, scores: np.ndarray, bins: int = PR_BINS) -> dict:
    """Compare the binned estimator against the exact one on the same data."""
    exact = pr_summary(labels, scores)
    h = PrHistogram(bins)
    h.add(labels, scores)
    binned = h.summary()
    return {
        "pairs": int(labels.size),
        "bins": bins,
        "exact": {"average_precision": exact["average_precision"],
                   "best_f1": exact["best_f1"]["f1"]},
        "binned": {"average_precision": binned["average_precision"],
                    "best_f1": binned["best_f1"]["f1"]},
        "abs_error_average_precision": abs(exact["average_precision"] - binned["average_precision"]),
        "abs_error_best_f1": abs(exact["best_f1"]["f1"] - binned["best_f1"]["f1"]),
    }


def pr_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.zeros_like(precision)
    denom = precision + recall
    np.divide(2 * precision * recall, denom, out=f1, where=denom > 0)
    best = int(np.argmax(f1))
    at_half = None
    if thresholds.size:
        idx = int(np.searchsorted(thresholds, 0.5))
        idx = min(idx, precision.size - 1)
        at_half = {"threshold": 0.5,
                   "precision": float(precision[idx]),
                   "recall": float(recall[idx]),
                   "f1": float(f1[idx])}
    return {
        "pairs": int(labels.size),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, scores)),
        "best_f1": {
            "f1": float(f1[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
            "threshold": float(thresholds[best]) if best < thresholds.size else 1.0,
        },
        "at_threshold_0.5": at_half,
    }


# ---------------------------------------------------------------- export

def export_scenario_matrices(build: dict, embeddings, all_cols_ids, table_to_silo,
                             pred, benchmark_root: Path, out_dir: Path) -> dict:
    """Write one source-by-target score matrix per scenario, in smx format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_scenario = {}
    for t in build["tables"]:
        by_scenario.setdefault((t["family"], t["scenario"]), {})[t["side"]] = t["table"]

    written, skipped = [], []
    for (family, scenario), sides in sorted(by_scenario.items()):
        if set(sides) != {"source", "target"}:
            skipped.append({"family": family, "scenario": scenario, "reason": "missing side"})
            continue
        s_tab, t_tab = sides["source"], sides["target"]
        s_silo, t_silo = table_to_silo[s_tab], table_to_silo[t_tab]
        if s_silo == t_silo:
            skipped.append({"family": family, "scenario": scenario,
                            "reason": "both tables in the same silo: not an inter-silo prediction"})
            continue

        manifest_path = benchmark_root / family / scenario / "manifest.json"
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        s_labels = [c["label"] for c in manifest["tables"]["source"]["columns"]]
        t_labels = [c["label"] for c in manifest["tables"]["target"]["columns"]]

        try:
            s_idx = [all_cols_ids[s_silo][(s_tab, lab)] for lab in s_labels]
            t_idx = [all_cols_ids[t_silo][(t_tab, lab)] for lab in t_labels]
        except KeyError as exc:
            skipped.append({"family": family, "scenario": scenario,
                            "reason": f"column not found in silo index: {exc}"})
            continue

        matrix = batched_scores(pred, embeddings[s_silo][s_idx], embeddings[t_silo][t_idx])
        dest = out_dir / family / f"{scenario}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(dest, matrix, delimiter=",", fmt="%.6f")
        written.append({"family": family, "scenario": scenario,
                        "shape": list(matrix.shape), "path": str(dest.relative_to(out_dir))})

    return {"written": len(written), "skipped": skipped, "matrices": written}


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", type=Path, help="directory produced by sima_silos.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10, help="epochs per relatedness graph (paper: 10 x 10 silos)")
    ap.add_argument("--embed", type=int, default=256, help="GraphSAGE output dimension (paper: 256)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-incremental", action="store_true",
                    help="train on all relatedness graphs at once instead of incrementally")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--train-silos", default=None,
                    help="comma-separated silo ids to train the predictor on "
                         "(default: build.json's train_silos, else all). Silos left out still get "
                         "their own relatedness graph for message passing, so their columns are "
                         "embedded but their domains were never a training target.")
    ap.add_argument("--profile-cache", type=Path, default=None,
                    help="pickle of column profiles (default: <benchmark>/profiles.pickle)")
    ap.add_argument("--exact-pr-max", type=int, default=50_000_000,
                    help="above this many inter-silo pairs, use the validated binned PR "
                         "estimator instead of materialising every pair (which costs several GB "
                         "and a sort of the same length)")
    ap.add_argument("--verify-scoring", type=int, default=200,
                    help="sample size for checking batched scoring against SiMa's own (0 to skip)")
    ap.add_argument("--export-matrices", action="store_true",
                    help="also write per-scenario score matrices in smx format")
    ap.add_argument("--benchmark-root", type=Path, default=REPO / "harness" / "generated" / "benchmark-full")
    args = ap.parse_args(argv)

    bench = args.benchmark
    with open(bench / "build.json", encoding="utf-8") as fh:
        build = json.load(fh)
    args.out.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    run = {
        "benchmark": str(bench),
        "build_settings": build["settings"],
        "build_totals": build["totals"],
        "run_settings": {
            "epochs_per_graph": args.epochs,
            "embed_size": args.embed,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "incremental": not args.no_incremental,
            "seed": args.seed,
            "negative_sampling": "NS3 (train.negative_sampling_3)",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    }
    import dgl  # after torch, so the backend message is already handled
    run["environment"]["dgl"] = dgl.__version__

    # ---- 1. columns ------------------------------------------------------
    cols_to_ids, files_to_paths = generate_ids_paths(str(bench / "datasets"))
    print(f"tables {len(files_to_paths)}  columns {len(cols_to_ids)}")

    # ---- 2. profiles -----------------------------------------------------
    cache = args.profile_cache if args.profile_cache is not None else bench / "profiles.pickle"
    profiles, prof_meta = profile_columns(files_to_paths, cache)
    missing = [c for c in cols_to_ids if c not in profiles]
    if missing:
        raise SystemExit(f"{len(missing)} columns have no profile, e.g. {missing[:3]}")
    widths = {len(v) for v in profiles.values()}
    if len(widths) != 1:
        raise SystemExit(f"profiles have inconsistent widths: {sorted(widths)}")
    width = widths.pop()
    names = feature_names(width)
    profiles_tensor, san = sanitize(create_profiles_tensor(profiles, cols_to_ids), names)
    run["profiling"] = {**prof_meta, "feature_width": width, **san}
    print(f"profiles: {width} features, {san['non_finite_features']} non-finite restored to "
          f"scipy<=1.8 constants {san['non_finite_by_kind']}")

    # ---- 3. relatedness graphs ------------------------------------------
    ground_truth = load_ground_truth(bench / "ground_truth" / "matches.csv")
    silos = load_silos(bench / "silos.json")
    t0 = time.perf_counter()
    graphs, all_columns, all_cols_ids, all_ids_cols = create_relatedness_graphs_from_silos(
        cols_to_ids, profiles_tensor.tolist(), ground_truth, silos)
    graph_seconds = time.perf_counter() - t0

    graphs_neg = {i: negative_sampling_3(graphs[i]) for i in range(len(graphs))}
    run["graphs"] = {
        "seconds": round(graph_seconds, 3),
        "silos": len(graphs),
        "per_silo": [
            {"silo": i, "tables": len(silos[i]), "nodes": int(graphs[i].num_nodes()),
             "positive_edges": int(graphs[i].num_edges()),
             "negative_edges": int(graphs_neg[i].num_edges())}
            for i in range(len(graphs))
        ],
    }
    for row in run["graphs"]["per_silo"]:
        print(f"  silo {row['silo']}: {row['nodes']} columns, "
              f"{row['positive_edges']} positive / {row['negative_edges']} negative edges")

    # ---- 4. train --------------------------------------------------------
    if args.train_silos is not None:
        train_ids = [int(x) for x in args.train_silos.split(",") if x.strip() != ""]
    elif build.get("train_silos") and build.get("holdout_silos"):
        train_ids = list(build["train_silos"])
    else:
        train_ids = list(range(len(graphs)))
    unknown = [i for i in train_ids if i not in graphs]
    if unknown:
        raise SystemExit(f"--train-silos names silos that do not exist: {unknown}")
    if not train_ids:
        raise SystemExit("no training silos selected")
    held_ids = [i for i in range(len(graphs)) if i not in set(train_ids)]

    # train_model() indexes its inputs positionally, so pass compact lists.
    train_graphs = {k: graphs[i] for k, i in enumerate(train_ids)}
    train_graphs_neg = {k: graphs_neg[i] for k, i in enumerate(train_ids)}
    t0 = time.perf_counter()
    model = train_model(train_graphs, train_graphs_neg, args.epochs, args.embed,
                        incremental=not args.no_incremental, rate=args.lr, wd=args.weight_decay)
    run["training"] = {"seconds": round(time.perf_counter() - t0, 3),
                       "train_silos": train_ids, "held_out_silos": held_ids}
    print(f"trained on silos {train_ids} in {run['training']['seconds']}s"
          + (f"; held out {held_ids}" if held_ids else ""))

    # ---- 5. embed --------------------------------------------------------
    with torch.no_grad():
        embeddings = {j: model.gnn(graphs[j], graphs[j].ndata["feat"]).detach()
                      for j in range(len(graphs))}

    if args.verify_scoring:
        rng = random.Random(args.seed)
        run["scoring_verification"] = verify_scoring(
            model.predictor, embeddings, all_columns, all_cols_ids, args.verify_scoring, rng)
        print(f"scoring check: max |batched - SiMa| = "
              f"{run['scoring_verification']['max_abs_deviation']:.3e} over "
              f"{run['scoring_verification']['pairs_checked']} pairs")

    # ---- 6. score every inter-silo pair ---------------------------------
    import itertools
    t0 = time.perf_counter()
    total_pairs = sum(len(all_columns[a]) * len(all_columns[b])
                      for a, b in itertools.combinations(range(len(graphs)), 2))
    exact = total_pairs <= args.exact_pr_max
    print(f"{total_pairs} inter-silo pairs; "
          + ("exact PR" if exact else f"binned PR ({PR_BINS} bins, validated to <2e-5 AP)"))

    label_blocks, score_blocks = [], []
    held_label_blocks, held_score_blocks = [], []
    hist_all, hist_held = PrHistogram(), PrHistogram()
    held_set = set(held_ids)
    for a, b in itertools.combinations(range(len(graphs)), 2):
        ca, cb = all_columns[a], all_columns[b]
        if not ca or not cb:
            continue
        idx_a = [all_cols_ids[a][c] for c in ca]
        idx_b = [all_cols_ids[b][c] for c in cb]
        matrix = batched_scores(model.predictor, embeddings[a][idx_a], embeddings[b][idx_b])
        # Labelling stays SiMa's own compute_probabilities_labels; only the
        # accumulation is per silo pair, to bound peak memory.
        results = [(ca[i], cb[j], float(matrix[i, j]))
                   for i in range(len(ca)) for j in range(len(cb))]
        labels, probs = compute_probabilities_labels(results, ground_truth)
        lab_arr = np.asarray(labels, dtype=np.int8)
        sc_arr = np.asarray(probs, dtype=np.float32)
        hist_all.add(lab_arr, sc_arr)
        if exact:
            label_blocks.append(lab_arr)
            score_blocks.append(sc_arr)
        if a in held_set and b in held_set:
            hist_held.add(lab_arr, sc_arr)
            if exact:
                held_label_blocks.append(lab_arr)
                held_score_blocks.append(sc_arr)
        del results, labels, probs, lab_arr, sc_arr
    predict_seconds = time.perf_counter() - t0

    if exact:
        labels = np.concatenate(label_blocks)
        scores = np.concatenate(score_blocks).astype(np.float64)
        summary = pr_summary(labels, scores)
    else:
        labels = scores = None
        summary = hist_all.summary()
    del label_blocks, score_blocks
    run["inter_silo_prediction"] = {"seconds": round(predict_seconds, 3), **summary}
    m = run["inter_silo_prediction"]
    print(f"inter-silo pairs {m['pairs']} ({m['positives']} positive, "
          f"{m['positive_rate']:.4%}) in {m['seconds']}s")
    print(f"  average precision {m['average_precision']:.4f}")
    print(f"  best F1 {m['best_f1']['f1']:.4f} "
          f"(P {m['best_f1']['precision']:.4f} / R {m['best_f1']['recall']:.4f} "
          f"@ {m['best_f1']['threshold']:.4f})")
    if m["at_threshold_0.5"]:
        h = m["at_threshold_0.5"]
        print(f"  at 0.5: F1 {h['f1']:.4f} (P {h['precision']:.4f} / R {h['recall']:.4f})")

    if exact:
        np.savez_compressed(args.out / "inter_silo_scores.npz", labels=labels, scores=scores)
    else:
        np.savez_compressed(args.out / "inter_silo_pr_histogram.npz",
                            pos=hist_all.pos, neg=hist_all.neg, bins=hist_all.bins)

    if held_ids:
        if exact and held_label_blocks:
            h_labels = np.concatenate(held_label_blocks)
            h_scores = np.concatenate(held_score_blocks).astype(np.float64)
            held_summary = (pr_summary(h_labels, h_scores)
                            if h_labels.sum() and h_labels.sum() < h_labels.size else None)
        else:
            held_summary = hist_held.summary()
            if "skipped" in held_summary:
                held_summary = None
        if held_summary is not None:
            run["held_out_prediction"] = {"silos": held_ids, **held_summary}
            hm = run["held_out_prediction"]
            print(f"held-out silos {held_ids}: {hm['pairs']} pairs, "
                  f"AP {hm['average_precision']:.4f}, best F1 {hm['best_f1']['f1']:.4f} "
                  f"(P {hm['best_f1']['precision']:.4f} / R {hm['best_f1']['recall']:.4f})")
        else:
            run["held_out_prediction"] = {"silos": held_ids,
                                          "skipped": "held-out silo pairs have no both-class labels"}

    # ---- 7. optional smx export -----------------------------------------
    if args.export_matrices:
        table_to_silo = {t: i for i, ts in silos.items() for t in ts}
        run["smx_export"] = export_scenario_matrices(
            build, embeddings, all_cols_ids, table_to_silo, model.predictor,
            args.benchmark_root, args.out / "matrices")
        print(f"exported {run['smx_export']['written']} scenario matrices "
              f"({len(run['smx_export']['skipped'])} skipped)")

    with open(args.out / "run.json", "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2)
    torch.save({"gnn": model.gnn.state_dict(), "predictor": model.predictor.state_dict()},
               args.out / "model.pt")
    print(f"wrote {args.out / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
