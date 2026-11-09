"""Ablate hash-invariant column features on the encrypted paper benchmark.

The shipped ``*_md5_encrypted`` instances use independent per-side value
renamings.  Cross-side equality, labels, types inferred from lexical form, and
IND coverage are therefore unavailable.  This experiment deliberately uses
only properties preserved by a bijective renaming *inside* each table:

* the sorted value-frequency distribution (the existing PMF baseline),
* entropy, distinctness, collision probability, and dominant mass,
* the observed empty-cell rate (expected to be destroyed by this encoding),
* unary UCC/key roles,
* discovered FD roles qualified by positive gpdep, and
* dense unary positive-gpdep role profiles, without FD discovery thresholds,
* within-table IND roles qualified by gcov and binomial significance, and
* simulated shared-HMAC/PSI and bottom-k cross-side overlap channels.

All matrices are evaluated with the common Hungarian extractor.  The fixed
fusion uses 65% PMF / 35% added evidence, matching the PRISMA attribute versus
structure split used elsewhere in this harness.  A small gpdep weight sweep is
also emitted and is explicitly exploratory/in-sample.  The shared-key results
simulate a different encryption contract from the shipped independent salts;
they are an accuracy experiment, not a production cryptographic protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Appended, not prepended: the vendored wheels in .deps are built for cp312
# while this tree is usually run under 3.10, so putting them first shadows a
# working system numpy/scipy with extension modules that cannot load at all.
# As a fallback they still cover an interpreter that has neither installed.
sys.path.append(str(Path(__file__).resolve().parent / ".deps"))
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import binom, wilcoxon
except ImportError:  # pragma: no cover - keeps the script usable without the optional wheel
    linear_sum_assignment = None
    binom = None
    wilcoxon = None


DATASETS = (
    "TPC-DI-Unionable_md5_encrypted",
    "ChEMBL-Unionable_md5_encrypted",
    "Valentine-Wikidata-Unionable_md5_encrypted",
    "Magellan-Unionable_md5_encrypted",
)
SHORT_NAMES = {
    "TPC-DI-Unionable_md5_encrypted": "TPC-DI",
    "ChEMBL-Unionable_md5_encrypted": "ChEMBL",
    "Valentine-Wikidata-Unionable_md5_encrypted": "Wikidata",
    "Magellan-Unionable_md5_encrypted": "Magellan",
}
QP = np.linspace(0.0, 1.0, 5)
FD_RE = re.compile(r"^\[(.*?)\]\s*-->\s*(.*?)\s*\(pdep\s+([^,]+),\s*([^)]+)\)")
NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}
EXPERIMENT_HMAC_KEY = b"schematch-shared-key-ablation-v1"


def read_matrix(path: Path) -> tuple[np.ndarray, list[str] | None, list[str] | None]:
    """Read the plain or labelled dense CSV form used by schematch."""
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [row for row in csv.reader(fh) if row]
    col_labels = None
    try:
        float(rows[0][0])
    except ValueError:
        col_labels = rows.pop(0)[1:]
    row_labels = None
    try:
        float(rows[0][0])
    except ValueError:
        row_labels = [row.pop(0) for row in rows]
    return np.asarray([[float(cell) for cell in row] for row in rows]), row_labels, col_labels


def find_ground_truth(scenario: Path) -> Path | None:
    files = sorted((scenario / "ground_truth").glob("*.csv"))
    return files[0] if files else None


def profile_quantiles(matrix: np.ndarray) -> np.ndarray:
    features = []
    quantiles = np.linspace(0.0, 1.0, 9)
    for i in range(matrix.shape[0]):
        out = np.delete(matrix[i, :], i) if matrix.shape[0] > 1 else np.zeros(1)
        inc = np.delete(matrix[:, i], i) if matrix.shape[0] > 1 else np.zeros(1)
        features.append(np.concatenate((np.quantile(out, quantiles), np.quantile(inc, quantiles))))
    return np.asarray(features)


def quantile_scores(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, right = profile_quantiles(source), profile_quantiles(target)
    distance = np.abs(left[:, None, :] - right[None, :, :]).mean(axis=2)
    scale = float(distance.max())
    return 1.0 - distance / scale if scale > 0 else np.zeros(distance.shape)


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"empty table: {path}")
    header = [x.strip() for x in rows[0]]
    width = len(header)
    data = [row + [""] * (width - len(row)) for row in rows[1:]]
    columns = [[row[i] if i < len(row) else "" for row in data] for i in range(width)]
    return header, columns


def non_null_cipher_columns(
    encrypted: list[list[str]], plaintext: list[list[str]]
) -> list[list[str]]:
    """Apply the owner's plaintext null mask to ciphertext values.

    The shipped transformation hashes nulls as ordinary values.  Dependency
    discovery should operate on non-null observations, so this simulates the
    reserved NULL marker a deployable encrypted pipeline needs.  No plaintext
    value is used as evidence; it only decides whether the aligned ciphertext
    cell is null.
    """
    if len(encrypted) != len(plaintext):
        raise ValueError("plaintext/encrypted schema width differs")
    cleaned = []
    for enc_col, plain_col in zip(encrypted, plaintext):
        if len(enc_col) != len(plain_col):
            raise ValueError("plaintext/encrypted row count differs")
        cleaned.append(
            [cipher for cipher, raw in zip(enc_col, plain_col) if raw.strip().lower() not in NULLS]
        )
    return cleaned


def non_null_plain_columns(columns: list[list[str]]) -> list[list[str]]:
    return [
        [value.strip() for value in column if value.strip().lower() not in NULLS]
        for column in columns
    ]


def binomial_significance(n: int, k: int, probability: float) -> float:
    """-log10 P[Bin(n, probability) >= k], capped only after log-space evaluation."""
    if n <= 0 or k <= 0 or probability >= 1.0:
        return 0.0
    probability = max(0.0, probability)
    if probability == 0.0:
        return 300.0
    if binom is None:  # conservative normal-free fallback
        return 0.0
    log_probability = float(binom.logsf(k - 1, n, probability))
    if not math.isfinite(log_probability):
        return 300.0
    return min(300.0, max(0.0, -log_probability / math.log(10.0)))


def within_ind_evidence(columns: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Dense within-table IND gcov and significance for every ordered pair."""
    sets = [set(column) for column in columns]
    counters = [Counter(column) for column in columns]
    total = Counter()
    for counter in counters:
        total.update(counter)
    n_columns = len(columns)
    gcov = np.zeros((n_columns, n_columns))
    significance = np.zeros((n_columns, n_columns))
    for dependent in range(n_columns):
        source_values = sets[dependent]
        n = len(source_values)
        if n == 0:
            continue
        background = total.copy()
        background.subtract(counters[dependent])
        background_total = sum(count for count in background.values() if count > 0)
        for referenced in range(n_columns):
            if dependent == referenced:
                continue
            target_values = sets[referenced]
            k = len(source_values & target_values)
            if k == 0 or background_total <= 0:
                continue
            coverage = k / n
            b = sum(max(0, background[value]) for value in target_values) / background_total
            genuine = (coverage - b) / (1.0 - b) if b < 1.0 else 0.0
            gcov[dependent, referenced] = max(0.0, genuine)
            significance[dependent, referenced] = binomial_significance(n, k, b)
    return gcov, significance


def within_ind_role_profile(
    gcov: np.ndarray, significance: np.ndarray | None = None
) -> np.ndarray:
    """Permutation-invariant outgoing/incoming qualified-IND descriptor."""
    features = []
    n = gcov.shape[0]
    for column in range(n):
        row: list[float] = []
        directions = (
            (np.delete(gcov[column, :], column), np.delete(significance[column, :], column)
             if significance is not None else None),
            (np.delete(gcov[:, column], column), np.delete(significance[:, column], column)
             if significance is not None else None),
        )
        for strengths, sigs in directions:
            positive = strengths > 0
            values = strengths[positive]
            row.append(float(positive.mean()) if positive.size else 0.0)
            row.extend(np.quantile(values, QP) if values.size else np.zeros(len(QP)))
            if sigs is not None:
                bounded = np.log1p(np.minimum(sigs[positive], 300.0)) / math.log(301.0)
                row.extend(np.quantile(bounded, QP) if bounded.size else np.zeros(len(QP)))
                joint = values * bounded
                row.extend(np.quantile(joint, QP) if joint.size else np.zeros(len(QP)))
        features.append(row)
    return np.asarray(features)


def shared_hmac_sets(columns: list[list[str]]) -> list[set[bytes]]:
    """Deterministic shared-key tokens; digests remain in memory only."""
    return [
        {
            hmac.new(EXPERIMENT_HMAC_KEY, value.encode("utf-8"), hashlib.sha256).digest()
            for value in column
        }
        for column in non_null_plain_columns(columns)
    ]


def psi_containment_scores(source: list[set], target: list[set]) -> np.ndarray:
    """Symmetric containment using only |A|, |B| and PSI intersection cardinality."""
    scores = np.zeros((len(source), len(target)))
    for i, left in enumerate(source):
        for j, right in enumerate(target):
            denominator = max(len(left), len(right))
            scores[i, j] = len(left & right) / denominator if denominator else 0.0
    return scores


def shared_hmac_genuine_scores(source: list[set], target: list[set]) -> np.ndarray:
    """Cross-side gcov weighted by binomial significance in both directions."""
    all_sets = source + target
    background_size = len(set().union(*all_sets)) if all_sets else 0
    scores = np.zeros((len(source), len(target)))
    if background_size == 0:
        return scores
    for i, left in enumerate(source):
        for j, right in enumerate(target):
            shared = len(left & right)
            if not left or not right or shared == 0:
                continue
            cov_lr, cov_rl = shared / len(left), shared / len(right)
            b_lr = min(len(right) / background_size, 1.0)
            b_rl = min(len(left) / background_size, 1.0)
            g_lr = (cov_lr - b_lr) / (1.0 - b_lr) if b_lr < 1.0 else 0.0
            g_rl = (cov_rl - b_rl) / (1.0 - b_rl) if b_rl < 1.0 else 0.0
            sig = min(
                binomial_significance(len(left), shared, b_lr),
                binomial_significance(len(right), shared, b_rl),
            )
            confidence = 1.0 - math.exp(-sig)
            scores[i, j] = max(0.0, min(g_lr, g_rl)) * confidence
    return scores


def bottom_k_sketches(columns: list[set[bytes]], k: int) -> list[tuple[set[int], int]]:
    sketches = []
    for column in columns:
        hashes = sorted(int.from_bytes(value[:8], "big") for value in column)
        sketches.append((set(hashes[:k]), len(hashes)))
    return sketches


def bottom_k_containment_scores(
    source: list[tuple[set[int], int]], target: list[tuple[set[int], int]], k: int
) -> np.ndarray:
    """Estimate containment from a mergeable bottom-k Jaccard sketch plus cardinality."""
    scores = np.zeros((len(source), len(target)))
    for i, (left, n_left) in enumerate(source):
        for j, (right, n_right) in enumerate(target):
            union_bottom = set(sorted(left | right)[:k])
            if not union_bottom or max(n_left, n_right) == 0:
                continue
            jaccard = sum(value in left and value in right for value in union_bottom) / len(union_bottom)
            intersection = jaccard * (n_left + n_right) / (1.0 + jaccard)
            scores[i, j] = min(1.0, intersection / max(n_left, n_right))
    return scores


def scalar_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.abs(left[:, None] - right[None, :]), 0.0, 1.0)


def pmf(column: list[str]) -> np.ndarray:
    if not column:
        return np.zeros(0)
    counts = sorted(Counter(column).values(), reverse=True)
    return np.asarray(counts, dtype=float) / len(column)


def pmf_scores(source: list[list[str]], target: list[list[str]]) -> np.ndarray:
    profiles_s = [pmf(col) for col in source]
    profiles_t = [pmf(col) for col in target]
    out = np.zeros((len(source), len(target)))
    for i, left in enumerate(profiles_s):
        for j, right in enumerate(profiles_t):
            if left.size == 0 or right.size == 0:
                out[i, j] = 1.0 if left.size == right.size else 0.0
                continue
            n = max(left.size, right.size)
            lp, rp = np.zeros(n), np.zeros(n)
            lp[: left.size], rp[: right.size] = left, right
            out[i, j] = 1.0 - np.linalg.norm(lp - rp) / math.sqrt(2.0)
    return np.clip(out, 0.0, 1.0)


def unary_statistics(columns: list[list[str]]) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = defaultdict(list)
    for col in columns:
        n = len(col)
        profile = pmf(col)
        if n == 0 or profile.size == 0:
            entropy = distinctness = collision = dominant = 0.0
        else:
            nz = profile[profile > 0]
            raw_entropy = float(-(nz * np.log(nz)).sum())
            entropy = raw_entropy / math.log(n) if n > 1 else 0.0
            distinctness = profile.size / n
            collision = float(np.square(profile).sum())
            dominant = float(profile[0])
        result["entropy"].append(entropy)
        result["distinctness"].append(distinctness)
        result["collision"].append(collision)
        result["dominant_mass"].append(dominant)
        # This is intentionally measured from the ciphertext table.  In the
        # shipped benchmark all plaintext nulls were renamed too, so it should
        # expose that missingness is not available to a downstream matcher.
        result["observed_null_rate"].append(
            sum(value == "" for value in col) / n if n else 0.0
        )
    return {name: np.asarray(values) for name, values in result.items()}


def local_name(qualified: str) -> str:
    return qualified.strip().rsplit(".", 1)[-1]


def read_uccs(path: Path) -> set[frozenset[str]]:
    keys: set[frozenset[str]] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            value = line.strip()
            if value.startswith("[") and value.endswith("]"):
                members = [x.strip() for x in value[1:-1].split(",") if x.strip()]
                if members:
                    keys.add(frozenset(local_name(member) for member in members))
    return keys


def unary_ucc_flags(keys: set[frozenset[str]], headers: list[str]) -> np.ndarray:
    unary = {next(iter(key)) for key in keys if len(key) == 1}
    return np.asarray([1.0 if header in unary else 0.0 for header in headers])


def discovered_fd_profile(
    path: Path, headers: list[str], candidate_keys: set[frozenset[str]]
) -> np.ndarray:
    """Positive-gpdep FD role evidence, excluding key-implied determinants.

    Exact ``sig`` is unavailable because the profiler output has neither the
    support counts nor the chance baseline.  Positive gpdep is the qualification
    that can be reproduced faithfully from the stored evidence.
    """
    index = {header: i for i, header in enumerate(headers)}
    outgoing: list[list[float]] = [[] for _ in headers]
    incoming: list[list[float]] = [[] for _ in headers]
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = FD_RE.match(line.strip())
            if not match:
                continue
            determinants = [local_name(x) for x in match.group(1).split(",")]
            dependent = local_name(match.group(2))
            try:
                gpdep = float(match.group(4))
            except ValueError:
                continue
            if gpdep <= 0:
                continue
            determinant_set = frozenset(determinants)
            if any(key <= determinant_set for key in candidate_keys):
                continue  # key-implied FD: syntactically valid but uninformative
            if dependent in index:
                incoming[index[dependent]].append(gpdep)
            for determinant in determinants:
                if determinant in index:
                    outgoing[index[determinant]].append(gpdep)

    features = []
    for out, inc in zip(outgoing, incoming):
        # Counts are log-compressed and normalised by the schema width so that
        # scenarios with different numbers of columns remain comparable.
        denom = math.log1p(max(1, len(headers) - 1))
        row = [math.log1p(len(out)) / denom, math.log1p(len(inc)) / denom]
        for values in (out, inc):
            row.extend(np.quantile(values, QP) if values else np.zeros(len(QP)))
        features.append(row)
    return np.asarray(features, dtype=float)


def positive_gpdep_profile(matrix: np.ndarray) -> np.ndarray:
    """Threshold-free role descriptor retaining only genuine (>0) strengths."""
    n = matrix.shape[0]
    features = []
    for i in range(n):
        out = np.delete(matrix[i, :], i) if n > 1 else np.zeros(1)
        inc = np.delete(matrix[:, i], i) if n > 1 else np.zeros(1)
        row: list[float] = []
        for values in (out, inc):
            positive = values[values > 0]
            row.append(float(positive.size / values.size) if values.size else 0.0)
            row.extend(np.quantile(positive, QP) if positive.size else np.zeros(len(QP)))
        features.append(row)
    return np.asarray(features)


def vector_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = np.abs(left[:, None, :] - right[None, :, :]).mean(axis=2)
    return np.clip(1.0 - distance, 0.0, 1.0)


def fuse(base: np.ndarray, evidence: np.ndarray, weight: float) -> np.ndarray:
    """Fuse evidence while treating a constant matrix as no information.

    Adding a constant is mathematically assignment-invariant but can change a
    floating-point solver's arbitrary resolution of exact ties.  Returning the
    base verbatim prevents such numerical changes from being misreported as a
    feature gain.
    """
    if float(np.ptp(evidence)) <= 1e-12:
        return base.copy()
    return (1.0 - weight) * base + weight * evidence


def scale_aligned_fuse(base: np.ndarray, evidence: np.ndarray, weight: float) -> np.ndarray:
    """Affine-align new evidence to an existing matcher's score range."""
    evidence_range = float(np.ptp(evidence))
    base_range = float(np.ptp(base))
    if evidence_range <= 1e-12 or base_range <= 1e-12:
        return base.copy()
    aligned = (evidence - float(evidence.min())) / evidence_range
    aligned = aligned * base_range + float(base.min())
    return (1.0 - weight) * base + weight * aligned


def side_metadata(scenario: Path, side: str, kind: str) -> Path:
    files = sorted((scenario / "metadata" / side).rglob(f"{kind}_results.txt"))
    return files[0] if files else Path("__missing__")


def hungarian_pairs(scores: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight rectangular assignment (O(n^3), no SciPy dependency)."""
    if linear_sum_assignment is not None:
        rows, columns = linear_sum_assignment(scores, maximize=True)
        return list(zip(rows.tolist(), columns.tolist()))
    transposed = scores.shape[0] > scores.shape[1]
    work = scores.T if transposed else scores
    n, m = work.shape
    cost = -work
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] > 0]
    return [(j, i) for i, j in pairs] if transposed else pairs


def evaluate_matrix(scores: np.ndarray, gt: np.ndarray) -> float:
    predicted = np.zeros(scores.shape, dtype=bool)
    for row, column in hungarian_pairs(scores):
        if scores[row, column] > 1e-9:
            predicted[row, column] = True
    truth = gt > 0.5
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../code/schematch/data")
    ap.add_argument("--gpdep-dir", default="unary_gpdep_encrypted")
    ap.add_argument("--out", default="reports/encrypted_feature_ablation_scenarios.csv")
    ap.add_argument("--summary", default="reports/encrypted_feature_ablation_summary.csv")
    ap.add_argument("--key-summary", default="reports/encrypted_feature_ablation_key_results.csv")
    ap.add_argument("--comparisons", default="reports/encrypted_feature_ablation_comparisons.csv")
    ap.add_argument(
        "--sf-dir", default="../code/schematch/results/09-02-2026_10-22-31",
        help="existing SF+FD+PMF result tree",
    )
    ap.add_argument(
        "--pded-dir", default="pded_distribution_out/w065",
        help="existing PDED+PMF matrix tree",
    )
    args = ap.parse_args()

    data_dir, gpdep_dir = Path(args.data_dir), Path(args.gpdep_dir)
    sf_dir, pded_dir = Path(args.sf_dir), Path(args.pded_dir)
    rows: list[dict[str, object]] = []
    feature_availability: dict[str, int] = defaultdict(int)

    for dataset in DATASETS:
        root = data_dir / dataset
        for scenario in sorted(path for path in root.iterdir() if path.is_dir()):
            src_csv = sorted((scenario / "source").glob("*.csv"))[0]
            tgt_csv = sorted((scenario / "target").glob("*.csv"))[0]
            src_headers, source = read_table(src_csv)
            tgt_headers, target = read_table(tgt_csv)
            plain_scenario = data_dir / dataset.removesuffix("_md5_encrypted") / scenario.name
            plain_src_csv = sorted((plain_scenario / "source").glob("*.csv"))[0]
            plain_tgt_csv = sorted((plain_scenario / "target").glob("*.csv"))[0]
            _, plain_source = read_table(plain_src_csv)
            _, plain_target = read_table(plain_tgt_csv)
            gt_path = find_ground_truth(scenario)
            if gt_path is None:
                raise ValueError(f"missing ground truth: {scenario}")
            gt, _, _ = read_matrix(gt_path)

            matrices: dict[str, np.ndarray] = {"Distribution (PMF)": pmf_scores(source, target)}
            stats_s, stats_t = unary_statistics(source), unary_statistics(target)
            for feature in stats_s:
                label = {
                    "entropy": "Entropy",
                    "distinctness": "Distinctness",
                    "collision": "Collision probability",
                    "dominant_mass": "Dominant mass",
                    "observed_null_rate": "Observed null rate",
                }[feature]
                matrices[label] = scalar_similarity(stats_s[feature], stats_t[feature])

            src_keys = read_uccs(side_metadata(scenario, "source", "UCC"))
            tgt_keys = read_uccs(side_metadata(scenario, "target", "UCC"))
            src_ucc = unary_ucc_flags(src_keys, src_headers)
            tgt_ucc = unary_ucc_flags(tgt_keys, tgt_headers)
            matrices["Unary key role"] = scalar_similarity(src_ucc, tgt_ucc)
            feature_availability["source_unary_ucc"] += int(src_ucc.sum())
            feature_availability["target_unary_ucc"] += int(tgt_ucc.sum())

            fd_s = discovered_fd_profile(
                side_metadata(scenario, "source", "FD"), src_headers, src_keys
            )
            fd_t = discovered_fd_profile(
                side_metadata(scenario, "target", "FD"), tgt_headers, tgt_keys
            )
            matrices["Qualified discovered-FD role"] = vector_similarity(fd_s, fd_t)

            gp_s_path = gpdep_dir / dataset / scenario.name / "gpdep_source.csv"
            gp_t_path = gpdep_dir / dataset / scenario.name / "gpdep_target.csv"
            gp_s, _, _ = read_matrix(gp_s_path)
            gp_t, _, _ = read_matrix(gp_t_path)
            matrices["Dense gpdep quantiles"] = quantile_scores(gp_s, gp_t)
            matrices["Dense positive-gpdep role"] = vector_similarity(
                positive_gpdep_profile(gp_s), positive_gpdep_profile(gp_t)
            )

            # Same-table INDs remain observable under each side's own salt.
            # The plaintext counterpart supplies only the null mask because the
            # shipped transform did not reserve a recognisable NULL token.
            clean_source = non_null_cipher_columns(source, plain_source)
            clean_target = non_null_cipher_columns(target, plain_target)
            ind_gcov_s, ind_sig_s = within_ind_evidence(clean_source)
            ind_gcov_t, ind_sig_t = within_ind_evidence(clean_target)
            matrices["Within-table IND gcov role"] = vector_similarity(
                within_ind_role_profile(ind_gcov_s),
                within_ind_role_profile(ind_gcov_t),
            )
            matrices["Within-table IND gcov+sig role"] = vector_similarity(
                within_ind_role_profile(ind_gcov_s, ind_sig_s),
                within_ind_role_profile(ind_gcov_t, ind_sig_t),
            )

            # Simulated alternative privacy contracts. Shared HMAC exposes
            # deterministic equality tokens; PSI-cardinality can produce the
            # same |intersection| score without disclosing those tokens. The
            # bottom-k variants exchange only a fixed-size overlap sketch.
            hmac_source = shared_hmac_sets(plain_source)
            hmac_target = shared_hmac_sets(plain_target)
            matrices["Shared HMAC / PSI-cardinality containment"] = psi_containment_scores(
                hmac_source, hmac_target
            )
            matrices["Shared HMAC gcov+sig"] = shared_hmac_genuine_scores(
                hmac_source, hmac_target
            )
            for sketch_size in (32, 128):
                matrices[f"Bottom-k({sketch_size}) overlap sketch"] = bottom_k_containment_scores(
                    bottom_k_sketches(hmac_source, sketch_size),
                    bottom_k_sketches(hmac_target, sketch_size),
                    sketch_size,
                )

            sf_candidates = sorted(
                (sf_dir / dataset / scenario.name / "_outputs" / "MatchingStepLine2").glob("*.csv")
            )
            pmf_candidates = sorted(
                (sf_dir / dataset / scenario.name / "_outputs" / "MatchingStepLine1").glob("*.csv")
            )
            pded_path = pded_dir / dataset / scenario.name / "pded.csv"
            # Prefer the persisted Java matrix so this baseline is byte-for-byte
            # the one used by the earlier SF/PDED feature-transfer report.
            if pmf_candidates:
                matrices["Distribution (PMF)"], _, _ = read_matrix(pmf_candidates[0])
            if sf_candidates:
                sf_base, _, _ = read_matrix(sf_candidates[0])
                matrices["SF+FD+PMF (existing)"] = sf_base
                for weight in (0.20, 0.35):
                    matrices[f"SF+FD+PMF + positive-gpdep (w={weight:.2f})"] = scale_aligned_fuse(
                        sf_base, matrices["Dense positive-gpdep role"], weight
                    )
                    for feature in ("Distinctness", "Entropy"):
                        matrices[f"SF+FD+PMF + {feature.lower()} (w={weight:.2f})"] = scale_aligned_fuse(
                            sf_base, matrices[feature], weight
                        )
                    for feature in (
                        "Within-table IND gcov role",
                        "Within-table IND gcov+sig role",
                        "Shared HMAC / PSI-cardinality containment",
                        "Shared HMAC gcov+sig",
                        "Bottom-k(128) overlap sketch",
                    ):
                        matrices[f"SF+FD+PMF + {feature} (w={weight:.2f})"] = scale_aligned_fuse(
                            sf_base, matrices[feature], weight
                        )
            if pded_path.is_file():
                pded_base, _, _ = read_matrix(pded_path)
                matrices["PDED+PMF (existing)"] = pded_base
                for weight in (0.20, 0.35):
                    matrices[f"PDED+PMF + positive-gpdep (w={weight:.2f})"] = scale_aligned_fuse(
                        pded_base, matrices["Dense positive-gpdep role"], weight
                    )
                    for feature in ("Distinctness", "Entropy"):
                        matrices[f"PDED+PMF + {feature.lower()} (w={weight:.2f})"] = scale_aligned_fuse(
                            pded_base, matrices[feature], weight
                        )
                    for feature in (
                        "Within-table IND gcov role",
                        "Within-table IND gcov+sig role",
                        "Shared HMAC / PSI-cardinality containment",
                        "Shared HMAC gcov+sig",
                        "Bottom-k(128) overlap sketch",
                    ):
                        matrices[f"PDED+PMF + {feature} (w={weight:.2f})"] = scale_aligned_fuse(
                            pded_base, matrices[feature], weight
                        )

            pmf_matrix = matrices["Distribution (PMF)"]
            added = (
                "Entropy",
                "Distinctness",
                "Collision probability",
                "Dominant mass",
                "Observed null rate",
                "Unary key role",
                "Qualified discovered-FD role",
                "Dense gpdep quantiles",
                "Dense positive-gpdep role",
                "Within-table IND gcov role",
                "Within-table IND gcov+sig role",
                "Shared HMAC / PSI-cardinality containment",
                "Shared HMAC gcov+sig",
                "Bottom-k(32) overlap sketch",
                "Bottom-k(128) overlap sketch",
            )
            for feature in added:
                matrices[f"PMF + {feature} (65/35)"] = fuse(pmf_matrix, matrices[feature], 0.35)

            # A compact combined candidate: PMF remains dominant; dependency
            # role and key evidence contribute only when present.
            matrices["PMF + positive-gpdep + key (65/25/10)"] = (
                0.65 * pmf_matrix
                + 0.25 * matrices["Dense positive-gpdep role"]
                + 0.10 * matrices["Unary key role"]
            )
            matrices["PMF + positive-gpdep + distinctness (65/20/15)"] = (
                0.65 * pmf_matrix
                + 0.20 * matrices["Dense positive-gpdep role"]
                + 0.15 * matrices["Distinctness"]
            )
            matrices["PMF + positive-gpdep + entropy (65/20/15)"] = (
                0.65 * pmf_matrix
                + 0.20 * matrices["Dense positive-gpdep role"]
                + 0.15 * matrices["Entropy"]
            )
            matrices["PMF + gpdep-quantiles + distinctness (65/20/15)"] = (
                0.65 * pmf_matrix
                + 0.20 * matrices["Dense gpdep quantiles"]
                + 0.15 * matrices["Distinctness"]
            )
            for weight in (0.10, 0.20, 0.35, 0.50):
                matrices[f"PMF + positive-gpdep (w={weight:.2f})"] = fuse(
                    pmf_matrix, matrices["Dense positive-gpdep role"], weight
                )

            for method, scores in matrices.items():
                rows.append(
                    {
                        "dataset": SHORT_NAMES[dataset],
                        "encrypted_dataset": dataset,
                        "scenario": scenario.name,
                        "method": method,
                        "f1": evaluate_matrix(scores, gt),
                    }
                )

    write_rows(Path(args.out), rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method"]))].append(float(row["f1"]))
    summary = [
        {
            "dataset": dataset,
            "scenarios": len(values),
            "method": method,
            "macro_f1": float(np.mean(values)),
        }
        for (dataset, method), values in grouped.items()
    ]
    order = {name: i for i, name in enumerate(SHORT_NAMES.values())}
    summary.sort(key=lambda row: (order[str(row["dataset"])], str(row["method"])))
    write_rows(Path(args.summary), summary)

    key_methods = (
        "Distribution (PMF)",
        "Entropy",
        "Distinctness",
        "Dense positive-gpdep role",
        "PMF + Entropy (65/35)",
        "PMF + Distinctness (65/35)",
        "PMF + Dense positive-gpdep role (65/35)",
        "SF+FD+PMF (existing)",
        "SF+FD+PMF + positive-gpdep (w=0.20)",
        "SF+FD+PMF + distinctness (w=0.20)",
        "PDED+PMF (existing)",
        "PDED+PMF + positive-gpdep (w=0.20)",
        "PDED+PMF + distinctness (w=0.35)",
        "Within-table IND gcov role",
        "Within-table IND gcov+sig role",
        "PMF + Within-table IND gcov role (65/35)",
        "PMF + Within-table IND gcov+sig role (65/35)",
        "Shared HMAC / PSI-cardinality containment",
        "Shared HMAC gcov+sig",
        "Bottom-k(32) overlap sketch",
        "Bottom-k(128) overlap sketch",
        "SF+FD+PMF + Within-table IND gcov role (w=0.35)",
        "SF+FD+PMF + Shared HMAC / PSI-cardinality containment (w=0.35)",
        "SF+FD+PMF + Bottom-k(128) overlap sketch (w=0.35)",
        "PDED+PMF + Within-table IND gcov role (w=0.35)",
        "PDED+PMF + Shared HMAC / PSI-cardinality containment (w=0.35)",
        "PDED+PMF + Bottom-k(128) overlap sketch (w=0.35)",
    )
    key_summary = [row for row in summary if row["method"] in key_methods]
    write_rows(Path(args.key_summary), key_summary)

    by_method: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in rows:
        by_method[str(row["method"])][
            (str(row["encrypted_dataset"]), str(row["scenario"]))
        ] = float(row["f1"])
    comparisons_to_make = (
        ("Distribution (PMF)", "PMF + Dense positive-gpdep role (65/35)"),
        ("Distribution (PMF)", "PMF + Entropy (65/35)"),
        ("Distribution (PMF)", "PMF + Distinctness (65/35)"),
        ("SF+FD+PMF (existing)", "SF+FD+PMF + positive-gpdep (w=0.20)"),
        ("SF+FD+PMF (existing)", "SF+FD+PMF + distinctness (w=0.20)"),
        ("PDED+PMF (existing)", "PDED+PMF + positive-gpdep (w=0.20)"),
        ("PDED+PMF (existing)", "PDED+PMF + distinctness (w=0.35)"),
        ("Distribution (PMF)", "PMF + Within-table IND gcov role (65/35)"),
        ("Distribution (PMF)", "PMF + Within-table IND gcov+sig role (65/35)"),
        (
            "PMF + Within-table IND gcov role (65/35)",
            "PMF + Within-table IND gcov+sig role (65/35)",
        ),
        ("Distribution (PMF)", "Shared HMAC / PSI-cardinality containment"),
        ("Distribution (PMF)", "Bottom-k(128) overlap sketch"),
        (
            "Shared HMAC / PSI-cardinality containment",
            "Bottom-k(128) overlap sketch",
        ),
        ("SF+FD+PMF (existing)", "SF+FD+PMF + Within-table IND gcov role (w=0.35)"),
        ("PDED+PMF (existing)", "PDED+PMF + Within-table IND gcov role (w=0.35)"),
        (
            "SF+FD+PMF (existing)",
            "SF+FD+PMF + Shared HMAC / PSI-cardinality containment (w=0.35)",
        ),
        (
            "PDED+PMF (existing)",
            "PDED+PMF + Shared HMAC / PSI-cardinality containment (w=0.35)",
        ),
        (
            "PDED+PMF (existing)",
            "PDED+PMF + Bottom-k(128) overlap sketch (w=0.35)",
        ),
    )
    comparison_rows = []
    for baseline, candidate in comparisons_to_make:
        keys = sorted(set(by_method[baseline]) & set(by_method[candidate]))
        deltas = np.asarray(
            [by_method[candidate][key] - by_method[baseline][key] for key in keys]
        )
        p_value = float(wilcoxon(deltas).pvalue) if wilcoxon is not None else float("nan")
        comparison_rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "scenarios": len(keys),
                "mean_f1_delta": float(deltas.mean()),
                "wins": int((deltas > 1e-12).sum()),
                "losses": int((deltas < -1e-12).sum()),
                "ties": int((np.abs(deltas) <= 1e-12).sum()),
                "wilcoxon_p": p_value,
            }
        )
    write_rows(Path(args.comparisons), comparison_rows)
    print(f"wrote {len(rows)} scenario results to {args.out}")
    print(f"wrote {len(summary)} aggregate results to {args.summary}")
    print(f"wrote {len(key_summary)} key results to {args.key_summary}")
    print(f"wrote {len(comparison_rows)} comparisons to {args.comparisons}")
    print("unary UCC occurrences:", dict(feature_availability))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
