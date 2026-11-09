"""Controlled unary FD/gpdep/U/NMI comparison on identical observations."""
from __future__ import annotations

import hashlib
import random
from collections import Counter

import numpy as np

from encrypted_feature_ablation import NULLS
from make_corrupt import typo
from smx.relational_roles import MI_EPS, _codes, _entropy, aligned_values

MEASURES = ("binary_fd", "gpdep", "u", "nmi")


def dependence_matrices(columns):
    """Pair-complete non-null observations, unary edges, zero diagonal.

    gpdep is the project's UNNORMALISED pdep - expected_pdep, positive part,
    not a [0,1]-rescaled genuine score. Keep signed values for null diagnostics.
    U and geometric NMI are plug-in estimates, without bias adjustment.
    Both constant endpoints and n<2 are uninformative for every measure.
    A separate common nonkey mask removes pairs where EITHER endpoint is unique
    on that pair's actual support. Exact binary FDs use joint-support counts,
    not numerical proximity to pdep=1 or projected composite determinants.
    """
    aligned = aligned_values(columns)
    coded = [_codes(c) for c in aligned]
    width = len(columns)
    matrices = {name: np.zeros((width, width)) for name in MEASURES}
    support = np.zeros((width, width), dtype=np.int64)
    nonkey = np.zeros((width, width), dtype=bool)
    signed = np.zeros((width, width))
    distinct = np.zeros((width, width), dtype=np.int64)
    for i, (x, _) in enumerate(coded):
        support[i, i] = int((x >= 0).sum())
        distinct[i, i] = len(set(x[x >= 0]))
        for j in range(i + 1, width):
            y, ny_total = coded[j]
            valid = (x >= 0) & (y >= 0)
            n = int(valid.sum())
            support[i, j] = support[j, i] = n
            if n < 2:
                continue
            a, b = x[valid], y[valid]
            cx, cy = np.bincount(a), np.bincount(b)
            nx, ny = int(np.count_nonzero(cx)), int(np.count_nonzero(cy))
            distinct[i, j], distinct[j, i] = nx, ny
            hx, hy = _entropy(cx), _entropy(cy)
            if hx <= MI_EPS or hy <= MI_EPS:
                continue
            keys, counts = np.unique(a * ny_total + b, return_counts=True)
            ax, by = keys // ny_total, keys % ny_total
            mi = max(0.0, hx + hy - _entropy(counts))
            if mi <= MI_EPS:
                mi = 0.0
            matrices["u"][i, j], matrices["u"][j, i] = min(1., mi/hy), min(1., mi/hx)
            matrices["nmi"][i, j] = matrices["nmi"][j, i] = min(1., mi/np.sqrt(hx*hy))
            matrices["binary_fd"][i, j] = len(counts) == nx
            matrices["binary_fd"][j, i] = len(counts) == ny
            nonkey[i, j] = nonkey[j, i] = nx < n and ny < n
            for p, q, det_count, ref_counts, group_counts in ((i,j,nx,cy,cx[ax]), (j,i,ny,cx,cy[by])):
                pdep = float(np.sum(counts.astype(float)**2 / group_counts) / n)
                collision = float(np.sum((ref_counts/n)**2))
                expected = collision + (det_count-1)/(n-1)*(1-collision)
                g = pdep - expected
                signed[p, q] = 0.0 if abs(g) <= MI_EPS else g
                matrices["gpdep"][p, q] = g if g > MI_EPS else 0.0
    return {"raw_"+k: v for k, v in matrices.items()} | {
        "nonkey_"+k: v*nonkey for k, v in matrices.items()
    }, {"support": support, "nonkey": nonkey, "distinct": distinct, "signed_gpdep": signed}


def seed_for(case_id, replicate, column, channel):
    return int.from_bytes(hashlib.sha256(f"dependence-v1:{case_id}:{replicate}:{column}:{channel}".encode()).digest()[:8], "big")


def corrupt_columns(columns, mode, rate, case_id, replicate):
    """Fresh target-only copies; nested selected positions at .10/.40.

    shuffle permutes selected non-null slots without changing their multiset.
    mixed reuses the old 50/50 typo/uniform-distinct-donor model. The actual
    changed fraction is reported: selected slots can remain identical.
    No GT, labels, profile data or outcome selection enters corruption.
    """
    if mode not in ("clean", "shuffle", "mixed") or not 0 <= rate <= 1:
        raise ValueError("Invalid corruption mode/rate")
    out, stats = [], []
    for j, col in enumerate(columns):
        changed = list(col)
        slots = [i for i, v in enumerate(col) if v.strip().lower() not in NULLS]
        rng = np.random.default_rng(seed_for(case_id, replicate, j, "positions"))
        chosen = rng.permutation(slots)[:round(rate*len(slots))] if mode != "clean" else []
        if mode == "shuffle" and len(chosen):
            order = np.random.default_rng(seed_for(case_id, replicate, j, "shuffle")).permutation(len(chosen))
            for k, pos in enumerate(chosen):
                changed[pos] = col[chosen[order[k]]]
        elif mode == "mixed":
            pool = sorted({col[i].strip() for i in slots})
            donor_index = {v: i for i, v in enumerate(pool)}
            edits = random.Random(seed_for(case_id, replicate, j, "edits"))
            for pos in chosen:
                value = col[pos].strip()
                if edits.random() < .5 or len(pool) < 2:
                    new = typo(value, edits)
                    # Keep the null mask fixed even if a typo spells a reserved
                    # marker. Prefix only that rare case; log actual changes.
                    changed[pos] = "value:" + new if new.strip().lower() in NULLS else new
                else:
                    k = edits.randrange(len(pool)-1)
                    k += k >= donor_index[value]
                    changed[pos] = pool[k]
        raw_changed = sum(a != b for a, b in zip(col, changed))
        normalized_changed = sum(a.strip() != b.strip() for a, b in zip(col, changed))
        if [v.strip().lower() in NULLS for v in col] != [v.strip().lower() in NULLS for v in changed]:
            raise ValueError("Corruption changed the null mask")
        marginal_equal = Counter(col) == Counter(changed)
        if mode == "shuffle" and not marginal_equal:
            raise ValueError("Shuffle changed column marginals")
        stats.append({"column": j, "non_null": len(slots), "selected": len(chosen),
                      "raw_changed": raw_changed, "normalized_changed": normalized_changed,
                      "marginal_equal": marginal_equal})
        out.append(changed)
    return out, stats


def agreement(clean, dirty):
    """All measures get the SAME error/correlation/edge-set diagnostics."""
    from scipy.stats import rankdata
    mask = ~np.eye(clean.shape[0], dtype=bool)
    a, b = clean[mask], dirty[mask]
    union = int(((a > 0) | (b > 0)).sum())
    def correlation(x, y):
        if len(x) < 2 or np.std(x) <= MI_EPS or np.std(y) <= MI_EPS:
            return None  # Undefined; never silently call flat matrices perfect.
        return float(np.corrcoef(x, y)[0, 1])
    return {"mae": float(np.mean(abs(a-b))) if len(a) else 0.,
            "max_error": float(np.max(abs(a-b))) if len(a) else 0.,
            "pearson": correlation(a,b), "spearman": correlation(rankdata(a),rankdata(b)),
            "positive_jaccard": float(((a>0)&(b>0)).sum()/union) if union else 1.,
            "clean_positive": int((a>0).sum()), "dirty_positive": int((b>0).sum()),
            "clean_mean": float(a.mean()) if len(a) else 0., "dirty_mean": float(b.mean()) if len(b) else 0.}
