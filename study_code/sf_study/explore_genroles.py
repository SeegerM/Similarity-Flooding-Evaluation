"""EXPLORATORY X72 (2026-10-06): a genuine role term, FD roles weighted by gpdep. Written before
any genuine feature was computed; design and predictions in EXPLORATORY.md X72.

  A(γ) = q+ + 5D + γ R_x,  x in (6, g, 13);  R_x = 1 − mean |f(i) − f(j)| over the variant's features
  R6  X46's six (FD left/right counts, UCC count, key flag, IND out/in)
  Rg  R6's four non-FD features + gin_sum, gin_max, gin_n, gout_sum, gout_max, gout_n, keyimp (gpdep-weighted FD roles)
  R13 R6 + the seven genuine features
gpdep is the second value of each FD line's "(pdep a, b)" annotation (clipped at 0). γ chosen leave-one-family-out as X46.

    ../harness/.venv-bdi/Scripts/python.exe explore_genroles.py --regime plain|md5 [--workers 10]
Output: results/<regime>/genroles_{scores.csv,summary.md}
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE / "tuning"))
sys.path.insert(0, str(HERE))
from explore_deproles import CONTROLS, FD_RE_FIXED, SF_ARMS, col, profiles, roles, sf_rows  # noqa: E402
from score_screen_metrics import all_metrics, permutations  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402

G = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)
VARIANTS = ("6", "g", "13")
TAU, KEYIMP = 0.5, 0.01
B, SEED = 10_000, 20261072
ANN = re.compile(r"\(pdep\s+([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)")


def genuine_fds(regime: str, ds: str, sc: str, side: str) -> list[tuple[tuple[str, ...], str, float]]:
    """(lhs, rhs, gpdep) per FD line of one side, labels in matrix form (side.column)."""
    meta = CORPUS[regime] / ds / sc / "metadata" / side
    out = []
    for f in sorted(meta.rglob("FD_results.txt")) if meta.is_dir() else []:
        for line in f.read_text(encoding="utf-8").splitlines():
            m = FD_RE_FIXED.match(line.strip())
            if not m:
                continue
            a = ANN.search(line)
            g = max(float(a.group(2)), 0.0) if a else 0.0
            lhs = tuple(col(x) for x in m.group("lhs").split(",") if x.strip())
            out.append((lhs, col(m.group("rhs")), g))
    return out


def genuine_features(labels: list[str], gfds) -> tuple[np.ndarray, list[str]]:
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    gin_sum, gin_max, gin_n = np.zeros(n), np.zeros(n), np.zeros(n)
    gout_sum, gout_max, gout_n = np.zeros(n), np.zeros(n), np.zeros(n)
    in_all, in_triv = np.zeros(n), np.zeros(n)
    for lhs, rhs, g in gfds:
        if rhs in idx:
            i = idx[rhs]
            gin_sum[i] += g
            gin_max[i] = max(gin_max[i], g)
            gin_n[i] += g >= TAU
            in_all[i] += 1
            in_triv[i] += g <= KEYIMP
        for x in lhs:
            if x in idx:
                j = idx[x]
                gout_sum[j] += g / len(lhs)
                gout_max[j] = max(gout_max[j], g)
                gout_n[j] += g >= TAU
    keyimp = np.divide(in_triv, in_all, out=np.zeros(n), where=in_all > 0)

    def norm(v):
        m = v.max()
        return v / m if m > 0 else v

    F = np.stack([norm(gin_sum), np.clip(gin_max, 0, 1), norm(gin_n), norm(gout_sum), np.clip(gout_max, 0, 1),
                  norm(gout_n), keyimp], axis=1)
    gendet = ["none" if gin_max[i] <= KEYIMP else "strong" if gin_max[i] >= TAU else "weak" for i in range(n)]
    return F, gendet


def sim(Fs: np.ndarray, Ft: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(Fs[:, None, :] - Ft[None, :, :]).mean(axis=2)


def side_features(regime, ds, sc, side, labels):
    F6, _ = roles(labels, *profiles(regime, ds, sc, side))
    Fg7, gendet = genuine_features(labels, genuine_fds(regime, ds, sc, side))
    return {"6": F6, "g": np.concatenate([F6[:, 2:], Fg7], axis=1), "13": np.concatenate([F6, Fg7], axis=1)}, gendet


def score_matcher(job):
    regime, m, cs = job
    seeds = HERE / "reference" / "seeds" / regime
    out = []
    for ds, sc, fam in cs:
        f_q, f_d = seeds / m / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        g_f = ground_truth_file(CORPUS[regime], ds, sc)
        if not (f_q.is_file() and f_d.is_file() and g_f is not None):
            continue
        q, rows, cols = read_matrix(f_q)
        D = read_matrix(f_d)[0]
        gt = read_matrix(g_f)[0] > 0.5
        if not (q.shape == D.shape == gt.shape) or not (np.isfinite(q).all() and np.isfinite(D).all()):
            continue
        qp = np.clip(q, 0, None)
        Fs, _ = side_features(regime, ds, sc, "source", list(rows))
        Ft, _ = side_features(regime, ds, sc, "target", list(cols))
        perms = permutations(gt.shape, f"{ds}/{sc}")
        base = qp + 5.0 * D
        row = {"seed": m, "dataset": ds, "scenario": sc, "family": fam}
        for v in VARIANTS:
            R = sim(Fs[v], Ft[v])
            for gm in G:
                if gm == 0.0 and v != "6":
                    continue
                r = all_metrics(base + gm * R, gt, perms)
                row[f"{v}|{gm}"] = (r["R_exp"] + r["F1_Hr"]) / 2
        out.append(row)
    return out


def keyed_and_label(job):
    """Per scenario: whether both sides have a unary key; gendet label counts on true and non-matching pairs."""
    regime, ds, sc, fam = job
    g_f = ground_truth_file(CORPUS[regime], ds, sc)
    _, rows, cols = read_matrix(HERE / "reference" / "seeds" / regime / "datatype_nanfix" / ds / sc / "seed.csv")
    gt = read_matrix(g_f)[0] > 0.5
    keyed = all(any(len(u) == 1 for u in profiles(regime, ds, sc, side)[1]) for side in ("source", "target"))
    _, ls = side_features(regime, ds, sc, "source", list(rows))
    _, lt = side_features(regime, ds, sc, "target", list(cols))
    same = np.array(ls, dtype=object)[:, None] == np.array(lt, dtype=object)[None, :]
    if same.shape != gt.shape:
        return ds, sc, fam, keyed, None, ls + lt
    return ds, sc, fam, keyed, (int((same & gt).sum()), int(gt.sum()), int((same & ~gt).sum()), int((~gt).sum())), ls + lt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("plain", "md5"), required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cs = cases_for(a.regime)
    sf = sf_rows(a.regime)
    seeds = sorted({k[0] for k in sf} - CONTROLS)
    seeds = [s for s in seeds if all((s, arm, d, c) in sf for arm in SF_ARMS for d, c, _ in cs)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(score_matcher, [(a.regime, s, cs) for s in seeds]) for r in part]
        side = list(pool.imap_unordered(keyed_and_label, [(a.regime, *c) for c in cs]))
    cover = defaultdict(int)
    for r in rows:
        cover[r["seed"]] += 1
    seeds = [s for s in seeds if cover[s] == len(cs)]
    rows = [r for r in rows if r["seed"] in seeds]
    out = HERE / "results" / a.regime
    with (out / "genroles_scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {(r["seed"], r["dataset"], r["scenario"]): r for r in rows}
    fams = defaultdict(list)
    for d, c, f in cs:
        fams[f].append((d, c))
    big = {f: v for f, v in fams.items() if len(v) >= 5}
    fam_of = {(d, c): f for d, c, f in cs}
    keyed = {(d, c): k for d, c, _, k, _, _ in side}

    def k_of(v, gm):
        return "6|0.0" if gm == 0.0 else f"{v}|{gm}"

    def macro(k, famset):
        return np.mean([np.mean([np.mean([by[(s, d, c)][k] for d, c in fams[f]]) for f in famset]) for s in seeds])

    def choose(v, train):
        sc_ = {gm: macro(k_of(v, gm), train) for gm in G}
        best = max(sc_.values())
        return min(gm for gm, x in sc_.items() if x >= best - 1e-12)

    choices = {v: {h: choose(v, [f for f in big if f != h]) for h in big} for v in VARIANTS}
    for v in VARIANTS:
        choices[v]["_other"] = choose(v, list(big))
    val = {f"R{v}": (lambda s, d, c, v=v: by[(s, d, c)][k_of(v, choices[v].get(fam_of[(d, c)], choices[v]["_other"]))])
           for v in VARIANTS}
    val["A_d"] = lambda s, d, c: by[(s, d, c)]["6|0.0"]
    for arm in SF_ARMS:
        val[f"SF {arm}"] = lambda s, d, c, arm=arm: sf[(s, arm, d, c)]
    rng = np.random.default_rng(SEED)

    def contrast(x, y, subset=None):
        per = {}
        for f, v in big.items():
            v2 = [dc for dc in v if subset is None or subset(dc)]
            if len(v2) >= 5:
                per[f] = np.array([np.mean([val[x](s, d, c) - val[y](s, d, c) for s in seeds]) for d, c in v2])
        if not per:
            return "– | –"
        est = float(np.mean([v.mean() for v in per.values()]))
        boots = np.mean([[rng.choice(v, size=v.size).mean() for _ in range(B)] for v in per.values()], axis=0)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        n = sum(v.size for v in per.values())
        return f"{est:+.4f} [{lo:+.4f}, {hi:+.4f}] | {len(per)} families, {n} scenarios"

    lines = [f"# X72 ({a.regime}): a genuine role term, FD roles weighted by gpdep (EXPLORATORY)", "",
             f"{len(seeds)} initial matchers, {len(cs)} scenarios; {sum(keyed.values())} keyed (both sides have a unary key). "
             "A = q+ + 5D + γ R_x; γ chosen leave-one-family-out from " + str(G) + ":", ""]
    for v, ch in choices.items():
        lines.append(f"- R{v}: " + "; ".join(f"{f} {g}" for f, g in ch.items()))
    lines += ["", "| contrast | J_tf [95% CI] | coverage |", "|---|---|---|"]
    pairs = [("Rg", "R6", None, ""), ("R13", "R6", None, ""), ("R13", "Rg", None, ""),
             ("Rg", "R6", lambda dc: keyed[dc], " (keyed)"), ("Rg", "R6", lambda dc: not keyed[dc], " (not keyed)")]
    for v in ("R6", "Rg", "R13"):
        pairs.append((v, "A_d", None, ""))
        for arm in ("melnik", "ucc_top1", "ucc1_fd_topk"):
            pairs.append((v, f"SF {arm}", None, ""))
    for x, y, sub, tag in pairs:
        lines.append(f"| {x} − {y}{tag} | {contrast(x, y, sub)} |")
    # gendet label (X75 style): pooled within family, family mean
    lab_f = defaultdict(lambda: np.zeros(4))
    allv = []
    for d, c, f, _, cnt, ls in side:
        allv += ls
        if cnt is not None and f in big:
            lab_f[f] += np.array(cnt, dtype=float)
    p1 = np.mean([v[0] / v[1] for v in lab_f.values()])
    p0 = np.mean([v[2] / v[3] for v in lab_f.values()])
    lift = np.mean([(v[0] / v[1]) / (v[2] / v[3]) for v in lab_f.values()])
    from collections import Counter
    cnts = Counter(allv)
    tot = sum(cnts.values())
    H = -sum(x / tot * math.log2(x / tot) for x in cnts.values())
    lines += ["", f"gendet label (none / weak / strong; X75 measures): p1 {p1:.3f}, p0 {p0:.3f}, lift {lift:.2f}, H {H:.2f} bits; "
              f"distribution {dict(cnts)}", ""]
    (out / "genroles_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
