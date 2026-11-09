"""Family 7 (format / pattern) as graph information: a pattern-class node per column, like a datatype node.

Class of a column: its dominant value shape (PatternProfileMatcher.shape: upper A, lower a, digit 9, whitespace
kept, other characters kept, runs collapsed, at most 32 characters) if at least half of its non-empty values have
it, else MIXED. Written as "@class:<shape>" edges with label `pattern` for extra-edges channel (dev build).
Control `shuffled`: the same classes permuted across the columns of each table (seeded by dataset/scenario/side).

Proposition 1, extended: with exactly one class node per column, SF(melnik + classes) = a·q + b(datatype pair,
class pair). Check: zero within-block inversions, blocks read from SF's output on the constant matrix under the
same arm (as check_prop1.py --classes constant does for melnik).

    python layer3b/f7.py classes | write | run --workers 4 | score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
from r7_build import cases  # noqa: E402
from r8_smoke import classpath, latest, make, outputs, run  # noqa: E402
from r8_score import blocks, met  # noqa: E402
from smx.adapters import read_matrix  # noqa: E402
from score_seeds import CORPUS, cases_for, ground_truth_file, one  # noqa: E402
from score_screen_metrics import permutations  # noqa: E402
from explore_posthoc import tiebreak  # noqa: E402

csv.field_size_limit(1 << 30)
# The same runner and scorer serve family 13 (layer3b/f13.py): SMX_L3B_FAMILY=f13 selects its edge files,
# configurations and results. An environment variable, because Windows pool workers re-import this module.
FAMILY = os.environ.get("SMX_L3B_FAMILY", "f7")
REAL = {"f7": "shape", "f13": "semtype"}[FAMILY]
SUFFIX = f"{FAMILY}_screen"
CLASSES = HERE / "classes_dev"
EDGES = STUDY / "data" / f"{FAMILY}_plain" / "screen"
RES = HERE / "results"
SEEDS = STUDY / "reference" / "seeds" / "plain"
SCREEN = STUDY / "tuning" / "corpus" / "plain" / "screen"
MIN_FAMILY = 5


def shape(v: str, max_length: int = 32) -> str:
    out, last = [], None
    for c in v:
        m = "A" if c.isupper() else "a" if c.islower() else "9" if c.isdigit() else " " if c.isspace() else c
        if out and m == last:
            continue
        out.append(m)
        last = m
        if len(out) >= max_length:
            break
    return "".join(out)


def column_classes(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        counts = [Counter() for _ in header]
        for row in r:
            for i, v in enumerate(row[: len(header)]):
                if v != "":
                    counts[i][shape(v)] += 1
    classes = []
    for c in counts:
        total = sum(c.values())
        top, n = c.most_common(1)[0] if c else ("", 0)
        classes.append(top if total and n / total >= 0.5 and top.strip() else "MIXED")
    return header, classes


def make_classes() -> None:
    diag = []
    for dataset, name, scen in cases("screen", "plain"):
        for side in ("source", "target"):
            header, cls = column_classes(scen / side / f"{side}.csv")
            rng = random.Random(hashlib.sha256(f"{dataset}/{name}/{side}".encode()).hexdigest())
            shuffled = cls[:]
            rng.shuffle(shuffled)
            for v, cl in (("shape", cls), ("shuffled", shuffled)):
                p = EDGES / v / dataset / name / f"{side}.tsv"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("".join(f"pattern\t{side}\t{h}\t@class:{c}\t1.0\n" for h, c in zip(header, cl)),
                             encoding="utf-8", newline="\n")
            diag.append({"dataset": dataset, "scenario": name, "side": side, "columns": len(header),
                         "classes": len(set(cls)), "mixed": cls.count("MIXED")})
    with (HERE / "f7_plain_screen_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(diag[0]))
        w.writeheader()
        w.writerows(diag)
    print(f"{len(diag)} tables; mean columns {mean(d['columns'] for d in diag):.1f}, mean classes "
          f"{mean(d['classes'] for d in diag):.1f}, mean MIXED columns {mean(d['mixed'] for d in diag):.1f} -> {EDGES}")


def write(seeds):
    root = HERE / "configs" / SUFFIX
    n = 0
    for seed in seeds + ["constant"]:
        for arm in (("melnik", "fd") if seed != "constant" else ("melnik",)):
            for v in (REAL, "shuffled"):
                make(f"{seed}__{arm}", f"{seed}__{arm}__{v}", root, RES / SUFFIX,
                     {"smx.sf.extraEdgesDir": (EDGES / v).as_posix()})
                n += 1
    make("levenshtein__melnik", "levenshtein__melnik__devnone", root, RES / SUFFIX, {})
    print(f"{n + 1} configurations -> {root}")


def run_all(workers, xmx):
    root = HERE / "configs" / SUFFIX
    todo = [c for c in sorted(root.iterdir(), key=lambda c: ("constant" not in c.name and "dev" not in c.name, c.name))
            if not (RES / SUFFIX / c.name / "DONE").is_file()]
    print(f"{len(todo)} to run, {workers} workers", flush=True)
    deps, failed, t0 = classpath(), [], time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(run, c, RES / SUFFIX / c.name, deps, xmx, str(CLASSES)) for c in todo]
        for k, f in enumerate(as_completed(futs), 1):
            label, code, el = f.result()
            failed += [label] if code else []
            print(f"[{k}/{len(todo)}] {label} {'ok' if code == 0 else 'FAILED'} ({el:.0f}s; {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    print(f"done; {len(failed)} failed {failed}")
    return 1 if failed else 0


def inversions(q, S, block):
    s, qq = S.ravel(), q.ravel()
    inv = pairs = 0
    for b in np.unique(block):
        idx = np.where(block == b)[0]
        if len(idx) < 2:
            continue
        dq = qq[idx][:, None] - qq[idx][None, :]
        dS = s[idx][:, None] - s[idx][None, :]
        inv += int(((dq > 1e-9) & (dS < -1e-6)).sum())
        pairs += int((dq > 1e-9).sum())
    return inv, pairs


def true_blocks(b_dt: np.ndarray, variant: str, ds: str, sn: str, shape: tuple[int, int]) -> np.ndarray:
    """Refined blocks by definition: the datatype block (SF(constant) under melnik) x the class pair from the
    edge files. The proxy (SF(constant) under the arm) can merge distinct class pairs with equal output."""
    cls = {}
    for side in ("source", "target"):
        lines = (EDGES / variant / ds / sn / f"{side}.tsv").read_text(encoding="utf-8").splitlines()
        cls[side] = [l.split("	")[3] for l in lines]
    n, m = shape
    keys = [f"{b_dt[i * m + j]}|{cls['source'][i]}|{cls['target'][j]}" for i in range(n) for j in range(m)]
    return np.unique(np.array(keys), return_inverse=True)[1]


def score_one(job):
    seed, arm, cases_ = job
    c_dt = latest(RES / "r7_screen" / "constant__melnik")
    c_cls = {v: latest(RES / SUFFIX / f"constant__melnik__{v}") for v in (REAL, "shuffled")}
    runs = {"none": latest(RES / "r7_screen" / f"{seed}__{arm}"),
            REAL: latest(RES / SUFFIX / f"{seed}__{arm}__{REAL}"),
            "shuffled": latest(RES / SUFFIX / f"{seed}__{arm}__shuffled")}
    out = []
    for ds, sn, fam in cases_:
        gt = read_matrix(ground_truth_file(CORPUS["plain"], ds, sn))[0] > 0.5
        q = read_matrix(SEEDS / seed / ds / sn / "seed.csv")[0]
        perms = permutations(gt.shape, f"{ds}/{sn}")
        mq = met(q, gt, perms)
        b_dt = blocks(read_matrix(one(c_dt / ds / sn / "_outputs" / "MatchingStepLine2"))[0])
        for v, r in runs.items():
            S = read_matrix(one(r / ds / sn / "_outputs" / "MatchingStepLine2"))[0]
            T, _ = tiebreak(q, S)
            row = {"seed": seed, "arm": arm, "variant": v, "dataset": ds, "scenario": sn, "family": fam,
                   **{f"q_{m}": x for m, x in mq.items()}, **{f"sf_{m}": x for m, x in met(S, gt, perms).items()},
                   "tb_J": met(T, gt, perms)["J"]}
            row["inv_dt"], row["pairs_dt"] = inversions(q, S, b_dt)
            if v in c_cls:
                b_cl = blocks(read_matrix(one(c_cls[v] / ds / sn / "_outputs" / "MatchingStepLine2"))[0])
                row["inv_cls"], row["pairs_cls"] = inversions(q, S, b_cl)
                row["inv_true"], row["pairs_true"] = inversions(q, S, true_blocks(b_dt, v, ds, sn, q.shape))
                row["blocks_cls"], row["blocks_dt"] = len(np.unique(b_cl)), len(np.unique(b_dt))
            out.append(row)
    return out


def macro(rows, key):
    by = defaultdict(list)
    for r in rows:
        by[r["family"]].append(r[key])
    fams = [f for f, v in by.items() if len(v) >= MIN_FAMILY]
    return mean(mean(by[f]) for f in fams)


def score():
    sys.stdout.reconfigure(encoding="utf-8")
    res = RES / SUFFIX
    refs = {(d, s): M for d, s, M in outputs(latest(RES / "r7_screen" / "levenshtein__melnik"))}
    worst = max(float(np.abs(M - refs[(d, s)]).max()) for d, s, M in outputs(latest(res / "levenshtein__melnik__devnone")))
    screen = {(d.name, s.name) for d in SCREEN.iterdir() if d.is_dir() for s in d.iterdir()}
    cases_ = [c for c in cases_for("plain") if (c[0], c[1]) in screen]
    seeds = sorted({p.name.split("__")[0] for p in res.iterdir() if "dev" not in p.name and not p.name.startswith("constant")})
    with Pool(6) as pool:
        rows = [r for part in pool.imap_unordered(score_one, [(s, a, cases_) for s in seeds for a in ("melnik", "fd")])
                for r in part]
    out = RES / f"{SUFFIX}_scores"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    idx = {(r["seed"], r["arm"], r["variant"], r["dataset"], r["scenario"]): r for r in rows}

    def eff(arm, a, b, m="J"):
        vals = []
        for s in seeds:
            sel = [{"family": r["family"], "_d": r[f"sf_{m}"] - (r[f"q_{m}"] if b == "q" else
                    idx[(s, arm, b, r["dataset"], r["scenario"])][f"sf_{m}"])}
                   for r in rows if r["seed"] == s and r["arm"] == arm and r["variant"] == a]
            vals.append(macro(sel, "_d"))
        return mean(vals)

    title = {"f7": "Family 7: pattern-class nodes", "f13": "Family 13: semantic-type nodes (rule-based typer)"}[FAMILY]
    lines = [f"# {title} (plain, 60 screen scenarios; exploratory)", "",
             f"Dev build with the channel off vs R7 smoke `melnik`: max |difference| {worst}"
             + (" — identical." if worst == 0 else " — DIFFERENT."), "",
             f"Matchers: {', '.join(seeds)}. Macro over families with ≥ {MIN_FAMILY} scenarios, mean over matchers.", ""]
    for arm in ("melnik", "fd"):
        lines += [f"## Arm `{arm}`", "",
                  "| variant | SF − q J | R_exp | F1_Hr | vs no classes J | R_exp | F1_Hr | tb − q J | inversions (datatype blocks) | inversions (refined blocks) | blocks dt → refined |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in ("none", REAL, "shuffled"):
            rs = [r for r in rows if r["arm"] == arm and r["variant"] == v]
            tb = mean(macro([{"family": r["family"], "_d": r["tb_J"] - r["q_J"]} for r in rs if r["seed"] == s], "_d") for s in seeds)
            i_dt = sum(r["inv_dt"] for r in rs) / max(1, sum(r["pairs_dt"] for r in rs))
            i_cl = (f"{sum(r['inv_cls'] for r in rs)} of {sum(r['pairs_cls'] for r in rs)} (true blocks: "
                    f"{sum(r['inv_true'] for r in rs)} of {sum(r['pairs_true'] for r in rs)})" if "inv_cls" in rs[0] else "")
            bl = (f"{mean(r['blocks_dt'] for r in rs):.1f} → {mean(r['blocks_cls'] for r in rs):.1f}" if "blocks_cls" in rs[0] else "")
            vs = "" if v == "none" else f"{eff(arm, v, 'none'):+.4f} | {eff(arm, v, 'none', 'R_exp'):+.4f} | {eff(arm, v, 'none', 'F1_Hr'):+.4f}"
            lines.append(f"| {v} | {eff(arm, v, 'q'):+.4f} | {eff(arm, v, 'q', 'R_exp'):+.4f} | {eff(arm, v, 'q', 'F1_Hr'):+.4f} | "
                         f"{vs if vs else '0 | 0 | 0'} | {tb:+.4f} | {i_dt:.4f} | {i_cl} | {bl} |")
        lines.append("")
    lines.append("Extended Proposition 1 (melnik + classes): zero inversions are predicted within the refined blocks.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("classes", "write", "run", "score"))
    ap.add_argument("--seeds", default="levenshtein,prefix_suffix,pmf,role_gcov")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--xmx", default="5g")
    a = ap.parse_args()
    if a.cmd == "classes":
        make_classes()
        return 0
    if a.cmd == "write":
        write(a.seeds.split(","))
        return 0
    if a.cmd == "run":
        return run_all(a.workers, a.xmx)
    return score()


if __name__ == "__main__":
    sys.exit(main())
