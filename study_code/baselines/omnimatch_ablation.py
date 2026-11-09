"""OMNIMATCH_ABLATION.md: driver for the OmniMatch information-matched ablation (EXPLORATORY).

Runs the port of the authors' release (code/schematch/.../schematch_py/omnimatch_release.py) per scenario on an imported
corpus tree and writes one score matrix per (arm, seed, scenario) as CSV, rows = source columns, columns = target
columns in header order (the ground-truth layout). Arms so far:
    F   OmniMatch as released (R-GCN over the top-3 similarity edges)
    N   the same with no edges (self-loops only): both columns' statistics, same training
The arms A (N + one-step edge term) and RF (the authors' random-forest baseline) are added after step 0.

    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_ablation.py run --arms F --seeds 20261015 --limit 5
    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_ablation.py run --arms F --seeds 1 2 3 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PM = HERE.parent / "code" / "schematch" / "src" / "main" / "resources" / "python_matchers"
sys.path.insert(0, str(PM))
sys.path.insert(0, str(HERE))
from schematch_py import omnimatch_release as om  # noqa: E402
import semstruct_run as sr  # noqa: E402  (case lists: Valentine protocol or an imported tree)

OUT = HERE / "reports" / "omnimatch_ablation_20261027"
OPTIONS = {"vectors": str(HERE / "third_party" / "fasttext_cc" / "cc.en.300.bin"), "dim": 50,
           "vectors_cache": str(HERE / "profile-cache" / "omnimatch" / "cc_en_300_dim50.npz")}


def header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        return next(csv.reader(fh), None) or []


def context(case: dict) -> dict | None:
    paths = sr.case_paths(case, "plain")
    if not paths:
        return None
    src, tgt, gt = paths
    folder = Path(case["plain"])
    return {"rt": str(folder.parent.parent), "ds": folder.parent.name, "sc": folder.name,
            "rows": [f"{src.stem}.{h}" for h in header(src)], "cols": [f"{tgt.stem}.{h}" for h in header(tgt)],
            "gt": gt}


@contextmanager
def arm_patch(arm: str):
    """N: no edges anywhere (training and test graphs), so only the self-loop weights act."""
    if arm == "F":
        yield
        return
    if arm == "N":
        original = om.topk_edges
        om.topk_edges = lambda pairs, feats, k: ([], [], [])
        try:
            yield
        finally:
            om.topk_edges = original
        return
    raise ValueError(arm)


GAMMAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def _edge_counts(pairs, feats, k) -> dict:
    """For each unordered node pair, under how many of the five similarities one is among the other's top-k."""
    src, dst, typ = om.topk_edges(pairs, feats, k)
    rels = {}
    for s, d, t in zip(src, dst, typ):
        rels.setdefault(frozenset((s, d)), set()).add(t)
    return {p: len(r) for p, r in rels.items()}


def run_all(ctx: dict, options: dict, arms) -> dict:
    """om.omnimatch, recomposed from the port's own helpers so that every arm shares one generated training set.
    F must equal om.omnimatch bit for bit (checked by `verify`). Returns {arm: |rows| x |cols| matrix} and, for A, the
    chosen gamma under key 'A_gamma'."""
    import torch
    options = dict(options or {})
    vecs = om._vectors_from(options)
    splitter = (om._WordSplitter(om._wordlist_file(str(options["wordlist"]))) if options.get("wordlist")
                else vecs.splitter())
    js_values = str(options.get("js_tokens", "header")) == "values"
    seed = int(options.get("seed", 20261015))
    k = int(options.get("k", 3))
    R = om._Rand(seed)
    max_rows = int(options.get("max_rows", om.x105.MAXROWS) or 0)
    tables = om._read_tables(ctx, "source", max_rows) + om._read_tables(ctx, "target", max_rows)
    gen = om.generate_pairs([(name, df) for name, df, _ in tables], options, R)
    prof, colobj, join_by_pair, nonjoin_by_pair = [], [], [], []
    for name, it, t0, t1, common in gen:
        side_ids = []
        for t in (t0, t1):
            profiles = om.table_profiles(t)
            idx = {}
            for c in t.columns:
                idx[c] = len(prof)
                prof.append(profiles[c])
                colobj.append(om._Column(t[c], c, vecs, splitter, js_values))
            side_ids.append(idx)
        i0, i1 = side_ids
        joins = [(i0[c], i1[c]) for c in common if c in i0 and c in i1]
        js = set(joins)
        join_by_pair.append(joins)
        nonjoin_by_pair.append([(a, b) for a in i0.values() for b in i1.values() if (a, b) not in js])
    tr_pairs = [p for lst in join_by_pair for p in lst] + [p for lst in nonjoin_by_pair for p in lst]
    tr_feats = [om.pair_features(colobj[a], colobj[b]) for a, b in tr_pairs]
    tr_edges = om.topk_edges(tr_pairs, tr_feats, k)
    n_sources = int(options.get("number_of_sources", 20))
    n_datasets = int(options.get("number_of_datasets", 2))
    names = list(dict.fromkeys(g[0] for g in gen))
    chosen = set(R.sample(names, min(n_sources, len(names))))
    use = [gi for gi, g in enumerate(gen) if g[0] in chosen and 2 * g[1] + 1 < n_datasets]
    J = [p for gi in use for p in join_by_pair[gi]]
    NJ = [p for gi in use for p in nonjoin_by_pair[gi]]
    A_, P_, N_ = om._triplets(J, NJ)
    S = {lab: (name, df, c) for name, df, labs in tables if name.startswith("source/") for lab, (c, _) in labs.items()}
    T = {lab: (name, df, c) for name, df, labs in tables if name.startswith("target/") for lab, (c, _) in labs.items()}
    tprof_cache = {}

    def test_node(lab, side):
        hit = side.get(lab)
        if hit is None:
            s = om.pd.Series([np.nan])
            return om._column_profile(s.astype(str)), om._Column(s, lab.split(".", 1)[-1], vecs, splitter, js_values)
        name, df, c = hit
        if name not in tprof_cache:
            tprof_cache[name] = om.table_profiles(df)
        return tprof_cache[name][c], om._Column(df[c], c, vecs, splitter, js_values)

    rows, cols_ = list(ctx["rows"]), list(ctx["cols"])
    te = [test_node(l, S) for l in rows] + [test_node(l, T) for l in cols_]
    ns, nt = len(rows), len(cols_)
    te_pairs = [(i, ns + j) for i in range(ns) for j in range(nt)]
    te_feats = [om.pair_features(te[a][1], te[b][1]) for a, b in te_pairs]
    te_edges = om.topk_edges(te_pairs, te_feats, k)
    test_prof = [t[0] for t in te]
    out = {}

    def train(edges, te_e):
        """om._train_and_embed, also returning the trained net and the training-node matrix (for gamma)."""
        was_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                X = om._node_matrix(prof) if prof else None
                G = om._graph(*edges)
                net = om._model(len(test_prof[0]), int(options.get("dimension", 256)), om.N_FEATURES)
                opt = torch.optim.Adam(net.parameters(), lr=float(options.get("lr", 0.001)), weight_decay=0)
                loss_fn = torch.nn.TripletMarginLoss(margin=float(options.get("margin", 0.5)),
                                                     p=float(options.get("norm", 2)))
                if A_ and X is not None:
                    a, p, n = (torch.tensor(x, dtype=torch.long) for x in (A_, P_, N_))
                    for _ in range(int(options.get("epochs", 30))):
                        opt.zero_grad()
                        h = net(X, G)
                        loss = loss_fn(h[a], h[p], h[n])
                        loss.backward()
                        opt.step()
                with torch.no_grad():
                    Zte = net(om._node_matrix(test_prof), om._graph(*te_e))
                    Ztr = net(X, G) if X is not None else None
        finally:
            torch.use_deterministic_algorithms(was_det)
        D = torch.cdist(Zte[:ns].double(), Zte[ns:].double()).numpy()
        return np.clip(np.nan_to_num(1.0 - D / 2.0, nan=0.0), 0.0, 1.0), Ztr

    empty = ([], [], [])
    if "F" in arms:
        out["F"], _ = train(tr_edges, te_edges)
    if "N" in arms or "A" in arms:
        out["N"], Ztr = train(empty, empty)
        if "A" in arms:
            # one-step edge term: gamma * (number of similarities under which s, t are among each other's top-k) / 5;
            # gamma chosen on the generated training triplets only (share of triplets ranked correctly)
            c_tr = _edge_counts(tr_pairs, tr_feats, k)
            c_te = _edge_counts(te_pairs, te_feats, k)
            best, best_g = -1.0, 0.0
            if A_ and Ztr is not None:
                import torch as _t
                def s_tr(u, v):
                    return 1.0 - float(_t.dist(Ztr[u].double(), Ztr[v].double())) / 2.0
                for g in GAMMAS:
                    ok = sum((s_tr(a, p) + g * c_tr.get(frozenset((a, p)), 0) / 5)
                             > (s_tr(a, n) + g * c_tr.get(frozenset((a, n)), 0) / 5) for a, p, n in zip(A_, P_, N_))
                    if ok / len(A_) > best + 1e-12:
                        best, best_g = ok / len(A_), g
            C = np.array([[c_te.get(frozenset((i, ns + j)), 0) for j in range(nt)] for i in range(ns)], dtype=float)
            out["A"] = out["N"] + best_g * C / 5
            out["A_gamma"] = best_g
    if "RF" in arms:
        # the release's rf_predictor: a random forest on the five pairwise similarity values of the generated pairs
        from sklearn.ensemble import RandomForestClassifier
        jset = {p for lst in join_by_pair for p in lst}
        y = np.array([p in jset for p in tr_pairs], dtype=int)
        if y.min() == y.max():
            out["RF"] = np.zeros((ns, nt))
        else:
            clf = RandomForestClassifier(random_state=seed).fit(np.array(tr_feats), y)
            P = clf.predict_proba(np.array(te_feats))[:, 1]
            out["RF"] = P.reshape(ns, nt)
    return out


def write_matrix(path: Path, M: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, M, delimiter=",", fmt="%.10g")


def command_run(args) -> int:
    if args.cases_root:
        sr.CASES_ROOT = Path(args.cases_root)
    cases = sr.load_cases()
    if args.cohort:
        cases = [c for c in cases if c.get("cohort") == args.cohort]
    if args.limit:
        rng = np.random.default_rng(20261027)                  # a fixed, spread-out subset (not the first families)
        cases = [cases[i] for i in sorted(rng.choice(len(cases), args.limit, replace=False))]
    Path(OPTIONS["vectors_cache"]).parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT
    timing = []
    for case in cases:
        ctx = context(case)
        if ctx is None:
            continue
        for seed in args.seeds:
            todo = [a for a in args.arms
                    if args.force or not (out / "matrices" / f"{a}__s{seed}" / f"{case['key']}.csv").exists()]
            if not todo:
                continue
            t0 = time.perf_counter()
            res = run_all({k: ctx[k] for k in ("rt", "ds", "sc", "rows", "cols")}, dict(OPTIONS, seed=seed), todo)
            dt = time.perf_counter() - t0
            for arm in todo:
                write_matrix(out / "matrices" / f"{arm}__s{seed}" / f"{case['key']}.csv", res[arm])
            timing.append({"case": case["id"], "key": case["key"], "arms": todo, "seed": seed, "seconds": dt,
                           "A_gamma": res.get("A_gamma")})
            print(f"{case['id']}  {'/'.join(todo)} s{seed}  {dt:.1f}s  gamma {res.get('A_gamma')}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"timing_{args.tag}.json").open("w", encoding="utf-8") as fh:
        json.dump(timing, fh, indent=1)
    return 0


def command_score(args) -> int:
    """Tie-free metrics (sf-matcher-study score_r7.metrics, the SF paper's J_tf) for every written matrix."""
    sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "layer3b"))
    sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "tuning"))
    from score_r7 import metrics as tf_metrics
    from score_screen_metrics import permutations
    if args.cases_root:
        sr.CASES_ROOT = Path(args.cases_root)
    out = Path(args.out) if args.out else OUT
    cases = {c["key"]: c for c in sr.load_cases()}
    rows = []
    for d in sorted((out / "matrices").iterdir()):
        arm, seed = d.name.split("__s")
        for f in sorted(d.glob("*.csv")):
            case = cases.get(f.stem)
            if case is None:
                continue
            gt = sr.read_ground_truth(sr.case_paths(case, "plain")[2])
            S = np.loadtxt(f, delimiter=",", ndmin=2)
            if S.shape != gt.shape:
                continue
            m = tf_metrics(S, gt, permutations(gt.shape, f"{case['dataset']}/{case['scenario']}"))
            rows.append({"key": f.stem, "family": case["family"], "cohort": case.get("cohort", "all"), "arm": arm,
                         "seed": int(seed), "J": m["J"], "R_exp": m["R_exp"], "F1_Hr": m["F1_Hr"]})
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} matrices scored -> {out / 'scores.csv'}")
    return 0


CONTRASTS = [("F", "N", "gate: the edges carry something (F over no edges)"),
             ("A", "F", "information-matched: one-step edge term minus propagation"),
             ("RF", "F", "the authors' random forest on the pair similarities minus OmniMatch"),
             ("A", "N", "the one-step edge term over no edges")]


def command_report(args) -> int:
    """Seed-averaged (seeds 1-3 per scenario, the step-0 rule) paired contrasts, macro over families with >= 5 scenarios,
    95% bootstrap over scenarios stratified by family (2,000), or over clusters of scenarios sharing a table when
    --cases-root is given; cohorts remaining338 and all on Valentine."""
    import pandas as pd
    out = Path(args.out) if args.out else OUT
    d = pd.read_csv(out / "scores.csv")
    d = d[d.seed.isin(args.seeds)]
    g = d.groupby(["key", "family", "cohort", "arm"])[["J", "R_exp", "F1_Hr"]].mean().reset_index()
    n_seeds = d.groupby(["key", "arm"]).seed.nunique()
    wide = {m: g.pivot_table(index=["key", "family", "cohort"], columns="arm", values=m) for m in ("J", "R_exp", "F1_Hr")}
    complete = [k for k in wide["J"].index if all(n_seeds.get((k[0], a), 0) == len(args.seeds) for a in ("F", "N", "A", "RF"))]
    rng = np.random.default_rng(20261027)
    lines = [f"# OmniMatch information-matched ablation ({args.label})", "",
             f"EXPLORATORY (OMNIMATCH_ABLATION.md). Each arm averaged over seeds {args.seeds} per scenario; paired "
             "differences; macro over families with >= 5 scenarios; 95% bootstrap (2,000).", ""]
    clusters = None
    if args.cases_root:
        sr.CASES_ROOT = Path(args.cases_root)
        cases = {c["key"]: c for c in sr.load_cases()}
        import hashlib
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for k in complete:
            p = sr.case_paths(cases[k[0]], "plain")
            for t in ("s" + hashlib.sha256(p[0].read_bytes()).hexdigest(), "t" + hashlib.sha256(p[1].read_bytes()).hexdigest()):
                parent[find(k[0])] = find(t)
        clusters = {k[0]: find(k[0]) for k in complete}
    cohorts = ["all"] if args.cases_root else [args.cohort, "all"]
    for cohort in cohorts:
        sel = [k for k in complete if cohort == "all" or k[2] == cohort]
        fam = {}
        for i, k in enumerate(sel):
            fam.setdefault(k[1], []).append(i)
        fams = {f: np.array(ix) for f, ix in fam.items() if len(ix) >= 5}
        lines += [f"## {cohort}: {len(sel)} scenarios ({', '.join(f'{f} {len(ix)}' for f, ix in fams.items())})", "",
                  "| arm | J_tf | R_exp | F1_Hr |", "|---|---|---|---|"]
        for arm in ("F", "N", "A", "RF"):
            v = {m: np.mean([wide[m].loc[[sel[i] for i in ix], arm].mean() for ix in fams.values()]) for m in wide}
            lines.append(f"| {arm} | {v['J']:.3f} | {v['R_exp']:.3f} | {v['F1_Hr']:.3f} |")
        lines += ["", "| contrast | meaning | J_tf [95%] | R_exp | F1_Hr |", "|---|---|---|---|---|"]
        for a, b, meaning in CONTRASTS:
            res = {}
            for m in wide:
                dd = (wide[m].loc[sel, a] - wide[m].loc[sel, b]).to_numpy()
                est = np.mean([dd[ix].mean() for ix in fams.values()])
                if clusters is not None:
                    cl = {}
                    for i, k in enumerate(sel):
                        cl.setdefault(clusters[k[0]], []).append(i)
                    groups = [np.array(v) for v in cl.values()]
                    boots = [dd[np.concatenate([groups[c] for c in rng.choice(len(groups), len(groups))])].mean()
                             for _ in range(2000)]
                else:
                    boots = [np.mean([dd[rng.choice(ix, len(ix))].mean() for ix in fams.values()]) for _ in range(2000)]
                res[m] = (est, np.percentile(boots, 2.5), np.percentile(boots, 97.5))
            j = res["J"]
            lines.append(f"| {a} - {b} | {meaning} | {j[0]:+.3f} [{j[1]:+.3f}, {j[2]:+.3f}] | {res['R_exp'][0]:+.3f} | "
                         f"{res['F1_Hr'][0]:+.3f} |")
        lines.append("")
    if clusters is not None:
        lines.insert(4, f"{len(set(clusters.values()))} clusters of scenarios sharing a table; bootstrap over clusters.")
    (out / f"report_{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def command_verify(args) -> int:
    """run_all's F must equal om.omnimatch bit for bit."""
    cases = sr.load_cases()
    rng = np.random.default_rng(20261027)
    for case in [cases[i] for i in sorted(rng.choice(len(cases), args.n, replace=False))]:
        ctx = context(case)
        c = {k: ctx[k] for k in ("rt", "ds", "sc", "rows", "cols")}
        ref = om.omnimatch(c, dict(OPTIONS, seed=7))
        mine = run_all(c, dict(OPTIONS, seed=7), ["F"])["F"]
        print(case["id"], "identical" if np.array_equal(ref, mine) else f"DIFF max {np.abs(ref - mine).max():.3g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--cases-root", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(func=command_score)
    rp = sub.add_parser("report")
    rp.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    rp.add_argument("--cases-root", default=None)
    rp.add_argument("--out", default=None)
    rp.add_argument("--label", default="valentine_plain")
    rp.add_argument("--cohort", default="remaining338",
                    help="Amendment 549: remaining505 with SEMSTRUCT_RUN=reports/valentine549_protocol")
    rp.set_defaults(func=command_report)
    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=3)
    v.set_defaults(func=command_verify)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["F"], choices=["F", "N", "A", "RF"])
    r.add_argument("--seeds", nargs="+", type=int, default=[20261015])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--cohort", default=None)
    r.add_argument("--cases-root", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--tag", default="run")
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=command_run)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
