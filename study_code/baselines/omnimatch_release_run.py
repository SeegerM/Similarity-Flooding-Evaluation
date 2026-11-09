"""Drive the OFFICIAL OmniMatch release (amazon-science/omnimatch @ 515f8ce, harness/third_party/omnimatch, CC BY-NC 4.0,
research use) per Valentine scenario and write score matrices for comparison with our port
(code/schematch/.../schematch_py/omnimatch_release.py). Nothing in third_party/omnimatch is modified: its modules are
imported from the original files (no bytecode written) and driven from here, replicating the release scripts' __main__
blocks call for call. Every adaptation is listed in reports/omnimatch_release_20261027/CHANGES.md.

Release environment (Python 3.8, torch 1.7.0, dgl 0.5.3):
    harness/.venv-omnimatch-release/Scripts/python.exe harness/omnimatch_release_run.py reduce
    harness/.venv-omnimatch-release/Scripts/python.exe harness/omnimatch_release_run.py run --seeds 1 2 3 --limit 20
Scoring and comparison (newer numpy/scipy):
    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_release_run.py score
    harness/.venv-bdi/Scripts/python.exe -I harness/omnimatch_release_run.py compare
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
import types
from collections import OrderedDict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"                                      # shared PC: at most 4 threads
sys.dont_write_bytecode = True                                # never write __pycache__ into third_party/omnimatch

HERE = Path(__file__).resolve().parent
RELEASE = HERE / "third_party" / "omnimatch"
OUT = HERE / "reports" / "omnimatch_release_20261027"
FT300 = HERE / "third_party" / "fasttext_cc" / "cc.en.300.bin"
FT50 = OUT / "cc.en.50.bin"
# OMNIMATCH_ABLATION.md Amendment 549: OMNIMATCH_PROTOCOL=reports/valentine549_protocol/protocol.json (with run --limit 0
# --only <keys> --out reports/omnimatch_release_549); unset = the 382 protocol, unchanged
PROTOCOL = Path(os.environ.get("OMNIMATCH_PROTOCOL", HERE / "reports" / "audited_md5_fused_20260904" / "protocol.json"))
if not PROTOCOL.is_absolute():
    PROTOCOL = HERE / PROTOCOL
PORT = HERE / "reports" / "omnimatch_ablation_20261027"

FEATURES = ["jaccard_frequent", "value_embeddings", "value_distribution", "jaccard_containment"]  # config order
BENCHMARK = "valentine"            # anything but city_government/culture_recreation -> '<stem>_<i>.csv' naming
PARAMS = dict(graph_construction="topk", k=3, dimension=256, epochs=30, learning_rate=0.001, margin=0.5, norm=2,
              number_of_datasets=2, number_of_sources=20)
GEN = dict(num_iterations=1, common_cols_l=1, common_cols_r=6, common_cols_step=1, overlaps_l=0.1, overlaps_r=0.9,
           overlaps_step=0.1, typo_prc=10, cov_prc=100)
MODELS = ("rgcn_margin", "rgcn_margin_rowmajor", "rgcn_margin_noedges", "mlp_cross_entropy", "rf")


# ------------------------------------------------------------------------------------------------ case list

def load_cases(limit=0):
    cases = json.loads(PROTOCOL.read_text(encoding="utf-8"))["cases"]
    if limit:
        import numpy as np
        rng = np.random.default_rng(20261027)                 # exactly as omnimatch_ablation.py command_run --limit
        cases = [cases[i] for i in sorted(rng.choice(len(cases), limit, replace=False))]
    return cases


def case_files(case):
    folder = Path(case["plain"])
    gt = sorted((folder / "ground_truth").glob("*.csv"))
    src = sorted((folder / "source").glob("*.csv"))
    tgt = sorted((folder / "target").glob("*.csv"))
    if not (gt and src and tgt):
        return None
    return src[0], tgt[0], gt[0]


def read_ground_truth(path):
    """As semstruct_run.read_ground_truth: 0/1 matrix, rows = source columns, cols = target columns, no header."""
    import numpy as np
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if row and any(cell.strip() for cell in row):
                rows.append([float(cell) if cell.strip() else 0.0 for cell in row])
    return np.array(rows) > 0.5


# ----------------------------------------------------------------------------------- release import + shims

_FT_CACHE = {}


def import_release():
    """Import the release modules from the original files, with the minimal shims recorded in CHANGES.md."""
    src = str(RELEASE / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    # (1) unidecode is not installed in the release venv (no installs allowed): NFKD + ASCII drop, identity on ASCII.
    if "unidecode" not in sys.modules:
        import unicodedata
        m = types.ModuleType("unidecode")
        m.unidecode = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        sys.modules["unidecode"] = m
    # (2) featurizer imports multiprocess_tools, which the release does not ship; only used with in_parallel=True,
    #     which the release never sets.
    if "multiprocess_tools" not in sys.modules:
        m = types.ModuleType("multiprocess_tools")

        def _missing(*a, **k):
            raise NotImplementedError("multiprocess_tools is not part of the release")
        m.run_multithread_jaccard_similarity = _missing
        sys.modules["multiprocess_tools"] = m
    import torch
    torch.set_num_threads(4)
    import training_generator as tg
    import tools
    # (2b) featurizer also imports three names that tools.py of the release does not define and featurizer never uses
    #      (fabricated_to_source_filename, drop_row_nums, fabricated_to_source_self): attached as raising placeholders.
    for _name in ("fabricated_to_source_filename", "drop_row_nums", "fabricated_to_source_self"):
        if not hasattr(tools, _name):
            def _absent(*a, _n=_name, **k):
                raise NotImplementedError(f"tools.{_n} is not part of the release")
            setattr(tools, _name, _absent)
    import featurizer as fz
    import omnimatch_predictors as op
    import omnimatch_models as om
    import rf_classifier as rfc

    # (3) training_generator.update_values forks mp.Pool(cpu_count()-1) per call; replaced by a serial pool so that the
    #     typo/perturbation randomness comes from the seeded python random / np.random of this process and CPU use
    #     stays bounded. Output is re-sorted into column order by the release itself.
    class _SerialPool:
        def __init__(self, processes=None):
            pass

        def imap_unordered(self, fn, it):
            return map(fn, it)

        def close(self):
            pass
    tg.mp = types.SimpleNamespace(Pool=_SerialPool, cpu_count=lambda: 2)

    # (4) featurizer hard-codes fasttext.load_model('/usr/local/matching/embedding_files/cc.en.50.bin') and ignores the
    #     embeddings_file config key; redirect that path to our cc.en.50.bin (made with fasttext.util.reduce_model),
    #     loaded once and cached (the release reloads it for every call).
    import fasttext as _ft

    def _load(path):
        key = str(FT50)
        if key not in _FT_CACHE:
            _FT_CACHE[key] = _ft.load_model(key)
        return _FT_CACHE[key]
    fz.fasttext = types.SimpleNamespace(load_model=_load)
    # cosmetic: no progress bars in the logs
    fz.tqdm = lambda x, *a, **k: x
    tools.tqdm = lambda x, *a, **k: x
    return types.SimpleNamespace(tg=tg, fz=fz, tools=tools, op=op, om=om, rfc=rfc, torch=torch)


def seed_all(R, seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    R.torch.manual_seed(seed)
    try:
        import dgl
        dgl.seed(seed)
    except Exception:
        pass


def dump(obj, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------------------------------------------------------------------------ pipeline steps

def generate_training(R, test_dir, train_dir):
    """training_generator.py __main__, call for call (num_iterations 1, common_cols range(1,6), overlaps 0.1..0.9)."""
    import pandas as pd
    from random import choice
    tg = R.tg
    common_cols = list(range(GEN["common_cols_l"], GEN["common_cols_r"], GEN["common_cols_step"]))
    overlaps = []
    ol = GEN["overlaps_l"]
    while ol <= GEN["overlaps_r"]:
        overlaps.append(ol)
        ol += GEN["overlaps_step"]
    log = []
    for i in range(GEN["num_iterations"]):
        for filename in os.listdir(test_dir):
            filepath = os.path.join(test_dir, filename)
            test = pd.read_csv(filepath)
            cc = min(choice(common_cols), len(test.columns))
            train_1_mid, train_2_mid, common = tg.split_horizontal(test, cc, pick_unique=choice([True, False]))
            overlap = choice(overlaps)
            typos, cascade = choice([True, False]), choice([True, False])
            # (argument evaluation order of the release: overlap, typos, cascade -> randint inside split_vertical)
            train_1, train_2 = tg.split_vertical(train_1_mid, train_2_mid, overlap=overlap, typos=typos,
                                                 cov_prc=GEN["cov_prc"], typo_prc=GEN["typo_prc"], cascade=cascade)
            train_1.to_csv(os.path.join(train_dir, f'{filename[:-4]}_{2*i}.csv'), index=False)
            train_2.to_csv(os.path.join(train_dir, f'{filename[:-4]}_{2*i+1}.csv'), index=False)
            log.append({"file": filename, "cc": cc, "common": [str(c) for c in common], "overlap": overlap,
                        "typos": typos, "cascade": cascade, "rows": [len(train_1), len(train_2)],
                        "cols": [train_1.shape[1], train_2.shape[1]]})
    return log


def training_pairs(fzr):
    """Full join/non-join lists between the members of each generated pair (stem_0.csv, stem_1.csv): joins are the
    shared (common) columns, every other T_0 x T_1 pair is a non-join, T_0-column-major."""
    joins, non_joins = [], []
    names = sorted(fzr.dataframes)
    for f0 in names:
        if not f0.endswith("_0.csv"):
            continue
        f1 = f0[:-6] + "_1.csv"
        if f1 not in fzr.dataframes:
            continue
        c0 = fzr.dataframes[f0].columns.tolist()
        c1 = fzr.dataframes[f1].columns.tolist()
        common = [c for c in c0 if c in set(c1)]
        for c in common:
            joins.append(((f0, c), (f1, c)))
        cs = set(common)
        for a in c0:
            for b in c1:
                if not (a == b and a in cs):
                    non_joins.append(((f0, a), (f1, b)))
    return joins, non_joins


def featurize(R, dataset_dir, features_dir, joins, non_joins):
    """featurizer.py __main__ for each of the four features (one feature switched on per run, as the README says) plus
    compute_column_features. featurize_column_pairs calls the undefined compute_header_uniqueness (whose result is
    unused), so its two _pair_featurization calls are made directly. Tokenization is shared across the four runs
    (deterministic, identical state)."""
    fzr = R.fz.Featurizer(str(dataset_dir))
    fzr.featurize_columns()
    dump(fzr.column_features, os.path.join(features_dir, "individual_features.pickle"))
    fzr.tokenize_columns(split_words=True)
    fzr.frequent_token_sets()
    for feature in FEATURES:
        pos = fzr._pair_featurization(joins, str(FT50), [feature])
        neg = fzr._pair_featurization(non_joins, str(FT50), [feature])
        dump(pos, os.path.join(features_dir, f"{feature}_pos.pickle"))
        dump(neg, os.path.join(features_dir, f"{feature}_neg.pickle"))


def column_map(df, header):
    """Header position -> the release's (rstripped, pandas-deduplicated) column name."""
    cols = df.columns.tolist()
    if len(cols) != len(header) or len(set(cols)) != len(cols):
        raise ValueError(f"column mismatch: {len(cols)} vs {len(header)} header cells")
    return cols


def prepare_test(R, case, work):
    """Seed-independent: test tables, test pair pickles (joins = ground-truth pairs, row-major; non-joins = the rest,
    row-major), test features. Cached in work/test_*."""
    import numpy as np
    import pandas as pd
    src, tgt, gtf = case_files(case)
    tdir, fdir = work / "test_tables", work / "test_features"
    done = fdir / "DONE"
    gt = read_ground_truth(gtf)
    if not done.exists():
        shutil.rmtree(tdir, ignore_errors=True)
        shutil.rmtree(fdir, ignore_errors=True)
        tdir.mkdir(parents=True)
        fdir.mkdir(parents=True)
        shutil.copyfile(src, tdir / "source.csv")
        shutil.copyfile(tgt, tdir / "target.csv")
    dfs = R.tools.csvs_to_dataframes(str(tdir))
    with src.open(encoding="utf-8", errors="replace", newline="") as fh:
        hs = next(csv.reader(fh))
    with tgt.open(encoding="utf-8", errors="replace", newline="") as fh:
        ht = next(csv.reader(fh))
    rows = column_map(dfs["source.csv"], hs)
    cols = column_map(dfs["target.csv"], ht)
    if gt.shape != (len(rows), len(cols)):
        raise ValueError(f"ground truth {gt.shape} vs {(len(rows), len(cols))}")
    allpairs = [(("source.csv", r), ("target.csv", c)) for r in rows for c in cols]
    flat = gt.reshape(-1)
    joins = [p for p, g in zip(allpairs, flat) if g]
    non_joins = [p for p, g in zip(allpairs, flat) if not g]
    if not done.exists():
        dump(joins, fdir / "join_pairs.pickle")
        dump(non_joins, fdir / "non_join_pairs.pickle")
        featurize(R, tdir, fdir, joins, non_joins)
        done.write_text("ok")
    return {"rows": rows, "cols": cols, "allpairs": allpairs, "gt": gt, "tdir": tdir, "fdir": fdir}


def build_graph(R, join_features, non_join_features, node_features_file, k, order=None, no_edges=False):
    """omnimatch_predictors.py __main__: column ids, top-k edges, dgl graph with column-L2-normalised node features.
    order: optional explicit pair order for the edge construction (row-major variant); no_edges: graph without edges."""
    import dgl
    import torch
    import torch.nn.functional as F
    op = R.op
    columns_to_ids, ids_to_columns, ids_to_features = op.get_column_ids(str(node_features_file))
    num_features = len(FEATURES) + int("jaccard_containment" in FEATURES)
    ft = torch.tensor(ids_to_features, dtype=torch.float)
    if no_edges:
        graph = dgl.graph((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
                          num_nodes=len(ids_to_features))
        edge_types = torch.tensor([], dtype=torch.long)
    else:
        if order is not None:
            merged = dict(join_features)
            merged.update(non_join_features)
            jf, njf = OrderedDict((p, merged[p]) for p in order), {}
        else:
            jf, njf = join_features, non_join_features
        edge_list, edge_types = op.create_edge_list_topk(jf, njf, columns_to_ids, num_features, k)
        et = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        graph = dgl.graph((et[0], et[1]))
        edge_types = torch.tensor(edge_types)
    graph.ndata["feat"] = F.normalize(ft, 2, 0)
    return graph, edge_types, columns_to_ids, ft, num_features


class _NoEdgeConv:
    """Context for the extra arm rgcn_margin_noedges (not in the release's model menu; it mirrors the port's arm N).
    dgl 0.5.3's update_all on a graph without edges returns a zero tensor of the *input* width, so RelGraphConv cannot
    run on an edgeless graph. Inside this context RelGraphConv.forward computes what it computes when every node
    receives no message: feat @ loop_weight + h_bias (no activation, no dropout in the release's layers)."""

    def __enter__(self):
        import dgl.nn.pytorch as dglnn
        self.cls = dglnn.RelGraphConv
        self.orig = self.cls.forward

        def fwd(conv, g, feat, etypes, norm=None):
            h = feat @ conv.loop_weight
            if conv.bias:
                h = h + conv.h_bias
            if conv.activation:
                h = conv.activation(h)
            return conv.dropout(h)
        self.cls.forward = fwd
        return self

    def __exit__(self, *exc):
        self.cls.forward = self.orig
        return False


def run_one(R, case, seed, out, force=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    key = case["key"]
    targets = {m: out / "matrices" / f"{m}__s{seed}" / f"{key}.csv" for m in MODELS}
    if not force and all(p.exists() for p in targets.values()):
        return None
    t0 = time.perf_counter()
    info = {"key": key, "id": case["id"], "seed": seed}
    work = out / "work" / key
    test = prepare_test(R, case, work)
    info["t_test"] = time.perf_counter() - t0
    sdir = work / f"s{seed}"
    shutil.rmtree(sdir, ignore_errors=True)
    train_dir, trf_dir, samples_dir = sdir / "train_tables", sdir / "train_features", sdir / "samples"
    for d in (train_dir, trf_dir, samples_dir):
        d.mkdir(parents=True)

    # ---- 1. training data (seeded)
    seed_all(R, seed)
    info["generator"] = generate_training(R, str(test["tdir"]), str(train_dir))
    # ---- 2. training pair lists + features
    fzr_cols = R.tools.csvs_to_dataframes(str(train_dir))
    holder = types.SimpleNamespace(dataframes=fzr_cols)
    joins, non_joins = training_pairs(holder)
    dump(joins, trf_dir / "join_pairs.pickle")
    dump(non_joins, trf_dir / "non_join_pairs.pickle")
    info["train_pairs"] = [len(joins), len(non_joins)]
    featurize(R, train_dir, trf_dir, joins, non_joins)
    info["t_features"] = time.perf_counter() - t0

    op, tools = R.op, R.tools
    k = PARAMS["k"]
    # ---- 3. training graph (omnimatch_predictors __main__)
    join_train_features = tools.get_features(FEATURES, str(trf_dir), pair_type="pos")
    non_join_train_features = tools.get_features(FEATURES, str(trf_dir), pair_type="neg")
    join_train_pairs = list(join_train_features.keys())
    non_join_train_pairs = list(non_join_train_features.keys())
    tr_graph, tr_et, tr_ids, tr_ft, num_features = build_graph(R, join_train_features, non_join_train_features,
                                                               trf_dir / "individual_features.pickle", k)
    info["train_nodes_edges"] = [tr_graph.number_of_nodes(), tr_graph.number_of_edges()]
    info["nan_node_features"] = [int(torch.isnan(tr_ft).sum())]
    all_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    source_to_fabricated = {}
    for filename in all_files:
        source_to_fabricated.setdefault(tools.fabricated_to_source(filename, BENCHMARK), []).append(filename)
    n_sources = min(PARAMS["number_of_sources"], len(source_to_fabricated))   # 'or all available'
    info["number_of_sources"] = n_sources

    # ---- 4. test graphs and features
    def test_feats():
        return (tools.get_features(FEATURES, str(test["fdir"]), pair_type="pos"),
                tools.get_features(FEATURES, str(test["fdir"]), pair_type="neg"))
    tnf = test["fdir"] / "individual_features.pickle"
    jt, njt = test_feats()
    join_test_pairs, non_join_test_pairs = list(jt.keys()), list(njt.keys())
    te_graph, te_et, te_ids, te_ft, _ = build_graph(R, jt, njt, tnf, k)                     # release order
    jt, njt = test_feats()
    te_graph_rm, te_et_rm, _, _, _ = build_graph(R, jt, njt, tnf, k, order=test["allpairs"])  # row-major order
    te_graph_ne, te_et_ne, _, _, _ = build_graph(R, jt, njt, tnf, k, no_edges=True)
    info["nan_node_features"].append(int(torch.isnan(te_ft).sum()))
    info["test_nodes_edges"] = [te_graph.number_of_nodes(), te_graph.number_of_edges()]
    jp_ids = [(te_ids[p[0]], te_ids[p[1]]) for p in join_test_pairs]
    njp_ids = [(te_ids[p[0]], te_ids[p[1]]) for p in non_join_test_pairs]
    pair_ids = [(te_ids[a], te_ids[b]) for a, b in test["allpairs"]]
    shape = (len(test["rows"]), len(test["cols"]))

    def matrix(emb, model, loss):
        s = np.array([op.compute_score(emb[a], emb[b], loss, model) for a, b in pair_ids], dtype=float)
        if loss == "rgcn_margin":
            s = 1.0 - s / 2.0                     # release ranks by 1 - d; d in [0, 2] for unit embeddings
        return s.reshape(shape)

    def release_metrics(emb, model, loss):
        try:
            m, _, _ = op.compute_metrics(jp_ids, njp_ids, te_ids, emb, model, loss)
            return {a: float(b) for a, b in m.items()}
        except Exception as e:                                # noqa: BLE001
            return {"error": repr(e)}

    res, metrics = {}, {}
    # ---- 5a. rgcn_margin (training graph; joins first as in the release)
    seed_all(R, seed)
    joins_train, nn_train = op.get_positive_negative_nodes(join_train_pairs, non_join_train_pairs, tr_ids,
                                                           PARAMS["number_of_datasets"], n_sources,
                                                           source_to_fabricated, "", str(samples_dir), BENCHMARK)
    info["triplet_joins"] = len(joins_train)
    model = R.om.train_model_triplet_loss(tr_graph, joins_train, nn_train, tr_ids, tr_et, PARAMS["epochs"],
                                          PARAMS["dimension"], num_features, PARAMS["margin"], PARAMS["norm"],
                                          rate=PARAMS["learning_rate"])
    with torch.no_grad():
        emb = model.gnn(te_graph, te_graph.ndata["feat"], te_et).detach()
        emb_rm = model.gnn(te_graph_rm, te_graph_rm.ndata["feat"], te_et_rm).detach()
    res["rgcn_margin"] = matrix(emb, model, "rgcn_margin")
    metrics["rgcn_margin"] = release_metrics(emb, model, "rgcn_margin")
    res["rgcn_margin_rowmajor"] = matrix(emb_rm, model, "rgcn_margin")
    metrics["rgcn_margin_rowmajor"] = release_metrics(emb_rm, model, "rgcn_margin")
    dump({"rows": test["rows"], "cols": test["cols"], "ids": te_ids, "emb": emb.numpy(), "emb_rowmajor": emb_rm.numpy()},
         sdir / "rgcn_embeddings.pickle")

    # ---- 5b. same model and training, no edges in either graph (self-loop weights only; = the port's arm N)
    seed_all(R, seed)
    tr_graph_ne, tr_et_ne, _, _, _ = build_graph(R, {}, {}, trf_dir / "individual_features.pickle", k, no_edges=True)
    joins_train2, nn_train2 = op.get_positive_negative_nodes(join_train_pairs, non_join_train_pairs, tr_ids,
                                                             PARAMS["number_of_datasets"], n_sources,
                                                             source_to_fabricated, "", str(samples_dir), BENCHMARK)
    with _NoEdgeConv():
        model_ne = R.om.train_model_triplet_loss(tr_graph_ne, joins_train2, nn_train2, tr_ids, tr_et_ne,
                                                 PARAMS["epochs"], PARAMS["dimension"], num_features, PARAMS["margin"],
                                                 PARAMS["norm"], rate=PARAMS["learning_rate"])
        with torch.no_grad():
            emb_ne = model_ne.gnn(te_graph_ne, te_graph_ne.ndata["feat"], te_et_ne).detach()
    res["rgcn_margin_noedges"] = matrix(emb_ne, model_ne, "rgcn_margin")
    metrics["rgcn_margin_noedges"] = release_metrics(emb_ne, model_ne, "rgcn_margin")

    # ---- 5c. mlp_cross_entropy (the release's no-graph model: MLP on the concatenated column statistics)
    seed_all(R, seed)
    columns_to_ids, _, ids_to_features = op.get_column_ids(str(trf_dir / "individual_features.pickle"))
    mlp = R.om.train_mlp_model(join_train_pairs, non_join_train_pairs, columns_to_ids, ids_to_features,
                               PARAMS["epochs"], rate=PARAMS["learning_rate"])
    emb_m = F.normalize(te_ft, 2, 0)
    with torch.no_grad():
        res["mlp_cross_entropy"] = matrix(emb_m, mlp, "mlp_cross_entropy")
    metrics["mlp_cross_entropy"] = release_metrics(emb_m, mlp, "mlp_cross_entropy")

    # ---- 5d. rf_predictor (random forest, 100 trees, on the 5 pair similarities of the sampled generated pairs)
    seed_all(R, seed)
    samples = sorted(samples_dir.glob("*.txt"))
    trp = tools.get_features(FEATURES, str(trf_dir), pair_type="pos")
    trn = tools.get_features(FEATURES, str(trf_dir), pair_type="neg")
    tep, ten = test_feats()
    X_train, y_train, X_test, y_test, groups = tools.prepare_data_datasets_self(
        trp, trn, tep, ten, sample_file=str(samples[0]), benchmark=BENCHMARK)
    clf = R.rfc.Classifier(no_est=100)
    clf.train_classifier(X_train, y_train)
    metrics["rf"] = {"PR-AUC": float(clf.compute_metrics(X_test, y_test))}
    merged = dict(tep)
    merged.update(ten)
    Xall = np.array([merged[p] for p in test["allpairs"]], dtype=object)
    res["rf"] = clf.classifier.predict_proba(Xall)[:, 1].reshape(shape)

    for m, M in res.items():
        targets[m].parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(targets[m], M, delimiter=",", fmt="%.10g")
    info["release_metrics"] = metrics
    info["nan_in_matrices"] = {m: int(np.isnan(M).sum()) for m, M in res.items()}
    info["seconds"] = time.perf_counter() - t0
    # training tables are large and reproducible from the seed: keep only the features/pickles
    shutil.rmtree(train_dir, ignore_errors=True)
    return info


def command_run(args):
    R = import_release()
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.limit)
    if args.only:
        cases = [c for c in cases if c["key"] in set(args.only)]
    log = out / "runs.jsonl"
    for case in cases:
        if case_files(case) is None:
            continue
        for seed in args.seeds:
            try:
                info = run_one(R, case, seed, out, force=args.force)
                if info is None:
                    continue
                info["status"] = "ok"
            except Exception as e:                            # record and continue
                info = {"key": case["key"], "id": case["id"], "seed": seed, "status": "failed", "error": repr(e),
                        "traceback": traceback.format_exc()}
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(info, default=str) + "\n")
            print(f"{case['id']} s{seed} {info['status']} {info.get('seconds', 0):.0f}s "
                  f"{info.get('release_metrics', info.get('error', ''))}", flush=True)
    return 0


# ------------------------------------------------------------------------------------- scoring (.venv-bdi)

def command_score(args):
    import numpy as np
    sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "layer3b"))
    sys.path.insert(0, str(HERE.parent / "sf-matcher-study" / "tuning"))
    sys.path.insert(0, str(HERE))
    from score_r7 import metrics as tf_metrics
    from score_screen_metrics import permutations
    out = Path(args.out) if args.out else OUT
    cases = {c["key"]: c for c in load_cases()}
    rows = []
    for d in sorted((out / "matrices").iterdir()):
        model, seed = d.name.split("__s")
        for f in sorted(d.glob("*.csv")):
            case = cases[f.stem]
            gt = read_ground_truth(case_files(case)[2])
            S = np.loadtxt(f, delimiter=",", ndmin=2)
            if S.shape != gt.shape:
                print("shape mismatch", f)
                continue
            m = tf_metrics(S, gt, permutations(gt.shape, f"{case['dataset']}/{case['scenario']}"))
            rows.append({"key": f.stem, "family": case["family"], "cohort": case.get("cohort", "all"), "arm": model,
                         "seed": int(seed), "J": m["J"], "R_exp": m["R_exp"], "F1_Hr": m["F1_Hr"]})
    with (out / "scores.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} matrices scored -> {out / 'scores.csv'}")
    return 0


def command_compare(args):
    import numpy as np
    import pandas as pd
    out = Path(args.out) if args.out else OUT
    keys = [c["key"] for c in load_cases(20)]
    rel = pd.read_csv(out / "scores.csv")
    rel["arm"] = "release:" + rel["arm"]
    port = pd.read_csv(PORT / "scores.csv")
    port = port[port["seed"].isin([1, 2, 3]) & port["arm"].isin(["F", "N", "RF"])]
    port["arm"] = "port:" + port["arm"]
    d = pd.concat([rel, port])
    d = d[d["key"].isin(keys)]
    per = d.groupby(["arm", "key"])[["J", "R_exp", "F1_Hr"]].mean()          # seeds averaged per scenario
    nseeds = d.groupby(["arm", "key"])["seed"].nunique()
    arms = sorted(per.index.get_level_values(0).unique())
    lines = ["| arm | scenarios | seeds/scenario | J_tf | R_exp | F1_Hr |", "|---|---|---|---|---|---|"]
    for a in arms:
        x = per.loc[a]
        lines.append(f"| {a} | {len(x)} | {nseeds.loc[a].min()}-{nseeds.loc[a].max()} | {x['J'].mean():.3f} | "
                     f"{x['R_exp'].mean():.3f} | {x['F1_Hr'].mean():.3f} |")
    contrasts = [("release:rgcn_margin", "release:mlp_cross_entropy"),
                 ("release:rgcn_margin", "port:F"),
                 ("release:rf", "port:RF"),
                 ("release:rgcn_margin", "release:rgcn_margin_noedges"),
                 ("release:rgcn_margin_noedges", "port:N"),
                 ("release:rgcn_margin_rowmajor", "port:F"),
                 ("release:rgcn_margin", "release:rgcn_margin_rowmajor"),
                 ("release:rf", "release:rgcn_margin"),
                 ("port:F", "port:N")]
    lines += ["", "| contrast | scenarios | J_tf diff [95% bootstrap] | R_exp | F1_Hr |", "|---|---|---|---|---|"]
    rng = np.random.default_rng(20261027)
    for a, b in contrasts:
        if a not in arms or b not in arms:
            continue
        common = sorted(set(per.loc[a].index) & set(per.loc[b].index))
        diff = per.loc[a].loc[common] - per.loc[b].loc[common]
        J = diff["J"].to_numpy()
        boots = [J[rng.integers(0, len(J), len(J))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        lines.append(f"| {a} - {b} | {len(common)} | {J.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{diff['R_exp'].mean():+.3f} | {diff['F1_Hr'].mean():+.3f} |")
    # per-scenario table
    wide = per["J"].unstack(0)
    wide.to_csv(out / "per_scenario_J.csv")
    text = "\n".join(lines)
    (out / "comparison.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(wide.round(3).to_string())
    return 0


# ------------------------------------------------------------------------------------------------ fastText

def command_reduce(args):
    """cc.en.300 -> cc.en.50 with fasttext.util.reduce_model, as the release expects (featurizer loads cc.en.50.bin)."""
    import fasttext
    import fasttext.util
    OUT.mkdir(parents=True, exist_ok=True)
    m = fasttext.load_model(str(FT300))
    print("loaded", m.get_dimension(), flush=True)
    fasttext.util.reduce_model(m, 50)
    print("reduced", m.get_dimension(), flush=True)
    m.save_model(str(FT50))
    print("saved", FT50, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reduce")
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--only", nargs="*")
    r.add_argument("--force", action="store_true")
    r.add_argument("--out")
    for name in ("score", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--out")
    a = ap.parse_args()
    return {"reduce": command_reduce, "run": command_run, "score": command_score, "compare": command_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
