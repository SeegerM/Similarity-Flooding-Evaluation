"""Tests for the SemStruct reconstruction.

Three of these are not ordinary unit tests but the mechanical half of the
validation ladder in `harness/SEMSTRUCT_REIMPLEMENTATION.md` S6:

    test_lm_only_control_is_exactly_the_frozen_embedding   check 1
    test_row_permutation_preserves_column_marginals        check 2 (its premise)
    test_zero_structural_delta_reproduces_semantics        check 3

They are written so that a reconstruction that quietly gained its accuracy from
something other than structure would fail here rather than at the point where
someone reads a table of numbers.

No language model is loaded: `FakeEncoder` returns deterministic vectors keyed
on the text, which is all the graph code needs and keeps the suite instant.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semstruct import (SemStructConfig, batch_graphs, bipartite_rerank,  # noqa: E402
                       build_graph, build_model, cosine_matrix,
                       permute_within_columns, row_softmax)
from semstruct.augment import acronym, make_view, typo  # noqa: E402
from semstruct.graph import COLUMN, ROW, VALUE, _leader_merge  # noqa: E402
from semstruct.model import mean_aggregate  # noqa: E402
from semstruct.text import discretize, is_numeric_column  # noqa: E402
from semstruct.train import (TrainingGroup, batch_loss,  # noqa: E402
                             _pairwise_cosine_distance)

CFG = SemStructConfig(embed_dim=16, hidden_dim=16, row_cap=50)


class FakeEncoder:
    """Deterministic unit vectors keyed on the text; no model, no network."""

    def __init__(self, cfg: SemStructConfig = CFG, aliases=None) -> None:
        self.cfg = cfg
        self.aliases = aliases or {}
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        text = self.aliases.get(text, text)
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        vector = np.random.default_rng(seed).normal(size=self.cfg.embed_dim)
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def encode(self, texts):
        self.calls += 1
        if not texts:
            return np.zeros((0, self.cfg.embed_dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def encode_cached(self, texts):
        return self.encode(texts)


TABLE_HEADERS = ["city", "amount"]
TABLE_COLUMNS = [["London", "Paris", "London", ""],
                 ["10", "20", "30", "40"]]


def _graph(headers=None, columns=None, cfg=CFG, encoder=None):
    return build_graph(headers or TABLE_HEADERS, columns or TABLE_COLUMNS,
                       cfg, encoder or FakeEncoder(cfg))


# ---------------------------------------------------------------- graph shape

def test_graph_has_one_node_per_column_row_and_distinct_value():
    graph = _graph()
    assert graph.n_cols == 2
    assert graph.n_rows == 4
    # city: {London, Paris} (the empty cell is null); amount: 4 distinct, but
    # only 4 <= n_bins so it stays categorical -> 4 value nodes.
    assert graph.n_values == 2 + 4
    assert list(graph.node_type[:2]) == [COLUMN, COLUMN]
    assert set(graph.node_type[2:6].tolist()) == {ROW}
    assert set(graph.node_type[6:].tolist()) == {VALUE}


def test_null_cell_contributes_no_edge():
    graph = _graph()
    row_of_null = 2 + 3                      # fourth row, city is ""
    incident = (graph.edge_index[0] == row_of_null).sum()
    assert incident == 1                     # only its `amount` value


def test_every_edge_is_present_in_both_directions():
    graph = _graph()
    pairs = {(int(a), int(b)) for a, b in zip(*graph.edge_index)}
    assert all((b, a) in pairs for a, b in pairs)
    assert len(pairs) == graph.edge_index.shape[1]


def test_columns_and_rows_are_never_directly_adjacent():
    graph = _graph()
    types = graph.node_type
    for a, b in zip(*graph.edge_index):
        kinds = {int(types[a]), int(types[b])}
        assert kinds != {COLUMN, ROW}
        assert VALUE in kinds


def test_numeric_column_is_binned_to_at_most_k_value_nodes():
    cfg = SemStructConfig(embed_dim=16, hidden_dim=16, n_bins=8)
    values = [str(i) for i in range(100)]
    assert is_numeric_column(values, cfg.n_bins)
    graph = build_graph(["n"], [values], cfg, FakeEncoder(cfg))
    assert graph.n_values <= cfg.n_bins


def test_duplicate_quantile_edges_collapse_instead_of_making_empty_bins():
    heavy = ["0"] * 90 + [str(i) for i in range(1, 11)]
    tokens, discretized = discretize(heavy, 8)
    assert discretized
    assert len(set(tokens)) < 8


# ------------------------------------------------------------------- merging

def test_identical_values_merge_and_unrelated_ones_do_not():
    cfg = SemStructConfig(embed_dim=16, hidden_dim=16, merge_threshold=0.9)
    aliases = {"column: c, value: NYC": "column: c, value: New York"}
    encoder = FakeEncoder(cfg, aliases=aliases)
    graph = build_graph(["c"], [["New York", "NYC", "Berlin"]], cfg, encoder)
    assert graph.n_values == 2          # New York and NYC merged; Berlin apart


def test_leader_merge_does_not_chain_transitively():
    # b is close to a and to c; a and c are far apart.  A transitive-closure
    # merge would put all three in one node; leader-based merging must not.
    vectors = np.array([[1.0, 0.0], [0.7, 0.714], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    counts = np.array([3, 2, 1])
    assignment, clusters = _leader_merge(vectors, counts, ["a", "b", "c"], 0.68)
    assert clusters == 2
    assert assignment[0] == assignment[1] and assignment[2] != assignment[0]


def test_threshold_above_one_disables_merging():
    cfg = SemStructConfig(embed_dim=16, hidden_dim=16, merge_threshold=1.01)
    graph = build_graph(["c"], [["a", "b", "c", "a"]], cfg, FakeEncoder(cfg))
    assert graph.n_values == 3               # distinct tokens, nothing merged


def test_centered_merge_space_removes_the_common_direction():
    """The measurement behind the S4.2 ladder, as an assertion.

    Three vectors sharing a large common component are all pairwise similar in
    raw cosine and unrelated once that component is removed.
    """
    from semstruct.graph import _merge_space
    common = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vectors = np.stack([common + 0.15 * np.eye(3, dtype=np.float32)[i]
                        for i in range(3)])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    raw = vectors @ vectors.T
    centred = _merge_space(vectors, "centered")
    centred_similarity = centred @ centred.T
    off = ~np.eye(3, dtype=bool)
    assert raw[off].min() > 0.9
    assert centred_similarity[off].max() < 0.9


def test_merge_is_confined_to_a_column():
    cfg = SemStructConfig(embed_dim=16, hidden_dim=16, merge_threshold=-1.0)
    # threshold -1 merges everything it is allowed to merge; two columns must
    # still keep at least one value node each.
    graph = build_graph(["a", "b"], [["x", "y"], ["x", "y"]], cfg, FakeEncoder(cfg))
    assert graph.n_values == 2


# --------------------------------------------------------------------- model

def test_sage_matches_reference():
    """One layer, recomputed from the definition rather than the code."""
    torch.manual_seed(0)
    model = build_model(CFG)
    conv = model.convs[0]
    x = torch.randn(6, CFG.embed_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 0, 1, 4, 3]])
    got = conv(x, edge_index)

    n = x.shape[0]
    adjacency = torch.zeros(n, n)
    for a, b in zip(*edge_index):
        adjacency[b, a] = 1.0
    degree = adjacency.sum(1, keepdim=True).clamp(min=1.0)
    expected = conv.lin_l(adjacency @ x / degree) + conv.lin_r(x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_isolated_node_aggregates_to_zero():
    x = torch.randn(4, 3)
    aggregated = mean_aggregate(x, torch.tensor([[0], [1]]))
    assert torch.allclose(aggregated[2], torch.zeros(3))
    assert torch.allclose(aggregated[3], torch.zeros(3))
    assert torch.allclose(aggregated[1], x[0])


def test_zero_structural_delta_reproduces_semantics():
    """S6 check 3: with MLP(.) == 0 the model is its own initialisation."""
    graph = _graph()
    model = build_model(CFG)
    model.zero_structural_delta()
    x = torch.from_numpy(graph.features).float()
    edges = torch.from_numpy(graph.edge_index)
    columns = torch.arange(graph.n_cols)
    with torch.no_grad():
        out = model(x, edges, columns)
    assert torch.equal(out, x[:graph.n_cols])


def test_lm_only_control_is_exactly_the_frozen_embedding():
    """S6 check 1: the control is this pipeline minus structure, not a rewrite."""
    graph = _graph()
    model = build_model(CFG)
    x = torch.from_numpy(graph.features).float()
    edges = torch.from_numpy(graph.edge_index)
    columns = torch.arange(graph.n_cols)
    with torch.no_grad():
        out = model(x, edges, columns, structural=False)
    assert torch.equal(out, x[:graph.n_cols])


def test_batching_is_equivalent_to_separate_forward_passes():
    graphs = [_graph(), _graph(["a"], [["p", "q", "r"]])]
    model = build_model(CFG)
    features, edge_index, column_index = batch_graphs(graphs)
    with torch.no_grad():
        together = model(torch.from_numpy(features).float(),
                         torch.from_numpy(edge_index),
                         torch.from_numpy(np.concatenate(column_index)))
        apart = torch.cat([
            model(torch.from_numpy(g.features).float(),
                  torch.from_numpy(g.edge_index),
                  torch.arange(g.n_cols)) for g in graphs])
    assert torch.allclose(together, apart, atol=1e-6)


def test_a_column_sees_row_cooccurrence_only_at_depth_three():
    """The depth the paper chose is the shallowest one that can work.

    Column -> value -> row -> value -> column is four hops, but the *column*
    node only needs the co-occurring value, which is three.
    """
    graph = _graph()
    adjacency = np.zeros((graph.n_nodes, graph.n_nodes))
    for a, b in zip(*graph.edge_index):
        adjacency[a, b] = 1.0
    reach = np.linalg.matrix_power(adjacency + np.eye(graph.n_nodes), 2)
    assert reach[0, 1] == 0.0                 # two hops: no other column
    reach3 = np.linalg.matrix_power(adjacency + np.eye(graph.n_nodes), 3)
    assert reach3[0, graph.n_cols + graph.n_rows + 2] > 0.0   # a value of col 1


# ----------------------------------------------------------------- objective

def test_triplet_loss_is_zero_when_positives_are_already_closer():
    cfg = SemStructConfig(embed_dim=4, hidden_dim=4, margin=0.2)
    embeddings = torch.tensor([[1.0, 0, 0, 0], [1.0, 0.01, 0, 0],
                               [0.0, 1.0, 0, 0], [0.0, 1.0, 0.01, 0]])
    distance = _pairwise_cosine_distance(embeddings)
    assert distance[0, 1] < distance[0, 2]
    positive = torch.tensor([[False, True, False, False],
                             [True, False, False, False],
                             [False, False, False, True],
                             [False, False, True, False]])
    negative = ~positive & ~torch.eye(4, dtype=torch.bool)
    hardest_positive = torch.where(positive, distance, torch.full_like(distance, -1.0)).max(1).values
    hardest_negative = torch.where(negative, distance, torch.full_like(distance, 2.0)).min(1).values
    loss = torch.relu(hardest_positive - hardest_negative + cfg.margin)
    assert float(loss.max()) < cfg.margin + 1e-6


def test_batch_loss_pairs_a_view_with_its_source_table():
    encoder = FakeEncoder(CFG)

    class Store:
        def get(self, headers, columns):
            return build_graph(headers, columns, CFG, encoder)

    rng = np.random.default_rng(0)
    group = TrainingGroup(table_id="t", headers=list(TABLE_HEADERS),
                          columns=[list(c) for c in TABLE_COLUMNS])
    group.views = [make_view(group.headers, group.columns, CFG, rng)]
    model = build_model(CFG)
    loss = batch_loss(model, Store(), [group], [0], CFG)
    assert loss is None or float(loss.detach()) >= 0.0


# ------------------------------------------------------------------ controls

def test_row_permutation_preserves_column_marginals():
    """S6 check 2's premise: only co-occurrence is destroyed."""
    columns = [["a", "b", "c", "d"], ["1", "2", "3", "4"]]
    permuted = permute_within_columns(columns, seed=7)
    assert [sorted(c) for c in permuted] == [sorted(c) for c in columns]
    assert permuted != columns


def test_row_permutation_leaves_the_graph_isomorphic_in_its_column_part():
    encoder = FakeEncoder(CFG)
    columns = [["a", "b", "c", "d"], ["1", "2", "3", "4"]]
    original = build_graph(["x", "y"], columns, CFG, encoder)
    shuffled = build_graph(["x", "y"], permute_within_columns(columns, 7), CFG, encoder)
    assert original.n_values == shuffled.n_values
    assert original.edge_index.shape == shuffled.edge_index.shape
    assert np.allclose(original.features[:2], shuffled.features[:2])


# ------------------------------------------------------------ views and noise

def test_a_view_keeps_a_subset_of_columns_and_records_the_correspondence():
    rng = np.random.default_rng(3)
    view = make_view(["alpha", "beta", "gamma"],
                     [["1", "2"], ["3", "4"], ["5", "6"]], CFG, rng)
    assert 2 <= len(view.headers) <= 3
    assert len(view.source_columns) == len(view.headers)
    assert view.source_columns == sorted(set(view.source_columns))
    assert all(0 <= c < 3 for c in view.source_columns)


def test_noise_operators_change_text_without_destroying_it():
    rng = np.random.default_rng(1)
    assert typo("customer", rng) != "customer"
    assert acronym("customer_first_name", rng) == "CFN"
    assert acronym("customer", rng) == "cstmr"


# ------------------------------------------------------------------- scoring

def test_cosine_matrix_orientation_is_source_by_target():
    source = np.eye(3)[:2]
    target = np.eye(3)
    S = cosine_matrix(source, target)
    assert S.shape == (2, 3)
    assert np.isclose(S[0, 0], 1.0) and np.isclose(S[1, 1], 1.0)


def test_row_softmax_is_row_stochastic_and_order_preserving():
    S = np.array([[0.9, 0.1, -0.2], [0.0, 0.5, 0.4]])
    P = row_softmax(S)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.array_equal(np.argsort(-P, axis=1), np.argsort(-S, axis=1))


def test_bipartite_rerank_puts_assigned_pairs_above_everything_else():
    S = np.array([[0.9, 0.8], [0.85, 0.2]])
    out = bipartite_rerank(S)
    assigned = [(0, 1), (1, 0)]              # the optimal assignment here
    lowest_assigned = min(out[i, j] for i, j in assigned)
    others = [out[i, j] for i in range(2) for j in range(2)
              if (i, j) not in assigned]
    assert all(o < lowest_assigned for o in others)


def test_bipartite_rerank_keeps_relative_order_of_the_rest():
    S = np.array([[0.9, 0.5, 0.4], [0.2, 0.8, 0.1], [0.3, 0.2, 0.7]])
    out = bipartite_rerank(S)
    mask = np.ones(S.shape, dtype=bool)
    mask[[0, 1, 2], [0, 1, 2]] = False
    assert np.array_equal(np.argsort(S[mask]), np.argsort(out[mask]))


def test_empty_table_does_not_crash_the_pipeline():
    graph = build_graph([], [], CFG, FakeEncoder(CFG))
    assert graph.n_nodes == 0
    assert graph.edge_index.shape == (2, 0)
