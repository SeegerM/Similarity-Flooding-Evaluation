"""Tabular graph construction (PAPER S3.1-S3.2).

One table becomes one heterogeneous graph with three node types:

    column nodes   one per attribute; these are what the matcher compares
    row  nodes     one per record; zero-initialised topological conduits
    value nodes    one per distinct value *of a column*, after merging

and two undirected edge types: column-value (the value occurs in the column)
and row-value (the value occurs in the record).  Columns are never adjacent to
rows directly -- a column reaches a co-occurring column only through a value,
a row, and a second value, which is why the depth-3 encoder is the shallowest
one that can see row-level co-occurrence at all.

Three construction points are ours rather than the paper's, and all three are
recorded in SEMSTRUCT_REIMPLEMENTATION.md S3:

  * value nodes are keyed **per column**.  The paper's own value serialisation
    is "column: {}, value: {}", so the same string under two headers already
    has two different embeddings; keying globally would give one node two
    contradictory features.
  * merging is **within a column** and **leader-based**, so it cannot chain a
    long transitive path into one node.
  * a merged node's feature is the **mean of its members**, renormalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SemStructConfig
from .encoder import Encoder
from .text import discretize, serialize_column, serialize_value

COLUMN, ROW, VALUE = 0, 1, 2


@dataclass
class TableGraph:
    features: np.ndarray        # (n_nodes, d) float32, row nodes already zeroed
    edge_index: np.ndarray      # (2, n_edges) int64, both directions present
    n_cols: int
    n_rows: int
    n_values: int
    column_labels: List[str]
    node_type: np.ndarray       # (n_nodes,) int8
    merged_from: Optional[List[int]] = None   # audit trail, not used by the model

    @property
    def n_nodes(self) -> int:
        return self.features.shape[0]

    def column_features(self) -> np.ndarray:
        return self.features[: self.n_cols]


def _merge_space(vectors: np.ndarray, similarity: str) -> np.ndarray:
    """The vectors the *threshold* is applied to (never the node features).

    `raw` is the literal reading of PAPER S3.2.1.  `centered` subtracts the
    column's mean value embedding before renormalising, which removes the
    common direction that makes every e5-small cosine sit near 0.95; see
    SEMSTRUCT_REIMPLEMENTATION.md S4.2 for the measurement that makes this a
    declared arm rather than a silent correction.
    """
    if similarity == "raw":
        return vectors
    if similarity != "centered":
        raise ValueError(f"unknown merge_similarity {similarity!r}")
    if len(vectors) < 2:
        return vectors
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    norms = np.clip(np.linalg.norm(centred, axis=1, keepdims=True), 1e-9, None)
    return (centred / norms).astype(np.float32)


def _leader_merge(vectors: np.ndarray, counts: np.ndarray,
                  tokens: Sequence[str], threshold: float,
                  similarity: str = "raw", strict: bool = False) -> Tuple[np.ndarray, int]:
    """Leader-based greedy merge; returns (assignment, n_clusters).

    Deterministic order: descending frequency, then the token string.  A value
    joins the first leader it is similar enough to, so a chain of pairwise
    similarities below the threshold at its ends never collapses into one node.
    A threshold above 1.0 disables merging: value nodes are already exact
    duplicates of each other by construction, so nothing is left to merge.
    """
    n = len(tokens)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    if threshold > 1.0:
        return np.arange(n, dtype=np.int64), n
    vectors = _merge_space(vectors, similarity)
    order = sorted(range(n), key=lambda i: (-int(counts[i]), tokens[i]))
    assignment = np.full(n, -1, dtype=np.int64)
    leaders: List[int] = []
    leader_matrix = np.zeros((0, vectors.shape[1]), dtype=np.float32)
    for index in order:
        if leaders:
            sims = leader_matrix @ vectors[index]
            best = int(np.argmax(sims))
            if (float(sims[best]) > threshold) if strict else (float(sims[best]) >= threshold):
                assignment[index] = best
                continue
        assignment[index] = len(leaders)
        leaders.append(index)
        leader_matrix = np.vstack([leader_matrix, vectors[index][None, :]])
    return assignment, len(leaders)


def build_graph(headers: Sequence[str], columns: Sequence[Sequence[str]],
                cfg: SemStructConfig, encoder: Encoder) -> TableGraph:
    """Build one table's graph.  `columns` is column-major, already row-capped."""
    # These three fields document choices the code implements one way only.
    # Guarded rather than ignored, so a configuration cannot claim something
    # the graph does not do.
    if not cfg.drop_nulls:
        raise ValueError("null handling other than dropping is not part of the "
                         "frozen variant; see SEMSTRUCT_REIMPLEMENTATION.md S3.2")
    if cfg.value_scope != "per_column" or cfg.merge_scope != "per_column":
        raise ValueError("value and merge scope are per-column in the frozen "
                         "variant; see SEMSTRUCT_REIMPLEMENTATION.md S3.2")

    headers = list(headers)
    n_cols = len(headers)
    n_rows = len(columns[0]) if n_cols else 0
    for column in columns:
        if len(column) != n_rows:
            raise ValueError("ragged table passed to build_graph")

    # 1. discretise, then collect each column's distinct value tokens
    tokens_per_column: List[List[Optional[str]]] = []
    uniques: List[List[str]] = []
    counts: List[np.ndarray] = []
    for column in columns:
        tokens, _ = discretize(column, cfg.n_bins)
        tokens_per_column.append(tokens)
        tally: Dict[str, int] = {}
        for token in tokens:
            if token is None:
                continue
            tally[token] = tally.get(token, 0) + 1
        keys = sorted(tally)
        uniques.append(keys)
        counts.append(np.array([tally[k] for k in keys], dtype=np.int64))

    # 2. embed column headers and value tokens with the frozen PLM
    column_texts = [serialize_column(h, cfg.column_serialization) for h in headers]
    value_texts: List[str] = []
    for header, keys in zip(headers, uniques):
        value_texts.extend(
            serialize_value(header, key, cfg.value_template) for key in keys)
    embeddings = encoder.encode_cached(column_texts + value_texts)
    column_vectors = embeddings[:n_cols]
    value_vectors = embeddings[n_cols:]

    # 3. merge value nodes within each column
    node_vectors: List[np.ndarray] = []
    token_to_node: List[Dict[str, int]] = []
    merged_from: List[int] = []
    offset = 0
    for column_index, keys in enumerate(uniques):
        block = value_vectors[offset:offset + len(keys)]
        offset += len(keys)
        assignment, n_clusters = _leader_merge(
            block, counts[column_index], keys, cfg.merge_threshold,
            cfg.merge_similarity, cfg.merge_strict)
        base = len(node_vectors)
        sums = np.zeros((n_clusters, cfg.embed_dim), dtype=np.float32)
        members = np.zeros(n_clusters, dtype=np.int64)
        for local, cluster in enumerate(assignment):
            sums[cluster] += block[local]
            members[cluster] += 1
        for cluster in range(n_clusters):
            vector = sums[cluster] / max(1, int(members[cluster]))
            norm = float(np.linalg.norm(vector))
            node_vectors.append(vector / norm if norm > 0 else vector)
            merged_from.append(int(members[cluster]))
        token_to_node.append(
            {key: base + int(assignment[local]) for local, key in enumerate(keys)})

    n_values = len(node_vectors)

    # 4. lay out nodes: columns, then rows, then values
    n_nodes = n_cols + n_rows + n_values
    features = np.zeros((n_nodes, cfg.embed_dim), dtype=np.float32)
    if n_cols:
        features[:n_cols] = column_vectors
    if cfg.row_init == "zero":
        pass                                    # PAPER S5.3
    elif cfg.row_init == "random":
        rng = np.random.default_rng(cfg.seed)
        features[n_cols:n_cols + n_rows] = (
            rng.normal(size=(n_rows, cfg.embed_dim)).astype(np.float32)
            / np.sqrt(cfg.embed_dim))
    else:
        raise ValueError(
            f"row_init {cfg.row_init!r} is not part of the frozen variant")
    if n_values:
        features[n_cols + n_rows:] = np.stack(node_vectors)

    node_type = np.empty(n_nodes, dtype=np.int8)
    node_type[:n_cols] = COLUMN
    node_type[n_cols:n_cols + n_rows] = ROW
    node_type[n_cols + n_rows:] = VALUE

    # 5. edges
    value_base = n_cols + n_rows
    sources: List[int] = []
    targets: List[int] = []
    for column_index in range(n_cols):
        for node in sorted(set(token_to_node[column_index].values())):
            sources.append(column_index)
            targets.append(value_base + node)
    for row_index in range(n_rows):
        row_node = n_cols + row_index
        for column_index in range(n_cols):
            token = tokens_per_column[column_index][row_index]
            if token is None:
                continue
            sources.append(row_node)
            targets.append(value_base + token_to_node[column_index][token])

    if sources:
        forward = np.array([sources, targets], dtype=np.int64)
        edge_index = np.concatenate([forward, forward[::-1]], axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return TableGraph(features=features, edge_index=edge_index, n_cols=n_cols,
                      n_rows=n_rows, n_values=n_values,
                      column_labels=headers, node_type=node_type,
                      merged_from=merged_from)


def batch_graphs(graphs: Sequence[TableGraph]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Disjoint union of several graphs, plus each graph's column-node indices.

    Message passing never crosses a table boundary, so the union is exactly
    equivalent to running the encoder on each graph separately -- it is only a
    way to spend one forward pass instead of sixteen.
    """
    features, edges, column_indices = [], [], []
    offset = 0
    for graph in graphs:
        features.append(graph.features)
        if graph.edge_index.size:
            edges.append(graph.edge_index + offset)
        column_indices.append(np.arange(graph.n_cols, dtype=np.int64) + offset)
        offset += graph.n_nodes
    stacked = (np.concatenate(features, axis=0) if features
               else np.zeros((0, 0), dtype=np.float32))
    edge_index = (np.concatenate(edges, axis=1) if edges
                  else np.zeros((2, 0), dtype=np.int64))
    return stacked, edge_index, column_indices
