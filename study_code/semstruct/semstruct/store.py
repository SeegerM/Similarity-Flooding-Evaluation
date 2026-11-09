"""A content-addressed cache of built graphs.

Graph construction is dominated by the frozen encoder, and the encoder is a
pure function of the text, so a graph is a pure function of (table content,
configuration).  Caching it keyed on exactly that pair is what makes training
affordable on CPU: a view that reuses 90% of its parent's values pays for the
10% it perturbed, and a table appearing in several scenarios is built once.

The key covers every configuration field that can change a graph.  A field
added to `SemStructConfig` is therefore *automatically* part of the key --
which is the safe direction to fail, since a stale hit would be silent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from dataclasses import replace

from .config import AMENDMENT2_DEFAULTS, DEFAULT, SemStructConfig
from .encoder import Encoder
from .graph import ROW, TableGraph, build_graph
from .tables import table_digest

# Fields added for the 2026-10-27 ablation that do not change a graph (left out of the cache key).
MODEL_ONLY_NEW = ("delta_init", "row_value_edges", "center_features")


class GraphStore:
    def __init__(self, cfg: SemStructConfig, encoder: Encoder,
                 cache_dir: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, headers: Sequence[str], columns: Sequence[Sequence[str]]) -> str:
        # Model-side fields of the 2026-10-27 ablation do not change a graph: the two
        # new fields are left out and n_layers is keyed at its default, so every key
        # of a default configuration is exactly what it was before the ablation.
        fields = self.cfg.to_dict()
        for name in MODEL_ONLY_NEW:
            fields.pop(name, None)
        fields["n_layers"] = DEFAULT.n_layers
        fields["mining"] = DEFAULT.mining          # Amendment 1: model-side only
        # Amendment 2: model- and view-side fields never change a graph built from given cells; the one
        # graph-side field (merge_strict) enters the key only when it differs from its default.
        for name in ("positive_mining", "batch_mode", "mlp_dropout", "view_sampling", "view_row_range", "valavg",
                     "mlp_only"):
            fields.pop(name, None)
        if fields.get("merge_strict") == AMENDMENT2_DEFAULTS["merge_strict"]:
            fields.pop("merge_strict", None)
        payload = json.dumps(fields, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8"))
        digest.update(table_digest(headers, columns).encode("ascii"))
        return digest.hexdigest()

    def get(self, headers: Sequence[str], columns: Sequence[Sequence[str]]
            ) -> TableGraph:
        if not self.cache_dir:
            self.misses += 1
            return build_graph(headers, columns, self.cfg, self.encoder)
        path = self.cache_dir / f"{self._key(headers, columns)}.npz"
        if path.exists():
            try:
                graph = _load(path)
                self.hits += 1
                return self._view(graph)
            except (OSError, ValueError, KeyError):
                pass                      # a truncated cache file is not fatal
        self.misses += 1
        graph = build_graph(headers, columns, self.cfg, self.encoder)
        _save(path, graph)
        return self._view(graph)

    def _view(self, graph: TableGraph) -> TableGraph:
        """The graph the model sees: without row-value edges for the one-hop arm."""
        if not self.cfg.row_value_edges:
            graph = drop_row_edges(graph)
        if self.cfg.center_features:
            graph = center_graph(graph)
        return graph


def center_graph(graph: TableGraph) -> TableGraph:
    """Subtract the graph's mean column/value feature from its column and value nodes; row nodes stay as they are."""
    keep = graph.node_type != ROW
    if not keep.any():
        return graph
    features = graph.features.copy()
    features[keep] = features[keep] - features[keep].mean(axis=0, keepdims=True)
    return replace(graph, features=features)


def drop_row_edges(graph: TableGraph) -> TableGraph:
    """Keep only edges between columns and values: drop every edge with a row-node end."""
    e = graph.edge_index
    if e.size == 0:
        return graph
    lo, hi = graph.n_cols, graph.n_cols + graph.n_rows
    keep = ~(((e[0] >= lo) & (e[0] < hi)) | ((e[1] >= lo) & (e[1] < hi)))
    return replace(graph, edge_index=e[:, keep])


def _save(path: Path, graph: TableGraph) -> None:
    temporary = path.with_suffix(f".{id(graph)}.tmp.npz")
    np.savez_compressed(
        temporary,
        features=graph.features,
        edge_index=graph.edge_index,
        node_type=graph.node_type,
        shape=np.array([graph.n_cols, graph.n_rows, graph.n_values], dtype=np.int64),
        column_labels=np.array(graph.column_labels, dtype=object),
        merged_from=np.array(graph.merged_from or [], dtype=np.int64),
    )
    temporary.replace(path)


def _load(path: Path) -> TableGraph:
    with np.load(path, allow_pickle=True) as data:
        n_cols, n_rows, n_values = (int(x) for x in data["shape"])
        return TableGraph(
            features=data["features"],
            edge_index=data["edge_index"],
            n_cols=n_cols, n_rows=n_rows, n_values=n_values,
            column_labels=[str(x) for x in data["column_labels"].tolist()],
            node_type=data["node_type"],
            merged_from=data["merged_from"].tolist(),
        )
