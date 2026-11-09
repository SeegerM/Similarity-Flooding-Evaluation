"""From two tables to one score matrix.

The primary artefact is the **raw cosine** matrix over final column embeddings:

    S_ij = cos( h_final(s_i), h_final(t_j) )

That is deliberate.  This project's evaluation framework separates score
generation from correspondence extraction and runs every matcher's matrix
through the *same* extractors, so a competitor that ships its own selection
policy must be split the same way or the comparison confounds scoring quality
with selection policy.  SemStruct's own transformation -- softmax-normalised
cosine followed by Magneto's bipartite reranking (PAPER S4.2.1) -- is
reproduced here as a secondary, system-faithful arm rather than folded into the
primary matrix.

The reranker is a transcription of `code/magneto-matcher/.../bp_reranker.py`:
solve the assignment problem, keep the assigned pairs' scores, and rescale
every other pair to sit just below the smallest assigned score with its
relative order intact.
"""

from __future__ import annotations

import numpy as np
import torch

from .graph import TableGraph
from .model import SemStructEncoder


def column_embeddings(model: SemStructEncoder, graph: TableGraph,
                      structural: bool = True) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(graph.features)).float()
    edges = torch.from_numpy(np.ascontiguousarray(graph.edge_index))
    columns = torch.arange(graph.n_cols, dtype=torch.int64)
    with torch.no_grad():
        out = model(x, edges, columns, structural=structural)
    return out.numpy()


def cosine_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.size == 0 or target.size == 0:
        return np.zeros((source.shape[0], target.shape[0]), dtype=float)
    a = source / np.clip(np.linalg.norm(source, axis=1, keepdims=True), 1e-12, None)
    b = target / np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-12, None)
    return (a @ b.T).astype(float)


def score_matrix(model: SemStructEncoder, source: TableGraph, target: TableGraph,
                 structural: bool = True) -> np.ndarray:
    return cosine_matrix(column_embeddings(model, source, structural),
                         column_embeddings(model, target, structural))


def row_softmax(S: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """PAPER S4.2.1, "softmax-normalized cosine similarity".

    Row-wise, over the target columns, at temperature 1.  The axis and the
    temperature are both unspecified upstream; PREREG, and it matters only for
    the secondary arm -- a row-wise monotone transform leaves every extractor
    in this harness that is row-local unchanged.
    """
    if S.size == 0:
        return S
    scaled = S / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def bipartite_rerank(S: np.ndarray) -> np.ndarray:
    """Magneto's `arrange_bipartite_matches`, on a dense matrix.

    Assigned pairs keep their score; every other pair is scaled to sit just
    below the smallest assigned score, preserving relative order.
    """
    from scipy.optimize import linear_sum_assignment
    if S.size == 0:
        return S
    out = S.astype(float).copy()
    rows, cols = linear_sum_assignment(out, maximize=True)
    assigned = np.zeros(out.shape, dtype=bool)
    assigned[rows, cols] = True
    if not assigned.any() or assigned.all():
        return out
    smallest_assigned = float(out[assigned].min())
    largest_other = float(out[~assigned].max())
    scale = ((smallest_assigned - 0.01) / largest_other) if largest_other > 0 else 1.0
    out[~assigned] = out[~assigned] * scale
    return out


def native_matrix(S: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """The paper's own pipeline: softmax, then bipartite reranking."""
    return bipartite_rerank(row_softmax(S, temperature))
