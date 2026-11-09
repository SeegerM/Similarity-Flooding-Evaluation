"""REMA-ARC: a bounded reproduction of Koutras et al., SEAData 2020.

`papers/SEAData5.pdf`. There is no released implementation -- the paper is
four pages, the TU Delft record carries no files, and the authors' pages link
Valentine and OmniMatch but not this. So this is a reconstruction from the
text, and `REMA_REPRODUCTION.md` records which lines come from the paper and
which are our declared defaults. Read that document before reading this file.

The pipeline, in the paper's own four stages:

  1. encode both relations into ONE undirected Attribute-Row-Cell graph
  2. generate random walks over it
  3. train skip-gram embeddings on the walks
  4. column vector := mean of its cell-value embeddings; score := cosine

Stage 1 is the whole method. A cell value occurring in several tuples is *one*
node, so identical values shared by the two tables are the bridges that put
both relations in a single connected component and therefore in a single
embedding space (paper S3.1, "Remarks and Node Merging"). Two consequences that
matter more here than in the original paper:

  * REMA-ARC reads no attribute labels. Attribute nodes are identified by
    (side, position), never by name, so nothing in the graph knows what a
    column is called. It is a pure instance-based matcher.
  * It is therefore entirely carried by cross-table value identity. Under a
    per-side-salted recoding the two relations share no value, the graph falls
    into two components, and cosine similarity across them is whatever the
    random initialisation and negative sampling left behind. That is a
    prediction, not a defect, and `REMA_REPRODUCTION.md` states it in advance.

Needs `gensim`. In this checkout only `.venv-bdi` has it built for cp310:

    harness/.venv-bdi/Scripts/python.exe -m ...
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

# Paper S4.1, "Parameter tuning": 10 walks of length 100 per node, 128
# dimensions, window 10, skip-gram. These five are quoted, not chosen.
WALKS_PER_NODE = 10
WALK_LENGTH = 100
DIMENSIONS = 128
WINDOW = 10
SKIP_GRAM = 1

# Not in the paper. Declared in REMA_REPRODUCTION.md, fixed here.
EPOCHS = 1              # the node2vec lineage's default, not gensim's 5
NEGATIVE = 5            # gensim's default
MIN_COUNT = 1           # every cell value must keep a vector
SEED = 20260908
WALK_CHUNK = 20000      # walks per generated block; fixes the random stream

ROW = 0
VALUE = 1
ATTRIBUTE = 2

NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}


def is_null(value: str) -> bool:
    """The project's null rule, identical to `make_md5_audited.is_null`.

    Shared so that a cell counted as absent in the recoded corpus is counted
    as absent here too. It also keeps the empty string out of the graph: a
    single node adjacent to almost every row would be a hub that every walk
    passes through, which is the FINDINGS 33 failure in graph form.
    """
    return value.strip().lower() in NULLS


@dataclass(frozen=True)
class ArcGraph:
    """An undirected simple graph in CSR form, plus what each node is.

    `indptr`/`indices` are the usual CSR pair. `kind` labels every node ROW,
    VALUE or ATTRIBUTE. `value_nodes` gives, per side and per column, the
    node ids of that column's distinct non-null cell values -- which is
    exactly what stage 4 averages.
    """

    indptr: np.ndarray
    indices: np.ndarray
    kind: np.ndarray
    value_nodes: tuple[list[list[int]], list[list[int]]]
    n_rows_used: tuple[int, int]

    @property
    def n_nodes(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def n_edges(self) -> int:
        return int(self.indices.size // 2)

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def counts(self) -> dict[str, int]:
        return {"rows": int((self.kind == ROW).sum()),
                "values": int((self.kind == VALUE).sum()),
                "attributes": int((self.kind == ATTRIBUTE).sum()),
                "nodes": self.n_nodes, "edges": self.n_edges,
                "isolated": int((self.degree == 0).sum())}


def build_arc_graph(source: Sequence[Sequence[str]],
                    target: Sequence[Sequence[str]],
                    row_cap: int = 0) -> ArcGraph:
    """Stage 1. Both relations, one graph.

    `source`/`target` are column-major: `side[j][i]` is row `i` of column `j`,
    as `valentine_sweep.strict_columns` returns them.

    Edges, both directions stored:
        row node       -- every non-null cell value in that tuple  (RC graph)
        attribute node -- every distinct non-null value in that column  (ARC)

    A cell value is keyed by its stripped string and shared globally: across
    tuples, across columns and across the two relations. Multi-edges are
    collapsed, so a tuple holding the same value twice contributes one edge.
    """
    sides = (source, target)
    for side in sides:
        if side and len({len(c) for c in side}) != 1:
            raise ValueError("ragged table: columns differ in length")

    rows_used = tuple(min(len(s[0]), row_cap) if s and row_cap > 0
                      else (len(s[0]) if s else 0) for s in sides)

    node_kind: list[int] = []
    value_id: dict[str, int] = {}

    def new_node(k: int) -> int:
        node_kind.append(k)
        return len(node_kind) - 1

    row_id = [[new_node(ROW) for _ in range(rows_used[s])] for s in (0, 1)]
    attr_id = [[new_node(ATTRIBUTE) for _ in side] for side in sides]

    edges: set[tuple[int, int]] = set()
    value_nodes: tuple[list[list[int]], list[list[int]]] = ([], [])

    for s, side in enumerate(sides):
        for j, column in enumerate(side):
            seen: dict[str, None] = {}
            for i in range(rows_used[s]):
                raw = column[i]
                if is_null(raw):
                    continue
                value = raw.strip()
                node = value_id.get(value)
                if node is None:
                    node = value_id[value] = new_node(VALUE)
                edges.add((row_id[s][i], node))
                seen[value] = None
            for value in seen:
                edges.add((attr_id[s][j], value_id[value]))
            value_nodes[s].append([value_id[v] for v in seen])

    n = len(node_kind)
    if edges:
        left = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        right = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        rows = np.concatenate([left, right])
        cols = np.concatenate([right, left])
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        np.cumsum(indptr, out=indptr)
    else:
        cols = np.empty(0, dtype=np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)

    return ArcGraph(indptr=indptr, indices=np.ascontiguousarray(cols),
                    kind=np.asarray(node_kind, dtype=np.int8),
                    value_nodes=value_nodes, n_rows_used=rows_used)


def high_degree_starts(graph: ArcGraph) -> np.ndarray:
    """Nodes whose degree exceeds the graph's mean degree.

    The paper's `REMA-ARC-HighDegree` starts walks only from these (S4.1), and
    reports that it gives "identical results" to starting from every node
    while costing much less. In an ARC graph the mean is dragged down by the
    long tail of values occurring once -- degree 2, one row and one attribute
    -- so this typically keeps the row and attribute nodes and drops most
    value nodes.
    """
    degree = graph.degree
    if degree.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(degree > degree.mean()).astype(np.int64)


def walk_starts(graph: ArcGraph, walks_per_node: int = WALKS_PER_NODE,
                high_degree: bool = False) -> np.ndarray:
    """The start node of every walk, in order: `walks_per_node` per node."""
    starts = high_degree_starts(graph) if high_degree \
        else np.arange(graph.n_nodes, dtype=np.int64)
    return np.repeat(starts, walks_per_node)


def _walk_block(graph: ArcGraph, block: np.ndarray, length: int,
                rng: np.random.Generator) -> np.ndarray:
    """Step one batch of walks forward, all of them at once.

    One uniform draw per active walk per step, turned into a neighbour by
    offsetting into CSR. Padding is -1, which only ever appears for a walk
    started at an isolated node: every node a walk can *reach* has degree at
    least one, so a walk cannot die part-way.
    """
    degree = graph.degree
    walks = np.full((block.size, length), -1, dtype=np.int64)
    walks[:, 0] = block
    current = block.copy()
    active = degree[current] > 0
    for step in range(1, length):
        if not active.any():
            break
        cur = current[active]
        draw = rng.random(cur.size)
        nxt = graph.indices[graph.indptr[cur]
                            + np.minimum((draw * degree[cur]).astype(np.int64),
                                         degree[cur] - 1)]
        walks[active, step] = nxt
        current[active] = nxt
    return walks


def walk_chunks(graph: ArcGraph, walks_per_node: int = WALKS_PER_NODE,
                length: int = WALK_LENGTH, seed: int = SEED,
                high_degree: bool = False, chunk: int = WALK_CHUNK
                ) -> Iterator[np.ndarray]:
    """Stage 2, in blocks. `walks_per_node` uniform walks of `length` per node.

    Chunked rather than materialised whole, because the whole thing does not
    fit. The walk budget is per node, so a whole-table TPC-DI scenario has
    around 280k nodes and 2.8M walks -- a single `(2.8e6, 100)` int64 array is
    22 GB. One chunk of 20k walks is 16 MB, and the corpus is regenerated per
    epoch instead of stored, which costs nothing: walk generation was 0.11s
    against 25s of Word2Vec at the frozen cap.

    Each chunk draws from its own `default_rng([seed, index])`, so the output
    is a deterministic function of (seed, chunk size) and does not depend on
    how much memory the machine had. `WALK_CHUNK` is therefore part of the
    frozen configuration, not a tuning knob.

    "Simple random walks", i.e. REMA-ARC, not REMA-JUST: the paper's JUST
    variant biases the next step by node type, and reproducing it needs
    transition rules the four pages do not give. `high_degree` restricts the
    *start* set only, which is the paper's HighDegree variant; the walk itself
    is unchanged and can still visit any node.
    """
    if length < 1 or walks_per_node < 1:
        raise ValueError("walks_per_node and length must be positive")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    starts = walk_starts(graph, walks_per_node, high_degree)
    for index, begin in enumerate(range(0, starts.size, chunk)):
        yield _walk_block(graph, starts[begin:begin + chunk], length,
                          np.random.default_rng([seed, index]))


def random_walks(graph: ArcGraph, walks_per_node: int = WALKS_PER_NODE,
                 length: int = WALK_LENGTH, seed: int = SEED,
                 high_degree: bool = False, chunk: int = WALK_CHUNK
                 ) -> np.ndarray:
    """Every walk as one array. For tests and small graphs; see `walk_chunks`."""
    blocks = list(walk_chunks(graph, walks_per_node, length, seed,
                              high_degree, chunk))
    if not blocks:
        return np.empty((0, length), dtype=np.int64)
    return np.vstack(blocks)


def walk_corpus_size(graph: ArcGraph, walks_per_node: int = WALKS_PER_NODE,
                     length: int = WALK_LENGTH, high_degree: bool = False
                     ) -> tuple[int, int]:
    """(walks, tokens) without generating anything.

    A walk from an isolated node is one token long and every other walk is
    `length` tokens, so the corpus size is exact arithmetic. Computed rather
    than counted because the streaming corpus is never held in memory.
    """
    starts = walk_starts(graph, walks_per_node, high_degree)
    if starts.size == 0:
        return 0, 0
    isolated = int((graph.degree[starts] == 0).sum())
    return int(starts.size), isolated + (int(starts.size) - isolated) * length


class WalkCorpus:
    """gensim wants an iterable of token lists, re-iterated once per epoch.

    Regenerates the walks on each pass from `walk_chunks` and converts them
    through a table of interned node-id strings, so each pass allocates lists
    of *existing* strings rather than 100 new ones per walk, and no pass holds
    more than one chunk. Storing the corpus as Python strings instead would
    cost more than an order of magnitude in memory at the sizes we run.
    """

    def __init__(self, graph: ArcGraph, walks_per_node: int = WALKS_PER_NODE,
                 length: int = WALK_LENGTH, seed: int = SEED,
                 high_degree: bool = False, chunk: int = WALK_CHUNK):
        self._graph = graph
        self._kwargs = {"walks_per_node": walks_per_node, "length": length,
                        "seed": seed, "high_degree": high_degree,
                        "chunk": chunk}
        self._tokens = [str(i) for i in range(graph.n_nodes)]
        self._walks, self._token_count = walk_corpus_size(
            graph, walks_per_node, length, high_degree)

    def __len__(self) -> int:
        return self._walks

    @property
    def token_count(self) -> int:
        return self._token_count

    def __iter__(self) -> Iterator[list[str]]:
        tokens = self._tokens
        for block in walk_chunks(self._graph, **self._kwargs):
            for walk in block:
                row = walk[walk >= 0]
                if row.size:
                    yield [tokens[i] for i in row.tolist()]


def stable_hash(token: str) -> int:
    """A hash that does not move between processes.

    gensim seeds each word's initial vector from `hash(word)`, and CPython
    salts `str.__hash__` per process, so the default makes an otherwise fully
    seeded run irreproducible. This replaces it with SHA-256, which is why two
    runs of this module produce byte-identical matrices.
    """
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:7], "big")


def train_embeddings(corpus: WalkCorpus, dimensions: int = DIMENSIONS,
                     window: int = WINDOW, epochs: int = EPOCHS,
                     seed: int = SEED, workers: int = 1):
    """Stage 3. Skip-gram Word2Vec over the walks.

    `workers=1` is not a performance choice: gensim's multi-worker training
    interleaves batches non-deterministically, and a matcher whose score
    matrix changes between identical runs cannot be compared against the
    frozen artifacts in this project.
    """
    from gensim.models import Word2Vec

    return Word2Vec(sentences=corpus, vector_size=dimensions,
                    window=window, sg=SKIP_GRAM, negative=NEGATIVE,
                    min_count=MIN_COUNT, epochs=epochs, seed=seed,
                    workers=workers, hashfxn=stable_hash)


def column_vectors(model, nodes_per_column: Sequence[Sequence[int]],
                   dimensions: int = DIMENSIONS) -> np.ndarray:
    """Stage 4a. A column is the mean of its distinct cell-value embeddings.

    Paper S3.4: "we derive the vector representation of each column as the
    mean of all their corresponding cell value embeddings". One node per
    distinct value, so the mean is over distinct values, unweighted by how
    often each occurs -- declared in REMA_REPRODUCTION.md.

    An all-null column has no cell values and gets the zero vector, which
    scores 0 against everything rather than raising.
    """
    key = model.wv
    out = np.zeros((len(nodes_per_column), dimensions), dtype=np.float64)
    for j, nodes in enumerate(nodes_per_column):
        vectors = [key[str(node)] for node in nodes if str(node) in key]
        if vectors:
            out[j] = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
    return out


def cosine_scores(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stage 4b. Raw cosine, in [-1, 1]; a zero vector scores 0.

    Deliberately *not* clipped to [0, 1] the way this project's own channels
    are. Clipping collapses every negative cell to an exact tie, and a tied
    block is the one thing the harness's extractors are most sensitive to
    (`test_exact_tie_block_inflates_match_set`). The reported extractors --
    Hungarian and mutual-top1 -- are invariant to monotone rescaling, so
    nothing in the headline depends on this, and the artifact stays faithful.
    """
    ln = np.linalg.norm(left, axis=1, keepdims=True)
    rn = np.linalg.norm(right, axis=1, keepdims=True)
    left = left / np.where(ln == 0, 1, ln)
    right = right / np.where(rn == 0, 1, rn)
    return left @ right.T


def greedy_order_based(scores: np.ndarray, median_threshold: bool = False
                       ) -> np.ndarray:
    """REMA's own extractor, for the native-versus-common comparison.

    Paper S3.4: sort all pairs by similarity, walk down the list, materialise
    a pair only if both of its columns are still unmatched. The optional
    extension keeps only pairs scoring at least the median of all pairwise
    similarities.

    Kept here rather than added to `smx.core.EXTRACTORS`, which every other
    script iterates by default and which must not change under them.
    """
    out = np.zeros(scores.shape, dtype=bool)
    if scores.size == 0:
        return out
    floor = float(np.median(scores)) if median_threshold else -np.inf
    order = np.argsort(-scores, axis=None, kind="stable")
    rows, cols = np.unravel_index(order, scores.shape)
    used_r, used_c = set(), set()
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i in used_r or j in used_c:
            continue
        if scores[i, j] < floor:
            break
        out[i, j] = True
        used_r.add(i)
        used_c.add(j)
    return out


def rema_arc(source: Sequence[Sequence[str]], target: Sequence[Sequence[str]],
             row_cap: int = 0, seed: int = SEED, epochs: int = EPOCHS,
             workers: int = 1, high_degree: bool = False
             ) -> tuple[np.ndarray, dict]:
    """The whole pipeline: two column-major tables in, score matrix out.

    Returns the matrix and a provenance record -- graph size, corpus size and
    every parameter -- so a run can be audited without re-reading the code.
    """
    graph = build_arc_graph(source, target, row_cap=row_cap)
    corpus = WalkCorpus(graph, seed=seed, high_degree=high_degree)
    model = train_embeddings(corpus, epochs=epochs, seed=seed, workers=workers)
    left = column_vectors(model, graph.value_nodes[0])
    right = column_vectors(model, graph.value_nodes[1])

    shared = len(set(v for col in graph.value_nodes[0] for v in col)
                 & set(v for col in graph.value_nodes[1] for v in col))
    info = {"graph": graph.counts(),
            "rows_used": list(graph.n_rows_used),
            "shared_value_nodes": shared,
            "walks": len(corpus),
            "walk_tokens": corpus.token_count,
            "vocabulary": len(model.wv),
            "params": {"walks_per_node": WALKS_PER_NODE,
                       "walk_length": WALK_LENGTH, "dimensions": DIMENSIONS,
                       "window": WINDOW, "sg": SKIP_GRAM, "negative": NEGATIVE,
                       "min_count": MIN_COUNT, "epochs": epochs,
                       "seed": seed, "workers": workers, "row_cap": row_cap,
                       "high_degree": high_degree, "walk_chunk": WALK_CHUNK}}
    return cosine_scores(left, right), info
