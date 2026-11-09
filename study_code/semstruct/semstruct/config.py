"""The frozen SemStruct-reimpl configuration, with provenance for every value.

Kang, Srinivas, Mihindukulasooriya, Shirai, Ram, Samulowitz and Seneviratne,
*"SemStruct: Contextualizing Semantic Embeddings with Structural Information
for Schema Matching"*, KDD '26 (`harness/papers/3770855.3817963.pdf`).

There is no released implementation (see `harness/SEMSTRUCT_REIMPLEMENTATION.md`
S1), so this is a reconstruction from the text.  Every field below carries one
of three provenance markers, and the marker is the point of this file:

    PAPER    quoted or unambiguously implied by the paper
    PREREG   the paper is silent; fixed here before any evaluation scenario
             was scored, and justified in SEMSTRUCT_REIMPLEMENTATION.md S3
    COST     fixed by a declared cost rule, not by accuracy

Nothing in this file may be changed after a matrix has been scored without
recording the change as an amendment in that document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class SemStructConfig:
    # ---------------------------------------------------------------- semantic
    # PAPER A.1.2: "the frozen e5-small checkpoint".
    plm: str = "intfloat/e5-small"
    # PREREG.  E5 requires an asymmetric instruction prefix and the paper does
    # not say which.  The E5 model card prescribes "query: " on both sides for
    # symmetric similarity tasks, which is what column-to-column matching is.
    e5_prefix: str = "query: "
    # PREREG.  Mean pooling over the attention mask then L2 normalisation is
    # E5's own published convention; the paper does not restate it.
    pool: str = "mean"
    l2_normalize_features: bool = True
    # PREREG.  Long text values exist in SOTAB-style data; 64 word-pieces holds
    # a header plus a value comfortably and bounds the encoder cost.
    max_tokens: int = 64
    embed_batch: int = 128
    embed_dim: int = 384          # PAPER A.1.2, implied by e5-small

    # ------------------------------------------------------------ serialisation
    # PAPER Table 1 + S5.4: the `header` strategy, which their own ablation
    # ranks best overall.  One variant only -- see SEMSTRUCT_REIMPLEMENTATION S2.
    column_serialization: str = "header"
    # PAPER S3.2.2: "column: {}, value: {}".
    value_template: str = "column: {column}, value: {value}"

    # ------------------------------------------------------------------- graph
    n_bins: int = 8               # PAPER S3.1 and A.1.2: k = 8
    merge_threshold: float = 0.9  # PAPER S3.2.1: tau = 0.9
    # PREREG.  *What* the threshold is a threshold on.  The paper says "cosine
    # similarity" without saying on which scale, and the scale is a property of
    # the checkpoint rather than of the rule: on frozen e5-small, arbitrary
    # short strings sit at cosine 0.92-0.98, so a literal tau = 0.9 merges
    # 76-96% of all value *pairs* and keeps only 3.8% of value tokens as nodes,
    # emptying the graph of value distinctions; the same threshold on
    # column-centred cosine keeps 96.1% and makes the step a no-op.  Measured
    # over the whole class sample in SEMSTRUCT_REIMPLEMENTATION.md S4.2/S4.3.
    # "raw" is the literal reading and stays the primary arm; "centred" is the
    # anisotropy-corrected reading the paper's own citation [6] motivates.
    # A threshold above 1.0 disables merging entirely.
    merge_similarity: str = "raw"       # "raw" | "centered"
    # PAPER S3.2/S5.3: zero-init row nodes ("topological conduits").
    row_init: str = "zero"
    # PREREG.  Value nodes are keyed per column, not globally: the paper's own
    # value serialisation embeds the column name, so two identical strings in
    # different columns are different nodes by construction.
    value_scope: str = "per_column"
    # PREREG.  Merging is within a column and leader-based (no transitive
    # chaining); see SEMSTRUCT_REIMPLEMENTATION S3.
    merge_scope: str = "per_column"
    # PREREG.  A null cell contributes no value node and no edge.  The
    # alternative -- one shared null node -- is a hub adjacent to most rows.
    drop_nulls: bool = True
    # COST.  Rows read per table, prefix.  Set by the cost rule in
    # SEMSTRUCT_REIMPLEMENTATION S4, not by accuracy.
    row_cap: int = 400

    # --------------------------------------------------------------------- GNN
    gnn: str = "sage"             # PAPER A.1.2: GraphSAGE backbone
    n_layers: int = 3             # PAPER A.1.2
    hidden_dim: int = 384         # PAPER A.1.2
    mlp_layers: int = 2           # PAPER Eq. 2: "a simple 2-layer feedforward"
    # PREREG.  PyTorch Geometric SAGEConv defaults, reimplemented in plain
    # torch: mean aggregation, root weight, bias, no normalisation, no dropout,
    # ReLU between layers and none after the last.
    activation: str = "relu"
    dropout: float = 0.0
    sage_normalize: bool = False

    # ---------------------------------------------------------------- training
    optimizer: str = "adam"       # PAPER A.1.2
    lr: float = 5e-4              # PAPER A.1.2, Valentine
    epochs: int = 10              # PAPER A.1.2, Valentine
    batch_tables: int = 16        # PAPER A.1.2: "batch size of 16 tables"
    margin: float = 0.2           # PAPER S3.4.1
    distance: str = "cosine"      # PAPER S3.4.1
    mining: str = "hardest"       # PAPER S3.4.1
    # PREREG.  Adam's torch defaults; the paper names the optimiser only.
    weight_decay: float = 0.0
    grad_clip: float = 1.0        # PREREG: stability, not tuned
    # PREREG.  Views per source table, and the sampling rates for a view.  The
    # paper says "randomly sampling rows and columns" without rates.
    views_per_table: int = 4
    view_row_frac: Tuple[float, float] = (0.5, 1.0)
    view_col_frac: Tuple[float, float] = (0.5, 1.0)
    view_min_cols: int = 2
    p_header_noise: float = 0.5   # PREREG: per kept column
    p_value_noisy_column: float = 0.3
    p_value_noise_cell: float = 0.3

    # ------------------------------------------------------------------ scoring
    # PREREG.  The primary artefact is the raw cosine matrix, handed to the
    # harness's common extractors.  The paper's own softmax + bipartite
    # reranking is available as a secondary, system-faithful arm.
    score: str = "cosine"

    # -------------------------------------------------------------- determinism
    seed: int = 20260908

    # ------------------------------------------- information-matched ablation
    # harness/SEMSTRUCT_ABLATION.md (2026-10-27, specified in advance).  Both are
    # model-side only: GraphStore leaves them out of the graph-cache key, so the
    # cached graphs are reused.  Defaults reproduce the 2026-09-08 behaviour.
    # "default": PyTorch's init of the MLP's last layer; "zero": zero-initialise
    # it, so training starts from the LM-only control (repair R1 of the collapse).
    delta_init: str = "default"
    # False drops the row-value edges after loading a graph: a column then sees
    # only its own values (with n_layers = 1, the one-hop information-matched arm).
    row_value_edges: bool = True
    # Repair R2 (tried only because R1 failed its pre-stated rule): subtract each graph's mean column/value feature
    # (row nodes stay zero), against the anisotropy of the frozen features.  Applied after loading, like the above.
    center_features: bool = False

    # ------------------------------------- Amendment 2: the revised training setup
    # harness/SEMSTRUCT_ABLATION.md, Amendment 2 (2026-10-28).  The defaults reproduce every earlier run;
    # AUTHORS below collects the Amendment 2 values.
    # "hardest": the farthest positive (PAPER S3.4.1); "easy": the closest positive (Amendment 2).
    positive_mining: str = "hardest"
    # "groups": 8 source tables x (original + one view); "one_group": 16 views of one source table.
    batch_mode: str = "groups"
    # Dropout between the MLP's two layers (Amendment 2: 0.3).
    mlp_dropout: float = 0.0
    # "frac": rows and columns as fractions (view_row_frac, view_col_frac); "authors": rows a uniform
    # integer in view_row_range clipped to the table, columns a uniform integer in [ceil(n/2), n].
    view_sampling: str = "frac"
    view_row_range: Tuple[int, int] = (100, 2000)
    # Merge only above the threshold (Amendment 2: strictly greater); False keeps the earlier ">=".
    merge_strict: bool = False
    # The ValAvg arm (PAPER S5.2): the mean of the column's column-conditioned value
    # embeddings through the same MLP and residual, no GNN; needs row_value_edges = False.
    valavg: bool = False
    # The MLP rung of PAPER Fig. 5: the same MLP and residual on the frozen column embedding itself, no values, no GNN.
    mlp_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = SemStructConfig()

# Amendment 2: the revised training setup (2026-10-28).
AUTHORS = SemStructConfig(
    plm="intfloat/e5-small-v2", e5_prefix="",
    positive_mining="easy", batch_mode="one_group", mlp_dropout=0.3,
    view_sampling="authors", view_row_range=(100, 2000),
    p_header_noise=0.3, p_value_noisy_column=1.0, p_value_noise_cell=0.15,
    row_cap=2000, merge_strict=True)

# Fields added by Amendment 2, with their defaults; GraphStore and the run script's fingerprint drop
# them at these values so that every earlier cache key and checkpoint fingerprint is unchanged.
AMENDMENT2_DEFAULTS = {
    "positive_mining": "hardest", "batch_mode": "groups", "mlp_dropout": 0.0,
    "view_sampling": "frac", "view_row_range": (100, 2000), "merge_strict": False, "valavg": False,
    "mlp_only": False}
