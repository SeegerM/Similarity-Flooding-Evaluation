"""The structural encoder: GraphSAGE + a 2-layer MLP behind a residual.

PAPER A.1.2 fixes a 3-layer GraphSAGE backbone at hidden dimension 384, and
PAPER Eq. 2 fixes the read-out:

    h_c_final = h_c_(0) + MLP(h_c_(L))

The paper runs on PyTorch Geometric.  PyG is not installed in this checkout and
adding it would pull a CUDA-oriented dependency tree for a model that is four
matrix multiplies, so `SAGEConv` below is a plain-torch reimplementation of
PyG's `SAGEConv` **at its defaults**, which is what the paper's one-line
description leaves us:

    out_i = lin_l( mean_{j in N(i)} x_j ) + lin_r( x_i )

with `lin_l` carrying the bias, `lin_r` not, `aggr='mean'`, `root_weight=True`,
`normalize=False`, `project=False`.  Isolated nodes aggregate to zero, exactly
as PyG's scatter-mean does.  That equivalence is asserted by
`tests/test_semstruct.py::test_sage_matches_reference`, which recomputes one
layer from the definition rather than from this implementation.

The residual is the reason the zero-delta control in
SEMSTRUCT_REIMPLEMENTATION.md S6 is meaningful: zero the MLP's last layer and
the model must reproduce the frozen semantic initialisation bit for bit.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .config import SemStructConfig


def mean_aggregate(x: Tensor, edge_index: Tensor) -> Tensor:
    """Mean of each node's neighbours; zero where a node has none."""
    n_nodes = x.shape[0]
    out = torch.zeros_like(x)
    if edge_index.numel() == 0:
        return out
    source, target = edge_index[0], edge_index[1]
    out.index_add_(0, target, x.index_select(0, source))
    degree = torch.zeros(n_nodes, dtype=x.dtype, device=x.device)
    degree.index_add_(0, target, torch.ones_like(source, dtype=x.dtype))
    return out / degree.clamp(min=1.0).unsqueeze(-1)


class SAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, normalize: bool = False) -> None:
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=True)     # neighbours
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)    # root
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        out = self.lin_l(mean_aggregate(x, edge_index)) + self.lin_r(x)
        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2.0, dim=-1)
        return out


class SemStructEncoder(nn.Module):
    """Frozen semantic features in, refined column embeddings out."""

    def __init__(self, cfg: SemStructConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.gnn != "sage":
            raise ValueError(
                f"gnn {cfg.gnn!r} is not part of the frozen variant; the paper's "
                "GAT/HGT arms are excluded, see SEMSTRUCT_REIMPLEMENTATION.md S2")
        dims = [cfg.embed_dim] + [cfg.hidden_dim] * cfg.n_layers
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1], normalize=cfg.sage_normalize)
            for i in range(cfg.n_layers)])
        if cfg.mlp_layers != 2:
            raise ValueError("PAPER Eq. 2 fixes a 2-layer MLP")
        # Amendment 2: the MLP has Dropout(0.3) between its layers; at the default 0 the module list,
        # and so every earlier checkpoint's state-dict keys, is unchanged.
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            *([nn.Dropout(cfg.mlp_dropout)] if cfg.mlp_dropout else []),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )
        if cfg.valavg and cfg.row_value_edges:
            raise ValueError("valavg averages a column's values: it needs row_value_edges = False")
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout else None

    def forward(self, x: Tensor, edge_index: Tensor,
                column_index: Tensor, structural: bool = True) -> Tensor:
        """Return the final column embeddings, Eq. 2.

        `structural=False` is the LM-only control (S6 check 1): it skips the
        GNN and the MLP entirely and returns the frozen semantic vectors, so
        the control is guaranteed to be *this* pipeline minus structure rather
        than a separately written baseline that might differ in serialisation.
        """
        semantic = x.index_select(0, column_index)
        if not structural:
            return semantic
        if self.cfg.mlp_only:
            # PAPER Fig. 5, the MLP rung: the trained MLP on the frozen column embedding alone (no values, no GNN)
            return semantic + self.mlp(semantic)
        if self.cfg.valavg:
            # PAPER S5.2 ValAvg: without row edges a column's neighbours are exactly
            # its (column-conditioned) value nodes, so their mean is the centroid; same MLP and residual.
            centroid = mean_aggregate(x, edge_index).index_select(0, column_index)
            return semantic + self.mlp(centroid)
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = torch.relu(h)
                if self.dropout is not None:
                    h = self.dropout(h)
        return semantic + self.mlp(h.index_select(0, column_index))

    @torch.no_grad()
    def zero_structural_delta(self) -> None:
        """Force MLP(.) == 0, for the S6 check-3 control."""
        last = self.mlp[-1]
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()


def build_model(cfg: SemStructConfig, seed: Optional[int] = None) -> SemStructEncoder:
    torch.manual_seed(cfg.seed if seed is None else seed)
    model = SemStructEncoder(cfg)
    if cfg.delta_init == "zero":
        # repair R1 (harness/SEMSTRUCT_ABLATION.md): start as the LM-only control
        model.zero_structural_delta()
    elif cfg.delta_init != "default":
        raise ValueError(f"delta_init {cfg.delta_init!r}")
    return model
