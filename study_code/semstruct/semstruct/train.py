"""Self-supervised training of the structural encoder (PAPER S3.4).

The objective is the paper's, restated exactly:

    L = sum_{a in B} [ max_{p in P} dist(a, p) - min_{n in N} dist(a, n) + alpha ]

with cosine distance, alpha = 0.2, and hardest-positive / hardest-negative
mining inside the batch (PAPER S3.4.1).  Supervision comes only from synthetic
views of *source* tables: the evaluation targets are never built, never
embedded and never seen, which is the paper's own leakage rule (S3.4).

Two things here are ours, both in `config.py` under PREREG and argued in
SEMSTRUCT_REIMPLEMENTATION.md S3:

  * the batch is 16 graphs formed as 8 source tables x (original + one view),
    which is the smallest construction satisfying both "batch size of 16
    tables" (A.1.2) and "every table has at least one positive match" (S3.4.2);
  * the anchor sum is a **mean**, not a sum.  With a sum, the effective step
    size would scale with how many columns the sampled tables happen to have,
    so the paper's single learning rate would mean different things on ChEMBL
    and OpenData.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .augment import View, make_views
from .config import SemStructConfig
from .graph import batch_graphs
from .model import SemStructEncoder, build_model
from .store import GraphStore


@dataclass
class TrainingGroup:
    """One source table and its views; the unit of positive supervision."""
    table_id: str
    headers: List[str]
    columns: List[List[str]]
    views: List[View] = field(default_factory=list)


def build_groups(tables: Sequence[Tuple[str, List[str], List[List[str]]]],
                 cfg: SemStructConfig, rng: np.random.Generator
                 ) -> List[TrainingGroup]:
    groups = []
    for table_id, headers, columns in tables:
        if len(headers) < cfg.view_min_cols or not columns or not columns[0]:
            continue
        groups.append(TrainingGroup(
            table_id=table_id, headers=list(headers), columns=[list(c) for c in columns],
            views=make_views(headers, columns, cfg, rng)))
    return groups


def _pairwise_cosine_distance(embeddings: torch.Tensor) -> torch.Tensor:
    normalised = torch.nn.functional.normalize(embeddings, p=2.0, dim=1)
    return 1.0 - normalised @ normalised.t()


def batch_loss(model: SemStructEncoder, store: GraphStore,
               groups: Sequence[TrainingGroup], picks: Sequence[int],
               cfg: SemStructConfig) -> Optional[torch.Tensor]:
    """Loss for one batch of (group, chosen view) pairs, or None if no anchor
    in the batch has both a positive and a negative."""
    members = []
    for slot, (group, pick) in enumerate(zip(groups, picks)):
        view = group.views[pick]
        members.append((slot, group.headers, group.columns, list(range(len(group.headers)))))
        members.append((slot, view.headers, view.columns, view.source_columns))
    return _members_loss(model, store, members, cfg)


def group_views_loss(model: SemStructEncoder, store: GraphStore, group: TrainingGroup,
                     view_indices: Sequence[int], cfg: SemStructConfig) -> Optional[torch.Tensor]:
    """Amendment 2 (one-group batching): several views of one source table; positives are the same source
    column in another view, negatives every other source column."""
    members = [(0, group.views[i].headers, group.views[i].columns, group.views[i].source_columns)
               for i in view_indices]
    return _members_loss(model, store, members, cfg)


def _members_loss(model: SemStructEncoder, store: GraphStore, members, cfg: SemStructConfig
                  ) -> Optional[torch.Tensor]:
    graphs, group_ids, source_columns = [], [], []
    for slot, headers, columns, sources in members:
        graphs.append(store.get(headers, columns))
        group_ids.append(np.full(len(headers), slot, dtype=np.int64))
        source_columns.append(np.array(sources, dtype=np.int64))

    features, edge_index, column_index = batch_graphs(graphs)
    x = torch.from_numpy(np.ascontiguousarray(features)).float()
    edges = torch.from_numpy(np.ascontiguousarray(edge_index))
    columns = torch.from_numpy(np.concatenate(column_index))

    embeddings = model(x, edges, columns)
    distance = _pairwise_cosine_distance(embeddings)

    group_of = torch.from_numpy(np.concatenate(group_ids))
    source_of = torch.from_numpy(np.concatenate(source_columns))
    graph_of = torch.from_numpy(np.concatenate(
        [np.full(len(idx), i, dtype=np.int64) for i, idx in enumerate(column_index)]))

    same_group = group_of[:, None] == group_of[None, :]
    same_source = source_of[:, None] == source_of[None, :]
    same_graph = graph_of[:, None] == graph_of[None, :]
    positive = same_group & same_source & ~same_graph
    negative = ~(same_group & same_source)

    has_positive = positive.any(dim=1)
    has_negative = negative.any(dim=1)
    usable = has_positive & has_negative
    if not bool(usable.any()):
        return None

    if cfg.positive_mining == "hardest":
        hardest_positive = torch.where(positive, distance,
                                       torch.full_like(distance, -1.0)).max(dim=1).values
    elif cfg.positive_mining == "easy":
        # Amendment 2: the closest positive
        hardest_positive = torch.where(positive, distance,
                                       torch.full_like(distance, 2.0)).min(dim=1).values
    else:
        raise ValueError(f"positive_mining {cfg.positive_mining!r}")
    if cfg.mining == "hardest":
        hardest_negative = torch.where(negative, distance,
                                       torch.full_like(distance, 2.0)).min(dim=1).values
    elif cfg.mining == "semihard":
        # harness/SEMSTRUCT_ABLATION.md, Amendment 1: the closest negative that is still farther than the
        # positive; where an anchor has none, its farthest negative (no anchor is dropped)
        farther = negative & (distance > hardest_positive[:, None])
        semi = torch.where(farther, distance, torch.full_like(distance, 2.0)).min(dim=1).values
        farthest = torch.where(negative, distance, torch.full_like(distance, -1.0)).max(dim=1).values
        hardest_negative = torch.where(farther.any(dim=1), semi, farthest)
    else:
        raise ValueError(f"mining {cfg.mining!r}")
    triplet = torch.relu(hardest_positive - hardest_negative + cfg.margin)
    return triplet[usable].mean()


def _batches(n_groups: int, per_batch: int, rng: np.random.Generator
             ) -> List[np.ndarray]:
    order = rng.permutation(n_groups)
    return [order[i:i + per_batch] for i in range(0, n_groups, per_batch)]


def train(groups: Sequence[TrainingGroup], store: GraphStore,
          cfg: SemStructConfig, validation: Sequence[TrainingGroup] = (),
          patience: int = 5, log: Optional[Callable[[str], None]] = None
          ) -> Tuple[SemStructEncoder, Dict[str, list]]:
    """Train, with early stopping on validation loss (PAPER A.1.2, patience 5)."""
    model = build_model(cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    per_batch = max(1, cfg.batch_tables // 2)      # each group contributes 2 graphs

    history: Dict[str, list] = {"train": [], "validation": []}
    best_state = copy.deepcopy(model.state_dict())
    best_loss, since_best = float("inf"), 0

    def training_batches():
        if cfg.batch_mode == "groups":
            for chunk in _batches(len(groups), per_batch, rng):
                selected = [groups[i] for i in chunk]
                picks = [int(rng.integers(0, len(g.views))) for g in selected]
                yield lambda selected=selected, picks=picks: batch_loss(model, store, selected, picks, cfg)
        elif cfg.batch_mode == "one_group":
            # Amendment 2: every view once per epoch, in batches of 16 views of one source table, batch order shuffled
            plan = []
            for g, group in enumerate(groups):
                order = rng.permutation(len(group.views))
                plan += [(g, order[i:i + cfg.batch_tables]) for i in range(0, len(order), cfg.batch_tables)]
            for k in rng.permutation(len(plan)):
                g, idx = plan[k]
                if len(idx) >= 2:
                    yield lambda g=g, idx=idx: group_views_loss(model, store, groups[g], idx, cfg)
        else:
            raise ValueError(f"batch_mode {cfg.batch_mode!r}")

    def validation_batches():
        if cfg.batch_mode == "one_group":
            for group in validation:                 # fixed views: a stable yardstick
                idx = list(range(min(cfg.batch_tables, len(group.views))))
                if len(idx) >= 2:
                    yield lambda group=group, idx=idx: group_views_loss(model, store, group, idx, cfg)
            return
        for chunk in _batches(len(validation), per_batch, np.random.default_rng(cfg.seed)):
            selected = [validation[i] for i in chunk]
            picks = [0] * len(selected)              # fixed view: a stable yardstick
            yield lambda selected=selected, picks=picks: batch_loss(model, store, selected, picks, cfg)

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for make_loss in training_batches():
            loss = make_loss()
            if loss is None:
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history["train"].append(train_loss)

        validation_loss = float("nan")
        if validation:
            model.eval()
            with torch.no_grad():
                values = []
                for make_loss in validation_batches():
                    loss = make_loss()
                    if loss is not None:
                        values.append(float(loss.item()))
                validation_loss = float(np.mean(values)) if values else float("nan")
        history["validation"].append(validation_loss)

        if log:
            log(f"epoch {epoch + 1}/{cfg.epochs}  train {train_loss:.4f}  "
                f"validation {validation_loss:.4f}")

        watched = validation_loss if validation else train_loss
        if watched == watched and watched < best_loss - 1e-6:
            best_loss, since_best = watched, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                if log:
                    log(f"early stop at epoch {epoch + 1} (patience {patience})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, history
