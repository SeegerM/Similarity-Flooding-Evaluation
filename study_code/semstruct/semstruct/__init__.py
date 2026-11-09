"""SemStruct-reimpl: a paper-faithful reimplementation of Kang et al., KDD 2026.

    Inwon Kang, Kavitha Srinivas, Nandana Mihindukulasooriya, Sola Shirai,
    Parikshit Ram, Horst Samulowitz, Oshani Seneviratne.
    "SemStruct: Contextualizing Semantic Embeddings with Structural Information
    for Schema Matching."  KDD '26.  doi:10.1145/3770855.3817963

**This is not the authors' implementation.**  No public release exists (see
`harness/SEMSTRUCT_REIMPLEMENTATION.md` S1), so every number produced here is
*our reconstruction of SemStruct, not SemStruct*, and must be labelled that way
wherever it is reported.  `config.py` marks every constant as taken from the
paper, fixed by us, or fixed by a cost rule.
"""

from .config import AMENDMENT2_DEFAULTS, AUTHORS, DEFAULT, SemStructConfig
from .encoder import Encoder
from .graph import TableGraph, batch_graphs, build_graph
from .match import (bipartite_rerank, column_embeddings, cosine_matrix,
                    native_matrix, row_softmax, score_matrix)
from .model import SemStructEncoder, build_model
from .store import GraphStore
from .tables import permute_within_columns, read_table
from .train import TrainingGroup, build_groups, train

__all__ = [
    "DEFAULT", "AUTHORS", "AMENDMENT2_DEFAULTS", "SemStructConfig", "Encoder", "TableGraph", "build_graph",
    "batch_graphs", "SemStructEncoder", "build_model", "GraphStore",
    "read_table", "permute_within_columns", "TrainingGroup", "build_groups",
    "train", "score_matrix", "column_embeddings", "cosine_matrix",
    "row_softmax", "bipartite_rerank", "native_matrix",
]

__version__ = "0.1.0"
