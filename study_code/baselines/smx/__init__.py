"""smx - a matcher-agnostic harness for dependency-aware schema matching.

Every matcher is reduced to a single artifact: a score matrix S over
(source columns x target columns).  All diagnostics and all correspondence
extraction happen downstream of S, identically for every matcher, so that
scoring quality and selection policy can be measured separately.
"""

from .core import (
    ScoreMatrix,
    scoring_diagnostics,
    EXTRACTORS,
    extract,
    match_metrics,
    evaluate,
)

__all__ = [
    "ScoreMatrix",
    "scoring_diagnostics",
    "EXTRACTORS",
    "extract",
    "match_metrics",
    "evaluate",
]
