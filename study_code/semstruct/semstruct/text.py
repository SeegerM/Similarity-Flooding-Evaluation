"""Cell/column text handling: the null rule, numeric detection, serialisation.

The null rule is deliberately identical to the project's own
(`harness/make_md5_audited.is_null`), so that a cell counted absent by the
recoded-corpus builder is counted absent here too.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

NULLS = {"", "null", "none", "nan", "na", "n/a", "-"}

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def is_null(value: str) -> bool:
    return value.strip().lower() in NULLS


def as_float(value: str) -> Optional[float]:
    text = value.strip().replace(",", "")
    if not _NUMERIC.match(text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def is_numeric_column(values: Sequence[str], n_bins: int) -> bool:
    """Numeric iff every non-null cell parses and there are more than `n_bins`
    distinct values.

    PREREG.  The paper says only "for numerical columns".  Requiring *every*
    non-null cell to parse keeps mixed columns categorical, which is the
    conservative reading: discretising a column that is mostly text would
    destroy exactly the semantic content the value nodes carry.  The
    cardinality condition is what makes discretisation a no-op on a column that
    already has at most k distinct values (a coded flag, say), where binning
    would only rename the categories as ranges.
    """
    seen = set()
    any_value = False
    for raw in values:
        if is_null(raw):
            continue
        any_value = True
        number = as_float(raw)
        if number is None:
            return False
        seen.add(number)
    return any_value and len(seen) > n_bins


def _format_edge(x: float) -> str:
    return f"{x:.6g}"


def quantile_bins(numbers: Sequence[float], n_bins: int) -> List[float]:
    """Quantile edges with duplicates collapsed.

    PREREG.  The paper does not say what happens when a distribution is so
    skewed that several quantiles coincide (a column that is 90% zero).  Equal
    edges are collapsed, so such a column simply gets fewer than k bins rather
    than several empty ones that no value can ever fall into.
    """
    if not numbers:
        return []
    ordered = sorted(numbers)
    edges = []
    for i in range(n_bins + 1):
        q = i / n_bins
        pos = q * (len(ordered) - 1)
        low = int(math.floor(pos))
        high = min(low + 1, len(ordered) - 1)
        frac = pos - low
        edges.append(ordered[low] * (1 - frac) + ordered[high] * frac)
    collapsed = [edges[0]]
    for edge in edges[1:]:
        if edge > collapsed[-1]:
            collapsed.append(edge)
    return collapsed


def bin_label(number: float, edges: Sequence[float]) -> str:
    """Right-closed bins, the last one inclusive on both sides."""
    if len(edges) < 2:
        return _format_edge(edges[0]) if edges else "0"
    for i in range(1, len(edges)):
        if number <= edges[i] or i == len(edges) - 1:
            return f"[{_format_edge(edges[i - 1])}, {_format_edge(edges[i])}]"
    return f"[{_format_edge(edges[-2])}, {_format_edge(edges[-1])}]"


def discretize(values: Sequence[str], n_bins: int) -> Tuple[List[Optional[str]], bool]:
    """Map a column's cells to value tokens; None marks a null.

    Returns (tokens, was_discretized).
    """
    if is_numeric_column(values, n_bins):
        numbers = [as_float(v) for v in values if not is_null(v)]
        edges = quantile_bins([n for n in numbers if n is not None], n_bins)
        out: List[Optional[str]] = []
        for raw in values:
            if is_null(raw):
                out.append(None)
                continue
            number = as_float(raw)
            out.append(None if number is None else bin_label(number, edges))
        return out, True
    return [None if is_null(v) else v.strip() for v in values], False


def serialize_column(header: str, strategy: str = "header") -> str:
    """PAPER Table 1.  Only `header` is built: it is the frozen variant."""
    if strategy != "header":
        raise ValueError(
            f"serialization {strategy!r} is not part of the frozen variant; "
            "see SEMSTRUCT_REIMPLEMENTATION.md S2"
        )
    return header


def serialize_value(header: str, value: str, template: str) -> str:
    return template.format(column=header, value=value)
