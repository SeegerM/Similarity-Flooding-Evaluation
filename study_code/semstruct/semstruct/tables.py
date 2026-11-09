"""Reading a CSV table column-major, under an explicit row cap.

A prefix cap rather than a random sample: it is deterministic without a seed
and it matches the row-cap precedent already in this repository (REMA uses a
1,000-row prefix, LIMA profiling the same).  What a prefix costs is a real
question -- see SEMSTRUCT_REIMPLEMENTATION.md S4 -- but it is measurable and it
does not add a second source of randomness to a method that already has one.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def read_table(path: Path, row_cap: Optional[int] = None
               ) -> Tuple[List[str], List[List[str]]]:
    """Return (headers, columns) with `columns` column-major.

    Ragged rows are padded to the header width rather than rejected: unlike the
    score matrices, where a short row silently corrupts every metric, a short
    data row is a benign CSV quirk and dropping the whole table would lose a
    scenario.  Over-long rows are truncated, and both are counted by the caller
    through `table_digest` if it needs to know.
    """
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration:
            return [], []
        columns: List[List[str]] = [[] for _ in headers]
        for count, values in enumerate(reader):
            if row_cap is not None and count >= row_cap:
                break
            if len(values) < len(headers):
                values = values + [""] * (len(headers) - len(values))
            for column, value in zip(columns, values):
                column.append(value)
    return headers, columns


def table_digest(headers: Sequence[str], columns: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for header, column in zip(headers, columns):
        digest.update(header.encode("utf-8", "surrogatepass"))
        digest.update(b"\x01")
        for value in column:
            digest.update(value.encode("utf-8", "surrogatepass"))
            digest.update(b"\x00")
        digest.update(b"\x02")
    return digest.hexdigest()


def permute_within_columns(columns: Sequence[Sequence[str]], seed: int
                           ) -> List[List[str]]:
    """The row-permutation control (SEMSTRUCT_REIMPLEMENTATION.md S6, check 2).

    Each column is permuted independently, so every column keeps its exact
    multiset of values -- its cardinality, its distribution, its value nodes and
    its column-value edges are all unchanged -- while the row co-occurrence
    that SemStruct claims to exploit is destroyed.  Anything the method gains
    over the LM-only control that survives this permutation was not structural.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for column in columns:
        order = rng.permutation(len(column))
        out.append([column[i] for i in order])
    return out
