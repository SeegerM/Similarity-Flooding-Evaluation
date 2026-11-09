"""Synthetic views of a source table, for the self-supervised objective.

PAPER S3.4: "given a table T, we generate a set of augmented views by randomly
sampling rows and columns.  We further augment each view by applying
column-header and value level noise (typos, formatting changes, acronyms)."
Since a view is derived from the table, the column correspondence is known, and
that is the entire supervision signal.

The paper gives the *kinds* of noise and none of the rates, so every rate here
is ours; they are listed in `config.py` with the PREREG marker and justified in
SEMSTRUCT_REIMPLEMENTATION.md S3.  The operators themselves are written to
imitate the perturbations Valentine's own fabricator applies -- that is what
the noise is meant to stand in for -- without reading Valentine's code, so the
model is not being trained on the exact transformation it will be tested on.

Everything is driven by an explicit `numpy.random.Generator`, so a run is a
function of the seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .config import SemStructConfig

_SPLIT = re.compile(r"[_\s\-.]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_VOWELS = "aeiouAEIOU"
_NUMERIC = re.compile(r"^[+-]?\d+(\.\d+)?$")

# Small neighbourhood on a QWERTY layout; a "typo" that lands on a key next to
# the intended one is the realistic case, and it is what makes the character
# noise different from uniform random corruption.
_NEIGHBOURS = {
    "a": "qsz", "b": "vgn", "c": "xdv", "d": "sfce", "e": "wrd", "f": "dgvr",
    "g": "fhbt", "h": "gjny", "i": "uok", "j": "hkmu", "k": "jli", "l": "ko",
    "m": "nj", "n": "bmh", "o": "ipl", "p": "ol", "q": "wa", "r": "etf",
    "s": "adwx", "t": "ryg", "u": "yij", "v": "cfb", "w": "qes", "x": "zsc",
    "y": "tuh", "z": "asx",
}


def typo(text: str, rng: np.random.Generator) -> str:
    if len(text) < 2:
        return text
    op = rng.integers(0, 4)
    i = int(rng.integers(0, len(text)))
    if op == 0 and i < len(text) - 1:                      # transpose
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if op == 1:                                            # delete
        return text[:i] + text[i + 1:]
    if op == 2:                                            # duplicate
        return text[:i] + text[i] + text[i:]
    character = text[i]
    options = _NEIGHBOURS.get(character.lower())
    if not options:
        return text
    replacement = options[int(rng.integers(0, len(options)))]
    if character.isupper():
        replacement = replacement.upper()
    return text[:i] + replacement + text[i + 1:]


def reformat(text: str, rng: np.random.Generator) -> str:
    if _NUMERIC.match(text.strip()):
        stripped = text.strip()
        op = rng.integers(0, 3)
        if op == 0 and "." not in stripped:
            return stripped + ".0"
        if op == 1 and "." in stripped:
            return stripped.split(".")[0]
        try:                                               # thousands grouping
            return f"{int(float(stripped)):,}"
        except (ValueError, OverflowError):
            return stripped
    op = rng.integers(0, 4)
    if op == 0:
        return text.upper()
    if op == 1:
        return text.lower()
    if op == 2:
        return text.title()
    separators = ["_", " ", "-"]
    parts = [p for p in _SPLIT.split(text) if p]
    if len(parts) < 2:
        return text.strip()
    return separators[int(rng.integers(0, len(separators)))].join(parts)


def acronym(text: str, rng: np.random.Generator) -> str:
    parts = [p for p in _SPLIT.split(_CAMEL.sub(" ", text)) if p]
    if len(parts) > 1:
        return "".join(p[0] for p in parts).upper()
    if len(text) > 3:                                      # drop vowels
        head, tail = text[0], text[1:]
        squeezed = head + "".join(c for c in tail if c not in _VOWELS)
        return squeezed if len(squeezed) > 1 else text[:3]
    return text


def perturb_header(header: str, rng: np.random.Generator) -> str:
    return (typo, reformat, acronym)[int(rng.integers(0, 3))](header, rng)


def perturb_value(value: str, rng: np.random.Generator) -> str:
    return (typo, reformat)[int(rng.integers(0, 2))](value, rng)


@dataclass
class View:
    headers: List[str]
    columns: List[List[str]]
    source_columns: List[int]      # view column j came from source column [j]


def make_view(headers: Sequence[str], columns: Sequence[Sequence[str]],
              cfg: SemStructConfig, rng: np.random.Generator) -> View:
    n_cols = len(headers)
    n_rows = len(columns[0]) if n_cols else 0

    if cfg.view_sampling == "authors":
        # Amendment 2: columns a uniform integer in [ceil(n/2), n], rows a uniform integer in
        # [100, min(2000, n)] (all rows when the table has fewer than 100), both in random order without replacement
        keep_cols = max(min(cfg.view_min_cols, n_cols), int(rng.integers(-(-n_cols // 2), n_cols + 1)))
        chosen = rng.choice(n_cols, size=keep_cols, replace=False).tolist()
        low, high = cfg.view_row_range
        high = min(high, n_rows)
        keep_rows = n_rows if high < low else int(rng.integers(low, high + 1))
        rows = rng.choice(n_rows, size=keep_rows, replace=False).tolist() if n_rows else []
    elif cfg.view_sampling == "frac":
        low, high = cfg.view_col_frac
        keep_cols = max(min(cfg.view_min_cols, n_cols),
                        int(round(n_cols * float(rng.uniform(low, high)))))
        chosen = sorted(rng.choice(n_cols, size=keep_cols, replace=False).tolist())

        low, high = cfg.view_row_frac
        keep_rows = max(1, int(round(n_rows * float(rng.uniform(low, high)))))
        rows = sorted(rng.choice(n_rows, size=keep_rows, replace=False).tolist()) \
            if n_rows else []
    else:
        raise ValueError(f"view_sampling {cfg.view_sampling!r}")

    out_headers: List[str] = []
    out_columns: List[List[str]] = []
    for column_index in chosen:
        header = headers[column_index]
        if rng.random() < cfg.p_header_noise:
            header = perturb_header(header, rng)
        cells = [columns[column_index][r] for r in rows]
        if rng.random() < cfg.p_value_noisy_column:
            cells = [perturb_value(c, rng) if c and rng.random() < cfg.p_value_noise_cell
                     else c for c in cells]
        out_headers.append(header)
        out_columns.append(cells)
    return View(headers=out_headers, columns=out_columns, source_columns=chosen)


def make_views(headers: Sequence[str], columns: Sequence[Sequence[str]],
               cfg: SemStructConfig, rng: np.random.Generator) -> List[View]:
    return [make_view(headers, columns, cfg, rng)
            for _ in range(cfg.views_per_table)]
