"""Lossless-enough LIMA adaptation and canonical denial-constraint parsing."""
from __future__ import annotations

import base64
import csv
import math
import re
from pathlib import Path

NULLS = {"", "null", "none", "nan", "na", "n/a", "nil", "missing", "unknown", "?"}
PREDICATE = re.compile(r"^(c\d{4})(?:\(Double\))?(!=|=|<|>)(c\d{4})(?:\(Double\))?$")


def encode_category(value: str, row: int) -> str:
    if value.strip().lower() in NULLS:
        return f"null_row_{row:09d}"
    return "v_" + base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")


def numeric_column(values: list[str]) -> list[str] | None:
    converted = []
    for value in values:
        if value.strip().lower() in NULLS:
            # LIMA has no explicit missing-value semantics. Do not turn NaN into
            # a third ordered comparison state; retain equality-only semantics.
            return None
        try:
            number = float(value.strip())
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        converted.append(format(number, ".17g"))
    return converted


def adapt_table(path: Path, table: dict, destination: Path, row_cap: int) -> tuple[list[dict], int]:
    """Validate the whole export and write a deterministic prefix for LIMA."""
    csv.field_size_limit(2**31 - 1)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected = [c["label"] for c in table["columns"]]
        if header != expected:
            raise ValueError(f"Header/manifest mismatch: {path}")
        rows, row_count = [], 0
        for index, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"Ragged row {index + 2}: {path}")
            row_count += 1
            if len(rows) < row_cap:
                rows.append(row)
    if row_count != table["rows"]:
        raise ValueError(f"Row count/manifest mismatch: {path}")
    if len(rows) < 2:
        raise ValueError("LIMA requires at least two rows")
    values_by_column = list(map(list, zip(*rows)))
    numeric = [numeric_column(values) for values in values_by_column]
    adapted_header = [f"c{i:04d}(Double)" if values is not None else f"c{i:04d}"
                      for i, values in enumerate(numeric)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(adapted_header)
        for row_index, original in enumerate(rows):
            writer.writerow([numeric[column][row_index] if numeric[column] is not None
                             else encode_category(value, row_index)
                             for column, value in enumerate(original)])
    mapping = [{"id": column["id"], "index": column["index"], "label": column["label"],
                "lima_header": adapted_header[index],
                "dtype": "DOUBLE" if numeric[index] is not None else "CATEGORY"}
               for index, column in enumerate(table["columns"])]
    return mapping, len(rows)


def parse_constraint(text: str, columns: list[dict]) -> dict:
    if not text.startswith("!(") or not text.endswith(")"):
        raise ValueError(f"Unrecognised LIMA constraint: {text}")
    predicates = []
    for item in text[2:-1].split(" & "):
        match = PREDICATE.fullmatch(item)
        if not match or match.group(1) != match.group(3):
            raise ValueError(f"Unrecognised LIMA predicate: {item}")
        index = int(match.group(1)[1:])
        if index >= len(columns) or columns[index]["id"] != match.group(1):
            raise ValueError(f"LIMA column outside mapping: {item}")
        predicates.append({"column": match.group(1), "operator": match.group(2),
                           "tuple_variables": ["t0", "t1"]})
    return {"kind": "dc", "text": text, "predicates": predicates,
            "columns": sorted({p["column"] for p in predicates}),
            "arity": len({p["column"] for p in predicates}),
            "predicate_count": len(predicates)}


def read_relatedness(path: Path, columns: list[dict]) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    headers = [c["lima_header"] for c in columns]
    if not rows or rows[0][1:] != headers or [row[0] for row in rows[1:]] != headers:
        raise ValueError("LIMA relatedness labels/orientation changed")
    values = [[float(value) for value in row[1:]] for row in rows[1:]]
    if len(values) != len(columns) or any(len(row) != len(columns) for row in values):
        raise ValueError("Invalid LIMA relatedness shape")
    if any(not math.isfinite(value) or value < 0 for row in values for value in row):
        raise ValueError("Invalid LIMA relatedness value")
    return values


def legacy_dc_lines(records: list[dict], labels: dict[str, str]) -> list[str]:
    """Project canonical LIMA DCs to the DCFinder-compatible parser grammar."""
    operators = {"=": "==", "!=": "<>", "<": "<", ">": ">"}
    lines = []
    for record in records:
        terms = [f"t0.{labels[predicate['column']]} {operators[predicate['operator']]} "
                 f"t1.{labels[predicate['column']]}" for predicate in record["predicates"]]
        lines.append("[DC:  " + "  ".join(terms) + " ]")
    return sorted(lines)
