"""E10: does how a benchmark was built favour some matchers? (independent row sampling, positional ground truth)

Paper: Sections III-D and VI-C, Fig. 6. Inputs: results/baselines/ (the sampling controls on TPC-H, AdventureWorks
and Chinook) and the constant-matrix checks. Runtime: seconds after the shared analysis run.

    python experiments/E10_benchmark_construction.py
"""
import sys

from _common import V, report, rows

v = V()
c = {}
items = []
for db, lab in (("Tpch", "TPC-H"), ("Aw", "AdventureWorks"), ("Chin", "Chinook")):
    for m, ml in (("Pmf", "PMF (value distributions)"), ("CharTri", "character trigrams"), ("Mag", "Magneto"),
                  ("Lev", "Levenshtein (names only)")):
        items.append((f"{lab}, {ml}: shared rows -> independently sampled rows (change in J_tf)",
                      f"eaSmp{db}{m}Change"))
sections = [
    ("Initial matchers when the two sides share rows or are sampled independently", rows(v, items, c)),
    ("Positional ground truth: Hungarian F1 of a constant matrix", rows(v, [
        ("Valentine, ties broken by index order", "xcConstHVal"),
        ("Valentine, ties broken at random", "xcConstRVal"),
        ("MaDI use cases, index order", "xcConstHMadi"),
        ("HDXSM, index order", "xcConstHHdx"),
        ("GOBY, index order", "xcConstHGoby"),
        ("T2D, index order", "xcConstHTtd"),
    ], c)),
]
sys.exit(report("E10", "Benchmark construction",
                "Do benchmarks split from one source favour matchers that exploit shared values, and does positional "
                "ground truth reward uninformative matchers?", sections, c))
