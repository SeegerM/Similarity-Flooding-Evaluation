"""Timing test for SEMSTRUCT_ABLATION.md Amendment 2 (the revised setup).  No accuracy is computed.

Measures, cold (no disk caches), with the v1 checkpoint as a cost stand-in for e5-small-v2 (same architecture/size):
  1. how many distinct Valentine source tables there are, and their sizes at 2000 rows;
  2. the build of three source tables (smallest, median, largest by cells) at 2000 rows;
  3. the build of 10 views per table under the Amendment 2 view sampling and noise (encoder memo warm from the source);
  4. one forward+backward step of `full` and `valavg` on a batch of 16 views of one table;
  5. the cold build of 10 evaluation scenarios (both tables) at row caps 400 and 2000.
Then projects the cost rule of Amendment 2 for V in {25, 50, 100, 200, 400, 1000}.

    .venv-embed/Scripts/python.exe -I semstruct_timing_a2.py [--threads 12]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code" / "semstruct"))

import semstruct_run as sr  # noqa: E402
from semstruct import AUTHORS, Encoder, GraphStore, TrainingGroup, build_model, read_table  # noqa: E402
from semstruct.augment import make_view  # noqa: E402
from semstruct.train import group_views_loss  # noqa: E402

OUT = HERE / "reports" / "semstruct_a2_20261028"
CANDIDATES = (25, 50, 100, 200, 400, 1000)
BUDGET_H = 14.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)

    cfg = replace(AUTHORS, plm="intfloat/e5-small", views_per_table=10)   # v1 stand-in for the cost only
    cases = sr.load_cases()
    tables = sr.training_tables(cases, cfg, None, 0, "plain")
    sizes = sorted(((len(h) * len(c[0]), d, h, c) for d, h, c in tables), key=lambda t: t[0])
    print(f"{len(tables)} distinct source tables; cells at <=2000 rows: "
          f"min {sizes[0][0]}, median {sizes[len(sizes) // 2][0]}, max {sizes[-1][0]}", flush=True)
    picks = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]

    result = {"threads": args.threads, "n_source_tables": len(tables),
              "source_table_cells": [s[0] for s in sizes], "tables": []}
    rng = np.random.default_rng(20261028)
    for cells, digest, headers, columns in picks:
        encoder = Encoder(cfg, cache_dir=None, threads=args.threads)
        store = GraphStore(cfg, encoder, cache_dir=None)
        started = time.perf_counter()
        store.get(headers, columns)
        source_s = time.perf_counter() - started
        view_s = []
        views = [make_view(headers, columns, cfg, rng) for _ in range(16)]
        for view in views[:10]:
            started = time.perf_counter()
            store.get(view.headers, view.columns)
            view_s.append(time.perf_counter() - started)
        group = TrainingGroup(table_id=digest, headers=list(headers), columns=[list(c) for c in columns], views=views)
        for view in views[10:]:
            store.get(view.headers, view.columns)            # finish the 16-view batch's graphs before timing steps
        step = {}
        for arm, arm_cfg in (("full", cfg), ("valavg", replace(cfg, valavg=True, row_value_edges=False))):
            arm_store = GraphStore(arm_cfg, encoder, cache_dir=None)
            model = build_model(arm_cfg)
            model.train()
            optimiser = torch.optim.Adam(model.parameters(), lr=arm_cfg.lr)
            started = time.perf_counter()
            loss = group_views_loss(model, arm_store, group, list(range(16)), arm_cfg)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            step[arm] = time.perf_counter() - started    # includes rebuilding 16 view graphs from the warm memo
        row = {"cells": cells, "n_cols": len(headers), "n_rows": len(columns[0]), "source_s": source_s,
               "view_s_mean": float(np.mean(view_s)), "view_s_max": float(np.max(view_s)), "step_s": step}
        result["tables"].append(row)
        print(f"  table {len(headers)} cols x {len(columns[0])} rows: source {source_s:.1f}s, view mean "
              f"{row['view_s_mean']:.2f}s (max {row['view_s_max']:.2f}), step full {step['full']:.1f}s, "
              f"valavg {step['valavg']:.1f}s", flush=True)

    sample = [cases[i] for i in sorted(np.random.default_rng(20261028).choice(len(cases), 10, replace=False))]
    result["eval"] = {}
    for cap in (400, 2000):
        ecfg = replace(cfg, row_cap=cap)
        encoder = Encoder(ecfg, cache_dir=None, threads=args.threads)
        store = GraphStore(ecfg, encoder, cache_dir=None)
        seconds = []
        for case in sample:
            paths = sr.case_paths(case, "plain")
            started = time.perf_counter()
            for path in paths[:2]:
                h, c = read_table(path, cap)
                if h:
                    store.get(h, c)
            seconds.append(time.perf_counter() - started)
        result["eval"][cap] = {"mean_s_per_pair": float(np.mean(seconds)), "max": float(np.max(seconds))}
        print(f"  evaluation cap {cap}: {np.mean(seconds):.1f}s per scenario (max {np.max(seconds):.1f})", flush=True)

    # projection under the cost rule
    n_tables = len(tables)
    source_h = n_tables * float(np.mean([t["source_s"] for t in result["tables"]])) / 3600
    view_s = float(np.mean([t["view_s_mean"] for t in result["tables"]]))
    step_full = float(np.mean([t["step_s"]["full"] for t in result["tables"]]))
    step_valavg = float(np.mean([t["step_s"]["valavg"] for t in result["tables"]]))
    projection = {}
    for cap in (400, 2000):
        eval_h = len(cases) * result["eval"][cap]["mean_s_per_pair"] / 3600
        for v in CANDIDATES:
            graphs_h = source_h + n_tables * v * view_s / 3600
            batches = n_tables * v / 16
            # graphs come from the disk cache after the first epoch; the step time above includes graph loading
            # from memory, so it is an upper bound per batch.  10 epochs, both arms.
            train_h = 10 * batches * (step_full + step_valavg) / 3600
            projection[f"cap{cap}_V{v}"] = {"graphs_h": graphs_h, "train_h": train_h, "eval_h": eval_h,
                                            "total_h": graphs_h + train_h + eval_h}
    chosen = {cap: max([v for v in CANDIDATES if projection[f"cap{cap}_V{v}"]["total_h"] <= BUDGET_H], default=None)
              for cap in (400, 2000)}
    result["projection"], result["chosen_V"] = projection, chosen
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{'arm':<12}" + "".join(f"{'V=' + str(v):>10}" for v in CANDIDATES))
    for cap in (400, 2000):
        print(f"{'eval ' + str(cap):<12}" + "".join(f"{projection[f'cap{cap}_V{v}']['total_h']:>9.1f}h" for v in CANDIDATES))
    print(f"chosen by the rule (<= {BUDGET_H} h): {chosen}; wrote {OUT / 'timing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
