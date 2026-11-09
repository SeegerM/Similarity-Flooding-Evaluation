"""Command line entry point: turn matcher outputs into one tidy CSV.

    python -m smx.cli single --scores S.csv --gt GT.csv --tag system=sf
    python -m smx.cli schematch --output-dir ... --data-dir ... -o report.csv

Every row of the report is (identifying metadata) x (extraction policy),
carrying both the selection-free scoring diagnostics and the match metrics.
Because the scoring diagnostics do not depend on the extractor, grouping the
report by extractor answers "which selection policy is responsible?" while
holding the score matrix fixed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

from .adapters import (
    discover_pded_runs,
    discover_schematch_runs,
    discover_sima_runs,
    load_pair,
)
from .core import EXTRACTORS, evaluate

# Columns that lead the report; anything else is appended alphabetically.
LEAD_COLUMNS = [
    "system",
    "dataset",
    "scenario",
    "config",
    "extractor",
    "n_source",
    "n_target",
    "n_ground_truth",
    "n_match",
    "match_ratio",
    "precision",
    "recall",
    "f1",
    "overall",
    "tie_mass_row_max",
    "mean_ties_at_row_max",
    "frac_rows_tied_at_max",
    "mean_row_entropy",
    "mrr_pessimistic",
    "mrr_optimistic",
    "hits_at_1_pessimistic",
    "hits_at_1_optimistic",
    "mean_rank_pessimistic",
    "mean_margin",
]


def write_report(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        print("no runs matched; nothing written", file=sys.stderr)
        return
    seen = set()
    for row in rows:
        seen.update(row.keys())
    ordered = [c for c in LEAD_COLUMNS if c in seen]
    ordered += sorted(c for c in seen if c not in set(ordered))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {out_path}")


def _parse_tags(tags: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for tag in tags or []:
        if "=" not in tag:
            raise SystemExit(f"--tag expects key=value, got {tag!r}")
        key, value = tag.split("=", 1)
        meta[key] = value
    return meta


def cmd_single(args: argparse.Namespace) -> int:
    sm = load_pair(Path(args.scores), Path(args.gt), meta=_parse_tags(args.tag))
    rows = evaluate(sm, policies=args.extractors)
    write_report(rows, Path(args.out))
    _print_summary(rows)
    return 0


def cmd_schematch(args: argparse.Namespace) -> int:
    specs = discover_schematch_runs(
        Path(args.output_dir), Path(args.data_dir), dataset=args.dataset,
        system=args.system,
    )
    if not specs:
        print(
            f"no schematch runs found under {args.output_dir} "
            f"with ground truth under {args.data_dir}",
            file=sys.stderr,
        )
        return 1
    rows: List[Dict[str, object]] = []
    failures = 0
    for spec in specs:
        try:
            sm = load_pair(
                Path(spec["scores_path"]),
                Path(spec["ground_truth_path"]),
                meta=dict(spec["meta"]),
            )
        except ValueError as exc:
            failures += 1
            print(f"skipping {spec['scores_path']}: {exc}", file=sys.stderr)
            continue
        rows.extend(evaluate(sm, policies=args.extractors))
    if failures:
        print(f"{failures} run(s) skipped", file=sys.stderr)
    write_report(rows, Path(args.out))
    _print_summary(rows)
    return 0


def cmd_pded(args: argparse.Namespace) -> int:
    specs = discover_pded_runs(
        Path(args.output_dir), Path(args.data_dir), dataset=args.dataset
    )
    if not specs:
        print(
            f"no PDED runs found under {args.output_dir} "
            f"with ground truth under {args.data_dir}",
            file=sys.stderr,
        )
        return 1
    rows: List[Dict[str, object]] = []
    for spec in specs:
        sm = load_pair(
            Path(spec["scores_path"]),
            Path(spec["ground_truth_path"]),
            meta=dict(spec["meta"]),
        )
        rows.extend(evaluate(sm, policies=args.extractors))
    write_report(rows, Path(args.out))
    _print_summary(rows)
    return 0


def cmd_sima(args: argparse.Namespace) -> int:
    specs = discover_sima_runs(
        Path(args.matrices_dir), Path(args.data_dir), dataset=args.dataset,
        config=args.config,
    )
    if not specs:
        print(
            f"no SiMa matrices found under {args.matrices_dir} "
            f"with ground truth under {args.data_dir}",
            file=sys.stderr,
        )
        return 1
    rows: List[Dict[str, object]] = []
    for spec in specs:
        sm = load_pair(
            Path(spec["scores_path"]),
            Path(spec["ground_truth_path"]),
            meta=dict(spec["meta"]),
        )
        rows.extend(evaluate(sm, policies=args.extractors))
    write_report(rows, Path(args.out))
    _print_summary(rows)
    return 0


def _print_summary(rows: List[Dict[str, object]]) -> None:
    """Print the |M| comparison across extractors -- the headline diagnostic."""
    if not rows:
        return
    by_extractor: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_extractor.setdefault(str(row.get("extractor", "?")), []).append(row)

    print()
    header = f"{'extractor':<20} {'|M|/|GT|':>9} {'P':>7} {'R':>7} {'F1':>7} {'ties':>7}"
    print(header)
    print("-" * len(header))
    for name in sorted(by_extractor):
        group = by_extractor[name]

        def mean(key: str) -> float:
            vals = [
                float(r[key])
                for r in group
                if r.get(key) is not None and float(r[key]) == float(r[key])
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        print(
            f"{name:<20} {mean('match_ratio'):>9.2f} {mean('precision'):>7.3f} "
            f"{mean('recall'):>7.3f} {mean('f1'):>7.3f} "
            f"{mean('mean_ties_at_row_max'):>7.2f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smx", description="score-matrix harness for schema matchers"
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=list(EXTRACTORS),
        choices=list(EXTRACTORS),
        help="extraction policies to apply (default: all)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", help="evaluate one score matrix")
    p_single.add_argument("--scores", required=True)
    p_single.add_argument("--gt", required=True)
    p_single.add_argument("--tag", action="append", default=[], help="key=value")
    p_single.add_argument("-o", "--out", default="report.csv")
    p_single.set_defaults(func=cmd_single)

    p_sm = sub.add_parser("schematch", help="walk a schematch output tree")
    p_sm.add_argument("--output-dir", required=True)
    p_sm.add_argument("--data-dir", required=True)
    p_sm.add_argument("--dataset", default=None)
    p_sm.add_argument(
        "--system", default="schematch",
        help="system label stored in the report (for example: prisma)",
    )
    p_sm.add_argument("-o", "--out", default="report.csv")
    p_sm.set_defaults(func=cmd_schematch)

    p_pded = sub.add_parser("pded", help="walk PDED soft-matrix output")
    p_pded.add_argument("--output-dir", required=True)
    p_pded.add_argument("--data-dir", required=True)
    p_pded.add_argument("--dataset", default=None)
    p_pded.add_argument("-o", "--out", default="report.csv")
    p_pded.set_defaults(func=cmd_pded)

    p_sima = sub.add_parser("sima", help="walk SiMa's exported per-scenario matrices")
    p_sima.add_argument("--matrices-dir", required=True,
                        help="the matrices/ directory written by sima_run.py --export-matrices")
    p_sima.add_argument("--data-dir", required=True)
    p_sima.add_argument("--dataset", default=None)
    p_sima.add_argument("--config", default="sima",
                        help="config label stored in the report")
    p_sima.add_argument("-o", "--out", default="report.csv")
    p_sima.set_defaults(func=cmd_sima)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
