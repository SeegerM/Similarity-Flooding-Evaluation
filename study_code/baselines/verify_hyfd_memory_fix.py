"""Compare optimized HyFD records with previously completed native profiles."""
import argparse
import json
from pathlib import Path
from smx.datasets import json_write
from smx.profiling import Profiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=("source", "target"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.manifest.parent / "metadata/profiling.json").read_text(encoding="utf-8"))
    reference = json.loads(Path(state[args.side]["fd"]["path"]).read_text(encoding="utf-8"))
    result = Profiler(Path("profile-cache"), heap="512m", timeout=300.).profile(
        args.manifest, args.side, "fd", threshold=reference["spec"]["threshold"],
        max_arity=reference["spec"]["max_arity"])
    comparison = {"reference": state[args.side]["fd"], "optimized": result,
                  "reference_heap": reference["spec"]["heap"], "identical_records": False}
    if result["status"] == "complete":
        actual = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        comparison["identical_records"] = actual["records"] == reference["records"]
        comparison["reference_seconds"] = reference["elapsed_seconds"]
        comparison["optimized_seconds"] = actual["elapsed_seconds"]
    json_write(args.report, comparison)
    print(json.dumps(comparison, indent=2), flush=True)
    return 0 if comparison["identical_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
