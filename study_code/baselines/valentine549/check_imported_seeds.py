"""Every imported seed of results/<regime>/ must have one seed matrix and one SF matrix for each of the 168 scenarios."""
import sys
from pathlib import Path

study = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(study))
from score_seeds import cases_for, latest_run, one  # noqa: E402

for regime in ("pnew", "mnew"):
    cases = cases_for(regime)
    short = {}
    for seed_dir in sorted(p for p in (study / "results" / regime).iterdir() if (p / "DONE").is_file()):
        run = latest_run(seed_dir)
        missing = sum(1 for d, s, _ in cases
                      if not (one(run / d / s / "_outputs" / "MatchingStepLine1") and one(run / d / s / "_outputs" / "MatchingStepLine2")))
        if missing:
            short[seed_dir.name] = missing
    n = sum(1 for p in (study / "results" / regime).iterdir() if (p / "DONE").is_file())
    print(f"{regime}: {n} seeds, {len(cases)} scenarios each; incomplete: {short or 'none'}")
