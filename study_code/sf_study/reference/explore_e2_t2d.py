"""EXPLORATORY T2D replication (23:35: E2 stays registered on ctl only; T2D runs later as an
exploratory replication). B, 2026-10-04.

Runs the frozen registered scorer reference/score_e2.py (26a03aa9…) unchanged, with one difference: the t2d corpus
path points at harness re-import with profiles, harness/generated/t2d-usable, instead of t2d-sm-wh. The SF
outputs are read from tuning/results/e2_t2d/ (exploratory run, built like E2 stage 2).

The override is set at module level, so the pool's worker processes (which re-import this module on Windows) see it.
The output is relabelled: tuning/results/e2_t2d_scores is moved to tuning/results/t2d_exploratory_scores, the
report's header says EXPLORATORY, and the verdict words are replaced ("CONFIRMED" -> "consistent with the
prediction", "not confirmed" -> "not consistent"), so no number from it can be quoted as a registered verdict.

    ../harness/.venv-bdi/Scripts/python.exe reference/explore_e2_t2d.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

import score_e2  # noqa: E402
import score_layer1  # noqa: E402

T2D = STUDY.parent / "harness" / "generated" / "t2d-usable"
score_e2.CORPORA["t2d"] = T2D
score_layer1.CORPUS["t2d"] = T2D


def additive_rows_guarded(job) -> list[dict]:
    """score_e2.additive_rows with one addition (2026-10-05, after the first run crashed before any number): a scenario
    whose initial matrix q is not finite is skipped, as score_layer1.score_seed already does for the SF rows. On T2D
    some initial matchers return non-finite cells; the frozen function passed them to all_metrics, which refused."""
    import numpy as np
    corpus, seed, cs = job
    seeds = HERE / "seeds" / corpus
    sf_root = STUDY / "tuning" / "results" / f"e2_{corpus}" / f"{seed}__melnik"
    sf_dir = score_layer1.run_dir(sf_root) if sf_root.is_dir() else None
    out = []
    for ds, sc, fam in cs:
        f_q, f_d = seeds / seed / ds / sc / "seed.csv", seeds / "datatype_nanfix" / ds / sc / "seed.csv"
        f_s = score_e2.one(sf_dir / ds / sc / "_outputs" / "MatchingStepLine2") if sf_dir else None
        g = score_e2.ground_truth_file(score_e2.CORPORA[corpus], ds, sc)
        if not (f_q.is_file() and f_d.is_file() and f_s and g):
            continue
        gt = score_e2.read_matrix(g)[0] > 0.5
        q_raw = score_e2.read_matrix(f_q)[0]
        if not np.isfinite(q_raw).all():
            continue
        q, D = np.clip(q_raw, 0, None), np.nan_to_num(score_e2.read_matrix(f_d)[0])
        if q.shape != gt.shape or D.shape != gt.shape:
            continue
        m = score_e2.all_metrics(q + 5.0 * D, gt, score_e2.permutations(gt.shape, f"{ds}/{sc}"))
        out.append({"seed": seed, "arm": "additive_d", "dataset": ds, "scenario": sc, "family": fam,
                    "sf_R_exp": m["R_exp"], "sf_F1_Hr": m["F1_Hr"]})
    return out


score_e2.additive_rows = additive_rows_guarded   # module level: pool workers re-import this module


def main() -> int:
    if not T2D.is_dir():
        raise SystemExit(f"{T2D} missing")
    sys.argv = [sys.argv[0], "--corpus", "t2d"]
    rc = score_e2.main()
    src = STUDY / "tuning" / "results" / "e2_t2d_scores"
    dst = STUDY / "tuning" / "results" / "t2d_exploratory_scores"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    rep = dst / "report.md"
    text = rep.read_text(encoding="utf-8")
    text = text.replace("registered analysis", "EXPLORATORY replication (not registered; corpus t2d-usable)")
    text = text.replace("not confirmed", "not consistent").replace("CONFIRMED", "consistent with the prediction")
    text = text.replace(" (decided on t2d)", "")
    rep.write_text("> EXPLORATORY: the frozen E2 scorer run on A's profiled re-import of T2D (harness/generated/"
                   "t2d-usable). E2's registered verdicts are on ctl only. The additive term skips scenarios whose "
                   "initial matrix is not finite (as the SF rows already do).\n\n" + text, encoding="utf-8")
    print(f"exploratory T2D report: {rep}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
