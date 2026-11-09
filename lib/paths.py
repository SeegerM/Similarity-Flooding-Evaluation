"""Where the analysis code reads from and writes to.

Cached results live in results/ with the folder layout the study code writes, so the output of a rerun
(study_code/) can replace them file for file. Intermediate values go to a temporary folder (or $SFE_OUTPUT);
the experiment scripts in experiments/ turn them into readable reports in reports/.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "results" / "sf_study"            # Similarity Flooding study: initial matchers, SF runs, screens
BASELINES = ROOT / "results" / "baselines"       # learned systems, sampling controls, Valentine protocol
DERIVED = ROOT / "results" / "derived"           # per-row tables of the later analyses (see results/derived/README.md)
STUDY_CODE = ROOT / "study_code" / "sf_study"
BASELINES_CODE = ROOT / "study_code" / "baselines"
OUT = Path(os.environ.get("SFE_OUTPUT", Path(tempfile.gettempdir()) / "sfe_values")).resolve()
OUT.mkdir(parents=True, exist_ok=True)

# the analysis reuses metric and configuration definitions of the study code
for p in (BASELINES_CODE, STUDY_CODE / "tuning", STUDY_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
