"""X79 (b) (EXPLORATORY): the controlled construction-knob corpus.
Corpus data/knobs/<dataset>/<scenario>__<knob>/ (data/make_knobs.py; 166 bases x 12 variants = 1,992 scenarios; profiles
carried over). On the hosts: $SMX_REMOTE_ROOT/smx/corpus/knobs.

Stage 1 (`seeds`): panel of 15 initial matchers plus datatype_nanfix (cloned from datatype as in E2 / Amendment 12;
datatype itself is generated only as the template and not run), --no-sf -> seedstudy_knobs (Java), seedpy_knobs (seed server).
Stage 2 (`arms`): first-line matrices -> reference/seeds/knobs/<m>/; SF arms under tuning/configs/x79_knobs (dispatch.py
needs an underscore in the suffix, so the runs land in tuning/results/x79_knobs/<m>__<arm>):
    <m>__melnik, <m>__fd                  clones of layer1_plain/<m>__<arm>
    <m>__ucc_top1, <m>__ucc1_fd_topk      clones of gvfull_plain/<m>__<arm>
    levenshtein__melnik_twin              the determinism twin
Only the corpus path, datasets, seed directory and resultsDir change (asserted, as make_final_configs.py). A matcher is
kept for a dataset only if every scenario of it has a first-line matrix (gaps listed).

    python servers/make_knobs_configs.py seeds|arms
"""

from __future__ import annotations

import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDY / "servers"))
import make_final_configs as mfc  # noqa: E402

CORPUS, SUFFIX = "knobs", "x79_knobs"
PANEL = ["levenshtein", "jaro_winkler", "prefix_suffix", "qgram3_jaccard", "coma", "cupid", "sbert_mpnet", "pmf", "ks",
         "jsd_values", "containment", "unary_statistics_invariant", "distinct_count", "kang_naughton", "role_gcov"]
ROOT = STUDY / "data" / CORPUS
mfc.CORPORA[CORPUS] = ROOT
mfc.make_configs.CORPUS[CORPUS] = ROOT
mfc.make_configs.EXTERNAL.add(CORPUS)
mfc.score_seeds.CORPUS[CORPUS] = ROOT
mfc.score_seeds.EXTERNAL.add(CORPUS)


def seeds() -> int:
    mfc.MATCHERS = PANEL + ["datatype"]          # datatype: template for datatype_nanfix only
    java_only = mfc.is_py
    import export_seed_study as ess
    real_export = ess.export

    def export(corpus, include, python=False, only=None):
        include = set(include) - {"datatype"}
        return real_export(corpus, include=include, python=python, only=set(only or include) - {"datatype"})
    ess.export = export
    rc = mfc.seeds(CORPUS)
    assert java_only is mfc.is_py
    return rc


def arms() -> int:
    """make_final_configs.arms with this corpus, the panel, and SUFFIX as the configuration/results folder."""
    mfc.MATCHERS = PANEL
    real_cfg = mfc.CFG
    # arms() writes CFG/<corpus>/<label> with resultsDir tuning/results/<corpus>/<label>; route both to SUFFIX
    orig_arms = mfc.arms
    import re
    src = Path(mfc.__file__).read_text(encoding="utf-8")
    body = src[src.index("def arms(corpus: str) -> int:"):src.index('if __name__ == "__main__":')]
    body = body.replace('out_root = CFG / corpus', 'out_root = CFG / SUFFIX')
    body = body.replace('(STUDY / "tuning" / "results" / corpus / label)', '(STUDY / "tuning" / "results" / SUFFIX / label)')
    body = body.replace('g2.count(f"{corpus}/{label}\\"")', 'g2.count(f"{SUFFIX}/{label}\\"")')
    assert body.count("SUFFIX") == 3, body.count("SUFFIX")
    ns = dict(vars(mfc))
    ns["SUFFIX"] = SUFFIX
    exec(compile(body, mfc.__file__ + " (arms, X79)", "exec"), ns)
    assert real_cfg == ns["CFG"] and orig_arms is mfc.arms and re
    return ns["arms"](CORPUS)


if __name__ == "__main__":
    raise SystemExit({"seeds": seeds, "arms": arms}[sys.argv[1]]())
