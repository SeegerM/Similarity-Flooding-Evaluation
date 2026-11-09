"""Write output/corpus_forest.tex: the per-corpus figure that replaces the RQ1-across-corpora figure and the synthesis table
(redesign 1 of FLOAT_REVIEW.md, 2026-10-26). Rows are corpora grouped by how they were built; three panels
share the rows: (A) Spearman rho between the initial matcher's quality and SF's gain; (B) the information-matched one-step
term minus SF as published in J_tf; (B') SF stopped after two updates minus SF run to convergence, in J_tf.
The figure references macro NAMES only (every macro is checked to exist in output/*_numbers.tex);
macros whose stored sign is SF minus the term use the flipped values \\vA, \\vB, \\vL that main.tex defines.

    python make_corpus_forest.py
"""

from __future__ import annotations

import re

from paths import OUT
figs = OUT
macros = {}
for f in figs.glob("*_numbers.tex"):
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", f.read_text(encoding="utf-8")):
        macros[name] = val
DERIVED = {"vA": ("drPlRefVsAdEst",), "vB": ("drMdRefVsAdEst",), "vL": ("tcHOneEst",)}   # defined in main.tex


def est(base: str | None):
    """(est, lo, hi) macro names for a base; flipped bases are given as 'v*'; lo/hi None if no interval."""
    if base is None:
        return None
    if base in DERIVED:
        assert DERIVED[base][0] in macros, base
        return (base, base + "lo", base + "hi")
    e = base if base in macros else base + "Est"
    if e not in macros:
        raise SystemExit(f"missing macro \\{base}(Est)")
    root = base[:-3] if base.endswith("Est") else base
    lo, hi = root + "Lo", root + "Hi"
    if root + "ClLo" in macros and root + "ClHi" in macros:   # clustered interval (scenarios sharing a table), §III-E
        lo, hi = root + "ClLo", root + "ClHi"
    ok = macros.get(lo, "") not in ("",) and macros.get(hi, "") not in ("",)
    return (e, lo, hi) if ok else (e, None, None)


# (label, construction group, rho base, one-step base, two-update base, registered flags (rho, one-step))
ROWS = [
    ("Built from one source: single tables", None),
    ("Valentine readable", ("rqcVPlRho", "vA", "ldTwoConvJValEst", (False, False))),
    ("Valentine recoded", ("rqcVMdRho", "vB", "ldTwoConvJVMdEst", (False, False))),
    ("New databases (readable)", ("wpCtlPlRho", "finPlFOneEst", "rlxTwoConvCtlPl", (False, True))),
    ("New databases (recoded)", ("wpCtlOpRho", "finOpFOneEst", "rlxTwoConvCtlOp", (False, True))),
    ("Lahman/MusicBrainz", ("etwoPOnePlEst", "etwoPThreeEst", "svLmSEst", (True, True))),
    ("MaDI products", ("wpMadiPRho", "xtMadiPAdEst", "ldTwoConvJMadiPEst", (False, False))),
    ("Built from one source: multi-table", None),
    ("3-DB multi-table", (None, "vL", "rlxTwoConvInd", (False, True))),
    ("Lahman/MusicBrainz multi-table", (None, "mtMacroAdtSfEst", "svLmMEst", (False, False))),
    ("Produced independently: single tables", None),
    ("T2D web tables", ("ttdPOneEst", "ttdPThreeEst", "svTtdEst", (False, False))),
    ("Synthea$\\to$OMOP pairs", (None, "mtSynPairAdSfEst", "svSynEst", (False, False))),
    ("GDC-SM", ("wpGdcRho", "xtGdcAdEst", "ldTwoConvJGdcEst", (False, False))),
    ("HDXSM", ("wpHdxRho", "xtHdxAdEst", "ldTwoConvJHdxEst", (False, False))),
    ("GOBY", ("wpGobyRho", "xtGobyAdEst", "ldTwoConvJGobyEst", (False, False))),
    ("MaDI use cases", ("wpMadiRho", "xtMadiAdEst", "ldTwoConvJMadiEst", (False, False))),
    ("Designed independently: multi-table", None),
    ("Independent schema pairs", ("wpIndepRho", "finAFFiveEst", "ldTwoConvJIndepEst", (False, True))),
]
PANELS = [  # (title, xmin, xmax, ticks, zero-side note)
    ("(A) $\\rho$: quality vs SF's gain", -1.0, 0.6, "-1,-0.5,0,0.5", "weaker gain more"),
    ("(B) one-step term $-$ SF ($J_\\text{tf}$)", -0.06, 0.10, "-0.05,0,0.05", "term better"),
    ("(B$'$) two updates $-$ converged ($J_\\text{tf}$)", -0.02, 0.05, "-0.02,0,0.02,0.04", "two better"),
]


# fewer than five independent units (Section III-E): drawn hollow, without an interval, in every panel
DESCRIPTIVE = {"MaDI products", "MaDI use cases"}
# the held-out corpora on which the two-update stop was tested after being specified in advance (svNonInf of svN)
HELDOUT = {"T2D web tables", r"Synthea$\to$OMOP pairs", "Lahman/MusicBrainz", "Lahman/MusicBrainz multi-table"}


def marks(p: int) -> list[str]:
    out = []
    for y, (label, spec) in enumerate(ROWS, start=1):
        if spec is None:
            continue
        rho, one, two, reg = spec
        base = (rho, one, two)[p]
        names = est(base)
        if names is None:
            continue
        e, lo, hi = names
        registered = reg[p] if p < 2 else False
        if label in DESCRIPTIVE:   # fewer than five independent units: descriptive only (Section III-E), no interval
            lo = hi = None
        if p == 2 and label in HELDOUT:   # held-out corpora of the pre-specified two-update stop
            if lo:
                out.append(rf"\draw[black!75, line width=0.7pt] (axis cs:\{lo},{y}) -- (axis cs:\{hi},{y});")
            out.append(rf"\addplot[only marks, mark=square*, mark size=1.9pt, black] coordinates {{(\{e},{y})}};")
            continue
        if lo:
            out.append(rf"\draw[black!75, line width=0.7pt] (axis cs:\{lo},{y}) -- (axis cs:\{hi},{y});")
            mark = "diamond*, mark size=2.8pt" if registered else "*, mark size=1.9pt"
        else:
            mark = "o, mark size=1.9pt"
        out.append(rf"\addplot[only marks, mark={mark}, black] coordinates {{(\{e},{y})}};")
    return out


n = len(ROWS)
shade = [i for i, (_, s) in enumerate(ROWS, start=1)]
first_ind = next(i for i, (l, s) in enumerate(ROWS, start=1) if s is None and l.startswith("Produced"))
assert not any("," in l for l, _ in ROWS), "labels must not contain commas (pgfplots splits its label list at commas)"
labels = ", ".join(rf"\textbf{{{l}}}" if s is None else l for l, s in ROWS)
tex = ["% Generated by make_corpus_forest.py from macro names only (numbers in figures/*_numbers.tex); do not edit.",
       "\\begin{figure*}[t]", "\\centering", "\\begin{tikzpicture}"]
for p, (title, xmin, xmax, ticks, note) in enumerate(PANELS):
    at = "" if p == 0 else f", at={{(panel{p - 1}.east)}}, anchor=west, xshift=0.45cm"
    ylab = "ytick=\\empty, "      # row labels are drawn as nodes (pgfplots' label lists choke on these labels)
    tex.append(f"\\begin{{axis}}[name=panel{p}{at}, width=0.235\\textwidth, scale only axis, height=7.0cm, "
               f"xmin={xmin}, xmax={xmax}, ymin=0.3, ymax={n + 0.7}, y dir=reverse, {ylab}"
               f"ytick style={{draw=none}}, xtick={{{ticks}}}, scaled x ticks=false, "
               f"xticklabel style={{font=\\scriptsize, /pgf/number format/fixed}}, title={{{title}}}, "
               f"title style={{font=\\footnotesize, yshift=-4pt}}, axis x line*=bottom, axis y line*=left, "
               f"y axis line style={{draw=none}}, clip=false]")
    tex.append(rf"\fill[black!5] (axis cs:{xmin},{first_ind - 0.5}) rectangle (axis cs:{xmax},{n + 0.5});")
    tex.append(rf"\draw[black!70, line width=0.5pt] (axis cs:0,0.3) -- (axis cs:0,{n + 0.7});")
    if p == 0:
        for y, (l, s) in enumerate(ROWS, start=1):
            text = rf"\textbf{{{l}}}" if s is None else l
            tex.append(rf"\node[font=\scriptsize, anchor=east, inner sep=1pt] at (axis cs:{xmin},{y}) {{{text}\ }};")
    tex += marks(p)
    tex.append("\\end{axis}")
tex += ["\\end{tikzpicture}",
        # caption shortened 2026-11-03 (notes_response_2026-11-03.md, section 5): headline dropped, Section V states it
        "\\caption{SF's effect by corpus construction (shaded: produced or designed independently). (A) Spearman $\\rho$ "
        "between initial quality and SF's gain; bounds alone produce such a correlation (\\S\\ref{sec:rq1}); intervals "
        "resample initial matchers. (B) Information-matched one-step term minus SF as published, in $J_\\text{tf}$: "
        "datatype term on single tables (it lacks SF's literal similarities, \\S\\ref{sec:info-dt}), table-context term "
        "on multi-table schemas. (B$'$) SF after two updates minus converged SF; squares: held-out corpora of the stop "
        "specified in advance (passes above $\\rvNIMargin$). Markers as in \\S\\ref{sec:methods}; no mark in (A): not "
        "computed; MIMIC not shown.}",
        "\\label{fig:forest}", "\\end{figure*}"]
(OUT / "corpus_forest.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print(f"wrote figures/corpus_forest.tex: {sum(1 for _, s in ROWS if s)} corpora, {len(PANELS)} panels")
