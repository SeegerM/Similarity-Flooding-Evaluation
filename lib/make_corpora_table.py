"""Write output/corpora_compact.tex, the paper's corpus table (corpora grouped by how they were built).

Every count is read from the long version's generated tables (make_corpus_tables.py):
- output/corpus_char.tex: scenarios, independent units and tables per side;
- output/data.tex, panel (b): scenario counts per regime for the corpora corpus_char does not list
  (independent multi-table, Lahman/MusicBrainz multi-table) or lists for one regime only.
- figures/abs_numbers.tex (make_abs_numbers.py): the absolute-level column (macro names \absQ*/\absSf*), which replaced
  the former tables-per-side column on 2026-10-28.
Nothing is typed by hand; the script fails if a corpus is missing. Run after the long version's tables change
(refresh_shared.py calls it). Run: python make_corpora_table.py
"""
import re

from paths import OUT, STUDY
main = OUT
# The Valentine row's scenario and unit counts are macros of the 549 run (\srcNScen readable, \rqFourNScen recoded,
# \srcNSources independent source tables): the long version's corpus tables (make_corpus_tables.py) describe the
# 382-case copy. Every other row is not Valentine.


def rows(text: str) -> dict[str, list[str]]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if "&" not in line or not line.endswith("\\\\") or line.startswith("\\multicolumn"):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        out[cells[0]] = cells[1:]
    return out


char = rows((main / "corpus_char.tex").read_text(encoding="utf-8"))
data_text = (main / "data.tex").read_text(encoding="utf-8")
other = rows(data_text[data_text.index("(b) Other corpora"):])


def c(name: str, i: int) -> str:  # corpus_char: 0 scenarios, 1 units, 2 tables per side
    return char[name][i]


def o(name: str) -> str:  # data.tex (b): scenarios, "+" between regimes
    return other[name][0]


valentine = c("Valentine (readable)", 0)
# (corpus, scenarios, units, tables per side, used for)
one_source = [
    ("Valentine (5 families)", f"{valentine}+{valentine}", c("Valentine (readable)", 1), "1/1", "all claims; tuning"),
    # display label: "3-DB multi-table" (the source table still calls it "Indep. multi-table"; renamed here so it no
    # longer reads as "produced independently", after the ICDE review, 2026-10-26)
    (r"3-DB multi-table", o(r"Indep.\ multi-table (3 DBs)"), "3", "multi", "table context"),
    ("Lahman/MusicBrainz", o("Lahman/MusicBrainz"), c("Lahman/MusicBrainz (readable)", 1), "1/1", "replication"),
    ("Lahman/MB multi-table", o("Lahman/MB multi-table"), "2", "multi", "more databases"),
    ("New databases (4)", o("New databases (4)"), c("New databases (readable)", 1), "1/1", "final test"),
    ("MaDI products", c("MaDI products", 0), c("MaDI products", 1), "1/1", "control: one table"),
]
one_source[0] = ("Valentine (5 families)", r"\srcNScen+\rqFourNScen", r"\srcNSources", "1/1", "all claims; tuning")
independent = [
    ("T2D web tables", c("T2D web tables", 0), c("T2D web tables", 1), "1/1", "boundary; role term"),
    (r"Synthea$\to$OMOP", o("Synthea--OMOP"), c("Synthea--OMOP pairs", 1), "1/1; multi", "independent design"),
    (r"Indep.\ schema pairs", c("Independent schema pairs", 0), c("Independent schema pairs", 1),
     c("Independent schema pairs", 2), "final test"),
    ("GDC-SM", c("GDC-SM", 0), c("GDC-SM", 1), "1/1", "external check"),
    ("HDXSM", c("HDXSM", 0), c("HDXSM", 1), "1/1", "external check"),
    ("GOBY", c("GOBY", 0), c("GOBY", 1), "1/1", "external check"),
    ("MaDI use cases", c("MaDI use cases", 0), c("MaDI use cases", 1), "1/1", "external check"),
    (r"MIMIC$\to$OMOP", c("MIMIC--OMOP", 0), c("MIMIC--OMOP", 1), c("MIMIC--OMOP", 2), "descriptive only"),
]

for r in one_source + independent:
    if r[0] == "Valentine (5 families)":
        continue   # macro cells, checked against the 549 macros below
    for cell in r[1:4]:
        if not re.fullmatch(r"[0-9+/;a-z ]+", cell):
            raise SystemExit(f"unexpected cell {cell!r} in row {r[0]!r}")

# Construction columns (redesign 3 of FLOAT_REVIEW.md, 2026-10-26) replace "Used for": per corpus, the
# median value overlap of true pairs (\prOvl*), the median name similarity of true / other pairs (\prName*, \prNameNon*)
# and the Hungarian F1 of a constant matrix with ties broken by index / at random (\xcConstH*, \xcConstR*). Macro names
# only; the key is the corpus's suffix in pr_numbers / xc_numbers (None: not computed for that corpus).
KEY = {"Valentine (5 families)": ("Val", "Val"), "3-DB multi-table": (None, None), "Lahman/MusicBrainz": ("CHAR:Lahman/MusicBrainz (readable)", "LmPl"),
       "Lahman/MB multi-table": (None, None), "New databases (4)": ("CtlPl", "CtlPl"), "MaDI products": ("MadiP", "MadiP"),
       "T2D web tables": ("Ttd", "Ttd"), r"Synthea$\to$OMOP": ("Syn", "Syn"), r"Indep.\ schema pairs": ("Indep", "Indep"),
       "GDC-SM": ("Gdc", "Gdc"), "HDXSM": ("Hdx", "Hdx"), "GOBY": ("Goby", "Goby"), "MaDI use cases": ("Madi", "Madi"),
       r"MIMIC$\to$OMOP": ("Mimic", "Mimic")}
figs = OUT
macros = set()
values = {}
for f in figs.glob("*_numbers.tex"):
    macros |= set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read_text(encoding="utf-8")))
    values.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", f.read_text(encoding="utf-8")))
missing = [m for m in ("srcNScen", "rqFourNScen", "srcNSources") if m not in macros]
if missing:
    raise SystemExit(f"missing 549 macros {missing} for the Valentine row")
# both regimes cover the same 549 scenarios (the base of results/sf_study/results/valentine549/cases.csv)
n_cases = sum(1 for _ in open(STUDY / "results" / "valentine549" / "cases.csv", encoding="utf-8")) - 1
if not values["srcNScen"] == values["rqFourNScen"] == str(n_cases):
    raise SystemExit(f"Valentine counts disagree: {values['srcNScen']}, {values['rqFourNScen']}, cases.csv {n_cases}")
# Absolute levels (reviewers asked for them; replace the former "Tab." column, 2026-10-28): \absQ<k>/\absSf<k> from this
# draft's figures/abs_numbers.tex (make_abs_numbers.py): mean J_tf of the corpus's initial matchers alone / of SF as
# published after them. Key per corpus; None: no stored absolute scores (none at present).
ABS = {"Valentine (5 families)": "Val", "3-DB multi-table": "Ind", "Lahman/MusicBrainz": "LmPl",
       "Lahman/MB multi-table": "LmM", "New databases (4)": "CtlPl", "MaDI products": "MadiP", "T2D web tables": "Ttd",
       r"Synthea$\to$OMOP": "Syn", r"Indep.\ schema pairs": "Indep", "GDC-SM": "Gdc", "HDXSM": "Hdx", "GOBY": "Goby",
       "MaDI use cases": "Madi", r"MIMIC$\to$OMOP": "Mimic"}
abs_macros = set(re.findall(r"\\newcommand\{\\(\w+)\}", (OUT / "abs_numbers.tex").read_text(encoding="utf-8")))


def absolute(name: str) -> str:
    k = ABS[name]
    if k is None:
        return "--"
    missing = [m for m in (f"absQ{k}", f"absSf{k}") if m not in abs_macros]
    if missing:
        raise SystemExit(f"missing macros {missing} for {name}; run make_abs_numbers.py")
    return rf"\absQ{k}/\absSf{k}"


def construction(name: str) -> list[str]:
    pr, xc = KEY[name]
    if pr and pr.startswith("CHAR:"):  # no pr_* macros for this corpus: overlap and names from corpus_char (generated)
        row = char[pr[5:]]               # cells: 0 scen, 1 units, 2 tables, 3 columns, 4 diagonal, 5 const, 6 overlap, 7 names
        ovl, names = row[6].split("/")[0], row[7]
        const = rf"\xcConstH{xc}/\xcConstR{xc}"
        assert re.fullmatch(r"[0-9.]+", ovl) and re.fullmatch(r"[0-9.]+/[0-9.]+", names), (ovl, names)
        return [ovl, names, const]
    need = ([f"prOvl{pr}", f"prName{pr}", f"prNameNon{pr}"] if pr else []) + ([f"xcConstH{xc}", f"xcConstR{xc}"] if xc else [])
    missing = [m for m in need if m not in macros]
    if missing:
        raise SystemExit(f"missing macros {missing} for {name}")
    ovl = rf"\prOvl{pr}" if pr else "--"
    names = rf"\prName{pr}/\prNameNon{pr}" if pr else "--"
    const = rf"\xcConstH{xc}/\xcConstR{xc}" if xc else "--"
    return [ovl, names, const]


def block(title: str, items) -> list[str]:
    """One group of rows. Mat.: the matcher panel behind the J_tf q/SF column (macro xrPanel<key>, make_xr_numbers.py from
    abs_numbers_report.md); Synthea's table pairs and whole schemas are separated by "/"."""
    lines = [rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{title}}}}}\\"]
    for r in items:
        if f"xrPanel{ABS[r[0]]}" not in macros:
            raise SystemExit(f"missing macro xrPanel{ABS[r[0]]} for {r[0]}; run make_xr_numbers.py")
        scen = r[1].replace("+", "/") if r[0] == r"Synthea$\to$OMOP" else r[1]
        lines.append(" & ".join([r[0], scen, r[2], rf"\xrPanel{ABS[r[0]]}", absolute(r[0])] + construction(r[0]))
                     + r"\\")
    return lines


tex = [
    "% Generated by make_corpora_table.py from output/{corpus_char,data}.tex (make_corpus_tables.py) and",
    "% figures/abs_numbers.tex (make_abs_numbers.py); do not edit."
    "\n% Valentine row counts from the 549 macros (src, rq4).",
    # the absolute-level macros are read here unless main.tex already reads them (works either way)
    r"\makeatletter\@ifundefined{absQVal}{\input{figures/abs_numbers}}{}\makeatother",
    r"\begin{table}[t]",
    r"\caption{Corpora by how they were built (readable regime). Scen.: scenarios, readable+recoded (Synthea$\to$OMOP: table",
    r"pairs/whole schemas). Units: independent sources (\S\ref{sec:methods}). Mat.: initial matchers behind $\Jtf$ $q$/SF,",
    r"the initial matchers alone / SF as published (panels differ; compare within a row). Ovl.: median value overlap of",
    r"true pairs. Names: median name similarity, true/other pairs. Const.\ F1: constant matrix, ties by index/at random.",
    r"--: not computed.}",
    r"\label{tab:corpora}",
    r"\scriptsize\setlength{\tabcolsep}{1.6pt}",
    r"\begin{tabular}{@{}lrrrcccc@{}}",
    r"\toprule",
    r"Corpus & Scen. & Units & Mat. & $\Jtf$ $q$/SF & Ovl. & Names & Const.\ F1\\",
    r"\midrule",
    *block("Built from one source", one_source),
    r"\midrule",
    *block("Produced independently", independent),
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
(OUT / "corpora_compact.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print("wrote tables/corpora_compact.tex:", len(one_source) + len(independent), "corpora")
