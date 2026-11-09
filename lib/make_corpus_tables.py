"""Generate the paper's inventory tables from the code, so the draft cannot drift from what runs.

Reads the seed registry (experiments/sf_study/make_configs.py: SEEDS, OPTIONAL_SEEDS, FILE_SEEDS),
the tuning space (experiments/sf_study/tuning/space.py), the audited protocol and the tuning splits,
and writes:

    tables/seed_families.tex   one row per matcher family: members, inputs, implementation
    tables/seed_list.tex       every seed configuration (supplementary material), split over several floats
    tables/data.tex            datasets x scenario counts x tuning splits
    tables/tuning_space.tex    the pre-registered SF tuning space

Every registered seed must have an entry in META below; the script stops otherwise.

    python make_corpus_tables.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from paths import BASELINES, OUT, STUDY  # noqa: E402


META = {
    # seed: (family, display name, inputs, impl, cite keys)
    "levenshtein": ("str", "Levenshtein", "L", "J", "levenshtein1966"),
    "jaro_winkler": ("str", "Jaro-Winkler", "L", "J", "winkler1990"),
    "cosine": ("str", "Cosine (character)", "L", "J", ""),
    "hamming": ("str", "Hamming", "L", "J", ""),
    "lcs": ("str", "Longest common subsequence", "L", "J", "do2002coma"),
    "prefix_suffix": ("str", "Prefix/suffix StringMatch (SF's initial matcher)", "L", "J+", "melnik2002similarity"),
    "prefix_suffix_tfidf": ("str", "Prefix/suffix + TF-IDF", "L", "J+", "melnik2002similarity,koutras2021valentine"),
    "monge_elkan_jw": ("str", "Monge-Elkan (Jaro-Winkler inner)", "L", "J+", "monge1996"),
    "monge_elkan_jw_expand": ("str", "Monge-Elkan + abbreviation expansion", "L", "J+", "monge1996"),
    "monge_elkan_porter": ("str", "Monge-Elkan + Porter stemming", "L", "J+", "monge1996,porter1980"),
    "soft_tfidf": ("str", "SoftTF-IDF", "L", "J+", "cohen2003comparison"),
    "qgram3_jaccard": ("str", "Character 3-grams (Jaccard)", "L", "J+", "ukkonen1992"),
    "qgram2_dice": ("str", "Character 2-grams (Dice)", "L", "J+", "ukkonen1992"),
    "smoa": ("str", "SMOA", "L", "J+", "stoilos2005smoa"),
    "double_metaphone": ("str", "Double Metaphone", "L", "J+", "philips2000"),
    "soundex": ("str", "Soundex", "L", "J+", ""),
    "smith_waterman": ("str", "Smith-Waterman", "L", "J+", "smith1981"),
    "smith_waterman_gotoh": ("str", "Smith-Waterman-Gotoh (affine gaps)", "L", "J+", "gotoh1982"),
    "needleman_wunsch": ("str", "Needleman-Wunsch", "L", "J+", "needleman1970"),
    "norms": ("str", "NORMS-style schema-aware abbreviation expansion", "L", "J+", "sorrentino2010norms"),
    "wordnet_wup": ("lex", "WordNet Wu-Palmer", "L,X", "J+", "miller1995wordnet,wu1994"),
    "wordnet_path": ("lex", "WordNet path", "L,X", "J+", "miller1995wordnet"),
    "wordnet_lch": ("lex", "WordNet Leacock-Chodorow", "L,X", "J+", "leacock1998"),
    "wordnet_res": ("lex", "WordNet Resnik (intrinsic IC)", "L,X", "J+", "resnik1995,seco2004"),
    "wordnet_lin": ("lex", "WordNet Lin (intrinsic IC)", "L,X", "J+", "lin1998,seco2004"),
    "wordnet_jcn": ("lex", "WordNet Jiang-Conrath (intrinsic IC)", "L,X", "J+", "jiang1997,seco2004"),
    "wordnet_lin_brown": ("lex", "WordNet Lin (corpus IC)", "L,X", "S", "lin1998"),
    "thesaurus_generic": ("lex", "Synonym thesaurus (generic example)", "L,X", "J+", "madhavan2001cupid,do2002coma"),
    "thesaurus_umls": ("lex", "UMLS / OMOP vocabulary", "L,X", "S", "bodenreider2004umls"),
    "datatype": ("meta", "Data-type compatibility", "I", "J", ""),
    "distinct_count": ("meta", "Distinct count", "I", "J", ""),
    "extrema": ("meta", "Extrema", "I", "J", ""),
    "null_percentage": ("meta", "Null percentage", "I", "J", ""),
    "uniqueness": ("meta", "Uniqueness", "I", "J", ""),
    "unary_statistics": ("meta", "Unary statistics profile", "I", "J+", "abedjan2018profiling"),
    "unary_statistics_invariant": ("meta", "Unary statistics (recoding-invariant)", "I", "J+", "abedjan2018profiling"),
    "semint": ("meta", "SemInt attribute signatures", "I", "J+", "li2000semint"),
    "containment": ("ovl", "Value containment (Tversky)", "I", "J+", "zhu2016lshensemble"),
    "jaccard_exact": ("ovl", "Value Jaccard", "I", "J+", ""),
    "fuzzy_jaccard": ("ovl", "Fuzzy Jaccard (JaccardLevenshtein)", "I", "J+", "koutras2021valentine"),
    "valentine_jaccard": ("ovl", "Valentine JaccardLevenshtein", "I", "P", "koutras2021valentine"),
    "minhash_jaccard": ("ovl", "MinHash Jaccard", "I", "J+", "broder1997"),
    "minhash_containment": ("ovl", "MinHash containment", "I", "J+", "broder1997,zhu2016lshensemble"),
    "unionability": ("ovl", "Set-unionability (table domain)", "I", "J+", "nargesian2018tus"),
    "unionability_pair": ("ovl", "Set-unionability (pair domain)", "I", "J+", "nargesian2018tus"),
    "char3_soft": ("ovl", "Soft overlap, character 3-grams", "I", "P", ""),
    "token_soft": ("ovl", "Soft overlap, tokens", "I", "P", ""),
    "pmf": ("dist", "Sorted value-frequency PMF", "I", "J", "hellenberg2025prisma"),
    "ks": ("dist", "Kolmogorov-Smirnov", "I", "J+", ""),
    "wasserstein": ("dist", "1-D Wasserstein (EMD)", "I", "J+", "zhang2011distribution"),
    "jsd_values": ("dist", "Jensen-Shannon on values", "I", "J+", ""),
    "valentine_distribution": ("dist", "Valentine Distribution-based", "I", "P", "zhang2011distribution,koutras2021valentine"),
    "distribution_shape": ("dist", "Scale-invariant quantile fingerprint", "I", "J+", "du2024isresmat"),
    "distribution_shape_z": ("dist", "Quantile fingerprint (z-scored)", "I", "J+", "du2024isresmat"),
    "pattern_shape": ("fmt", "Value shape profile", "I", "J+", "bogatu2020d3l"),
    "pattern_charclass": ("fmt", "Character-class composition", "I", "J+", "bogatu2020d3l"),
    "char_distribution": ("fmt", "Sherlock character features", "I", "J+", "hulsebos2019sherlock"),
    "kang_naughton": ("dep", "Kang-Naughton MI graph", "D", "P", "kang2003opaque"),
    "kang_naughton_profile": ("dep", "Kang-Naughton MI profile", "D", "P", "kang2003opaque"),
    "role_gcov": ("dep", "Within-table inclusion role (gcov)", "D", "P", ""),
    "role_gcov_sig": ("dep", "gcov role with significance", "D", "P", ""),
    "role_uncertainty": ("dep", "Uncertainty-coefficient role U", "D", "P", ""),
    "prisma": ("dep", "PRISMA (FD-based)", "D,I", "F", "hellenberg2025prisma"),
    "coma": ("comp", "COMA (schema)", "L", "P", "do2002coma"),
    "coma_instances": ("comp", "COMA (schema + instances)", "L,I", "P", "do2002coma"),
    "cupid": ("comp", "Cupid", "L,X", "P", "madhavan2001cupid"),
    "composite_average": ("comp", "COMA-style average aggregation", "L,I,X", "J+", "do2002coma"),
    "composite_max": ("comp", "COMA-style max aggregation", "L,I,X", "J+", "do2002coma"),
    "composite_confidence": ("comp", "Harmony-style confidence voting", "L,I,X", "J+", "mork2008harmony"),
    "composite_dominants": ("comp", "Predictor-weighted aggregation", "L,I,X", "J+", "sagi2013prediction"),
    "isresmat": ("learn", "ISResMat (instance)", "I,X", "P", "du2024isresmat"),
    "isresmat_hybrid": ("learn", "ISResMat (hybrid)", "L,I,X", "P", "du2024isresmat"),
    "learned_lofo": ("learn", "Learned combination (LSD/YAM-style)", "L,I,D", "F", "doan2001lsd,duchateau2009yam,madhavan2005corpus"),
    "semstruct": ("graph", "SemStruct", "L,I,X", "F", "kang2026semstruct"),
    "semstruct_lm_only": ("graph", "SemStruct, LM only (control)", "L,I,X", "F", "kang2026semstruct"),
    "rema": ("graph", "REMA", "I", "F", "koutras2020rema"),
    "rema_shuffled": ("graph", "REMA, shuffled (control)", "I", "F", "koutras2020rema"),
    "sima_tpcdi": ("graph", "SiMa (TPC-DI-Unionable only)", "I,X", "F", "koutras2022sima"),
    "embdi": ("graph", "EmbDI", "I", "S", "cappuzzo2020embdi"),
    "sbert_minilm": ("emb", "Sentence embeddings, MiniLM", "L,X", "P", "reimers2019sbert"),
    "sbert_mpnet": ("emb", "Sentence embeddings, mpnet", "L,X", "P", "reimers2019sbert"),
    "sbert_mpnet_values": ("emb", "mpnet, label + sample values", "L,I,X", "P", "reimers2019sbert,nargesian2018tus"),
    "sbert_e5": ("emb", "E5-small", "L,X", "P", "wang2022e5"),
    "bert_base": ("emb", "BERT-base (mean pooling)", "L,X", "P", "devlin2019bert"),
    "embedding_random": ("emb", "Random embedding (control)", "L", "P", ""),
    "word_vectors": ("emb", "Word vectors (fastText/GloVe)", "L,X", "S", ""),
    "semantic_types": ("types", "Type-annotator adapter (Sherlock, Sato, DoDuo)", "I,X", "S", "hulsebos2019sherlock,zhang2020sato,suhara2022doduo"),
    "magneto": ("llm", "Magneto (zero-shot, bipartite)", "L,I,X", "P", "liu2025magneto"),
    "llm_opus": ("llm", "LLM scoring of all pairs", "L,I,X", "S", "parciak2024llm"),
    "llm_opus_rerank": ("llm", "LLM retrieve-then-rerank", "L,I,X", "S", "liu2025magneto"),
    "probabilistic_levenshtein": ("unc", "Probabilistic mappings over top-K", "L", "J+", "dong2007uncertainty,murty1968"),
    "topk_uniform_levenshtein": ("unc", "Top-K mapping frequency", "L", "J+", "gal2006topk,murty1968"),
    "constant": ("ctrl", "Constant matrix (structure only)", "--", "J+", ""),
    "random": ("ctrl", "Random matrix", "--", "J", ""),
    "oracle": ("ctrl", "Ground-truth oracle", "G", "J+", ""),
    "oracle_noise025": ("ctrl", "Oracle + 25% noise", "G", "J+", ""),
    "oracle_noise050": ("ctrl", "Oracle + 50% noise", "G", "J+", ""),
    "oracle_erase050": ("ctrl", "Oracle, half the true pairs erased", "G", "J+", ""),
    "oracle_decoy100": ("ctrl", "Oracle with same-row decoys", "G", "J+", ""),
}


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def corpus_counts() -> dict[tuple[str, str], int]:
    """Scenarios per (corpus, regime) from the X77/X81 property files (results/sf_study/results)."""
    out = Counter()
    for f in (STUDY / "results" / "benchmark_properties.csv", STUDY / "results" / "x81" / "benchmark_properties_x81.csv"):
        if f.is_file():
            for r in csv.DictReader(open(f, encoding="utf-8")):
                key = r["corpus"]
                if key == "madi":
                    key = "madi_products" if r["scenario"].startswith("products__") else "madi_indep"
                if key == "synthea":
                    key = "synthea" if "__" in r["scenario"] else "synthea_full"
                out[(key, r["regime"])] += 1
    return out


# the other corpora: (label, counts key(s), built from one source?, what it tests, evidence status)
OTHER_CORPORA = [
    ("Indep.\\ multi-table (3 DBs)", ["indep"], True, "table context", "registered"),
    ("Lahman/MusicBrainz", ["ctl"], True, "replication", "registered"),
    ("Lahman/MB multi-table", ["indep2"], True, "replication", "exploratory"),
    ("New databases (4)", ["final_controlled"], True, "frozen terms", "registered"),
    ("T2D web tables", ["t2d"], False, "role term", "registered$^a$"),
    ("Synthea--OMOP", ["synthea", "synthea_full"], False, "pairs; whole schemas", "exploratory"),
    ("Indep.\\ schema pairs", ["final_indep"], False, "table context", "registered"),
    ("GDC-SM", ["gdcsm"], False, "frozen terms", "exploratory"),
    ("HDXSM", ["hdxsm"], False, "frozen terms", "exploratory"),
    ("GOBY", ["goby"], False, "frozen terms", "exploratory"),
    ("MaDI use cases", ["madi_indep"], False, "frozen terms", "exploratory"),
    ("MaDI products", ["madi_products"], True, "frozen terms (control)", "exploratory"),
    ("MIMIC--OMOP", ["mimic"], False, "frozen terms", "exploratory"),
]


def other_corpora_tabular() -> list[str]:
    cnt = corpus_counts()
    rows = []
    for name, keys, split, tests, status in OTHER_CORPORA:
        per = [sum(v for (c, _), v in cnt.items() if c == k) for k in keys]
        regimes = {r for (c, r) in cnt if c in keys}
        n = "+".join(str(x) for x in per) if len(per) > 1 else str(per[0])
        if len(regimes) > 1:   # readable and recoded halves
            half = per[0] // 2
            n = f"{half}+{half}"
        rows.append(f"{name} & {n} & {'one source' if split else 'indep.'} & {tests} & {status} \\\\")
    return [r"\begin{tabular}[t]{@{}lrlll@{}}", r"\toprule",
            r"Corpus & Scen. & Built & Tests & Status \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]


def data_table() -> str:
    cases = json.loads((BASELINES / "reports" / "audited_md5_fused_20260904" / "protocol.json")
                       .read_text(encoding="utf-8"))["cases"]
    splits = json.loads((STUDY / "tuning" / "splits.json").read_text(encoding="utf-8"))
    split_of = {}
    for s in ("lockbox", "dev", "screen"):
        for k in splits[s]:
            split_of.setdefault(k, set()).add(s)
    rows, totals = [], Counter()
    by = defaultdict(list)
    for c in cases:
        by[c["dataset"]].append(c)
    for ds in sorted(by):
        cs = by[ds]
        n = len(cs)
        lb = sum("lockbox" in split_of.get(c["key"], ()) for c in cs)
        dv = sum("dev" in split_of.get(c["key"], ()) for c in cs)
        sc = sum("screen" in split_of.get(c["key"], ()) for c in cs)
        totals.update({"n": n, "lb": lb, "dv": dv, "sc": sc})
        rows.append(f"{tex(cs[0]['family'])} & {tex(cs[0]['scenario_type'])} & {n} & {lb} & {dv} & {sc} \\\\")
    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{The corpora of the study. (a) The audited Valentine corpus: scenarios (source--target table pairs)"
        r" per family and scenario type, and how the tuning protocol (\S\ref{sec:setup}) splits them. \emph{Dev}"
        r" (75\%): used to choose SF's configuration. \emph{Lockbox} (25\%): held out and scored once,"
        r" for the chosen configurations only. \emph{Screen}: a subset of dev, used to prune the"
        r" configurations before the dev stage. Splits are stratified by family and type. Every"
        r" scenario exists in two regimes, plain and audited MD5. (b) The other corpora: scenarios (readable+recoded"
        r" where both exist; pairs+whole schemas for Synthea--OMOP), whether both sides come from one source,"
        r" what they test, and the evidence status. $^a$The role-term comparison; the rest of T2D is exploratory."
        r" Table~\ref{tab:corpora} characterises them.}",
        r"\label{tab:data}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{minipage}[t]{0.45\textwidth}\centering (a) Valentine\par\smallskip",
        r"\begin{tabular}[t]{@{}llrrrr@{}}",
        r"\toprule",
        r"Family & Type & Scenarios & Lockbox & Dev & Screen \\",
        r"\midrule",
        *rows,
        r"\midrule",
        f"\\multicolumn{{2}}{{@{{}}l}}{{Total}} & {totals['n']} & {totals['lb']} & {totals['dv']} & {totals['sc']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.54\textwidth}\centering (b) Other corpora\par\smallskip",
        *other_corpora_tabular(),
        r"\end{minipage}",
        r"\end{table*}",
    ]) + "\n"


CORPUS_NAMES = {"valentine_plain": "Valentine (readable)", "final_controlled_plain": "New databases (readable)",
                "ctl_plain": "Lahman/MusicBrainz (readable)", "madi_products": "MaDI products", "t2d": "T2D web tables",
                "synthea": "Synthea--OMOP pairs", "final_indep": "Independent schema pairs", "gdcsm": "GDC-SM",
                "hdxsm": "HDXSM", "goby": "GOBY", "madi_indep": "MaDI use cases", "mimic": "MIMIC--OMOP"}


def corpus_characterisation_table() -> str:
    """Every corpus of the study, grouped by how it was built (EXPLORATORY X77, X81, X85; X86 clusters).
    Reads results/sf_study/results/x85/corpus_table.csv (explore_x85_characterise.py)."""
    src = STUDY / "results" / "x85" / "corpus_table.csv"
    if not src.is_file():
        return ""
    rows = list(csv.DictReader(open(src, encoding="utf-8")))

    def f(r, k, d=2):
        v = float(r[k])
        return "--" if v != v else f"{v:.{d}f}"
    body, group = [], None
    for r in rows:
        g = r["built_from_one_source"] == "True"
        if g != group:
            if group is not None:
                body.append(r"\midrule")
            body.append(r"\multicolumn{11}{@{}l}{\emph{" + ("Built from one source" if g else "Produced independently")
                        + r"}} \\")
            group = g
        tables = f"{f(r, 'tables_src', 0)}/{f(r, 'tables_tgt', 0)}"
        body.append(f"{CORPUS_NAMES.get(r['corpus'], tex(r['corpus']))} & {r['scenarios']} & {r['units']} & {tables} "
                    f"& {f(r, 'cols_src', 0)}$\\times${f(r, 'cols_tgt', 0)} & {f(r, 'diag')} "
                    f"& {f(r, 'const_F1_H')}/{f(r, 'const_F1_Hr')} & {f(r, 'ovl_true')}/{f(r, 'ovl_non')} "
                    f"& {f(r, 'name_true')}/{f(r, 'name_non')} & {f(r, 'num_share')} & {f(r, 'fd_per_col', 1)} \\\\")
    return "\n".join([
        r"\begin{table*}[t]",
        r"\caption{Every corpus of the study, by how it was built. Medians over scenarios unless noted. Units:"
        r" independent units of the corpus (source tables or databases for corpora built from one source; schema"
        r" domains for the independent schema pairs; groups of scenarios that share a table elsewhere, GDC-SM by"
        r" its source studies, which share one target). Tables per side; columns; share of true pairs on the"
        r" diagonal (single tables only); Hungarian F1 of an all-equal matrix, mean over scenarios, with ties broken by"
        r" index / at random; value-set overlap and name similarity of true pairs / other pairs; share of numeric"
        r" columns; FDs per column.}",
        r"\label{tab:corpora}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        r"\toprule",
        r"Corpus & Scen. & Units & Tables & Columns & Diagonal & Const.\ F1 H/Hr & Value overlap & Name sim."
        r" & Numeric & FDs/col. \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]) + "\n"


def main() -> int:
    out = OUT
    out.mkdir(exist_ok=True)
    (out / "data.tex").write_text(data_table(), encoding="utf-8")
    (out / "corpus_char.tex").write_text(corpus_characterisation_table(), encoding="utf-8")
    print(f"tables written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
