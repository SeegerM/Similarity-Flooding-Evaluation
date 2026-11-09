"""Write one Schematch configuration directory per SF seed.

Each directory holds exactly one first-line matcher (the seed) and Similarity Flooding as the
second-line matcher with `initialMatrix: RAW_FIRST_LINE`. One seed per directory is not a
convenience: Schematch keeps first-line matchers in a HashMap and SF seeds itself from
`getFirstLineMatchers().get(0)`, so a directory with two first-line matchers has an unspecified
seed.

SF itself is held at the harness's production configuration (QUICK encoding, INV_PROD
coefficients, fixpoint formula C, FD threshold 0.0, no constraint weighting, UCC/IND off; see
harness/java/AuditedPmfSf.java and SF_FORMULA_C_SWITCH.md). Only the seed varies.

Usage:
    python make_configs.py                       # all seeds, plain regime, full corpus
    python make_configs.py --regime md5          # audited-MD5 corpus copy
    python make_configs.py --datasets Magellan-Unionable --tag smoke
    python make_configs.py --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = {
    "plain": ROOT / "harness" / "generated" / "benchmark-full",
    "md5": ROOT / "harness" / "generated" / "benchmark-full-md5",
    # the independent-sampling corpus, converted by data/convert_indep.py (multi-table, external test set)
    "indep": HERE / "data" / "indep_plain",
    "indepsmoke": HERE / "data" / "indep_test" / "corpus",
    # E2 (Amendment 7): the controlled corpus, Lahman and MusicBrainz, plain and opaque, profiled
    "ctl": ROOT / "harness" / "generated" / "controlled-imported-v3",
    # the 168 Valentine scenarios of the official release that the 382-case corpus lacked (2026-10-30), readable and
    # audited MD5; profiled with the fixed HyFD (output/cluster-start-2026-10-30/cohort-549.json)
    "pnew": ROOT / "harness" / "generated" / "valentine-new168",
    "mnew": ROOT / "harness" / "generated" / "valentine-new168-md5",
}
EXTERNAL = {"indep", "indepsmoke", "ctl", "pnew", "mnew"}  # not in the 382-case protocol: every dataset of the corpus
PROTOCOL = ROOT / "harness" / "reports" / "audited_md5_fused_20260904" / "protocol.json"

# name -> (class, package, params). Floats are written with a decimal point because
# Matcher.configure only converts Double to float.
SEEDS: dict[str, tuple[str, str, dict]] = {
    # --- Schematch label matchers that already existed (never used as SF seeds before) --------
    "levenshtein": ("LevenshteinMatcher", "similarity.label", {}),
    "jaro_winkler": ("JaroWinklerMatcher", "similarity.label", {}),
    "cosine": ("CosineMatcher", "similarity.label", {}),
    "hamming": ("HammingMatcher", "similarity.label", {}),
    "lcs": ("LongestCommonSubsequenceMatcher", "similarity.label", {}),
    # --- new label matchers -----------------------------------------------------------------
    "prefix_suffix": ("PrefixSuffixMatcher", "similarity.label", {"useIdf": False}),
    "prefix_suffix_tfidf": ("PrefixSuffixMatcher", "similarity.label", {"useIdf": True}),
    "monge_elkan_jw": ("MongeElkanMatcher", "similarity.label", {"inner": "JARO_WINKLER"}),
    "monge_elkan_jw_expand": ("MongeElkanMatcher", "similarity.label",
                              {"inner": "JARO_WINKLER", "normalization": "EXPAND"}),
    "soft_tfidf": ("SoftTfIdfMatcher", "similarity.label", {"theta": 0.9}),
    "qgram3_jaccard": ("QGramMatcher", "similarity.label", {"q": 3, "coefficient": "JACCARD"}),
    "qgram2_dice": ("QGramMatcher", "similarity.label", {"q": 2, "coefficient": "DICE"}),
    "smoa": ("SmoaMatcher", "similarity.label", {}),
    "double_metaphone": ("PhoneticMatcher", "similarity.label", {"algorithm": "DOUBLE_METAPHONE"}),
    "soundex": ("PhoneticMatcher", "similarity.label", {"algorithm": "SOUNDEX"}),
    "wordnet_wup": ("WordNetMatcher", "similarity.label", {"measure": "WUP"}),
    "wordnet_lin": ("WordNetMatcher", "similarity.label", {"measure": "LIN"}),
    "wordnet_path": ("WordNetMatcher", "similarity.label", {"measure": "PATH"}),
    # --- Schematch metadata matchers that already existed ----------------------------------
    "datatype": ("DatatypeMatcher", "metadata", {}),
    "distinct_count": ("DistinctCountMatcher", "metadata", {}),
    "extrema": ("ExtremaMatcher", "metadata", {}),
    "null_percentage": ("NullPercentageMatcher", "metadata", {}),
    "uniqueness": ("UniquenessMatcher", "metadata", {}),
    # --- new metadata / instance matchers ---------------------------------------------------
    "unary_statistics": ("UnaryStatisticsMatcher", "metadata", {}),
    "unary_statistics_invariant": ("UnaryStatisticsMatcher", "metadata", {"recodingInvariant": True}),
    "pmf": ("DistributionMatcher", "similarity.instance", {}),
    "containment": ("ContainmentMatcher", "similarity.instance", {"alpha": 1.0, "beta": 0.0}),
    "jaccard_exact": ("ContainmentMatcher", "similarity.instance", {"alpha": 1.0, "beta": 1.0}),
    "fuzzy_jaccard": ("FuzzyJaccardMatcher", "similarity.instance", {}),
    "minhash_jaccard": ("MinHashMatcher", "similarity.instance", {"estimate": "JACCARD"}),
    "minhash_containment": ("MinHashMatcher", "similarity.instance", {"estimate": "CONTAINMENT"}),
    "ks": ("DistributionTestMatcher", "similarity.instance", {"test": "KS"}),
    "wasserstein": ("DistributionTestMatcher", "similarity.instance", {"test": "WASSERSTEIN"}),
    "jsd_values": ("DistributionTestMatcher", "similarity.instance", {"test": "JSD", "domain": "VALUE"}),
    "pattern_shape": ("PatternProfileMatcher", "similarity.instance", {"granularity": "SHAPE"}),
    "pattern_charclass": ("PatternProfileMatcher", "similarity.instance", {"granularity": "CHARCLASS"}),
    # --- controls (never reported as matchers) ----------------------------------------------
    "constant": ("ConstantMatcher", "control", {"value": 1.0}),
    "random": ("RandomMatcher", "sota", {}),
    "oracle": ("OracleMatcher", "control", {}),
    "oracle_noise025": ("OracleMatcher", "control", {"noise": 0.25}),
    "oracle_noise050": ("OracleMatcher", "control", {"noise": 0.5}),
    "oracle_erase050": ("OracleMatcher", "control", {"erase": 0.5}),
    "oracle_decoy100": ("OracleMatcher", "control", {"decoy": 1.0}),
    # --- Python matchers called from Schematch (needs seed_server.py running) --------------
    "sbert_minilm": ("PythonServerMatcher", "external",
                     {"method": "sbert", "options": {"model": "all-MiniLM-L6-v2"}}),
    "sbert_mpnet": ("PythonServerMatcher", "external",
                    {"method": "sbert", "options": {"model": "all-mpnet-base-v2"}}),
    "sbert_mpnet_values": ("PythonServerMatcher", "external",
                           {"method": "sbert", "options": {"model": "all-mpnet-base-v2",
                                                           "text": "label_values"}}),
    "embedding_random": ("PythonServerMatcher", "external", {"method": "embedding_random"}),
    "valentine_distribution": ("PythonServerMatcher", "external", {"method": "valentine_distribution"}),
    "valentine_jaccard": ("PythonServerMatcher", "external", {"method": "valentine_jaccard"}),
    "coma": ("PythonServerMatcher", "external", {"method": "coma"}),
    "coma_instances": ("PythonServerMatcher", "external", {"method": "coma_instances"}),
    "cupid": ("PythonServerMatcher", "external", {"method": "cupid"}),
    "magneto": ("PythonServerMatcher", "external", {"method": "magneto"}),
    # ======================= second layer ===================================================
    # --- label: alignment, stemming, schema-aware abbreviations, thesaurus, WordNet IC -------
    "smith_waterman": ("AlignmentMatcher", "similarity.label", {"algorithm": "SMITH_WATERMAN"}),
    "needleman_wunsch": ("AlignmentMatcher", "similarity.label", {"algorithm": "NEEDLEMAN_WUNSCH"}),
    "smith_waterman_gotoh": ("AlignmentMatcher", "similarity.label", {"algorithm": "SMITH_WATERMAN_GOTOH"}),
    "monge_elkan_porter": ("MongeElkanMatcher", "similarity.label",
                           {"inner": "JARO_WINKLER", "normalization": "PORTER"}),
    "norms": ("NormsMatcher", "similarity.label", {}),
    "thesaurus_generic": ("ThesaurusMatcher", "similarity.label",
                          {"thesaurusFile": (HERE / "resources" / "thesaurus_generic.csv").as_posix()}),
    "wordnet_lch": ("WordNetMatcher", "similarity.label", {"measure": "LCH"}),
    "wordnet_res": ("WordNetMatcher", "similarity.label", {"measure": "RES"}),
    "wordnet_jcn": ("WordNetMatcher", "similarity.label", {"measure": "JCN"}),
    # --- instances / metadata ------------------------------------------------------------------
    "unionability": ("UnionabilityMatcher", "similarity.instance", {"domain": "TABLES"}),
    "unionability_pair": ("UnionabilityMatcher", "similarity.instance", {"domain": "PAIR"}),
    "distribution_shape": ("DistributionShapeMatcher", "similarity.instance", {"normalization": "MINMAX"}),
    "distribution_shape_z": ("DistributionShapeMatcher", "similarity.instance", {"normalization": "ZSCORE"}),
    "char_distribution": ("CharDistributionMatcher", "similarity.instance", {}),
    "semint": ("SemIntMatcher", "metadata", {}),
    # --- composite / uncertainty ---------------------------------------------------------------
    "composite_average": ("CompositeMatcher", "ensemble", {"components": "COMPOSITE", "aggregation": "AVERAGE"}),
    "composite_max": ("CompositeMatcher", "ensemble", {"components": "COMPOSITE", "aggregation": "MAX"}),
    "composite_confidence": ("CompositeMatcher", "ensemble",
                             {"components": "COMPOSITE", "aggregation": "CONFIDENCE"}),
    "composite_dominants": ("CompositeMatcher", "ensemble",
                            {"components": "COMPOSITE", "aggregation": "PREDICTOR", "predictor": "DOMINANTS"}),
    "probabilistic_levenshtein": ("ProbabilisticMappingMatcher", "ensemble", {"k": 20}),
    "topk_uniform_levenshtein": ("ProbabilisticMappingMatcher", "ensemble", {"k": 20, "weighting": "UNIFORM"}),
    # --- Python, called from Schematch ---------------------------------------------------------
    "sbert_e5": ("PythonServerMatcher", "external",
                 {"method": "sbert", "options": {"model": "intfloat/e5-small", "prefix": "query: "}}),
    "bert_base": ("PythonServerMatcher", "external", {"method": "sbert", "options": {
        "model": (HERE.parent / "code" / "ISResMat" / "data" / "pretrained_model" / "bert-base-uncased").as_posix()}}),
    "isresmat": ("PythonServerMatcher", "external", {"method": "isresmat", "options": {"col_name_prob": 0}}),
    "isresmat_hybrid": ("PythonServerMatcher", "external",
                        {"method": "isresmat", "options": {"col_name_prob": 0.5}}),
    "kang_naughton": ("PythonServerMatcher", "external", {"method": "kang_naughton"}),
    "kang_naughton_profile": ("PythonServerMatcher", "external", {"method": "kang_naughton_profile"}),
    "char3_soft": ("PythonServerMatcher", "external", {"method": "char3_soft"}),
    "token_soft": ("PythonServerMatcher", "external", {"method": "token_soft"}),
    # --- the harness's within-table dependency roles (the thesis's own family) -----------------
    "role_gcov": ("PythonServerMatcher", "external", {"method": "role_gcov"}),
    "role_gcov_sig": ("PythonServerMatcher", "external", {"method": "role_gcov_sig"}),
    "role_uncertainty": ("PythonServerMatcher", "external", {"method": "role_uncertainty"}),
}

# The standard composite: one matcher per first-layer family (names, lexical, sets, distributions),
# equal weights. Fixed here, not tuned.
COMPOSITE = [
    {"name": "LevenshteinMatcher", "package": "similarity.label"},
    {"name": "PrefixSuffixMatcher", "package": "similarity.label"},
    {"name": "WordNetMatcher", "package": "similarity.label"},
    {"name": "ContainmentMatcher", "package": "similarity.instance"},
    {"name": "DistributionMatcher", "package": "similarity.instance"},
]

# Seeds that need something this environment does not have; only generated when named with --seeds.
OPTIONAL_SEEDS: dict[str, tuple[str, str, dict]] = {
    "word_vectors": ("PythonServerMatcher", "external",
                     {"method": "word_vectors", "options": {"vectors": "<path to .vec/.txt>"}}),
    "semantic_types": ("PythonServerMatcher", "external",
                       {"method": "semantic_types", "options": {"adapter": "<module:function>"}}),
    "llm_opus": ("PythonServerMatcher", "external", {"method": "llm", "options": {"effort": "medium"}}),
    "llm_opus_rerank": ("PythonServerMatcher", "external",
                        {"method": "llm", "options": {"effort": "medium", "candidates": 3}}),
    "embdi": ("EmbdiMatcher", "sota", {}),
    "thesaurus_umls": ("ThesaurusMatcher", "similarity.label",
                       {"thesaurusFile": "<path to MRCONSO.RRF>", "format": "MRCONSO"}),
    "wordnet_lin_brown": ("WordNetMatcher", "similarity.label",
                          {"measure": "LIN", "icFile": "<path to ic-brown.dat>"}),
}

# FILE seeds: computed outside Schematch and read by SF directly ({suffix}: the config suffix,
# {regime}: plain | md5). The first-line matcher of such a config is a placeholder; score_seeds.py
# reads the seed from the directory recorded in the config's seed_file_dir.txt. A FILE seed only
# runs on the datasets it covers completely, because SF stops on a missing seed file.
FILE_SEEDS: dict[str, str] = {
    "learned_lofo": "seeds_learned/{suffix}/learned_lofo",
    # the harness's PRISMA seeds (make_file_seeds.py --source prisma); Magellan has none
    "prisma": "../harness/seeds_prisma/{regime}",
    # cached learned systems, converted by import_file_seeds.py
    "semstruct": "seeds_imported/{regime}/semstruct",
    "semstruct_lm_only": "seeds_imported/{regime}/semstruct_lm_only",
    "rema": "seeds_imported/{regime}/rema",
    "rema_shuffled": "seeds_imported/{regime}/rema_shuffled",
    "sima_tpcdi": "seeds_imported/{regime}/sima_tpcdi",
}


def covered(seed_dir: Path, corpus: Path, names: list[str]) -> tuple[list[str], list[str]]:
    """Datasets for which every scenario in the corpus has a seed file, and the rest."""
    keep, drop = [], []
    for name in names:
        scenarios = [p.name for p in (corpus / name).iterdir() if p.is_dir()]
        ok = scenarios and all((seed_dir / name / s / "seed.csv").is_file() for s in scenarios)
        (keep if ok else drop).append(name)
    return keep, drop

SF = {
    "propCoeffPolicy": "INV_PROD",
    "fixpoint": "C",
    "graphEncoding": "QUICK",
    "indFilterThreshold": "",
    "uccFilterThreshold": "",
    "fdFilterThreshold": "0.0",
    "constraintWeighting": "NONE",
    "dependencyTopK": "",
    "fdAllowListFile": "",
    "initialMatrix": "RAW_FIRST_LINE",
    "initialMatrixDir": "",
}

GENERAL = """applicationName: "schematch"
defaultSeparator: ","
defaultTablePairSeparator: "___"
defaultSourceDatabaseDir: "source"
defaultTargetDatabaseDir: "target"
defaultGroundTruthDir: "ground_truth"
cacheDir: "cache"
resultsDir: "{results}"
performanceDir: "_performances"
performanceSummaryFileName: "performance_summary.csv"
performanceOverviewFilePrefix: "performance_overview_"
outputDir: "_outputs"
defaultDatasetBasePath: "{corpus}"

evaluateAttributes: false
saveOutputPerTablePair: false
saveOutputVerbose: true

saveOutputTablePairGeneration: false
evaluateTablePairGeneration: false

# The seed matrix itself, so seed quality and SF uplift come from the same run.
saveOutputFirstLineMatchers: true
evaluateFirstLineMatchers: false
readCacheFirstLineMatchers: false
writeCacheFirstLineMatchers: false

# This Schematch fork couples dependency loading to the first-line boosting flag (see the harness
# configs); SF reads RAW_FIRST_LINE, so the boosted matrix is computed but neither saved nor used.
runSimMatrixBoostingOnFirstLineMatchers: true
saveOutputSimMatrixBoostingOnFirstLineMatchers: false
evaluateSimMatrixBoostingOnFirstLineMatchers: false
readCacheSimMatrixBoostingOnFirstLineMatchers: false
writeCacheSimMatrixBoostingOnFirstLineMatchers: false

runSecondLineMatchers: {run_sf}
saveOutputSecondLineMatchers: {run_sf}
evaluateSecondLineMatchers: false
readCacheSecondLineMatchers: false
writeCacheSecondLineMatchers: false

runSimMatrixBoostingOnSecondLineMatchers: false
saveOutputSimMatrixBoostingOnSecondLineMatchers: false
evaluateSimMatrixBoostingOnSecondLineMatchers: false
readCacheSimMatrixBoostingOnSecondLineMatchers: false
writeCacheSimMatrixBoostingOnSecondLineMatchers: false
"""


def scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, dict):
        v = json.dumps(v)
        return "'" + v.replace("'", "''") + "'"
    return json.dumps(str(v))


def matcher_doc(cls: str, package: str, params: dict) -> str:
    lines = ["---", f'name: "{cls}"', f'packageName: "{package}"']
    if params:
        lines.append("params:")
        for k, v in params.items():
            lines.append(f"  {k}: [{scalar(v)}]")
    return "\n".join(lines) + "\n"


def datasets_doc(names: list[str]) -> str:
    return "".join(f'---\nname: "{n}"\npath: "{n}"\n' for n in names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=sorted(CORPUS), default="plain")
    ap.add_argument("--datasets", default="", help="comma-separated; default: every dataset")
    ap.add_argument("--seeds", default="", help="comma-separated subset of SEEDS")
    ap.add_argument("--tag", default="", help="suffix for the config and results directories")
    ap.add_argument("--no-sf", action="store_true", help="seed matrices only")
    ap.add_argument("--server-port", type=int, default=0,
                    help="seed_server port for PythonServerMatcher seeds (default: the matcher's 5101)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for group, table in (("seed", SEEDS), ("optional", OPTIONAL_SEEDS)):
            for name, (cls, package, params) in table.items():
                print(f"{group:8s} {name:28s} {package}.{cls} {params or ''}")
        for name, d in FILE_SEEDS.items():
            print(f"{'file':8s} {name:28s} initialMatrix FILE from {d}")
        return 0

    corpus = CORPUS[args.regime]
    if args.datasets:
        names = [d for d in args.datasets.split(",") if d]
    elif args.regime in EXTERNAL:
        names = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    else:
        # The datasets of the 382-case audited protocol. The corpus folder also holds GDC, whose
        # domain catalogues have no FD profile, so SF refuses them ("not_applicable").
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        names = sorted({c["dataset"] for c in protocol["cases"]
                        if (corpus / c["dataset"]).is_dir()})
    seeds = [s for s in args.seeds.split(",") if s] or list(SEEDS)
    known = {**SEEDS, **OPTIONAL_SEEDS}
    unknown = [s for s in seeds if s not in known and s not in FILE_SEEDS]
    if unknown:
        raise SystemExit(f"unknown seeds: {unknown}")

    suffix = f"{args.regime}" + (f"_{args.tag}" if args.tag else "")
    written = 0
    for seed in seeds:
        sf = dict(SF)
        seed_names = names
        if seed in FILE_SEEDS:
            seed_dir = (HERE / FILE_SEEDS[seed].format(suffix=suffix, regime=args.regime)).resolve()
            cls, package, params = "ConstantMatcher", "control", {"value": 0.0}
            sf.update(initialMatrix="FILE", initialMatrixDir=seed_dir.as_posix())
            if seed_dir.is_dir():
                seed_names, dropped = covered(seed_dir, corpus, names)
                if dropped:
                    print(f"{seed}: not covered, left out: {', '.join(dropped)}")
                if not seed_names:
                    print(f"{seed}: covers no dataset completely; no config written")
                    continue
        else:
            cls, package, params = known[seed]
            params = dict(params)
            if cls == "PythonServerMatcher" and args.server_port:
                params["serverPort"] = args.server_port
            if params.get("components") == "COMPOSITE":
                params["components"] = json.dumps(COMPOSITE)
        out = HERE / "configs" / suffix / seed
        out.mkdir(parents=True, exist_ok=True)
        results = (HERE / "results" / suffix / seed).as_posix()
        (out / "general.yaml").write_text(
            GENERAL.format(results=results, corpus=corpus.as_posix(),
                           run_sf="false" if args.no_sf else "true"), encoding="utf-8")
        (out / "datasets.yaml").write_text(datasets_doc(seed_names), encoding="utf-8")
        (out / "first_line_matchers.yaml").write_text(matcher_doc(cls, package, params), encoding="utf-8")
        (out / "second_line_matchers.yaml").write_text(
            matcher_doc("SimilarityFlooding", "similarityFlooding", sf), encoding="utf-8")
        if seed in FILE_SEEDS:
            (out / "seed_file_dir.txt").write_text(sf["initialMatrixDir"] + "\n", encoding="utf-8")
        written += 1
    print(f"wrote {written} configs under {HERE / 'configs' / suffix} "
          f"({len(names)} datasets from {corpus})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
