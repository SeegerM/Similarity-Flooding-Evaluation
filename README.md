# What Does Propagation Add in Similarity Flooding?

Graph-based schema matchers such as **Similarity Flooding (SF)** propagate an initial matcher's similarity scores over
a graph of schema information (data types, keys, dependencies, tables). When they beat their initial matcher, the gain
is usually credited to propagation. We separate three possible sources of such a gain: the **information** in the
graph, its **propagation**, and the **evaluation**. To do so, we compare SF with *one-step terms* that receive the same
information without propagating it, across 85 initial matchers and 14 corpora, some built from one source table and
some produced independently.

Every experiment below has **one script** that recomputes its results from the cached scores in `results/` and prints
them as readable tables. No Java, GPU or cluster is needed for that.

## Quick start

You need Python 3.10 or newer.

```shell
pip install -r requirements.txt
python run_all.py                                # all experiments, about a minute
python experiments/E04_pooling_over_sources.py   # or a single one
```

Each script prints its results, saves them to `reports/<id>.md`, and compares the numbers with the values the paper
reports (`checks/<id>.json`). The last line of every report says whether they all match. The first script you run
also runs the shared analysis code once (about 15 seconds) and caches its values in a temporary folder.

## The experiments

| # | Question | Paper | Script | Main inputs |
|---|---|---|---|---|
| E01 | Without key or FD edges, can SF reorder candidates within a datatype block? (numerical check of Proposition 1) | §II, Prop. 1 | [`E01_block_invariant.py`](experiments/E01_block_invariant.py) | exported SF graphs and iterates of 504 runs |
| E02 | How much of SF's gain depends on its initial matcher, and is the falling gain more than score bounds produce? | §IV, Fig. 3–4 | [`E02_gain_vs_initial_matcher.py`](experiments/E02_gain_vs_initial_matcher.py) | 77 initial matchers alone and with SF on 549 Valentine scenarios |
| E03 | With the same information, is a one-step term or SF ahead, per corpus? | §V, Table II | [`E03_matched_controls.py`](experiments/E03_matched_controls.py) | per-cell contrasts of all corpora |
| E04 | Averaged over independent source groups, are one-step terms ahead on source-built data, and is that different from independent data? | §III-E, contribution 3 | [`E04_pooling_over_sources.py`](experiments/E04_pooling_over_sources.py) | as E03 |
| E05 | Which registered predictions held, and which survive a Holm correction over all of them? | §V-F, Table III | [`E05_registered_tests.py`](experiments/E05_registered_tests.py) | the registered scorers' reports |
| E06 | Is the key graph's lead propagation, column order, or key information a one-step term can use too? | §V-C | [`E06_key_graph_controls.py`](experiments/E06_key_graph_controls.py) | key-graph runs with original and permuted columns, key-membership term |
| E07 | How much do the metric, the tie rule and the assignment rule change SF's measured gain? | §VI-B, Table IV | [`E07_evaluation_audit.py`](experiments/E07_evaluation_audit.py) | all initial matchers under several scoring rules |
| E08 | Is SF stopped after two updates as good as converged SF on held-out corpora? | §V-E, Table III | [`E08_two_updates.py`](experiments/E08_two_updates.py) | held-out iteration runs |
| E09 | Does tuning SF's parameters help on data kept apart from tuning? (registered lockbox) | §VI-A, Table IV | [`E09_tuning_lockbox.py`](experiments/E09_tuning_lockbox.py) | the single lockbox evaluation |
| E10 | Do benchmarks split from one source favour some matchers, and does positional ground truth reward guessing? | §III-D, §VI-C, Fig. 6 | [`E10_benchmark_construction.py`](experiments/E10_benchmark_construction.py) | independent row sampling on three databases; constant matrices |
| E11 | Does message passing in learned matchers add over simpler use of the same information? | §V-G, supplement | [`E11_learned_matchers.py`](experiments/E11_learned_matchers.py) | SemStruct reconstruction, OmniMatch |
| E12 | What does propagation cost compared with a one-step term? | §VII, supplement | [`E12_cost.py`](experiments/E12_cost.py) | dedicated timing pass |

Results marked *registered* in the paper were fixed, with their analysis, before their data were scored (E05, E08,
E09 and parts of E03); everything else is exploratory. E05 lists every registered hypothesis.

## What is in this repository

```
README.md            this file
run_all.py           runs every experiment and checks it against the paper
requirements.txt     numpy, scipy, pandas
experiments/         one script per experiment (E01 ... E12); _common.py holds shared helpers
lib/                 the analysis code the experiment scripts call
results/             cached results of every run the paper reports (the scripts' input)
  sf_study/            the Similarity Flooding study: initial matchers, SF runs, registered reports
  baselines/           learned systems, sampling controls, the Valentine scenario protocol
  derived/             per-row tables of the later analyses (see results/derived/README.md)
checks/              the paper's numbers, which the experiment scripts compare with
registrations/       the registration documents of the registered tests
study_code/          the code that produced results/ (needs Schematch, the benchmark data and a cluster)
  sf_study/            configuration generators, runners, scorers, data converters
  baselines/           the Python matching harness (smx), baseline and control experiments
  semstruct/           our reimplementation of SemStruct
  later_analyses/      the scripts that produced results/derived/ from the stored similarity matrices
  environment/         package lists of the Python environments, the benchmark catalog
reports/             written by the experiment scripts (not part of the repository)
```

`results/` mirrors the folders the study code writes to: a scorer that writes
`study_code/sf_study/tuning/results/<batch>/report.md` produced `results/sf_study/tuning/results/<batch>/report.md`.
The `report.md` files are scorer outputs that the analysis parses. Score files over 10 MB are stored gzipped and listed
in `results/GZIPPED.tsv` with the SHA-256 of the uncompressed file; the scripts unpack and check them automatically.

## Rerunning the experiments from scratch

The experiment scripts start from the cached scores. Recomputing those scores takes a CPU cluster: the study ran on
10 hosts with about 520 parallel tasks over several nights, plus one GPU host for ISResMat and SemStruct. A single SF
run on one scenario takes seconds; the largest batches have 10,000 to 17,000 runs.

1. **Framework.** Check out [Schematch](https://github.com/UMR-Big-Data-Analytics/schematch) and build it with Java 21
   and Maven 3.9. The study's runners call its Similarity Flooding matcher with the switches recorded in each
   configuration. The Python matchers run behind Schematch's matcher server.
2. **Data.** The ground truth and dependency profiles of the prepared scenarios are released as
   [Schema Matching Benchmark Datasets v2.0.0](https://github.com/UMR-Big-Data-Analytics/Schema-Matching-Benchmark-Datasets/releases/tag/v2.0.0)
   (tag commit `e1351e9`); `study_code/environment/benchmark-catalog-v2.0.0.json` pins its archives and checksums.
   Sources of the corpora:

   | corpus | source | conversion |
   |---|---|---|
   | Valentine (ChEMBL, OpenData, TPC-DI, Magellan, Wikidata) | [Zenodo 5084605](https://zenodo.org/records/5084605), the official release | 549 of the 551 scenarios; `study_code/baselines/valentine549/`, `python -m smx.profiling`, `make_md5_audited.py` for the recoded variant; scenario list `results/sf_study/results/valentine549/cases.csv` |
   | GOBY | [goby-benchmark.github.io](https://goby-benchmark.github.io/) | `study_code/sf_study/data/convert_external_raw_c.py` |
   | MaDI-Bench | [wbsg-uni-mannheim/MaDI-Bench](https://github.com/wbsg-uni-mannheim/MaDI-Bench), commit `655faa7`; not redistributed | `study_code/sf_study/data/convert_external_raw_c.py` |
   | T2D-SM | [WDC Schema Matching Benchmark](https://webdatacommons.org/structureddata/smb/) | `study_code/sf_study/data/convert_external_raw.py` |
   | GDC-SM | [Zenodo 14963588](https://zenodo.org/records/14963588) and the original supplementary files | `study_code/sf_study/data/convert_external_raw.py` |
   | HDXSM | [MAGICS-LAB/SMUTF](https://github.com/MAGICS-LAB/SMUTF) | `study_code/sf_study/data/convert_external_raw.py` |
   | Synthea → OMOP | [SMAT](https://github.com/JZCS2018/SMAT), [Synthea sample data](https://synthetichealth.github.io/synthea-sample-data/), [Eunomia](https://github.com/OHDSI/Eunomia) | `study_code/sf_study/data/convert_synthea.py` |
   | MIMIC → OMOP | [MIMIC-IV demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/), [its OMOP version 0.9](https://physionet.org/content/mimic-iv-demo-omop/0.9/), ground truth of Parciak et al. | `study_code/sf_study/data/convert_mimic_omop.py` |
   | 3-DB multi-table (TPC-H, AdventureWorks, Chinook) | public database dumps | `study_code/baselines/independent_sampling.py`, then `study_code/sf_study/data/convert_indep.py` |
   | Lahman / MusicBrainz | [Lahman](https://sabr.org/lahman-database/), [MusicBrainz dumps](https://musicbrainz.org/doc/MusicBrainz_Database/Download) | `study_code/baselines/smx/controlled.py`, `study_code/sf_study/data/make_multitable_ext.py` |
   | Mixed databases (AdventureWorks, IMDb, Sakila, SCOPe) | public database dumps | `study_code/sf_study/data/build_final_controlled.py` |
   | Independent schema pairs (bibliography, music) | public schema designs | `study_code/sf_study/data/convert_final_indep.py`, with our hand-made ground truth in `study_code/sf_study/data/final_gt/` |

3. **Python environments.** `study_code/environment/` lists the packages of the five environments: the main one
   (`venv-bdi`), and one each for ISResMat and SemStruct's revised training, embedding matchers, the OmniMatch
   release, and SiMa.
4. **Initial matchers.** `study_code/sf_study/make_configs.py` and `run_seeds.py` compute the initial similarity
   matrices; `score_seeds.py` and `score_seeds_tiefree.py` score them.
5. **Similarity Flooding runs.** Each batch has a configuration generator (`servers/make_*_configs.py`,
   `reference/make_layer1_configs.py`, `tuning/make_tuning_configs.py`). Run a batch locally with
   `tuning/run_tuning.py` or on a cluster with `servers/dispatch.py push|start|status|collect`, then run its scorer
   (for example `reference/score_e2.py`). The first lines of every cached report name the scorer that wrote it.
6. **The 549 Valentine scenarios.** `study_code/sf_study/reference/assemble_549.py` builds every Valentine input of
   the analysis from the runs on the 381 scenarios of an earlier, incomplete copy, the reruns of every FD-dependent
   configuration on the fixed FD profiles, and the runs on the 168 new scenarios.
7. **Later analyses.** `study_code/later_analyses/` rescored the stored similarity matrices into the tables in
   `results/derived/` (column permutation of the key graph, the key-membership term, Recall@GT per cell, assignment
   rules). Each script's docstring says what it reads.
8. **Re-analysis.** Copy the new reports and tables into `results/` and run `python run_all.py`.

The cluster scripts contain placeholders instead of the authors' infrastructure: `<cpu-host-00>` to `<cpu-host-09>`
and `<gpu-host>` for host names, `$SMX_REMOTE_ROOT` and `<CLUSTER_HOME>` for the run directories, `$DATA_ROOT` for a
local collection of the public databases, and `<RESEARCH_ROOT>` for the study's working tree. Set them before running
a cluster script. `study_code/sf_study/smoke/run_smoke.py` is a one-machine check of SF as published and with FD
edges on the seven Magellan scenarios.

## Registered and exploratory results

The registration documents are in `registrations/`. The binding file of the layer study is
`PREREGISTRATION_LAYERS.md`; its amendments are appended to it with the time they were appended. The tuning study is
registered in `PREREGISTRATION.md`. The other files are the drafts as approved:

| document | registers |
|---|---|
| `PREREGISTRATION.md`, `AMENDMENT_E1_DRAFT.md` | the configuration search, the tie-breaker analysis and the single lockbox evaluation |
| `PREREGISTRATION_LAYERS.md` | SF as published, the layer-1 and layer-3 hypotheses and Amendments 1–12 |
| `AMENDMENT_ADDITIVE_DRAFT.md` (Amendment 5) | the additive datatype-and-class term against SF |
| `AMENDMENT_E2_DRAFT.md` (Amendment 7) | the replication on Lahman/MusicBrainz and T2D |
| `AMENDMENT_CLASSFULL_DRAFT.md`, `AMENDMENT_CLASSFULL_ERRATUM1.md` (Amendment 8) | class nodes on the full matcher panel |
| `AMENDMENT_TABLECTX_DRAFT.md` (Amendment 9) | the table-aware additive term |
| `AMENDMENT_FINAL_DRAFT.md` (Amendment 12) | the final test of the recommendations on untouched data |
| `SEMSTRUCT_REIMPLEMENTATION.md` | the SemStruct reconstruction and its checks |

Two copies differ from the study's frozen files: `PREREGISTRATION_LAYERS.md` in three redacted local paths (a host
name and a data folder) and `AMENDMENT_FINAL_DRAFT.md` in one (the data folder), so their SHA-256 hashes differ from
the ones recorded in the study log. All other files are byte-identical to the frozen ones.

## Companion repositories

- **[Schematch](https://github.com/UMR-Big-Data-Analytics/schematch)**: the matching framework with our Similarity
  Flooding implementation and the initial matchers.
- **[Schema Matching Benchmark Datasets](https://github.com/UMR-Big-Data-Analytics/Schema-Matching-Benchmark-Datasets)**:
  ground truth and dependency profiles of the benchmark scenarios, as versioned release archives.

## Licence

The code (`experiments/`, `lib/`, `study_code/`, `run_all.py`) is licensed under the Apache License 2.0 (`LICENSE`).
The results, checks and registration documents (`results/`, `checks/`, `registrations/`) are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The corpora keep their own licences.

Copyright 2026 Marcian Seeger, Julian Trösser, Fabian Panse and Thorsten Papenbrock.
