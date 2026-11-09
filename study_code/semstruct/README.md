# SemStruct-reimpl

A paper-faithful reimplementation of

> Inwon Kang, Kavitha Srinivas, Nandana Mihindukulasooriya, Sola Shirai,
> Parikshit Ram, Horst Samulowitz, Oshani Seneviratne.
> **"SemStruct: Contextualizing Semantic Embeddings with Structural Information
> for Schema Matching."** KDD '26. [doi:10.1145/3770855.3817963](https://doi.org/10.1145/3770855.3817963)

**This is not the authors' implementation, and no public one exists.** Every
number produced here is *our reconstruction of SemStruct, not SemStruct*, and
must be labelled `SemStruct-reimpl` wherever it is reported. The preregistration
— what the paper fixes, what we had to fix, and the checks that decide whether
the reconstruction is using the information the paper claims to use — is
[`preregistration/SEMSTRUCT_REIMPLEMENTATION.md`](../../preregistration/SEMSTRUCT_REIMPLEMENTATION.md).
Read that first.

## What the method does

A table becomes a heterogeneous graph:

```
column ---- value ---- row ---- value ---- column
        (occurs in)  (occurs in)
```

Column and value nodes get frozen `e5-small` embeddings (a value is serialised
as `column: {header}, value: {value}`); row nodes start at zero and act as
conduits. A 3-layer GraphSAGE propagates over that graph, and the refined
column embedding is the semantic one plus a learned delta:

```
h_final(c) = h_0(c) + MLP(h_L(c))
```

The language model is never trained. The structural encoder is trained
self-supervised, on synthetic views of source tables only, with a cosine
triplet loss at margin 0.2 and hardest-positive / hardest-negative mining
inside the batch.

Matching score for a pair of schemas is the cosine between final column
embeddings, which is where this project's harness takes over: the score matrix
goes through the same extractors as every other matcher here, and the paper's
own softmax + bipartite reranking is a separate, secondary arm.

## Layout

```
semstruct/config.py     the frozen configuration; every field marked PAPER, PREREG or COST
semstruct/text.py       null rule, numeric detection, quantile binning, serialisation
semstruct/encoder.py    frozen E5, mean-pooled and L2-normalised, with a content-addressed cache
semstruct/graph.py      graph construction, value-node merging, batching
semstruct/model.py      SAGEConv (PyG defaults, plain torch) + 2-layer MLP behind the residual
semstruct/augment.py    synthetic views: row/column sampling, typo/format/acronym noise
semstruct/train.py      batch construction, triplet loss, early stopping
semstruct/match.py      cosine score matrix; softmax + Magneto-style bipartite reranking
semstruct/store.py      content-addressed graph cache
tests/                  28 tests, including the three mechanical validation controls
```

There is no PyTorch Geometric dependency: `SAGEConv` is four matrix multiplies
and is reimplemented at PyG's defaults, checked against the definition by
`test_sage_matches_reference`. Requirements are `torch`, `transformers`,
`numpy` and `scipy` — all present in the `venv-embed` environment (`../environment/venv-embed.freeze.txt`).

## Running

```shell
python experiments/semstruct/tests/run_tests.py
```

Against this project's corpus, through `../baselines/semstruct_run.py` (from `experiments/baselines/`):

```shell
python semstruct_run.py cost
python semstruct_run.py train
python semstruct_run.py score --regime plain
python semstruct_run.py verify
python semstruct_run.py report
```

Standalone, on two CSV files:

```python
from dataclasses import replace
from semstruct import DEFAULT, Encoder, GraphStore, build_model, read_table, score_matrix

cfg = replace(DEFAULT, row_cap=400)
store = GraphStore(cfg, Encoder(cfg, cache_dir="cache"))
source = store.get(*read_table("source.csv", cfg.row_cap))
target = store.get(*read_table("target.csv", cfg.row_cap))
S = score_matrix(build_model(cfg), source, target)      # untrained; see `train`
```

## The one thing to know before reading any result

The published merge threshold does not transfer to the published checkpoint.
`tau = 0.9` on raw `e5-small` cosine merges 76–96% of all value pairs, because
E5 is anisotropic and two arbitrary short strings already sit near 0.95. Across
the whole class sample it keeps **3.8%** of value tokens as nodes — under the
recoded regime, one table's 1,097 value tokens become 4 nodes. The same
threshold on column-centred cosine keeps 96.1% and the step becomes a no-op.
Neither matches the paper's description of what the merge is for.

The primary arm keeps the literal reading; `--merge centered` and `--merge none`
are declared alternatives, all three fixed before any scenario was scored. The
measurement and the reasoning are in S4.2 of the preregistration.
