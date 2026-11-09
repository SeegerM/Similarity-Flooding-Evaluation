# Later analyses

These scripts produced the tables in `results/derived/` (see the table there). They read the stored similarity
matrices of the study's runs, the corpora with their dependency profiles and the initial matrices, which are not part
of this repository, and they assume the study's working tree (`<RESEARCH_ROOT>`: `sf-matcher-study/`, `harness/` and
the manuscript folder these scripts lived in). They are included so that every cached table can be traced to the
code that computed it; the experiment scripts in `experiments/` do not run them.

| script | what it computes |
|---|---|
| `recall_cells.py`, `recall_multitable.py`, `recall_roles.py`, `recall_er10.py` | Table II cells in expected Recall@GT from the stored runs, with the same matcher panels, weights and bootstrap designs as the J_tf cells |
| `recall_keygraph.py` | key graph minus SF as published per corpus |
| `perm_keygraph.py`, `perm_keygraph549.py` | the key graph with permuted columns (60-scenario pilot, then all 549 Valentine scenarios) |
| `w1_roles_perm.py` | the role term against the key graph with permuted columns |
| `w4_keyterm.py`, `w4_keyterm_madi.py` | the key-membership term (Schematch's key choice reimplemented) on Valentine and the external corpora |
| `f1_policies.py`, `rescore_f1_policies_gain.py` | SF's F1 gain under several assignment rules |
| `make_loso.py` | leave-one-source-out cells, the common-panel recomputation and the role term per scenario |
| `make_cluster_maps.py` | the scenario-to-cluster maps of the external corpora |
