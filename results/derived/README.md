# Per-row tables of the later analyses

These tables were computed from the stored similarity matrices (not included, tens of GB) by the scripts in
`study_code/later_analyses/`. The experiment scripts read them directly. One row is one initial matcher on one
scenario unless stated otherwise. Files over 10 MB are stored gzipped (see `results/GZIPPED.tsv`).

| file | content | produced by | used in |
|---|---|---|---|
| `recall/recall_cells.csv` | Table II cells in expected Recall@GT and J_tf (one row per corpus and comparison) | `recall_cells.py` | E03, E04 |
| `recall/recall_multitable.csv`, `recall/recall_roles.csv`, `recall/recall_er10.csv` | the multi-table, Valentine role and further Table II cells | `recall_multitable.py`, `recall_roles.py`, `recall_er10.py` | E03, E04 |
| `recall/recall_keygraph.csv` | key graph minus SF as published per corpus | `recall_keygraph.py` | E06 |
| `recall/perm_keygraph_plain549_rows.csv`, `recall/perm_keygraph_md549_rows.csv` | key graph with original and permuted columns, and SF as published, on 549 Valentine scenarios | `perm_keygraph549.py` | E06 |
| `recall/w4_keyterm_rows.csv` | key-membership term, key graph (original and permuted), SF and role term on Valentine | `w4_keyterm.py` | E06 |
| `recall/w4_keyterm_ext_rows.csv` | the same on the external corpora, with the key lift as a term and in the graph | `w4_keyterm_madi.py` | E06 |
| `recall/gain_policies_rows.csv` | every initial matcher alone and with SF, scored under several assignment rules | `rescore_f1_policies_gain.py` | E07 |
| `loso/panel_cells.csv` | every Table II cell on its own and on the common matcher panel | `make_loso.py` | E03 |
| `loso/role_recall_plain.csv`, `loso/role_recall_md5.csv` | the role term in Recall@GT per matcher and scenario | `make_loso.py` | E06 |
| `loso/loso_cells.csv` | Table II cells with one source left out | `make_loso.py` | (supplement) |
| `clusters.json` | scenario-to-cluster maps of the external corpora (shared tables, studies) for the cluster bootstrap | `make_cluster_maps.py` | E06 |
