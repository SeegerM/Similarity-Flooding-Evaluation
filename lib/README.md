# Analysis code

The experiment scripts in `experiments/` call this code; you do not need to run anything here directly.

- `values.py` unpacks the gzipped score files, runs the `make_*_numbers.py` scripts once and collects every value
  they compute (about 4,600, by name). The scripts were written for the paper's build: they write their values as
  LaTeX macro definitions (`\newcommand{\name}{value}`) into a temporary value folder, which their docstrings call
  `output/`. `values.py` reads them back as plain name-value pairs; nothing in the repository depends on LaTeX.
- `make_rr_numbers.py` reads the registered role replication's reports.
- `make_results_table.py`, `make_results_table_R.py` (Table II in J_tf and Recall@GT), `w1_pooling.py` (E04),
  `w2_dual.py` (cells whose verdict differs between the metrics), `w6_registry.py` and `make_registered_table.py`
  (E05), `keygraph.py` (E06), `f1_rules.py` (E07) and `recall_cells.py` (the contrast designs) implement the later
  analyses. They read `results/derived/` and the values above.
- `paths.py` names the folders.
