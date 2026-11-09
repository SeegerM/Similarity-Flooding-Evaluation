# H4, grouped resampling (EXPLORATORY, ER11)

Same panel (73 matchers) and statistic as the registered H4 (`h4_report.md`); groups resampled within families, 10000 draws, seed 20261003.

- Groups, independent corpus (database / sampling seed): {'AdventureWorks': 3, 'Chinook': 3, 'TPCH': 3}
- Groups, Valentine plain (column splits): {'ChEMBL': 13, 'Magellan': 7, 'OpenData': 11, 'TPC-DI': 15}

Difference J_tf(independent) − J_tf(plain): **+0.0449**, grouped 95% interval **[+0.0362, +0.0525]** (registered scenario-level interval in `h4_report.md`).

Conditional on these three databases and their three sampling seeds each; not an interval for new databases.
