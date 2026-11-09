"""E04: are one-step terms ahead on data built from one source and not on independently produced data? (pooled)

Table II's cells are not independent (readable and recoded variants share sources; some corpora share databases).
This experiment gives each of 11 independent source groups one score, the mean of its one-step term minus SF cells,
and tests the difference between the two kinds of data by exact permutation over all assignments of the labels.

Paper: contribution (3), Section III-E and the supplement's pooling section. Inputs: as E03. Runtime: about a minute.

    python experiments/E04_pooling_over_sources.py
"""
import re
import sys

from _common import V, report

V()
import w1_pooling  # noqa: E402

lines = w1_pooling.main()
text = '\n'.join(lines)
checked = {}
for metric, block in (('R', text.split('## J_tf')[0]), ('J', text.split('## J_tf')[1])):
    m = re.search(r'\| all cells \| raw \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([0-9.]+) \(of (\d+)\)', block)
    checked[f'{metric} source-built mean'], checked[f'{metric} independent mean'] = m.group(1).strip(), m.group(2).strip()
    checked[f'{metric} difference'], checked[f'{metric} exact p'] = m.group(3).strip(), m.group(4)
    g = re.search(r'all cells, source-built: mean of (\d+) unit scores ([^ ]+) \[([^,]+), ([^\]]+)\]', block)
    checked[f'{metric} source-built groups'] = g.group(1)
    checked[f'{metric} source-built interval'] = f'[{g.group(3)}, {g.group(4)}]'
sections = [('Results (all variants)', text.split('\n', 2)[2] if text.startswith('#') else text)]
sys.exit(report('E04', 'Pooling over independent source groups',
                'Averaged over independent source groups, are one-step terms ahead of SF on data built from one '
                'source, and is that different from independently produced data?', sections, checked,
                note='Positive values: the one-step term is ahead. Exploratory.'))
