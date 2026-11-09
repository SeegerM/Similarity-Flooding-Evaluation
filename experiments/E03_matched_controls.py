"""E03: one-step terms that receive SF's information, against SF, per corpus (the paper's Table II).

Each cell is a one-step term minus SF with the same information in its graph (datatypes, column classes, column roles
from keys and FDs, table context), and SF after two updates minus converged SF. Shown in expected Recall@GT (the
ranking diagnostic) and J_tf (the registered metric: mean of Recall@GT and tie-free Hungarian F1).

Paper: Section V, Table II. Inputs: the shared analysis values (J_tf cells) and results/derived/recall/recall_*.csv
and results/derived/loso/panel_cells.csv (Recall@GT cells, rescored from the stored runs by
study_code/later_analyses/recall_cells.py and its siblings). Runtime: about a minute.

    python experiments/E03_matched_controls.py
"""
import sys

from _common import V, report

V()   # computes the shared values that the table code reads
import make_results_table as TJ          # noqa: E402  (J_tf cells)
import make_results_table_R as TR        # noqa: E402  (Recall@GT cells)
from w1_pooling import registered_role_J  # noqa: E402
from w2_dual import DIFF                  # noqa: E402

NAMES = {'dt': 'Datatype', 'cl': 'Class', 'ro': 'Roles', 'tc': 'Table ctx.', 'tu': 'Two upd. - conv.'}


def cell(x):
    if not isinstance(x, dict):
        return '--'
    s = '%+.3f' % x['e']
    if x['l'] is None:
        return f'({s})'
    side = ' term' if x['l'] > 0 else ' SF' if x['h'] < 0 else ''
    s = f"{s} [{x['l']:+.3f}, {x['h']:+.3f}]{side}"
    return f'_{s}_' if x.get('nm') else s


reg = registered_role_J()
diff = DIFF()
hdr = '| corpus | sources | ' + ' | '.join(NAMES.values()) + ' |'
md_r, md_j = [hdr, '|---|---|' + '---|' * 5], [hdr, '|---|---|' + '---|' * 5]
checked = {}
for i, ((label, g, cr), (_, _, cj)) in enumerate(zip(TR.ROWS_R, TJ.ROWS)):
    name = label.replace('$\\to$', ' to ').replace('\\ ', ' ')
    rr, jj = [], []
    for k in NAMES:
        r = cr[k]
        j = reg.get(label) if cj[k] == TJ.NI_R and k == 'ro' else cj[k]
        rr.append(cell(r) + (' (j)' if (label, k) in diff else ''))
        jj.append(cell(j))
        if isinstance(r, dict):
            checked[f'R {name} {k}'] = '%+.3f' % r['e']
        if isinstance(j, dict):
            checked[f'J {name} {k}'] = '%+.3f' % j['e']
    src = str(TJ.SOURCES[i]) if hasattr(TJ, 'SOURCES') else ''
    md_r.append(f'| {name} | {src} | ' + ' | '.join(rr) + ' |')
    md_j.append(f'| {name} | {src} | ' + ' | '.join(jj) + ' |')
cnt = TR.counts()
checked.update({'cells marked j': str(len(diff)),
                'source-built: term ahead / SF ahead / with interval': '%d / %d / %d' % cnt[0],
                'independent: term ahead / SF ahead / with interval': '%d / %d / %d' % cnt[1]})
legend = ('Cells: estimate [95% interval]; "term" or "SF" marks the side that is ahead when the interval excludes 0; '
          '(value): no interval; _italic_: not information-matched (no table organization); (j): the verdict differs '
          'in J_tf; --: not run or not applicable. Corpora with fewer than five independent sources have intervals '
          'that are conditional on those sources. Roles: against the top-1 key graph (mixed DBs: the best '
          'constraint graph); Lahman/MusicBrainz and T2D roles: the registered test\'s runs.')
sections = [
    ('Expected Recall@GT', '\n'.join(md_r) + '\n\n' + legend),
    ('J_tf', '\n'.join(md_j)),
    ('Counts of cells with an interval (Recall@GT)', [
        ('built from one source: term ahead / SF ahead / cells', checked['source-built: term ahead / SF ahead / with interval']),
        ('produced independently: term ahead / SF ahead / cells', checked['independent: term ahead / SF ahead / with interval']),
        ('cells whose verdict differs between the two metrics', checked['cells marked j'])]),
]
sys.exit(report('E03', 'Information-matched one-step terms against SF (Table II)',
                'When SF and a one-step term receive the same information, which is ahead, per corpus?', sections,
                checked, note='Counting cells is descriptive; E04 pools them over independent source groups.'))
