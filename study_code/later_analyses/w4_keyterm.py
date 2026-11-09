"""Review 7, W4: the identical-information control for the top-1 key graph (2026-11-08; EXPLORATORY).

By the refined-block lemma, SF with the top-1 key graph (key columns linked to a shared key-size node) can only re-offset
the datatype x key-membership blocks. Its one-step counterpart with exactly that information is the key-membership term
    A_key = q+ + 5 D + gamma [i in key_s and j in key_t]
with the datatype term of the paper (q+ + 5D) and the key that Schematch's key graph chose.

Key choice, reimplemented from Schematch (UniqueColumnCombination.getPrimaryKeyScore, DependencyFilter.keepTopK,
InputReader.readUCCFile): UCCs with at most 3 columns, primary-key score = mean of 1/|U|, mean over columns of
1/max(1, longest value - 8), 0.5 (1/(left+1) + 1/(between+1)) for the columns' positions, and the share of columns whose
lower-cased label ends in id, key, nr or no. Top-1 by score; ties keep Java HashSet iteration order (hash of the column
list, Column.hashCode = (label + "___" + table).hashCode()), which is emulated here. With permuted columns only the
position part changes (the UCC files name columns by label and were copied unchanged).

Validation: the predicted change of the key under the permutation is compared with whether the stored key-graph output
changed (figures/recall/perm_keygraph_<reg>_rows.csv).

gamma is chosen leave-one-family-out on J_tf from (0.05, ..., 10), as the role term's weight was. Contrasts
as Table II (family macro over families with >= 5 scenarios; 76 matchers): key term - key graph, key term - SF as
published, key term - role term; original and permuted column order.
Writes figures/recall/w4_keyterm.md and w4_keyterm_rows.csv.
Run:  SMX_VALENTINE=549 ../../harness/.venv-bdi/Scripts/python.exe w4_keyterm.py
"""
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
for p in (ST.parent / 'harness', ST / 'tuning', ST, ST / 'reference'):
    sys.path.insert(0, str(p))
os.environ.setdefault('SMX_VALENTINE', '549')

G = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)   # widened 2026-11-08: the first run chose the grid edge 1.0 everywhere
REG = {'plain': 'plain549', 'md5': 'md549'}
SUFFIXES = ('id', 'key', 'nr', 'no')


# ---------------------------------------------------------------- Java hashing and HashSet order
def jstr(s):
    h = 0
    b = s.encode('utf-16-be')
    for i in range(0, len(b), 2):
        h = (31 * h + ((b[i] << 8) | b[i + 1])) & 0xFFFFFFFF
    return h


def jlist(hs):
    h = 1
    for x in hs:
        h = (31 * h + x) & 0xFFFFFFFF
    return h


def ucc_hash(cols, table):
    return (31 + jlist([jstr(c + '___' + table) for c in cols])) & 0xFFFFFFFF


def hashset_order(items, hashes):
    """Iteration order of a java.util.HashSet filled with `items` in this order (no treeification)."""
    seen, uniq, hs = set(), [], []
    for it, h in zip(items, hashes):
        if it not in seen:
            seen.add(it)
            uniq.append(it)
            hs.append(h)
    cap = 16
    while len(uniq) > cap * 0.75:
        cap *= 2
    bucket = [((h ^ (h >> 16)) & (cap - 1)) for h in hs]
    order = sorted(range(len(uniq)), key=lambda i: (bucket[i], i))
    return [uniq[i] for i in order], [hs[i] for i in order]


# ---------------------------------------------------------------- key choice
from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=8)
def read_table(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        r = csv.reader(f)
        labels = next(r)
        maxlen = [0] * len(labels)
        any_val = [False] * len(labels)
        for row in r:
            for j, v in enumerate(row[:len(labels)]):
                any_val[j] = True
                n = len(v.encode('utf-16-le')) // 2
                if n > maxlen[j]:
                    maxlen[j] = n
    return labels, maxlen, any_val


def resolve(name, labels):
    name = name.strip()
    k = name.rfind(' (pdep ')
    name = (name[:k] if k >= 0 else name).strip()
    if name in labels:
        return name
    i = name.find('.')
    while i >= 0:
        if name[i + 1:] in labels:
            return name[i + 1:]
        i = name.find('.', i + 1)
    raise KeyError(name)


def parse_cols(line):
    out, s = [], line
    while ']' in s:
        start = s.index('[') if '[' in s else 0
        end = s.index(',') if ',' in s else s.index(']')
        out.append(s[start + 1:end])
        s = s[end + 1:]
    return out


def choose_key(scdir, side, position=None):
    """Columns of the top-1 UCC of one side, or () if none; position: {label: index} for permuted columns."""
    table = side
    labels, maxlen, any_val = read_table(scdir / side / f'{side}.csv')
    pos = position or {l: i for i, l in enumerate(labels)}
    by_pos = sorted(labels, key=lambda l: pos[l])
    f = scdir / 'metadata' / side / table / 'UCC_results.txt'
    if not f.is_file():
        return ()
    uccs = []
    for line in f.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.strip().lower() == '[]':
            continue
        cols = tuple(resolve(c, labels) for c in parse_cols(line))
        uccs.append(cols)
    if not uccs:
        return ()
    a, ha = hashset_order(uccs, [ucc_hash(c, table) for c in uccs])        # readUCCFile's HashSet
    keep = [(c, h) for c, h in zip(a, ha) if 0 < len(c) <= 3]
    if not keep:
        return ()
    b, _ = hashset_order([c for c, _ in keep], [h for _, h in keep])         # filter's HashSet

    def score(c):
        card = 1.0 / len(c)
        val = np.mean([1.0 / max(1, (maxlen[labels.index(x)] if any_val[labels.index(x)] else 2 ** 31 - 1) - 8)
                       for x in c])
        ps = [pos[x] for x in c]
        lo, hi = min(ps), max(ps)
        left = lo
        between = sum(1 for l in by_pos[lo:hi + 1] if l not in c)
        p = 0.5 * (1.0 / (left + 1) + 1.0 / (between + 1))
        suf = sum(1 for x in c for s in SUFFIXES if x.lower().endswith(s)) / len(c)
        return (card + val + p + suf) / 4.0
    if len(b) == 1:
        return b[0]
    ranked = sorted(b, key=score, reverse=True)                             # stable, as Java's List.sort
    return ranked[0]


def keys_for(regime, ds, sc, perm):
    import valentine549 as V
    d = V.corpus_dir(regime, ds, sc)
    out = {}
    for side in ('source', 'target'):
        labels = read_table(d / side / f'{side}.csv')[0]
        orig = choose_key(d, side)
        pp = perm[side]                                     # new position i holds original column pp[i]
        position = {labels[o]: i for i, o in enumerate(pp)}
        out[side] = (orig, choose_key(d, side, position))
    return out


# ---------------------------------------------------------------- scoring
def init():
    os.chdir(ST)


def work(job):
    regime, m, cases, keys = job
    import valentine549 as V
    from score_screen_metrics import all_metrics, permutations
    from score_seeds import CORPUS, ground_truth_file
    from smx.adapters import read_matrix
    out = []
    for ds, sc, fam in cases:
        k = keys.get(f'{ds}/{sc}')
        if k is None:
            continue
        f_q, f_d = V.seed_file(regime, m, ds, sc), V.seed_file(regime, 'datatype_nanfix', ds, sc)
        g = ground_truth_file((V.NEW_TREE[regime] if V.is_new(ds, sc) else CORPUS[regime]), ds, sc)
        if not (f_q.is_file() and f_d.is_file() and g is not None):
            continue
        q, rows, cols = read_matrix(f_q)
        D = read_matrix(f_d)[0]
        gt = read_matrix(g)[0] > 0.5
        if not (q.shape == D.shape == gt.shape) or not (np.isfinite(q).all() and np.isfinite(D).all()):
            continue
        rl = [r.split('.', 1)[1] if '.' in r else r for r in rows]
        cl = [c.split('.', 1)[1] if '.' in c else c for c in cols]
        base = np.clip(q, 0, None) + 5.0 * D
        pr = permutations(gt.shape, f'{ds}/{sc}')
        row = dict(seed=m, dataset=ds, scenario=sc, group=fam)
        for o, idx in (('o', 0), ('p', 1)):
            ks, kt = set(k['source'][idx]), set(k['target'][idx])
            K = np.outer([l in ks for l in rl], [l in kt for l in cl]).astype(float)
            for gm in G:
                r = all_metrics(base + gm * K, gt, pr)
                row[f'k{o}_{gm}'] = (r['R_exp'] + r['F1_Hr']) / 2
                row[f'k{o}_{gm}_R'] = r['R_exp']
        out.append(row)
    return out


def main():
    os.chdir(ST)
    import valentine549 as V
    from score_final import cell_values, macro_contrast
    L = ['# Key-membership term vs the top-1 key graph (Valentine 549, EXPLORATORY, 2026-11-08)', '']
    allrows = []
    for regime in ('plain', 'md5'):
        reg = REG[regime]
        res = ST / 'tuning' / 'results' / f'xperm_key_{reg}'
        perms = json.loads((res / 'permutations.json').read_text(encoding='utf-8'))
        dep = list(csv.DictReader(open(ST / 'results' / reg / 'deproles_scores.csv', encoding='utf8')))
        seeds = sorted({r['seed'] for r in dep})
        fam = {(r['dataset'], r['scenario']): r['family'] for r in dep}
        big = {f for f in set(fam.values()) if sum(1 for v in fam.values() if v == f) >= 5}
        cases = [c for c in V.cases549() if fam.get((c[0], c[1])) in big]
        keys, fails = {}, 0
        for ds, sc, _ in cases:
            try:
                keys[f'{ds}/{sc}'] = keys_for(regime, ds, sc, perms[f'{ds}/{sc}'])
            except (KeyError, FileNotFoundError, StopIteration):
                fails += 1
        # validation against the stored key-graph outputs
        pk = list(csv.DictReader(open(HERE / 'figures' / 'recall' / f'perm_keygraph_{reg}_rows.csv', encoding='utf8')))
        changed = {}
        for r in pk:
            changed.setdefault(f"{r['dataset']}/{r['scenario']}", []).append(float(r['changed']) > 0)
        tab = {(a, b): 0 for a in (True, False) for b in (True, False)}
        for key, k in keys.items():
            if key in changed:
                # with a key on one side only there is no pair of key-size nodes, so the key graph equals SF
                # (all 180 OpenData scenarios and 2 TPC-DI ones); a change of that one key cannot show
                both = all(k[s][i] for s in ('source', 'target') for i in (0, 1))
                pred = both and ((set(k['source'][0]) != set(k['source'][1])) or
                                 (set(k['target'][0]) != set(k['target'][1])))
                obs = any(changed[key])
                tab[(pred, obs)] += 1
        agree = tab[(True, True)] + tab[(False, False)]
        L += [f'## {reg}', '', f'{len(keys)} scenarios with a reproduced key choice ({fails} could not be read).',
              f'Validation (predicted key change under the permutation vs stored key-graph output changed): agree '
              f'{agree} of {sum(tab.values())}; predicted & observed {tab[(True, True)]}, predicted only '
              f'{tab[(True, False)]}, observed only {tab[(False, True)]}, neither {tab[(False, False)]}.', '']
        with Pool(10, initializer=init) as pool:
            parts = pool.map(work, [(regime, s, cases, keys) for s in seeds])
        rows = [r for p in parts for r in p]
        pkr = {(r['seed'], r['dataset'], r['scenario']): r for r in pk}
        rp = {}
        for r in csv.DictReader(open(HERE / 'figures' / 'loso' / f'role_recall_{regime}.csv', encoding='utf8')):
            rp[(r['matcher'], r['dataset'], r['scenario'])] = float(r['role_R'])
        CH = {'plain': {'ChEMBL': 0.2, 'Magellan': 0.5, 'OpenData': 0.2, 'TPC-DI': 0.5, '_other': 0.2},
              'md5': {'_other': 1.0}}[regime]
        rj = {}
        for r in dep:
            gg = CH.get(r['family'], CH['_other'])
            rj[(r['seed'], r['dataset'], r['scenario'])] = float(r['A_%s_0.0' % gg])
        # gamma: leave one family out on J_tf, by the key term in the original order
        fams = sorted({r['group'] for r in rows})
        choice = {}
        for f in fams:
            tr = [r for r in rows if r['group'] != f]
            # review 8 W4: the criterion is flat above gamma = 1 (with beta = 5 a large gamma only ranks key pairs
            # first), so take the smallest gamma within 0.001 of the best instead of the grid edge
            fit = {gm: np.mean([np.mean([r[f'ko_{gm}'] for r in tr if r['group'] == ff]) for ff in fams if ff != f])
                   for gm in G}
            best = max(fit.values())
            choice[f] = min(gm for gm in G if fit[gm] >= best - 0.001)
        L.append('gamma chosen leave-one-family-out: ' + ', '.join(f'{f} {choice[f]}' for f in fams))
        wide = []
        for r in rows:
            k = (r['seed'], r['dataset'], r['scenario'])
            if k not in pkr or k not in rp or k not in rj:
                continue
            gm = choice[r['group']]
            p = pkr[k]
            w = dict(seed=r['seed'], group=r['group'], dataset=r['dataset'], scenario=r['scenario'],
                     kt=r[f'ko_{gm}'], kt_R=r[f'ko_{gm}_R'], ktp=r[f'kp_{gm}'], ktp_R=r[f'kp_{gm}_R'],
                     key=float(p['key']), key_R=float(p['key_R']), keyp=float(p['keyp']), keyp_R=float(p['keyp_R']),
                     sf=float(p['sf']), sf_R=float(p['sf_R']), role=rj[k], role_R=rp[k])
            wide.append(w)
            allrows.append(dict(w, regime=reg))
        L += ['', '| contrast | J_tf [95%] | Recall@GT [95%] |', '|---|---|---|']
        for a, b, name in (('kt', 'key', 'key term - key graph (original order)'),
                           ('ktp', 'keyp', 'key term - key graph (permuted columns)'),
                           ('kt', 'sf', 'key term - SF as published'),
                           ('kt', 'role', 'key term - role term'),
                           ('role', 'key', 'role term - key graph (reference)')):
            res2 = []
            for suf in ('', '_R'):
                inc, means, _ = cell_values(wide, [a + suf, b + suf])
                res2.append(macro_contrast(means, a + suf, b + suf, np.random.default_rng(20261076)))
            j, rr = res2
            L.append(f'| {name} | {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}] | {rr[0]:+.4f} [{rr[1]:+.4f}, {rr[2]:+.4f}] |')
        L.append(f'\n{len({w["seed"] for w in wide})} matchers, {len({(w["dataset"], w["scenario"]) for w in wide})} '
                 f'scenarios.\n')
    (HERE / 'figures' / 'recall' / 'w4_keyterm.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    with open(HERE / 'figures' / 'recall' / 'w4_keyterm_rows.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        wr.writeheader()
        wr.writerows(allrows)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
