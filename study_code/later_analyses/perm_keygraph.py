"""Does the key graph's lead over SF as published on Valentine depend on column order? (2026-11-06; EXPLORATORY pilot)

The key graph (top-1 UCC edges) chooses its key with a score whose position part rates columns further left as more
key-like, and Valentine's ground truth is positional. Pilot on X29's 60 readable screen scenarios
(sf-matcher-study/tuning/corpus/plain/screen_perm: CSV columns and truth permuted per side, FD/UCC/IND files copied,
since they name their columns; permutations in results/plain/permutation_check_perms.json). SF as published is
already known to be order-invariant there (X29: identical output).

  --prepare  reorder every matcher's stored initial matrix (reference/seeds/plain/<m>) to the permuted tables and copy
             the original key-graph configuration of each matcher (tuning/configs/gvfull_plain/<m>__ucc_top1),
             changing only the corpus, seed and results folders and the dataset list;
  --run      one Schematch JVM per matcher (tuning/run_tuning.run_one);
  --score    original key graph (tuning/results/gvfull_plain/<m>__ucc_top1), permuted key graph put back into the
             original order, and SF as published (tuning/results/layer1_plain/<m>__melnik), scored with the paper's
             scorer on the original truth; family macro, scenario bootstrap within family.

Run (from this folder):  ../../harness/.venv-bdi/Scripts/python.exe perm_keygraph.py --prepare --run --score
"""
import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / 'sf-matcher-study'
TUNING = ST / 'tuning'
for p in (ST.parent / 'harness', TUNING, ST, ST / 'reference'):
    sys.path.insert(0, str(p))

REGIME = 'plain'


def configure(regime):
    """plain: X29's permuted screen copy; md5: a copy built here with X29's rule (same permutation function)."""
    global REGIME, PERM_CORPUS, ORIG_CORPUS, PERMS, SEEDS, PSEEDS, CFG, RES, ORIG_CFG
    REGIME = regime
    ORIG_CORPUS = TUNING / 'corpus' / regime / 'screen'
    if regime == 'plain':
        PERM_CORPUS = TUNING / 'corpus' / 'plain' / 'screen_perm'
        PERMS = ST / 'results' / 'plain' / 'permutation_check_perms.json'
    else:
        PERM_CORPUS = TUNING / 'corpus' / regime / 'screen_perm_key'
        PERMS = ST / 'results' / regime / 'perm_key_perms.json'
    SEEDS = ST / 'reference' / 'seeds' / regime
    PSEEDS = ST / 'reference' / 'seeds' / f'{regime}_perm_key'
    CFG = TUNING / 'configs' / f'xperm_key_{regime}'
    RES = TUNING / 'results' / f'xperm_key_{regime}'
    ORIG_CFG = TUNING / 'configs' / f'gvfull_{regime}'


configure('plain')


def build_perm_corpus():
    """X29's construction (explore_permutation_check.prepare) for a regime without a permuted copy."""
    import hashlib
    from explore_permutation_check import perm_for
    if PERM_CORPUS.exists():
        shutil.rmtree(PERM_CORPUS)
    perms = {}
    for ds_dir in sorted(d for d in ORIG_CORPUS.iterdir() if d.is_dir()):
        for sc_dir in sorted(d for d in ds_dir.iterdir() if d.is_dir()):
            ds, sc = ds_dir.name, sc_dir.name
            src, dst = sc_dir.resolve(), PERM_CORPUS / ds / sc
            shutil.copytree(src / 'metadata', dst / 'metadata')
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f.name)
            p = {}
            for side in ('source', 'target'):
                rr = rows(src / side / f'{side}.csv')
                p[side] = perm_for(f'{ds}/{sc}/{side}', len(rr[0]))
                write(dst / side / f'{side}.csv', [[r[i] for i in p[side]] if len(r) == len(rr[0]) else r for r in rr])
            status = dst / 'metadata' / 'profile-status.properties'
            if status.is_file():
                lines = status.read_text(encoding='utf-8').splitlines()
                for side in ('source', 'target'):
                    h = hashlib.sha256((dst / side / f'{side}.csv').read_bytes()).hexdigest()
                    lines = [f'{side}.table.sha256={h}' if l.startswith(f'{side}.table.sha256=') else l for l in lines]
                status.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            gt_files = list((src / 'ground_truth').iterdir())
            assert len(gt_files) == 1
            g = rows(gt_files[0])
            write(dst / 'ground_truth' / gt_files[0].name, [[g[i][j] for j in p['target']] for i in p['source']])
            perms[f'{ds}/{sc}'] = p
    PERMS.parent.mkdir(parents=True, exist_ok=True)
    PERMS.write_text(json.dumps(perms), encoding='utf-8')
    print(f'{len(perms)} {REGIME} scenarios permuted into {PERM_CORPUS}')
CONTROLS = {'constant', 'random', 'oracle', 'oracle_noise025', 'oracle_noise050', 'oracle_erase050',
            'oracle_decoy100', 'embedding_random', 'rema_shuffled', 'semstruct_lm_only', 'datatype_nanfix'}


def rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.reader(f))


def write(path, rr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rr)


def matchers():
    return sorted(d.name.split('__')[0] for d in ORIG_CFG.iterdir()
                  if d.name.endswith('__ucc_top1') and d.name.split('__')[0] not in CONTROLS)


def prepare():
    perms = json.loads(PERMS.read_text(encoding='utf-8'))
    datasets = sorted({k.split('/')[0] for k in perms})
    for p in (PSEEDS, CFG):
        if p.exists():
            shutil.rmtree(p)
    ms = []
    for m in matchers():
        ok = True
        for key, p in perms.items():
            ds, sc = key.split('/')
            f = SEEDS / m / ds / sc / 'seed.csv'
            if not f.is_file():
                ok = False
                break
            s = rows(f)
            head, body = s[0], s[1:]
            new = [[head[0]] + [head[1 + j] for j in p['target']]]
            new += [[body[i][0]] + [body[i][1 + j] for j in p['target']] for i in p['source']]
            write(PSEEDS / m / ds / sc / 'seed.csv', new)
        if not ok:
            shutil.rmtree(PSEEDS / m, ignore_errors=True)
            continue
        src = ORIG_CFG / f'{m}__ucc_top1'
        dst = CFG / f'{m}__ucc_top1'
        shutil.copytree(src, dst)
        rep = {
            f'defaultDatasetBasePath: "<CLUSTER_HOME>/smx/corpus/{REGIME}"':
                f'defaultDatasetBasePath: "{PERM_CORPUS.as_posix()}"',
            (TUNING / 'results' / f'gvfull_{REGIME}' / f'{m}__ucc_top1').as_posix(): (RES / f'{m}__ucc_top1').as_posix(),
            (SEEDS / m).as_posix(): (PSEEDS / m).as_posix(),
        }
        for f in dst.iterdir():
            if f.suffix not in ('.yaml', '.properties'):
                continue
            t = f.read_text(encoding='utf-8')
            for a, b in rep.items():
                t = t.replace(a, b)
            f.write_text(t, encoding='utf-8')
        (dst / 'datasets.yaml').write_text(''.join(f'---\nname: "{d}"\npath: "{d}"\n' for d in datasets),
                                           encoding='utf-8')
        g = (dst / 'general.yaml').read_text(encoding='utf-8')
        s = (dst / 'second_line_matchers.yaml').read_text(encoding='utf-8')
        assert PERM_CORPUS.as_posix() in g and (RES / f'{m}__ucc_top1').as_posix() in g, (m, 'general.yaml')
        assert (PSEEDS / m).as_posix() in s, (m, 'seed dir')
        ms.append(m)
    print(f'{len(perms)} scenarios, {len(ms)} matchers prepared')


def run(workers, xmx):
    from run_seeds import classpath
    from run_tuning import run_one
    deps = classpath()
    todo = [c for c in sorted(CFG.iterdir()) if not (RES / c.name / 'DONE').is_file()]
    print(f'{len(todo)} configurations to run', flush=True)
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(run_one, c, RES / c.name, deps, xmx) for c in todo]
        for f in as_completed(futs):
            label, code, secs = f.result()
            print(f'{label:40s} {"ok" if code == 0 else f"FAILED {code}"} ({secs:.0f}s)', flush=True)


def score():
    from score_screen_metrics import recall_expected, hungarian_random, permutations
    from score_seeds import latest_run, one
    from smx.adapters import read_matrix
    from score_final import macro_contrast, cell_values
    perms = json.loads(PERMS.read_text(encoding='utf-8'))
    fam = {}
    for key in perms:
        ds = key.split('/')[0]
        fam[key] = ds.split('-')[0]
    out = []
    for m in sorted(d.name.split('__')[0] for d in CFG.iterdir()):
        o_key = latest_run(TUNING / 'results' / f'gvfull_{REGIME}' / f'{m}__ucc_top1')
        p_key = latest_run(RES / f'{m}__ucc_top1')
        o_sf = latest_run(TUNING / 'results' / f'layer1_{REGIME}' / f'{m}__melnik')
        if not (o_key and p_key and o_sf):
            continue
        for key, p in perms.items():
            ds, sc = key.split('/')
            fs = [one(r / ds / sc / '_outputs' / 'MatchingStepLine2') for r in (o_key, p_key, o_sf)]
            g = one(ORIG_CORPUS / ds / sc / 'ground_truth')
            if not all(fs) or g is None:
                continue
            K, P, S = (read_matrix(f)[0] for f in fs)
            gt = read_matrix(g)[0] > 0.5
            if not (K.shape == P.shape == S.shape == gt.shape) or not all(np.isfinite(M).all() for M in (K, P, S)):
                continue
            back = np.empty_like(P)
            back[np.ix_(p['source'], p['target'])] = P
            pr = permutations(gt.shape, f'{ds}/{sc}')
            r = {}
            for name, M in (('key', K), ('keyp', back), ('sf', S)):
                Mc = np.clip(M, 0, None)
                re_, f1 = recall_expected(Mc, gt), hungarian_random(Mc, gt, pr)
                r[name], r[name + '_R'] = (re_ + f1) / 2, re_
            out.append(dict(seed=m, group=fam[key], dataset=ds, scenario=sc, changed=float(np.abs(back - K).max() > 1e-9),
                            **r))
    keys = ['key', 'keyp', 'sf', 'key_R', 'keyp_R', 'sf_R']
    inc, means, _ = cell_values(out, keys)
    rng = lambda: np.random.default_rng(20261106)
    res = {}
    for metric, suf in (('J', ''), ('R', '_R')):
        res[metric] = {'key - SF (original order)': macro_contrast(means, 'key' + suf, 'sf' + suf, rng()),
                       'key - SF (permuted columns)': macro_contrast(means, 'keyp' + suf, 'sf' + suf, rng()),
                       'permuted - original key': macro_contrast(means, 'keyp' + suf, 'key' + suf, rng())}
    ch = np.mean([r['changed'] for r in out if r['seed'] in inc])
    lines = [f'# Key graph vs SF as published under column permutation (Valentine {REGIME} screen, EXPLORATORY)', '',
             f'{len(means)} scenarios, {len(inc)} matchers; key-graph output changed by the permutation in {ch:.1%} '
             'of matcher-scenario cases.', '', '| contrast | J_tf [95%] | Recall@GT [95%] |', '|---|---|---|']
    for k in res['J']:
        j, r = res['J'][k], res['R'][k]
        lines.append(f'| {k} | {j[0]:+.4f} [{j[1]:+.4f}, {j[2]:+.4f}] | {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}] |')
    (HERE / 'figures' / 'recall').mkdir(parents=True, exist_ok=True)
    (HERE / 'figures' / 'recall' / f'perm_keygraph_{REGIME}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with open(HERE / 'figures' / 'recall' / f'perm_keygraph_{REGIME}_rows.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--xmx', default='4g')
    ap.add_argument('--regime', choices=('plain', 'md5'), default='plain')
    a = ap.parse_args()
    configure(a.regime)
    if a.prepare:
        if a.regime != 'plain':
            build_perm_corpus()
        prepare()
    if a.run:
        run(a.workers, a.xmx)
    if a.score:
        score()


if __name__ == '__main__':
    main()
