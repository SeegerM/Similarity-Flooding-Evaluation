"""Brief A, 2026-11-06: source and initial-matcher-panel sensitivity of Table II.

Reads the study's stored scores and, for the few Recall-only aggregate cells,
rescoring stored matrices; never runs SF. New outputs only, in figures/loso.
The registered/first-reported J_tf extraction remains unchanged. All Recall
analyses are exploratory. Run with harness/.venv-bdi/Scripts/python.exe -B.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

os.environ['SMX_VALENTINE'] = '549'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ST = ROOT / 'sf-matcher-study'
OUT = HERE / 'figures' / 'loso'
for p in (ROOT / 'harness', ST / 'tuning', ST, ST / 'reference', ST / 'layer3b', HERE):
    sys.path.insert(0, str(p))
import numpy as np
import pandas as pd
import recall_cells as RC
from score_final import cell_values
from explore_x81 import clusters
from make_results_table import ROWS

B = 10_000
SEED = 20261076
COL_NAMES = {'dt': 'Datatype', 'cl': 'Classes', 'ro': 'Roles', 'tc': 'Table context', 'tu': 'Two updates'}
CANON = { (r['dataset'], r['scenario']) for r in RC.read('results', 'valentine549', 'cases.csv') }


@dataclass
class Cell:
    frame: pd.DataFrame
    design: str = 'macro'
    cluster_corpus: str | None = None
    bootstrap_seed: int = SEED
    note: str = ''
    label: str = ''
    col: str = ''


def capture(rows, x, y, design='macro', corpus=None):
    keys = [x, y, x + '_R', y + '_R']
    clean = RC.clean(rows, keys)
    inc, _, _ = cell_values(clean, keys)
    data = [dict(matcher=r['seed'], group=str(r['group']), dataset=r['dataset'], scenario=r['scenario'],
                 J=float(r[x])-float(r[y]), R=float(r[x+'_R'])-float(r[y+'_R']))
            for r in clean if r['seed'] in inc]
    if not data:
        return None
    return Cell(pd.DataFrame(data), design, corpus)


def source_id(dataset, scenario):
    # The study's published source_id rule (explore_benchmark_properties.py).
    # The repeated DBLP prefix is grouped; other Magellan pairs retain their
    # dataset identifiers. These are provenance proxies, not proven independence.
    return f"{dataset.split('-')[0]}/{scenario.split('_')[0]}"


def role_job(job):
    regime, ds, sc, family, matchers = job
    import valentine549 as V
    from explore_deproles import profiles, roles
    from score_seeds import CORPUS, ground_truth_file
    from score_screen_metrics import recall_expected
    from smx.adapters import read_matrix
    from recall_roles import CHOICE
    os.chdir(ST)
    dq = V.seed_file(regime, 'datatype_nanfix', ds, sc)
    truth = ground_truth_file(V.NEW_TREE[regime] if V.is_new(ds, sc) else CORPUS[regime], ds, sc)
    if truth is None or not dq.is_file():
        return []
    D, labels_s, labels_t = read_matrix(dq)
    gt = read_matrix(truth)[0] > .5
    fs, _ = roles(list(labels_s), *profiles(regime, ds, sc, 'source'))
    ft, _ = roles(list(labels_t), *profiles(regime, ds, sc, 'target'))
    role = 1.0 - np.abs(fs[:, None, :] - ft[None, :, :]).mean(axis=2)
    gamma = CHOICE[regime].get(family, CHOICE[regime]['_other'])
    out = []
    for m in matchers:
        f = V.seed_file(regime, m, ds, sc)
        if not f.is_file():
            continue
        q, rs, cs = read_matrix(f)
        if list(rs) != list(labels_s) or list(cs) != list(labels_t):
            raise ValueError(f'Role matrix labels disagree: {regime}/{m}/{ds}/{sc}')
        if q.shape != gt.shape or not np.isfinite(q).all() or not np.isfinite(D).all():
            continue
        out.append(dict(matcher=m, dataset=ds, scenario=sc,
                        role_R=recall_expected(np.clip(q, 0, None)+5*D+gamma*role, gt)))
    return out


def load_roles(workers):
    import valentine549 as V
    from recall_roles import CHOICE
    out = {}
    for regime, label in [('plain','Valentine (readable)'), ('md5','Valentine (recoded)')]:
        dep = RC.read('results', V.vreg(regime), 'deproles_scores.csv')
        fam = {(r['dataset'],r['scenario']):r['family'] for r in dep}
        big = {f for f in fam.values() if list(fam.values()).count(f)>=5}
        sf = {(r['seed'],r['dataset'],r['scenario']):r
              for r in V.assembled(f'gvfull_{V.vreg(regime)}_scores') if r['arm']=='ucc_top1'}
        p = OUT / f'role_recall_{regime}.csv'
        if p.exists():
            scored = pd.read_csv(p)
        else:
            ms = sorted({r['seed'] for r in dep})
            jobs = [(regime,d,s,f,ms) for d,s,f in V.cases549() if fam.get((d,s)) in big]
            parts = []
            with Pool(workers) as pool:
                for i, part in enumerate(pool.imap(role_job, jobs),1):
                    parts.extend(part)
                    if i % 50 == 0: print(f'roles {regime}: {i}/{len(jobs)} scenarios', flush=True)
            scored = pd.DataFrame(parts)
            scored.to_csv(p,index=False)
        lookup = scored.set_index(['matcher','dataset','scenario']).role_R.to_dict()
        rows = []
        for r in dep:
            k = (r['seed'],r['dataset'],r['scenario'])
            if k not in sf or k not in lookup or r['family'] not in big:
                continue
            s = sf[k]
            gamma = CHOICE[regime].get(r['family'],CHOICE[regime]['_other'])
            rows.append(dict(seed=r['seed'],dataset=r['dataset'],scenario=r['scenario'],group=r['family'],
                             a=float(r[f'A_{gamma}_0.0']), a_R=lookup[k],
                             b=(float(s['sf_R_exp'])+float(s['sf_F1_Hr']))/2,b_R=float(s['sf_R_exp'])))
        out[(label,'ro')] = capture(rows,'a','b')
        print(f'roles {regime} loaded',flush=True)
    return out


def multi_job(job):
    import explore_multitable_ext as X
    from score_screen_metrics import recall_expected
    X.J = lambda s,g,p: recall_expected(s,g)
    # FD/UCC completeness is frozen from the stored original scores below;
    # only the melnik comparator is needed to recompute these Recall contrasts.
    X.ARMS = ('melnik',)
    os.chdir(ST)
    return X.score_case(job)


def load_multitable(workers):
    import explore_multitable_ext as X
    out = {}
    for corp, label, cols in [('synthea',r'Synthea$\to$OMOP',{'dt':('A_d','melnik')}),
                              ('indep2','Lahman/MB multi-table',{'dt':('A_d','melnik'),'tc':('A_dt','melnik')})]:
        cs = X.cases(corp)
        if corp=='synthea': cs=[c for c in cs if c[2]=='SyntheaOMOP-Pair']
        ms=sorted(p.name for p in (ST/'reference'/'seeds'/corp).iterdir() if p.is_dir() and p.name not in X.CONTROLS)
        p=OUT/f'multitable_recall_{corp}.csv'
        if p.exists(): rr=pd.read_csv(p)
        else:
            import score_layer1
            dirs={}
            for d in (ST/'tuning'/'results'/X.CORPORA[corp][1]).iterdir():
                if d.is_dir() and d.name.endswith('__melnik'):
                    dirs[(d.name.split('__',1)[0],'melnik')]=score_layer1.run_dir(d)
            jobs=[(corp,d,s,c,ms,dirs) for d,s,c in cs]
            with Pool(workers) as pool: parts=pool.map(multi_job,jobs)
            rr=pd.DataFrame([r for part in parts for r in part])
            rr.to_csv(p,index=False)
        jr=pd.read_csv(ST/'results'/'multitable_fix'/corp/'multitable_scores.csv')
        keys=['seed','dataset','scenario','cell']
        rr=rr.rename(columns={c:c+'_R' for c in rr if c not in keys})
        jr=jr.rename(columns={c:c+'_J' for c in jr if c not in keys})
        rr=rr.merge(jr,on=keys,validate='one_to_one')
        need=['q','A_d','A_dt','A_role','A_rt',*X.ARMS]
        frames=[]
        for cell,g in rr.groupby('cell'):
            cases={(d,s) for d,s,c in cs if c==cell}
            if not cases: continue
            g=g[[ (d,s) in cases for d,s in zip(g.dataset,g.scenario) ]]
            # The old scores retain joint-arm eligibility; Recall only needs
            # datatype/table-context controls and the published comparator.
            g=g.dropna(subset=[f'{n}_J' for n in need]+[f'{n}_R' for n in ('q','A_d','A_dt','melnik')])
            counts=g.groupby('seed').size()
            complete=counts[counts==len(cases)].index
            frames.append(g[g.seed.isin(complete)])
        g=pd.concat(frames,ignore_index=True)
        for col,(a,b) in cols.items():
            frame=pd.DataFrame(dict(matcher=g.seed,group=g.cell,dataset=g.dataset,scenario=g.scenario,
                                    J=g[a+'_J']-g[b+'_J'],R=g[a+'_R']-g[b+'_R']))
            out[(label,col)]=Cell(frame,bootstrap_seed=X.SEED,
                                 note='All arms complete within each database, as original multi-table analysis.')
        print(f'multitable {corp} loaded',flush=True)
    return out


def er10_job(job):
    import score_layer1 as S
    os.chdir(ST)
    return S.score_seed(job)


def registered_role_job(job):
    corpus, ds, sc, family, matchers, dirs = job
    import score_roles_reg as S
    import explore_deproles as X
    from score_screen_metrics import recall_expected
    from smx.adapters import read_matrix
    from explore_composite import load, sf_out
    # These registered arms deliberately preserve their original FD parser,
    # as documented in supplement/protocol.tex; do not use fixed-profile mirrors.
    X.FD_RE = X.FD_RE_FROZEN
    os.chdir(ST)
    qroot=ST/'reference'/'seeds'/corpus
    truth=S.ground_truth_file(S.CORPORA[corpus],ds,sc)
    if truth is None:return []
    D,rs,cs=read_matrix(qroot/'datatype_nanfix'/ds/sc/'seed.csv')
    gt=read_matrix(truth)[0]>.5
    R=S.role_matrix(corpus,ds,sc,rs,cs)
    out=[]
    for m in matchers:
        q=load(qroot/m/ds/sc/'seed.csv',list(rs),list(cs))
        sf=load(sf_out(dirs.get(m),ds,sc),list(rs),list(cs))
        if q is None or sf is None:continue
        out.append(dict(matcher=m,dataset=ds,scenario=sc,
                        role_R=recall_expected(np.clip(q,0,None)+5*np.nan_to_num(D)+.2*R,gt),
                        key_R=recall_expected(sf,gt)))
    return out


def load_registered_roles(workers):
    import score_roles_reg as S
    import score_layer1
    out={}
    for corpus,label in [('ctl','Lahman/MusicBrainz'),('t2d','T2D web tables')]:
        raw=RC.read('tuning','results',f'rolesreg_{corpus}_scores','scores.csv')
        raw=[r for r in raw if r['regime']=='plain']
        arms=['A_d','A_role',*S.ARMS]
        by=defaultdict(dict)
        for r in raw:by[(r['seed'],r['dataset'],r['scenario'])][r['arm']]=float(r['J'])
        cs=S.cases(corpus);cs=[c for c in cs if c[3]=='plain']
        ms=sorted(m for m in {r['seed'] for r in raw} if m not in S.CONTROLS and
                  all(all(a in by[(m,d,s)] for a in arms) for d,s,_,_ in cs))
        cache=OUT/f'registered_role_recall_{corpus}.csv'
        if cache.exists():rvals=pd.read_csv(cache)
        else:
            dirs={m:score_layer1.run_dir(ST/'tuning'/'results'/f'e2x_{corpus}'/f'{m}__ucc_top1') for m in ms}
            with Pool(workers) as pool:parts=pool.map(registered_role_job,[(corpus,d,s,f,ms,dirs) for d,s,f,_ in cs])
            rvals=pd.DataFrame([r for part in parts for r in part]);rvals.to_csv(cache,index=False)
        lookup=rvals.set_index(['matcher','dataset','scenario']).to_dict('index')
        data=[]
        for m in ms:
            for d,s,f,_ in cs:
                k=(m,d,s)
                if k in lookup:
                    data.append(dict(matcher=m,group=f,dataset=d,scenario=s,
                                     J=by[k]['A_role']-by[k]['ucc_top1'],R=lookup[k]['role_R']-lookup[k]['key_R']))
        out[(label,'ro')]=Cell(pd.DataFrame(data),bootstrap_seed=S.SEED,
                               note='NI-only registered J_tf cell: original joint-arm panel and frozen FD parser; exploratory Recall rescored.')
    return out


def load_er10(workers):
    import explore_er10 as E
    from score_seeds import cases_for
    cs=cases_for('indep')
    fam={(d,s):f for d,s,f in cs}
    p=OUT/'er10_scored.csv'
    if p.exists(): sf=pd.read_csv(p)
    else:
        by=defaultdict(list)
        for d in (E.RES/'er10_indep').iterdir():
            if (d/'DONE').is_file() and d.name.split('__',1)[1]=='melnik_shape': by[d.name.split('__',1)[0]].append(str(d))
        with Pool(workers) as pool: parts=pool.map(er10_job,[('indep',m,ds,cs) for m,ds in sorted(by.items())])
        sf=pd.DataFrame([r for part in parts for r in part]);sf.to_csv(p,index=False)
    ad=pd.read_csv(E.RES/'layer1_indep_scores'/'additive_scores.csv')
    keys=['seed','dataset','scenario']
    g=ad.merge(sf[keys+['sf_R_exp','sf_F1_Hr']],on=keys,suffixes=('_base','_shape'),validate='one_to_one')
    allcases={(d,s) for d,s in zip(ad.dataset,ad.scenario)}
    need=['sf_R_exp_base','sf_F1_Hr_base','A_p_R_exp','A_p_F1_Hr','A_d_R_exp','A_d_F1_Hr','sf_R_exp_shape','sf_F1_Hr_shape']
    g=g.dropna(subset=need)
    cnt=g.groupby('seed').size(); g=g[g.seed.isin(cnt[cnt==len(allcases)].index)]
    out={}
    for col,a,b in [('cl','A_p','sf_shape'),('dt','A_d','sf_base')]:
        if b=='sf_shape': r,f=g.sf_R_exp_shape,g.sf_F1_Hr_shape
        else: r,f=g.sf_R_exp_base,g.sf_F1_Hr_base
        dr=g[a+'_R_exp']-r;dj=(dr+g[a+'_F1_Hr']-f)/2
        frame=pd.DataFrame(dict(matcher=g.seed,group=[fam[k] for k in zip(g.dataset,g.scenario)],
                                dataset=g.dataset,scenario=g.scenario,J=dj,R=dr))
        out[('3-DB multi-table',col)]=Cell(frame,bootstrap_seed=E.SEED,
                                         note='Joint completeness over SF, SF+class, datatype and class controls.')
    print('3-DB classes and datatype loaded',flush=True)
    return out


def registry(workers):
    old=RC.contrast;RC.contrast=capture
    try: raw=RC.cells()
    finally: RC.contrast=old
    cells={k:v for k,v in raw.items() if isinstance(v,Cell)}
    # final_A was calculated directly rather than through contrast().
    fa=RC.clean([r for r in RC.num(RC.read('tuning','results','final_A_scores','scores.csv')) if r['truth']=='primary'],
                ['A_dt','melnik','A_dt_R','melnik_R'])
    cells[('Indep.\\ schema pairs','tc')]=capture(fa,'A_dt','melnik')
    # X93 uses whole-source clusters; it is not the family macro used by dt/cl/ro.
    df=pd.read_csv(ST/'tuning'/'results'/'x93_549_scores'/'rows.csv')
    names={'valentine_plain':'Valentine (readable)','valentine_md5':'Valentine (recoded)',
           'gdcsm':'GDC-SM','hdxsm':'HDXSM','goby':'GOBY','madi_products':'MaDI products',
           'madi_indep':'MaDI use cases','final_indep':'Indep.\\ schema pairs'}
    for corp,label in names.items():
        g=df[(df.corpus==corp)&df.arm.isin(['SF_t2','SF_conv'])]
        w=g.pivot(index=['matcher','dataset','scenario'],columns='arm',values=['J','R_exp'])
        ncase=len(w.reset_index()[['dataset','scenario']].drop_duplicates())
        w=w.dropna();cnt=w.groupby(level='matcher').size();w=w[w.index.get_level_values('matcher').isin(cnt[cnt==ncase].index)]
        f=w.reset_index()[['matcher','dataset','scenario']].copy();f.columns=f.columns.get_level_values(0)
        f['group']='all'; f['J']=(w[('J','SF_t2')]-w[('J','SF_conv')]).to_numpy()
        f['R']=(w[('R_exp','SF_t2')]-w[('R_exp','SF_conv')]).to_numpy()
        cells[(label,'tu')]=Cell(f,'source_cluster' if corp.startswith('valentine') else 'cluster',corp,
                                bootstrap_seed=20261008,note='X93 cohort; readable Valentine excludes screen cases.')
    cells.update(load_roles(workers));cells.update(load_multitable(workers));cells.update(load_er10(workers))
    cells.update(load_registered_roles(workers))
    for (lab,col),cell in cells.items():
        cell.label,cell.col=lab,col
        cell.frame=cell.frame.drop_duplicates(['matcher','dataset','scenario'])
        if lab.startswith('Valentine'):
            cell.frame['source']=[source_id(d,s) for d,s in zip(cell.frame.dataset,cell.frame.scenario)]
        else: cell.frame['source']=cell.frame.group.astype(str)
    return cells


def estimates(cell,frame=None, metric='R', design=None):
    frame=cell.frame if frame is None else frame
    d=frame.groupby(['group','dataset','scenario'],sort=True)[metric].mean().reset_index()
    if d.empty:return np.nan,np.nan,np.nan
    design=design or cell.design
    rng=np.random.default_rng(cell.bootstrap_seed)
    if design in ('cluster','source_cluster'):
        if design=='source_cluster': units=[source_id(d,s) for d,s in zip(d.dataset,d.scenario)]
        else:
            # explore_x81 receives bare scenario names; X93 expects dataset/scenario
            corp=cell.cluster_corpus
            raw=[s if corp in ('madi_indep','madi_products') else (f'{ds}/{s}' if cell.col=='tu' else s)
                 for ds,s in zip(d.dataset,d.scenario)]
            cm=clusters(corp,raw);units=[cm[s] for s in raw]
        d['unit']=units
        if len(set(units))>=5:
            g=d.groupby('unit')[metric].agg(['sum','count'])
            ix=rng.integers(0,len(g),size=(B,len(g)))
            boot=g['sum'].to_numpy()[ix].sum(axis=1)/g['count'].to_numpy()[ix].sum(axis=1)
            return float(d[metric].mean()),*np.percentile(boot,[2.5,97.5])
        if cell.col=='tu':return float(d[metric].mean()),np.nan,np.nan
        # Exactly recall_cells.contrast's fallback: conditional within-group CI.
    arrays=[g[metric].to_numpy() for _,g in d.groupby('group',sort=True)]
    boots=np.mean([a[rng.integers(0,len(a),size=(B,len(a)))].mean(axis=1) for a in arrays],axis=0)
    return float(np.mean([a.mean() for a in arrays])),*np.percentile(boots,[2.5,97.5])


def direction(x):
    if not np.isfinite(x):return 'unavailable'
    return 'positive' if x>1e-12 else ('negative' if x< -1e-12 else 'zero')
def status(t):
    if not np.isfinite(t[0]):return 'unavailable'
    return 'positive' if t[1]>0 else ('negative' if t[2]<0 else ('no interval' if not np.isfinite(t[1]) else 'includes 0'))
def fmt(t):
    if not np.isfinite(t[0]):return 'unavailable'
    return f'{t[0]:+.4f}'+(f' [{t[1]:+.4f}, {t[2]:+.4f}]' if np.isfinite(t[1]) else ' [no interval]')


def write_csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False,float_format='%.9g')


def analyse(cells):
    table={(lab,col):v for lab,group,cols in ROWS for col,v in cols.items() if isinstance(v,dict)}
    common={col:set.intersection(*(set(c.frame.matcher) for (lab,k),c in cells.items() if k==col)) for col in COL_NAMES}
    panel=[];loso=[];sources=[];summary=[];paired=[];unit_summary=[]
    for (lab,col),cell in cells.items():
        f=cell.frame
        ids=sorted(set(f.matcher));cp=sorted(common[col]);cf=f[f.matcher.isin(cp)]
        sc=f[['group','dataset','scenario','source']].drop_duplicates()
        groups=sc.groupby('group').size().to_dict()
        for source,g in sc.groupby('source'):
            weight=sum(1/len(groups)/groups[row.group] for row in g.itertuples())
            sources.append(dict(corpus=lab,col=col,source=source,scenarios=len(g),raw_share=len(g)/len(sc),
                                macro_weight=weight,scenario_mean_weight=len(g)/len(sc),
                                original_design=cell.design,canonical_valentine=sum((d,s) in CANON for d,s in zip(g.dataset,g.scenario))))
        for metric in ('R','J'):
            base=estimates(cell,metric=metric)
            ct=estimates(cell,cf,metric) if cp else (np.nan,np.nan,np.nan)
            signchange=np.isfinite(ct[0]) and direction(base[0])!=direction(ct[0])
            panel.append(dict(corpus=lab,col=col,metric=metric,n_matchers=len(ids),matchers=';'.join(ids),
                              n_common=len(cp),common_matchers=';'.join(cp),estimate=base[0],lo=base[1],hi=base[2],
                              common_estimate=ct[0],common_lo=ct[1],common_hi=ct[2],
                              direction_changed=signchange,status_changed=status(base)!=status(ct),
                              panel='flips' if signchange else ('same' if cp==ids else 'smaller'),
                              scenarios=len(sc),groups=len(groups),design=cell.design,note=cell.note))
            if not lab.startswith('Valentine'):continue
            outcomes=[]
            own_values={}
            for source in sorted(set(f.source)):
                reduced=f[f.source!=source]
                t=estimates(cell,reduced,metric)
                outcomes.append(t)
                loso.append(dict(corpus=lab,col=col,metric=metric,left_out=source,n_matchers=len(ids),
                                 scenarios_removed=int((sc.source==source).sum()),scenarios_remaining=len(reduced[['dataset','scenario']].drop_duplicates()),
                                 baseline_estimate=base[0],baseline_lo=base[1],baseline_hi=base[2],estimate=t[0],lo=t[1],hi=t[2],
                                 sign_survives=direction(t[0])==direction(base[0]),
                                 interval_status_survives=status(t)==status(base),original_design=cell.design,
                                 panel_frozen=True,family_eligibility_frozen=True))
                # A source's own point estimate: retain original family weights
                # within that source; no scenario CI presented as source evidence.
                own=estimates(cell,f[f.source==source],metric,'macro')[0]
                own_values[source]=own
                sources.append(dict(corpus=lab,col=col,source=source,metric=metric,source_estimate=own,
                                    source_unit_ci='not reported'))
            for scope in ('all provenance units','three dominant source tables'):
                vv=np.array([v for s,v in own_values.items() if scope=='all provenance units' or
                             s in ('ChEMBL/assays','OpenData/miller2','TPC/prospect')])
                lo=hi=np.nan
                if len(vv)>=5:
                    rng=np.random.default_rng(SEED)
                    lo,hi=np.percentile(vv[rng.integers(0,len(vv),size=(B,len(vv)))].mean(axis=1),[2.5,97.5])
                unit_summary.append(dict(corpus=lab,col=col,metric=metric,scope=scope,units=len(vv),
                                         estimate=float(vv.mean()),lo=lo,hi=hi,
                                         design='equal source-unit means; provenance independence not established'))
            summary.append(dict(corpus=lab,col=col,metric=metric,baseline=base,
                                all_sign=all(direction(t[0])==direction(base[0]) for t in outcomes),
                                all_status=all(status(t)==status(base) for t in outcomes),
                                worst=min(t[0] for t in outcomes),best=max(t[0] for t in outcomes)))
        temp=f.copy();temp['corpus']=lab;temp['col']=col;paired.append(temp)
    write_csv(OUT/'paired_cells.csv',pd.concat(paired,ignore_index=True).to_dict('records'))
    write_csv(OUT/'panel_cells.csv',panel);write_csv(OUT/'loso_cells.csv',loso);write_csv(OUT/'source_cells.csv',sources)
    write_csv(OUT/'source_unit_cells.csv',unit_summary)
    (OUT/'common_panels.json').write_text(json.dumps({k:sorted(v) for k,v in common.items()},indent=2),encoding='utf8')
    # Corpus-level tally per comparison, keeping conflicting Valentine regimes
    # as a mixed result rather than inventing a majority vote.
    counts=[]
    for metric in ('R','J'):
        for col in COL_NAMES:
            rows=[r for r in panel if r['col']==col and r['metric']==metric and (r['corpus'],col) in table
                  and not table[(r['corpus'],col)]['nm']]
            for mode in ('separate regimes','Valentine collapsed','exclude Valentine'):
                for use_common in (False,True):
                    vals=defaultdict(list)
                    for r in rows:
                        name=r['corpus']
                        if name.startswith('Valentine'):
                            if mode=='exclude Valentine':continue
                            if mode=='Valentine collapsed':name='Valentine'
                        t=(r['common_estimate'],r['common_lo'],r['common_hi']) if use_common else (r['estimate'],r['lo'],r['hi'])
                        # Preserve which cells originally had intervals; do not
                        # turn descriptive cells into new significant results.
                        if table[(r['corpus'],col)]['l'] is None:t=(t[0],np.nan,np.nan)
                        vals[name].append(t)
                    ps=[];ss=[]
                    for name,ts in vals.items():
                        ds={direction(t[0]) for t in ts};st={status(t) for t in ts}
                        ps.append(next(iter(ds)) if len(ds)==1 else 'mixed')
                        ss.append(next(iter(st)) if len(st)==1 else 'mixed')
                    counts.append(dict(metric=metric,col=col,mode=mode,panel='common' if use_common else 'original',corpora=len(vals),
                                       point_positive=ps.count('positive'),point_negative=ps.count('negative'),point_zero=ps.count('zero'),point_mixed=ps.count('mixed'),point_unavailable=ps.count('unavailable'),
                                       ci_positive=ss.count('positive'),ci_negative=ss.count('negative'),ci_includes_zero=ss.count('includes 0'),
                                       ci_missing=ss.count('no interval'),ci_mixed=ss.count('mixed')))
    write_csv(OUT/'corpus_counts.csv',counts)
    return panel,loso,sources,summary,counts,table


def report(result,cells):
    panel,loso,sources,summary,counts,table=result
    p=['# Source and initial-matcher-panel robustness — 2026-11-06','',
       'This analysis addresses Brief A. No SF runs, manuscript edits, commits or cluster jobs were made. '
       'All numbers are paired contrasts: term minus the graph comparator, or two updates minus converged SF. '
       'Positive favours the term/two updates. Recall@GT is exploratory; J_tf uses the original stored extraction rule, '
       'not the proposed normalized F1.','',
       '## Design and scope','',
       '- Original complete initial-matcher panels are frozen before every leave-one-source-out deletion. '
       'Original eligible families are fixed; remaining families are equally weighted after deletion. '
       'A family that becomes empty necessarily leaves the macro average; its deletion changes the target population.',
       '- Intervals use 10,000 percentile draws with each cell’s original resampling unit: within-family/database '
       'scenarios for macro cells, whole provenance clusters for X93 two-update cells and whole external clusters '
       'where at least five exist. They are sensitivity intervals, not new registered tests. X93 originally consumed '
       'one random stream across corpora, metrics and several contrasts; this script resets the stream per cell, '
       'so interval endpoints can differ by Monte Carlo variation despite an identical bootstrap design.',
       '- A source is identified with the study’s existing family/scenario-prefix provenance rule. '
       'This yields 10 Valentine units: assays, miller2, prospect; six Magellan prefix groups (two DBLP scenarios share '
       'one group); and musicians. These proxies do not establish independence. Amazon reuse and other pair overlap '
       'require provenance inspection before treating them as independent clusters.',
       '- The three 180-scenario source tables account for 540/549 scenarios. This does not mean 98.4% of the '
       'family-macro estimate: the eligible 7-scenario Magellan family has 25% weight when ChEMBL, OpenData, TPC-DI '
       'and Magellan are the four groups. Wikidata’s two scenarios are excluded from those cells. '
       'The source CSV records both raw shares and macro weights.',
       '- The two class cells retain their earlier-copy coverage. Their helper includes all five families: '
       'the two Wikidata scenarios receive 20% macro weight, unlike datatype and role cells that exclude this '
       'small family. The readable class cohort has 382 cases; the recoded class cohort is smaller. '
       'X93 readable two-update results exclude the '
       'tuning screen, and their mean/whole-source bootstrap differ from the datatype/class/role family macro. '
       'These cohorts are not silently expanded to 549.',
       '- The common panel is the exact-ID intersection over all available corpora in a comparison column, '
       'including the explicitly information-unmatched datatype controls. It is a sensitivity analysis, not an '
       'intrinsically preferred or representative panel. It contains only nine initial matchers. No matcher aliases are merged.',
       '- “Original” in the panel table means the original initial-matcher panel, not that every interval was '
       'published in Table II. Conditional scenario-bootstrap intervals are displayed for sensitivity analysis '
       'even where the paper retains only a point estimate. Original interval availability is preserved in the '
       'corpus-count CSV. Six common-panel point-sign changes are not six significant reversals; the intervals '
       'and effect sizes must be assessed separately.',
       '- No interval is offered for a mean over only the three dominant source tables; their individual effects '
       'are shown below. Source-level generalization is not established by scenario-resampling intervals.','',
       '## Valentine leave-one-source-out results','',
       '| Corpus | Comparison | Metric | Original estimate [95%] | Leave-out range | Sign survives all | Interval status survives all |',
       '|---|---|---|---|---|---|---|']
    for r in sorted(summary,key=lambda r:(r['corpus'],r['col'],r['metric'])):
        p.append(f"| {r['corpus']} | {COL_NAMES[r['col']]} | {r['metric']} | {fmt(r['baseline'])} | {r['worst']:+.4f} to {r['best']:+.4f} | {'yes' if r['all_sign'] else 'NO'} | {'yes' if r['all_status'] else 'NO'} |")
    p+=['','### Source- or panel-sensitive directions','']
    flags=[r for r in loso if not r['sign_survives']]
    for r in flags:p.append(f"- {r['corpus']}, {COL_NAMES[r['col']]}, {r['metric']}: removing {r['left_out']} changes {r['baseline_estimate']:+.4f} to {r['estimate']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}].")
    for r in panel:
        if r['direction_changed']:p.append(f"- Common-panel direction changes: {r['corpus']}, {COL_NAMES[r['col']]}, {r['metric']}: {r['estimate']:+.4f} → {r['common_estimate']:+.4f}; {r['n_matchers']} → {r['n_common']} initial matchers.")
    if not flags and not any(r['direction_changed'] for r in panel):p.append('- No point-sign changes found.')
    p+=['','## Three dominant Valentine sources, each as one unit','',
        'Each entry is the mean contrast for that source on the original frozen panel. '
        'No confidence interval over three source units is reported. Magellan and Wikidata are retained separately in the CSV.','',
        '| Corpus | Comparison | Metric | assays | miller2 | prospect | Equal mean of these 3 |',
        '|---|---|---|---|---|---|---|']
    for (lab,col),cell in sorted(cells.items()):
        if not lab.startswith('Valentine'):continue
        for metric in ('R','J'):
            vals=[]
            for src in ('ChEMBL/assays','OpenData/miller2','TPC/prospect'):
                f=cell.frame[cell.frame.source==src]
                vals.append(estimates(cell,f,metric,'macro')[0] if not f.empty else np.nan)
            p.append(f"| {lab} | {COL_NAMES[col]} | {metric} | "+' | '.join(f'{v:+.4f}' for v in vals)+f' | {np.mean(vals):+.4f} |')
    p+=['','## Per-cell initial-matcher panel sensitivity','',
        'S = same panel, M = smaller common panel with unchanged point-sign, F = point-sign flips. '
        'Use counts (for example `76/9 M`) in a compact panel column, with full IDs in the supplement. '
        'If R and J_tf differ, mark both rather than using one unqualified flag.','',
        '| Corpus | Comparison | Metric | Original n | Common n | Original estimate [95%] | Common-panel estimate [95%] | Panel |',
        '|---|---|---|---|---|---|---|---|']
    for r in sorted(panel,key=lambda r:(r['corpus'],r['col'],r['metric'])):
        p.append(f"| {r['corpus']} | {COL_NAMES[r['col']]} | {r['metric']} | {r['n_matchers']} | {r['n_common']} | {fmt((r['estimate'],r['lo'],r['hi']))} | {fmt((r['common_estimate'],r['common_lo'],r['common_hi']))} | {r['panel']} |")
    p+=['','## Corpus-level counts by comparison','',
        'Only numerical Table II cells are counted; NI-only role entries are not invented as numeric contrasts. '
        'Information-unmatched datatype cells are excluded. Counts are descriptive; corpus variants and comparison '
        'columns are not independent replications. “Mixed” preserves disagreement between Valentine regimes. '
        'The CSV additionally supplies common-panel and interval-based counts. NI-only entries are included '
        'in the panel intersection and per-cell sensitivity table, with their original joint-arm panels and '
        'frozen registered FD parser, but are excluded from these numeric Table II counts.','',
        '| Metric | Comparison | Valentine handling | Corpora | Positive | Negative | Zero | Mixed |',
        '|---|---|---|---|---|---|---|---|']
    for r in counts:
        if r['panel']=='original':p.append(f"| {r['metric']} | {COL_NAMES[r['col']]} | {r['mode']} | {r['corpora']} | {r['point_positive']} | {r['point_negative']} | {r['point_zero']} | {r['point_mixed']} |")
    p+=['','## Source counts and estimator weights','',
        '| Corpus | Comparison | Source | Scenarios | Raw share | Family-macro weight |',
        '|---|---|---|---|---|---|']
    for r in sources:
        if r['corpus'].startswith('Valentine') and 'scenarios' in r:
            p.append(f"| {r['corpus']} | {COL_NAMES[r['col']]} | {r['source']} | {r['scenarios']} | {r['raw_share']:.3f} | {r['macro_weight']:.3f} |")
    p+=['','## Files and reproducibility','',
        '- `make_loso.py`: local analysis, maximum two worker processes by default; `SMX_VALENTINE=549`.',
        '- `figures/loso/loso_cells.csv`: every leave-out estimate, interval and robustness flag.',
        '- `figures/loso/panel_cells.csv`: every original and common panel, including complete initial-matcher IDs.',
        '- `figures/loso/source_cells.csv`: source counts, weights and per-source estimates.',
        '- `figures/loso/source_unit_cells.csv`: equal-source-unit means, descriptive bootstrap over all provenance units, '
        'and the three-dominant-source summary without an interval. The all-unit bootstrap assumes independence '
        'of provenance proxies; it does not prove that assumption.',
        '- `figures/loso/corpus_counts.csv`: separate, collapsed and excluded Valentine counts, by metric and panel.',
        '- `figures/loso/paired_cells.csv` and rescoring CSVs: per-initial-matcher paired inputs, enabling audit without recomputing matrices.',
        '', '### Interpretation limits','',
        'Point-sign robustness and interval robustness are distinct. A small effect crossing zero is not a substantive '
        'reversal by itself. Within-source scenario resampling does not establish transfer to new sources. '
        'These checks do not validate a new F1 extraction policy or correct the original scorer’s absolute tie tolerance.']
    # A round-trip check against the already generated Recall summaries. The
    # source files round to four decimals, hence the explicit rounding bound.
    reference=[]
    for name in ('recall_cells.csv','recall_roles.csv','recall_multitable.csv','recall_er10.csv'):
        reference.append(pd.read_csv(HERE/'figures'/'recall'/name))
    ref=pd.concat(reference,ignore_index=True).drop_duplicates(['corpus','col'],keep='last')
    check=pd.DataFrame(panel).query("metric == 'R'").merge(ref,on=['corpus','col'])
    check['R_abs_difference']=(check.estimate-check.R).abs()
    assert len(check)==51, f'Expected 51 numerical Recall cells; got {len(check)}'
    assert check.R_abs_difference.max()<=0.000051, check.loc[check.R_abs_difference.idxmax()].to_dict()
    write_csv(OUT/'baseline_check.csv',check[['corpus','col','estimate','R','R_abs_difference']].to_dict('records'))
    cp=json.loads((OUT/'common_panels.json').read_text(encoding='utf8'))
    assert all(len(v)>0 for v in cp.values()), 'Empty common panel'
    p+=['','## Verification','',
        f'- All 51 numerical Recall cells reproduce the existing four-decimal summaries: maximum absolute '
        f'point-estimate difference {check.R_abs_difference.max():.8f} (rounding bound 0.00005).',
        f'- Complete output: {len(cells)} numerical/NI-only cells, {len(panel)} metric-specific panel rows, '
        f'{len(loso)} metric-specific leave-out rows.',
        '- Every comparison column has a nonempty common panel of nine initial matchers.',
        '- Registered NI-only role cells are included with original panels of 10 (Lahman/MusicBrainz) and '
        '9 (T2D); their new Recall values are exploratory.',
        '- No new F1 scorer was invoked. The protected manuscript and scoring files were not edited.']
    (HERE/'LOSO_REPORT_2026-11-06.md').write_text('\n'.join(p)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=2:raise ValueError('This task is limited to two worker processes.')
    sys.stdout.reconfigure(encoding='utf8');OUT.mkdir(parents=True,exist_ok=True)
    os.chdir(ST)
    cells=registry(args.workers)
    print(f'Loaded {len(cells)} cells; computing sensitivity intervals',flush=True)
    result=analyse(cells);report(result,cells)
    print(f'Wrote {HERE / "LOSO_REPORT_2026-11-06.md"}',flush=True)


if __name__=='__main__':main()
