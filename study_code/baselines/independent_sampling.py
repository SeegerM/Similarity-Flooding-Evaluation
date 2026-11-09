"""Frozen independent-sample robustness study; see INDEPENDENT_SAMPLING_20260929.md."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, sqlite3, sys, pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import grain_confirm as gc
import grain_pilot as gp
from smx.grain import measure_table, grain_from_measures
from smx.core import mutual_top1
from encrypted_feature_ablation import pmf_scores, scale_aligned_fuse, NULLS
from regime_tables import label_matrix
from best_combination import char3_channel
from edbt_followups import allowed_map, loao_mask, permute_classes

HERE = Path(__file__).resolve().parent
OUT = HERE / 'reports/independent_sampling_20260929'
DATA = HERE / 'generated/independent_sampling_20260929'
SEEDS = (20260929, 20260930, 20260931)
CONDITIONS = ('identical', 'half', 'disjoint', 'independent')
CHINOOK = {
    'name':'Chinook', 'root':DATA/'raw/chinook', 'delimiter':';', 'base':'Invoice',
    'tables':{t:(t+'.csv', k) for t,k in [('Artist','ArtistId'),('Album','AlbumId'),
        ('Genre','GenreId'),('MediaType','MediaTypeId'),('Track','TrackId'),
        ('Customer','CustomerId'),('Invoice','InvoiceId'),('InvoiceLine','InvoiceLineId')]},
    'fks':{('Album','ArtistId'):'Artist',('Track','AlbumId'):'Album',('Track','GenreId'):'Genre',
        ('Track','MediaTypeId'):'MediaType',('Invoice','CustomerId'):'Customer',
        ('InvoiceLine','InvoiceId'):'Invoice',('InvoiceLine','TrackId'):'Track'},
    'variants':{
        'N':{t:(t,[]) for t in ('Artist','Album','Genre','MediaType','Track','Customer','Invoice','InvoiceLine')},
        'D1':{'line':('InvoiceLine',[('TrackId',None),('AlbumId',None),('ArtistId',None)]),
              'invoice':('Invoice',[('CustomerId',None)]),'genre':('Genre',[]),'media':('MediaType',[])},
        'D2':{'line':('InvoiceLine',[('InvoiceId',None),('CustomerId',None)]),
              'track':('Track',[('GenreId',None),('MediaTypeId',None)]),
              'album':('Album',[('ArtistId',None)])}}}
DBS = {d['name']:d for d in (gc.TPCH,gc.AW,CHINOOK)}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=1,ensure_ascii=False),encoding='utf-8')

def csvout(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def prepare():
    if (OUT/'manifest.json').exists(): raise RuntimeError('Manifest already exists')
    con=sqlite3.connect(f'file:{(DATA/"raw/Chinook.sqlite").as_posix()}?mode=ro',uri=True)
    for table,(filename,key) in CHINOOK['tables'].items():
        cur=con.execute('SELECT * FROM "'+table+'"')
        path=CHINOOK['root']/filename; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f,delimiter=';'); w.writerow([c[0] for c in cur.description]); w.writerows(cur)
    con.close()
    inputs=[HERE/'INDEPENDENT_SAMPLING_20260929.md',DATA/'raw/Chinook.sqlite']
    inputs += list(HERE.glob('*.py'))+list((HERE/'smx').rglob('*.py'))
    inputs += [d['root']/f for d in DBS.values() for f,k in d['tables'].values()]
    specs={name:{**d,'fks':[(list(k),v) for k,v in d['fks'].items()]} for name,d in DBS.items()}
    manifest={'seeds':SEEDS,'conditions':CONDITIONS,'database_specs':specs,
        'inputs':{str(p):digest(p) for p in inputs},'python':sys.version,
        'numpy':np.__version__,'source_url':'https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite'}
    # Paths are strings in the serialized scientific specification.
    dump(OUT/'manifest.json',json.loads(json.dumps(manifest,default=str)))
    print('Frozen',len(inputs),'inputs',flush=True)

def verify_inputs():
    m=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    for p,h in m['inputs'].items():
        assert digest(p)==h, 'Frozen input changed: '+p
    return m

def choose(keys,n,seed,condition,source=None):
    rng=random.Random(seed)
    if source is None: return set(rng.sample(keys,n))
    if condition=='identical': return set(source)
    if condition=='independent': return set(rng.sample(keys,n))
    shared=n//2 if condition=='half' else 0
    outside=[x for x in keys if x not in source]
    return set(rng.sample(sorted(source),shared)+rng.sample(outside,n-shared))

def closure(db,selected):
    base=db['base']; bk=db['tables'][base][1]
    rows={base:list(gc.read_rows(db,base,lambda r:r[bk] in selected))}
    keys={base:set(selected)}
    for (t,c),ref in db['fks'].items():
        if ref==base and t not in rows: rows[t]=list(gc.read_rows(db,t,lambda r,c=c:r[c] in selected))
    changed=True
    while changed:
        changed=False
        for (t,c),ref in db['fks'].items():
            if t not in rows or ref==base: continue
            needed={r[c] for r in rows[t] if r[c]}; have=keys.get(ref,set())
            if needed-have:
                k=db['tables'][ref][1]; want=needed|have
                new=list(gc.read_rows(db,ref,lambda r,k=k:r[k] in want))
                got={r[k] for r in new}
                assert needed <= got, (db['name'],t,c,'missing FK',len(needed-got))
                rows[ref]=new; keys[ref]=got; changed=True
    for t in db['tables']: assert rows.get(t), (db['name'],t,'empty table')
    return rows

def canonical(v):
    return '' if v.strip().lower() in NULLS else v.strip()

def recode(cols,key):
    return [[hashlib.sha256(f'{key}|{i}|{v}'.encode()).hexdigest() if v else '' for v in c]
            for i,c in enumerate(cols)]

def representation(db,variant,rows,side,folder):
    result={}
    for table,(prov,trows,raw) in sorted(gc.build_variant(db,rows,variant).items()):
        cols=[[canonical(r[i]) for r in trows] for i in range(len(raw))]
        names=gc.labels_for(db,prov,raw,side,'plain',variant,table)
        gc.write_table(folder/variant/side/(table+'.csv'),names,list(zip(*cols)))
        m=measure_table(cols); g=grain_from_measures(m)
        opaque=recode(cols,f'20260929|{side}|{variant}|{table}')
        og=grain_from_measures(measure_table(opaque))
        assert g.home==og.home and g.edges==og.edges and g.distinct==og.distinct
        for a,b in zip(cols,opaque):
            from edbt_followups import partition_check
            partition_check(a,b)
        result[table]={'cols':cols,'opaque':opaque,'labels':names,'prov':prov,
            'homes':[(table,g.class_key(i)) for i in range(len(cols))],
            'entities':[gc.entity(db,p) for p in prov],
            'cards':{(table,g.class_key(i)):m.rows if g.class_key(i)=='FACT' else
                 min(m.distinct[int(j)] for j in g.class_key(i).split('+')) for i in range(len(cols))},
            'summary':g.summary()}
    return result

def flatten(rep,regime,side):
    out={k:[] for k in ('cols','labels','prov','homes','entities')}; out['cards']={}
    for table,r in rep.items():
        for k in ('prov','homes','entities'): out[k].extend(r[k])
        out['cols'].extend(r['cols' if regime=='plain' else 'opaque'])
        out['labels'].extend(r['labels'] if regime=='plain' else
            [hashlib.sha256(f'{side}|{table}|{i}'.encode()).hexdigest()[:16] for i in range(len(r['cols']))])
        out['cards'].update(r['cards'])
    return out

def score_pair(db,seed,condition,sv,tv,srep,trep,folder):
    records=[]
    for regime in ('plain','opaque'):
        s,t=flatten(srep,regime,'source'),flatten(trep,regime,'target')
        gt=np.asarray([[gc.corresponds(a,b) for b in t['prov']] for a in s['prov']],bool)
        assert gt.any()
        pmf=pmf_scores(s['cols'],t['cols']); lev=np.asarray(label_matrix(s['labels'],t['labels']))
        bases={'pmf':pmf,'levenshtein':lev,'seed':scale_aligned_fuse(lev,pmf,.35),
               'char3':char3_channel(s['cols'],t['cols'],20260903)}
        np.savez_compressed(folder/f'{sv}_{tv}_{regime}.npz',gt=gt,**bases)
        for matcher,S in bases.items():
            si,ti=s['homes'],t['homes']; se,te=s['entities'],t['entities']
            masks={'base':np.ones(S.shape,bool),
                'CVA':allowed_map(si,ti,gp._vote_map(si,ti,S)),
                'H':allowed_map(si,ti,gp._hungarian_map(si,ti,S)),
                'HC':allowed_map(si,ti,gp._hungarian_map(si,ti,S,(s['cards'],t['cards']))),
                'LOAO':loao_mask(si,ti,S),
                'inferred_GT':allowed_map(si,ti,gp._class_map(si,ti,gt,False)),
                'annotated_GT':allowed_map(se,te,gp._class_map(se,te,gt,True))}
            for k in range(10):
                rs=permute_classes(se,f'{db}|{seed}|{sv}|{tv}|s|{k}')
                rt=permute_classes(te,f'{db}|{seed}|{sv}|{tv}|t|{k}')
                masks[f'random_GT_{k}']=allowed_map(rs,rt,gp._class_map(rs,rt,gt,True))
            baseline=gp._metrics(S,gt); M=mutual_top1(S)
            for arm,mask in masks.items():
                masked=np.where(mask,S,float(S.min())-1); mets=gp._metrics(masked,gt)
                MM=mutual_top1(masked)
                if arm=='CVA': assert np.all(M<=MM)
                records.append({'database':db,'seed':seed,'condition':condition,'pair':sv+'_'+tv,
                    'regime':regime,'matcher':matcher,'arm':arm,'gt':int(gt.sum()),
                    'retained':float(mask.mean()),'lost_gt':int((gt&~mask).sum()),
                    'removed_fp':int((M&~gt&~MM).sum()),'lost_tp':int((M&gt&~MM).sum()),
                    **mets,**{'base_'+k:v for k,v in baseline.items()},
                    **{'delta_'+k:mets[k]-v for k,v in baseline.items()}})
    return records

def run_job(job):
    dbname,seed=job; db=DBS[dbname]; folder=OUT/'cases'/f'{dbname}_{seed}'
    if (folder/'done.json').exists(): return dbname,seed,'cached'
    folder.mkdir(parents=True,exist_ok=True)
    n=150 if dbname=='Chinook' else 1500
    key=db['tables'][db['base']][1]
    population=[r[key] for r in gc.read_rows(db,db['base'])]
    assert len(set(population))==len(population) and len(population)>=2*n
    src=choose(population,n,seed,'source'); sr=closure(db,src)
    sf=DATA/f'{dbname}_{seed}'/'source'
    sv={v:representation(db,v,sr,'source',sf) for v in db['variants']}
    records=[]; overlaps=[]; samples={}
    profiles={'source':{v:{t:r['summary'] for t,r in rep.items()} for v,rep in sv.items()}}
    for condition in CONDITIONS:
        selected=choose(population,n,seed+100000,condition,src)
        tr=sr if condition=='identical' else closure(db,selected)
        samples[condition]={'source':sorted(src),'target':sorted(selected),
            'root_intersection':len(src&selected),'population':len(population)}
        tf=DATA/f'{dbname}_{seed}'/condition
        tv={v:representation(db,v,tr,'target',tf) for v in db['variants']}
        profiles[condition]={v:{t:r['summary'] for t,r in rep.items()} for v,rep in tv.items()}
        for table,(_,pk) in db['tables'].items():
            a={r[pk] if pk else tuple(r.values()) for r in sr[table]}
            b={r[pk] if pk else tuple(r.values()) for r in tr[table]}
            overlaps.append({'database':dbname,'seed':seed,'condition':condition,'table':table,
                'source':len(a),'target':len(b),'shared':len(a&b),'jaccard':len(a&b)/len(a|b)})
        dest=folder/condition; dest.mkdir(exist_ok=True)
        for a,b,_ in gc.PAIRINGS[:3]: records.extend(score_pair(dbname,seed,condition,a,b,sv[a],tv[b],dest))
        print(dbname,seed,condition,'scored',flush=True)
    csvout(folder/'metrics.csv',records); csvout(folder/'overlap.csv',overlaps)
    dump(folder/'samples.json',samples); dump(folder/'profiles.json',profiles)
    dump(folder/'done.json',{'rows':len(records)})
    return dbname,seed,len(records)

def summarize():
    records=[]; overlap=[]
    for db in DBS:
        for seed in SEEDS:
            folder=OUT/'cases'/f'{db}_{seed}'; assert (folder/'done.json').exists()
            for name,dest in [('metrics',records),('overlap',overlap)]:
                with (folder/(name+'.csv')).open(encoding='utf-8') as f: dest.extend(csv.DictReader(f))
    csvout(OUT/'metrics.csv',records); csvout(OUT/'overlap.csv',overlap)
    groups=defaultdict(lambda:defaultdict(list))
    for r in records:
        arm='random_GT_mean' if r['arm'].startswith('random_GT_') else r['arm']
        groups[(r['database'],r['condition'],r['regime'],r['matcher'],arm)][r['seed']].append(r)
    summary=[]
    for key,seeds in sorted(groups.items()):
        row=dict(zip(('database','condition','regime','matcher','arm'),key))
        for metric in ('f1_mutual','f1_hungarian','recall_at_gt','mrr'):
            for prefix in ('','delta_'):
                vals=[np.mean([float(r[prefix+metric]) for r in rs]) for rs in seeds.values()]
                row[prefix+metric]=float(np.mean(vals)); row[prefix+metric+'_seed_min']=float(min(vals)); row[prefix+metric+'_seed_max']=float(max(vals))
        summary.append(row)
    csvout(OUT/'summary.csv',summary)
    verify_inputs()
    files=list((OUT/'cases').rglob('*'))+list(DATA.rglob('*.csv'))
    dump(OUT/'output_digests.json',{str(p):digest(p) for p in files if p.is_file()})
    print('Complete',len(records),'metric rows;',len(summary),'summary rows',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['prepare','run','summarize']); ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='summarize': summarize()
    else:
        verify_inputs()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs=[pool.submit(run_job,(db,seed)) for db in DBS for seed in SEEDS]
            for j in as_completed(jobs): print(j.result(),flush=True)
        summarize()
