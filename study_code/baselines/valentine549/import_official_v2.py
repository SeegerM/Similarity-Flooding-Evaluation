from pathlib import Path
import csv,hashlib,json,sys,time
workspace=Path(sys.argv[1]).resolve();work=workspace/'output/cluster-start-2026-10-30';sys.path.insert(0,str(workspace/'harness'))
from smx.datasets import export_one,digest,json_write
raw=workspace/'code/download_valentine_20261027/extracted';report=json.loads((work/'official-release-audit.json').read_text());valid=[r for r in report['scenarios'] if r['status']=='valid'];invalid=[r for r in report['scenarios'] if r['status']!='valid']
assert len(valid)==549 and len(invalid)==2 and all(r['family']=='Wikidata' for r in invalid)
output=workspace/'harness/generated/benchmark-official-20261030b'
old=json.loads((workspace/'harness/reports/audited_md5_fused_20260904/protocol.json').read_text())['cases'];old_ids={r['id']:r for r in old};existing=[];added=[];corrections=[]
for index,record in enumerate(valid):
 manifest_path=output/record['dataset']/record['scenario']/'manifest.json'
 if not manifest_path.exists():export_one(raw,output,record,False)
 manifest=json.loads(manifest_path.read_text())
 if record['family']=='Magellan' and not manifest.get('repeated_header_removed'):
  source=manifest_path.parent/manifest['tables']['source']['path']
  with source.open(encoding='utf-8',newline='') as f:rows=list(csv.reader(f))
  if len(rows)>1 and rows[1]==rows[0]:
   with source.open('w',encoding='utf-8',newline='') as f:csv.writer(f,lineterminator='\n').writerows([rows[0],*rows[2:]])
   manifest['tables']['source']['sha256']=digest(source);manifest['tables']['source']['rows']-=1
   manifest['repeated_header_removed']={'side':'source','data_row':1,'reason':'Official Magellan release repeats its header; same documented correction as the historical corpus.'}
   json_write(manifest_path,manifest);corrections.append(record['id'])
 for side in ['source','target']:
  assert digest(manifest_path.parent/manifest['tables'][side]['path'])==manifest['tables'][side]['sha256']
 match=old_ids.get(record['id'])
 if match is None and record['family']=='TPC-DI' and record['scenario']=='prospect_horizontal_50_ec_av':
  candidates=[r for r in old if r['family']=='TPC-DI' and r['scenario']==record['scenario']];assert len(candidates)==1;match=candidates[0]
 row={'id':record['id'],'path':str(manifest_path.parent.relative_to(output)).replace('\\','/'),'manifest_sha256':digest(manifest_path),'tables':{side:manifest['tables'][side]['sha256'] for side in ['source','target']}}
 if match is not None:row['prior_id']=match['id'];row['prior_key']=match['key'];existing.append(row)
 else:row['key']=hashlib.sha256(record['id'].encode()).hexdigest()[:16];added.append(row)
 if (index+1)%100==0:print('Imported',index+1,'of',len(valid),flush=True)
assert len(existing)==382 and len(added)==167,(len(existing),len(added))
manifest={'input':'official Valentine v1, archive md5 98da6d6701806538e5c685f4426b95f6','total_release':551,'valid':549,'excluded':invalid,'prior_cohort':382,'new_cohort':167,'prior_cases':existing,'new_cases':added,'magellan_header_corrections':corrections,'forbidden_compute_hosts':'Any compute host other than <cpu-host-00> or <cpu-host-01> until explicitly authorized','allowed_compute_hosts':['<cpu-host-00>','<cpu-host-01>']}
json_write(work/'cohort-549.json',manifest)
print(json.dumps({'imported':549,'new':len(added),'prior':len(existing),'magellan_headers_removed':len(corrections),'new_table_bytes':sum((output/r['path']/side/(side+'.csv')).stat().st_size for r in added for side in ['source','target'])},indent=2))
