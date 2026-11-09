from __future__ import annotations
import argparse,concurrent.futures,csv,hashlib,json,os,pathlib,resource,socket,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
ALLOWED={'<cpu-host-00>','<cpu-host-01>'}
if socket.gethostname().split('.')[0] not in ALLOWED:raise SystemExit('This batch is restricted to <cpu-host-00> and <cpu-host-01>')
sys.path.insert(0,str(ROOT/'python'))
from smx.profiling import Profiler,stable_hash,materialize
from smx.datasets import digest,json_write
BASE_RUN=subprocess.run
GIB=1024**3
PYTHON_LIMIT=2*GIB
JAVA_LIMIT=32*GIB
JAVA=pathlib.Path.home()/'opt/jdk21/bin/java'
JAVA_OPTIONS='-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m -XX:MaxDirectMemorySize=256m'

def native_limits():resource.setrlimit(resource.RLIMIT_AS,(JAVA_LIMIT,JAVA_LIMIT))
def native_run(command,*args,**kwargs):
 if command and str(command[0])==str(JAVA):kwargs['preexec_fn']=native_limits
 return BASE_RUN(command,*args,**kwargs)

def initialise():
 resource.setrlimit(resource.RLIMIT_AS,(PYTHON_LIMIT,JAVA_LIMIT))
 subprocess.run=native_run
 os.environ['JAVA_TOOL_OPTIONS']=JAVA_OPTIONS
 os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
 csv.field_size_limit(2**31-1)

class FrozenProfiler(Profiler):
 def __init__(self):
  super().__init__(ROOT/'profile-cache',java=str(JAVA),timeout=7200,heap='8g')
  self.runtime=json.loads((ROOT/'runtime.json').read_text())
  self.java_identity=json.loads((ROOT/'outputs/java-identity.json').read_text())
 def tool(self,kind):
  spec=self.runtime['tools'][kind]
  identity={'snapshot_native_fingerprint':spec['native_fingerprint'],'frozen_source_fingerprint':spec['source_fingerprint'],'java':self.java_identity,'java_tool_options':JAVA_OPTIONS}
  fingerprint=stable_hash(identity)
  path=ROOT/'outputs/tool-identities'/(kind+'.json')
  if not path.exists():json_write(path,identity)
  return {'fingerprint':fingerprint,'bridge':spec['bridge'],'classpath':os.pathsep.join(str(ROOT/p) for p in spec['classpath']),'identity_path':str(path)}

def run_case(relative):
 path=ROOT/relative/'manifest.json';profiler=FrozenProfiler();state={};start=time.monotonic()
 for side in ['source','target']:
  state[side]={}
  for kind in ['fd','ucc','ind']:
   deadline=time.monotonic()+7400
   while True:
    try:
     result=profiler.profile(path,side,kind,.95,1 if kind=='ind' else 2,'distinct');break
    except RuntimeError as exc:
     if 'already running' not in str(exc) or time.monotonic()>deadline:result={'status':'failed','error':str(exc)};break
     time.sleep(1)
    except Exception as exc:result={'status':'failed','error':str(exc)};break
   state[side][kind]=result
   json_write(path.parent/'metadata/profiling.json',state)
 success=all(v.get('status') in {'complete','not_applicable'} for side in state.values() for v in side.values())
 materialized=False;error=None
 if success:
  try:materialize(path,state);materialized=True
  except Exception as exc:error=str(exc)
 status={'scenario':relative,'canonical_complete':success,'materialized':materialized,'profiles':{s:{k:{'status':v['status'],'count':v.get('count'),'key':v.get('key')} for k,v in kinds.items()} for s,kinds in state.items()},'error':error,'seconds':round(time.monotonic()-start,2),'host':socket.gethostname().split('.')[0]}
 key=hashlib.sha256(relative.encode()).hexdigest()[:16]
 json_write(ROOT/'outputs/cases'/(key+'.json'),status)
 return status

def pilot():
 folder=ROOT/'pilot-data/Pilot/case';data=b'id,group,copy\n1,a,a\n2,a,a\n3,b,b\n4,b,b\n'
 manifest={'id':'profiling-gate','csv_policy':{'encoding':'utf-8','delimiter':',','quotechar':'"','escapechar':None,'null_token':'','type_inference':False,'trim_header_whitespace':False},'tables':{}}
 for side in ['source','target']:
  p=folder/side/(side+'.csv');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  manifest['tables'][side]={'path':side+'/'+side+'.csv','sha256':digest(p),'semantics':'observations','rows':4,'columns':[{'id':f'c{i:04d}','index':i,'label':label,'original_label':label} for i,label in enumerate(['id','group','copy'])]}
 json_write(folder/'manifest.json',manifest);marker=folder/'metadata/profile-status.properties';marker.parent.mkdir(exist_ok=True);marker.write_text('source.fd=not_profiled\ntarget.fd=not_profiled\n')
 with concurrent.futures.ProcessPoolExecutor(max_workers=1,initializer=initialise) as pool:status=pool.submit(run_case,'pilot-data/Pilot/case').result()
 assert status['canonical_complete'] and status['materialized'],status
 for side in ['source','target']:
  state=json.loads((folder/'metadata/profiling.json').read_text())[side]
  fd=json.loads(pathlib.Path(state['fd']['path']).read_text())['records']
  actual={(tuple(x['lhs']),tuple(x['rhs'])) for x in fd}
  expected={(('c0000',),('c0001',)),(('c0000',),('c0002',)),(('c0001',),('c0002',)),(('c0002',),('c0001',))}
  assert actual==expected,actual
  assert state['ucc']['count']==1 and state['ind']['count']==2,state
 json_write(ROOT/'outputs/pilot-ok.json',status)
 print(json.dumps({'pilot_passed':True,'host':socket.gethostname().split('.')[0],'counts':status['profiles']},indent=2),flush=True)

def check_runtime():
 runtime=json.loads((ROOT/'runtime.json').read_text())
 for name,expected in runtime['files_sha256'].items():
  if digest(ROOT/name)!=expected:raise ValueError('Frozen runtime file changed: '+name)
 version=BASE_RUN([str(JAVA),'-version'],capture_output=True,text=True,check=True).stderr.strip()
 json_write(ROOT/'outputs/java-identity.json',{'version':version,'executable_sha256':digest(JAVA)})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
 if not 1<=args.workers<=12:raise SystemExit('Worker cap is 12 per allowed host')
 check_runtime()
 if args.pilot:pilot();return
 if not (ROOT/'outputs/pilot-ok.json').exists():raise SystemExit('Run the profiling gate first')
 jobs=json.loads((ROOT/'jobs.json').read_text());todo=[]
 for job in jobs:
  key=hashlib.sha256(job.encode()).hexdigest()[:16];p=ROOT/'outputs/cases'/(key+'.json')
  if not p.exists() or not json.loads(p.read_text()).get('materialized'):todo.append(job)
 mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines()) if v.split()[0].isdigit()}
 if mem['MemAvailable']<args.workers*11*GIB:raise SystemExit('Not enough free RAM for this worker cap')
 start=time.time();state={'host':socket.gethostname().split('.')[0],'workers':args.workers,'total':len(jobs),'pending_at_start':len(todo),'done':len(jobs)-len(todo),'failed':0,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'allowed_hosts':sorted(ALLOWED),'java_heap':'8g','python_address_space_limit_gb':2,'java_address_space_limit_gb':32}
 json_write(ROOT/'outputs/status.json',state)
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=initialise) as pool:
  futures={pool.submit(run_case,job):job for job in todo}
  for future in concurrent.futures.as_completed(futures):
   job=futures[future]
   try:result=future.result();ok=result['materialized']
   except Exception as exc:ok=False;result={'scenario':job,'error':str(exc)}
   state['done']+=1;state['failed']+=not ok;state['elapsed_seconds']=round(time.time()-start,1);state['last']=result
   json_write(ROOT/'outputs/status.json',state)
   print(json.dumps({'done':state['done'],'total':state['total'],'failed':state['failed'],'scenario':job}),flush=True)
 state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());json_write(ROOT/'outputs/status.json',state)
if __name__=='__main__':main()
