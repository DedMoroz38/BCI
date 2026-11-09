"""Run a fixed CV phase or the test phase after frozen LOSO selection."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,sys,subprocess,argparse,time
HERE=Path(__file__).resolve().parent
from temporal_features import CANDIDATES
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cv','test'],required=True);p.add_argument('--workers',type=int,default=4,choices=range(1,5));args=p.parse_args()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(HERE.parent/'.cache/matplotlib'))
models=['forest20','forest12','logreg'] if args.phase=='cv' else ['forest20','logreg']
jobs=[(v,m) for m in models for v in CANDIDATES]
def run(job):
 v,m=job
 while not all((HERE/'cache'/f'{v}_band{b}.complete.json').exists() for b in range(6)):time.sleep(2)
 log=HERE/'logs'/f'{v}_{m}_{args.phase}.log'
 print('START',args.phase,v,m,flush=True)
 with log.open('w') as f:
  r=subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'--variant',v,'--model',m,'--protocol',args.phase],env=env,stdout=f,stderr=subprocess.STDOUT)
 print('DONE' if r.returncode==0 else 'FAILED',args.phase,v,m,flush=True)
 return job,r.returncode
failed=[]
with ThreadPoolExecutor(max_workers=args.workers) as pool:
 for f in as_completed([pool.submit(run,j) for j in jobs]):
  j,c=f.result()
  if c:failed.append((j,c))
if failed:raise SystemExit(str(failed))
print('PHASE COMPLETE',args.phase,flush=True)
