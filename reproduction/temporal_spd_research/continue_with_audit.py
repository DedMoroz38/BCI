"""Finish both preregistered CV stages, freeze selections, then test fixed functions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time,os,subprocess,sys,json
HERE=Path(__file__).resolve().parent;FOLLOW=HERE/'followup_mixed'
from temporal_features import CANDIDATES
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(HERE.parent/'.cache/matplotlib'))
def count(folder):
 return len([f for f in folder.glob('*.json') if json.loads(f.read_text())['protocol']!='test'])
while True:
 counts=[count(HERE/'results'/v) for v in CANDIDATES]
 if counts==[27]*6 and count(FOLLOW/'results/pyramid2_mixed')==27:break
 time.sleep(5)
for directory in [FOLLOW,HERE]:
 with (directory/'logs/selection.log').open('w') as f:
  subprocess.run([sys.executable,str(directory/'select_model.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['git','add','selection.json','followup_mixed/selection.json'],cwd=HERE,check=True)
subprocess.run(['git','commit','-m','research(selection): freeze raw and numerically eligible LOSO rankings before test'],cwd=HERE,check=True)
print('Both selections committed BEFORE all new public tests.',flush=True)
def test(directory,workers):
 with (directory/'logs/launch-test.log').open('w') as f:
  subprocess.run([sys.executable,'-u',str(directory/'launch.py'),'--phase','test','--workers',str(workers)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(test,HERE,3),pool.submit(test,FOLLOW,1)]
 for f in fs:f.result()
print('All 203 new evaluation jobs complete.',flush=True)
