"""Finish the predeclared experiment without losing the CV-before-test boundary."""
from pathlib import Path
import time,os,subprocess,sys,json
HERE=Path(__file__).resolve().parent
from temporal_features import CANDIDATES
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR=str(HERE.parent/'.cache/matplotlib'))
while True:
 counts=[len([f for f in (HERE/'results'/v).glob('*.json') if json.loads(f.read_text())['protocol']!='test']) for v in CANDIDATES]
 if counts==[27]*6:break
 time.sleep(5)
print('CV complete; freezing selection before any new test.',flush=True)
with (HERE/'logs/selection.log').open('w') as f:
 subprocess.run([sys.executable,str(HERE/'select_model.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['git','add','selection.json'],cwd=HERE,check=True)
subprocess.run(['git','commit','-m','research(selection): freeze LOSO winner before public test'],cwd=HERE,check=True)
print('Selection committed; starting fixed public-test phase.',flush=True)
with (HERE/'logs/launch-test.log').open('w') as f:
 subprocess.run([sys.executable,'-u',str(HERE/'launch.py'),'--phase','test','--workers','4'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('All 174 new evaluations complete.',flush=True)
