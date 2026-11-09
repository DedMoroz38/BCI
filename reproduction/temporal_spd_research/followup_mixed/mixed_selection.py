"""Select only numerically eligible R(X) before any new public-test scores."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,numpy as np
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=PARENT.parent
sys.path.insert(0,str(PARENT))
from temporal_features import CANDIDATES as PRIMARY
EXCLUDED={'pyramid2_oas':'400 mathematically null tangent coordinates amplified by source scaler; measured variance <=4.53e-27','quadrature_oas':'Complex structure forces null self-quadrature tangent coordinates; not eligible without numerical handling'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def folders():
 d={v:PARENT/'results'/v for v in PRIMARY};d['pyramid2_mixed']=HERE/'results/pyramid2_mixed';return d

def cv_snapshot():
 out={}
 for v,directory in folders().items():
  files=[f for f in directory.glob('*.json') if json.loads(f.read_text())['protocol']!='test']
  if len(files)!=27:raise RuntimeError(f'{v}: incomplete {len(files)}/27 CV')
  for f in files:
   pred=f.with_name(f.stem+'_predictions.npz')
   out[str(f.relative_to(PARENT))]=sha(f);out[str(pred.relative_to(PARENT))]=sha(pred)
 return out

def validate_selection():
 d=json.loads((HERE/'selection.json').read_text())
 if d['cv_sha256']!=cv_snapshot():raise RuntimeError('CV changed after final selection')
 return d

def main():
 if (HERE/'selection.json').exists():validate_selection();return
 snap=cv_snapshot()
 if list((PARENT/'results').rglob('*test*.json')) or list((HERE/'results').rglob('*test*.json')):raise RuntimeError('New test scores exist before final selection')
 allfolders=folders();allfolders.update(original_oas=ROOT/'results/eeg',delay1_oas=ROOT/'representation_ablation/results/delay1_oas')
 rows=[]
 for v,directory in allfolders.items():
  fs=list(directory.glob('forest20_loso*.json'))
  if len(fs)!=8:raise RuntimeError('Primary folds incomplete')
  rows.append(dict(variant=v,loso_mean=float(np.mean([json.loads(f.read_text())['session_mean'] for f in fs])),eligible=v not in EXCLUDED))
 rows.sort(key=lambda r:(-r['loso_mean'],r['variant']))
 winner=next(r['variant'] for r in rows if r['eligible'])
 d=dict(created_utc=datetime.now(timezone.utc).isoformat(),criterion='forest20 mean LOSO session-balanced accuracy, numerical eligibility first',winner=winner,ranking=rows,excluded=EXCLUDED,cv_sha256=snap,protocol_sha256=sha(HERE/'PROTOCOL.md'),policy='Frozen before all new test scores; numeric quality gate and fixed mixing preregistered')
 (HERE/'selection.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='cv_sha256'},indent=2))
if __name__=='__main__':main()
