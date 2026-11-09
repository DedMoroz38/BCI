"""Freeze the declared forest20 LOSO winner before computing new test scores."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent
from temporal_features import CANDIDATES

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cv_snapshot():
 out={}
 for v in CANDIDATES:
  files=sorted((HERE/'results'/v).glob('*.json'))
  cv=[f for f in files if json.loads(f.read_text())['protocol']!='test']
  if len(cv)!=27:raise RuntimeError(f'{v}: CV incomplete {len(cv)}/27')
  for f in cv:
   pred=f.with_name(f.stem+'_predictions.npz')
   out[str(f.relative_to(HERE))]=sha(f);out[str(pred.relative_to(HERE))]=sha(pred)
 return out

def validate_selection():
 d=json.loads((HERE/'selection.json').read_text())
 if cv_snapshot()!=d['cv_sha256']:raise RuntimeError('CV outputs changed since selection')
 return d

def main():
 if (HERE/'selection.json').exists():validate_selection();return
 snapshot=cv_snapshot()
 if list((HERE/'results').rglob('*test*.json')):raise RuntimeError('Test results exist before selection')
 rows=[]
 folders={v:HERE/'results'/v for v in CANDIDATES}
 folders.update(original_oas=HERE.parent/'results/eeg',delay1_oas=HERE.parent/'representation_ablation/results/delay1_oas')
 for v,folder in folders.items():
  fs=sorted(folder.glob('forest20_loso*.json'))
  if len(fs)!=8:raise RuntimeError('LOSO folds incomplete')
  rows.append(dict(variant=v,loso_mean=float(np.mean([json.loads(f.read_text())['session_mean'] for f in fs]))))
 rows.sort(key=lambda r:(-r['loso_mean'],r['variant']))
 d=dict(created_utc=datetime.now(timezone.utc).isoformat(),criterion='forest20 mean session-balanced LOSO across 8 subjects',winner=rows[0]['variant'],ranking=rows,cv_sha256=snapshot,policy='Selection fixed before any new test evaluations; previous public test already seen')
 (HERE/'selection.json').write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='cv_sha256'},indent=2))
if __name__=='__main__':main()
