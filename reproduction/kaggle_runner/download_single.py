"""Guarded paginated download of the single-model experiment's saved outputs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
from kaggle.api.kaggle_api_extended import KaggleApi
p=argparse.ArgumentParser();p.add_argument('target',type=Path)
p.add_argument('--slug',default='den1shh/proteus-single-attention-end-to-end')
p.add_argument('--summaries-only',action='store_true');a=p.parse_args()
api=KaggleApi();api.authenticate();token=None;seen=set();pages=0;count=0
pattern=r'^single-results/'
if a.summaries_only:pattern=r'^single-results/(?:[^/]+\.(?:json|csv|md)|(?:attention_gram|attention_exp|baseline)/[^/]+(?:\.json|_predictions\.npz))$'
while True:
    files,token=api.kernels_output(a.slug,str(a.target),file_pattern=pattern,page_size=200,page_token=token,quiet=True)
    pages+=1;count+=len(files);print('OUTPUT PAGE',pages,'selected',len(files),'total',count,flush=True)
    if not token:break
    if token in seen:raise RuntimeError('Repeated pagination token')
    seen.add(token)
root=a.target/'single-results'
counts={n:len(list((root/(n)).glob('*_predictions.npz'))) for n in ['baseline','attention_gram','attention_exp']}
hashes={str(f.relative_to(a.target)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.target.rglob('*') if f.is_file()}
receipt=dict(source=a.slug,downloaded_utc=datetime.now(timezone.utc).isoformat(),pages=pages,prediction_jobs=counts,files_sha256=hashes)
(a.target/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
print('PREDICTION JOBS',counts,flush=True)
if not all(v==10 for v in counts.values()):raise RuntimeError('Incomplete run; artifacts preserved for diagnosis')
