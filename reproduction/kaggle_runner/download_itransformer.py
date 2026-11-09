"""Guarded paginated download of all iTransformer curves/checkpoints/results."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time,requests
from kaggle.api.kaggle_api_extended import KaggleApi
p=argparse.ArgumentParser();p.add_argument('target',type=Path)
p.add_argument('--slug',default='den1shh/proteus-itransformer-vs-lag4');a=p.parse_args()
api=KaggleApi();api.authenticate();token=None;seen=set();pages=0;count=0
while True:
    for attempt in range(5):
        try:
            files,token=api.kernels_output(a.slug,str(a.target),file_pattern=r'^it-results/',page_size=200,page_token=token,quiet=True)
            break
        except requests.RequestException as error:
            if attempt==4:raise
            print('Retry interrupted output download',type(error).__name__,flush=True);time.sleep(2)
    pages+=1;count+=len(files);print('OUTPUT PAGE',pages,'selected',len(files),'total',count,flush=True)
    if not token:break
    if token in seen:raise RuntimeError('Repeated pagination token')
    seen.add(token)
root=a.target/'it-results';counts={name:len(list((root/name).glob('*_predictions.npz'))) for name in ['baseline','itransformer']}
hashes={str(f.relative_to(a.target)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.target.rglob('*') if f.is_file()}
receipt=dict(source=a.slug,downloaded_utc=datetime.now(timezone.utc).isoformat(),pages=pages,prediction_jobs=counts,files_sha256=hashes)
(a.target/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
print('PREDICTION JOBS',counts,flush=True)
assert counts=={'baseline':10,'itransformer':30},'Incomplete outputs;preserved for diagnosis'
assert len(list((root/'itransformer').rglob('curves.png')))==30
