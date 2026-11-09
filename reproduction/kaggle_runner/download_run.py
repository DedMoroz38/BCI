"""Download all paginated saved results,using the SDK's guarded output paths."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,requests
from kaggle.api.kaggle_api_extended import KaggleApi

parser=argparse.ArgumentParser();parser.add_argument('slug');parser.add_argument('target',type=Path)
parser.add_argument('--include-source',action='store_true');args=parser.parse_args()
pattern=r'^classifier-results/'
if args.include_source:
    pattern=r'^(classifier-results/|riemann-project/(?!reproduction/(?:data|temporal_spd_research/cache)/))'
api=KaggleApi();api.authenticate();token=None;seen=set();pages=0;files=[]
while True:
    try:
        current,token=api.kernels_output(args.slug,str(args.target),file_pattern=pattern,
                                       page_size=200,page_token=token,quiet=True)
    except requests.HTTPError as error:
        raise SystemExit(f'Kaggle download failed with HTTP {error.response.status_code}')
    files.extend(current);pages+=1
    print('Downloaded page',pages,'selected files',len(current),'total',len(files),flush=True)
    if not token:break
    if token in seen:raise RuntimeError('Repeated Kaggle pagination token')
    seen.add(token)
results=args.target/'classifier-results'
names=['baseline','lda_lw','svm_linear','svm_rbf','ridge','elasticnet','mlp']
counts={name:len(list((results/name).glob('*.json'))) for name in names}
assert all(n==10 for n in counts.values()),counts
assert len(list(results.rglob('*_predictions.npz')))==70
hashes={str(p.relative_to(args.target)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.target.rglob('*') if p.is_file()}
receipt=dict(source=args.slug,downloaded_utc=datetime.now(timezone.utc).isoformat(),
             pages=pages,file_count=len(hashes),jobs=counts,files_sha256=hashes)
(args.target/'download-receipt.json').write_text(json.dumps(receipt,indent=2))
print('Verified download completeness:70 job records and70 prediction files',flush=True)
