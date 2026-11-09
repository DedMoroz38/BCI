"""Persist read-only Kaggle events; training is remote and survives chat closure."""
from pathlib import Path
import argparse,json,time,requests
from kaggle.api.kaggle_api_extended import KaggleApi
p=argparse.ArgumentParser();p.add_argument('version',default='1',nargs='?');a=p.parse_args()
slug='den1shh/proteus-attention-spd-ablation/'+a.version
api=KaggleApi();api.authenticate();seen=set()
path=Path(__file__).resolve().parent/f'live-v{a.version}.jsonl'
if path.exists():
    for line in path.read_text().splitlines():seen.update(json.loads(line).get('data','').splitlines())
with path.open('a',buffering=1) as log:
    failures=0
    while True:
        try:
            for event in api.kernels_logs_stream(slug):
                log.write(json.dumps(event)+'\n')
                for line in event.get('data','').splitlines():
                    if line in seen:continue
                    seen.add(line)
                    if any(w in line for w in ['GPU','passed','ATTENTION','PACK','FEATURES','DONE','Error','Traceback','FROZEN','VERIFIED']):
                        print(f"t={event.get('time','?')} {line}",flush=True)
            break
        except (requests.exceptions.ChunkedEncodingError,requests.exceptions.ConnectionError,requests.exceptions.Timeout):
            failures+=1
            if failures>=10:raise
            time.sleep(2)
