"""Persist live read-only Kaggle events and print concise progress with run time."""
from pathlib import Path
import json,sys,time,requests
from kaggle.api.kaggle_api_extended import KaggleApi

version=sys.argv[1] if len(sys.argv)>1 else '4'
slug='den1shh/proteus-lag4-classifier-ablation/'+version
here=Path(__file__).resolve().parent
api=KaggleApi();api.authenticate();seen=set();path=here/f'live-v{version}.jsonl'
if path.exists():
    for row in path.read_text().splitlines():
        seen.update(json.loads(row).get('data','').splitlines())
with path.open('a',buffering=1) as log:
    failures=0
    while True:
        try:
            for event in api.kernels_logs_stream(slug):
                log.write(json.dumps(event)+'\n')
                for line in event.get('data','').splitlines():
                    if line in seen:continue
                    seen.add(line);failures=0
                    if any(word in line for word in ['FEATURES','CLASSIFIER','DONE MODEL','FROZEN SELECTION',
                            'VERIFIED ALL','EXPERIMENT COMPLETE','passed,','Error','Traceback']):
                        print(f"t={event.get('time','?')} {line}",flush=True)
            break
        except (requests.exceptions.ChunkedEncodingError,requests.exceptions.ConnectionError,requests.exceptions.Timeout):
            failures+=1
            if failures>=5:raise
            print('Log connection interrupted; reconnecting to the same running job',flush=True)
            time.sleep(1)
