"""Read-only remote status/log tracking; independent of the cancelled CPU run."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='2');p.add_argument('--slug',default='den1shh/proteus-author-ensemble-vs-lag4');p.add_argument('--logs',action='store_true');a=p.parse_args()
    slug=a.slug+'/'+a.version
    label='recovery' if 'recovery' in a.slug else 'gpu'
    api=KaggleApi();api.authenticate()
    if a.logs:
        failures=0
        with (HERE/f'live-{label}-v{a.version}.jsonl').open('a',buffering=1) as log:
            while failures<10:
                try:
                    for event in api.kernels_logs_stream(slug):
                        log.write(json.dumps(event)+'\n')
                        for line in event.get('data','').splitlines():
                            if any(w in line for w in ('GPU','CUDA','passed','START','DONE','FINISHED','Error','Traceback','FROZEN')):
                                print(event.get('time'),line,flush=True)
                    return
                except Exception as e:
                    failures+=1;print('LOG RETRY',type(e).__name__,flush=True);time.sleep(2)
        return
    previous=None
    for _ in range(2880):
        try:
            response=api.kernels_status(slug);state=response.status.name
            value=dict(slug=slug,state=state,checked_utc=datetime.now(timezone.utc).isoformat(),
                       failure_message=response.failure_message,url='https://www.kaggle.com/code/'+a.slug)
            (HERE/f'completion-{label}-v{a.version}.json').write_text(json.dumps(value,indent=2))
            if state!=previous:print('KAGGLE',state,flush=True);previous=state
            if state in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELED','FAILED'):return
        except Exception as e:print('STATUS RETRY',type(e).__name__,flush=True)
        time.sleep(30)

if __name__=='__main__':main()
