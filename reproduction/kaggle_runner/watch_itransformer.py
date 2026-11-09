"""Read-only completion monitor; download and independently verify when done."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from kaggle.api.kaggle_api_extended import KaggleApi
HERE=Path(__file__).resolve().parent;EXP=HERE.parent/'itransformer_ablation'
p=argparse.ArgumentParser();p.add_argument('version');a=p.parse_args()
slug='den1shh/proteus-itransformer-vs-lag4/'+a.version
status_path=EXP/f'completion-v{a.version}.json';target=HERE/'downloads'/f'itransformer-v{a.version}'
api=KaggleApi();api.authenticate();previous=None
for _ in range(24*60*2):
    try:
        response=api.kernels_status(slug);state=response.status.name
    except Exception as e:
        print('STATUS RETRY',type(e).__name__,flush=True);time.sleep(30);continue
    if state!=previous:
        print('KAGGLE STATUS',state,flush=True);previous=state
    status_path.write_text(json.dumps(dict(slug=slug,state=state,checked_utc=datetime.now(timezone.utc).isoformat(),download_target=str(target)),indent=2))
    if state in ['COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','CANCELED','FAILED']:
        if state!='COMPLETE':raise RuntimeError('Remote run ended with '+state+'; inspect logs and saved outputs')
        subprocess.run([sys.executable,str(HERE/'download_itransformer.py'),str(target),'--slug',slug],check=True)
        env=dict(os.environ,IT_OUTPUT=str(target/'it-results'))
        subprocess.run([str(HERE.parent/'.venv/bin/python'),str(EXP/'analyze_itransformer.py')],env=env,check=True)
        report=target/'it-results/REPORT.md'
        (EXP/'LATEST_REPORT.md').write_text(report.read_text())
        status_path.write_text(json.dumps(dict(slug=slug,state='COMPLETE_VERIFIED',verified_utc=datetime.now(timezone.utc).isoformat(),report=str(report),download_target=str(target)),indent=2))
        print(report.read_text(),flush=True)
        break
    time.sleep(30)
else:raise TimeoutError('Completion monitor exceeded24hours')
