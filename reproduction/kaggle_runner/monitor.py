"""Read Kaggle execution logs through the installed skill; retain full local logs."""
from pathlib import Path
import json,os,subprocess,sys

here=Path(__file__).resolve().parent
skill='/Users/den1shh/.codex/skills/kaggle/scripts/kaggle_skill.py'
slug='den1shh/proteus-lag4-classifier-ablation'
env=dict(os.environ,PATH=str(here/'.venv/bin')+os.pathsep+os.environ['PATH'])
for action in ['status','logs']:
    r=subprocess.run([sys.executable,skill,'cli','--','kernels',action,slug],
                     capture_output=True,text=True,env=env)
    raw=r.stdout+r.stderr
    (here/f'latest-{action}.txt').write_text(raw)
    if action=='status':print(raw.strip(),flush=True);continue
    start=raw.find('[{');end=raw.rfind(']')
    if start>=0 and end>start:
        rows=json.loads(raw[start:end+1]);log=''.join(x.get('data','') for x in rows)
        (here/'execution.log').write_text(log)
        print(log[-6000:],flush=True)
    else:print(raw[-2000:],flush=True)
