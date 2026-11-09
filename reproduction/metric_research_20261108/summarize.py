"""Summarize complete folds only; partial seed sets never masquerade as full CV."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import sha,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    records=[]
    for path in (root/'models').glob('*/*/seed*/prediction.json'):
        r=json.loads(path.read_text());assert sha(path.with_suffix('.npz'))==r['prediction_sha256']
        assert sha(path.parent/'model.pt')==r['model_sha256'];records.append(r)
    df=pd.DataFrame(records)
    if df.empty:write_json(root/'summary.json',dict(status='NO_COMPLETED_FITS'));return
    df.to_csv(root/'fold_scores.csv',index=False)
    rows=[]
    for (config,seed),g in df.groupby(['config','seed']):
        cv=g[g.tag.str.startswith('loso_')];late=g[g.tag=='later_last']
        rows.append(dict(config=config,seed=int(seed),loso_folds=len(cv),later_folds=len(late),
                         loso_mean_person_cell=float(cv.cell_mean.mean()),later_cell=float(late.cell_mean.iloc[0]) if len(late) else None,
                         complete_protocol=len(cv)==10 and len(late)==1))
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    lines=['# Wave1: nested person-held-out CNNs','',
        'Complete protocol requires10held-out people and one later-session fold covering2people.',
        'No old public-test tuning;all10public people enter the new CV.','',pd.DataFrame(rows).to_string(index=False),
        '',f'Completed network fits:{len(df)}. Partial seed sets are labeled explicitly.',
        'Next:combine saved predictions with matching local all10 Lag4 controls;rank official pooled OOF cellBA and report person-mean separately.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'summary.json',dict(fits=len(df),complete_config_seeds=sum(r['complete_protocol'] for r in rows),rows=rows))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
