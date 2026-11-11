"""Freeze metadata-only splits and archive the exact reused algorithms."""
from pathlib import Path
import ast,json,shutil
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def extract(path,names):
    text=path.read_text();tree=ast.parse(text)
    return '\n\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)

def main():
    old=ROOT/'mfm_models_20261110'
    shutil.copyfile(old/'core.py',HERE/'base.py');shutil.copyfile(old/'models.py',HERE/'models.py')
    shutil.copyfile(ROOT/'transfer_deep_dive_20261110/nets.py',HERE/'transfer_nets.py')
    shutil.copyfile(ROOT/'transfer_deep_dive_20261110/averaging.py',HERE/'averaging.py')
    shutil.copyfile(ROOT/'transduction_research_20261109/core.py',HERE/'tta_core.py')
    shutil.copyfile(ROOT/'winner_research_20261109/SWAD_LICENSE',HERE/'SWAD_LICENSE')
    header='''"""Same local Lag4 geometry/heads, now fit to8 source participants."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,time,traceback,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from collections import Counter
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from core import *
from models import StructuredLR
MATRIX_NAMES=('lag4_matched_lr50','lag4_nuclear001_50','lag4_nuclear005_50')
'''
    body=extract(old/'run.py',{'draw_specs','matrix_job'})
    body=body.replace(",\n                   ('lag4_rank8_50',dict(rank=8)),('lag4_rank16_50',dict(rank=16))",'')
    i=body.index("            for name,r,load in ");j=body.index('        joblib.dump(state',i);body=body[:i]+body[j:]
    footer='''
def job(args):
    cache,out,tag,kind=args
    try:
        matrix_job(cache,out,tag,kind)
        return dict(fold=tag,kind=kind,status='complete')
    except Exception as e:
        r=dict(fold=tag,kind=kind,status='failed',error=str(e),traceback=traceback.format_exc())
        write_json(Path(out)/'failures'/f'{kind}_{tag}.json',r);print(r,flush=True);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default=str(ROOT/'mfm_models_20261110/cache'))
    p.add_argument('--out',default=str(HERE/'results_pair_cpu'));p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();start=time.monotonic();records=[]
    jobs=[(a.cache,a.out,t,k) for t in pair_tags() for k in ('lag4','mssa8_mix25')]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,j) for j in jobs]):
            records.append(f.result());write_json(Path(a.out)/'progress.json',dict(jobs=records,planned=len(jobs),seconds=time.monotonic()-start))
            from evaluate import summarize
            summarize([('pair8_2',Path(a.out))],Path(a.out)/'validation')
    write_json(Path(a.out)/'runtime.json',dict(jobs=records,seconds=time.monotonic()-start))
    print('CPU COMPLETE',len(records),round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
'''
    (HERE/'classical.py').write_text(header+body+footer)
    with np.load(old/'cache/meta.npz') as z:meta={k:z[k] for k in ('y','subject','session','context','run','onset')}
    people=np.unique(meta['subject']);order=np.random.default_rng(20261110).permutation(people)
    folds={}
    for i in range(len(order)):
        held=[str(order[i]),str(order[(i+1)%len(order)])]
        te=np.flatnonzero(np.isin(meta['subject'],held));tr=np.flatnonzero(~np.isin(meta['subject'],held))
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=33).split(np.zeros(len(tr)),groups=meta['subject'][tr]))
        tag=f'pair{i:02}_'+ '_'.join(p.split('/')[-1] for p in held)
        folds[tag]=dict(train_people=np.unique(meta['subject'][tr]).tolist(),test_people=held,
                        train_ids=tr.tolist(),test_ids=te.tolist(),inner_train_ids=tr[it].tolist(),inner_val_ids=tr[iv].tolist())
    result=dict(seed=20261110,order=order.tolist(),folds=folds,first_session='1',stream_bins=[[1,20],[21,60],[61,None]],
                primary='mean over repeats within each held-out person,then equal mean over10people; first session cellBA')
    (HERE/'splits.json').write_text(json.dumps(result,indent=2))
    print('Frozen pairs:',[f['test_people'] for f in folds.values()])
    print('No phenotype/model-score information used in pair schedule.')

if __name__=='__main__':main()
