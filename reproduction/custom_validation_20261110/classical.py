"""Same local Lag4 geometry/heads, now fit to8 source participants."""
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
def draw_specs(m,trees=50):
    rng=np.random.default_rng(0);people=np.unique(m['subject'])
    for _ in range(trees):
        band=int(rng.integers(6));ch=np.sort(rng.choice(41,20,replace=False));counts=Counter(rng.choice(people,len(people)))
        yield band,ch,np.array([counts[s] for s in m['subject']],dtype='float64')

def matrix_job(cache,out,tag,representation='lag4',trees=50):
    begin=time.monotonic();cache=Path(cache);out=Path(out);m=metadata(cache);tr,te=split(m,tag)
    mt=subset(m,tr);me=subset(m,te);y=mt['y'];old_ids=np.load(cache/'old_ids.npy')
    names=MATRIX_NAMES if representation=='lag4' else (representation+f'_lr{trees}',)
    if all((out/'predictions'/tag/(n+'.json')).exists() for n in names):return
    accum={n:[] for n in names};failed={};folder=out/'models'/tag/representation;folder.mkdir(parents=True,exist_ok=True)
    for tree,(band,ch,boot) in enumerate(draw_specs(mt,trees)):
        keep=boot>0;counts=np.bincount(y[keep],minlength=3)
        w=boot[keep]*len(y[keep])/(3*counts[y[keep]])
        rows=(2*ch[:,None]+np.array([0,1])).reshape(-1)
        if representation=='lag4':
            big=np.load(ROOT/'temporal_spd_research/cache'/f'lag4_oas_band{band}.npy',mmap_mode='r')
            global_ids=old_ids
        else:
            big=np.load(cache/f'{representation}_band{band}.npy',mmap_mode='r');global_ids=np.arange(len(m['y']))
        # Slice before eigendecomposition: preserve the existing forest geometry.
        ids=np.r_[tr,te];C=np.asarray(big[np.ix_(global_ids[ids],rows,rows)])
        a,b,prior=tangent_pair(C,np.arange(len(tr)),np.arange(len(tr),len(ids)),subset(m,ids),return_source=True)
        scaler=StandardScaler().fit(a);a=scaler.transform(a);b=scaler.transform(b)
        base=LogisticRegression(C=.01,max_iter=400,solver='lbfgs').fit(a[keep],y[keep],sample_weight=w)
        fitted=dict(coef=base.coef_,intercept=base.intercept_,weight_sum=float(w.sum()))
        state=dict(band=band,physical_channels=ch,log_prior=prior,scaler=scaler,bootstrap=boot,heads={names[0]:base})
        accum[names[0]].append(base.predict_proba(b))
        if representation=='lag4':
            plans=[('lag4_nuclear001_50',dict(nuclear=.01)),('lag4_nuclear005_50',dict(nuclear=.05))]
            for name,kwargs in plans:
                if name in failed:continue
                try:
                    model=StructuredLR(40,**kwargs).fit(a[keep],y[keep],w,fitted)
                    accum[name].append(model.predict_proba(b));state['heads'][name]=model
                except Exception as e:failed[name]=dict(tree=tree,error=str(e),traceback=traceback.format_exc())
        joblib.dump(state,folder/f'tree{tree:02}.joblib',compress=1)
        write_json(folder/'progress.json',dict(trees=tree+1,planned=trees,seconds=time.monotonic()-begin,failed=failed))
        if (tree+1)%5==0:print('TREES',tag,representation,tree+1,round(time.monotonic()-begin,1),'seconds',flush=True)
    for name,ps in accum.items():
        if len(ps)==trees:
            save_pair(out,tag,name,me,np.mean(ps,axis=0),trees=trees,representation=representation,seconds=time.monotonic()-begin)
    write_json(out/'jobs'/f'{representation}_{tag}.json',dict(family=representation,tag=tag,status='complete',seconds=time.monotonic()-begin,failed=failed))
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
