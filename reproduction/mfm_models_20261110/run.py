"""Classical and structured models: all10 LOSO + later session, CPU only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse,traceback,time,json,warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from core import *
from models import Scaler,classifier,HOSVD,HOPLS2,WeightedPLS,StructuredLR

MATRIX_NAMES=('lag4_matched_lr50','lag4_nuclear001_50','lag4_nuclear005_50','lag4_rank8_50','lag4_rank16_50','lag4_pls8_50','lag4_hopls4_50','lag4_hopls8_50')

def tf(x):
    _,_,z=stft(x,fs=120,window='hann',nperseg=120,noverlap=60,nfft=120,boundary=None,padded=False,axis=-1)
    return np.log(np.maximum(np.abs(z[...,4:41,:])**2,1e-20)).astype('float32')

def causal_log_center(x,meta,source):
    out=np.empty_like(x)
    for ids in streams(meta):
        avg=(10*source+np.cumsum(x[ids].mean(-1),axis=0))/(10+np.arange(1,len(ids)+1))[:,None,None]
        out[ids]=x[ids]-avg[...,None]
    return out

def spectral_job(cache,out,tag,mode='spectral'):
    begin=time.monotonic();cache=Path(cache);out=Path(out);m=metadata(cache);tr,te=split(m,tag)
    mt=subset(m,tr);me=subset(m,te);y=mt['y'];w=person_weights(mt)
    x=np.load(cache/'filtered.npy',mmap_mode='r');a=np.asarray(x[tr],dtype='float64');b=np.asarray(x[te],dtype='float64')
    prior=None
    if mode in ('spectral','spectral_source'):
        prior=np.mean(a@a.transpose(0,2,1)/480,axis=0)
        a=tf(ea(a,mt,prior));b=tf(ea(b,me,prior))
    else:a=tf(a);b=tf(b)
    source=np.average(a.mean(-1),axis=0,weights=w)
    if mode in ('spectral','spectral_center'):
        a=causal_log_center(a,mt,source);b=causal_log_center(b,me,source)
    elif mode=='spectral_relative':
        # Only remove overall per-window amplitude, retain frequency structure.
        a=a-a.mean(axis=(1,2,3),keepdims=True);b=b-b.mean(axis=(1,2,3),keepdims=True)
    scale=Scaler().fit(a,w);a=scale.transform(a).astype('float32');b=scale.transform(b).astype('float32')
    folder=out/'models'/tag/mode;folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(input_cov_prior=prior,log_prior=source,scaler=scale,mode=mode),folder/'preprocess.joblib')
    plans=[dict(name=f'stft_lr_c{C}',kind='lr',C=C) for C in (.01,.1)]
    plans += [dict(name=f'stft_pls{r}',kind='pls',r=r) for r in (4,8,16)]
    plans += [dict(name=f'stft_hopls{r}_l{l[0]}{l[1]}{l[2]}',kind='hopls',r=r,ranks=l) for r,l in ((4,(4,4,2)),(8,(4,4,2)),(8,(8,6,3)))]
    plans += [dict(name='stft_tucker_'+''.join(map(str,l)),kind='tucker',ranks=l) for l in ((4,4,2),(8,6,3),(12,8,4))]
    if mode!='spectral':
        for cfg in plans:cfg['name']=cfg['name'].replace('stft_',mode+'_',1)
    failures=[]
    for cfg in plans:
        if (out/'predictions'/tag/(cfg['name']+'.json')).exists():continue
        try:
            start=time.monotonic();kind=cfg['kind'];model=None
            if kind=='lr':aa=a.reshape(len(a),-1);bb=b.reshape(len(b),-1)
            else:
                if kind=='pls':model=WeightedPLS(cfg['r'])
                elif kind=='hopls':model=HOPLS2(cfg['r'],cfg['ranks'])
                else:model=HOSVD(cfg['ranks'])
                model.fit(a,y,w);aa=model.transform(a);bb=model.transform(b)
            p,head=classifier(aa,bb,y,w,C=cfg.get('C',1.))
            train_p=head['clf'].predict_proba(head['scaler'].transform(aa))
            save_pair(out,tag,cfg['name'],me,p,config=cfg,seconds=time.monotonic()-start,features=aa.shape[1],train=score(mt,train_p))
            joblib.dump(dict(projection=model,head=head),folder/(cfg['name']+'.joblib'))
            del aa,bb,model,head
        except Exception as e:
            failures.append(dict(config=cfg,error=str(e),traceback=traceback.format_exc()));print('FAILED spectral',tag,cfg['name'],str(e),flush=True)
    write_json(out/'jobs'/f'{mode}_{tag}.json',dict(family=mode,tag=tag,status='complete',seconds=time.monotonic()-begin,failures=failures))

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
            plans=[('lag4_nuclear001_50',dict(nuclear=.01)),('lag4_nuclear005_50',dict(nuclear=.05)),
                   ('lag4_rank8_50',dict(rank=8)),('lag4_rank16_50',dict(rank=16))]
            for name,kwargs in plans:
                if name in failed:continue
                try:
                    model=StructuredLR(40,**kwargs).fit(a[keep],y[keep],w,fitted)
                    accum[name].append(model.predict_proba(b));state['heads'][name]=model
                except Exception as e:failed[name]=dict(tree=tree,error=str(e),traceback=traceback.format_exc())
            for name,r,load in [('lag4_pls8_50',8,None),('lag4_hopls4_50',4,(4,4)),('lag4_hopls8_50',8,(8,8))]:
                if name in failed:continue
                try:
                    if load is None:
                        projection=WeightedPLS(r);aa=a[keep];bb=b
                    else:
                        projection=HOPLS2(r,load);aa=unupper(a[keep],40).astype('float32');bb=unupper(b,40).astype('float32')
                    projection.fit(aa,y[keep],w)
                    pp,head=classifier(projection.transform(aa),projection.transform(bb),y[keep],w,C=1.)
                    accum[name].append(pp);state['heads'][name]=dict(projection=projection,head=head)
                except Exception as e:failed[name]=dict(tree=tree,error=str(e),traceback=traceback.format_exc())
        joblib.dump(state,folder/f'tree{tree:02}.joblib',compress=1)
        write_json(folder/'progress.json',dict(trees=tree+1,planned=trees,seconds=time.monotonic()-begin,failed=failed))
        if (tree+1)%5==0:print('TREES',tag,representation,tree+1,round(time.monotonic()-begin,1),'seconds',flush=True)
    for name,ps in accum.items():
        if len(ps)==trees:
            save_pair(out,tag,name,me,np.mean(ps,axis=0),trees=trees,representation=representation,seconds=time.monotonic()-begin)
    write_json(out/'jobs'/f'{representation}_{tag}.json',dict(family=representation,tag=tag,status='complete',seconds=time.monotonic()-begin,failed=failed))

def wrapped(job):
    kind,cache,out,tag,trees=job
    try:
        if kind.startswith('spectral'):spectral_job(cache,out,tag,kind)
        else:matrix_job(cache,out,tag,kind,trees)
        return dict(family=kind,tag=tag,status='complete')
    except Exception as e:
        r=dict(family=kind,tag=tag,status='failed',error=str(e),traceback=traceback.format_exc())
        write_json(Path(out)/'jobs'/f'{kind}_{tag}.json',r);print('JOB_FAILED',r,flush=True);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',default=str(HERE/'cache'));p.add_argument('--out',default=str(HERE/'results'))
    p.add_argument('--workers',type=int,default=2);p.add_argument('--families',nargs='+',default=['spectral','lag4','mssa8_mix25','mssa16_mix50'])
    p.add_argument('--tags',nargs='+',default=TAGS);p.add_argument('--trees',type=int,default=50);args=p.parse_args()
    start=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    from prepare import raw,mssa_band
    raw(args.cache);m=metadata(args.cache)
    for tag in args.tags:
        ref=reference(tag);save_pair(out,tag,'historical_lag4_50',subset(m,ref['data_ids']),ref['lag_raw'],source='saved existing probabilities; no parity gate')
    records=[]
    # Spectral jobs run first so complete standalone results arrive quickly.
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        for family in args.families:
            if family.startswith('mssa'):
                futures=[pool.submit(mssa_band,args.cache,b) for b in range(6)]
                for f in as_completed(futures):f.result()
            futures=[pool.submit(wrapped,(family,args.cache,args.out,tag,args.trees)) for tag in args.tags]
            for f in as_completed(futures):
                records.append(f.result());summarize(out)
                write_json(out/'progress.json',dict(seconds=time.monotonic()-start,jobs=records))
    summarize(out);write_json(out/'runtime.json',dict(seconds=time.monotonic()-start,jobs=records))
    print('ALL FINISHED',round(time.monotonic()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
