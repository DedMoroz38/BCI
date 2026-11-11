"""Frozen mSSA + nuclear LR experiment. CPU-only, source-fit and causal prediction."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json,time,traceback,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
HERE=Path(__file__).resolve().parent
REPRO=HERE.parents[1]
CUSTOM=REPRO/'custom_validation_20261110'
sys.path.insert(0,str(REPRO/'mfm_models_20261110'))
sys.path.insert(0,str(CUSTOM))
import numpy as np
import pandas as pd
import joblib
from core import META,subset,streams,spectral,upper,unupper,metadata,write_json,balance,score
from models import StructuredLR
from evaluate import observations
CACHE=REPRO/'mfm_models_20261110/cache'
BASELINE=CUSTOM/'results_pair_cpu'
OUT=HERE/'results'
CONFIGS={'mssa8_mix25_nuclear001_50':.01,'mssa8_mix25_nuclear005_50':.05}
COMPARATORS=('mssa8_mix25_lr50','lag4_matched_lr50','lag4_nuclear001_50','lag4_nuclear005_50')


def causal_features(c,m,source,prior=5):
    """Transform using fixed source prior and only the current stream prefix."""
    c=np.asarray(c,dtype='float64');logs=spectral(c,'log')
    out=np.empty((len(c),c.shape[-1]*(c.shape[-1]+1)//2))
    for ids in streams(m):
        refs=(prior*source+np.cumsum(logs[ids],axis=0))/(prior+np.arange(1,len(ids)+1))[:,None,None]
        whiten=spectral(-.5*refs,'exp')
        out[ids]=upper(spectral(whiten@c[ids]@whiten,'log'))
    return out


def transform_pair(source_c,target_c,source_m,target_m,prior,scaler):
    return (scaler.transform(causal_features(source_c,source_m,prior)),
            scaler.transform(causal_features(target_c,target_m,prior)))


def save_prediction(tag,name,m,p,ids):
    if not np.isfinite(p).all():raise FloatingPointError('nonfinite probabilities')
    p=np.maximum(p,1e-15);p/=p.sum(1,keepdims=True)
    folder=OUT/'predictions'/tag;folder.mkdir(parents=True,exist_ok=True)
    with (folder/(name+'.npz.tmp')).open('wb') as handle:
        np.savez_compressed(handle,**m,proba=p,data_ids=ids)
    (folder/(name+'.npz.tmp')).replace(folder/(name+'.npz'))
    write_json(folder/(name+'.json'),dict(tag=tag,name=name,trees=50,**score(m,p)))


def fold_job(tag,record):
    begin=time.monotonic();failed={};done={name:0 for name in CONFIGS}
    try:
        m=metadata(CACHE);tr=np.asarray(record['train_ids']);te=np.asarray(record['test_ids'])
        sm=subset(m,tr);tm=subset(m,te)
        if set(sm['subject'])&set(tm['subject']):raise ValueError('source/target people overlap')
        if len(m['y'])!=2655:raise ValueError('metadata must include all 2655 windows')
        folder=OUT/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'split_ids.npz',train_ids=tr,test_ids=te)
        sums={name:np.zeros((len(te),3)) for name in CONFIGS}
        for tree in range(50):
            source_path=BASELINE/'models'/tag/'mssa8_mix25'/f'tree{tree:02}.joblib'
            state=joblib.load(source_path)
            ch=state['physical_channels'];rows=(2*ch[:,None]+np.array([0,1])).reshape(-1)
            cov=np.load(CACHE/f"mssa8_mix25_band{state['band']}.npy",mmap_mode='r')
            # The only fitted geometry is the saved source-only state. No target is passed to fit.
            a=state['scaler'].transform(causal_features(cov[np.ix_(tr,rows,rows)],sm,state['log_prior']))
            boot=state['bootstrap'];keep=boot>0;y=sm['y'];counts=np.bincount(y[keep],minlength=3)
            w=boot[keep]*len(y[keep])/(3*counts[y[keep]])
            initial=state['heads']['mssa8_mix25_lr50']
            init=dict(coef=initial.coef_,intercept=initial.intercept_,weight_sum=float(w.sum()))
            heads={}
            for name,tau in CONFIGS.items():
                if name in failed:continue
                try:
                    heads[name]=StructuredLR(40,nuclear=tau,iterations=60).fit(a[keep],y[keep],w,init,C=.01)
                    if not np.isfinite(heads[name].coef).all():raise FloatingPointError('nonfinite classifier weights')
                except Exception as exc:
                    failed[name]=dict(tree=tree,error=str(exc),traceback=traceback.format_exc())
                    print('HEAD FAILURE',tag,name,failed[name],flush=True)
            b=state['scaler'].transform(causal_features(cov[np.ix_(te,rows,rows)],tm,state['log_prior']))
            for name,head in heads.items():
                p=head.predict_proba(b)
                if not np.isfinite(p).all():raise FloatingPointError(f'nonfinite probabilities {name}')
                sums[name]+=p;done[name]+=1
            saved=dict(band=state['band'],physical_channels=ch,log_prior=state['log_prior'],
                       scaler=state['scaler'],bootstrap=boot,heads=heads,source_model=str(source_path),
                       initial_weight_sum=float(w.sum()),train_ids=tr,test_ids=te)
            joblib.dump(saved,folder/f'tree{tree:02}.joblib',compress=1)
            write_json(folder/'progress.json',dict(trees=tree+1,planned=50,seconds=time.monotonic()-begin,failed=failed,completed_heads=done))
            if (tree+1)%5==0:print('TREES',tag,tree+1,round(time.monotonic()-begin,1),flush=True)
        for name in CONFIGS:
            if done[name]==50 and name not in failed:
                p=sums[name]/50
                save_prediction(tag,name,tm,p,te)
                save_prediction(tag,name+'__balance',tm,balance(p,tm),te)
        result=dict(fold=tag,status='complete' if not failed else 'partial',failed=failed,completed_heads=done,seconds=time.monotonic()-begin)
    except Exception as exc:
        result=dict(fold=tag,status='failed',failed=failed,error=str(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-begin)
        print('FOLD FAILURE',result,flush=True)
    write_json(OUT/'jobs'/f'{tag}.json',result)
    return result


def summarize():
    expected=json.loads((CUSTOM/'splits.json').read_text())['folds'];detail=[]
    for root,names in [(BASELINE,COMPARATORS),(OUT,tuple(CONFIGS))]:
        for tag in expected:
            for raw in names:
                for name in (raw,raw+'__balance'):
                    path=root/'predictions'/tag/(name+'.npz')
                    if not path.exists():continue
                    with np.load(path) as data:
                        detail+=observations('pair8_2',tag,name,{k:data[k] for k in META},data['proba'])
    df=pd.DataFrame(detail);df.to_csv(HERE/'per_person_repeat.csv',index=False)
    rows=[];people=[];incomplete=[]
    for name,d in df.groupby('model'):
        counts=d.groupby('person').size()
        if set(d.fold)!=set(expected) or len(counts)!=10 or not counts.eq(2).all():
            incomplete.append(dict(model=name,folds=d.fold.nunique(),people=len(counts)));continue
        agg=d.groupby('person')[['first_ba','first_nll','all_ba','first20_ba','middle21_60_ba','tail61_ba']].mean()
        for person,row in agg.iterrows():people.append(dict(model=name,person=person,**row.to_dict()))
        rows.append(dict(model=name,first_session_ba=float(agg.first_ba.mean()),worst3=float(np.sort(agg.first_ba)[:3].mean()),
                         first_nll=float(agg.first_nll.mean()),all_session_cell_ba=float(np.average(d.all_ba,weights=d.all_cells)),
                         first20_ba=float(agg.first20_ba.mean()),middle21_60_ba=float(agg.middle21_60_ba.mean()),
                         tail61_ba=float(agg.tail61_ba.mean()),mean_repeat_sd=float(d.groupby('person').first_ba.std(ddof=0).mean()),folds=10))
    comparison=pd.DataFrame(rows).sort_values('first_session_ba',ascending=False)
    comparison.to_csv(HERE/'comparison.csv',index=False)
    pd.DataFrame(people).to_csv(HERE/'per_person.csv',index=False)
    pd.DataFrame(incomplete,columns=['model','folds','people']).to_csv(HERE/'incomplete.csv',index=False)
    return comparison


def main():
    begin=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    folds=json.loads((CUSTOM/'splits.json').read_text())['folds'];records=[]
    write_json(HERE/'STATUS.json',dict(status='running',planned_folds=10,workers=2,blas_threads=1))
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(fold_job,tag,record) for tag,record in folds.items()]
        for future in as_completed(futures):
            records.append(future.result())
            write_json(HERE/'STATUS.json',dict(status='running',finished_folds=len(records),planned_folds=10,records=records,seconds=time.monotonic()-begin))
    table=summarize();seconds=time.monotonic()-begin
    status=dict(status='complete' if all(r['status']=='complete' for r in records) else 'partial',finished_folds=len(records),
                planned_folds=10,failed_folds=sum(r['status']=='failed' for r in records),
                failed_head_configs=sum(len(r.get('failed',{})) for r in records),records=records,seconds=seconds,workers=2,blas_threads=1)
    write_json(HERE/'STATUS.json',status)
    report='# CPU synergy results\n\nPrimary: first-session cell BA; mean repeat metric per person, then equal mean over 10 people.\n\n'
    report+=table.to_string(index=False)+'\n\n'
    report+=f"Runtime: {seconds:.1f} s; failed folds: {status['failed_folds']}; failed fold/head configurations: {status['failed_head_configs']}.\n"
    report+='\nExploratory comparison on reused validation participants; no independent test claim. Protocol frozen in PROTOCOL.md.\n'
    (HERE/'REPORT.md').write_text(report)
    print(table.to_string(index=False),flush=True);print('COMPLETE',seconds,flush=True)

if __name__=='__main__':main()
