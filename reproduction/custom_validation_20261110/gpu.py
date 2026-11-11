"""Retrain existing top ensemble components on8people;60fits,no old weights."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,time,json,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context,current_process
import numpy as np
import torch
from core import *
from neural_training import fit_plain,fit_point,fit_early,predict,atomic_torch
from tta_core import causal_geometry,features

CONFIGS=[dict(name='swad_plain',recipe='swad',family='plain'),dict(name='point_meanstd',recipe='point',family='context'),dict(name='early_and_filter',recipe='early',family='context',prior=10,scope='filter',early=True)]
_PREPARED=None

def prepare(cache,tag,device):
    global _PREPARED
    if _PREPARED is not None and _PREPARED['tag']==tag:return _PREPARED
    _PREPARED=None
    if torch.cuda.is_available():torch.cuda.empty_cache()
    m=metadata(cache);X=np.load(Path(cache)/'X.npy',mmap_mode='r');tr,te=split(m,tag)
    rec=json.loads((HERE/'splits.json').read_text())['folds'][tag]
    it=np.asarray(rec['inner_train_ids']);iv=np.asarray(rec['inner_val_ids'])
    xi,xv,iprior=get_aligned(X,m,it,iv);xt,xe,prior=get_aligned(X,m,tr,te)
    _PREPARED=dict(tag=tag,train=subset(m,tr),test=subset(m,te),inner_train=subset(m,it),inner_val=subset(m,iv),prior=prior)
    for key,value in dict(xi=xi,xv=xv,xt=xt,xe=xe).items():_PREPARED[key]=torch.from_numpy(value).to(device)
    return _PREPARED

def job(args):
    cfg,seed,tag,cache,out,deadline=args;folder=Path(out)/'models'/tag/cfg['name']/f'seed{seed}'
    try:
        if (folder/'complete.json').exists():return dict(tag=tag,model=cfg['name'],seed=seed,status='complete')
        torch.set_num_threads(2);identity=current_process()._identity;gpu=(identity[0]-1)%max(torch.cuda.device_count(),1)
        device=f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda'):torch.cuda.set_device(gpu)
        p=prepare(cache,tag,device);fit={'swad':fit_plain,'point':fit_point,'early':fit_early}[cfg['recipe']]
        net,selection=fit(cfg,seed,p['xi'],p['inner_train'],folder/'selection',device,deadline,val_x=p['xv'],val_meta=p['inner_val'])
        del net
        net,refit=fit(cfg,seed,p['xt'],p['train'],folder/'refit',device,deadline,epochs=selection['refit_epochs'],fixed_average=selection['averaging_epochs'])
        z=predict(net,p['xe'],p['test']);np.save(folder/'logits.npy',z)
        atomic_torch(dict(config=cfg,state={k:v.detach().cpu() for k,v in net.state_dict().items()}),folder/'model.pt')
        np.savez_compressed(folder/'alignment.npz',cov_prior=p['prior'])
        r=dict(tag=tag,model=cfg['name'],seed=seed,status='complete',selection=selection,refit=refit,
               train_people=np.unique(p['train']['subject']).tolist(),test_people=np.unique(p['test']['subject']).tolist(),
               inner_train_people=np.unique(p['inner_train']['subject']).tolist(),inner_val_people=np.unique(p['inner_val']['subject']).tolist())
        write_json(folder/'complete.json',r);print('FIT DONE',tag,cfg['name'],seed,flush=True);return r
    except Exception as e:
        r=dict(tag=tag,model=cfg['name'],seed=seed,status='failed',error=str(e),traceback=traceback.format_exc())
        write_json(folder/'failure.json',r);print('FIT FAILED',r,flush=True);return r

def combine(cache,out):
    cache=Path(cache);out=Path(out);m=metadata(cache)
    logs=[np.load(cache/f'laglog_{i}.npy',mmap_mode='r') for i in range(5)]
    for tag in pair_tags():
        tr,te=split(m,tag);me=subset(m,te)
        with np.load(HERE/'classical_inputs'/f'{tag}.npz') as z:lag=z['lag4_matched_lr50'].copy()
        components={};zs={}
        for cfg in CONFIGS:
            files=[out/'models'/tag/cfg['name']/f'seed{s}/logits.npy' for s in (0,1)]
            if not all(f.exists() for f in files):continue
            zz=[np.load(f) for f in files];zs[cfg['recipe']]=zz
            pp=np.mean([softmax(z,axis=1) for z in zz],axis=0);components[cfg['recipe']]=pp
            save_pair(out,tag,cfg['name']+'_mean2',me,pp)
        if len(components)!=3:continue
        mixed=.7*lag+.3*np.mean([softmax(z/4,axis=1) for z in zs['swad']],axis=0)
        q0=np.empty_like(mixed);source=np.stack([np.mean(a[tr],axis=0,dtype='float64') for a in logs])
        target=np.stack([a[te] for a in logs],axis=1).astype('float64')
        cfg=dict(tau=.5,beta=1.,k=20,iters=5,shrink=.5,warm=20,ramp=0,strength=.5)
        for ids in streams(me):
            f=features(target[ids],source,'lag_running',np.full((len(ids),3),1/3))
            q0[ids]=causal_geometry((f@f.T,),mixed[None,ids],[cfg])[0,0]
        q1=.8*q0+.2*balance(components['point'],me)
        early=balance(components['early'],me);q2=.7*q1+.3*early
        p0,p1=[softmax(z,axis=1) for z in zs['early']];mean=(p0+p1)/2
        js=.5*np.sum(p0*np.log(np.maximum(p0,1e-15)/np.maximum(mean,1e-15))+p1*np.log(np.maximum(p1,1e-15)/np.maximum(mean,1e-15)),axis=1)
        a=.1+.2*np.exp(-js/.05);gate=(1-a[:,None])*q1+a[:,None]*early
        for name,p in [('full_6272_recipe',q0),('full_6323_recipe',q1),('full_6458_fixed30_recipe',q2),('full_6451_gate_recipe',gate)]:save(out,tag,name,me,p)
    from evaluate import summarize
    summarize([('pair8_2',HERE/'classical_results'),('pair8_2',out)],out/'validation')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='/kaggle/working/custom-validation-results')
    parser.add_argument('--cache',default='/tmp/custom-validation-cache');parser.add_argument('--minutes',type=float,default=180);a=parser.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+60*a.minutes
    for tag in pair_tags():
        with np.load(HERE/'classical_inputs'/f'{tag}.npz') as z:
            mm={k:z[k] for k in META}
            for name in ['lag4_matched_lr50','lag4_nuclear001_50','lag4_nuclear005_50','mssa8_mix25_lr50']:
                save_pair(HERE/'classical_results',tag,name,mm,z[name])
    if not (cache/'X.npy').exists():
        parts=[]
        for s in ('train','val','test'):
            path=next(Path('/kaggle/input').rglob(f'proteus_{s}.npz'))
            with np.load(path) as z:parts.append({k:z[k] for k in ('X',)+META})
        for k in ('X',):np.save(cache/f'{k}.npy',np.concatenate([p[k] for p in parts]))
        np.savez(cache/'meta.npz',**{k:np.concatenate([p[k] for p in parts]) for k in META});del parts
    from window_cache import prep_job
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:list(pool.map(prep_job,[(str(cache),'lag',b) for b in range(5)]))
    tasks=[(cfg,s,tag,str(cache),str(out),deadline) for tag in pair_tags() for cfg in CONFIGS for s in (0,1)]
    records=[]
    with ProcessPoolExecutor(max_workers=min(2,max(1,torch.cuda.device_count())),mp_context=get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):
            records.append(f.result());write_json(out/'progress.json',dict(planned=len(tasks),jobs=records,seconds=time.monotonic()-start))
    combine(cache,out);write_json(out/'runtime.json',dict(jobs=records,seconds=time.monotonic()-start))
    print('GPU VALIDATION COMPLETE',flush=True)

if __name__=='__main__':main()
