"""Frozen 4-policy x2-seed x10-pair source-only study; no historical refits."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context,current_process
import numpy as np
import torch
from core import *
from novel import CONFIGS,new_net,SessionPredictor
from neural_training import fit_early,predict,atomic_torch

_PREPARED=None

def prepare(cache,tag,device):
    global _PREPARED
    if _PREPARED is not None and _PREPARED['tag']==tag:return _PREPARED
    _PREPARED=None
    if torch.cuda.is_available():torch.cuda.empty_cache()
    m=metadata(cache);X=np.load(Path(cache)/'X.npy',mmap_mode='r');tr,te=split(m,tag)
    rec=json.loads((HERE/'splits.json').read_text())['folds'][tag]
    it=np.asarray(rec['inner_train_ids']);iv=np.asarray(rec['inner_val_ids'])
    assert not set(m['subject'][tr]) & set(m['subject'][te])
    assert not set(m['subject'][it]) & set(m['subject'][iv])
    xi,xv,iprior=get_aligned(X,m,it,iv);xt,xe,prior=get_aligned(X,m,tr,te)
    p=dict(tag=tag,train=subset(m,tr),test=subset(m,te),inner_train=subset(m,it),
           inner_val=subset(m,iv),prior=prior)
    for key,value in dict(xi=xi,xv=xv,xt=xt,xe=xe).items():p[key]=torch.from_numpy(value).to(device)
    _PREPARED=p
    return p

def run_fit(args):
    cfg,seed,tag,cache,out,deadline=args
    folder=Path(out)/'models'/tag/cfg['name']/f'seed{seed}'
    try:
        if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
        torch.set_num_threads(2)
        identity=current_process()._identity
        gpu=((identity[0]-1) if identity else 0)%max(torch.cuda.device_count(),1)
        device=f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda'):torch.cuda.set_device(gpu)
        p=prepare(cache,tag,device)
        net,selection=fit_early(cfg,seed,p['xi'],p['inner_train'],folder/'selection',device,deadline,
                                 val_x=p['xv'],val_meta=p['inner_val'])
        del net
        net,refit=fit_early(cfg,seed,p['xt'],p['train'],folder/'refit',device,deadline,
                             epochs=selection['refit_epochs'],fixed_average=selection['averaging_epochs'])
        z,h=predict(net,p['xe'],p['test'],hidden=True)
        np.save(folder/'logits.npy',z);np.save(folder/'hidden.npy',h)
        atomic_torch(dict(config=cfg,state={k:v.detach().cpu() for k,v in net.state_dict().items()}),folder/'model.pt')
        np.savez_compressed(folder/'alignment.npz',cov_prior=p['prior'])
        # Actual bounded-memory B1 inference on one stream: retain timings and equivalence.
        ids=next(streams(p['test']))[:12];adapter=SessionPredictor(net)
        if device.startswith('cuda'):torch.cuda.synchronize()
        begin=time.monotonic();single=np.concatenate([adapter.predict_logits(p['xe'][i:i+1]).cpu().numpy() for i in ids])
        if device.startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.monotonic()-begin
        difference=float(np.max(np.abs(single-z[ids])))
        if not np.allclose(single,z[ids],atol=2e-4,rtol=2e-4):raise RuntimeError(f'B1-prefix discrepancy {difference}')
        record=dict(tag=tag,model=cfg['name'],seed=seed,status='complete',selection=selection,refit=refit,
                    b1_ms_per_window=1000*elapsed/len(ids),b1_prefix_max_logit_difference=difference,
                    train_people=np.unique(p['train']['subject']).tolist(),test_people=np.unique(p['test']['subject']).tolist(),
                    inner_train_people=np.unique(p['inner_train']['subject']).tolist(),inner_val_people=np.unique(p['inner_val']['subject']).tolist())
        write_json(folder/'complete.json',record)
        print('FIT DONE',tag,cfg['name'],seed,flush=True)
        return record
    except Exception as e:
        record=dict(tag=tag,model=cfg['name'],seed=seed,status='incomplete' if isinstance(e,TimeoutError) else 'failed',
                    error=str(e),traceback=traceback.format_exc())
        write_json(folder/'failure.json',record);print('FIT FAILED',record,flush=True)
        return record

def combine(cache,out):
    out=Path(out);m=metadata(cache)
    for tag in pair_tags():
        _,te=split(m,tag);me=subset(m,te)
        baseline_file=HERE/'baseline_inputs'/f'{tag}.npz'
        with np.load(baseline_file) as b:
            for key in META:assert np.array_equal(b[key],me[key]),(tag,key)
            baseline=b['proba']
        save(out,tag,'reference_full_6458_recipe',me,baseline)
        for cfg in CONFIGS:
            folders=[out/'models'/tag/cfg['name']/f'seed{s}' for s in (0,1)]
            if not all((f/'complete.json').exists() for f in folders):continue
            ps=[]
            for s,f in enumerate(folders):
                p=softmax(np.load(f/'logits.npy'),axis=1);ps.append(p)
                save_pair(out,tag,f"{cfg['name']}_seed{s}",me,p)
            mean=np.mean(ps,axis=0);save_pair(out,tag,cfg['name']+'_mean2',me,mean)
            balanced=balance(mean,me)
            for weight in (.1,.2):
                save(out,tag,f"full_add{int(weight*100)}_{cfg['name']}",me,(1-weight)*baseline+weight*balanced)
    from evaluate import summarize
    summarize([('pair8_2',out)],out/'validation')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='/kaggle/working/literature-wave-results')
    parser.add_argument('--cache',default='/tmp/literature-wave-cache')
    parser.add_argument('--minutes',type=float,default=120)
    parser.add_argument('--combine-only',action='store_true')
    a=parser.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.combine_only:combine(a.cache,out);return
    start=time.monotonic();deadline=start+60*a.minutes
    cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    if not (cache/'X.npy').exists():
        parts=[]
        for s in ('train','val','test'):
            path=next(Path('/kaggle/input').rglob(f'proteus_{s}.npz'))
            with np.load(path) as z:parts.append({k:z[k] for k in ('X',)+META})
        np.save(cache/'X.npy',np.concatenate([p['X'] for p in parts]))
        np.savez(cache/'meta.npz',**{k:np.concatenate([p[k] for p in parts]) for k in META});del parts
    tasks=[(cfg,s,t,str(cache),str(out),deadline) for cfg in CONFIGS for t in pair_tags() for s in (0,1)]
    records=[]
    with ProcessPoolExecutor(max_workers=min(2,max(1,torch.cuda.device_count())),mp_context=get_context('spawn')) as pool:
        pending={};iterator=iter(tasks)
        for _ in range(min(2,max(1,torch.cuda.device_count()))):
            task=next(iterator,None)
            if task is not None:pending[pool.submit(run_fit,task)]=task
        while pending:
            future=next(as_completed(pending));pending.pop(future);records.append(future.result())
            write_json(out/'progress.json',dict(planned=len(tasks),jobs=records,seconds=time.monotonic()-start))
            if time.monotonic()<deadline-90:
                task=next(iterator,None)
                if task is not None:pending[pool.submit(run_fit,task)]=task
    combine(cache,out)
    write_json(out/'runtime.json',dict(planned=len(tasks),jobs=records,seconds=time.monotonic()-start,
                                      complete=sum(r['status']=='complete' for r in records),
                                      failed=sum(r['status']=='failed' for r in records),
                                      unscheduled=len(tasks)-len(records)))
    print('LITERATURE WAVE FINISHED',flush=True)

if __name__=='__main__':main()
