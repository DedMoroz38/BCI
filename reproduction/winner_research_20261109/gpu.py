"""Two GPU workers; actual causal models, inner-NLL selection, no controls."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,csv,time,traceback,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context,current_process
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit
from common import *
from nets import CONFIGS,new_net
from averaging import StateAverage,LossValley,mean_states,clone_state

_DATA=None;_PREPARED=None

def atomic_torch(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def prepare(tag,family,cache,device):
    global _DATA,_PREPARED
    if _DATA is None:
        with np.load(Path(cache)/'meta.npz') as z:meta={k:z[k] for k in META}
        _DATA=(np.load(Path(cache)/'X.npy',mmap_mode='r'),meta)
    if _PREPARED is not None and _PREPARED['key']==(tag,family):return _PREPARED
    _PREPARED=None
    if torch.cuda.is_available():torch.cuda.empty_cache()
    X,meta=_DATA;tr,te=train_test(meta,tag)
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=33).split(np.zeros(len(tr)),groups=meta['subject'][tr]))
    it,iv=tr[a],tr[b]
    if family=='context':
        xi,xv,iprior=get_aligned(X,meta,it,iv);xt,xe,prior=get_aligned(X,meta,tr,te)
    else:
        xi=np.ascontiguousarray(filtering(X[it],(4,36)),dtype='float32');xv=np.ascontiguousarray(filtering(X[iv],(4,36)),dtype='float32')
        xt=np.ascontiguousarray(filtering(X[tr],(4,36)),dtype='float32');xe=np.ascontiguousarray(filtering(X[te],(4,36)),dtype='float32');iprior=prior=np.zeros((41,41))
    values=dict(key=(tag,family),train=subset(meta,tr),test=subset(meta,te),inner_train=subset(meta,it),inner_val=subset(meta,iv),
                prior=prior,inner_prior=iprior,tr=tr,te=te)
    values.update({k:torch.from_numpy(v).to(device) for k,v in dict(xi=xi,xv=xv,xt=xt,xe=xe).items()})
    _PREPARED=values;return values

@torch.no_grad()
def predict(net,x,meta,hidden=False):
    net.eval();z=np.empty((len(x),3));latent=None
    for ids in streams(meta):
        idx=torch.as_tensor(ids,device=x.device);h=net.latent(x[idx]);v=net.out(h)
        z[ids]=v.detach().cpu().numpy()
        if hidden:
            if latent is None:latent=np.empty((len(x),h.shape[1]),dtype='float32')
            latent[ids]=h.detach().cpu().numpy()
    return (z,latent) if hidden else z

def cell_weights(meta):
    y=meta['y'];keys=np.array(['|'.join(map(str,r)) for r in zip(meta['subject'],meta['session'],meta['context'])]);out=np.empty(len(y))
    for key in np.unique(keys):
        ids=np.flatnonzero(keys==key);counts=np.bincount(y[ids],minlength=3)
        out[ids]=len(y)/(len(np.unique(keys))*3*np.maximum(counts[y[ids]],1))
    return out

def metrics(net,x,meta):
    z=predict(net,x,meta);p=softmax(z,axis=1);r=score(meta,p)
    r['cell_nll']=float(np.mean(-np.log(np.maximum(p[np.arange(len(p)),meta['y']],1e-15))*cell_weights(meta)))
    return r

def plot(folder,rows,selected):
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        for ax,metric,label in ((axes[0],'nll','Cell-balanced NLL'),(axes[1],'ba','Cell BA')):
            for part in ('train','val'):
                values=[r for r in rows if r.get(part+'_'+metric) is not None]
                if values:ax.plot([r['epoch'] for r in values],[r[part+'_'+metric] for r in values],label=part)
            ax.axvline(selected,ls='--',color='gray');ax.set_xlabel('Epoch');ax.set_title(label);ax.grid(alpha=.2)
        axes[0].legend();fig.tight_layout();fig.savefig(Path(folder)/'curves.png',dpi=140);plt.close(fig)
    except Exception as e:print('PLOT_FAILED',str(e),flush=True)

def fit(cfg,seed,x,meta,folder,device,deadline,val_x=None,val_meta=None,epochs=30,fixed_average=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);selecting=val_meta is not None
    net=new_net(cfg,seed,device);groups=list(streams(meta));y=torch.as_tensor(meta['y'],dtype=torch.long)
    weights=torch.as_tensor(class_weights(meta['y']),dtype=torch.float32)
    if cfg['family']=='tsm':
        import geoopt
        manifold=[net.bimap.W];other=[p for p in net.parameters() if p is not net.bimap.W]
        opt=geoopt.optim.RiemannianAdam([dict(params=other,weight_decay=1e-4),dict(params=manifold,weight_decay=0)],lr=.001,stabilize=10)
    else:opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100)
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;bad=0;best_state=None;history=[];snapshots={};valley=LossValley(3,6,.1)
    begin=time.monotonic()
    for epoch in range(1,epochs+1):
        if time.monotonic()>deadline:raise TimeoutError('GPU study wall budget reached; prior completed fits retained')
        net.refresh_source(x,groups);net.train()
        if hasattr(net,'norm'):net.norm.eval()
        segment=StateAverage();loss_sum=0.;seen=0
        for g in rng.permutation(len(groups)):
            ids=groups[g];idx=torch.as_tensor(ids,device=x.device)
            z=net(x[idx]);yy=y[ids].to(z.device);ww=weights[ids].to(z.device,dtype=z.dtype)
            loss=(F.cross_entropy(z,yy,reduction='none')*ww).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            if cfg['family']=='context':segment.update(net.state_dict())
            loss_sum+=float(loss.detach())*len(ids);seen+=len(ids)
        scheduler.step()
        if cfg['family']=='context':
            snapshots[epoch]=segment.result();monitored=new_net(cfg,seed,device);monitored.load_state_dict(snapshots[epoch]);monitored.refresh_source(x,groups)
        else:monitored=net;monitored.refresh_source(x,groups)
        validation=metrics(monitored,val_x,val_meta) if selecting else None
        train_eval=metrics(monitored,x,meta) if selecting and (epoch==1 or epoch%5==0 or epoch==epochs) else None
        if selecting:
            value=validation['cell_nll']
            if value<best-1e-4:best=value;best_epoch=epoch;bad=0;best_state=clone_state(monitored.state_dict());atomic_torch(best_state,folder/'best.pt')
            else:bad+=1
            if cfg['family']=='context':valley.update(epoch,value)
        row=dict(epoch=epoch,optimization_ce=loss_sum/seen,train_nll=train_eval['cell_nll'] if train_eval else None,
                 train_ba=train_eval['cell'] if train_eval else None,val_nll=validation['cell_nll'] if validation else None,val_ba=validation['cell'] if validation else None)
        history.append(row)
        with (folder/'curves.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        print('EPOCH',cfg['name'],seed,folder.parent.parent.name,epoch,'loss',round(loss_sum/seen,4),
              'valNLL',round(validation['cell_nll'],4) if validation else '-', 'seconds',round(time.monotonic()-begin,1),flush=True)
        if cfg['family']=='context':del monitored
        if selecting and bad>=6:break
    if cfg['family']=='context':
        selected=valley.final_epochs() if selecting else list(fixed_average or [epochs])
        net.load_state_dict(mean_states([snapshots[e] for e in selected]))
    else:
        selected=[best_epoch] if selecting else [epochs]
        if selecting:net.load_state_dict(best_state)
    net.refresh_source(x,groups);net.eval();atomic_torch(clone_state(net.state_dict()),folder/'final.pt')
    report=dict(best_epoch=best_epoch if selecting else epochs,averaging_epochs=selected,refit_epochs=max(selected),
                seconds=time.monotonic()-begin,parameters=sum(p.numel() for p in net.parameters()),config=cfg,seed=seed)
    write_json(folder/'fit.json',report)
    if selecting:plot(folder,history,best_epoch)
    return net,report

def worker(args):
    cfg,seed,tag,cache,out,deadline=args;out=Path(out);folder=out/'models'/tag/cfg['name']/f'seed{seed}'
    if time.monotonic()>deadline:return dict(tag=tag,config=cfg['name'],seed=seed,status='budget_pending')
    try:
        torch.set_num_threads(2);identity=current_process()._identity;gpu=((identity[0]-1) if identity else 0)%max(torch.cuda.device_count(),1)
        device=f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():torch.cuda.set_device(gpu)
        prepared=prepare(tag,cfg['family'],cache,device)
        net,selection=fit(cfg,seed,prepared['xi'],prepared['inner_train'],folder/'selection',device,deadline,
                          val_x=prepared['xv'],val_meta=prepared['inner_val'])
        del net
        net,refit=fit(cfg,seed,prepared['xt'],prepared['train'],folder/'refit',device,deadline,
                      epochs=selection['refit_epochs'],fixed_average=selection['averaging_epochs'])
        z,h=predict(net,prepared['xe'],prepared['test'],True);folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'logits.npy',z);np.save(folder/'hidden.npy',h)
        atomic_torch(dict(config=cfg,state=clone_state(net.state_dict()),seed=seed),folder/'model.pt')
        np.savez_compressed(folder/'alignment.npz',cov_prior=prepared['prior'],band=[1,58] if cfg['family']=='context' else [4,36],prior_weight=5)
        save_prediction(out,tag,cfg['name']+f'_seed{seed}',prepared['test'],softmax(z,axis=1),config=cfg,seed=seed,
                        train_people=np.unique(prepared['train']['subject']).tolist(),inner_train_people=np.unique(prepared['inner_train']['subject']).tolist(),
                        inner_val_people=np.unique(prepared['inner_val']['subject']).tolist(),selection=selection,refit=refit)
        del net
        return dict(tag=tag,config=cfg['name'],seed=seed,status='complete')
    except Exception as e:
        failure=dict(tag=tag,config=cfg['name'],seed=seed,status='failed',error=str(e),traceback=traceback.format_exc())
        write_json(folder/'failure.json',failure);print('MODEL_FAILED_CONTINUING',tag,cfg['name'],seed,str(e),flush=True);return failure

def combine(out,cache):
    out=Path(out)
    with np.load(Path(cache)/'meta.npz') as z:meta={k:z[k] for k in META}
    for tag in TAGS:
        tr,te=train_test(meta,tag);mt=subset(meta,te);ref=reference(tag);priors=[];names=[]
        for cfg in CONFIGS:
            folders=[out/'models'/tag/cfg['name']/f'seed{s}' for s in (0,1)]
            if not all((p/'logits.npy').exists() and (out/'predictions'/tag/f"{cfg['name']}_seed{s}.json").exists() for s,p in enumerate(folders)):continue
            z=[np.load(p/'logits.npy') for p in folders]
            p=np.mean([softmax(v,axis=1) for v in z],axis=0);name=cfg['name']+'_mean2'
            save_expert(out,tag,name,mt,p,seeds=[0,1])
            cnn=np.mean([softmax(v/4,axis=1) for v in z],axis=0);mixed=.7*ref['lag_raw']+.3*cnn
            save_prediction(out,tag,name+'__lag4_mix_balance',mt,balanced(mt,mixed))
            priors.append(mixed);names.append(name+'__lag4_mix_tta')
        if priors:
            try:
                grams=tta_grams(cache,meta,tr,te);result=apply_tta_many(grams,mt,np.stack(priors))
                for name,p in zip(names,result):save_prediction(out,tag,name,mt,p)
            except Exception as e:write_json(out/'combination_failures'/f'{tag}.json',dict(error=str(e),traceback=traceback.format_exc()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',default='/kaggle/input');parser.add_argument('--cache',default='/tmp/winner-gpu-cache')
    parser.add_argument('--out',default='/kaggle/working/winner-gpu-results');parser.add_argument('--minutes',type=float,default=690);args=parser.parse_args()
    start=time.monotonic();deadline=start+args.minutes*60;cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True);Path(args.out).mkdir(parents=True,exist_ok=True)
    data=load_data(args.data_root);np.save(cache/'X.npy',data.pop('X'));np.savez(cache/'meta.npz',**data);del data
    from cpu import prep_job
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:list(pool.map(prep_job,[(str(cache),'lag',b) for b in range(5)]))
    print('DEVICES',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],flush=True)
    jobs=[(cfg,seed,tag,str(cache),args.out,deadline) for cfg in CONFIGS for tag in TAGS for seed in (0,1)]
    records=[]
    with ProcessPoolExecutor(max_workers=min(2,max(1,torch.cuda.device_count())),mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for f in as_completed(futures):
            try:records.append(f.result())
            except Exception as e:records.append(dict(status='worker_failed',error=str(e)))
            write_json(Path(args.out)/'progress.json',dict(completed=len(records),planned=len(jobs),jobs=records))
    combine(args.out,args.cache);summarize(args.out)
    write_json(Path(args.out)/'runtime.json',dict(seconds=time.monotonic()-start,planned_fits=len(jobs),jobs=records))

if __name__=='__main__':main()
