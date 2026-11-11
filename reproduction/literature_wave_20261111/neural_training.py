"""Previous Shallow/meanstd/early recipes,refit only inside each source split."""
from pathlib import Path
from copy import deepcopy
import csv,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import *
from averaging import StateAverage,LossValley,mean_states,clone_state
from transfer_nets import TransferNet
class Shallow(nn.Module):
    """Same parameter names and operations as the saved author ShallowNet."""
    def __init__(self,n_chans=41,n_times=480,n_classes=3,n_filters=16,kernel=13,pool=35,stride=7,dropout=.5):
        super().__init__();self.config=dict(n_chans=n_chans,n_times=n_times,n_classes=n_classes,n_filters=n_filters,kernel=kernel,pool=pool,stride=stride,dropout=dropout)
        self.temporal=nn.Conv2d(1,n_filters,(1,kernel))
        self.spatial=nn.Conv2d(n_filters,n_filters,(n_chans,1),bias=False)
        self.norm=nn.BatchNorm2d(n_filters);self.pool=nn.AvgPool2d((1,pool),(1,stride));self.drop=nn.Dropout(dropout)
        self.out=nn.Linear(n_filters*((n_times-kernel+1-pool)//stride+1),n_classes)
    def encode(self,x):
        h=self.norm(self.spatial(self.temporal(x[:,None])))
        return torch.log(torch.clamp(self.pool(h*h),min=1e-6)).flatten(1)
    def forward(self,x):return self.out(self.drop(self.encode(x)))

class ContextShallow(Shallow):
    def __init__(self,mode,prior=10,device='cpu'):
        super().__init__();self.mode=mode;self.prior=prior;d=self.out.in_features
        self.register_buffer('source_mean',torch.zeros(d));self.register_buffer('source_var',torch.ones(d))
        if mode=='deep':
            self.adapter=nn.Sequential(nn.Linear(d,16),nn.Tanh(),nn.Linear(16,2*d))
            nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        self.to(device)
    def latent(self,x):
        h=self.encode(x);n=self.prior+torch.arange(1,len(h)+1,device=h.device,dtype=h.dtype)[:,None]
        mean=(self.prior*self.source_mean+h.cumsum(0))/n
        if self.mode=='mean':return h-mean+self.source_mean
        if self.mode=='meanstd':
            second=(self.prior*(self.source_var+self.source_mean.square())+h.square().cumsum(0))/n
            var=(second-mean.square()).clamp_min(.001*self.source_var.clamp_min(1e-6))
            return (h-mean)*self.source_var.clamp_min(1e-6).sqrt()/var.sqrt()+self.source_mean
        context=(mean-self.source_mean)/self.source_var.clamp_min(1e-6).sqrt()
        gamma,beta=self.adapter(context).chunk(2,dim=-1)
        return (1+.25*gamma.tanh())*h+.25*self.source_var.clamp_min(1e-6).sqrt()*beta.tanh()
    def forward(self,x):return self.out(self.drop(self.latent(x)))
    @torch.no_grad()
    def refresh_source(self,x,groups):
        # Source-only BN recalibration, then fixed during causal prefix passes.
        self.eval();self.norm.reset_running_stats();self.norm.train();seen=0
        for b in x.split(64):
            self.norm.momentum=len(b)/(seen+len(b));self.encode(b);seen+=len(b)
        self.norm.momentum=.1;self.norm.eval()
        h=torch.cat([self.encode(b) for b in x.split(64)])
        self.source_mean.copy_(h.mean(0));self.source_var.copy_(h.var(0,unbiased=False).clamp_min(1e-6))

def atomic_torch(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

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

def fit_point(cfg,seed,x,meta,folder,device,deadline,val_x=None,val_meta=None,epochs=30,fixed_average=None):
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

def fit_early(cfg,seed,x,meta,folder,device,deadline,val_x=None,val_meta=None,epochs=30,fixed_average=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);selecting=val_meta is not None
    net=new_net(cfg,seed,device);groups=list(streams(meta));people=[np.flatnonzero(meta['subject']==p) for p in np.unique(meta['subject'])];y=torch.as_tensor(meta['y'],dtype=torch.long)
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
        net.refresh_source(x,groups,people);net.train()
        if hasattr(net,'norm'):net.norm.eval()
        segment=StateAverage();loss_sum=0.;seen=0
        for g in rng.permutation(len(groups)):
            ids=groups[g];idx=torch.as_tensor(ids,device=x.device)
            batch=x[idx]
            if cfg.get('shift'):
                width=cfg['shift'];offset=torch.randint(-width,width+1,(len(idx),1,1),device=x.device)
                positions=torch.arange(batch.shape[-1],device=x.device)[None,None,:]+width+offset
                batch=F.pad(batch,(width,width),mode='reflect').gather(-1,positions.expand(-1,batch.shape[1],-1))
            z=net(batch);yy=y[ids].to(z.device);ww=weights[ids].to(z.device,dtype=z.dtype)
            loss=(F.cross_entropy(z,yy,reduction='none',label_smoothing=cfg.get('ls',0.))*ww).mean()
            loss=loss+net.regularization()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            if cfg['family']=='context':segment.update(net.state_dict())
            loss_sum+=float(loss.detach())*len(ids);seen+=len(ids)
        scheduler.step()
        if cfg['family']=='context':
            snapshots[epoch]=segment.result();monitored=deepcopy(net);monitored.load_state_dict(snapshots[epoch]);monitored.refresh_source(x,groups,people)
        else:monitored=net;monitored.refresh_source(x,groups,people)
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
    net.refresh_source(x,groups,people);net.eval();atomic_torch(clone_state(net.state_dict()),folder/'final.pt')
    report=dict(best_epoch=best_epoch if selecting else epochs,averaging_epochs=selected,refit_epochs=max(selected),
                seconds=time.monotonic()-begin,parameters=sum(p.numel() for p in net.parameters()),config=cfg,seed=seed)
    write_json(folder/'fit.json',report)
    if selecting:plot(folder,history,best_epoch)
    return net,report

def new_net(cfg,seed,device):
    from novel import new_net as factory
    return factory(cfg,seed,device)

@torch.no_grad()
def predict(net,x,meta,hidden=False):
    net.eval();z=np.empty((len(x),3));latent=None
    for ids in streams(meta):
        index=torch.as_tensor(ids,device=x.device)
        h=net.latent(x[index]) if hasattr(net,'latent') else net.encode(x[index])
        z[ids]=net.out(h).detach().cpu().numpy()
        if hidden:
            if latent is None:latent=np.empty((len(x),h.shape[1]),dtype='float32')
            latent[ids]=h.detach().cpu().numpy()
    return (z,latent) if hidden else z

@torch.no_grad()
def refresh_bn(net,x):
    net.eval();net.norm.reset_running_stats();net.norm.train();seen=0
    for batch in x.split(64):
        net.norm.momentum=len(batch)/(seen+len(batch));net(batch);seen+=len(batch)
    net.norm.momentum=.1;net.eval();return net

def fit_plain(cfg,seed,x,meta,folder,device,deadline,val_x=None,val_meta=None,epochs=30,fixed_average=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);selecting=val_meta is not None
    net=new_net(cfg,seed,device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100)
    shuffle=torch.Generator().manual_seed(seed)
    y=torch.as_tensor(meta['y'],dtype=torch.long,device=device)
    weights=torch.as_tensor(class_weights(meta['y']),dtype=torch.float32,device=device)
    best=float('inf');best_epoch=0;bad=0;history=[];snapshots={};valley=LossValley(3,6,.1);start=time.monotonic()
    for epoch in range(1,epochs+1):
        if time.monotonic()>deadline:raise TimeoutError('GPU time budget reached')
        net.train();segment=StateAverage();total=0;seen=0
        permutation=torch.randperm(len(y),generator=shuffle).to(device)
        for idx in permutation.split(64):
            z=net(x[idx]);loss=(F.cross_entropy(z,y[idx],reduction='none')*weights[idx]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad();loss.backward();opt.step();segment.update(net.state_dict())
            total+=float(loss.detach())*len(idx);seen+=len(idx)
        schedule.step();snapshots[epoch]=segment.result()
        monitored=deepcopy(net);monitored.load_state_dict(snapshots[epoch]);refresh_bn(monitored,x)
        validation=metrics(monitored,val_x,val_meta) if selecting else None
        training=metrics(monitored,x,meta) if selecting and (epoch==1 or epoch%5==0) else None
        if selecting:
            value=validation['cell_nll']
            if value<best-1e-4:best=value;best_epoch=epoch;bad=0
            else:bad+=1
            valley.update(epoch,value)
        history.append(dict(epoch=epoch,optimization_ce=total/seen,
                            train_nll=training['cell_nll'] if training else None,train_ba=training['cell'] if training else None,
                            val_nll=validation['cell_nll'] if validation else None,val_ba=validation['cell'] if validation else None))
        with (folder/'curves.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[-1]));writer.writeheader();writer.writerows(history)
        print('SWAD EPOCH',seed,epoch,round(total/seen,4),flush=True)
        del monitored
        if selecting and bad>=6:break
    selected=valley.final_epochs() if selecting else list(fixed_average or [epochs])
    net.load_state_dict(mean_states([snapshots[e] for e in selected]));refresh_bn(net,x)
    atomic_torch(clone_state(net.state_dict()),folder/'final.pt')
    r=dict(best_epoch=best_epoch if selecting else epochs,averaging_epochs=selected,refit_epochs=max(selected),seconds=time.monotonic()-start)
    write_json(folder/'fit.json',r)
    if selecting:plot(folder,history,best_epoch)
    return net,r
