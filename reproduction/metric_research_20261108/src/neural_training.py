"""Small aligned CNNs with person-held-out epoch selection and durable fits."""
import csv,hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import sosfiltfilt
from common import AUTHOR,sha,write_json,cells,loss_weights,scores,softmax
from bci.ensemble import aligned_windows,DEEP_BAND
from bci.stream import ShallowNet,bandpass

CONFIGS={
    'shallow40_class':dict(n_filters=40,loss='class'),
    'shallow16_class':dict(n_filters=16,loss='class'),
    'shallow16_subject':dict(n_filters=16,loss='subject'),
}

class BudgetStop(RuntimeError):pass

def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,tmp);os.replace(tmp,path)

def cpu_state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

class Stopper:
    def __init__(self,patience=6):self.patience=patience;self.best=float('inf');self.best_epoch=0;self.bad=0
    @property
    def stopped(self):return self.bad>=self.patience
    def update(self,value,epoch):
        improved=value<self.best-1e-4
        if improved:self.best=value;self.best_epoch=epoch;self.bad=0
        else:self.bad+=1
        return improved

def new_net(config,seed,device):
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    return ShallowNet(41,480,n_filters=config['n_filters'],dropout=.5).to(device)

def aligned_pair(train,test):
    """All fitted quantities come exclusively from train,not from test/val."""
    sos=bandpass(DEEP_BAND,120)
    xf=sosfiltfilt(sos,np.asarray(train.X,np.float64),axis=-1)
    prior=np.mean([x@x.T/x.shape[-1] for x in xf],axis=0)
    a=np.ascontiguousarray(aligned_windows(train,xf,prior,5),dtype=np.float32)
    del xf
    xf=sosfiltfilt(sos,np.asarray(test.X,np.float64),axis=-1)
    b=np.ascontiguousarray(aligned_windows(test,xf,prior,5),dtype=np.float32)
    return a,b,prior

def align_with_prior(test,prior):
    xf=sosfiltfilt(bandpass(DEEP_BAND,120),np.asarray(test.X,np.float64),axis=-1)
    return np.ascontiguousarray(aligned_windows(test,xf,prior,5),dtype=np.float32)

def logits(net,x,batch=128):
    net.eval()
    with torch.no_grad():return torch.cat([net(v) for v in x.split(batch)]).cpu().numpy()

def evaluate(net,x,w):
    z=logits(net,x);p=softmax(z,axis=1)
    nll=-np.log(np.maximum(p[np.arange(len(w)),w.y],1e-15))
    weight=loss_weights(w.y,cells(w),'cell')
    return dict(ce=float(nll.mean()),cell_nll=float(np.mean(weight*nll)),cell_ba=scores(w,p)['cell_mean'])

def save_history(path,rows):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def plot(folder,rows,best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.3))
    for ax,key,title in zip(axes,('ce','cell_nll','cell_ba'),('CE (eval mode)','Cell-balanced NLL (selection)','Cell balanced accuracy')):
        for prefix,color in (('train','#2676b8'),('val','#d75c35')):
            values=[r for r in rows if r.get(prefix+'_'+key) is not None]
            if values:ax.plot([r['epoch'] for r in values],[r[prefix+'_'+key] for r in values],label=prefix,color=color)
        ax.axvline(best,color='gray',ls='--');ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Inner held-out people only; training evaluated every5epochs')
    fig.tight_layout();fig.savefig(folder/'curves.png',dpi=120);plt.close(fig)

def fit(x,y_windows,config,seed,folder,device,*,val_x=None,val_windows=None,epochs=30,deadline=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    selecting=val_windows is not None
    settings=dict(config=config,seed=seed,epochs=epochs,selecting=selecting,
        people=sorted(set(y_windows.subject)),windows=len(y_windows),
        val_people=sorted(set(val_windows.subject)) if selecting else [],
        source_sha=sha(Path(__file__)),lr=.001,decay=.01,batch64=True,schedule_Tmax=100)
    token=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    if (folder/'complete.json').exists():
        previous=json.loads((folder/'complete.json').read_text())
        if previous['settings_hash']!=token:raise RuntimeError('Changed completed training settings')
        if sha(folder/'final.pt')!=previous['final_sha256']:raise RuntimeError('Changed final weights')
        net=new_net(config,seed,device);net.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True))
        return net.eval(),previous
    if deadline is not None and time.time()>=deadline:raise BudgetStop('Training budget reached')
    net=new_net(config,seed,device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    generator=torch.Generator().manual_seed(seed)
    kind=config['loss'];groups=y_windows.subject if kind=='subject' else cells(y_windows)
    w=torch.as_tensor(loss_weights(y_windows.y,groups,kind),dtype=torch.float32,device=device)
    y=torch.as_tensor(y_windows.y,dtype=torch.long,device=device)
    stop=Stopper();history=[];start_epoch=1;begin=time.monotonic();best_state=None
    if (folder/'latest.pt').exists():
        state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        if state['settings_hash']!=token:raise RuntimeError('Changed partial training settings')
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['shuffle']);torch.set_rng_state(state['rng_cpu'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state(state['rng_cuda'],device)
        start_epoch=state['epoch']+1;history=state['history'];stop.best=state['best'];stop.best_epoch=state['best_epoch'];stop.bad=state['bad']
        best_state=state['best_model']
    for epoch in range(start_epoch,epochs+1):
        if selecting and stop.stopped:break
        if deadline is not None and time.time()>=deadline:raise BudgetStop('Training budget reached')
        net.train();total=0.;seen=0
        permutation=torch.randperm(len(y),generator=generator).to(device)
        for idx in permutation.split(64):
            if deadline is not None and time.time()>=deadline:raise BudgetStop('Training budget reached;previous complete epoch is preserved')
            prediction=net(x[idx]);loss=(F.cross_entropy(prediction,y[idx],reduction='none')*w[idx]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())*len(idx);seen+=len(idx)
        scheduler.step()
        validation=evaluate(net,val_x,val_windows) if selecting else None
        if selecting:
            improved=stop.update(validation['cell_nll'],epoch)
            if improved:
                best_state=cpu_state(net);atomic_save(best_state,folder/'best.pt')
        monitor=epoch==1 or epoch%5==0 or (selecting and stop.stopped) or epoch==epochs
        training=evaluate(net,x,y_windows) if monitor else None
        row=dict(epoch=epoch,optimization_ce=total/seen)
        for prefix,values in (('train',training),('val',validation)):
            row.update({prefix+'_'+key:(values[key] if values else None) for key in ('ce','cell_nll','cell_ba')})
        history.append(row);save_history(folder/'curves.csv',history)
        state=dict(settings_hash=token,epoch=epoch,model=cpu_state(net),best_model=best_state,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),shuffle=generator.get_state(),rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
            history=history,best=stop.best if selecting else 0.,best_epoch=stop.best_epoch,bad=stop.bad)
        atomic_save(state,folder/'latest.pt')
        if monitor:
            if selecting:plot(folder,history,stop.best_epoch)
            print('CNN_EPOCH',folder.parent.parent.name,folder.parent.name,seed,epoch,
                  'trainCE',round(training['ce'],4),'valCE',round(validation['ce'],4) if validation else '-',flush=True)
        if selecting and stop.stopped:break
    selected=stop.best_epoch if selecting else epochs
    if selecting:net.load_state_dict(best_state)
    atomic_save(cpu_state(net),folder/'final.pt')
    record=dict(settings_hash=token,settings=settings,best_epoch=selected,stopped_epoch=len(history),
                seconds=time.monotonic()-begin,parameters=sum(p.numel() for p in net.parameters()),final_sha256=sha(folder/'final.pt'))
    write_json(folder/'complete.json',record)
    return net.eval(),record
