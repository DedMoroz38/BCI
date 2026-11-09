"""Nested person-held-out selection with comparable eval-mode learning curves."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit
from classifier import EEGiTransformer,fit_scale


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)

def subject_split(subjects):
    subjects=np.asarray(subjects)
    train,val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=33).split(np.zeros(len(subjects)),groups=subjects))
    if set(subjects[train])&set(subjects[val]):raise RuntimeError('Person leakage')
    return train,val

class EarlyStopper:
    def __init__(self,patience=10,min_delta=1e-4):
        self.patience,self.min_delta=patience,min_delta;self.best=float('inf');self.best_epoch=0;self.bad=0
    def update(self,value,epoch):
        if not np.isfinite(value):raise RuntimeError('Nonfinite validation loss')
        if value<self.best-self.min_delta:
            self.best=value;self.best_epoch=epoch;self.bad=0;return True
        self.bad+=1;return False
    @property
    def stopped(self):return self.bad>=self.patience

def class_weights(y):
    counts=torch.bincount(y,minlength=3).float()
    if (counts==0).any():raise ValueError('All3 training classes required')
    return len(y)/(3*counts)

def _new(seed,x,device):
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested')
        torch.cuda.manual_seed_all(seed);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model=EEGiTransformer().to(device);fit_scale(model,x)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    return model,optimizer,torch.Generator().manual_seed(seed)

def _epoch(model,optimizer,x,y,weights,generator):
    model.train();order=torch.randperm(len(y),generator=generator).to(x.device)
    total=torch.zeros((),device=x.device)
    for start in range(0,len(y),64):
        idx=order[start:start+64];optimizer.zero_grad(set_to_none=True)
        ce=F.cross_entropy(model(x[idx]),y[idx],reduction='none');w=weights[y[idx]]
        loss=(ce*w).sum()/w.sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step();total+=(ce.detach()*w).sum()
    return float((total/weights[y].sum()).cpu())

@torch.no_grad()
def predict(model,x,*,device='cuda'):
    model.eval();x=torch.as_tensor(x,dtype=torch.float32,device=device);outputs=[]
    for start in range(0,len(x),128):outputs.append(model(x[start:start+128]).softmax(-1).cpu().numpy().astype(float))
    p=np.concatenate(outputs);return p/p.sum(1,keepdims=True)

@torch.no_grad()
def evaluate(model,x,y,groups,weights):
    model.eval();ce_sum=torch.zeros((),device=x.device);weighted=ce_sum.clone();outputs=[]
    for start in range(0,len(y),128):
        logits=model(x[start:start+128]);target=y[start:start+128]
        ce=F.cross_entropy(logits,target,reduction='none')
        ce_sum+=ce.sum();weighted+=(ce*weights[target]).sum()
        outputs.append(logits.argmax(1).cpu().numpy())
    pred=np.concatenate(outputs);truth=y.cpu().numpy();groups=np.asarray(groups)
    def ba(yt,yp):
        return float(np.mean([np.mean(yp[yt==c]==c) if np.any(yt==c) else 0 for c in np.union1d(yt,yp)]))
    return dict(ce=float((ce_sum/len(y)).cpu()),weighted_ce=float((weighted/weights[y].sum()).cpu()),
                session_ba=float(np.mean([ba(truth[groups==g],pred[groups==g]) for g in np.unique(groups)])),
                accuracy=float(np.mean(truth==pred)))

def save_csv(path,rows):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def plot_curves(folder,rows,best_epoch,baseline=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
    epochs=[r['epoch'] for r in rows]
    for ax,key,title in zip(axes,['ce','weighted_ce','session_ba'],['Cross-entropy (eval mode)','Class-weighted CE (selection)','Session balanced accuracy']):
        ax.plot(epochs,[r['train_'+key] for r in rows],label='Train',color='#2676b8')
        ax.plot(epochs,[r['val_'+key] for r in rows],label='Validation: held-out people',color='#d75c35')
        if baseline:
            for part,color in [('train','#2676b8'),('val','#d75c35')]:
                ax.axhline(baseline[part][key],ls=':',alpha=.65,color=color,label='Lag4 '+part)
        ax.axvline(best_epoch,color='gray',ls='--',alpha=.6)
        ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Inner selection only; outer evaluation is never used for these curves',fontsize=10)
    path=Path(folder)/'curves.png';fig.savefig(path,dpi=130);plt.close(fig)
    print('CURVE '+str(path.resolve()),flush=True)

def _saved(folder,name):
    path=Path(folder)/name
    if not path.exists():return None
    record=json.loads(path.read_text())
    for rel,value in record['artifacts'].items():
        if digest(Path(folder)/rel)!=value:raise RuntimeError('Changed training artifact: '+rel)
    return record

def train_selection(x,y,groups,val_x,val_y,val_groups,*,seed,folder,device='cuda',max_epochs=80,patience=10,baseline=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    settings=dict(seed=seed,max_epochs=max_epochs,patience=patience,min_delta=1e-4,
                  baseline_sha256=hashlib.sha256(json.dumps(baseline,sort_keys=True).encode()).hexdigest())
    previous=_saved(folder,'selection.json')
    if previous is not None:
        if previous['settings']!=settings:raise RuntimeError('Changed selection configuration')
        return previous
    begin=time.monotonic()
    x=torch.as_tensor(x,dtype=torch.float32,device=device);y=torch.as_tensor(y,dtype=torch.long,device=device)
    val_x=torch.as_tensor(val_x,dtype=torch.float32,device=device);val_y=torch.as_tensor(val_y,dtype=torch.long,device=device)
    model,optimizer,generator=_new(seed,x,device);weights=class_weights(y)
    stop=EarlyStopper(patience);history=[]
    for epoch in range(1,max_epochs+1):
        optimized=_epoch(model,optimizer,x,y,weights,generator)
        train=evaluate(model,x,y,groups,weights);val=evaluate(model,val_x,val_y,val_groups,weights)
        row=dict(epoch=epoch,optimization_weighted_ce=optimized,**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()})
        history.append(row);improved=stop.update(val['weighted_ce'],epoch)
        if improved:
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=epoch,seed=seed),folder/'best.pt')
        save_csv(folder/'curves.csv',history)
        if epoch==1 or epoch%5==0 or stop.stopped or epoch==max_epochs:
            plot_curves(folder,history,stop.best_epoch,baseline)
            print('IT SELECT',folder.parent.name,'seed',seed,'epoch',epoch,'trainCE',round(train['ce'],5),'valCE',round(val['ce'],5),'valBA',round(val['session_ba'],5),flush=True)
        if stop.stopped:break
    # Always show the final curve and the selected epoch.
    plot_curves(folder,history,stop.best_epoch,baseline)
    record=dict(settings=settings,seed=seed,best_epoch=stop.best_epoch,best_validation_weighted_ce=stop.best,
                stopped_epoch=len(history),seconds=time.monotonic()-begin,parameter_count=sum(p.numel() for p in model.parameters()),
                criterion='inner-val class-weighted CE;min_delta1e-4;patience'+str(patience),
                artifacts={name:digest(folder/name) for name in ['best.pt','curves.csv','curves.png']})
    write_json(folder/'selection.json',record)
    return record

def refit(x,y,groups,*,seed,epochs,folder,device='cuda'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    previous=_saved(folder,'refit.json')
    if previous is not None:
        if previous['seed']!=seed or previous['epochs']!=epochs:raise RuntimeError('Changed refit configuration')
        model=EEGiTransformer().to(device);model.load_state_dict(torch.load(folder/'final.pt',weights_only=True,map_location=device)['state'])
        return dict(model=model.eval(),history=previous['history'],seconds=previous['seconds'])
    begin=time.monotonic();x=torch.as_tensor(x,dtype=torch.float32,device=device);y=torch.as_tensor(y,dtype=torch.long,device=device)
    model,optimizer,generator=_new(seed,x,device);weights=class_weights(y);history=[]
    for epoch in range(1,epochs+1):
        optimized=_epoch(model,optimizer,x,y,weights,generator)
        train=evaluate(model,x,y,groups,weights)
        history.append(dict(epoch=epoch,optimization_weighted_ce=optimized,**train))
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},epochs=epochs,seed=seed),folder/'final.pt')
    save_csv(folder/'train_history.csv',history)
    record=dict(seed=seed,epochs=epochs,history=history,seconds=time.monotonic()-begin,
                artifacts={name:digest(folder/name) for name in ['final.pt','train_history.csv']})
    write_json(folder/'refit.json',record)
    print('IT REFIT',folder.parent.name,'seed',seed,'epochs',epochs,'seconds',round(record['seconds'],2),flush=True)
    return dict(model=model.eval(),history=history,seconds=record['seconds'])
