"""One final jointly trained LR; full differentiable session history and scaler."""
import math,sys,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'attention_spd_ablation'))
from attention import AttentionSPD,log_spd

class _ExpSymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a):
        a=(a+a.transpose(-1,-2))/2
        vals,vec=torch.linalg.eigh(a.cpu() if a.is_cuda else a)
        vals,vec=vals.to(a.device),vec.to(a.device)
        e=vals.exp()
        if not torch.isfinite(e).all():raise RuntimeError('Nonfinite symmetric exponential')
        ctx.save_for_backward(vals,vec,e)
        return (vec*e.unsqueeze(-2))@vec.transpose(-1,-2)
    @staticmethod
    def backward(ctx,g):
        vals,vec,e=ctx.saved_tensors
        a,b=vals.unsqueeze(-1),vals.unsqueeze(-2);delta=a-b
        ea,eb=e.unsqueeze(-1),e.unsqueeze(-2)
        close=delta.abs()<1e-7
        safe=torch.where(close,torch.ones_like(delta),delta)
        loewner=torch.where(close,(ea+eb)/2,(ea-eb)/safe)
        g=(g+g.transpose(-1,-2))/2
        return vec@(loewner*(vec.transpose(-1,-2)@g@vec))@vec.transpose(-1,-2)

def exp_symmetric(a):return _ExpSymmetric.apply(a)

def session_tangents(r,logr,prior,streams):
    """All historical reference paths remain differentiable; sessions reset."""
    pieces=[];positions=[]
    for stream in streams:
        idx=torch.as_tensor(stream,dtype=torch.long,device=r.device)
        current=logr[idx]
        denom=torch.arange(1,len(idx)+1,device=r.device,dtype=r.dtype)+5
        refs=(5*prior+current.cumsum(0))/denom[:,None,None]
        pieces.append(refs);positions.append(idx)
    positions=torch.cat(positions)
    if len(positions)!=len(r) or len(torch.unique(positions))!=len(r):raise ValueError('Invalid session indices')
    refs=torch.cat(pieces)[torch.argsort(positions)]
    whitener=exp_symmetric(-.5*refs)
    tangent=log_spd(whitener@r@whitener)
    idx=torch.triu_indices(r.shape[-1],r.shape[-1],device=r.device)
    scaling=torch.where(idx[0]==idx[1],torch.ones((),device=r.device,dtype=r.dtype),
                        torch.full((),math.sqrt(2),device=r.device,dtype=r.dtype))
    return tangent[:,idx[0],idx[1]]*scaling

class SingleE2E(nn.Module):
    def __init__(self,mode,*,channels=82,samples=476):
        super().__init__();self.mode=mode
        self.encoder=AttentionSPD(mode,[0],channels=channels,samples=samples)
        d=channels*(channels+1)//2
        generator=torch.Generator().manual_seed(0)
        self.head=nn.Linear(d,3,dtype=torch.float64)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(3,d,generator=generator,dtype=torch.float64)*.01)
            self.head.bias.zero_()
        self.register_buffer('log_prior',torch.zeros(channels,channels,dtype=torch.float64))
        self.register_buffer('feature_mean',torch.zeros(d,dtype=torch.float64))
        self.register_buffer('feature_scale',torch.ones(d,dtype=torch.float64))
    def matrices(self,x):
        if self.mode=='attention_exp':
            lr=self.encoder.intermediate(x[None])[0].double()
            return exp_symmetric(lr),lr
        r=self.encoder(x[None])[0]
        return r,log_spd(r)
    def training_features(self,x,streams):
        r,lr=self.matrices(x);prior=lr.mean(0)
        features=session_tangents(r,lr,prior,streams)
        mean=features.mean(0);variance=(features-mean).square().mean(0)
        scale=torch.where(variance>1e-12,variance.clamp_min(1e-12).sqrt(),torch.ones_like(variance))
        return (features-mean)/scale,dict(prior=prior,mean=mean,scale=scale)
    def train_logits(self,x,streams):
        features,stats=self.training_features(x,streams)
        return self.head(features),stats
    @torch.no_grad()
    def update_training_stats(self,x,streams):
        _,stats=self.training_features(x,streams)
        self.log_prior.copy_(stats['prior']);self.feature_mean.copy_(stats['mean']);self.feature_scale.copy_(stats['scale'])
    @torch.no_grad()
    def predict_proba(self,x,streams):
        x=torch.as_tensor(x,dtype=torch.float32,device=self.encoder.wq.device)
        r,lr=self.matrices(x)
        features=session_tangents(r,lr,self.log_prior,streams)
        logits=self.head((features-self.feature_mean)/self.feature_scale)
        return logits.softmax(-1).cpu().numpy()


def fit_single(x,y,streams,*,mode,device='cuda',epochs=80):
    begin=time.monotonic();device=torch.device(device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=torch.as_tensor(x,dtype=torch.float32,device=device)
    y=torch.as_tensor(y,dtype=torch.long,device=device)
    model=SingleE2E(mode,channels=x.shape[1],samples=x.shape[2]).to(device)
    n=len(x)
    with torch.no_grad():
        centered=x-x.mean(-1,keepdim=True)
        model.encoder.input_scale.fill_(float(centered.square().mean().sqrt().clamp_min(1e-6).cpu()))
        features,stats=model.training_features(x,streams)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        initializer=LogisticRegression(C=.01,max_iter=2000,class_weight='balanced',random_state=0)
        initializer.fit(features.cpu().numpy(),y.cpu().numpy())
    with torch.no_grad():
        model.head.weight.copy_(torch.as_tensor(initializer.coef_,device=device))
        model.head.bias.copy_(torch.as_tensor(initializer.intercept_,device=device))
    initial=model.head.weight.detach().cpu().clone()
    del features,stats,initializer
    counts=torch.bincount(y,minlength=3).double()
    if (counts==0).any():raise ValueError('All three training classes required')
    weights=(n/(3*counts))[y]
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'weight_decay':.001},
                                {'params':model.head.parameters(),'weight_decay':0.}],lr=.001)
    history=[]
    for epoch in range(epochs):
        start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        logits,stats=model.train_logits(x,streams)
        ce=(F.cross_entropy(logits,y,reduction='none')*weights).sum()/n
        regularizer=model.head.weight.square().sum()/(2*.01*n)
        loss=ce+regularizer
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite end-to-end objective')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        qgrad=float(model.encoder.wq.grad.norm().detach().cpu());kgrad=float(model.encoder.wk.grad.norm().detach().cpu())
        optimizer.step()
        row=dict(epoch=epoch+1,objective=float(loss.detach().cpu()),crossentropy=float(ce.detach().cpu()),
                 gradient_norm=float(grad.cpu()),q_gradient_norm=qgrad,k_gradient_norm=kgrad,
                 epoch_seconds=time.monotonic()-start)
        history.append(row)
        if epoch==0 or (epoch+1)%10==0 or epoch==epochs-1:
            print('E2E EPOCH',mode,epoch+1,'loss',round(row['objective'],6),'seconds',round(row['epoch_seconds'],2),flush=True)
        del logits,stats,loss,ce,regularizer
    model.update_training_stats(x,streams);model.eval()
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(model=model,history=history,seconds=time.monotonic()-begin,
                initial_head_weight=initial,retained_head_weight=model.head.weight.detach().cpu().clone(),
                warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught])
