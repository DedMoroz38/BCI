"""Independent channel-attention SPD encoders; no evaluation data in fitting."""
import math,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class _LogSPD(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a):
        a=(a+a.transpose(-1,-2))/2
        # cuSOLVER serial dispatch dominates batches of tiny40x40 problems on
        # T4. Offload only this double-precision symmetric eigendecomposition;
        # projections,attention,Loewner backward and optimizer remain on CUDA.
        eig_input=a.cpu() if a.is_cuda else a
        vals,vec=torch.linalg.eigh(eig_input)
        vals=vals.to(a.device);vec=vec.to(a.device)
        if (vals<=0).any() or not torch.isfinite(vals).all():
            raise RuntimeError('Matrix log requires finite SPD input')
        logs=vals.log();ctx.save_for_backward(vals,vec)
        return (vec*logs.unsqueeze(-2))@vec.transpose(-1,-2)
    @staticmethod
    def backward(ctx,g):
        vals,vec=ctx.saved_tensors
        a=vals.unsqueeze(-1);b=vals.unsqueeze(-2)
        delta=a-b
        # log(a)-log(b) divided by a-b, with analytic repeated-eigenvalue limit.
        close=delta.abs()<=1e-7*torch.maximum(a,b)
        safe=torch.where(close,torch.ones_like(delta),delta)
        loewner=torch.where(close,2/(a+b),(a.log()-b.log())/safe)
        g=(g+g.transpose(-1,-2))/2
        return vec@(loewner*(vec.transpose(-1,-2)@g@vec))@vec.transpose(-1,-2)

def log_spd(a):return _LogSPD.apply(a)

class AttentionSPD(nn.Module):
    def __init__(self,mode,tree_ids,*,channels=40,samples=476):
        super().__init__()
        if mode not in ('attention_gram','attention_exp'):raise ValueError(mode)
        self.mode,self.channels,self.samples=mode,channels,samples
        self.tree_ids=list(tree_ids)
        def initialize(width,offset):
            rows=[]
            for tree in tree_ids:
                g=torch.Generator().manual_seed(1000+tree*10+offset)
                rows.append(torch.randn(samples,width,generator=g)/math.sqrt(samples))
            return nn.Parameter(torch.stack(rows))
        self.wq=initialize(16,1);self.wk=initialize(16,2)
        if mode=='attention_gram':self.wv=initialize(64,3)
        self.register_buffer('input_scale',torch.ones(len(tree_ids),1,1,1))
        indices=torch.triu_indices(channels,channels)
        self.register_buffer('indices',indices)
        self.register_buffer('upper_scale',torch.where(indices[0]==indices[1],1.,math.sqrt(2)))
    def intermediate(self,x):
        u=(x-x.mean(-1,keepdim=True))/self.input_scale
        q=F.layer_norm(u@self.wq[:,None],(16,))
        k=F.layer_norm(u@self.wk[:,None],(16,))
        s=(q@k.transpose(-1,-2))/4
        if self.mode=='attention_exp':
            s=s.tanh()
            return (s+s.transpose(-1,-2))/(2*self.channels)
        a=s.softmax(-1);v=u@self.wv[:,None]
        z=v+.1*(a@v);z=z-z.mean(-1,keepdim=True)
        z=z.double()
        r=(z@z.transpose(-1,-2))/64
        return r+.01*torch.eye(self.channels,device=x.device,dtype=z.dtype)
    def forward(self,x):
        r=self.intermediate(x).double()
        if self.mode=='attention_exp':r=torch.matrix_exp(r)
        return (r+r.transpose(-1,-2))/2
    def log_matrix(self,x):
        r=self.intermediate(x).double()
        return r if self.mode=='attention_exp' else log_spd(r)
    def features(self,x):
        r=self.log_matrix(x)
        return (r[...,self.indices[0],self.indices[1]]*self.upper_scale).float()


def train_pack(x,y,effective_weights,*,mode,tree_ids,device='cuda',epochs=80,batch_size=128):
    begin=time.monotonic()
    device=torch.device(device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    x=torch.as_tensor(x,dtype=torch.float32,device=device)
    y=torch.as_tensor(y,dtype=torch.long,device=device)
    w=torch.as_tensor(effective_weights,dtype=torch.float32,device=device)
    t,n,c,s=x.shape
    if w.shape!=(t,n) or (w<0).any() or (w.sum(1)<=0).any():raise ValueError('Invalid weights')
    model=AttentionSPD(mode,tree_ids,channels=c,samples=s).to(device)
    totals=w.sum(1)
    # Scaling and fixed auxiliary standardization ignore zero-bootstrap windows.
    with torch.no_grad():
        centered=x-x.mean(-1,keepdim=True)
        positive=(w>0).float()
        scale=((centered.square().mean((-1,-2))*positive).sum(1)/positive.sum(1)).sqrt().clamp_min(1e-6)
        model.input_scale.copy_(scale[:,None,None,None])
        su=torch.zeros(t,c*(c+1)//2,device=device);sq=su.clone()
        for start in range(0,n,batch_size):
            f=model.features(x[:,start:start+batch_size]);p=positive[:,start:start+batch_size,None]
            su+=(f*p).sum(1);sq+=(f.square()*p).sum(1)
        mean=su/positive.sum(1,keepdim=True)
        std=(sq/positive.sum(1,keepdim=True)-mean.square()).clamp_min(0).sqrt().clamp_min(.02)
    d=mean.shape[-1]
    head=nn.Parameter(torch.zeros(t,d,3,device=device));bias=nn.Parameter(torch.zeros(t,1,3,device=device))
    optimizer=torch.optim.AdamW([{'params':model.parameters(),'weight_decay':.001},
                                {'params':[head,bias],'weight_decay':0.}],lr=.001)
    generator=torch.Generator().manual_seed(0)
    history=[]
    for epoch in range(epochs):
        # Exclude rows with no active tree at all; zero-weight tree/window pairs
        # still cannot affect gradients because this network has no batch state.
        active=torch.nonzero(w.sum(0)>0).flatten()
        order=active[torch.randperm(len(active),generator=generator).to(device)]
        objectives=torch.zeros(t,device=device)
        for start in range(0,len(order),batch_size):
            idx=order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            f=(model.features(x[:,idx])-mean[:,None])/std[:,None]
            logits=f@head+bias
            ce=F.cross_entropy(logits.reshape(-1,3),y[idx].expand(t,-1).reshape(-1),reduction='none').reshape(t,-1)
            empirical=(ce*w[:,idx]).sum(1)/totals*(len(order)/len(idx))
            regularizer=head.square().sum((1,2))/(2*.01*totals)
            losses=empirical+regularizer;loss=losses.sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite attention training loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([*model.parameters(),head,bias],5.,error_if_nonfinite=True)
            optimizer.step();objectives+=losses.detach()*len(idx)/len(order)
        if epoch==0 or (epoch+1)%10==0 or epoch==epochs-1:
            row=dict(epoch=epoch+1,objective=float(objectives.mean().cpu()),
                     per_tree=objectives.cpu().tolist(),last_gradient_norm=float(grad.cpu()))
            history.append(row)
            print('ATTENTION EPOCH',mode,tree_ids,epoch+1,'objective',round(row['objective'],6),flush=True)
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(model=model.eval(),history=history,seconds=time.monotonic()-begin,
                auxiliary=dict(head=head.detach().cpu(),bias=bias.detach().cpu(),mean=mean.cpu(),std=std.cpu()))

@torch.no_grad()
def encode_pack(model,x,batch_size=128):
    device=model.wq.device
    out=[]
    for start in range(0,x.shape[1],batch_size):
        out.append(model(torch.as_tensor(x[:,start:start+batch_size],dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(out,axis=1)
