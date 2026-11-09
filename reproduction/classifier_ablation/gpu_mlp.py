"""Train all50 independent small MLPs together using batched CUDA matmuls."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def fit_predict(x,y,weights,eval_x,*,device='cuda',epochs=300):
    start=time.monotonic()
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('The requested MLP experiment requires a CUDA GPU')
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(device)
    x=torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    target=torch.as_tensor(np.asarray(y),dtype=torch.long,device=device)
    w=torch.as_tensor(np.asarray(weights),dtype=torch.float32,device=device)
    test=torch.as_tensor(np.asarray(eval_x),dtype=torch.float32,device=device)
    b,n,d=x.shape;hidden=32
    if target.shape!=(n,) or w.shape!=(b,n):raise ValueError('Invalid batched shapes')
    if (w<0).any() or (w.sum(1)<=0).any():raise ValueError('Invalid weights')
    rng=np.random.RandomState(0)
    a1=np.sqrt(6/(d+hidden));a2=np.sqrt(6/(hidden+3))
    init1=rng.uniform(-a1,a1,size=(d,hidden)).astype('float32')
    init2=rng.uniform(-a2,a2,size=(hidden,3)).astype('float32')
    W1=nn.Parameter(torch.tensor(init1,device=device).expand(b,-1,-1).clone())
    b1=nn.Parameter(torch.zeros((b,1,hidden),device=device))
    W2=nn.Parameter(torch.tensor(init2,device=device).expand(b,-1,-1).clone())
    b2=nn.Parameter(torch.zeros((b,1,3),device=device))
    params=[W1,b1,W2,b2]
    optimizer=torch.optim.Adam(params,lr=1e-3)
    totals=w.sum(1)
    def logits(values):return torch.bmm(F.relu(torch.bmm(values,W1)+b1),W2)+b2
    history=[]
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        z=logits(x)
        ce=F.cross_entropy(z.reshape(-1,3),target.expand(b,-1).reshape(-1),reduction='none').reshape(b,n)
        losses=(ce*w).sum(1)/totals+(W1.square().sum((1,2))+W2.square().sum((1,2)))/(2*totals)
        loss=losses.sum()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
        loss.backward();optimizer.step()
        if epoch in {0,epochs-1} or (epoch+1)%50==0:
            history.append(dict(epoch=epoch+1,mean_objective=float(losses.detach().mean().cpu())))
    with torch.no_grad():p=logits(test).softmax(-1).cpu().numpy().astype(float)
    if device.type=='cuda':torch.cuda.synchronize()
    return dict(tree_proba=p,epochs=epochs,device=str(device),loss_history=history,
                fit_predict_seconds=time.monotonic()-start,
                parameters={name:a.detach().cpu().numpy() for name,a in zip(['W1','b1','W2','b2'],params)})
