"""Training-only signal transforms and parameter averaging policies."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from common import AUTHOR
from bci.stream import ShallowNet


def weighted_loss(z,y,w,smoothing=0.):
    return (F.cross_entropy(z,y,reduction='none',label_smoothing=smoothing)*w).mean()


def uniform(shape,x,g):return torch.rand(shape,device=x.device,dtype=x.dtype,generator=g)


def beta(n,alpha,x,g):
    a=torch.full((n,),alpha,device=x.device,dtype=x.dtype)
    u=torch._standard_gamma(a,generator=g);v=torch._standard_gamma(a,generator=g)
    return u/(u+v).clamp_min(torch.finfo(x.dtype).tiny)


def partners(y,subjects,g):
    eligible=(y[:,None]==y[None,:])&(subjects[:,None]!=subjects[None,:])
    u=torch.rand(eligible.shape,device=y.device,generator=g).masked_fill(~eligible,-1)
    selected=u.argmax(1)
    return torch.where(eligible.any(1),selected,torch.arange(len(y),device=y.device))


def augment(x,y,subjects,cfg,g):
    active={k:v for k,v in cfg.items() if k!='style' and v}
    if not active:return x
    z=x.clone();n,c,t=z.shape
    if cfg.get('global_gain'):z=z*(1+(2*uniform((n,1,1),z,g)-1)*cfg['global_gain'])
    if cfg.get('coordinate_gain'):z=z*(1+(2*uniform((n,c,1),z,g)-1)*cfg['coordinate_gain'])
    if cfg.get('noise'):
        rms=z.square().mean(-1,keepdim=True).sqrt()
        z=z+torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=g)*rms*cfg['noise']
    if cfg.get('drop'):
        k=int(cfg['drop']);assert 0<k<c
        selected=uniform((n,c),z,g).argsort(1)[:,:k]
        keep=torch.ones((n,c),device=z.device,dtype=z.dtype).scatter(1,selected,0)
        z=z*keep[:,:,None]
    if cfg.get('mask'):
        length=int(cfg['mask']);assert 0<length<t
        start=torch.randint(t-length+1,(n,1,1),device=z.device,generator=g)
        time=torch.arange(t,device=z.device)[None,None,:]
        z=z*((time<start)|(time>=start+length))
    if cfg.get('shift'):
        width=int(cfg['shift']);assert 0<width<t
        shifts=torch.randint(-width,width+1,(n,1,1),device=z.device,generator=g)
        idx=torch.arange(t,device=z.device)[None,None,:]+width+shifts
        z=F.pad(z,(width,width),mode='reflect').gather(-1,idx.expand(n,c,t))
    if cfg.get('flip'):z=z*torch.where(uniform((n,1,1),z,g)<cfg['flip'],-1.,1.)
    if cfg.get('mix'):
        p=partners(y,subjects,g);lam=beta(n,cfg['mix'],z,g)[:,None,None]
        z=lam*z+(1-lam)*z[p]
    return z


def mix_style(h,y,subjects,alpha,g):
    if y is None or subjects is None or float(uniform((),h,g))>=.5:return h
    p=partners(y,subjects,g)
    mu=h.mean(-1,keepdim=True).detach();var=h.var(-1,keepdim=True,unbiased=False).detach()
    sd=(var+1e-6).sqrt();lam=beta(len(h),alpha,h,g).view(-1,1,1,1)
    value=(h-mu)/sd*(lam*sd+(1-lam)*sd[p])+lam*mu+(1-lam)*mu[p]
    valid=(var.flatten(1).amax(1)>1e-6).view(-1,1,1,1)
    return torch.where(valid,value,h)


class RegularizedNet(ShallowNet):
    def __init__(self,n_filters,style=0.):
        super().__init__(41,480,n_filters=n_filters,dropout=.5);self.style=style
    def forward(self,x,y=None,subjects=None,generator=None):
        if not (self.training and self.style):return super().forward(x)
        h=self.norm(self.spatial(self.temporal(x[:,None])))
        h=torch.log(torch.clamp(self.pool(h*h),min=1e-6))
        h=mix_style(h,y,subjects,self.style,generator)
        return self.out(self.drop(h).flatten(1))


def new_net(cfg,seed,device):
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    return RegularizedNet(cfg['n_filters'],cfg.get('augment',{}).get('style',0.)).to(device)


def clone_state(state):return {k:v.detach().cpu().clone() for k,v in state.items()}


def mean_states(states):
    assert states
    result=clone_state(states[-1])
    for key,value in result.items():
        if value.is_floating_point():
            value.copy_(states[0][key]);
            for n,state in enumerate(states[1:],2):value.add_((state[key]-value)/n)
    return result


class StateAverage:
    def __init__(self,kind='mean',decay=0.):self.kind=kind;self.decay=decay;self.n=0;self.state=None
    def update(self,state):
        if self.state is None:self.state=clone_state(state)
        else:
            factor=1-self.decay if self.kind=='ema' else 1/(self.n+1)
            for k,v in self.state.items():
                if v.is_floating_point():v.add_((state[k].detach().cpu()-v)*factor)
                else:v.copy_(state[k].detach().cpu())
        self.n+=1
    def result(self):return clone_state(self.state)
    def state_dict(self):return dict(kind=self.kind,decay=self.decay,n=self.n,state=self.state)
    def load_state_dict(self,s):self.kind=s['kind'];self.decay=s['decay'];self.n=s['n'];self.state=s['state']


@torch.no_grad()
def refresh_bn(net,x,batch=64):
    # Only BatchNorm trains: no dropout,augmentations,or validation data.
    net.eval();modules=[m for m in net.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    momenta={m:m.momentum for m in modules}
    for m in modules:m.reset_running_stats();m.train()
    seen=0
    for v in x.split(batch):
        for m in modules:m.momentum=len(v)/(seen+len(v))
        net(v);seen+=len(v)
    for m in modules:m.momentum=momenta[m]
    net.eval();return net


class LossValley:
    """Scalar/epoch-index adaptation of khanrc/swad LossValley (MIT).

    Dense parameter segment means are maintained by the training loop.
    The original source/license are archived under literature/vendor/.
    """
    def __init__(self,n_converge=3,n_tolerance=6,ratio=.05):
        self.nc=n_converge;self.nt=n_tolerance;self.ratio=ratio
        self.cq=deque(maxlen=n_converge);self.sq=deque(maxlen=n_tolerance)
        self.start=None;self.threshold=None;self.dead=False;self.selected=[]
    def update(self,epoch,loss):
        if self.dead:return
        item=(epoch,float(loss));self.cq.append(item);self.sq.append(item)
        if self.start is None:
            if len(self.cq)<self.nc:return
            if int(np.argmin([v for _,v in self.cq]))==0:
                # Author AveragedModel starts with n_averaged=0: the first
                # update replaces its initial copied segment. Preserve that
                # behavior,including repeated segment weights in later queues.
                self.start=self.cq[0][0];self.selected=[]
                self.threshold=np.mean([v for _,v in self.cq])*(1+self.ratio)
                if self.nt<self.nc:self.selected += [k for k,_ in list(self.cq)[1:1+self.nc-self.nt]]
                elif self.nt>self.nc:
                    q=list(self.sq)[:self.nt-self.nc+1];start=0
                    for i in reversed(range(len(q))):
                        if q[i][1]>self.threshold:start=i+1;break
                    self.selected += [k for k,_ in q[start+1:]]
            return
        if self.sq[0][0]<self.start:return
        if min(v for _,v in self.sq)>self.threshold:self.dead=True;return
        self.selected.append(self.sq[0][0])
    def final_epochs(self):
        if self.start is None:return [self.cq[-1][0]]
        selected=list(self.selected)
        if not self.dead:
            q=list(self.sq)[1:]
            for i,(epoch,_) in enumerate(q):
                if min(v for _,v in q[i:])>self.threshold:break
                selected.append(epoch)
        return sorted(selected or [self.start])
    def state_dict(self):return dict(nc=self.nc,nt=self.nt,ratio=self.ratio,cq=list(self.cq),sq=list(self.sq),start=self.start,threshold=float(self.threshold) if self.threshold is not None else None,dead=self.dead,selected=self.selected)
    def load_state_dict(self,s):
        self.__init__(s['nc'],s['nt'],s['ratio']);self.cq.extend(s['cq']);self.sq.extend(s['sq'])
        for key in ('start','threshold','dead','selected'):setattr(self,key,s[key])
