"""Compact causal moment alignment, driven by the literature review."""
import numpy as np
import torch
from torch import nn

class TransferNet(nn.Module):
    def __init__(self,cfg,device='cpu'):
        super().__init__();self.cfg=cfg;self.nf=12 if cfg.get('multiscale') else 16
        self.prior=float(cfg.get('prior',10));self.flat=cfg.get('scope')=='point'
        if cfg.get('multiscale'):
            self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(1,4,(1,k),padding='same',padding_mode='reflect'),
                nn.Conv2d(4,4,(41,1),bias=False)) for k in (13,25,51)])
        else:
            self.temporal=nn.Conv2d(1,16,(1,13));self.spatial=nn.Conv2d(16,16,(41,1),bias=False)
        self.norm=nn.BatchNorm2d(self.nf);self.pool=nn.AvgPool2d((1,35),(1,7));self.drop=nn.Dropout(.5)
        self.out=nn.Linear(self.nf*62,3)
        d=self.nf*62 if self.flat else self.nf
        self.register_buffer('source_mean',torch.zeros(d));self.register_buffer('source_var',torch.ones(d))
        self.register_buffer('early_mean',torch.zeros(self.nf));self.register_buffer('early_var',torch.ones(self.nf))
        if cfg.get('learned'):self.alpha_logit=nn.Parameter(torch.zeros(self.nf))
        self.to(device)

    def pre(self,x):
        if hasattr(self,'branches'):
            h=torch.cat([b(x[:,None]) for b in self.branches],dim=1)
            return self.norm(h)[...,6:-6].squeeze(2)
        return self.norm(self.spatial(self.temporal(x[:,None]))).squeeze(2)

    def source_moments(self,h,people):
        mean=h.mean(-1) if h.ndim==3 else h
        second=h.square().mean(-1) if h.ndim==3 else h.square()
        if self.cfg.get('person_prior'):
            m=torch.stack([mean[idx].mean(0) for idx in people]).mean(0)
            s=torch.stack([second[idx].mean(0) for idx in people]).mean(0)
        else:m=mean.mean(0);s=second.mean(0)
        return m,(s-m.square()).clamp_min(1e-6)

    def moments(self,mean,second,source_mean,source_var,learned=False,decay=None):
        n=len(mean);t=torch.arange(1,n+1,device=mean.device,dtype=mean.dtype)[:,None]
        if decay:
            factor=torch.pow(torch.as_tensor(decay,device=mean.device,dtype=mean.dtype),t)
            mass=self.prior*factor+(1-factor)/(1-decay)
            mu=factor*(self.prior*source_mean+(mean/factor).cumsum(0))/mass
            e2=factor*(self.prior*(source_var+source_mean.square())+(second/factor).cumsum(0))/mass
            return mu,(e2-mu.square()).clamp_min(.001*source_var.clamp_min(1e-6))
        target_mean=mean.cumsum(0)/t;target_second=second.cumsum(0)/t
        target_var=(target_second-target_mean.square()).clamp_min(0)
        a=t/(self.prior+t)
        if learned:a=a*self.alpha_logit.sigmoid()[None]
        mu=(1-a)*source_mean+a*target_mean
        var=(1-a)*source_var+a*target_var+a*(1-a)*(target_mean-source_mean).square()
        return mu,var.clamp_min(.001*source_var.clamp_min(1e-6))

    def align_early(self,h):
        mu,var=self.moments(h.mean(-1),h.square().mean(-1),self.early_mean,self.early_var)
        scale=(self.early_var/var).sqrt().clamp(.5,2.)
        return (h-mu[...,None])*scale[...,None]+self.early_mean[None,:,None]

    def pool_log(self,h):return torch.log(self.pool(h[:, :,None,:].square()).clamp_min(1e-6)).squeeze(2)

    def latent(self,x):
        h=self.pre(x)
        if self.cfg.get('early'):h=self.align_early(h)
        h=self.pool_log(h)
        value=h.flatten(1) if self.flat else h
        mean=value if self.flat else value.mean(-1)
        second=value.square() if self.flat else value.square().mean(-1)
        mu,var=self.moments(mean,second,self.source_mean,self.source_var,
                            self.cfg.get('learned',False),self.cfg.get('decay'))
        scale=(self.source_var/var).sqrt()
        if self.cfg.get('clip'):scale=scale.clamp(.5,2.)
        if self.flat:aligned=(value-mu)*scale+self.source_mean
        else:aligned=(value-mu[...,None])*scale[...,None]+self.source_mean[None,:,None]
        strength=self.cfg.get('strength',1.)
        return ((1-strength)*value+strength*aligned).flatten(1)

    def forward(self,x):return self.out(self.drop(self.latent(x)))

    def regularization(self):
        if hasattr(self,'alpha_logit'):return .01*(self.alpha_logit.sigmoid()-.5).square().mean()
        return 0.

    @torch.no_grad()
    def refresh_source(self,x,groups,people):
        self.eval();self.norm.reset_running_stats();self.norm.train();seen=0
        for b in x.split(64):
            self.norm.momentum=len(b)/(seen+len(b));self.pre(b);seen+=len(b)
        self.norm.momentum=.1;self.norm.eval()
        pre=torch.cat([self.pre(b) for b in x.split(64)])
        em,ev=self.source_moments(pre,people);self.early_mean.copy_(em);self.early_var.copy_(ev)
        if self.cfg.get('early'):
            h=torch.empty((len(x),self.nf,62),device=x.device,dtype=x.dtype)
            for idx in groups:h[idx]=self.pool_log(self.align_early(pre[idx]))
        else:h=self.pool_log(pre)
        value=h.flatten(1) if self.flat else h
        m,v=self.source_moments(value,people);self.source_mean.copy_(m);self.source_var.copy_(v)

def c(name,**kwargs):return dict(name=name,family='context',prior=10,scope='filter',**kwargs)

CONFIGS=[c('filter_p10'),c('filter_p40',**{'prior_override':40}),
         c('filter_person_prior',person_prior=True),
         c('filter_half',strength=.5),c('filter_clip',clip=True),
         c('filter_ttn16',learned=True),c('early_and_filter',early=True),
         c('multiscale12_filter',multiscale=True),c('point_p40',scope_override='point',prior_override=40),
         c('point_half',scope_override='point',strength=.5),c('point_person_prior',scope_override='point',person_prior=True),
         c('filter_memory50',decay=.98),c('filter_shift12',shift=12),c('filter_ls005',ls=.05)]
for cfg in CONFIGS:
    if 'prior_override' in cfg:cfg['prior']=cfg.pop('prior_override')
    if 'scope_override' in cfg:cfg['scope']=cfg.pop('scope_override')

def new_net(cfg,seed,device):
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    return TransferNet(cfg,device)
