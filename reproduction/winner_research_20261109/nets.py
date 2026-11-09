"""ShallowNet-compatible encoder and explicitly causal session adapters."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

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

class CausalTSMNet(nn.Module):
    """Author TSMNet convolution/BiMap/ReEig/LogEig; new prefix LE normalization.

    This is an adaptation, not a claim of reproducing author SPDDSMBN.
    Author custom spectral backward and Stiefel parameter are retained.
    """
    def __init__(self,lag=False,device='cpu',prior=10):
        super().__init__();sys.path.insert(0,str(Path(__file__).parent/'vendor/TSMNet'))
        import spdnets.modules as modules
        import spdnets.functionals as functions
        self.ops=functions;self.lag=lag;self.prior=prior;self.device_cnn=device
        channels=82 if lag else 41;self.dim=20
        self.cnn=nn.Sequential(nn.Conv2d(1,4,(1,25),padding='same',padding_mode='reflect'),
                              nn.Conv2d(4,40,(channels,1)),nn.Flatten(start_dim=2)).to(device)
        self.cov=modules.CovariancePool()
        self.bimap=modules.BiMap((1,40,20),dtype=torch.double,device='cpu')
        self.rectify=modules.ReEig(1e-4);self.logeig=modules.LogEig(20)
        self.out=nn.Linear(210,3).double()
        self.log_scale=nn.Parameter(torch.zeros((),dtype=torch.double))
        self.register_buffer('reference_log',torch.zeros(20,20,dtype=torch.double))
        self.register_buffer('source_dispersion',torch.ones((),dtype=torch.double))
    def geometry(self,x):
        if self.lag:x=torch.stack([x[...,4:],x[...,:-4]],dim=-2).reshape(len(x),82,-1)
        h=self.cnn(x[:,None]);C=self.cov(h).to(device='cpu',dtype=torch.double)
        return self.rectify(self.bimap(C))
    def mapped(self,C):
        L=self.ops.sym_logm.apply(C);n=self.prior+torch.arange(1,len(C)+1,dtype=C.dtype)[:,None,None]
        running=(self.prior*self.reference_log+L.cumsum(0))/n
        W=self.ops.sym_expm.apply(-.5*running)
        return self.logeig(W@C@W)
    def latent(self,x):
        h=self.mapped(self.geometry(x));s=h.square().mean(-1)
        n=self.prior+torch.arange(1,len(h)+1,dtype=h.dtype)
        dispersion=(self.prior*self.source_dispersion+s.cumsum(0))/n
        return h/dispersion.clamp_min(1e-5).sqrt()[:,None]*self.log_scale.exp()
    def forward(self,x):return self.out(self.latent(x))
    @torch.no_grad()
    def refresh_source(self,x,groups):
        self.eval();covs=torch.cat([self.geometry(b) for b in x.split(64)])
        self.reference_log.copy_(self.ops.sym_logm.apply(covs).mean(0))
        total=0.;count=0
        for idx in groups:
            h=self.mapped(covs[idx]);total+=float(h.square().mean(-1).sum());count+=len(idx)
        self.source_dispersion.fill_(max(total/count,1e-5))

CONFIGS=[dict(name='latent_mean_p10',family='context',mode='mean',prior=10),
         dict(name='latent_meanstd_p10',family='context',mode='meanstd',prior=10),
         dict(name='latent_deepset_p10',family='context',mode='deep',prior=10),
         dict(name='latent_deepset_p40',family='context',mode='deep',prior=40),
         dict(name='tsmnet_causal_le',family='tsm',lag=False),
         dict(name='tsmnet_causal_le_lag4',family='tsm',lag=True)]

def new_net(cfg,seed,device):
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    if cfg['family']=='context':return ContextShallow(cfg['mode'],cfg['prior'],device)
    return CausalTSMNet(cfg['lag'],device)
