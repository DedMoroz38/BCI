"""New causal spectral-shape and Cholesky covariance hypotheses, not paper replicas."""
import torch
from torch.nn import functional as F
from transfer_nets import TransferNet

CONFIGS=[dict(name=f'{label}{int(100*strength):03}',kind=kind,
              alignment_strength=strength,recipe='novel',family='context',
              prior=10,scope='filter',early=True)
         for kind,label in [('spectral','spectral_shape'),('covariance','covariance')]
         for strength in (.25,.5)]

class NovelNet(TransferNet):
    def __init__(self,cfg,device='cpu'):
        super().__init__(cfg,device)
        self.register_buffer('source_covariance',torch.eye(self.nf,device=device))
        self.register_buffer('source_psd',torch.ones(self.nf,235,device=device))
        self._collecting=False
        self._stream_state=None

    def _prefix(self,value,source,key):
        """Uncentered moments; streams are separate calls or explicit B1 states."""
        state=self._stream_state
        if state is None:
            count=torch.arange(1,len(value)+1,device=value.device,dtype=value.dtype)
            denom=(self.prior+count).reshape((-1,)+(1,)*(value.ndim-1))
            return (self.prior*source+value.cumsum(0))/denom
        if len(value)!=1:raise ValueError('Incremental inference requires B=1')
        total,n=state.get(key,(torch.zeros_like(source),0))
        total=total+value[0];n+=1;state[key]=(total.detach(),n)
        return ((self.prior*source+total)/(self.prior+n))[None]

    def moments(self,mean,second,source_mean,source_var,learned=False,decay=None):
        if self._stream_state is None:
            return super().moments(mean,second,source_mean,source_var,learned,decay)
        key='early' if source_mean is self.early_mean else 'late'
        mu=self._prefix(mean,source_mean,key+'_mean')
        e2=self._prefix(second,source_var+source_mean.square(),key+'_second')
        return mu,(e2-mu.square()).clamp_min(.001*source_var.clamp_min(1e-6))

    @staticmethod
    def shape_psd(h):
        centered=h-h.mean(-1,keepdim=True)
        window=torch.hann_window(h.shape[-1],device=h.device,dtype=h.dtype)
        psd=torch.fft.rfft(centered*window).abs().square()
        n,d,f=psd.shape
        psd=F.avg_pool1d(F.pad(psd.reshape(n*d,1,f),(4,4),mode='replicate'),9,stride=1).reshape(n,d,f)
        return psd/psd.mean(-1,keepdim=True).clamp_min(1e-8)

    def spectral_adjust(self,h):
        psd=self.shape_psd(h)
        target=self._prefix(psd,self.source_psd,'shape_psd')
        gain=(self.source_psd.clamp_min(1e-5)/target.clamp_min(1e-5)).sqrt().clamp(2/3,1.5)
        mean=h.mean(-1,keepdim=True);centered=h-mean
        filtered=torch.fft.irfft(torch.fft.rfft(centered)*gain,n=h.shape[-1])
        alpha=self.cfg['alignment_strength']
        mixed=(1-alpha)*centered+alpha*filtered
        # Correct finite-window endpoint effects while retaining perwindow power.
        mixed=mixed-mixed.mean(-1,keepdim=True)
        energy=centered.square().mean(-1,keepdim=True)
        scale=(energy.clamp_min(1e-12)/mixed.square().mean(-1,keepdim=True).clamp_min(1e-12)).sqrt()
        mixed=torch.where(energy>1e-12,mixed*scale,centered)
        return mixed+mean

    def stable_covariance(self,cov):
        diagonal=torch.diag_embed(cov.diagonal(dim1=-2,dim2=-1))
        source_scale=self.source_covariance.diagonal().mean().clamp_min(1e-4)
        eye=torch.eye(self.nf,device=cov.device,dtype=cov.dtype)
        return .8*(cov+cov.transpose(-1,-2))*.5+.2*diagonal+1e-4*source_scale*eye

    def align_early(self,h):
        diagonal=super().align_early(h)
        if self._collecting or self.cfg['alignment_strength']==0:return diagonal
        if self.cfg['kind']=='spectral':return self.spectral_adjust(diagonal)
        if self.cfg['kind']!='covariance':raise ValueError(self.cfg['kind'])
        source_second=self.source_covariance+self.early_mean[:,None]*self.early_mean[None,:]
        mu=self._prefix(h.mean(-1),self.early_mean,'cov_mean')
        second=self._prefix(h@h.transpose(-1,-2)/h.shape[-1],source_second,'cov_second')
        covariance=second-mu[:,:,None]*mu[:,None,:]
        lt=torch.linalg.cholesky(self.stable_covariance(covariance))
        ls=torch.linalg.cholesky(self.stable_covariance(self.source_covariance))
        white=torch.linalg.solve_triangular(lt,h-mu[...,None],upper=False)
        recolored=ls@white+self.early_mean[None,:,None]
        a=self.cfg['alignment_strength']
        return (1-a)*diagonal+a*recolored

    @torch.no_grad()
    def refresh_source(self,x,groups,people):
        if self._stream_state is not None:raise RuntimeError('Source refresh during inference')
        self._collecting=True
        try:
            # Parent recomputes BN and raw early moments on source inputs only.
            super().refresh_source(x,groups,people)
            pre=torch.cat([self.pre(b) for b in x.split(64)])
            second=(pre@pre.transpose(-1,-2)/pre.shape[-1]).mean(0)
            self.source_covariance.copy_(second-self.early_mean[:,None]*self.early_mean[None,:])
            if self.cfg['kind']=='spectral':
                total=torch.zeros_like(self.source_psd)
                for idx in groups:
                    total+=self.shape_psd(super().align_early(pre[idx])).sum(0)
                self.source_psd.copy_(total/len(pre))
        finally:self._collecting=False
        pooled=torch.empty((len(x),self.nf,62),device=x.device,dtype=x.dtype)
        for idx in groups:pooled[idx]=self.pool_log(self.align_early(pre[idx]))
        m,v=self.source_moments(pooled,people)
        self.source_mean.copy_(m);self.source_var.copy_(v)
        self.eval()

class SessionPredictor:
    """Bounded-memory B1 inference; reset at every participant-session boundary."""
    def __init__(self,net):self.net=net.eval();self.reset_state()
    def reset_state(self):self.state={}
    @torch.no_grad()
    def predict_logits(self,x):
        if x.shape[0]!=1:raise ValueError('Expected one chronological window')
        self.net._stream_state=self.state
        try:return self.net(x)
        finally:self.net._stream_state=None
    def predict(self,x):return self.predict_logits(x).argmax(-1)

def new_net(cfg,seed,device):
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
    return NovelNet(cfg,device)
