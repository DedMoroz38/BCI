"""Literature-driven per-window SPD transforms; original model code is unchanged."""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys
import numpy as np
from scipy.signal import butter,sosfiltfilt,hilbert

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Brain-Decoding/track2_bci'))
from bci.riemann import RiemannFeatures
from pyriemann.estimation import Covariances

CONTROL='original_oas'
CANDIDATES=('lag4_oas','stationary_oas','pyramid2_oas','quadrature_oas','path_integral_oas','signature2_oas')
MULTIPLICITY={v:(2 if v in CANDIDATES[:4] else 1) for v in CANDIDATES}
DEFINITIONS={
 'lag4_oas':'OAS of electrode-major [x(t),x(t-4)], 476 aligned samples',
 'stationary_oas':'OAS of globally centered, zero-padded adjacent shifts across T+1 positions; exactly block Toeplitz',
 'pyramid2_oas':'Electrode-interleaved block diagonal of OAS of first/second 240-sample halves',
 'quadrature_oas':'OAS of interleaved real/Hilbert quadratures, projected as (S+J S J^T)/2',
 'path_integral_oas':'OAS on exact two-point Gauss-Legendre quadrature nodes of every piecewise-linear interval',
 'signature2_oas':'Original OAS + window RMS^2 * Phi Phi^T/64; four exact depth-2 time-path signature coordinates on 16 ordered patches',
}
SOURCES={
 'lag4_oas':['https://arxiv.org/abs/2302.04508'],
 'stationary_oas':['https://www.stat.uchicago.edu/~wbwu/papers/cov-reviewJune15-2011-2.pdf','https://arxiv.org/abs/1706.03161'],
 'pyramid2_oas':['https://www.ijcai.org/Proceedings/13/Papers/363.pdf'],
 'quadrature_oas':['https://doi.org/10.1146/annurev-statistics-031219-041138'],
 'path_integral_oas':['https://jmlr.org/papers/v20/16-314.html'],
 'signature2_oas':['https://jmlr.org/papers/v20/16-314.html','https://proceedings.mlr.press/v151/dyer22a.html'],
}


def expanded_channels(channels):
 return (2*np.asarray(channels,dtype=int)[:,None]+np.array([0,1])).ravel()


def continuous_nodes(x):
 a,b=x[...,:-1],x[...,1:]
 mid=(a+b)/2;delta=(b-a)/(2*np.sqrt(3))
 return np.stack([mid-delta,mid+delta],axis=-1).reshape(x.shape[:-1]+(-1,))


def stationary_vectors(x):
 x=x-x.mean(axis=-1,keepdims=True)
 zeros=np.zeros(x.shape[:-1]+(1,))
 now=np.concatenate([x,zeros],axis=-1)
 prev=np.concatenate([zeros,x],axis=-1)
 return np.stack([now,prev],axis=-2).reshape(x.shape[:-2]+(2*x.shape[-2],x.shape[-1]+1))


def signature_coordinates(x):
 """[S_x,S_tx,S_xt,S_xx], exact on a time-augmented linear interpolant."""
 n=x.shape[-1];dt=1/(n-1)
 dx=np.diff(x,axis=-1)
 sx=dx.sum(-1)
 stx=(dx*((np.arange(n-1)+.5)*dt)).sum(-1)
 sxt=((x[...,:-1]-x[...,:1])+.5*dx).sum(-1)*dt
 return np.stack([sx,stx,sxt,.5*sx*sx],axis=-1)


def compute_matrices(X,band,representation):
 x=sosfiltfilt(butter(4,band,btype='bandpass',fs=120,output='sos'),np.asarray(X,dtype=np.float64),axis=-1)
 oas=Covariances('oas')
 n,h,t=x.shape
 if representation==CONTROL:return oas.transform(x)
 if representation=='lag4_oas':
  z=np.stack([x[...,4:],x[...,:-4]],axis=-2).reshape(n,2*h,t-4)
  return oas.transform(z)
 if representation=='stationary_oas':return oas.transform(stationary_vectors(x))
 if representation=='pyramid2_oas':
  a=oas.transform(x[...,:t//2]);b=oas.transform(x[...,t//2:])
  out=np.zeros((n,2*h,2*h));out[:,0::2,0::2]=a;out[:,1::2,1::2]=b
  return out
 if representation=='quadrature_oas':
  z=np.stack([x,np.imag(hilbert(x,axis=-1))],axis=-2).reshape(n,2*h,t)
  s=oas.transform(z)
  perm=np.arange(2*h).reshape(h,2)[:,::-1].ravel();sign=np.tile([-1.,1.],h)
  rotated=s[:,perm,:][:,:,perm]*np.outer(sign,sign)
  return (s+rotated)/2
 if representation=='path_integral_oas':return oas.transform(continuous_nodes(x))
 if representation=='signature2_oas':
  scale=np.maximum(np.sqrt(np.mean(x*x,axis=(-1,-2))),1e-12)
  normalized=x/scale[:,None,None]
  patches=np.array_split(normalized,16,axis=-1)
  if min(a.shape[-1] for a in patches)<2:raise ValueError('Signature patches need at least two samples')
  phi=np.concatenate([signature_coordinates(a) for a in patches],axis=-1)
  addition=(phi@np.swapaxes(phi,-1,-2))*(scale*scale/phi.shape[-1])[:,None,None]
  return oas.transform(x)+addition
 raise ValueError(representation)


class TemporalFeatures(RiemannFeatures):
 def __init__(self,band=(4,40),prior_weight=5,*,representation=CONTROL,cache=None):
  super().__init__(band,prior_weight)
  self.representation,self.cache=representation,cache
 def covariances(self,X):
  if self.representation==CONTROL:return super().covariances(X)
  if self.cache is not None:return self.cache.lookup(self.representation,self.band,X)
  return compute_matrices(X,self.band,self.representation)


@contextmanager
def representation_factory(representation,cache):
 from bci.models import forest,logreg
 a,b,sub=forest.RiemannFeatures,logreg.RiemannFeatures,forest._sub
 factory=partial(TemporalFeatures,representation=representation,cache=cache)
 forest.RiemannFeatures=logreg.RiemannFeatures=factory
 if MULTIPLICITY.get(representation,1)==2:
  forest._sub=lambda cov,ch:sub(cov,expanded_channels(ch))
 try:yield
 finally:forest.RiemannFeatures,logreg.RiemannFeatures,forest._sub=a,b,sub


def fit_model(train,model_name,representation,*,cache=None,n_trees=50,n_channels=None):
 from bci.models import forest,logreg
 with representation_factory(representation,cache):
  if model_name=='logreg':return logreg.fit(train,C=.01,prior_weight=5,seed=0)
  count={'forest12':12,'forest20':20}[model_name] if n_channels is None else n_channels
  model=forest.fit(train,n_trees=n_trees,n_channels=count,C=.01,prior_weight=5,seed=0)
  if MULTIPLICITY.get(representation,1)==2:
   for tree in model.trees:
    tree.physical_channels=tree.channels.copy();tree.channels=expanded_channels(tree.physical_channels)
  return model
