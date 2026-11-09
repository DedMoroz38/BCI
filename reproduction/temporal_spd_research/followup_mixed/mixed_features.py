"""Information-preserving, numerically stable encoding of temporal covariances."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import temporal_features as base
from temporal_cache import MatrixCache as BaseCache

CONTROL='original_oas'
CANDIDATES=('pyramid2_mixed',)
MULTIPLICITY={'pyramid2_mixed':2}
DEFINITIONS={'pyramid2_mixed':'Per-electrode orthogonal Hadamard congruence of the two-half OAS block covariance: U=(A+B)/2,V=(A-B)/2,R=[[U,V],[V,U]], interleaved physical copies'}
SOURCES={'pyramid2_mixed':['https://www.ijcai.org/Proceedings/13/Papers/363.pdf']}
expanded_channels=base.expanded_channels


def mix_covariances(c):
 a,b=c[...,0::2,0::2],c[...,1::2,1::2]
 out=np.empty_like(c);u=(a+b)/2;v=(a-b)/2
 out[...,0::2,0::2]=u;out[...,1::2,1::2]=u
 out[...,0::2,1::2]=v;out[...,1::2,0::2]=v
 return out


class MatrixCache:
 def __init__(self,directory=None):self.base=BaseCache(HERE.parent/'cache')
 def __deepcopy__(self,memo):memo[id(self)]=self;return self
 def lookup(self,representation,band,X):
  if representation!='pyramid2_oas':raise ValueError(representation)
  return mix_covariances(self.base.lookup('pyramid2_oas',band,X))


class LiveCache:
 def __deepcopy__(self,memo):memo[id(self)]=self;return self
 def lookup(self,representation,band,X):
  if representation!='pyramid2_oas':raise ValueError(representation)
  return mix_covariances(base.compute_matrices(X,band,'pyramid2_oas'))


def fit_model(train,model_name,representation='pyramid2_mixed',*,cache=None,n_trees=50,n_channels=None):
 if representation!='pyramid2_mixed':raise ValueError(representation)
 return base.fit_model(train,model_name,'pyramid2_oas',cache=LiveCache() if cache is None else cache,n_trees=n_trees,n_channels=n_channels)
