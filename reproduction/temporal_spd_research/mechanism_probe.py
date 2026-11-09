"""EXPLORATORY, label-free mechanism probes; does not select model parameters."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.signal import butter,sosfiltfilt
from pyriemann.estimation import Covariances
import temporal_features as f
HERE=Path(__file__).resolve().parent

def main():
 with np.load(HERE.parent/'data/proteus_train.npz') as d:
  raw=d['X'][:4].astype(float)
 x=sosfiltfilt(butter(4,(4,40),btype='bandpass',fs=120,output='sos'),raw,axis=-1)
 # Isolate matrix construction AFTER identical filtering; otherwise shuffling
 # before filtering would also change the original covariance by spectral effects.
 original_filter=f.sosfiltfilt
 f.sosfiltfilt=lambda sos,a,axis:a
 try:
  def matrix(a,v):
   if v=='delay1_oas':
    n,h,t=a.shape
    z=np.stack([a[...,1:],a[...,:-1]],axis=-2).reshape(n,2*h,t-1)
    return Covariances('oas').transform(z)
   return f.compute_matrices(a,(4,40),v)
  rng=np.random.default_rng(123);shuffled=x[...,rng.permutation(x.shape[-1])]
  rows=[]
  for v in (f.CONTROL,'delay1_oas')+f.CANDIDATES:
   r=matrix(x,v)
   for tag,a in [('reverse',x[...,::-1]),('shuffle',shuffled)]:
    distance=np.linalg.norm(matrix(a,v)-r,axis=(-1,-2))/np.linalg.norm(r,axis=(-1,-2))
    rows.append(dict(variant=v,intervention=tag,mean_relative_frobenius=float(distance.mean())))
 finally:f.sosfiltfilt=original_filter
 with (HERE/'mechanism_probe.csv').open('w') as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
