"""Small causal transforms,matching the supplied repository mechanisms."""
import hashlib,sys
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from sklearn.covariance import oas
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'Brain-Decoding/track2_bci'))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def check_digest(path,expected):
    if sha(path)!=expected:raise RuntimeError(f'Changed artifact: {path}')

def matrices(x,band,family):
    x=sosfiltfilt(butter(4,band,btype='bandpass',fs=120,output='sos'),np.asarray(x,np.float64),axis=-1)
    if family=='adaptive':
        tau=4 if tuple(band)==(4,40) else max(1,int(round(120/(4*np.mean(band)))))
        n,c,t=x.shape
        x=np.stack([x[...,tau:],x[...,:-tau]],axis=-2).reshape(n,2*c,t-tau)
    elif family!='original':raise ValueError(family)
    return np.stack([oas(a.T)[0] for a in x])

def stretch(x,groups,weight=10):
    out=np.empty_like(x)
    for idx in groups:
        values=x[idx]
        square=np.cumsum(np.concatenate([np.full((1,x.shape[1]),float(weight)),values*values]),axis=0)[1:]
        out[idx]=values/np.sqrt(square/(weight+np.arange(1,len(idx)+1))[:,None])
    return out

def balance(p,groups,alpha,weight=10):
    out=np.empty_like(p)
    for idx in groups:
        values=p[idx]
        cumulative=np.cumsum(np.concatenate([np.zeros((1,3)),values]),axis=0)[1:]
        marginal=(cumulative+weight/3)/(weight+np.arange(1,len(idx)+1))[:,None]
        q=values/marginal**alpha
        out[idx]=q/q.sum(1,keepdims=True)
    return out
