"""Independent dictionary metrics and the frozen Lag4/CNN fusion recipe."""
import numpy as np
from common import softmax,balance

def load(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}

def score(meta,p):
    assert np.isfinite(p).all() and np.all(p>=0)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    y=meta['y'];pred=p.argmax(1)
    def ba(idx):
        return np.mean([np.mean(pred[idx][y[idx]==c]==c) if np.any(y[idx]==c) else 0 for c in np.union1d(y[idx],pred[idx])])
    stream=np.char.add(np.char.add(meta['subject'],'/'),meta['session'])
    values=[ba(np.flatnonzero((stream==s)&(meta['context']==c))) for s,c in set(zip(stream,meta['context']))]
    return dict(cell_mean=float(np.mean(values)),accuracy=float(np.mean(y==pred)),
        log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
        brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))

def balanced_mix(meta,base,logits):
    q=np.mean([softmax(z/4,axis=1) for z in logits],axis=0);p=.7*base+.3*q
    stream=np.char.add(np.char.add(meta['subject'],'/'),meta['session'])
    return balance(p,[np.flatnonzero(stream==s) for s in np.unique(stream)],1)
