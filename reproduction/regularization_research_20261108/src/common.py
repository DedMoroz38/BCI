"""Shared split,metric and artifact rules; no training performed at import."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent
AUTHOR=ROOT/'ensemble_author_20261108/track2_bci'
if not any(name=='bci' for name in sys.modules):sys.path.insert(0,str(AUTHOR))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def metadata(w):return {k:getattr(w,k) for k in ('y','subject','session','context','run','onset')}

def cells(w):return np.char.add(np.char.add(w.stream,'/'),w.context)

def folds(protocol='all10',test=False):
    from bci.data import concat,load_split
    from bci.evaluation import later_sessions,leave_one_subject_out
    parts=[load_split(s,ROOT/'data') for s in ('train','val','test')]
    if protocol=='legacy' and test:return [('test_official',parts[0],parts[2])]
    full=concat(*parts) if protocol=='all10' else concat(*parts[:2])
    jobs=[(f'loso_{s.split("/")[-1]}',tr,te) for s,tr,te in leave_one_subject_out(full)]
    jobs.extend(('later_last',tr,te) for _,tr,te in later_sessions(full))
    return jobs

def inner_split(subjects):
    from sklearn.model_selection import GroupShuffleSplit
    return next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=33).split(np.zeros(len(subjects)),groups=subjects))

def loss_weights(y,groups,kind):
    y=np.asarray(y);groups=np.asarray(groups)
    if kind=='class':groups=np.zeros(len(y),int)
    elif kind not in ('subject','cell'):raise ValueError(kind)
    unique=np.unique(groups);weights=np.empty(len(y),float)
    for g in unique:
        idx=np.flatnonzero(groups==g);counts=np.bincount(y[idx],minlength=3)
        if np.any(counts==0):raise ValueError('Each training/validation domain must contain3classes')
        weights[idx]=len(y)/(len(unique)*3*counts[y[idx]])
    return weights

def balance(p,groups,alpha):
    p=np.asarray(p);out=np.empty_like(p)
    for idx in groups:
        prior=(np.cumsum(p[idx],axis=0)+10/3)/(10+np.arange(1,len(idx)+1))[:,None]
        q=p[idx]/prior**alpha;out[idx]=q/q.sum(1,keepdims=True)
    return out

def mix(p,q,weight,pool='arithmetic'):
    if pool=='arithmetic':out=(1-weight)*p+weight*q
    elif pool=='geometric':out=softmax((1-weight)*np.log(np.maximum(p,1e-15))+weight*np.log(np.maximum(q,1e-15)),axis=1)
    else:raise ValueError(pool)
    return out/out.sum(1,keepdims=True)

def scores(w,p):
    from bci.evaluation import score
    p=np.asarray(p);np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    assert np.isfinite(p).all() and np.all(p>=0)
    return dict(score(w,p.argmax(1)),log_loss=float(-np.log(np.maximum(p[np.arange(len(w)),w.y],1e-15)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[w.y])**2,axis=1))))

def save_prediction(path,w,p,**fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**metadata(w),proba=p,pred=p.argmax(1))
    record=dict(prediction_sha256=sha(path),**fields,**scores(w,p));write_json(path.with_suffix('.json'),record)
    return record
