"""Standalone, CPU-only PROTEUS research. No historical parity gates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
META=('y','subject','session','context','run','onset')
TAGS=[f'loso_P{i:02}' for i in range(1,11)]+['later_last']
BANDS=((4,40),(4,8),(8,13),(13,20),(20,30),(30,40))

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def metadata(cache):
    with np.load(Path(cache)/'meta.npz') as z:return {k:z[k] for k in META}

def subset(m,ids):return {k:m[k][ids] for k in META}

def streams(m):
    keys=np.char.add(np.char.add(m['subject'],'/'),m['session'])
    for k in np.unique(keys):
        ids=np.flatnonzero(keys==k)
        yield ids[np.lexsort((m['onset'][ids],m['run'][ids]))]

def reference(tag):
    with np.load(ROOT/'transfer_deep_dive_20261110/reference'/f'{tag}.npz') as z:
        return {k:z[k] for k in z.files}

def split(meta,tag):
    te=reference(tag)['data_ids'];mask=np.ones(len(meta['y']),bool);mask[te]=False
    return np.flatnonzero(mask),te

def score(m,p):
    y=m['y'];pred=p.argmax(1)
    keys=np.array(['|'.join(map(str,t)) for t in zip(m['subject'],m['session'],m['context'])])
    ba=[]
    for k in np.unique(keys):
        ids=np.flatnonzero(keys==k);classes=np.union1d(y[ids],pred[ids])
        ba.append(np.mean([np.mean(pred[ids][y[ids]==c]==c) if np.any(y[ids]==c) else 0 for c in classes]))
    return dict(cell=float(np.mean(ba)),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))

def balance(p,m):
    out=np.empty_like(p)
    for ids in streams(m):
        marginal=(np.cumsum(p[ids],axis=0)+10/3)/(10+np.arange(1,len(ids)+1))[:,None]
        q=p[ids]/np.maximum(marginal,1e-12);out[ids]=q/q.sum(1,keepdims=True)
    return out

def save(out,tag,name,m,p,**extra):
    if not np.isfinite(p).all():raise FloatingPointError('Nonfinite new predictions')
    p=np.maximum(p,1e-15);p/=p.sum(1,keepdims=True)
    folder=Path(out)/'predictions'/tag;folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/(name+'.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**m,proba=p)
    tmp.replace(folder/(name+'.npz'))
    r=dict(tag=tag,name=name,**score(m,p),**extra);write_json(folder/(name+'.json'),r)
    print('RESULT',tag,name,round(r['cell'],6),flush=True)
    return r

def save_pair(out,tag,name,m,p,**extra):
    save(out,tag,name,m,p,**extra)
    save(out,tag,name+'__balance',m,balance(p,m),**extra)

def person_weights(m,classes=True):
    n=len(m['y']);w=np.empty(n)
    for s in np.unique(m['subject']):
        ids=np.flatnonzero(m['subject']==s)
        if classes:
            counts=np.bincount(m['y'][ids],minlength=3)
            w[ids]=1/np.maximum(counts[m['y'][ids]],1)
        else:w[ids]=1/len(ids)
    return w/w.mean()

def spectral(a,kind):
    w,v=np.linalg.eigh((a+a.swapaxes(-1,-2))*.5)
    if kind=='log':w=np.log(np.maximum(w,np.max(w,axis=-1,keepdims=True)*1e-12))
    elif kind=='exp':w=np.exp(w)
    elif kind=='invsqrt':w=1/np.sqrt(np.maximum(w,np.max(w,axis=-1,keepdims=True)*1e-12))
    return (v*w[...,None,:])@v.swapaxes(-1,-2)

def upper(a):
    ij=np.triu_indices(a.shape[-1]);q=a[...,ij[0],ij[1]].copy()
    q[...,ij[0]!=ij[1]]*=np.sqrt(2)
    return q

def unupper(x,d):
    ij=np.triu_indices(d);q=np.zeros(x.shape[:-1]+(d,d),dtype=x.dtype)
    v=x.copy();v[...,ij[0]!=ij[1]]/=np.sqrt(2)
    q[...,ij[0],ij[1]]=v;q[...,ij[1],ij[0]]=v
    return q

def tangent_pair(C,tr,te,m,prior=5,return_source=False):
    a=np.asarray(C[tr],dtype='float64');b=np.asarray(C[te],dtype='float64')
    la=spectral(a,'log');lb=spectral(b,'log');source=la.mean(0)
    def transform(c,l,mt):
        out=np.empty((len(c),c.shape[-1]*(c.shape[-1]+1)//2))
        for ids in streams(mt):
            logref=(prior*source+np.cumsum(l[ids],axis=0))/(prior+np.arange(1,len(ids)+1))[:,None,None]
            W=spectral(-.5*logref,'exp');out[ids]=upper(spectral(W@c[ids]@W,'log'))
        return out
    pair=(transform(a,la,subset(m,tr)),transform(b,lb,subset(m,te)))
    return pair+(source,) if return_source else pair

def filtering(x,band):
    return sosfiltfilt(butter(4,band,btype='bandpass',fs=120,output='sos'),np.asarray(x,dtype='float64'),axis=-1)

def oas(x):
    x=x-x.mean(-1,keepdims=True);n=x.shape[-1];d=x.shape[-2]
    s=x@x.swapaxes(-1,-2)/n;alpha=np.mean(s*s,axis=(-1,-2));mu=np.trace(s,axis1=-2,axis2=-1)/d
    r=np.minimum((alpha+mu*mu)/np.maximum((n+1)*(alpha-mu*mu/d),1e-100),1)
    return (1-r[...,None,None])*s+(r*mu)[...,None,None]*np.eye(d)

def ea(x,mt,source,prior=5):
    out=np.empty_like(x,dtype='float32')
    for ids in streams(mt):
        total=prior*source.copy()
        for j,i in enumerate(ids):
            total+=x[i]@x[i].T/x.shape[-1]
            W=spectral(total/(prior+j+1),'invsqrt');out[i]=W@x[i]
    return out

def summarize(out):
    import pandas as pd
    out=Path(out);records=[];files={}
    for f in sorted((out/'predictions').glob('*/*.json')):
        r=json.loads(f.read_text());records.append(r);files.setdefault(r['name'],{})[r['tag']]=f.with_suffix('.npz')
    if not records:return
    pd.DataFrame(records).to_csv(out/'fold_scores.csv',index=False)
    rows=[];people=[]
    for name,tags in files.items():
        if not all(t in tags for t in TAGS[:10]):continue
        parts=[]
        for t in TAGS[:10]:
            with np.load(tags[t]) as z:part={k:z[k] for k in META+('proba',)}
            parts.append(part);people.append(dict(name=name,tag=t,**score(part,part['proba'])))
        pooled={k:np.concatenate([v[k] for v in parts]) for k in META+('proba',)}
        vals=[score(v,v['proba'])['cell'] for v in parts];late=None
        if 'later_last' in tags:
            with np.load(tags['later_last']) as z:late=score(z,z['proba'])['cell']
        rows.append(dict(name=name,oof_cell=score(pooled,pooled['proba'])['cell'],mean_person=float(np.mean(vals)),
                         worst3=float(np.mean(np.sort(vals)[:3])),later_cell=late,nll=score(pooled,pooled['proba'])['nll']))
    df=pd.DataFrame(rows)
    if len(df):df=df.sort_values('oof_cell',ascending=False)
    df.to_csv(out/'comparison.csv',index=False);pd.DataFrame(people).to_csv(out/'by_person.csv',index=False)
    report='# Standalone MFM models\n\nNo CNN training or CNN fusion. Full-ensemble historical reference64.58% is not a standalone Lag4 baseline.\n\n'
    report+=f'Prediction records: {len(records)}; complete10-person configurations: {len(df)}.\n\n'
    report+=df.to_string(index=False) if len(df) else 'No complete10-fold model yet.'
    (out/'REPORT.md').write_text(report+'\n')
