"""Data, causal transformations and complete-fold reporting. No parity gates."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.special import softmax
from tta_core import spectral,upper,balance,features,causal_geometry

HERE=Path(__file__).resolve().parent
META=('y','subject','session','context','run','onset')
TAGS=[f'loso_P{i:02}' for i in range(1,11)]+['later_last']
BANDS=((4,8),(8,13),(13,20),(20,30),(30,40))
TTA_BANDS=((4,8),(8,13),(13,20),(20,30),(4,40))

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def streams(meta):
    keys=np.char.add(np.char.add(meta['subject'],'/'),meta['session'])
    for key in np.unique(keys):
        ids=np.flatnonzero(keys==key)
        yield ids[np.lexsort((meta['onset'][ids],meta['run'][ids]))]

def subset(meta,ids):return {k:meta[k][ids] for k in META}

def load_data(root):
    parts=[]
    for split in ('train','val','test'):
        p=next(Path(root).rglob(f'proteus_{split}.npz'))
        with np.load(p) as z:parts.append({k:z[k] for k in ('X',)+META})
    return {k:np.concatenate([v[k] for v in parts]) for k in ('X',)+META}

def reference(tag):
    with np.load(HERE/'reference'/f'{tag}.npz') as z:return {k:z[k] for k in z.files}

def train_test(meta,tag):
    te=reference(tag)['data_ids'];mask=np.ones(len(meta['y']),bool);mask[te]=False
    return np.flatnonzero(mask),te

def score(meta,p):
    y=meta['y'];pred=np.asarray(p).argmax(-1)
    keys=np.array(['|'.join(map(str,r)) for r in zip(meta['subject'],meta['session'],meta['context'])])
    values=[]
    for key in np.unique(keys):
        ids=np.flatnonzero(keys==key);classes=np.union1d(y[ids],pred[ids])
        values.append(np.mean([np.mean(pred[ids][y[ids]==c]==c) if np.any(y[ids]==c) else 0 for c in classes]))
    return dict(cell=float(np.mean(values)),nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))

def balanced(meta,p):
    q=np.empty_like(p)
    for ids in streams(meta):q[ids]=balance(p[ids])
    return q

def save_prediction(out,tag,name,meta,p,**info):
    folder=Path(out)/'predictions'/tag;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}.npz';tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**meta,proba=p)
    tmp.replace(path)
    record=dict(tag=tag,name=name,**score(meta,p),**info)
    write_json(path.with_suffix('.json'),record)
    print('RESULT',tag,name,round(record['cell'],6),flush=True)
    return record

def save_expert(out,tag,name,meta,p,**info):
    ref=reference(tag);save_prediction(out,tag,name,meta,p,**info)
    ep=balanced(meta,p)
    for w in (.1,.2):
        save_prediction(out,tag,name+f'__late_w{w}',meta,(1-w)*ref['champion']+w*ep,
                        fusion='late',expert=name,weight=w)

def oas(x):
    x=x-x.mean(-1,keepdims=True);n=x.shape[-1];d=x.shape[-2]
    s=x@np.swapaxes(x,-1,-2)/n;alpha=np.mean(s*s,axis=(-1,-2));mu=np.trace(s,axis1=-2,axis2=-1)/d
    den=(n+1)*(alpha-mu*mu/d);r=np.minimum((alpha+mu*mu)/np.maximum(den,1e-100),1.)
    return (1-r[...,None,None])*s+(r*mu)[...,None,None]*np.eye(d)

def filtering(x,band):
    return sosfiltfilt(butter(4,band,btype='bandpass',fs=120,output='sos'),np.asarray(x,dtype='float64'),axis=-1)

def align(x,meta,prior,weight=5):
    """x is already filtered; current and past within each stream only."""
    out=np.empty(x.shape,dtype='float32');d=x.shape[1]
    for ids in streams(meta):
        total=weight*prior.copy();n=weight
        for i in ids:
            total+=x[i]@x[i].T/x.shape[-1];n+=1
            w,v=np.linalg.eigh((total/n+total.T/n)/2)
            W=(v/np.sqrt(np.maximum(w,np.max(w)*1e-10)))@v.T
            out[i]=W@x[i]
    return out

def get_aligned(x,meta,tr,te,band=(1,58)):
    a=filtering(x[tr],band);b=filtering(x[te],band)
    prior=np.mean(a@np.swapaxes(a,-1,-2)/a.shape[-1],axis=0)
    return align(a,subset(meta,tr),prior),align(b,subset(meta,te),prior),prior

def tangent(covs,meta,source):
    L=spectral(covs,'log');result=np.empty((len(covs),covs.shape[1]*(covs.shape[1]+1)//2))
    for ids in streams(meta):
        result[ids]=features(L[ids,None],source[None],'ordinary_running',np.zeros((len(ids),3)))
    return result

def class_weights(y):
    n=np.bincount(y,minlength=3);return len(y)/(3*np.maximum(n[y],1))

def tta_grams(cache,meta,tr,te):
    """Shared label-free Lag4 kernel for all new priors in this outer fold."""
    arrays=[np.load(Path(cache)/f'laglog_{b}.npy',mmap_mode='r') for b in range(5)]
    source=np.stack([np.mean(a[tr],axis=0,dtype='float64') for a in arrays])
    logs=np.stack([a[te] for a in arrays],axis=1).astype('float64')
    result=[]
    for ids in streams(subset(meta,te)):
        z=features(logs[ids],source,'lag_running',reference_from_probs(len(ids)))
        result.append((ids,(z@z.T,)))
    return result

def reference_from_probs(n):return np.full((n,3),1/3)

def apply_tta_many(grams,meta,priors,deadline=None):
    cfg=dict(tau=.5,beta=1.,k=20,iters=5,shrink=.5,warm=20,ramp=0,strength=.5)
    out=np.empty((len(priors),len(meta['y']),3))
    for ids,g in grams:out[:,ids]=causal_geometry(g,priors[:,ids],[cfg],deadline)[0]
    return out

def summarize(out):
    import pandas as pd
    out=Path(out);records=[];groups={}
    for p in sorted((out/'predictions').glob('*/*.json')):
        r=json.loads(p.read_text());records.append(r);groups.setdefault(r['name'],{})[r['tag']]=p.with_suffix('.npz')
    pd.DataFrame(records).to_csv(out/'fold_scores.csv',index=False)
    rows=[]
    for name,files in groups.items():
        if not all(tag in files for tag in TAGS[:10]):continue
        data=[]
        for tag in TAGS[:10]:
            with np.load(files[tag]) as z:data.append({k:z[k] for k in META+('proba',)})
        m={k:np.concatenate([v[k] for v in data]) for k in META};p=np.concatenate([v['proba'] for v in data])
        values=score(m,p);late=None
        if 'later_last' in files:
            with np.load(files['later_last']) as z:late=score({k:z[k] for k in META},z['proba'])['cell']
        rows.append(dict(name=name,oof_cell=values['cell'],mean_person=float(np.mean([score(v,v['proba'])['cell'] for v in data])),
                         later_cell=late,nll=values['nll'],brier=values['brier'],delta_pp=100*(values['cell']-.627233744814203)))
    df=pd.DataFrame(rows)
    if len(df):df=df.sort_values('oof_cell',ascending=False);df.to_csv(out/'comparison.csv',index=False)
    report='# Winner-inspired research\n\n'+f'Completed prediction records: {len(records)}. Complete LOSO configurations: {len(rows)}.\n\n'
    report+='Historical full baseline: OOF62.7234%, later71.1102%. No baseline training or parity gates.\n'
    report+='Public CV is exploratory; new branches are also evaluated in the full ensemble.\n\n'
    report+=df.head(40).to_string(index=False) if len(df) else 'No complete ten-person configuration yet.'
    (out/'REPORT.md').write_text(report+'\n');print(report,flush=True)
    write_json(out/'summary.json',dict(predictions=len(records),complete_loso_configs=len(rows)))
