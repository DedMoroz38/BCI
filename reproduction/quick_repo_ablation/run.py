"""Fast fixed-size ablations,with historical50-tree probabilities as controls."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time,warnings,importlib.metadata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from transforms import ROOT,sha,check_digest,matrices,stretch,balance
from bci.data import load_split,concat
from bci.evaluation import leave_one_subject_out,later_sessions,score
from bci.models.forest import BANDS
from bci.riemann import RiemannFeatures
HERE=Path(__file__).resolve().parent;CACHE=HERE/'cache';OUT=HERE/'results'
VARIANTS=('baseline','mixed50','stretch50','balance05','balance10')

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def dev():return concat(load_split('train',ROOT/'data'),load_split('val',ROOT/'data'))

def folds():
    w=dev()
    return [(f'loso_{s.split("/")[-1]}',tr,te) for s,tr,te in leave_one_subject_out(w)]+[
        ('later_last',tr,te) for _,tr,te in later_sessions(w)]

def metadata(w):return {k:getattr(w,k) for k in ('y','subject','session','context','run','onset')}

def specs(w):
    rng=np.random.default_rng(0);subjects=sorted(set(w.subject));out=[]
    for k in range(50):
        band=int(rng.integers(6));channels=np.sort(rng.choice(41,20,replace=False))
        counts=Counter(rng.choice(subjects,len(subjects)))
        out.append((band,channels,np.array([counts[s] for s in w.subject],float)))
    return out

def reference(tag):
    suffix='loso_Dreyer2026Proteus_'+tag[5:] if tag.startswith('loso_') else 'later_last'
    return ROOT/'temporal_spd_research/results/lag4_oas'/('forest20_'+suffix+'.json')

def initialize():
    CACHE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    files=[HERE/n for n in ('run.py','transforms.py','analyze.py','PROTOCOL.md')]
    files+=list((ROOT/'Brain-Decoding/track2_bci/bci').rglob('*.py'))
    files += [ROOT/'BCI_DedMoroz38/Brain-Decoding/track2_bci/bci/riemann.py',
              ROOT/'BCI_DedMoroz38/Brain-Decoding/track2_bci/bci/models/forest.py',
              ROOT/'ensemble_author_20261108/track2_bci/bci/stream.py']
    for tag,tr,te in folds():
        p=ROOT/'classifier_ablation/results/baseline'/f'{tag}_predictions.npz'
        r=json.loads(p.with_name(tag+'.json').read_text())
        check_digest(p,r['predictions_sha256'])
        with np.load(p) as f:
            for k,v in metadata(te).items():np.testing.assert_array_equal(v,f[k])
            np.testing.assert_array_equal(f['proba'].argmax(1),f['pred'])
        ref=reference(tag);history=json.loads(ref.read_text())
        if sorted(set(tr.subject))!=history['train_subjects']:raise RuntimeError('Changed training people')
        draws=[dict(band=b,channels=c.tolist()) for b,c,_ in specs(tr)]
        if draws!=history['trees']:raise RuntimeError('Changed tree draws')
        files += [p,p.with_name(tag+'.json'),ref]
    oldcache=ROOT/'temporal_spd_research/cache'
    for k in range(6):
        p=oldcache/f'lag4_oas_band{k}.npy';receipt=p.with_suffix('.complete.json')
        check_digest(p,json.loads(receipt.read_text())['sha256']);files += [p,receipt]
    files += [oldcache/'manifest.json']
    manifest=dict(files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))},
                  data={s:sha(ROOT/'data'/f'proteus_{s}.npz') for s in ('train','val')},
                  versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pyriemann')},
                  variants=VARIANTS)
    # JSON tuples normalized before resume comparison.
    manifest=json.loads(json.dumps(manifest))
    path=HERE/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise RuntimeError('Changed experiment inputs/code')
    if not path.exists():write_json(path,manifest)
    return sha(path)

def prepare():
    w=dev();start=time.monotonic()
    code=sha(HERE/'transforms.py');data=[sha(ROOT/'data'/f'proteus_{s}.npz') for s in ('train','val')]
    with threadpool_limits(limits=1):
        for family in ('original','adaptive'):
            for b,band in enumerate(BANDS):
                if family=='adaptive' and b==0:continue
                path=CACHE/f'{family}_{b}.npy';receipt=path.with_suffix('.json')
                if receipt.exists():
                    r=json.loads(receipt.read_text())
                    if r['code']!=code or r['data']!=data:raise RuntimeError('Changed matrix configuration')
                    check_digest(path,r['sha256']);continue
                dim=41 if family=='original' else 82
                partial=path.with_suffix('.partial.npy')
                a=np.lib.format.open_memmap(partial,mode='w+',dtype='float64',shape=(len(w),dim,dim))
                for first in range(0,len(w),64):a[first:first+64]=matrices(w.X[first:first+64],band,family)
                a.flush();del a;partial.replace(path)
                write_json(receipt,dict(code=code,data=data,sha256=sha(path),band=list(band),family=family))
                print('CACHE',family,b,round(time.monotonic()-start,1),'seconds',flush=True)

class Covariances:
    def __init__(self,w):
        old=ROOT/'temporal_spd_research/cache';self.old=old
        config=json.loads((old/'manifest.json').read_text())
        import hashlib
        index={key:i for i,key in enumerate(config['window_keys'])}
        self.global_ids=np.array([index[hashlib.blake2b(np.ascontiguousarray(x).tobytes(),digest_size=20).hexdigest()] for x in w.X])
        self.arrays={}
    def subset(self,family,band,idx,channels):
        if family=='adaptive' and band==0:family='lag4'
        key=family,band
        if key not in self.arrays:
            path=self.old/f'lag4_oas_band{band}.npy' if family=='lag4' else CACHE/f'{family}_{band}.npy'
            self.arrays[key]=np.load(path,mmap_mode='r')
        ids=self.global_ids[idx] if family=='lag4' else idx
        rows=channels if family=='original' else (2*channels[:,None]+np.array([0,1])).ravel()
        return np.asarray(self.arrays[key][np.ix_(ids,rows,rows)])

def metrics(w,p):
    return dict(score(w,p.argmax(1)),log_loss=float(-np.log(np.maximum(p[np.arange(len(w)),w.y],1e-15)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[w.y])**2,axis=1))))

def run_fold(tag):
    h=sha(HERE/'manifest.json');folder=OUT/tag;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        record=json.loads((folder/'complete.json').read_text())
        if record['manifest']!=h:raise RuntimeError('Changed completed fold')
        for name,digest in record['files'].items():check_digest(folder/name,digest)
        print('RESUME',tag,flush=True);return
    _,tr,te=next(f for f in folds() if f[0]==tag);w=dev()
    keys=lambda q:list(zip(q.subject,q.session,q.run,q.onset))
    index={key:i for i,key in enumerate(keys(w))}
    train_ids=np.array([index[key] for key in keys(tr)]);eval_ids=np.array([index[key] for key in keys(te)])
    cov=Covariances(w);draws=specs(tr);start=time.monotonic()
    with np.load(ROOT/'classifier_ablation/results/baseline'/f'{tag}_predictions.npz') as f:
        base=f['proba'].copy();base_trees=f['tree_proba'].copy()
    mixed=base_trees.copy();stretched=[];control=[]
    print('START',tag,len(tr),len(te),flush=True)
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
        warnings.simplefilter('always')
        for k,(b,channels,weights) in enumerate(draws):
            def fit_family(family):
                features=RiemannFeatures(BANDS[b],5)
                xt,_=features.fit_transform(tr,cov.subset(family,b,train_ids,channels))
                xe,_=features.transform(te,cov.subset(family,b,eval_ids,channels))
                return xt,xe
            def predict(xt,xe):
                keep=weights>0
                clf=LogisticRegression(C=.01,max_iter=2000,class_weight='balanced')
                clf.fit(xt[keep],tr.y[keep],sample_weight=weights[keep])
                return clf.predict_proba(xe)
            xt,xe=fit_family('lag4')
            pc=predict(xt,xe);np.testing.assert_allclose(pc,base_trees[k],rtol=0,atol=1e-7)
            control.append(pc)
            stretched.append(predict(stretch(xt,list(tr.streams())),stretch(xe,list(te.streams()))))
            if k>=17:
                xt,xe=fit_family('original' if k<34 else 'adaptive')
                mixed[k]=predict(xt,xe)
            if (k+1)%10==0:print('EXPERTS',tag,k+1,'/50',round(time.monotonic()-start,1),'seconds',flush=True)
    np.testing.assert_array_equal(np.mean(control,axis=0).argmax(1),base.argmax(1))
    p={ 'baseline':base,'mixed50':mixed.mean(0),'stretch50':np.mean(stretched,axis=0),
        'balance05':balance(base,list(te.streams()),.5),'balance10':balance(base,list(te.streams()),1)}
    for name,proba in p.items():
        proba=proba/proba.sum(1,keepdims=True)
        if not np.isfinite(proba).all() or np.any(proba<0):raise RuntimeError('Invalid probabilities')
        pred=folder/(name+'.npz')
        np.savez_compressed(pred,**metadata(te),pred=proba.argmax(1),proba=proba)
        r=dict(variant=name,tag=tag,manifest=h,prediction_sha256=sha(pred),**metrics(te,proba))
        write_json(folder/(name+'.json'),r)
        print('DONE',tag,name,round(r['session_mean'],6),flush=True)
    write_json(folder/'timing.json',dict(total_seconds=time.monotonic()-start,refit_baseline_exact=True,
        warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
    write_json(folder/'complete.json',dict(manifest=h,files={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('prepare','pilot','remaining','all'),default='all');a=p.parse_args()
    initialize();prepare()
    if a.phase=='prepare':return
    jobs=[t for t,_,_ in folds()]
    if a.phase=='pilot':run_fold(jobs[0]);return
    with ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(run_fold,jobs))
    import subprocess
    subprocess.run([sys.executable,str(HERE/'analyze.py')],check=True)

if __name__=='__main__':main()
