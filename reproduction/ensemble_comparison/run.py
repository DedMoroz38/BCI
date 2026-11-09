"""Pinned local replication, with unchanged upstream training and inference."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,hashlib,importlib.metadata,json,platform,subprocess,sys,time,warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
AUTHOR=ROOT/'ensemble_author_20261108'
sys.path.insert(0,str(AUTHOR/'track2_bci'))
from bci.data import concat,load_split
from bci.evaluation import leave_one_subject_out,later_sessions,score
from bci.ensemble import VARIANTS,fit_components,fit_deep,assemble
from bci.stream import save_spec,load_spec,StreamModel
COMMIT='934f423a73e1ed50d653345d27256ecfe50ab009'
OUT=HERE/'results'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)

def immutable_json(p,v):
    p=Path(p)
    if p.exists():
        if json.loads(p.read_text())!=v:raise RuntimeError(f'Changed immutable artifact: {p}')
    else:write_json(p,v)

def metadata(w):
    return {k:getattr(w,k) for k in ('y','subject','session','context','run','onset')}

def check_metadata(w,values):
    for k,v in metadata(w).items():np.testing.assert_array_equal(v,values[k],err_msg=k)

def folds(phase):
    parts={s:load_split(s,ROOT/'data') for s in ('train','val','test')}
    dev=concat(parts['train'],parts['val'])
    if phase=='test':return [('test_official',parts['train'],parts['test'])]
    return [(f'loso_{s.split("/")[-1]}',tr,te) for s,tr,te in leave_one_subject_out(dev)]+[
        ('later_last',tr,te) for _,tr,te in later_sessions(dev)]

def probabilities(model,w):
    out=np.empty((len(w),3))
    for idx in w.streams():
        m=copy.deepcopy(model);m.reset_state()
        for i in idx:out[i]=m.proba(w.X[i])
    if not np.isfinite(out).all() or np.any(out<0):raise RuntimeError('Invalid probabilities')
    np.testing.assert_allclose(out.sum(1),1,atol=1e-10)
    return out

def measures(w,p):
    pred=p.argmax(1)
    return dict(score(w,pred),log_loss=float(-np.log(np.maximum(p[np.arange(len(w)),w.y],1e-15)).mean()),
                brier=float(np.mean(np.sum((p-np.eye(3)[w.y])**2,axis=1))))

def baseline_reference(tag):
    if tag.startswith('loso_'):suffix='loso_Dreyer2026Proteus_'+tag[5:]
    elif tag=='later_last':suffix='later_last'
    else:suffix='test_official'
    return ROOT/'temporal_spd_research/results/lag4_oas'/('forest20_'+suffix)

def initialize():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=AUTHOR,text=True).strip()
    if head!=COMMIT or subprocess.check_output(['git','diff','HEAD'],cwd=AUTHOR):
        raise RuntimeError('Changed author source')
    files=[HERE/x for x in ('run.py','baseline.py','analyze.py','PROTOCOL.md')]
    files+=list((AUTHOR/'track2_bci/bci').rglob('*.py'))
    files += list((ROOT/'Brain-Decoding/track2_bci/bci').rglob('*.py'))
    files += [ROOT/'temporal_spd_research/temporal_features.py',ROOT/'temporal_spd_research/temporal_cache.py',
              ROOT/'temporal_spd_research/cache/manifest.json']
    hashes={s:sha(ROOT/'data'/f'proteus_{s}.npz') for s in ('train','val','test')}
    expected=json.loads((ROOT/'results/eeg/manifest.json').read_text())['data_sha256']
    if hashes!=expected:raise RuntimeError('Data changed')
    receipts={}
    for k in range(6):
        p=ROOT/'temporal_spd_research/cache'/f'lag4_oas_band{k}.npy'
        receipt=p.with_suffix('.complete.json')
        files.append(receipt)
        digest=sha(p)
        if digest!=json.loads(receipt.read_text())['sha256']:raise RuntimeError('Cache changed')
        receipts[p.name]=digest
    refs={}
    for phase in ('cv','test'):
        for tag,tr,te in folds(phase):
            ref=baseline_reference(tag)
            with np.load(str(ref)+'_predictions.npz') as f:check_metadata(te,f)
            r=json.loads(ref.with_suffix('.json').read_text())
            if sorted(set(tr.subject))!=r['train_subjects']:raise RuntimeError('Train people changed')
            refs[tag]={ext:sha(str(ref)+ext) for ext in ('.json','_predictions.npz')}
    manifest=dict(author_commit=head,data=hashes,cache=receipts,references=refs,
                  code={str(p.relative_to(ROOT)):sha(p) for p in files},
                  versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','torch','pyriemann')},
                  python=sys.version,platform=platform.platform(),variants=VARIANTS,
                  config=dict(n_trees=50,n_channels=20,n_subject_trees=10,C=.01,prior_weight=5,
                              seed=0,deep_seeds=[0],deep_epochs=100,torch_threads=2,blas_threads=1))
    immutable_json(HERE/'manifest.json',manifest)
    return sha(HERE/'manifest.json')

def result(folder,variant,w,p,elapsed,manifest_hash):
    slug=list(VARIANTS).index(variant)
    name=f'variant{slug}'
    np.savez_compressed(folder/(name+'.npz'),**metadata(w),proba=p,pred=p.argmax(1))
    r=dict(variant=variant,tag=folder.name,manifest_sha256=manifest_hash,
           prediction_sha256=sha(folder/(name+'.npz')),eval_seconds=elapsed,**measures(w,p))
    write_json(folder/(name+'.json'),r)
    print('DONE',folder.name,variant,'cell',r['cell_mean'],'session',r['session_mean'],flush=True)

def check_done(folder,index,manifest_hash):
    f=folder/f'variant{index}.json'
    if not f.exists():return False
    r=json.loads(f.read_text());p=folder/f'variant{index}.npz'
    if r['manifest_sha256']!=manifest_hash or sha(p)!=r['prediction_sha256']:
        raise RuntimeError('Changed completed result')
    return True

def run_fold(tag,phase,manifest_hash):
    torch.set_num_threads(2)
    _,train,test=next(f for f in folds(phase) if f[0]==tag)
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    print('START',tag,len(train),len(test),flush=True)
    with (folder/'baseline.log').open('a') as log:
        subprocess.run([sys.executable,str(HERE/'baseline.py'),tag,phase,manifest_hash],stdout=log,stderr=subprocess.STDOUT,check=True)
    if all(check_done(folder,k,manifest_hash) for k in range(7)):return tag
    def save_parts(parts):
        save_spec(parts,folder/'parts.tmp.npz');(folder/'parts.tmp.npz').replace(folder/'parts.npz')
        write_json(folder/'parts.sha.json',{'sha256':sha(folder/'parts.npz'),'manifest':manifest_hash})
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1,user_api='blas'):
        warnings.simplefilter('always')
        if (folder/'parts.npz').exists():
            receipt=json.loads((folder/'parts.sha.json').read_text())
            if receipt!={'sha256':sha(folder/'parts.npz'),'manifest':manifest_hash}:raise RuntimeError('Changed components')
            parts=load_spec(folder/'parts.npz')
        else:
            start=time.monotonic()
            switches=[s for s in VARIANTS.values() if not s.get('deep')]
            parts=fit_components(train,switches)
            save_parts(parts)
            write_json(folder/'nondeep-fit.json',{'seconds':time.monotonic()-start})
        for variant,switches in VARIANTS.items():
            k=list(VARIANTS).index(variant)
            if switches.get('deep') or check_done(folder,k,manifest_hash):continue
            spec,model=assemble(parts,switches,train.ch_names)
            start=time.monotonic();p=probabilities(model,test)
            save_spec(spec,folder/f'model{k}.npz')
            # Exercise actual exported model contract, not only internal proba.
            idx=next(test.streams())[:3]
            prefix=test.select(np.isin(np.arange(len(test)),idx))
            restored=StreamModel(load_spec(folder/f'model{k}.npz'))
            np.testing.assert_array_equal(p[idx],probabilities(restored,prefix))
            restored.reset_state()
            np.testing.assert_array_equal(p[idx].argmax(1),restored.predict(torch.from_numpy(prefix.X)))
            result(folder,variant,test,p,time.monotonic()-start,manifest_hash)
        if 'deep' not in parts:
            print('DEEP START',tag,'100 epochs CPU',flush=True)
            start=time.monotonic()
            parts['deep']=fit_deep(train,device='cpu')
            save_parts(parts)
            write_json(folder/'deep-fit.json',{'seconds':time.monotonic()-start})
            print('DEEP FINISHED',tag,time.monotonic()-start,flush=True)
        for variant in ('+deep','all'):
            k=list(VARIANTS).index(variant)
            if check_done(folder,k,manifest_hash):continue
            spec,model=assemble(parts,VARIANTS[variant],train.ch_names)
            start=time.monotonic();p=probabilities(model,test)
            save_spec(spec,folder/f'model{k}.npz')
            idx=next(test.streams())[:3];prefix=test.select(np.isin(np.arange(len(test)),idx))
            restored=StreamModel(load_spec(folder/f'model{k}.npz'))
            np.testing.assert_array_equal(p[idx],probabilities(restored,prefix))
            restored.reset_state()
            np.testing.assert_array_equal(p[idx].argmax(1),restored.predict(torch.from_numpy(prefix.X)))
            result(folder,variant,test,p,time.monotonic()-start,manifest_hash)
        write_json(folder/'warnings.json',[{'category':w.category.__name__,'message':str(w.message)} for w in caught])
    return tag

def worker(args):return run_fold(*args)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args();h=initialize()
    for phase in ('cv','test'):
        if phase=='test':
            completed={str(p.relative_to(HERE)):sha(p) for p in OUT.glob('*/variant*.json') if p.parent.name!='test_official'}
            if len(completed)!=63:raise RuntimeError('Need all63 author CV scores before test')
            immutable_json(HERE/'cv-freeze.json',dict(manifest=h,results=completed))
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(worker,[(tag,phase,h) for tag,_,_ in folds(phase)]))
    subprocess.run([sys.executable,str(HERE/'analyze.py')],check=True)

if __name__=='__main__':main()
