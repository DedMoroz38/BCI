"""Exact single Lag4 versus author iTransformer/no lags; nested person validation."""
import argparse,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=Path(os.environ.get('IT_OUTPUT',str(HERE/'results')))
os.environ['SINGLE_OUTPUT']=str(OUT)
sys.path.insert(0,str(ROOT/'single_attention_ablation'))
import run_single as lag
from training import subject_split,train_selection,refit,predict,write_json,digest
from classifier import EEGiTransformer
SEEDS=[0,1,2]


def source_hash():
    files=[*HERE.glob('*.py'),HERE/'PROTOCOL.md',*list((HERE/'vendor').rglob('*'))]
    files=[p for p in files if p.is_file() and '__pycache__' not in str(p)]
    files+=[*list((ROOT/'single_attention_ablation').glob('*.py')),ROOT/'single_attention_ablation/PROTOCOL.md',ROOT/'attention_spd_ablation/attention.py']
    files+=list((ROOT/'classifier_ablation').glob('*.py'))+list((ROOT/'Brain-Decoding/track2_bci/bci').rglob('*.py'))
    files+=[ROOT/'temporal_spd_research'/n for n in ['temporal_features.py','temporal_cache.py']]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in sorted(files)},sort_keys=True).encode()).hexdigest()

def verify_cache(directory):
    directory=Path(directory)
    names=['manifest.json','lag4_oas_band0.complete.json','lag4_oas_band0.npy']
    receipt=json.loads((directory/names[1]).read_text())
    values={name:digest(directory/name) for name in names}
    if values[names[2]]!=receipt['sha256']:raise RuntimeError('Changed baseline matrix cache')
    return values

def validate_runtime(path,runtime):
    if path.exists():
        previous=json.loads(path.read_text())
        for key in ['source_sha256','input_sha256','cache_sha256','versions','device','gpu']:
            if previous[key]!=runtime[key]:raise RuntimeError('Changed runtime provenance: '+key)
    else:write_json(path,runtime)

def partition_subjects(training_subjects,evaluation_subjects,protocol):
    if protocol=='loso' and set(training_subjects)&set(evaluation_subjects):
        raise RuntimeError('An outer evaluation person entered training')
    return subject_split(training_subjects)

def filtered(w):
    return sosfiltfilt(butter(4,(4,40),btype='bandpass',fs=120,output='sos'),w.X.astype(float),axis=-1).astype('float32')

def ce_metrics(w,p,weights):
    losses=-np.log(np.maximum(p[np.arange(len(w)),w.y],np.finfo(float).eps));sw=weights[w.y]
    return dict(ce=float(losses.mean()),weighted_ce=float((losses*sw).sum()/sw.sum()),
                session_ba=lag.score(w,p.argmax(1))['session_mean'],accuracy=float((p.argmax(1)==w.y).mean()))

def inner_baseline(train,val):
    cache=lag.MatrixCache(ROOT/'temporal_spd_research/cache');f=lag.TemporalFeatures((4,40),5,representation='lag4_oas')
    with threadpool_limits(limits=1):
        xt,_=f.fit_transform(train,cache.lookup('lag4_oas',(4,40),train.X))
        xv,_=f.transform(val,cache.lookup('lag4_oas',(4,40),val.X))
        clf=LogisticRegression(C=.01,max_iter=2000,class_weight='balanced',random_state=0).fit(xt,train.y)
    weights=len(train)/(3*np.bincount(train.y,minlength=3))
    return dict(train=ce_metrics(train,clf.predict_proba(xt),weights),val=ce_metrics(val,clf.predict_proba(xv),weights))

def run_neural(tag,protocol,train,test,device):
    inner_train,inner_val=partition_subjects(train.subject,test.subject,protocol)
    fitw,valw=train.select(inner_train),train.select(inner_val)
    assert not set(fitw.subject)&set(valw.subject)
    if protocol=='loso':assert not set(train.subject)&set(test.subject)
    x=filtered(train);xe=filtered(test)
    split=dict(train_subjects=sorted(set(fitw.subject)),validation_subjects=sorted(set(valw.subject)),
               outer_evaluation_subjects=sorted(set(test.subject)),inner_train_windows=len(fitw),inner_val_windows=len(valw),
               known_person_outer_overlap=sorted(set(valw.subject)&set(test.subject)))
    folder=OUT/'itransformer'/tag;folder.mkdir(parents=True,exist_ok=True)
    manifest=dict(runtime_sha256=digest(OUT/'runtime.json'),outer_fingerprint=lag.fingerprint(tag,train,test),split=split)
    path=folder/'job-manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Changed trial provenance'
    else:write_json(path,manifest)
    base_path=folder/'inner-baseline.json'
    if base_path.exists():reference=json.loads(base_path.read_text())
    else:reference=inner_baseline(fitw,valw);write_json(base_path,reference)
    for seed in SEEDS:
        trial=folder/f'seed{seed}';trial.mkdir(exist_ok=True)
        result_path=OUT/'itransformer'/f'{tag}_seed{seed}.json'
        if result_path.exists():
            r=json.loads(result_path.read_text());assert r['job_manifest_sha256']==digest(path)
            for rel,sha in r['artifact_sha256'].items():assert digest(OUT/rel)==sha
            with np.load(OUT/r['predictions_file']) as a:
                for k,v in lag.metadata(test).items():np.testing.assert_array_equal(a[k],v)
                for k,v in lag.metrics(test,a['proba']).items():np.testing.assert_allclose(r[k],v,atol=1e-14,rtol=0)
            print('RESUME IT',tag,seed,flush=True);continue
        begin=time.monotonic()
        selected=train_selection(x[inner_train],train.y[inner_train],fitw.stream,x[inner_val],train.y[inner_val],valw.stream,
                    seed=seed,folder=trial/'selection',device=device,baseline=reference)
        fitted=refit(x,train.y,train.stream,seed=seed,epochs=selected['best_epoch'],folder=trial/'refit',device=device)
        model=fitted['model'];p=predict(model,xe,device=device)
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        start=time.monotonic()
        stream_p=np.concatenate([predict(model,xe[i:i+1],device=device) for i in range(len(test))])
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        inference_seconds=time.monotonic()-start
        np.testing.assert_allclose(p,stream_p,atol=1e-5,rtol=1e-5)
        np.testing.assert_array_equal(p.argmax(1),stream_p.argmax(1))
        restored=EEGiTransformer().to(device)
        restored.load_state_dict(torch.load(trial/'refit/final.pt',map_location=device,weights_only=True)['state'])
        np.testing.assert_allclose(p[:3],predict(restored,xe[:3],device=device),atol=1e-5,rtol=1e-5)
        predictions=OUT/'itransformer'/f'{tag}_seed{seed}_predictions.npz'
        np.savez_compressed(predictions,**lag.metadata(test),pred=p.argmax(1),proba=p)
        files=[predictions,path,base_path,*list((trial/'selection').glob('*')),*list((trial/'refit').glob('*'))]
        result=dict(model='itransformer',tag=tag,seed=seed,selected_epoch=selected['best_epoch'],
                    stopped_epoch=selected['stopped_epoch'],split=split,job_manifest_sha256=digest(path),
                    predictions_file=str(predictions.relative_to(OUT)),seconds=time.monotonic()-begin,
                    inference_ms_per_window=1000*inference_seconds/len(test),parameter_count=selected['parameter_count'],
                    artifact_sha256={str(f.relative_to(OUT)):digest(f) for f in files if f.is_file()},**lag.metrics(test,p))
        write_json(result_path,result)
        print('DONE ITRANSFORMER',tag,'seed',seed,'epochs',result['selected_epoch'],'session',result['session_mean'],'cell',result['cell_mean'],flush=True)
        del model,restored,fitted
        if str(device).startswith('cuda'):torch.cuda.empty_cache()

def freeze_selection():
    hashes={};scores={}
    paths=sorted((OUT/'baseline').glob('loso_*.json'));assert len(paths)==8
    scores['baseline']=float(np.mean([json.loads(p.read_text())['session_mean'] for p in paths]))
    for p in paths+[OUT/'baseline/later_last.json']:assert p.exists();hashes[str(p.relative_to(OUT))]=digest(p)
    values=[]
    for seed in SEEDS:
        paths=sorted((OUT/'itransformer').glob(f'loso_*_seed{seed}.json'));assert len(paths)==8
        values.extend([json.loads(p.read_text())['session_mean'] for p in paths])
        for p in paths+[OUT/'itransformer'/f'later_last_seed{seed}.json']:assert p.exists();hashes[str(p.relative_to(OUT))]=digest(p)
    scores['itransformer']=float(np.mean(values))
    record=dict(scores=scores,winner=max(scores,key=scores.get),metric='mean LOSO session BA;neural mean over seeds,no prediction ensemble',
                cv_hashes=hashes,source_sha256=source_hash(),public_test_previously_seen=True)
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==record
    write_json(path,record);print('FROZEN IT SELECTION',json.dumps(scores),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    runtime=dict(source_sha256=source_hash(),cache_sha256=verify_cache(ROOT/'temporal_spd_research/cache'),input_sha256={s:digest(ROOT/f'data/proteus_{s}.npz') for s in ['train','val','test']},
                 versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn','pyriemann','pandas','einops']},
                 torch=torch.__version__,device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    validate_runtime(OUT/'runtime.json',runtime)
    for tag,protocol,fold,train,test in lag.folds():
        if protocol=='test':freeze_selection()
        print('START IT FOLD',tag,len(train),len(test),flush=True)
        lag.run_baseline(tag,protocol,fold,train,test)
        run_neural(tag,protocol,train,test,args.device)
    print('ITRANSFORMER EXPERIMENT COMPLETE',flush=True)
if __name__=='__main__':main()
