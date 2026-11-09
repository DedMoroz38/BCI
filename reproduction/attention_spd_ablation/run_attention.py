"""Fixed two-representation comparison, exact inherited fold/baseline protocol."""
import argparse,hashlib,importlib.util,json,os,sys,time,warnings
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=Path(os.environ.get('ATTENTION_OUTPUT',str(HERE/'results')))
os.environ['CLASSIFIER_OUTPUT']=str(OUT/'baseline_reproduction')
os.environ['CLASSIFIER_CACHE']=os.environ.get('ATTENTION_CACHE',str(HERE/'cache'))+'/baseline'
sys.path[:0]=[str(ROOT/'classifier_ablation'),str(HERE)]
from common import folds,draw_specs,metadata,sha,write_json,BANDS,control_path
from prepare_features import prepare
from classifiers import fit_classifier,balanced_weights
from bci.riemann import RiemannFeatures
from bci.evaluation import score
spec=importlib.util.spec_from_file_location('frozen_classifier_run',ROOT/'classifier_ablation/run.py')
baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
from attention import AttentionSPD,train_pack,encode_pack
MODES=['attention_gram','attention_exp']

def source_hash():
    files=sorted(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    files+=sorted((ROOT/'classifier_ablation').glob('*.py'))
    files+=sorted((ROOT/'Brain-Decoding/track2_bci/bci').rglob('*.py'))
    files+=[ROOT/'temporal_spd_research'/n for n in ['temporal_features.py','temporal_cache.py']]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()

def frozen_runtime(path,record):
    if path.exists():
        previous=json.loads(path.read_text())
        for key in ['source_sha256','input_sha256']:
            if previous[key]!=record[key]:raise RuntimeError('Changed experiment provenance: '+key)
        return previous
    write_json(path,record)
    return record

def fold_fingerprint(tag,train,test,specs):
    runtime=json.loads((OUT/'runtime.json').read_text())
    record=dict(source_sha256=source_hash(),input_sha256=runtime['input_sha256'],tag=tag,specs=specs)
    for part,windows in [('train',train),('eval',test)]:
        record[part]={k:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k,a in metadata(windows).items()}
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()

def band_inputs(windows):
    outputs=[]
    for band in BANDS:
        a=sosfiltfilt(butter(4,band,btype='bandpass',fs=120,output='sos'),windows.X.astype(float),axis=-1)
        a=np.stack([a[...,4:],a[...,:-4]],axis=-2).reshape(len(a),82,476).astype('float32')
        outputs.append(a)
    return outputs

def pack_inputs(bands,specs):
    return np.stack([bands[s['band']][:,s['matrix_channels']] for s in specs])

def validate_result(path,test,fingerprint):
    result=json.loads(path.read_text())
    assert result['source_sha256']==source_hash(),'Source changed during resume'
    assert result['fold_fingerprint']==fingerprint,'Fold/input provenance changed'
    p=path.with_name(path.stem+'_predictions.npz');assert sha(p)==result['predictions_sha256']
    with np.load(p) as data:
        for k,a in metadata(test).items():np.testing.assert_array_equal(a,data[k])
        for k,v in baseline.metrics(data,data['proba']).items():
            np.testing.assert_allclose(result[k],v,atol=1e-14,rtol=0)
    return result

def run_variant(mode,tag,train,test,train_bands,eval_bands,specs,device):
    out=OUT/mode;out.mkdir(parents=True,exist_ok=True)
    target=out/(tag+'.json')
    fingerprint=fold_fingerprint(tag,train,test,specs)
    if target.exists():return validate_result(target,test,fingerprint)
    folder=out/tag;folder.mkdir(exist_ok=True)
    tree_predictions=[];all_history=[];warning_rows=[];begin=time.monotonic()
    for start in range(0,50,5):
        ids=list(range(start,start+5));these=specs[start:start+5]
        pack=folder/f'pack_{start:02d}.npz';receipt=folder/f'pack_{start:02d}.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            assert saved['source_sha256']==source_hash() and saved['pack_sha256']==sha(pack)
            assert saved['fold_fingerprint']==fingerprint,'Cannot mix different folds or inputs'
            assert saved['checkpoint_sha256']==sha(folder/f'pack_{start:02d}.pt')
            with np.load(pack) as data:tree_predictions.extend(data['tree_proba'])
            all_history.append(saved);warning_rows.extend(saved['warnings'])
            print('RESUME PACK',mode,tag,start,flush=True);continue
        x=pack_inputs(train_bands,these);xe=pack_inputs(eval_bands,these)
        bootstrap=np.array([[s['subject_counts'][p] for p in train.subject] for s in these],dtype=float)
        eff=np.stack([balanced_weights(train.y,w) for w in bootstrap])
        print('START PACK',mode,tag,start,'/50',flush=True)
        fitted=train_pack(x,train.y,eff,mode=mode,tree_ids=ids,device=device)
        model=fitted['model']
        rtrain=encode_pack(model,x);reval=encode_pack(model,xe)
        probs=[];priors=[];means=[];scales=[];coef=[];intercept=[];pack_warnings=[]
        with threadpool_limits(limits=1):
            for j,s in enumerate(these):
                f=RiemannFeatures(BANDS[s['band']],5)
                xt,_=f.fit_transform(train,rtrain[j]);xtest,_=f.transform(test,reval[j])
                keep=bootstrap[j]>0
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    clf=fit_classifier('baseline',xt[keep],train.y[keep],bootstrap[j,keep],train.subject[keep])
                pack_warnings.extend([dict(tree=ids[j],category=w.category.__name__,message=str(w.message)) for w in caught])
                probs.append(clf.predict_proba(xtest))
                priors.append(f.log_prior);means.append(f.x_scaler.mean_);scales.append(f.x_scaler.scale_)
                coef.append(clf.coef_);intercept.append(clf.intercept_)
                # A future evaluation window cannot affect a previous prediction.
                prefix=test.select(np.arange(len(test))<min(5,len(test)))
                xp,_=f.transform(prefix,reval[j,:len(prefix)])
                np.testing.assert_allclose(xp,xtest[:len(prefix)],rtol=1e-8,atol=1e-8)
        np.savez_compressed(pack,tree_proba=np.stack(probs),log_prior=np.stack(priors),
                            scaler_mean=np.stack(means),scaler_scale=np.stack(scales),
                            coef=np.stack(coef),intercept=np.stack(intercept))
        checkpoint=folder/f'pack_{start:02d}.pt'
        torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},
                        auxiliary=fitted['auxiliary'],tree_ids=ids,mode=mode),checkpoint)
        record=dict(source_sha256=source_hash(),fold_fingerprint=fingerprint,tree_ids=ids,specs=these,
                    history=fitted['history'],training_seconds=fitted['seconds'],
                    warnings=pack_warnings,pack_sha256=sha(pack),checkpoint_sha256=sha(checkpoint))
        write_json(receipt,record);tree_predictions.extend(probs);all_history.append(record)
        warning_rows.extend(pack_warnings)
        print('DONE PACK',mode,tag,start+5,'/50','training_seconds',round(fitted['seconds'],1),flush=True)
        del model,fitted,x,xe,rtrain,reval
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    tree_p=np.stack(tree_predictions);p=tree_p.mean(0);p/=p.sum(1,keepdims=True)
    assert tree_p.shape==(50,len(test),3) and np.isfinite(p).all()
    assert (p>=0).all() and np.allclose(p.sum(1),1)
    meta=metadata(test)
    computed=baseline.metrics(meta,p)
    for k,v in score(test,p.argmax(1)).items():np.testing.assert_allclose(v,computed[k],atol=1e-14,rtol=0)
    path=out/(tag+'_predictions.npz')
    np.savez_compressed(path,**meta,pred=p.argmax(1),proba=p,tree_proba=tree_p)
    result=dict(variant=mode,tag=tag,source_sha256=source_hash(),fold_fingerprint=fingerprint,specs=specs,
                train_subjects=sorted(set(train.subject)),eval_subjects=sorted(set(test.subject)),
                predictions_sha256=sha(path),seconds=time.monotonic()-begin,
                histories=all_history,warnings=warning_rows,**computed)
    write_json(target,result)
    print('DONE VARIANT',mode,tag,json.dumps(computed),flush=True)
    return result

def freeze_selection():
    records={};hashes={}
    for name in ['baseline',*MODES]:
        directory=OUT/'baseline_reproduction/baseline' if name=='baseline' else OUT/name
        paths=sorted(directory.glob('loso_*.json'))
        assert len(paths)==8
        records[name]=float(np.mean([json.loads(p.read_text())['session_mean'] for p in paths]))
        for p in paths+[directory/'later_last.json']:
            assert p.exists();hashes[str(p.relative_to(OUT))]=sha(p)
    selection=dict(scores=records,winner=max(records,key=records.get),source_sha256=source_hash(),cv_hashes=hashes,
                   selection_metric='mean LOSO session balanced accuracy',test_previously_seen=True)
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==selection
    write_json(path,selection);print('FROZEN ATTENTION SELECTION',json.dumps(records),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    frozen_runtime(OUT/'runtime.json',dict(source_sha256=source_hash(),torch=torch.__version__,
              device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              input_sha256={s:sha(ROOT/f'data/proteus_{s}.npz') for s in ['train','val','test']}))
    for tag,protocol,fold,train,test in folds():
        if protocol=='test':freeze_selection()
        prepare(tag,protocol,fold,train,test)
        baseline.run_classical('baseline',tag)
        specs=draw_specs(train)
        # No cached learned representation can cross an outer fold.
        train_bands=band_inputs(train);eval_bands=band_inputs(test)
        for mode in MODES:run_variant(mode,tag,train,test,train_bands,eval_bands,specs,args.device)
        del train_bands,eval_bands
    print('ATTENTION EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':main()
