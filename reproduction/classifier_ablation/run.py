"""Seven fixed classifier variants on the submitted Lag4 forest's frozen features."""
import argparse,json,time,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from common import HERE,ROOT,OUT,CACHE,sha,write_json,folds,control_path
from classifiers import VARIANTS,fit_classifier,balanced_weights
from prepare_features import prepare
from bci.evaluation import balanced_accuracy


def config_hash():
    import hashlib
    files=['PROTOCOL.md','classifiers.py','gpu_mlp.py','common.py','prepare_features.py','run.py']
    return hashlib.sha256(json.dumps({n:sha(HERE/n) for n in files},sort_keys=True).encode()).hexdigest()


def validate_record(name,tag):
    path=OUT/name/(tag+'.json');record=json.loads(path.read_text())
    if record['classifier_config_sha256']!=config_hash():raise RuntimeError('Classifier configuration changed')
    if sha(CACHE/tag/'manifest.json')!=record['feature_manifest_sha256']:
        raise RuntimeError('Saved feature manifest changed')
    predictions=OUT/name/(tag+'_predictions.npz')
    if sha(predictions)!=record['predictions_sha256']:raise RuntimeError('Changed predictions')
    with np.load(predictions) as p, np.load(CACHE/tag/'meta.npz') as original:
        for key in ['y','subject','session','context','run','onset']:
            np.testing.assert_array_equal(p[key],original[key])
        np.testing.assert_array_equal(p['pred'],p['proba'].argmax(1))
        recomputed=metrics(p,p['proba'])
        for key,val in recomputed.items():
            if not np.isclose(val,record[key],atol=1e-14,rtol=0):raise RuntimeError('Saved metric mismatch '+key)
    return record


def metrics(meta,p):
    p=np.asarray(p);y=meta['y'];pred=p.argmax(1)
    stream=np.char.add(np.char.add(meta['subject'],'/'),meta['session'])
    cells=np.char.add(np.char.add(stream,'/'),meta['context'])
    def mean_group(groups):
        return float(np.mean([balanced_accuracy(y[groups==g],pred[groups==g]) for g in np.unique(groups)]))
    return dict(cell_mean=mean_group(cells),session_mean=mean_group(stream),
                pooled=balanced_accuracy(y,pred),accuracy=float(np.mean(y==pred)),
                log_loss=float(log_loss(y,p,labels=[0,1,2])),
                brier=float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))))


def save_result(name,tag,tree_p,details):
    folder=CACHE/tag;manifest=json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'meta.npz') as m:meta={k:m[k] for k in ['y','subject','session','context','run','onset']}
    if tree_p.shape!=(50,len(meta['y']),3) or not np.isfinite(tree_p).all():raise RuntimeError('Invalid probabilities')
    if tree_p.min()<-1e-12 or not np.allclose(tree_p.sum(-1),1,atol=1e-6):raise RuntimeError('Not simplex output')
    # Remove float32 normalization roundoff from MLP's output.
    p=tree_p.mean(0);p=p/p.sum(1,keepdims=True)
    result=dict(variant=name,tag=tag,protocol=manifest['protocol'],fold=manifest['fold'],
                train_subjects=manifest['train_subjects'],eval_subjects=manifest['eval_subjects'],
                train_windows=manifest['train_windows'],eval_windows=manifest['eval_windows'],
                feature_manifest_sha256=sha(folder/'manifest.json'),
                classifier_config_sha256=config_hash(),**metrics(meta,p),**details)
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(tag+'_predictions.npz'),**meta,pred=p.argmax(1),proba=p,tree_proba=tree_p)
    result['predictions_sha256']=sha(out/(tag+'_predictions.npz'))
    write_json(out/(tag+'.json'),result)
    print('DONE MODEL',name,tag,'session=',round(result['session_mean'],8),
          'cell=',round(result['cell_mean'],8),'seconds=',round(details['classifier_seconds'],1),flush=True)
    return result


def run_classical(name,tag):
    output=OUT/name/(tag+'.json')
    if output.exists():
        record=validate_record(name,tag)
        print('RESUME MODEL',name,tag,flush=True);return record
    folder=CACHE/tag;manifest=json.loads((folder/'manifest.json').read_text())
    xt=np.load(folder/'train.npy',mmap_mode='r');xe=np.load(folder/'eval.npy',mmap_mode='r')
    with np.load(folder/'meta.npz') as m:y=m['train_y'];weights=m['weights'];groups=m['train_subject']
    probabilities=[];warning_rows=[];max_coef=0.;begin=time.monotonic()
    exported=np.load(ROOT/'submission_lag4/package/weights.npz') if name=='baseline' and manifest['protocol']=='test' else None
    with threadpool_limits(limits=1):
        for k in range(50):
            keep=weights[k]>0
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                clf=fit_classifier(name,xt[k,keep],y[keep],weights[k,keep],groups[keep])
            for w in caught:
                warning_rows.append(dict(tree=k,category=w.category.__name__,message=str(w.message)))
            np.testing.assert_array_equal(clf.classes_,[0,1,2])
            probabilities.append(clf.predict_proba(xe[k]))
            if exported is not None:
                max_coef=max(max_coef,float(np.max(np.abs(clf.coef_-exported['coef'][k]))),
                             float(np.max(np.abs(clf.intercept_-exported['intercept'][k]))))
            if (k+1)%10==0:print('CLASSIFIER',name,tag,k+1,'/50',flush=True)
    tree_p=np.stack(probabilities)
    details=dict(classifier_seconds=time.monotonic()-begin,warnings=warning_rows,
                 convergence_warning_count=sum(r['category']=='ConvergenceWarning' for r in warning_rows),
                 backend='sklearn CPU')
    if name=='baseline':
        old=control_path(manifest['protocol'],manifest['fold'])
        with np.load(old.with_name(old.stem+'_predictions.npz')) as ref:
            np.testing.assert_array_equal(tree_p.mean(0).argmax(1),ref['pred'])
        details['exact_reference_labels']=True
        if exported is not None:
            geom=np.load(folder/'geometry.npz')
            geom_diffs={k:float(np.max(np.abs(geom[k]-exported[k]))) for k in ['log_prior','scaler_mean','scaler_scale']}
            if max_coef>1e-6 or max(geom_diffs.values())>1e-7:raise RuntimeError(f'Submission parameters differ {max_coef} {geom_diffs}')
            details['submission_max_classifier_parameter_diff']=max_coef
            details['submission_geometry_max_diffs']=geom_diffs
            exported.close()
    return save_result(name,tag,tree_p,details)


def run_mlp(tag):
    output=OUT/'mlp'/(tag+'.json')
    if output.exists():
        return validate_record('mlp',tag)
    from gpu_mlp import fit_predict
    folder=CACHE/tag;xt=np.load(folder/'train.npy',mmap_mode='r');xe=np.load(folder/'eval.npy',mmap_mode='r')
    with np.load(folder/'meta.npz') as m:y=m['train_y'];weights=m['weights']
    eff=np.stack([balanced_weights(y,w) for w in weights])
    result=fit_predict(xt,y,eff,xe,device='cuda',epochs=300)
    param_dir=OUT/'mlp';param_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(param_dir/(tag+'_weights.npz'),**result['parameters'])
    return save_result('mlp',tag,result['tree_proba'],dict(classifier_seconds=result['fit_predict_seconds'],
                       warnings=[],convergence_warning_count=0,backend='PyTorch CUDA',epochs=300,
                       optimizer='Adam lr=.001; fixed epochs, no convergence claim',loss_history=result['loss_history']))


def freeze_selection():
    scores={};hashes={}
    for name in VARIANTS:
        records=[]
        for p in sorted((OUT/name).glob('loso_*.json')):
            r=validate_record(name,p.stem);records.append(r['session_mean']);hashes[str(p.relative_to(OUT))]=sha(p)
        if len(records)!=8:raise RuntimeError(f'Incomplete LOSO {name}: {len(records)}')
        later=OUT/name/'later_last.json'
        if not later.exists():raise RuntimeError('Incomplete later-session evaluation')
        validate_record(name,'later_last');hashes[str(later.relative_to(OUT))]=sha(later)
        scores[name]=float(np.mean(records))
    winner=sorted(scores,key=lambda n:(-scores[n],n))[0]
    result=dict(metric='mean session balanced accuracy over8 LOSO subjects',scores=scores,winner=winner,
                cv_result_sha256=hashes,protocol_sha256=sha(HERE/'PROTOCOL.md'),
                selection_before_new_public_test=True,public_test_already_seen_in_previous_research=True)
    path=OUT/'selection.json'
    if path.exists() and json.loads(path.read_text())!=result:raise RuntimeError('Frozen selection changed')
    write_json(path,result);print('FROZEN SELECTION',json.dumps(scores),'winner',winner,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','cv','test'],default='all')
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    print('CLASSIFIER EXPERIMENT START',args.phase,flush=True)
    if args.phase=='test':freeze_selection()
    # The saved protocol and source bundle are frozen before any model scores.
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        jobs=[]
        for tag,protocol,fold,train,test in folds(args.phase):
            if protocol=='test':
                # All CV classifiers must finish before selection and test work.
                for job in as_completed(jobs):job.result()
                jobs=[];freeze_selection()
            prepare(tag,protocol,fold,train,test)
            run_classical('baseline',tag)
            jobs += [pool.submit(run_classical,n,tag) for n in ['elasticnet','svm_linear','svm_rbf','lda_lw','ridge']]
            run_mlp(tag)
        for job in as_completed(jobs):job.result()
    if args.phase in ['all','cv']:freeze_selection()
    print('CLASSIFIER EXPERIMENT COMPLETE',args.phase,flush=True)

if __name__=='__main__':main()
