"""Three SINGLE models, full-gradient training, original10 folds, no forest."""
import argparse,hashlib,importlib.metadata,json,os,sys,time,warnings
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=Path(os.environ.get('SINGLE_OUTPUT',str(HERE/'results')))
sys.path[:0]=[str(ROOT/'classifier_ablation'),str(ROOT/'temporal_spd_research')]
from common import folds,metadata,sha,write_json
from temporal_features import TemporalFeatures,fit_model
from temporal_cache import MatrixCache
from bci.evaluation import score
from bci.riemann import SessionRecentering
from single_e2e import SingleE2E,fit_single
MODES=['baseline','attention_gram','attention_exp']

def source_hash():
    files=[*HERE.glob('*.py'),HERE/'PROTOCOL.md',ROOT/'attention_spd_ablation/attention.py']
    files+=list((ROOT/'classifier_ablation').glob('*.py'))
    files+=list((ROOT/'Brain-Decoding/track2_bci/bci').rglob('*.py'))
    files+=[ROOT/'temporal_spd_research'/n for n in ['temporal_features.py','temporal_cache.py']]
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},sort_keys=True).encode()).hexdigest()

def fingerprint(tag,train,test):
    runtime=json.loads((OUT/'runtime.json').read_text())
    payload=dict(source_sha256=source_hash(),input_sha256=runtime['input_sha256'],versions=runtime['versions'],tag=tag)
    for name,w in [('train',train),('eval',test)]:
        payload[name]={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in metadata(w).items()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def check_runtime(path,runtime):
    if path.exists():
        previous=json.loads(path.read_text())
        for key in ['source_sha256','input_sha256','versions']:
            if runtime[key]!=previous[key]:raise RuntimeError('Changed resume provenance: '+key)
    else:write_json(path,runtime)

def band_inputs(w):
    a=sosfiltfilt(butter(4,(4,40),btype='bandpass',fs=120,output='sos'),w.X.astype(float),axis=-1)
    return np.stack([a[...,4:],a[...,:-4]],axis=-2).reshape(len(a),82,476).astype('float32')

def metrics(w,p):
    if p.shape!=(len(w),3) or not np.isfinite(p).all() or (p<0).any():raise RuntimeError('Invalid probabilities')
    np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
    return dict(**score(w,p.argmax(1)),log_loss=float(log_loss(w.y,p,labels=[0,1,2])),
                brier=float(np.square(p-np.eye(3)[w.y]).sum(1).mean()))

def save_result(mode,tag,train,test,p,details):
    out=OUT/mode;out.mkdir(parents=True,exist_ok=True)
    prediction=out/(tag+'_predictions.npz')
    np.savez_compressed(prediction,**metadata(test),pred=p.argmax(1),proba=p)
    r=dict(model=mode,tag=tag,source_sha256=source_hash(),fingerprint=fingerprint(tag,train,test),
           train_subjects=sorted(set(train.subject)),eval_subjects=sorted(set(test.subject)),
           training_windows=len(train),eval_windows=len(test),predictions_sha256=sha(prediction),
           **metrics(test,p),**details)
    write_json(out/(tag+'.json'),r)
    print('DONE SINGLE',mode,tag,'session',r['session_mean'],'cell',r['cell_mean'],'seconds',r['seconds'],flush=True)
    return r

def resume(mode,tag,train,test):
    path=OUT/mode/(tag+'.json')
    if not path.exists():return False
    r=json.loads(path.read_text());assert r['fingerprint']==fingerprint(tag,train,test),'Changed experiment provenance'
    p=path.with_name(path.stem+'_predictions.npz');assert sha(p)==r['predictions_sha256']
    if mode!='baseline':assert sha(path.with_suffix('.pt'))==r['checkpoint_sha256']
    else:assert sha(path.with_name(path.stem+'_weights.npz'))==r['checkpoint_sha256']
    with np.load(p) as saved:
        for key,value in metadata(test).items():np.testing.assert_array_equal(value,saved[key])
        for key,value in metrics(test,saved['proba']).items():np.testing.assert_allclose(r[key],value,atol=1e-14,rtol=0)
    print('RESUME SINGLE',mode,tag,flush=True);return True

def run_baseline(tag,protocol,fold,train,test):
    if resume('baseline',tag,train,test):return
    begin=time.monotonic();cache=MatrixCache(ROOT/'temporal_spd_research/cache')
    rtrain=cache.lookup('lag4_oas',(4,40),train.X);reval=cache.lookup('lag4_oas',(4,40),test.X)
    f=TemporalFeatures((4,40),5,representation='lag4_oas')
    with threadpool_limits(limits=1):
        xt,_=f.fit_transform(train,rtrain);xe,_=f.transform(test,reval)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            clf=LogisticRegression(C=.01,max_iter=2000,class_weight='balanced',random_state=0).fit(xt,train.y)
        p=clf.predict_proba(xe)
        original=fit_model(train,'logreg','lag4_oas',cache=cache)
    np.testing.assert_allclose(clf.coef_,original.clf.coef_,atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(clf.intercept_,original.clf.intercept_,atol=1e-8,rtol=1e-8)
    np.testing.assert_array_equal(p.argmax(1),original.clf.predict(xe))
    reference=ROOT/'temporal_spd_research/results/lag4_oas'/f'logreg_{protocol}_{fold.replace("/","_")}_predictions.npz'
    historical=False
    if reference.exists():
        with np.load(reference) as old:
            for k,v in metadata(test).items():np.testing.assert_array_equal(v,old[k])
            np.testing.assert_array_equal(p.argmax(1),old['pred'])
        historical=True
    out=OUT/'baseline';out.mkdir(parents=True,exist_ok=True)
    weights=out/(tag+'_weights.npz')
    np.savez_compressed(weights,log_prior=f.log_prior,scaler_mean=f.x_scaler.mean_,scaler_scale=f.x_scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
    return save_result('baseline',tag,train,test,p,dict(seconds=time.monotonic()-begin,
                exact_original_single_model=True,historical_prediction_match=historical,
                checkpoint_sha256=sha(weights),warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]))

def independent_inference(model,x,windows,p):
    with torch.no_grad():
        tx=torch.as_tensor(x,device=model.encoder.wq.device)
        r,_=model.matrices(tx);r=r.cpu().numpy()
    prior=model.log_prior.cpu().numpy();mean=model.feature_mean.cpu().numpy();scale=model.feature_scale.cpu().numpy()
    coef=model.head.weight.detach().cpu().numpy();intercept=model.head.bias.detach().cpu().numpy()
    independent=np.empty_like(p)
    with threadpool_limits(limits=1):
        for idx in windows.streams():
            state=SessionRecentering(prior,5)
            for i in idx:
                features=(state(r[i])[0]-mean)/scale
                independent[i]=softmax(features@coef.T+intercept)
    np.testing.assert_allclose(p,independent,atol=1e-6,rtol=1e-6)
    np.testing.assert_array_equal(p.argmax(1),independent.argmax(1))
    return float(np.max(np.abs(p-independent)))

def run_attention(mode,tag,train,test,x,xe,device):
    if resume(mode,tag,train,test):return
    begin=time.monotonic();fitted=fit_single(x,train.y,list(train.streams()),mode=mode,device=device)
    model=fitted['model'];p=model.predict_proba(xe,list(test.streams()))
    maximum=independent_inference(model,xe,test,p)
    first=min(len(test),3)
    prefix=test.select(np.arange(len(test))<first)
    np.testing.assert_allclose(model.predict_proba(xe[:first],list(prefix.streams())),p[:first],atol=1e-7,rtol=1e-7)
    out=OUT/mode;out.mkdir(parents=True,exist_ok=True);checkpoint=out/(tag+'.pt')
    torch.save(dict(mode=mode,channels=82,samples=476,state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    retained_head=True,fingerprint=fingerprint(tag,train,test)),checkpoint)
    loaded=torch.load(checkpoint,weights_only=True,map_location='cpu')
    restored=SingleE2E(mode,channels=82,samples=476).to(device);restored.load_state_dict(loaded['state'])
    np.testing.assert_allclose(restored.predict_proba(xe[:first],list(prefix.streams())),p[:first],atol=1e-7,rtol=1e-7)
    result=save_result(mode,tag,train,test,p,dict(seconds=time.monotonic()-begin,training_seconds=fitted['seconds'],
            epochs=80,history=fitted['history'],warnings=fitted['warnings'],checkpoint_sha256=sha(checkpoint),
            independent_numpy_inference_max_difference=maximum,retained_final_head=True,full_history_gradient=True))
    del model,restored,fitted
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result

def freeze_selection():
    scores={};hashes={}
    for mode in MODES:
        paths=sorted((OUT/mode).glob('loso_*.json'));assert len(paths)==8
        scores[mode]=float(np.mean([json.loads(p.read_text())['session_mean'] for p in paths]))
        for p in paths+[OUT/mode/'later_last.json']:
            assert p.exists();hashes[str(p.relative_to(OUT))]=sha(p)
    selection=dict(scores=scores,winner=max(scores,key=scores.get),metric='mean LOSO session BA',
                   source_sha256=source_hash(),cv_hashes=hashes,public_test_previously_seen=True)
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==selection
    write_json(path,selection);print('FROZEN SINGLE SELECTION',json.dumps(scores),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda');args=parser.parse_args()
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    runtime=dict(source_sha256=source_hash(),input_sha256={s:sha(ROOT/f'data/proteus_{s}.npz') for s in ['train','val','test']},
                 versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','pyriemann','pandas','torch']},
                 torch=torch.__version__,device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    check_runtime(OUT/'runtime.json',runtime)
    for tag,protocol,fold,train,test in folds():
        if protocol=='test':freeze_selection()
        print('START FOLD',tag,len(train),len(test),flush=True)
        run_baseline(tag,protocol,fold,train,test)
        x,xe=band_inputs(train),band_inputs(test)
        for mode in MODES[1:]:run_attention(mode,tag,train,test,x,xe,args.device)
        del x,xe
    print('SINGLE E2E EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':main()
