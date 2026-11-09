"""Run old Lag4 in its own interpreter; prohibit collisions with new bci modules."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'Brain-Decoding/track2_bci'),str(ROOT/'temporal_spd_research')]
from bci.data import concat,load_split
from bci.evaluation import leave_one_subject_out,later_sessions,score
from temporal_features import fit_model
from temporal_cache import MatrixCache

def main():
    tag,phase,mhash=sys.argv[1:]
    folder=HERE/'results'/tag;out=folder/'baseline.json'
    if out.exists():
        r=json.loads(out.read_text())
        if r['manifest_sha256']!=mhash:raise RuntimeError('Changed baseline manifest')
        if hashlib.sha256((folder/'baseline.npz').read_bytes()).hexdigest()!=r['prediction_sha256']:raise RuntimeError('Changed baseline predictions')
        return
    parts={s:load_split(s,ROOT/'data') for s in ('train','val','test')}
    dev=concat(parts['train'],parts['val'])
    if phase=='test':tr,te=parts['train'],parts['test'];suffix='test_official'
    elif tag=='later_last':_,tr,te=next(later_sessions(dev));suffix='later_last'
    else:
        s,tr,te=next(f for f in leave_one_subject_out(dev) if f[0].endswith('/'+tag[5:]))
        suffix='loso_'+s.replace('/','_')
    reference=ROOT/'temporal_spd_research/results/lag4_oas'/f'forest20_{suffix}_predictions.npz'
    start=time.monotonic()
    with threadpool_limits(limits=1):
        model=fit_model(tr,'forest20','lag4_oas',cache=MatrixCache(ROOT/'temporal_spd_research/cache'))
        fit_seconds=time.monotonic()-start
        p=model.tree_probas(te).mean(0)
        with np.load(reference) as old:
            for k in ('y','subject','session','context','run','onset'):np.testing.assert_array_equal(getattr(te,k),old[k])
            np.testing.assert_array_equal(p.argmax(1),old['pred'])
        # Live B=1 contract vs batched feature inference for first stream prefix.
        idx=next(te.streams())[:3];model.reset_state()
        for i in idx:np.testing.assert_array_equal(model.predict(te.X[i:i+1]),p[i:i+1].argmax(1))
    np.savez_compressed(folder/'baseline.npz',proba=p,pred=p.argmax(1),**{k:getattr(te,k) for k in ('y','subject','session','context','run','onset')})
    np.savez_compressed(folder/'baseline-weights.npz',
                        coef=np.stack([t.clf.coef_ for t in model.trees]),
                        intercept=np.stack([t.clf.intercept_ for t in model.trees]),
                        channels=np.stack([t.channels for t in model.trees]),
                        band=np.array([t.band for t in model.trees]),
                        log_prior=np.stack([t.features.log_prior for t in model.trees]),
                        scaler_mean=np.stack([t.features.x_scaler.mean_ for t in model.trees]),
                        scaler_scale=np.stack([t.features.x_scaler.scale_ for t in model.trees]))
    r=dict(variant='Lag4',tag=tag,manifest_sha256=mhash,**score(te,p.argmax(1)),fit_seconds=fit_seconds,
           total_seconds=time.monotonic()-start,exact_old_predictions=True,
           log_loss=float(-np.log(np.maximum(p[np.arange(len(te)),te.y],1e-15)).mean()),
           brier=float(np.mean(np.sum((p-np.eye(3)[te.y])**2,axis=1))),
           prediction_sha256=hashlib.sha256((folder/'baseline.npz').read_bytes()).hexdigest())
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(r,indent=2));tmp.replace(out)
    print('BASELINE',tag,r['cell_mean'],r['total_seconds'],flush=True)

if __name__=='__main__':main()
