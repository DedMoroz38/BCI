"""Full-public-cohort Lag4 controls,computed locally on the verified cache."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,sys,time,warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'Brain-Decoding/track2_bci'),str(ROOT/'temporal_spd_research'),str(HERE/'src')]
import bci
from common import folds,sha,write_json,save_prediction,balance
from temporal_features import fit_model
from temporal_cache import MatrixCache
OUT=HERE/'experiments/all10_baseline/results'

def job(tag):
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    code={str(p.relative_to(HERE)):sha(p) for p in (Path(__file__),HERE/'src/common.py')}
    if (folder/'complete.json').exists():
        import json
        previous=json.loads((folder/'complete.json').read_text())
        if previous['code']!=code:raise RuntimeError('Changed baseline fitting code')
        for name,digest in previous['files'].items():
            if sha(folder/name)!=digest:raise RuntimeError('Changed completed baseline')
        return tag
    _,tr,te=next(j for j in folds('all10') if j[0]==tag);start=time.monotonic()
    print('ALL10_START',tag,len(tr),len(te),flush=True)
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=fit_model(tr,'forest20','lag4_oas',cache=MatrixCache(ROOT/'temporal_spd_research/cache'))
        tree=model.tree_probas(te);p=tree.mean(0);p=p/p.sum(1,keepdims=True)
    save_prediction(folder/'lag4.npz',te,p,tag=tag,train_people=sorted(set(tr.subject)),fit_seconds=time.monotonic()-start)
    save_prediction(folder/'lag4_balance1.npz',te,balance(p,list(te.streams()),1),tag=tag)
    np.savez_compressed(folder/'weights.npz',coef=np.stack([t.clf.coef_ for t in model.trees]),intercept=np.stack([t.clf.intercept_ for t in model.trees]),
                        band=np.array([t.band for t in model.trees]),channels=np.stack([t.channels for t in model.trees]),
                        log_prior=np.stack([t.features.log_prior for t in model.trees]),scaler_mean=np.stack([t.features.x_scaler.mean_ for t in model.trees]),scaler_scale=np.stack([t.features.x_scaler.scale_ for t in model.trees]))
    write_json(folder/'complete.json',dict(code=code,seconds=time.monotonic()-start,warnings=[str(w.message) for w in caught],files={p.name:sha(p) for p in folder.iterdir() if p.name!='complete.json'}))
    print('ALL10_DONE',tag,time.monotonic()-start,flush=True);return tag

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    import json
    expected=json.loads((ROOT/'results/eeg/manifest.json').read_text())['data_sha256']
    for split,digest in expected.items():assert sha(ROOT/'data'/f'proteus_{split}.npz')==digest
    for k in range(6):
        import json
        path=ROOT/'temporal_spd_research/cache'/f'lag4_oas_band{k}.npy';receipt=json.loads(path.with_suffix('.complete.json').read_text())
        assert sha(path)==receipt['sha256']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(job,[t for t,_,_ in folds('all10')]))
    print('ALL10_BASELINES_COMPLETE',flush=True)

if __name__=='__main__':main()
