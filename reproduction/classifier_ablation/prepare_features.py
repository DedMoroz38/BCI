"""Build one common exact feature cache per outer fold,without fitting classifiers."""
import json,time
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,ROOT,CACHE,BANDS,sha,write_json,draw_specs,tree_features,control_path,metadata
from temporal_cache import MatrixCache


def prepare(tag,protocol,fold,train,test):
    folder=CACHE/tag;folder.mkdir(parents=True,exist_ok=True)
    reference=control_path(protocol,fold)
    frozen=json.loads(reference.read_text())
    if sorted(set(train.subject))!=frozen['train_subjects'] or sorted(set(test.subject))!=frozen['eval_subjects']:
        raise RuntimeError('Outer subject split changed')
    with np.load(reference.with_name(reference.stem+'_predictions.npz')) as old:
        for key,a in metadata(test).items():np.testing.assert_array_equal(a,old[key])
    specs=draw_specs(train)
    if [dict(band=s['band'],channels=s['channels']) for s in specs]!=frozen['trees']:
        raise RuntimeError('Random tree draws differ from submitted baseline')
    geometry_hashes={str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'Brain-Decoding/track2_bci/bci/riemann.py',
        ROOT/'temporal_spd_research/temporal_features.py',
        ROOT/'temporal_spd_research/temporal_cache.py']}
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['geometry_sha256']!=geometry_hashes or previous['specs']!=specs:
            raise RuntimeError('Cannot resume changed geometry')
        for name,digest in previous['files_sha256'].items():
            if sha(folder/name)!=digest:raise RuntimeError('Changed feature cache '+name)
        print('RESUME FEATURES',tag,flush=True);return folder
    print('START FEATURES',tag,len(train),len(test),flush=True)
    begin=time.monotonic();cache=MatrixCache(ROOT/'temporal_spd_research/cache')
    covs_train=[cache.lookup('lag4_oas',band,train.X) for band in BANDS]
    covs_test=[cache.lookup('lag4_oas',band,test.X) for band in BANDS]
    xt=np.lib.format.open_memmap(folder/'train.npy',mode='w+',dtype='float64',shape=(50,len(train),820))
    xe=np.lib.format.open_memmap(folder/'eval.npy',mode='w+',dtype='float64',shape=(50,len(test),820))
    counts=[];priors=[];means=[];scales=[]
    with threadpool_limits(limits=1):
        for k,spec in enumerate(specs):
            f,a,b=tree_features(train,test,spec,cache,
                               cov_train=covs_train[spec['band']],cov_eval=covs_test[spec['band']])
            xt[k]=a;xe[k]=b
            counts.append([spec['subject_counts'][s] for s in train.subject])
            priors.append(f.log_prior);means.append(f.x_scaler.mean_);scales.append(f.x_scaler.scale_)
            if (k+1)%10==0:print('FEATURES',tag,k+1,'/50',flush=True)
    xt.flush();xe.flush();del xt,xe
    np.savez_compressed(folder/'meta.npz',train_y=train.y,train_subject=train.subject,
                        weights=np.asarray(counts,dtype=float),**metadata(test))
    np.savez_compressed(folder/'geometry.npz',log_prior=np.stack(priors),
                        scaler_mean=np.stack(means),scaler_scale=np.stack(scales))
    record=dict(tag=tag,protocol=protocol,fold=fold,specs=specs,
                train_subjects=frozen['train_subjects'],eval_subjects=frozen['eval_subjects'],
                train_windows=len(train),eval_windows=len(test),tangent_dimension=820,
                geometry_sha256=geometry_hashes,reference_sha256=sha(reference),
                feature_seconds=time.monotonic()-begin,
                files_sha256={name:sha(folder/name) for name in ['train.npy','eval.npy','meta.npz','geometry.npz']})
    write_json(manifest_path,record)
    print('DONE FEATURES',tag,round(record['feature_seconds'],2),'s',flush=True)
    return folder
