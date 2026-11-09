"""Train the already evaluated train-only configuration and export plain arrays."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,importlib.metadata,time
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'temporal_spd_research'))
from temporal_features import fit_model
from temporal_cache import MatrixCache
from bci.data import load_split


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 out=HERE/'package';out.mkdir(exist_ok=True)
 baseline=json.loads((ROOT/'temporal_spd_research/results/lag4_oas/forest20_test_official.json').read_text())
 frozen=json.loads((ROOT/'temporal_spd_research/experiment.json').read_text())
 for name in ['temporal_features.py','temporal_cache.py']:
  assert sha(ROOT/'temporal_spd_research'/name)==frozen['code_sha256'][name]
 assert sha(ROOT/'data/proteus_train.npz')==frozen['data_sha256']['train']
 train=load_split('train',ROOT/'data')
 assert len(train)==1521 and sorted(set(train.subject))==baseline['train_subjects']
 print('Training 50-tree lag4 forest, original6 train subjects only,1521 windows.',flush=True)
 start=time.monotonic()
 with threadpool_limits(limits=1):
  model=fit_model(train,'forest20','lag4_oas',cache=MatrixCache(ROOT/'temporal_spd_research/cache'))
 specs=[dict(band=t.band,channels=t.physical_channels.tolist()) for t in model.trees]
 assert specs==baseline['trees']
 assert all(np.array_equal(t.clf.classes_,[0,1,2]) for t in model.trees)
 assert len(model.trees)==50 and all(len(t.physical_channels)==20 for t in model.trees)
 arrays=dict(format_version=np.array(1),sfreq=np.array(120.),lag=np.array(4),prior_weight=np.array(5.),ch_names=train.ch_names.astype(str),sos_bank=np.stack([f.sos for f in model.bank]),tree_bands=np.array([t.band for t in model.trees]),channels=np.stack([t.channels for t in model.trees]),log_prior=np.stack([t.features.log_prior for t in model.trees]),scaler_mean=np.stack([t.features.x_scaler.mean_ for t in model.trees]),scaler_scale=np.stack([t.features.x_scaler.scale_ for t in model.trees]),coef=np.stack([t.clf.coef_ for t in model.trees]),intercept=np.stack([t.clf.intercept_ for t in model.trees]),classes=model.trees[0].clf.classes_)
 for name,a in arrays.items():
  if np.issubdtype(a.dtype,np.number):assert np.isfinite(a).all(),name
 np.savez_compressed(out/'weights.npz',**arrays)
 info=dict(model='Riemann-Forest-Lag4',n_trees=50,physical_channels_per_tree=20,lag_samples=4,lag_ms=1000*4/120,sfreq=120,prior_weight=5,C=.01,seed=0,training_windows=len(train),training_subjects=sorted(set(train.subject)),training_split='original train only; validation/test labels not used in fit',training_data_sha256=frozen['data_sha256']['train'],source_commit=frozen['source_commit'],inference_dependencies=['numpy','scipy','scikit-learn','benchmark_utils (provided by competition)'],reference_metrics={k:baseline[k] for k in ['session_mean','cell_mean','pooled','accuracy']},weights_sha256=sha(out/'weights.npz'),created_utc=datetime.now(timezone.utc).isoformat())
 (out/'model_info.json').write_text(json.dumps(info,indent=2))
 report=dict(fit_seconds=time.monotonic()-start,weights_bytes=(out/'weights.npz').stat().st_size,package_versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','pyriemann','benchopt']},numerical_source_hashes={n:frozen['code_sha256'][n] for n in ['temporal_features.py','temporal_cache.py']},original_tree_draws_match=True)
 (HERE/'build.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
