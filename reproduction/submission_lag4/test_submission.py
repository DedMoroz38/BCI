"""Catch covariance/feature/classifier export errors and streaming state leakage."""
import copy,importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'2026-competition'));sys.path.insert(0,str(ROOT/'temporal_spd_research'))
from temporal_features import fit_model,compute_matrices
from bci.data import Windows


def load_module():
 path=HERE/'package/submission.py'
 assert path.exists(),'Self-contained inference submission.py has not been implemented'
 spec=importlib.util.spec_from_file_location('lag4_submission_under_test',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def windows(n,seed):
 rng=np.random.default_rng(seed)
 return Windows(X=rng.normal(size=(n,6,480)).astype('float32'),y=np.arange(n)%3,subject=np.repeat(['a','b'],n//2),session=np.full(n,'1'),context=np.full(n,'Graz'),run=np.ones(n,int),onset=np.arange(n,dtype=float),ch_names=np.array(['A','B','C','D','E','F']))


@pytest.fixture
def trained(tmp_path):
 tr,te=windows(12,51),windows(6,52)
 model=fit_model(tr,'forest20','lag4_oas',n_trees=3,n_channels=3)
 # Independent fixture export: no production builder is used in parity tests.
 np.savez(tmp_path/'weights.npz',format_version=np.array(1),sfreq=np.array(120.),lag=np.array(4),prior_weight=np.array(5.),ch_names=tr.ch_names,sos_bank=np.stack([f.sos for f in model.bank]),tree_bands=np.array([t.band for t in model.trees]),channels=np.stack([t.channels for t in model.trees]),log_prior=np.stack([t.features.log_prior for t in model.trees]),scaler_mean=np.stack([t.features.x_scaler.mean_ for t in model.trees]),scaler_scale=np.stack([t.features.x_scaler.scale_ for t in model.trees]),coef=np.stack([t.clf.coef_ for t in model.trees]),intercept=np.stack([t.clf.intercept_ for t in model.trees]),classes=model.trees[0].clf.classes_)
 meta=dict(submission_dir=tmp_path,n_chans=6,n_times=480,n_classes=3,sfreq=120.,ch_names=list(tr.ch_names),device='cpu')
 return model,tr,te,meta


def test_covariances_match_original_lagged_full_channel_oas(trained):
 source,tr,te,meta=trained;m=load_module();packed=m.Solver.get_instance().load_model(meta)
 actual=packed.covariances(te.X[0])
 for band,mat in zip(source.bank,actual):
  np.testing.assert_allclose(mat,compute_matrices(te.X[:1],band.band,'lag4_oas')[0],rtol=1e-13,atol=1e-13)


def test_exported_probabilities_match_source_and_reset_per_stream(trained):
 source,tr,te,meta=trained;m=load_module();packed=m.Solver.get_instance().load_model(meta)
 expected=source.tree_probas(te).mean(0);actual=np.empty_like(expected)
 for idx in te.streams():
  packed.reset_state()
  for i in idx:actual[i]=packed.predict_proba(torch.from_numpy(te.X[i:i+1]))[0]
 np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
 np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
 clone=copy.deepcopy(packed);clone.reset_state();fresh=m.Solver.get_instance().load_model(meta)
 np.testing.assert_allclose(clone.predict_proba(te.X[:2]),fresh.predict_proba(te.X[:2]),atol=1e-12)


def test_batch_and_one_window_updates_are_equivalent(trained):
 _,_,te,meta=trained;m=load_module();a=m.Solver.get_instance().load_model(meta);b=m.Solver.get_instance().load_model(meta)
 np.testing.assert_allclose(a.predict_proba(te.X),np.concatenate([b.predict_proba(w[None]) for w in te.X]),atol=1e-12)
 a.reset_state();pred=a.predict(torch.from_numpy(te.X[:1]));assert pred.shape==(1,) and pred.dtype==np.int64


def test_channel_metadata_reorders_inputs_without_changing_prediction(trained):
 _,tr,te,meta=trained;m=load_module();a=m.Solver.get_instance().load_model(meta)
 changed=dict(meta,ch_names=list(tr.ch_names[::-1]));b=m.Solver.get_instance().load_model(changed)
 np.testing.assert_allclose(a.predict_proba(te.X),b.predict_proba(te.X[:,::-1,:]),atol=1e-12)


@pytest.mark.parametrize('change',[{'sfreq':100.},{'n_classes':2},{'n_chans':8},{'ch_names':['wrong']*6}])
def test_incompatible_metadata_fails_clearly(trained,change):
 _,_,_,meta=trained;m=load_module()
 with pytest.raises(ValueError):m.Solver.get_instance().load_model(dict(meta,**change))
