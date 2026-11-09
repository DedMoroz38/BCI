from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from temporal_features import CANDIDATES,compute_matrices,continuous_nodes,signature_coordinates,stationary_vectors,fit_model,expanded_channels


def test_continuous_quadrature_matches_analytic_integral():
 x=np.array([[1.,2.,-1.,4.],[2.,-2.,3.,1.]])
 z=continuous_nodes(x)
 a,b=x[:,:-1],x[:,1:]
 mu=(a+b).mean(-1)/2
 exact=(a@a.T+b@b.T)/9+(a@b.T+b@a.T)/18-np.outer(mu,mu)
 np.testing.assert_allclose(np.cov(z,bias=True),exact,atol=1e-12)
 np.testing.assert_allclose(z.mean(-1),mu,atol=1e-12)


def test_signature_exact_straight_path_and_integration_by_parts():
 t=np.linspace(0,1,30)
 f=signature_coordinates(np.stack([2+3*t,4+t*t]))
 np.testing.assert_allclose(f[0],[3,1.5,1.5,4.5],atol=1e-12)
 np.testing.assert_allclose(f[:,1]+f[:,2],f[:,0],atol=1e-12)
 np.testing.assert_allclose(signature_coordinates(np.stack([2+3*t,4+t*t])+123),f,atol=1e-12)


def test_stationary_gram_has_equal_diagonal_blocks_and_lag_covariance():
 x=np.random.default_rng(5).normal(size=(3,31));x-=x.mean(-1,keepdims=True)
 z=stationary_vectors(x)
 c=z@z.T
 ids0=np.arange(0,6,2);ids1=ids0+1
 np.testing.assert_allclose(c[np.ix_(ids0,ids0)],x@x.T,atol=1e-12)
 np.testing.assert_allclose(c[np.ix_(ids1,ids1)],x@x.T,atol=1e-12)
 np.testing.assert_allclose(c[np.ix_(ids0,ids1)],x[:,1:]@x[:,:-1].T,atol=1e-12)


@pytest.mark.parametrize('v',CANDIDATES)
def test_spd_batch_single_and_expected_dimension(v):
 x=np.random.default_rng(3).normal(size=(3,6,480)).astype('float32')
 r=compute_matrices(x,(4,40),v)
 d=12 if v in CANDIDATES[:4] else 6
 assert r.shape==(3,d,d) and np.linalg.eigvalsh(r).min()>0
 np.testing.assert_allclose(r,r.swapaxes(-1,-2),atol=1e-12)
 np.testing.assert_allclose(r[1],compute_matrices(x[1:2],(4,40),v)[0],atol=1e-11)


def test_pyramid_blocks_follow_half_order_and_have_zero_cross_blocks():
 from scipy.signal import butter,sosfiltfilt
 from pyriemann.estimation import Covariances
 x=np.random.default_rng(6).normal(size=(1,4,480)).astype('float32')
 f=sosfiltfilt(butter(4,(4,40),btype='bandpass',fs=120,output='sos'),x.astype(float),axis=-1)
 r=compute_matrices(x,(4,40),'pyramid2_oas')[0]
 ids=np.arange(0,8,2)
 np.testing.assert_allclose(r[np.ix_(ids,ids)],Covariances('oas').transform(f[...,:240])[0])
 np.testing.assert_array_equal(r[np.ix_(ids,ids+1)],0)


def test_quadrature_is_invariant_to_complex_structure_rotation():
 x=np.random.default_rng(7).normal(size=(1,4,480)).astype('float32')
 r=compute_matrices(x,(4,40),'quadrature_oas')[0]
 j=np.kron(np.eye(4),np.array([[0.,-1.],[1.,0.]]))
 np.testing.assert_allclose(r,j@r@j.T,atol=1e-12)


def test_signature_adds_psd_to_original_covariance():
 from scipy.signal import butter,sosfiltfilt
 from pyriemann.estimation import Covariances
 x=np.random.default_rng(8).normal(size=(1,6,480)).astype('float32')
 f=sosfiltfilt(butter(4,(4,40),btype='bandpass',fs=120,output='sos'),x.astype(float),axis=-1)
 c=Covariances('oas').transform(f)[0]
 r=compute_matrices(x,(4,40),'signature2_oas')[0]
 assert np.linalg.eigvalsh(r-c).min()>-1e-12


def windows(n,seed):
 from bci.data import Windows
 rng=np.random.default_rng(seed)
 return Windows(X=rng.normal(size=(n,6,480)).astype('float32'),y=np.arange(n)%3,
  subject=np.repeat(['a','b'],n//2),session=np.full(n,'1'),context=np.full(n,'Graz'),
  run=np.ones(n,int),onset=np.arange(n,dtype=float),ch_names=np.array([str(i) for i in range(6)]))


@pytest.mark.parametrize('v',CANDIDATES)
def test_original_models_physical_grouping_stream_reset_and_prefix(v):
 from bci.models import forest
 from bci.evaluation import predict_streams
 from temporal_features import TemporalFeatures,MULTIPLICITY
 tr,te=windows(12,9),windows(6,10)
 original=forest.fit(tr,n_trees=2,n_channels=3)
 model=fit_model(tr,'forest12',v,n_trees=2,n_channels=3)
 for a,b in zip(original.trees,model.trees):
  assert a.band==b.band
  np.testing.assert_array_equal(a.channels,getattr(b,'physical_channels',b.channels))
  expect=expanded_channels(a.channels) if MULTIPLICITY[v]==2 else a.channels
  np.testing.assert_array_equal(b.channels,expect)
 np.testing.assert_array_equal(predict_streams(model,te),model.tree_probas(te).mean(0).argmax(-1))
 np.testing.assert_array_equal(predict_streams(model,te),predict_streams(model,te))
 f=TemporalFeatures(representation=v);f.fit_transform(tr)
 a,_=f.transform(te.select(np.arange(6)<2));b,_=f.transform(te)
 np.testing.assert_allclose(a,b[:2],atol=1e-10)
