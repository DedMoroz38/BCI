import importlib.util
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from sklearn.covariance import oas
import transforms as t

def test_stretch_matches_repository_and_resets():
    rng=np.random.default_rng(5);x=rng.normal(size=(12,7));groups=[np.arange(7),np.arange(7,12)]
    spec=importlib.util.spec_from_file_location('ded_riemann',t.ROOT/'BCI_DedMoroz38/Brain-Decoding/track2_bci/bci/riemann.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expected=np.empty_like(x)
    for idx in groups:
        state=module.SessionStretch(10)
        expected[idx]=[state(v) for v in x[idx]]
    y=t.stretch(x,groups)
    np.testing.assert_allclose(y,expected,rtol=1e-14,atol=1e-14)
    np.testing.assert_array_equal(y[:4],t.stretch(x[:4],[np.arange(4)]))
    np.testing.assert_array_equal(y[7:],t.stretch(x[7:],[np.arange(5)]))

def test_adaptive_covariance_matches_block_order_implementation():
    rng=np.random.default_rng(1);x=rng.normal(size=(2,41,480));band=(4,8)
    f=sosfiltfilt(butter(4,band,btype='bandpass',fs=120,output='sos'),x,axis=-1)
    z=np.concatenate([f[...,5:],f[...,:-5]],axis=1)
    expected=np.stack([oas(a.T)[0] for a in z])
    perm=np.stack([np.arange(41),np.arange(41)+41],axis=1).ravel()
    actual=t.matrices(x,band,'adaptive')
    np.testing.assert_allclose(actual,expected[:,perm][:,:,perm],atol=1e-12)
    assert np.linalg.eigvalsh(actual).min()>0

def test_balance_prefix_reset_and_zero_strength():
    p=np.array([[.7,.2,.1],[.6,.2,.2],[.1,.3,.6],[.2,.2,.6]])
    groups=[np.arange(2),np.arange(2,4)]
    out=t.balance(p,groups,.5)
    np.testing.assert_array_equal(out[:1],t.balance(p[:1],[np.arange(1)],.5))
    np.testing.assert_array_equal(out[2:],t.balance(p[2:],[np.arange(2)],.5))
    np.testing.assert_allclose(t.balance(p,groups,0),p)
    np.testing.assert_allclose(out.sum(1),1)
    marginal=(p[0]+10/3)/11;q=p[0]/np.sqrt(marginal)
    np.testing.assert_allclose(out[0],q/q.sum())

def test_cache_receipt_rejects_changed_file(tmp_path):
    p=tmp_path/'a.npy';np.save(p,np.eye(2))
    digest=t.sha(p);t.check_digest(p,digest)
    np.save(p,np.ones((2,2)))
    import pytest
    with pytest.raises(RuntimeError):t.check_digest(p,digest)
