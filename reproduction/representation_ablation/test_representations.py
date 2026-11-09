"""Behavioral checks for the three requested SPD window representations."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import hilbert
from scipy.spatial.distance import pdist

sys.path.insert(0,str(Path(__file__).parent))
try:
    from representations import delay_embed, expanded_channels, amplitude_envelope, hsic_matrix, compute_matrices, fit_model, RepresentationFeatures
except ImportError:
    delay_embed=None


def test_delay_alignment_and_electrode_major_order():
    assert delay_embed is not None, 'Representation implementation is missing'
    x=np.array([[[0.,1.,2.,3.],[10.,11.,12.,13.]]])
    expected=np.array([[[1.,2.,3.],[0.,1.,2.],[11.,12.,13.],[10.,11.,12.]]])
    np.testing.assert_array_equal(delay_embed(x),expected)
    np.testing.assert_array_equal(expanded_channels(np.array([1,3])),[2,3,6,7])


def test_grouped_submatrix_keeps_all_delayed_copies():
    assert delay_embed is not None, 'Representation implementation is missing'
    x=np.arange(3*8,dtype=float).reshape(1,3,8)
    z=delay_embed(x)
    selected=z[:,expanded_channels(np.array([0,2]))]
    expected=delay_embed(x[:,[0,2]])
    np.testing.assert_array_equal(selected,expected)


def test_hilbert_envelope_recovers_known_amplitude_modulation():
    assert delay_embed is not None, 'Representation implementation is missing'
    t=np.arange(480)/120
    a=2+.25*np.sin(2*np.pi*t)
    x=a*np.cos(2*np.pi*10*t)
    np.testing.assert_allclose(amplitude_envelope(x),a,atol=1e-12)


def naive_hsic(x):
    h,t=x.shape
    H=np.eye(t)-np.ones((t,t))/t
    centered=[]
    for row in x:
        sigma=np.median(pdist(row[:,None],metric='euclidean'))
        if sigma<=1e-12:sigma=1.
        K=np.exp(-(row[:,None]-row[None,:])**2/(2*sigma*sigma))
        centered.append(H@K@H)
    S=np.array([[np.trace(a@b)/t**2 for b in centered] for a in centered])
    return S+max(1e-12,1e-6*np.trace(S)/h)*np.eye(h)


def test_hsic_equals_naive_full_centered_kernel_trace():
    assert delay_embed is not None, 'Representation implementation is missing'
    x=np.random.default_rng(2).normal(size=(4,17))
    np.testing.assert_allclose(hsic_matrix(x),naive_hsic(x),rtol=1e-11,atol=1e-13)


def test_hsic_detects_squared_dependence_with_zero_linear_covariance():
    assert delay_embed is not None, 'Representation implementation is missing'
    x=np.array([-2.,-1.,0.,1.,2.])
    assert abs(np.cov(x,x*x)[0,1])<1e-14
    matrix=hsic_matrix(np.stack([x,x*x]))
    assert matrix[0,1]>.001


def test_hsic_is_invariant_to_joint_time_permutation_and_spd_for_constants():
    assert delay_embed is not None, 'Representation implementation is missing'
    rng=np.random.default_rng(3)
    x=rng.normal(size=(4,31));perm=rng.permutation(31)
    np.testing.assert_allclose(hsic_matrix(x),hsic_matrix(x[:,perm]),atol=1e-12)
    np.testing.assert_allclose(hsic_matrix(np.zeros((3,32))),1e-12*np.eye(3),atol=1e-20)


@pytest.mark.parametrize('representation,dim',[('delay1_oas',12),('hsic_rbf',6),('envelope_oas',6)])
def test_matrices_are_spd_and_batch_equals_one_window(representation,dim):
    assert delay_embed is not None, 'Representation implementation is missing'
    X=np.random.default_rng(4).normal(size=(3,6,120)).astype('float32')
    batch=compute_matrices(X,(4,40),representation)
    assert batch.shape==(3,dim,dim)
    assert np.isfinite(batch).all() and np.linalg.eigvalsh(batch).min()>0
    for i in range(3):
        np.testing.assert_allclose(batch[i],compute_matrices(X[i:i+1],(4,40),representation)[0],atol=1e-11,rtol=1e-11)


def test_envelope_retains_original_oas_estimator():
    assert delay_embed is not None, 'Representation implementation is missing'
    from scipy.signal import butter,sosfiltfilt
    from pyriemann.estimation import Covariances
    X=np.random.default_rng(5).normal(size=(2,4,120)).astype('float32')
    filtered=sosfiltfilt(butter(4,(4,40),btype='bandpass',fs=120,output='sos'),X.astype(float),axis=-1)
    expected=Covariances('oas').transform(np.abs(hilbert(filtered,axis=-1)))
    np.testing.assert_allclose(compute_matrices(X,(4,40),'envelope_oas'),expected,atol=1e-12)


def toy_windows(n,seed):
    from bci.data import Windows
    rng=np.random.default_rng(seed)
    return Windows(X=rng.normal(size=(n,6,120)).astype('float32'),y=np.arange(n)%3,
        subject=np.repeat(['a','b'],n//2),session=np.full(n,'1'),context=np.full(n,'Graz'),
        run=np.ones(n,int),onset=np.arange(n,dtype=float),ch_names=np.array([str(i) for i in range(6)]))


def test_original_control_has_identical_coefficients_and_predictions():
    assert delay_embed is not None, 'Representation implementation is missing'
    from bci.models import forest,logreg
    from bci.evaluation import predict_streams
    tr,te=toy_windows(12,6),toy_windows(6,7)
    original_factory,original_sub=forest.RiemannFeatures,forest._sub
    a=forest.fit(tr,n_trees=2,n_channels=3)
    b=fit_model(tr,'forest12','original_oas',n_trees=2,n_channels=3)
    assert forest.RiemannFeatures is original_factory and forest._sub is original_sub
    assert np.array_equal(predict_streams(a,te),predict_streams(b,te))
    for left,right in zip(a.trees,b.trees):
        assert left.band==right.band and np.array_equal(left.channels,right.channels)
        assert np.array_equal(left.clf.coef_,right.clf.coef_)
    a=logreg.fit(tr);b=fit_model(tr,'logreg','original_oas')
    assert np.array_equal(a.clf.coef_,b.clf.coef_)


@pytest.mark.parametrize('representation',['delay1_oas','hsic_rbf','envelope_oas'])
def test_physical_trees_causal_features_and_stream_reset_parity(representation):
    assert delay_embed is not None, 'Representation implementation is missing'
    from bci.models import forest
    from bci.evaluation import predict_streams
    tr,te=toy_windows(12,8),toy_windows(6,9)
    baseline=forest.fit(tr,n_trees=2,n_channels=3)
    model=fit_model(tr,'forest12',representation,n_trees=2,n_channels=3)
    for a,b in zip(baseline.trees,model.trees):
        physical=getattr(b,'physical_channels',b.channels)
        assert np.array_equal(a.channels,physical) and a.band==b.band
        if representation=='delay1_oas':
            np.testing.assert_array_equal(b.channels,expanded_channels(physical))
            assert b.clf.n_features_in_==21  # 6*7/2, three physical electrodes with two copies
    predicted=predict_streams(model,te)
    np.testing.assert_array_equal(predicted,model.tree_probas(te).mean(0).argmax(-1))
    np.testing.assert_array_equal(predicted,predict_streams(model,te))
    features=RepresentationFeatures(representation=representation)
    features.fit_transform(tr)
    prefix=te.select(np.arange(len(te))<2)
    small,_=features.transform(prefix);full,_=features.transform(te)
    np.testing.assert_allclose(small,full[:2],rtol=1e-10,atol=1e-10)
