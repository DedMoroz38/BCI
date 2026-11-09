"""Analytic formula, prefix weighting and causal model checks for five new means."""
import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import sqrtm, logm as scipy_logm

sys.path.insert(0, str(Path(__file__).parent))
try:
    from literature_references import LiteratureSession, LiteratureFeatures, fit_model
except ImportError:
    LiteratureSession = None

VARIANTS = ['harmonic_cumulative', 'poweuclid_half', 'kullback_sym', 'logchol_cumulative', 'wasserstein_recursive']

@pytest.mark.parametrize('variant,want', [
    ('harmonic_cumulative', 27/23), ('poweuclid_half', 16/9),
    ('kullback_sym', np.sqrt(63/23)), ('logchol_cumulative', 9**(1/6)),
    ('wasserstein_recursive', 16/9)])
def test_hand_derived_scalar_reference(variant, want):
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    s = LiteratureSession(np.zeros((1, 1)), 5, variant)
    x, _ = s(np.array([[9.]]))
    np.testing.assert_allclose(s.reference, [[want]], rtol=1e-11)
    np.testing.assert_allclose(x, [np.log(9/want)], atol=1e-11)

@pytest.mark.parametrize('variant,library_mean', [
    ('harmonic_cumulative', 'mean_harmonic'), ('poweuclid_half', 'mean_poweuclid'),
    ('kullback_sym', 'mean_kullback_sym'), ('logchol_cumulative', 'mean_logchol')])
def test_weighted_noncommuting_prefix_equals_published_mean(variant, library_mean):
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    from pyriemann.geometry import mean
    prior = np.array([[2., .4], [.4, 1.]])
    a = np.array([[1., -.2], [-.2, 3.]])
    b = np.array([[4., .6], [.6, .8]])
    s = LiteratureSession(scipy_logm(prior), 5, variant)
    s(a); s(b)
    kwargs = dict(sample_weight=np.array([5., 1., 1.])/7)
    if variant == 'poweuclid_half': kwargs['p'] = .5
    expected = getattr(mean, library_mean)(np.stack([prior, a, b]), **kwargs)
    np.testing.assert_allclose(s.reference, expected, atol=2e-10)
    if variant == 'kullback_sym':
        arithmetic = (5*prior+a+b)/7
        precision = (5*np.linalg.inv(prior)+np.linalg.inv(a)+np.linalg.inv(b))/7
        np.testing.assert_allclose(s.reference @ precision @ s.reference, arithmetic, atol=2e-10)


def test_log_cholesky_averages_strict_lower_and_log_diagonal_separately():
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    prior = np.array([[4., 2.], [2., 5.]])
    cov = np.array([[9., -3.], [-3., 2.]])
    s = LiteratureSession(scipy_logm(prior), 5, 'logchol_cumulative')
    s(cov)
    # chol(prior)=[[2,0],[1,2]], chol(cov)=[[3,0],[-1,1]].
    expected_factor = np.array([[(2**5*3)**(1/6), 0.], [2/3, (2**5)**(1/6)]])
    np.testing.assert_allclose(s.reference, expected_factor @ expected_factor.T, atol=1e-10)


def test_bures_update_matches_independent_optimal_transport_formula():
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    prior = np.array([[2., .4], [.4, 1.]])
    cov = np.array([[1., -.2], [-.2, 3.]])
    half = sqrtm(prior); invhalf = np.linalg.inv(half)
    transport = invhalf @ sqrtm(half @ cov @ half) @ invhalf
    interpolator = (5/6)*np.eye(2)+(1/6)*transport
    expected = interpolator @ prior @ interpolator
    s = LiteratureSession(scipy_logm(prior), 5, 'wasserstein_recursive')
    s(cov)
    np.testing.assert_allclose(s.reference, expected, atol=1e-10)

@pytest.mark.parametrize('variant', VARIANTS)
def test_zero_prior_recovers_current_covariance_and_remains_spd(variant):
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    s = LiteratureSession(np.zeros((2, 2)), 0, variant)
    cov = np.array([[2., .3], [.3, 1.]])
    x, _ = s(cov)
    np.testing.assert_allclose(s.reference, cov, atol=1e-10)
    np.testing.assert_allclose(x, 0., atol=1e-10)
    s(np.array([[1., -.4], [-.4, 3.]]))
    assert np.linalg.eigvalsh(s.reference).min() > 0


def toy_windows(n, seed):
    from bci.data import Windows
    rng = np.random.default_rng(seed)
    return Windows(X=rng.normal(size=(n, 6, 480)).astype('float32'),
        y=np.arange(n)%3, subject=np.repeat(['a', 'b'], n//2),
        session=np.full(n, '1'), context=np.full(n, 'Graz'),
        run=np.ones(n,int), onset=np.arange(n,dtype=float),
        ch_names=np.array([str(i) for i in range(6)]))

@pytest.mark.parametrize('variant', VARIANTS)
def test_causal_prefix_session_reset_and_batch_stream_parity(variant):
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    from bci.evaluation import predict_streams
    train, test = toy_windows(24, 31), toy_windows(12, 32)
    model = fit_model(train, 'forest12', variant, n_trees=3, n_channels=3)
    expected = model.tree_probas(test).mean(0).argmax(-1)
    predicted = predict_streams(model, test)
    assert np.array_equal(predicted, expected)
    assert np.array_equal(predicted, predict_streams(model, test))
    features = LiteratureFeatures(variant=variant)
    features.fit_transform(train)
    prefix = test.select(np.arange(len(test))<3)
    x_small,_ = features.transform(prefix)
    x_full,_ = features.transform(test)
    np.testing.assert_allclose(x_small, x_full[:3], atol=1e-11, rtol=0)


def test_control_adapter_preserves_exact_coefficients_and_random_trees():
    assert LiteratureSession is not None, 'Literature reference implementation is missing'
    from bci.models import forest, logreg
    from bci.evaluation import predict_streams
    train, test = toy_windows(24, 33), toy_windows(12, 34)
    old_factory = forest.RiemannFeatures
    a = forest.fit(train, n_trees=3, n_channels=3)
    b = fit_model(train, 'forest12', 'logeuclid_cumulative', n_trees=3, n_channels=3)
    assert forest.RiemannFeatures is old_factory
    assert np.array_equal(predict_streams(a,test), predict_streams(b,test))
    for left,right in zip(a.trees,b.trees):
        assert left.band == right.band and np.array_equal(left.channels,right.channels)
        assert np.array_equal(left.clf.coef_,right.clf.coef_)
    a = logreg.fit(train)
    b = fit_model(train, 'logreg', 'logeuclid_cumulative')
    assert np.array_equal(a.clf.coef_,b.clf.coef_)
