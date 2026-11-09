"""Analytic and causal checks for changing reference updates only."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power, logm as scipy_logm

sys.path.insert(0, str(Path(__file__).parent))
try:
    from references import ReferenceSession, ReferenceFeatures, fit_model
except ImportError:
    ReferenceSession = ReferenceFeatures = fit_model = None


def test_arithmetic_uses_covariances_not_logarithms():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    s = ReferenceSession(np.zeros((1, 1)), 5, 'euclid_cumulative')
    x, fp = s(np.array([[9.]]))
    np.testing.assert_allclose(s.reference, [[7/3]], rtol=1e-12)
    np.testing.assert_allclose(x, [np.log(27/7)], rtol=1e-12)
    np.testing.assert_allclose(fp, [np.log(7/3)], rtol=1e-12)


def test_recursive_scalar_is_geometric_mean_with_prior_mass():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    s = ReferenceSession(np.zeros((1, 1)), 5, 'riemann_recursive')
    x, _ = s(np.array([[64.]]))
    np.testing.assert_allclose(s.reference, [[2.]], atol=1e-12)
    np.testing.assert_allclose(x, [np.log(32.)], atol=1e-12)


def test_noncommuting_geodesic_uses_covariance_domain():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    prior = np.array([[2., .4], [.4, 1.]])
    cov = np.array([[1., .2], [.2, 3.]])
    half = fractional_matrix_power(prior, .5)
    invhalf = fractional_matrix_power(prior, -.5)
    expected = half @ fractional_matrix_power(invhalf @ cov @ invhalf, 1/6) @ half
    s = ReferenceSession(scipy_logm(prior), 5, 'riemann_recursive')
    s(cov)
    np.testing.assert_allclose(s.reference, expected, atol=1e-11)


def test_ema_normalizes_decaying_prior_and_window_weights():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    s = ReferenceSession(np.zeros((1, 1)), 2, 'logeuclid_ema', decay=.5)
    s(np.array([[np.exp(2.)]]))
    x, _ = s(np.array([[np.exp(4.)]]))
    # S1=2,n1=2; S2=5,n2=2 -> log(reference)=2.5.
    np.testing.assert_allclose(s.reference, [[np.exp(2.5)]], atol=1e-12)
    np.testing.assert_allclose(x, [1.5], atol=1e-12)


def test_ema_forgets_an_old_outlier_faster_than_a_recent_one():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    old = ReferenceSession(np.zeros((1, 1)), 5, 'logeuclid_ema', decay=.5)
    recent = ReferenceSession(np.zeros((1, 1)), 5, 'logeuclid_ema', decay=.5)
    for a, b in zip([16., 1., 1., 1.], [1., 1., 1., 16.]):
        old(np.array([[a]])); recent(np.array([[b]]))
    # The outlier's log weight differs by .5^3, denominator is identical.
    np.testing.assert_allclose(np.log(recent.reference), 8*np.log(old.reference), atol=1e-12)


@pytest.mark.parametrize('variant', ['euclid_cumulative', 'riemann_recursive', 'logeuclid_ema'])
def test_reference_stays_spd_and_zero_prior_gives_current_window(variant):
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    s = ReferenceSession(np.zeros((2, 2)), 0, variant)
    cov = np.array([[2., .3], [.3, 1.]])
    x, _ = s(cov)
    np.testing.assert_allclose(s.reference, cov, atol=1e-11)
    np.testing.assert_allclose(x, 0., atol=1e-11)
    s(np.array([[.5, -.1], [-.1, 3.]]))
    assert np.linalg.eigvalsh(s.reference).min() > 0


def toy_windows(n, seed):
    from bci.data import Windows
    rng = np.random.default_rng(seed)
    return Windows(X=rng.normal(size=(n, 6, 480)).astype('float32'),
        y=np.arange(n)%3, subject=np.repeat(['a', 'b'], n//2),
        session=np.full(n, '1'), context=np.full(n, 'Graz'),
        run=np.ones(n, int), onset=np.arange(n, dtype=float),
        ch_names=np.array([str(i) for i in range(6)]))


def test_control_adapter_equals_original_training_and_predictions():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    from bci.models import forest, logreg
    from bci.evaluation import predict_streams
    train, test = toy_windows(24, 12), toy_windows(12, 13)
    for name, direct in [('logreg', lambda: logreg.fit(train)),
                         ('forest12', lambda: forest.fit(train, n_trees=3, n_channels=3))]:
        source_factory = forest.RiemannFeatures
        original = direct()
        adapted = fit_model(train, name, 'logeuclid_cumulative', n_trees=3, n_channels=3)
        assert forest.RiemannFeatures is source_factory
        assert np.array_equal(predict_streams(original, test), predict_streams(adapted, test))
        if name == 'logreg':
            assert np.array_equal(original.clf.coef_, adapted.clf.coef_)
        else:
            for a, b in zip(original.trees, adapted.trees):
                assert a.band == b.band and np.array_equal(a.channels, b.channels)
                assert np.array_equal(a.clf.coef_, b.clf.coef_)


@pytest.mark.parametrize('variant', ['euclid_cumulative', 'riemann_recursive', 'logeuclid_ema'])
def test_causal_features_session_reset_and_stream_prediction_parity(variant):
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    from bci.evaluation import predict_streams
    train, test = toy_windows(24, 14), toy_windows(12, 15)
    model = fit_model(train, 'forest12', variant, n_trees=3, n_channels=3)
    predicted = predict_streams(model, test)
    expected = model.tree_probas(test).mean(0).argmax(-1)
    assert np.array_equal(predicted, expected)
    assert np.array_equal(predicted, predict_streams(model, test))
    features = ReferenceFeatures(variant=variant)
    features.fit_transform(train)
    prefix = test.select(np.arange(len(test)) < 3)
    small, _ = features.transform(prefix)
    full, _ = features.transform(test)
    np.testing.assert_allclose(small, full[:3], rtol=0, atol=1e-12)


def test_invalid_configuration_is_rejected():
    assert ReferenceSession is not None, 'Reference-update implementation is missing'
    for kwargs in [dict(variant='unknown'), dict(variant='logeuclid_ema', decay=1.1),
                   dict(variant='logeuclid_ema', decay=0.)]:
        with pytest.raises(ValueError):
            ReferenceSession(np.zeros((1, 1)), 5, **kwargs)
