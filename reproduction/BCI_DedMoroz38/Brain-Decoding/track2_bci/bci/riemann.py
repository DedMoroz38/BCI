"""Riemannian features: covariances + causal session recentering.

For each window we get two vectors:
- ``x``:  the window's covariance in the tangent space at its session's
          reference point (the person is "removed");
- ``fp``: the session's fingerprint, where its reference point lies on the
          manifold relative to the training mean (the person "itself").

The reference point is a running log-Euclidean mean of the session's
covariances, so it only uses windows seen so far, exactly as on the platform.

Optionally the covariance is *augmented*: every channel gets a 90°-shifted
copy (Hilbert transform, or a quarter-period time lag), so the covariance also
holds phase-lagged coupling between channels, not only zero-lag coupling.
"""

import numpy as np
import torch
from pyriemann.estimation import Covariances
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.preprocessing import StandardScaler

try:  # pyriemann moved these to pyriemann.geometry
    from pyriemann.geometry.base import expm, invsqrtm, logm
    from pyriemann.geometry.mean import mean_riemann
except ImportError:
    from pyriemann.utils.base import expm, invsqrtm, logm
    from pyriemann.utils.mean import mean_riemann
try:
    from pyriemann.geometry.tangentspace import upper
except ImportError:
    from pyriemann.utils.tangentspace import upper

from bci.data import SFREQ

AUGMENTATIONS = (None, "hilbert", "lag", "lag3", "lag@<tau>")  # "lag@<tau>": e.g. "lag@3"
HILBERT_EDGE = 15  # samples cut from both ends after the Hilbert transform (edge artifacts), 0.125 s


def quarter_period_lag(band, sfreq=SFREQ):
    """Quarter period of the band centre in samples: the lag of a 90° phase shift."""
    return max(1, int(round(sfreq / (4 * np.mean(band)))))


def lag_spec(kind, band, sfreq=SFREQ):
    """(order, tau) of a lag augmentation: "lag" (2, quarter period), "lag3" (3, quarter
    period), "lag@<tau>" (2, tau samples); None for anything else."""
    if kind == "lag":
        return 2, quarter_period_lag(band, sfreq)
    if kind == "lag3":
        return 3, quarter_period_lag(band, sfreq)
    if isinstance(kind, str) and kind.startswith("lag@"):
        return 2, int(kind[4:])
    return None


def n_copies(kind):
    """How many stacked copies of every channel an augmentation produces."""
    if kind is None:
        return 1
    spec = lag_spec(kind, (8, 13))
    return 2 if spec is None else spec[0]


def check_augment(kind):
    if kind in (None, "hilbert") or lag_spec(kind, (8, 13)) is not None:
        return
    raise ValueError(f"augment must be one of {AUGMENTATIONS}, got {kind!r}")


def augment(X, band, kind, sfreq=SFREQ):
    """(N, C, T) band-passed windows -> (N, kC, T') with k shifted copies of every channel.

    - ``"hilbert"``: [x; H(x)], real and imaginary parts of the analytic signal;
    - ``"lag"``: [x(t); x(t - tau)], tau = quarter period at the band centre
      (theta 5, alpha 3, low beta 2, high beta / gamma 1 sample at 120 Hz);
    - ``"lag3"``: [x(t); x(t - tau); x(t - 2 tau)], the order-3 augmented
      covariance (delay embedding, Carrara & Papadopoulo);
    - ``"lag@<tau>"``: [x(t); x(t - tau)] with a fixed tau in samples.

    The cross blocks of the covariance are then lagged cross-covariances
    (phase-lagged / directed coupling), which a zero-lag covariance misses.
    Input channel i becomes rows i, i + C, i + 2C, ...
    """
    if kind is None:
        return X
    if kind == "hilbert":
        X = np.concatenate([X, np.imag(hilbert(X, axis=-1))], axis=1)
        return X[..., HILBERT_EDGE:-HILBERT_EDGE]
    spec = lag_spec(kind, band, sfreq)
    if spec is None:
        raise ValueError(f"augment must be one of {AUGMENTATIONS}, got {kind!r}")
    order, tau = spec
    T, span = X.shape[-1], (order - 1) * tau
    return np.concatenate([X[..., span - k * tau:T - k * tau] for k in range(order)], axis=1)


class SessionRecentering:
    """Running log-Euclidean mean of one session's covariances.

    Starts from the training mean (prior) counted as ``prior_weight`` windows.
    """

    def __init__(self, log_prior, prior_weight, stretch=None):
        self.log_prior = log_prior
        self.log_sum, self.n = prior_weight * log_prior, prior_weight
        self.stretch = None if stretch is None else SessionStretch(stretch)

    def __call__(self, cov):
        """-> (x, fp) for one covariance, updating the session reference."""
        self.log_sum = self.log_sum + logm(cov)
        self.n += 1
        log_ref = self.log_sum / self.n
        W = invsqrtm(expm(log_ref))
        return upper(logm(W @ cov @ W)), upper(log_ref - self.log_prior)


class SessionStretch:
    """Causal per-feature stretching (dispersion matching, the unsupervised step of
    Riemannian Procrustes analysis): each standardized tangent feature is divided by
    its running RMS in the session, which starts from 1 counted as ``weight`` windows."""

    def __init__(self, weight):
        self.weight, self.sum_sq, self.n = weight, None, weight

    def __call__(self, x):
        if self.sum_sq is None:
            self.sum_sq = np.full(x.shape, float(self.weight))
        self.sum_sq = self.sum_sq + x ** 2
        self.n += 1
        return x / np.sqrt(self.sum_sq / self.n)


class RiemannFeatures:
    """Fit on training windows, then turn any windows into scaled (x, fp).

    ``x`` is standardized per feature. ``fp`` is divided by one global number,
    so distances between fingerprints stay log-Euclidean distances.

    Parameters
    ----------
    band : (low, high) band-pass in Hz before the covariance.
    prior_weight : how many windows the training mean counts for at the start
        of a session (larger = the reference adapts more slowly).
    augment : None, ``"hilbert"``, ``"lag"``, ``"lag3"`` or ``"lag@<tau>"``, see :func:`augment`.
    stretch : None, or the prior weight (in windows) of causal per-session
        stretching of the standardized features, see :class:`SessionStretch`.
    """

    def __init__(self, band=(4, 40), prior_weight=5, augment=None, stretch=None):
        check_augment(augment)
        self.band = tuple(band)
        self.sos = butter(4, band, btype="bandpass", fs=SFREQ, output="sos")
        self.prior_weight = prior_weight
        self.augment = augment
        self.stretch = stretch

    def covariances(self, X):
        """(N, C, T) -> (N, C, C), or (N, 2C, 2C) when augmented."""
        X = sosfiltfilt(self.sos, np.asarray(X, np.float64), axis=-1)
        return Covariances("oas").transform(augment(X, self.band, self.augment))

    def rows(self, channels, n_channels):
        """Rows of a channel subset in this covariance (with their shifted copies when augmented)."""
        copies = n_copies(self.augment)
        return np.concatenate([np.asarray(channels) + k * n_channels for k in range(copies)])

    def fit_transform(self, windows, covs=None):
        """``covs``: precomputed covariances of ``windows`` (else computed here)."""
        covs = self.covariances(windows.X) if covs is None else covs
        self.log_prior = np.mean([logm(c) for c in covs], axis=0)
        self.x_scaler, self.fp_scale = None, 1.0
        x, fp = self._session_features(windows, covs)  # unscaled, unstretched
        self.x_scaler, self.fp_scale = StandardScaler().fit(x), fp.std()
        x = self.x_scaler.transform(x)
        if getattr(self, "stretch", None) is not None:  # the same causal stretching as at test time
            for idx in windows.streams():
                stretch = SessionStretch(self.stretch)
                x[idx] = np.array([stretch(v) for v in x[idx]])
        return x, fp / self.fp_scale

    def transform(self, windows, covs=None):
        covs = self.covariances(windows.X) if covs is None else covs
        return self._session_features(windows, covs)

    def new_session(self):
        return SessionRecentering(self.log_prior, self.prior_weight, getattr(self, "stretch", None))

    def step(self, session, cov):
        """Scaled (and, with ``stretch``, stretched) (x, fp) of one window of a session."""
        x, fp = session(cov)
        if self.x_scaler is not None:
            x = self.x_scaler.transform(x[None])[0]
            if session.stretch is not None:
                x = session.stretch(x)
        return x, fp / self.fp_scale

    def _session_features(self, windows, covs):
        x, fp = [None] * len(windows), [None] * len(windows)
        for idx in windows.streams():
            session = self.new_session()
            for i in idx:
                x[i], fp[i] = self.step(session, covs[i])
        return np.array(x), np.array(fp)


class StreamingDecoder:
    """Track 2 contract: ``reset_state()`` + ``predict(X)`` for X of shape (1, C, T).

    ``classify(x, fp) -> class`` is any model working on RiemannFeatures.
    """

    def __init__(self, features, classify):
        self.features, self.classify = features, classify
        self.reset_state()

    def reset_state(self):
        self.session = self.features.new_session()

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        x, fp = self.features.step(self.session, self.features.covariances(X)[0])
        return np.array([self.classify(x, fp)])
