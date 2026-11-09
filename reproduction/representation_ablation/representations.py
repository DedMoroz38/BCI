"""Replace R(X) only; retain the original cumulative log-Euclidean pipeline."""
from contextlib import contextmanager
from functools import lru_cache, partial
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Brain-Decoding/track2_bci"))
from bci.data import SFREQ
from bci.riemann import RiemannFeatures
from pyriemann.estimation import Covariances

CONTROL = "original_oas"
CANDIDATES = ("delay1_oas", "hsic_rbf", "envelope_oas")
LAG = 1
RIDGE = 1e-6
DEFINITIONS = {
    "delay1_oas": "OAS covariance of electrode-major [x_i[t],x_i[t-1]], t=1..479; both copies selected together",
    "hsic_rbf": "R_ij=tr(H*K_i*H*H*K_j*H)/T^2 + eps*I; RBF sigma_i=median pairwise absolute differences per window/channel; eps=max(1e-12,1e-6*mean(diag(S)))",
    "envelope_oas": "OAS covariance of abs(hilbert(band_filtered_X)), full 480-sample window",
}
SOURCES = {
    "delay1_oas": "https://arxiv.org/abs/2302.04508",
    "hsic_rbf": "https://www.cs.cmu.edu/~arthurg/papers/GreBouSmoSch05.pdf",
    "envelope_oas": "https://arxiv.org/html/2111.03122v2",
}


def delay_embed(X, lag=LAG):
    X = np.asarray(X)
    if not 0 < lag < X.shape[-1]:
        raise ValueError("Delay must leave aligned samples")
    blocks = np.stack([X[..., lag:], X[..., :-lag]], axis=-2)
    return blocks.reshape(X.shape[:-2] + (2*X.shape[-2], X.shape[-1]-lag))


def expanded_channels(channels):
    return (2*np.asarray(channels, dtype=int)[:, None] + np.array([0, 1])).reshape(-1)


def amplitude_envelope(X):
    return np.abs(hilbert(X, axis=-1))


@lru_cache(maxsize=16)
def _triangles(t):
    i, j = np.triu_indices(t)
    off = np.triu_indices(t, 1)
    weight = np.where(i == j, 1., np.sqrt(2.))
    return i, j, off, weight


def hsic_matrix(X):
    """Exact full-sample T^-2 HSIC Gram, computed via symmetric vectorization."""
    X = np.asarray(X, dtype=np.float64)
    h, t = X.shape
    i, j, off, weight = _triangles(t)
    kernel = X[:, :, None] - X[:, None, :]
    sigma = np.median(np.abs(kernel[:, off[0], off[1]]), axis=-1)
    sigma = np.where(sigma > 1e-12, sigma, 1.)
    np.square(kernel, out=kernel)
    kernel *= (-.5 / sigma**2)[:, None, None]
    np.exp(kernel, out=kernel)
    row_mean = kernel.mean(axis=-1)
    grand_mean = row_mean.mean(axis=-1)
    kernel -= row_mean[:, :, None]
    kernel -= row_mean[:, None, :]
    kernel += grand_mean[:, None, None]
    # Symmetric vec_sqrt2 preserves tr(G_i G_j), using half the dense storage.
    vectors = kernel[:, i, j] * weight
    gram = (vectors @ vectors.T) / t**2
    gram = (gram + gram.T) / 2
    eps = max(1e-12, RIDGE * np.trace(gram) / h)
    return gram + eps*np.eye(h)


def compute_matrices(X, band, representation):
    """Pure per-window transform: no labels, fitted statistics or other windows."""
    X = np.asarray(X, dtype=np.float64)
    filtered = sosfiltfilt(butter(4, band, btype="bandpass", fs=SFREQ, output="sos"), X, axis=-1)
    if representation == "hsic_rbf":
        return np.stack([hsic_matrix(x) for x in filtered])
    if representation == "delay1_oas":
        filtered = delay_embed(filtered)
    elif representation == "envelope_oas":
        filtered = amplitude_envelope(filtered)
    elif representation != CONTROL:
        raise ValueError(f"Unknown representation: {representation}")
    return Covariances("oas").transform(filtered)


class RepresentationFeatures(RiemannFeatures):
    def __init__(self, band=(4, 40), prior_weight=5, *, representation=CONTROL, cache=None):
        if representation not in (CONTROL,) + CANDIDATES:
            raise ValueError(f"Unknown representation: {representation}")
        super().__init__(band, prior_weight)
        self.representation, self.cache = representation, cache

    def covariances(self, X):
        if self.representation == CONTROL:
            return super().covariances(X)
        if self.cache is not None:
            return self.cache.lookup(self.representation, self.band, X)
        return compute_matrices(X, self.band, self.representation)


@contextmanager
def representation_factory(representation, cache):
    from bci.models import forest, logreg
    old_forest, old_logreg, old_sub = forest.RiemannFeatures, logreg.RiemannFeatures, forest._sub
    factory = partial(RepresentationFeatures, representation=representation, cache=cache)
    forest.RiemannFeatures = logreg.RiemannFeatures = factory
    if representation == "delay1_oas":
        forest._sub = lambda cov, channels: old_sub(cov, expanded_channels(channels))
    try:
        yield
    finally:
        forest.RiemannFeatures, logreg.RiemannFeatures, forest._sub = old_forest, old_logreg, old_sub


def fit_model(train, model_name, representation, *, cache=None, n_trees=50, n_channels=None):
    from bci.models import forest, logreg
    with representation_factory(representation, cache):
        if model_name == "logreg":
            return logreg.fit(train, C=.01, prior_weight=5, seed=0)
        if model_name not in ("forest12", "forest20"):
            raise ValueError(f"Unknown model: {model_name}")
        channels = {"forest12": 12, "forest20": 20}[model_name] if n_channels is None else n_channels
        model = forest.fit(train, n_trees=n_trees, n_channels=channels, C=.01, prior_weight=5, seed=0)
        if representation == "delay1_oas":
            for tree in model.trees:
                tree.physical_channels = tree.channels.copy()
                tree.channels = expanded_channels(tree.physical_channels)
        return model
