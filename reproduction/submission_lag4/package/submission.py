"""Track02: trained Riemannian forest, lag4 OAS, cumulative log-Euclidean prior.

Self-contained inference. No project modules, data downloads, fitting, cache,
pickle or pyriemann dependency. Trained numeric parameters live in weights.npz.
"""
from pathlib import Path

import numpy as np
from scipy.signal import sosfiltfilt
from scipy.special import softmax
from sklearn.covariance import oas
from benchmark_utils.base_solver import CompetSolver


def _matrix_function(matrix, function, positive=False):
    values, vectors = np.linalg.eigh(matrix)
    if positive and (not np.isfinite(values).all() or values.min() <= 0):
        raise ValueError('Expected a finite positive definite covariance matrix')
    return vectors @ (function(values)[:, None] * vectors.T)


def _logm(matrix):
    return _matrix_function(matrix, np.log, positive=True)


class RiemannLag4:
    """CPU inference; each predict call advances only the current session."""

    def __init__(self, weights_path, meta):
        with np.load(weights_path, allow_pickle=False) as weights:
            self.weights = {name: weights[name].copy() for name in weights.files}
        w = self.weights
        if int(w['format_version']) != 1:
            raise ValueError('Unsupported weights format')
        self.sfreq = float(w['sfreq'])
        self.lag = int(w['lag'])
        self.prior_weight = float(w['prior_weight'])
        self.ch_names = list(w['ch_names'].astype(str))
        self.n_chans = len(self.ch_names)
        self.classes = w['classes'].astype(np.int64)
        if int(meta['n_chans']) != self.n_chans:
            raise ValueError('Channel count differs from trained weights')
        if int(meta['n_classes']) != len(self.classes):
            raise ValueError('Class count differs from trained weights')
        if not np.isclose(float(meta['sfreq']), self.sfreq, rtol=0, atol=1e-8):
            raise ValueError('Sampling frequency differs from trained weights')
        incoming = list(meta.get('ch_names', self.ch_names))
        if len(incoming) != self.n_chans or len(set(incoming)) != self.n_chans or set(incoming) != set(self.ch_names):
            raise ValueError('Channel names differ from trained weights')
        self.input_order = np.array([incoming.index(name) for name in self.ch_names])
        self.channels = w['channels'].astype(np.int64)
        self.tree_bands = w['tree_bands'].astype(np.int64)
        self.n_trees, dimension = self.channels.shape
        self.upper_indices = np.triu_indices(dimension)
        self.upper_weights = np.where(self.upper_indices[0] == self.upper_indices[1], 1., np.sqrt(2.))
        if w['log_prior'].shape != (self.n_trees, dimension, dimension):
            raise ValueError('Invalid prior dimensions')
        n_features = dimension * (dimension + 1) // 2
        if w['coef'].shape != (self.n_trees, len(self.classes), n_features):
            raise ValueError('Invalid classifier dimensions')
        if not np.all(w['scaler_scale'] > 0):
            raise ValueError('Invalid scaler scales')
        self.reset_state()

    def reset_state(self):
        self.log_sums = self.prior_weight * self.weights['log_prior']
        self.count = self.prior_weight

    def covariances(self, window):
        x = np.asarray(window, dtype=np.float64)[self.input_order]
        matrices = []
        for sos in self.weights['sos_bank']:
            filtered = sosfiltfilt(sos, x, axis=-1)
            z = np.stack([filtered[:, self.lag:], filtered[:, :-self.lag]], axis=-2)
            z = z.reshape(2 * self.n_chans, filtered.shape[-1] - self.lag)
            # Full82-channel OAS before selecting physical subspaces, as trained.
            matrices.append(oas(z.T)[0])
        return matrices

    def predict_proba(self, X):
        if hasattr(X, 'detach'):
            X = X.detach().cpu().numpy()
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1] != self.n_chans or X.shape[-1] <= self.lag:
            raise ValueError('Expected EEG windows of shape (B, trained_channels, T)')
        if not np.isfinite(X).all():
            raise ValueError('EEG input must be finite')
        results = []
        w = self.weights
        for window in X:
            covs = self.covariances(window)
            self.count += 1
            probabilities = []
            for k, channels in enumerate(self.channels):
                cov = covs[self.tree_bands[k]][np.ix_(channels, channels)]
                self.log_sums[k] = self.log_sums[k] + _logm(cov)
                log_ref = self.log_sums[k] / self.count
                reference = _matrix_function(log_ref, np.exp)
                whitening = _matrix_function(reference, lambda a: 1. / np.sqrt(a), positive=True)
                tangent = _logm(whitening @ cov @ whitening)
                features = tangent[self.upper_indices] * self.upper_weights
                features = (features - w['scaler_mean'][k]) / w['scaler_scale'][k]
                logits = features[None] @ w['coef'][k].T + w['intercept'][k]
                probabilities.append(softmax(logits, axis=1)[0])
            results.append(np.mean(probabilities, axis=0))
        return np.asarray(results).reshape(-1, len(self.classes))

    def predict(self, X):
        return self.classes[self.predict_proba(X).argmax(axis=1)].astype(np.int64)


class Solver(CompetSolver):
    name = 'Riemann-Forest-Lag4'
    requirements = []

    def load_model(self, meta):
        return RiemannLag4(Path(meta['submission_dir']) / 'weights.npz', meta)
