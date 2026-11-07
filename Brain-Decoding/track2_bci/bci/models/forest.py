"""Riemannian random forest: bagging + random subspaces on the SPD manifold.

Each "tree" is a small Riemannian model (session recentering + tangent space +
LogReg) that sees only
- a random view from the bank: a frequency band, optionally augmented with a
  90°-shifted copy of every channel (``augment``, see ``bci.riemann.augment``),
- a random subset of channels: the covariance sub-matrix of a channel subset
  is again SPD, so the whole Riemannian pipeline applies unchanged (an
  augmented view keeps each chosen channel together with its shifted copy),
- a bootstrap of the training *subjects* (sample weights = how many times a
  subject was drawn), which trains robustness to a change of person.

The forest averages the trees' probabilities: p(y | X) = mean_k p_k(y | X).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from bci.riemann import RiemannFeatures

# broadband + theta, alpha/mu, low beta, high beta, low gamma
BANDS = ((4, 40), (4, 8), (8, 13), (13, 20), (20, 30), (30, 40))


@dataclass
class Tree:
    view: int                  # index into the bank of views
    channels: np.ndarray       # channel subset
    rows: np.ndarray           # its rows in the view's covariance
    features: RiemannFeatures  # recentering state of this tree
    clf: LogisticRegression


class RiemannForest:
    """Track 2 contract: ``reset_state()`` + ``predict(X)`` for X of shape (1, C, T)."""

    def __init__(self, bank, trees):
        self.bank, self.trees = bank, trees
        self.reset_state()

    def reset_state(self):
        self.sessions = [t.features.new_session() for t in self.trees]

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        covs = [f.covariances(X)[0] for f in self.bank]  # one full covariance per view
        probas = []
        for tree, session in zip(self.trees, self.sessions):
            x, _ = tree.features.step(session, _sub(covs[tree.view], tree.rows))
            probas.append(tree.clf.predict_proba(x[None])[0])
        return np.array([np.mean(probas, axis=0).argmax()])

    def tree_probas(self, windows, covs=None):
        """(n_trees, N, 3) probabilities of every tree on ``windows`` (same as streaming).

        ``covs``: precomputed full covariances of ``windows``, one per view of the bank.
        """
        covs = [f.covariances(windows.X) for f in self.bank] if covs is None else covs
        probas = []
        for tree in self.trees:
            x, _ = tree.features.transform(windows, _sub(covs[tree.view], tree.rows))
            probas.append(tree.clf.predict_proba(x))
        return np.stack(probas)


def _sub(cov, rows):
    """Covariance sub-matrix of a row subset (works for (C, C) and (N, C, C))."""
    return cov[..., rows, :][..., rows]


RANDOM_LAGS = (1, 2, 3, 4, 5, 6)  # samples, for augment="randlag" (a control for the quarter period)


def views(bands=BANDS, augment=None):
    """Bank of (band, augmentation) views: every band plain, plus, with ``augment``,
    every narrow band (at most an octave) augmented; phase is not defined for a
    broad band such as 4-40 Hz. ``"randlag"``: every narrow band once per lag in
    ``RANDOM_LAGS`` (see :func:`view_weights`)."""
    bank = [(band, None) for band in bands]
    narrow = [band for band in bands if band[1] / band[0] <= 2]
    if augment == "randlag":
        bank += [(band, f"lag@{tau}") for band in narrow for tau in RANDOM_LAGS]
    elif augment is not None:
        bank += [(band, augment) for band in narrow]
    return bank


def view_weights(bank):
    """Draw probabilities of the views: uniform over (band, augmentation family), so
    ``randlag`` picks an augmented view as often as ``lag`` and then a random lag."""
    family = [(band, "lag@" if aug and aug.startswith("lag@") else aug) for band, aug in bank]
    counts = Counter(family)
    w = np.array([1.0 / counts[f] for f in family])
    return w / w.sum()


def fit(train, n_trees=50, n_channels=20, C=0.01, bands=BANDS, augment=None, prior_weight=5, seed=0,
        covs=None, stretch=None):
    """``covs``: precomputed full covariances of ``train``, one per ``views(bands, augment)``.
    ``stretch``: prior weight of per-session feature stretching (None = off)."""
    rng = np.random.default_rng(seed)
    bank_views = views(bands, augment)
    bank = [RiemannFeatures(band, prior_weight, aug, stretch) for band, aug in bank_views]
    p = view_weights(bank_views)
    if covs is None:
        covs = [f.covariances(train.X) for f in bank]  # full covariances, computed once per view
    subjects = sorted(set(train.subject))

    trees = []
    for _ in range(n_trees):
        # uniform banks keep rng.integers, so earlier seeds reproduce exactly
        view = int(rng.integers(len(bank))) if np.allclose(p, p[0]) else int(rng.choice(len(bank), p=p))
        channels = np.sort(rng.choice(train.X.shape[1], n_channels, replace=False))
        rows = bank[view].rows(channels, train.X.shape[1])
        drawn = Counter(rng.choice(subjects, len(subjects)))  # bootstrap of subjects
        weight = np.array([drawn[s] for s in train.subject], float)

        features = RiemannFeatures(bank[view].band, prior_weight, bank[view].augment, stretch)
        x, _ = features.fit_transform(train, _sub(covs[view], rows))
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(x[weight > 0], train.y[weight > 0], sample_weight=weight[weight > 0])
        trees.append(Tree(view, channels, rows, features, clf))
    return RiemannForest(bank, trees)
