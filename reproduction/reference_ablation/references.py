"""Controlled replacements of the original causal session-reference update.

The training prior, feature definition, scaler, covariance estimation and upstream
model fitting remain unchanged. Recursive AIRM is an inductive estimator, not a
batch Frechet mean. EMA uses normalized decaying prior/window weights.
"""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Brain-Decoding/track2_bci"))
from bci.riemann import RiemannFeatures, SessionRecentering, expm, invsqrtm, logm, upper
from pyriemann.geometry.geodesic import geodesic_riemann

CONTROL = "logeuclid_cumulative"
CANDIDATES = ("euclid_cumulative", "riemann_recursive", "logeuclid_ema")
DECAY = .95


class ReferenceSession:
    def __init__(self, log_prior, prior_weight, variant, decay=DECAY):
        if variant not in CANDIDATES or not 0 < decay < 1 or prior_weight < 0:
            raise ValueError("Invalid reference variant, decay or prior weight")
        self.variant, self.decay = variant, decay
        self.log_prior = np.asarray(log_prior)
        self.n = float(prior_weight)
        self.reference = expm(self.log_prior)
        self.matrix_sum = prior_weight * self.reference
        self.log_sum = prior_weight * self.log_prior

    def __call__(self, cov):
        if self.variant == "euclid_cumulative":
            self.n += 1
            self.matrix_sum = self.matrix_sum + cov
            self.reference = self.matrix_sum / self.n
            log_ref = logm(self.reference)
        elif self.variant == "riemann_recursive":
            self.n += 1
            self.reference = geodesic_riemann(self.reference, cov, alpha=1/self.n)
            log_ref = logm(self.reference)
        else:
            self.n = self.decay * self.n + 1
            self.log_sum = self.decay * self.log_sum + logm(cov)
            log_ref = self.log_sum / self.n
            self.reference = expm(log_ref)
        # Same feature/fingerprint map as the original SessionRecentering.
        W = invsqrtm(expm(log_ref))
        return upper(logm(W @ cov @ W)), upper(log_ref - self.log_prior)


class ReferenceFeatures(RiemannFeatures):
    def __init__(self, band=(4, 40), prior_weight=5, *, variant=CONTROL, decay=DECAY):
        if variant not in (CONTROL,) + CANDIDATES:
            raise ValueError(f"Unknown reference variant: {variant}")
        super().__init__(band, prior_weight)
        self.variant, self.decay = variant, decay

    def new_session(self):
        if self.variant == CONTROL:
            return SessionRecentering(self.log_prior, self.prior_weight)
        return ReferenceSession(self.log_prior, self.prior_weight, self.variant, self.decay)


@contextmanager
def reference_factory(variant):
    """Scoped factory injection; trained objects retain their own configuration."""
    from bci.models import forest, logreg
    old_forest, old_logreg = forest.RiemannFeatures, logreg.RiemannFeatures
    factory = partial(ReferenceFeatures, variant=variant)
    forest.RiemannFeatures = logreg.RiemannFeatures = factory
    try:
        yield
    finally:
        forest.RiemannFeatures, logreg.RiemannFeatures = old_forest, old_logreg


def fit_model(train, model_name, variant, *, n_trees=50, n_channels=None):
    from bci.models import forest, logreg
    with reference_factory(variant):
        if model_name == "logreg":
            return logreg.fit(train, C=.01, prior_weight=5, seed=0)
        if model_name not in ("forest12", "forest20"):
            raise ValueError(f"Unknown model: {model_name}")
        channels = {"forest12": 12, "forest20": 20}[model_name] if n_channels is None else n_channels
        return forest.fit(train, n_trees=n_trees, n_channels=channels, C=.01,
                          prior_weight=5, seed=0)
