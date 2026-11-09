"""Five published SPD estimators as controlled causal EEG reference updaters.

This adapts estimator definitions, not five full published EEG systems. The
original training prior, features, classifier and randomization are retained.
"""
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Brain-Decoding/track2_bci"))
from bci.riemann import RiemannFeatures, SessionRecentering, expm, invsqrtm, logm, upper
from pyriemann.geometry.base import sqrtm
from pyriemann.geometry.geodesic import geodesic_riemann, geodesic_wasserstein

CONTROL = "logeuclid_cumulative"
CANDIDATES = ("harmonic_cumulative", "poweuclid_half", "kullback_sym",
              "logchol_cumulative", "wasserstein_recursive")
DEFINITIONS = {
    "harmonic_cumulative": "H_t=((5*M0^-1+sum(R_i^-1))/(5+t))^-1; M_t=H_t",
    "poweuclid_half": "M_t=((5*sqrt(M0)+sum(sqrt(R_i)))/(5+t))^2; power-Euclidean p=.5",
    "kullback_sym": "A_t=(5*M0+sum(R_i))/(5+t); M_t=geodesic_riemann(A_t,H_t,.5)",
    "logchol_cumulative": "C_i=chol(R_i); average strict lower(C_i) and log(diag(C_i)), prior mass=5; M=Cbar*Cbar.T",
    "wasserstein_recursive": "M_t=geodesic_wasserstein(M_(t-1),R_t,1/(5+t))",
}
SOURCES = {
    "harmonic_cumulative": {"authors": "Moakher & Batchelor (2006); Nielsen (2024)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11675394/", "location": "Appendix A, Proposition A1: harmonic mean H"},
    "poweuclid_half": {"authors": "Dryden, Pennec & Peyrat (2010)",
        "url": "https://arxiv.org/abs/1009.3045", "location": "Section 2.1, power-Euclidean mean; p=.5 fixed here"},
    "kullback_sym": {"authors": "Moakher & Batchelor (2006); Nielsen (2024)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11675394/", "location": "Theorem A1, equation A5: centered-Gaussian Jeffreys centroid A#H"},
    "logchol_cumulative": {"authors": "Lin (2019)",
        "url": "https://arxiv.org/abs/1908.09326", "location": "Section 4.2, equation 4.3: Log-Cholesky average"},
    "wasserstein_recursive": {"authors": "Bhatia, Jain & Lim (2019)",
        "url": "https://arxiv.org/abs/1712.01504", "location": "Section 4, equations 38–41: BW geodesic; inductive streaming adaptation here"},
}


class LiteratureSession:
    def __init__(self, log_prior, prior_weight, variant):
        if variant not in CANDIDATES or prior_weight < 0:
            raise ValueError("Unknown literature reference or invalid prior mass")
        self.variant, self.log_prior = variant, np.asarray(log_prior)
        self.n = float(prior_weight)
        self.reference = expm(self.log_prior)
        self.eye = np.eye(self.reference.shape[0])
        if variant in ("harmonic_cumulative", "kullback_sym"):
            self.inverse_sum = prior_weight * np.linalg.solve(self.reference, self.eye)
        if variant == "kullback_sym":
            self.arithmetic_sum = prior_weight * self.reference
        if variant == "poweuclid_half":
            self.root_sum = prior_weight * sqrtm(self.reference)
        if variant == "logchol_cumulative":
            chol = np.linalg.cholesky(self.reference)
            self.lower_sum = prior_weight * np.tril(chol, -1)
            self.log_diagonal_sum = prior_weight * np.log(np.diag(chol))

    def __call__(self, cov):
        self.n += 1
        if self.variant in ("harmonic_cumulative", "kullback_sym"):
            self.inverse_sum = self.inverse_sum + np.linalg.solve(cov, self.eye)
            harmonic = np.linalg.solve(self.inverse_sum / self.n, self.eye)
            if self.variant == "harmonic_cumulative":
                self.reference = harmonic
            else:
                self.arithmetic_sum = self.arithmetic_sum + cov
                self.reference = geodesic_riemann(self.arithmetic_sum / self.n, harmonic, alpha=.5)
        elif self.variant == "poweuclid_half":
            self.root_sum = self.root_sum + sqrtm(cov)
            root_mean = self.root_sum / self.n
            self.reference = root_mean @ root_mean
        elif self.variant == "logchol_cumulative":
            chol = np.linalg.cholesky(cov)
            self.lower_sum = self.lower_sum + np.tril(chol, -1)
            self.log_diagonal_sum = self.log_diagonal_sum + np.log(np.diag(chol))
            factor = self.lower_sum / self.n + np.diag(np.exp(self.log_diagonal_sum / self.n))
            self.reference = factor @ factor.T
        else:
            self.reference = geodesic_wasserstein(self.reference, cov, alpha=1/self.n)
        log_ref = logm(self.reference)
        # Identical feature definition/fingerprint to the original code.
        W = invsqrtm(expm(log_ref))
        return upper(logm(W @ cov @ W)), upper(log_ref - self.log_prior)


class LiteratureFeatures(RiemannFeatures):
    def __init__(self, band=(4, 40), prior_weight=5, *, variant=CONTROL):
        if variant not in (CONTROL,) + CANDIDATES:
            raise ValueError(f"Unknown reference: {variant}")
        super().__init__(band, prior_weight)
        self.variant = variant

    def new_session(self):
        if self.variant == CONTROL:
            return SessionRecentering(self.log_prior, self.prior_weight)
        return LiteratureSession(self.log_prior, self.prior_weight, self.variant)


@contextmanager
def reference_factory(variant):
    from bci.models import forest, logreg
    old_forest, old_logreg = forest.RiemannFeatures, logreg.RiemannFeatures
    factory = partial(LiteratureFeatures, variant=variant)
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
        return forest.fit(train, n_trees=n_trees, n_channels=channels, C=.01, prior_weight=5, seed=0)
