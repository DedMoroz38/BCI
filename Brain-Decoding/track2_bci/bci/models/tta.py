"""Test-time adaptation of the forest's heads with pseudo-labels, inside one stream.

Causal self-training (in the spirit of online TTA for motor imagery, Wimpff et
al. 2023, and SPDIM/OSPDIM): every ``every`` windows, each tree's softmax head
(W, b) takes a few gradient steps on

    mean CE(W x_i + b, y_hat_i)  +  lam / 2 * ||(W, b) - (W0, b0)||^2

over the windows seen so far whose forest prediction is confident. Selection is
class-balanced (the most confident ``q`` fraction of each predicted class), so
the adaptation cannot collapse onto one class; the anchor to the trained head
(W0, b0) keeps it from drifting. The window being predicted is never used for
its own prediction.
"""

import numpy as np
from scipy.special import softmax


def _gd(X, Y, W, b, W0, b0, lam, steps):
    """Gradient descent on the anchored cross-entropy; step 1/L with L from the data."""
    n = len(X)
    v = np.ones(X.shape[1]) / np.sqrt(X.shape[1])
    for _ in range(10):  # power iteration: top eigenvalue of X^T X / n
        v = X.T @ (X @ v) / n
        v /= np.linalg.norm(v) + 1e-12
    L = 0.5 * (v @ (X.T @ (X @ v)) / n + 1.0) + lam  # softmax Hessian <= 1/2 (data + bias)
    for _ in range(steps):
        G = softmax(X @ W.T + b, axis=1) - Y
        W = W - (G.T @ X / n + lam * (W - W0)) / L
        b = b - (G.mean(0) + lam * (b - b0)) / L
    return W, b


def select(conf, y_hat, q):
    """Indices of the most confident ``q`` fraction of each predicted class."""
    keep = []
    for c in np.unique(y_hat):
        idx = np.flatnonzero(y_hat == c)
        k = max(1, int(round(q * len(idx))))
        keep.append(idx[np.argsort(-conf[idx])[:k]])
    return np.sort(np.concatenate(keep))


def adapt_stream(feats, heads, q=0.5, lam=1.0, every=10, warmup=20, steps=30, n_classes=3):
    """Causal predictions of a forest on one stream with pseudo-label adaptation.

    feats: list over trees of (N, d_k) features of the stream, in recording order.
    heads: list over trees of (coef (n_classes, d_k), intercept (n_classes,)).
    Returns (y_pred (N,), forest probabilities (N, n_classes)).
    """
    W0 = [np.array(c, float) for c, _ in heads]
    b0 = [np.array(i, float) for _, i in heads]
    W, b = [w.copy() for w in W0], [v.copy() for v in b0]
    N = len(feats[0])
    P = np.zeros((N, n_classes))
    for i in range(N):
        P[i] = np.mean([softmax(Wk @ xk[i] + bk) for Wk, bk, xk in zip(W, b, feats)], axis=0)
        seen = i + 1
        if seen >= warmup and seen % every == 0:
            # pseudo-labels of all windows seen so far, by the current heads
            P_seen = np.mean([softmax(xk[:seen] @ Wk.T + bk, axis=1) for Wk, bk, xk in zip(W, b, feats)], axis=0)
            y_hat = P_seen.argmax(1)
            keep = select(P_seen.max(1), y_hat, q)
            Y = np.eye(n_classes)[y_hat[keep]]
            for k, xk in enumerate(feats):
                W[k], b[k] = _gd(xk[keep], Y, W[k], b[k], W0[k], b0[k], lam, steps)
    return P.argmax(1), P
