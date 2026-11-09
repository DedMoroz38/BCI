"""Disclosed reconstruction of section 5: original synthetic code/seeds unavailable.

Uses the stated sample sizes, exact matrix logarithms, sklearn logistic regression,
and theoretical coordinate inclusion weights. Unspecified choices are fixed below
before execution and saved in the manifest; this is not an exact numeric replication.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H = 16


def pack(a):
    i, j = np.triu_indices(a.shape[-1])
    return a[..., i, j] * np.where(i == j, 1., np.sqrt(2.))


def unpack(x, h):
    i, j = np.triu_indices(h)
    a = np.zeros(x.shape[:-1] + (h, h))
    a[..., i, j] = x / np.where(i == j, 1., np.sqrt(2.))
    a[..., j, i] = a[..., i, j]
    return a


def matrix_function(a, function):
    values, vectors = np.linalg.eigh(a)
    if function is np.log and np.any(values <= 0):
        raise ValueError("Matrix logarithm requires SPD input")
    return (vectors * function(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


def subset_features(covariances, channels):
    return pack(matrix_function(covariances[..., channels, :][..., channels], np.log))


def eta(h, m):
    return m * ((m - 1)**2 + 2 * (h - 1)) / (h * (h - 1) * (m + 1))


def task(seed, sigma, direction="gaussian", n_train=400, n_test=2000):
    rng = np.random.default_rng(seed)
    d = H * (H + 1) // 2
    i, j = np.triu_indices(H)
    u = rng.normal(size=d)
    if direction == "equal":
        u = np.where(u >= 0, 1., -1.)
    elif direction == "diagonal":
        u[i != j] = 0
    u /= np.linalg.norm(u)
    labels, features, covariances = [], [], []
    for n in [n_train, n_test]:
        y = np.arange(n) % 2
        rng.shuffle(y)
        x = sigma * ((2*y[:, None]-1)*u + rng.normal(size=(n, d)))
        labels.append(y)
        features.append(x)
        covariances.append(matrix_function(unpack(x, H), np.exp))
    return rng, labels, features, covariances


def subsets(rng, m, k):
    return [np.sort(rng.choice(H, m, replace=False)) for _ in range(k)]


def inclusion(m, channel_sets=None):
    i, j = np.triu_indices(H)
    if channel_sets is None:
        return np.where(i == j, m/H, m*(m-1)/(H*(H-1)))
    return np.mean([np.isin(i, s) & np.isin(j, s) for s in channel_sets], axis=0)


def fit_decisions(covs, labels, channel_sets, c):
    from sklearn.linear_model import LogisticRegression
    all_covs = np.concatenate(covs)
    n_train = len(labels[0])
    decisions, probabilities = [], []
    # Full-channel models repeat the exact same features: cache the previous subset only.
    previous_key, previous_features = None, None
    for channels in channel_sets:
        key = tuple(channels)
        x = previous_features if key == previous_key else subset_features(all_covs, channels)
        previous_key, previous_features = key, x
        clf = LogisticRegression(C=float(c), max_iter=2000, solver="lbfgs")
        clf.fit(x[:n_train], labels[0])
        decisions.append(clf.decision_function(x[n_train:]))
        probabilities.append(clf.predict_proba(x[n_train:])[:, 1])
    return np.asarray(decisions), np.asarray(probabilities)


def residuals(decisions, y):
    r = decisions.copy()
    for label in np.unique(y):
        r[:, y == label] -= r[:, y == label].mean(axis=1, keepdims=True)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/synthetic")
    parser.add_argument("--section", choices=["all", "correlation", "variance", "equivalence", "power"], default="all")
    args = parser.parse_args()
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "reconstruction, original synthetic source/seeds unavailable",
        "h": H, "train": 400, "test": 2000, "C_correlation_variance": .001,
        "correlation": {"m": list(range(2, 17, 2)), "tasks": 5, "trees": 50},
        "variance": {"m": [4, 8, 12], "trees_pool": 200, "sigma": .15,
                     "K": [1, 2, 3, 5, 8, 12, 20, 30, 50], "ensemble_resamples": 200},
        "equivalence": {"tasks": 3, "trees": 200, "m": 8, "C": [1e-4, 1e-3, 1e-2, .1, 1., 10.]},
        "power": {"tasks": 10, "trees": 50, "sigma": .1, "train": 60,
                  "test": 2000, "m": [4, 8, 12], "C": [.01, .1, 1., 10., 100.]},
        "unspecified_choices": [
            "Balanced, shuffled labels; Gaussian direction normalized to unit norm",
            "Equal-modulus direction has seeded random signs",
            "Task seeds: correlation 1000+repeat; variance 2000; equivalence 3000+repeat; power 4000+repeat",
            "50 trees for correlation; 200-tree empirical pool for variance",
            "Variance: 200 independent ensembles sampled WITH replacement from the empirical pool",
            "Equivalence relative error = RMS(logit difference) / std(structured logits)",
            "Synthetic standard deviations use numpy ddof=0; sklearn default tolerance (as paper unspecified)",
            "Table 1 forest averages probabilities, structured model uses theoretical inclusion probabilities",
            "Table 1 chooses best C on test, as explicitly done in the paper; no further tuning",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with threadpool_limits(limits=1):
        if args.section in ["all", "correlation"]:
            rows = []
            for direction, sigma in [("equal", .02), ("gaussian", .02), ("gaussian", .15)]:
                for repeat in range(5):
                    rng, ys, xs, covs = task(1000+repeat, sigma, direction)
                    for m in range(2, 17, 2):
                        decisions, _ = fit_decisions(covs, ys, subsets(rng, m, 50), .001)
                        r = residuals(decisions, ys[1])
                        cov = r @ r.T / r.shape[1]
                        k = len(r)
                        rho = (cov.sum()-np.trace(cov))/(k*(k-1)) / np.diag(cov).mean()
                        rows.append(dict(direction=direction, sigma=sigma, repeat=repeat,
                                         m=m, correlation=rho, theory=eta(H, m)))
                    pd.DataFrame(rows).to_csv(out / "correlation.csv", index=False)
                    print(f"correlation: {direction} sigma={sigma}, repeat={repeat} done", flush=True)

        if args.section in ["all", "variance"]:
            rng, ys, xs, covs = task(2000, .15)
            rows = []
            for m in [4, 8, 12]:
                decisions, _ = fit_decisions(covs, ys, subsets(rng, m, 200), .001)
                r = residuals(decisions, ys[1])
                denominator = np.mean(r*r)
                gram = r @ r.T / r.shape[1]
                for k in manifest["variance"]["K"]:
                    indices = rng.integers(len(r), size=(200, k))
                    observed = np.array([gram[np.ix_(s, s)].mean()/denominator for s in indices])
                    rows.append(dict(m=m, K=k, variance_ratio=observed.mean(),
                                     resampling_std=observed.std(), theory=eta(H, m)+(1-eta(H, m))/k))
                pd.DataFrame(rows).to_csv(out / "variance.csv", index=False)
                print(f"variance: m={m} done", flush=True)

        if args.section in ["all", "equivalence"]:
            rows = []
            for sigma in [.02, .15]:
                for repeat in range(3):
                    rng, ys, xs, covs = task(3000+repeat, sigma)
                    ss = subsets(rng, 8, 200)
                    scale = np.sqrt(inclusion(8, ss))
                    # Features for the 200 exact trees are reused across the six C values.
                    all_covs = np.concatenate(covs)
                    exact = [subset_features(all_covs, s) for s in ss]
                    for c in manifest["equivalence"]["C"]:
                        logits = []
                        for x in exact:
                            clf = LogisticRegression(C=c, max_iter=2000)
                            clf.fit(x[:400], ys[0])
                            logits.append(clf.decision_function(x[400:]))
                        clf = LogisticRegression(C=c, max_iter=2000)
                        clf.fit(xs[0]*scale, ys[0])
                        structured = clf.decision_function(xs[1]*scale)
                        delta = np.sqrt(np.mean((np.mean(logits, axis=0)-structured)**2))/structured.std()
                        rows.append(dict(sigma=sigma, repeat=repeat, C=c, relative_error=delta))
                    pd.DataFrame(rows).to_csv(out / "equivalence.csv", index=False)
                    print(f"equivalence: sigma={sigma}, repeat={repeat} done", flush=True)

        if args.section in ["all", "power"]:
            rows = []
            for repeat in range(10):
                rng, ys, xs, covs = task(4000+repeat, .1, "diagonal", n_train=60)
                for m in [4, 8, 12]:
                    ss = subsets(rng, m, 50)
                    all_covs = np.concatenate(covs)
                    exact = [subset_features(all_covs, s) for s in ss]
                    scale = np.sqrt(inclusion(m))
                    for c in manifest["power"]["C"]:
                        p = []
                        for x in exact:
                            clf = LogisticRegression(C=c, max_iter=2000)
                            clf.fit(x[:60], ys[0])
                            p.append(clf.predict_proba(x[60:])[:, 1])
                        rows.append(dict(repeat=repeat, m=m, C=c, method="forest",
                                         accuracy=np.mean((np.mean(p, axis=0)>.5)==ys[1])))
                        clf = LogisticRegression(C=c, max_iter=2000)
                        clf.fit(xs[0]*scale, ys[0])
                        rows.append(dict(repeat=repeat, m=m, C=c, method="structured",
                                         accuracy=clf.score(xs[1]*scale, ys[1])))
                        if m == 4:
                            clf = LogisticRegression(C=c, max_iter=2000)
                            clf.fit(xs[0], ys[0])
                            rows.append(dict(repeat=repeat, m=16, C=c, method="full",
                                             accuracy=clf.score(xs[1], ys[1])))
                pd.DataFrame(rows).to_csv(out / "power_grid.csv", index=False)
                print(f"power: repeat={repeat} done", flush=True)
            grid = pd.DataFrame(rows)
            best = grid.loc[grid.groupby(["repeat", "method", "m"]).accuracy.idxmax()]
            best.to_csv(out / "power_best.csv", index=False)
            stats = best.groupby(["method", "m"]).accuracy.agg(["mean", lambda a: a.std(ddof=0)])
            stats.columns = ["mean", "std"]
            stats.to_csv(out / "table1.csv")
            print(stats.to_string(), flush=True)

    # Plot only after actual data files exist; theory is separate dashed lines.
    if all((out / f"{name}.csv").exists() for name in ["correlation", "variance", "equivalence"]):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        a = pd.read_csv(out / "correlation.csv")
        ms = np.arange(1, 17)
        axes[0].plot(ms, [eta(H, int(m)) for m in ms], "k-", label="theory")
        for (direction, sigma), g in a.groupby(["direction", "sigma"]):
            stats = g.groupby("m").correlation.agg(["mean", lambda v: v.std(ddof=0)])
            axes[0].errorbar(stats.index, stats.iloc[:, 0], yerr=stats.iloc[:, 1], fmt="o", ms=3,
                             label=f"{direction}, sigma={sigma}")
        axes[0].set(xlabel="Channels per tree m", ylabel="Tree correlation")
        axes[0].legend(fontsize=7)
        v = pd.read_csv(out / "variance.csv")
        for m, g in v.groupby("m"):
            line, = axes[1].plot(g.K, g.variance_ratio, label=f"m={m}")
            axes[1].plot(g.K, g.theory, "--", c=line.get_color())
        axes[1].set(xscale="log", yscale="log", xlabel="Trees K", ylabel="Relative variance")
        axes[1].legend(fontsize=8)
        e = pd.read_csv(out / "equivalence.csv")
        for sigma, g in e.groupby("sigma"):
            stats = g.groupby("C").relative_error.agg(["mean", lambda v: v.std(ddof=0)])
            axes[2].errorbar(stats.index, stats.iloc[:, 0], yerr=stats.iloc[:, 1], marker="o",
                             label=f"sigma={sigma}")
        axes[2].set(xscale="log", yscale="log", xlabel="C = 1 / (lambda N)", ylabel="Relative logit error")
        axes[2].legend(fontsize=8)
        fig.suptitle("Section 5 reconstruction — original synthetic seeds/code unavailable", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figure1.png", dpi=180)
        fig.savefig(out / "figure1.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
