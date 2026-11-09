"""Supplement: assess numerical solver precision without changing the main run."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from reproduce_synthetic import task, subsets, inclusion, subset_features

ROOT = Path(__file__).resolve().parents[1]
rows = []
with threadpool_limits(limits=1):
    for sigma in [.02, .15]:
        for repeat in range(3):
            rng, ys, xs, covs = task(3000 + repeat, sigma)
            ss = subsets(rng, 8, 200)
            scale = np.sqrt(inclusion(8, ss))
            features = [subset_features(np.concatenate(covs), s) for s in ss]
            for tol in [1e-4, 1e-10]:
                logits = []
                for x in features:
                    clf = LogisticRegression(C=.001, max_iter=2000, tol=tol).fit(x[:400], ys[0])
                    logits.append(clf.decision_function(x[400:]))
                clf = LogisticRegression(C=.001, max_iter=2000, tol=tol).fit(xs[0]*scale, ys[0])
                q = clf.decision_function(xs[1]*scale)
                error = float(np.sqrt(np.mean((np.mean(logits, axis=0)-q)**2))/q.std())
                rows.append(dict(sigma=sigma, repeat=repeat, tol=tol, relative_error=error))
                print(rows[-1], flush=True)
frame = pd.DataFrame(rows)
frame.to_csv(ROOT / 'results/synthetic/tolerance_diagnostic.csv', index=False)
print(frame.groupby(['sigma', 'tol']).relative_error.mean().to_string())
