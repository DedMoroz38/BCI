"""Step 3: pseudo-label test-time adaptation on top of a forest (bci/models/tta.py).

One job = one (base config, seed, fold): fit the forest once, then score the
unadapted forest and every TTA variant on the same trees (paired).
    python tta.py OUT SEEDS FOLDS N_JOBS     # e.g. tta.py . 0 loso 2
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed

OUT, SEEDS, FOLDS, N_JOBS = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3], int(sys.argv[4])
BASE = sys.argv[5] if len(sys.argv) > 5 else "lag100"
sys.argv = [sys.argv[0], OUT]
exec(open("/Users/egormaksimov/Documents/PROGRAMMING/BCI/save/next_steps/run.py").read().split("\nif __name__")[0])
from bci.models.forest import _sub
from bci.models.tta import adapt_stream

VARIANTS = {f"q{q}_lam{lam}": dict(q=q, lam=lam) for q in (0.3, 0.6) for lam in (0.1, 1.0)}


def tta_job(base, seed, fold):
    dev = dev_meta()
    is_test = split(dev, fold)
    train, test = dev.select(~is_test), dev.select(is_test)
    augment, n_trees, stretch = CONFIGS[base]
    paths = [cov_path(b, a) for b, a in forest.views(forest.BANDS, augment)]
    model = forest.fit(train, n_trees=n_trees, n_channels=20, C=0.01, augment=augment, seed=seed,
                       covs=FoldCovs(paths, np.flatnonzero(~is_test)), stretch=stretch)
    test_covs = FoldCovs(paths, np.flatnonzero(is_test))
    feats = [t.features.transform(test, _sub(test_covs[t.view], t.rows))[0] for t in model.trees]
    heads = [(t.clf.coef_, t.clf.intercept_) for t in model.trees]
    rows = []
    for name, kw in {"none": None, **VARIANTS}.items():
        t0 = time.time()
        pred = np.empty(len(test), np.int64)
        for idx in test.streams():
            f = [x[idx] for x in feats]
            if kw is None:
                P = np.mean([t.clf.predict_proba(x) for t, x in zip(model.trees, f)], axis=0)
                pred[idx] = P.argmax(1)
            else:
                pred[idx] = adapt_stream(f, heads, **kw)[0]
        rows.append(dict(base=base, variant=name, seed=seed, fold=fold,
                         sec=time.time() - t0, **score(test, pred)))
    pd.DataFrame(rows).to_csv(f"{OUT}/tta_results.csv", mode="a", index=False,
                              header=not os.path.exists(f"{OUT}/tta_results.csv"))


if __name__ == "__main__":
    folds = sorted(set(dev_meta().subject)) if FOLDS == "loso" else [LATER]
    jobs = [(BASE, s, f) for s in SEEDS for f in folds]
    if os.path.exists(f"{OUT}/tta_results.csv"):
        done = set(map(tuple, pd.read_csv(f"{OUT}/tta_results.csv")[["base", "seed", "fold"]].astype(str).values))
        jobs = [j for j in jobs if tuple(map(str, j)) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=10)(delayed(tta_job)(*j) for j in jobs)
