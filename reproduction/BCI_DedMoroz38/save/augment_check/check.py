"""Later-sessions dip check: plain vs lag at K=50/100 with 6 seeds, + LOSO for lag K=100.

Memory-safe: window-local covariances cached once (chunked), memory-mapped by
<=4 workers; each job appended to results.csv as it finishes (resumable).
    python check.py OUT
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed

OUT = sys.argv[1]
sys.path.insert(0, "/Users/egormaksimov/Documents/PROGRAMMING/BCI/Brain-Decoding/track2_bci")
DATA = "/Users/egormaksimov/Documents/PROGRAMMING/BCI/data"
from bci.data import Windows, concat, load_split
from bci.evaluation import later_sessions, score
from bci.models import forest
from bci.riemann import RiemannFeatures

CONFIGS = {"plain50": (None, 50), "lag50": ("lag", 50), "plain100": (None, 100), "lag100": ("lag", 100)}
LATER = "later sessions"


def cov_path(band, aug):
    return f"{OUT}/covs/{band[0]}-{band[1]}_{aug or 'plain'}.npy"


def dev_meta():
    dev = concat(load_split("train", DATA), load_split("val", DATA))
    fields = {k: getattr(dev, k) for k in Windows.__dataclass_fields__ if k not in ("X", "ch_names")}
    return Windows(X=np.zeros((len(dev), dev.X.shape[1], 1), np.float32), **fields, ch_names=dev.ch_names)


def precompute():
    dev = concat(load_split("train", DATA), load_split("val", DATA))
    for band, aug in forest.views(forest.BANDS, "lag"):
        if not os.path.exists(cov_path(band, aug)):
            f = RiemannFeatures(band, 5, aug)
            np.save(cov_path(band, aug), np.concatenate([f.covariances(dev.X[i:i + 200]) for i in range(0, len(dev), 200)]))


class FoldCovs:
    def __init__(self, paths, idx):
        self.paths, self.idx = paths, idx

    def __getitem__(self, view):
        return np.load(self.paths[view], mmap_mode="r")[self.idx]


def job(config, seed, fold):
    dev = dev_meta()
    if fold == LATER:
        _, _, test = next(later_sessions(dev))
        key = lambda w: list(zip(w.stream, w.run, w.onset))
        test_keys = set(key(test))
        assert len(test_keys) == len(test)
        is_test = np.array([k in test_keys for k in key(dev)])
    else:
        is_test = dev.subject == fold
    train, test = dev.select(~is_test), dev.select(is_test)
    augment, n_trees = CONFIGS[config]
    paths = [cov_path(b, a) for b, a in forest.views(forest.BANDS, augment)]
    t0 = time.time()
    model = forest.fit(train, n_trees=n_trees, n_channels=20, C=0.01, augment=augment, seed=seed,
                       covs=FoldCovs(paths, np.flatnonzero(~is_test)))
    P = model.tree_probas(test, covs=FoldCovs(paths, np.flatnonzero(is_test)))
    row = dict(config=config, seed=seed, fold=fold, minutes=(time.time() - t0) / 60,
               **score(test, P.mean(0).argmax(-1)))
    pd.DataFrame([row]).to_csv(f"{OUT}/results.csv", mode="a", index=False,
                               header=not os.path.exists(f"{OUT}/results.csv"))


if __name__ == "__main__":
    precompute()
    subjects = sorted(set(dev_meta().subject))
    jobs = [("lag100", s, f) for s in (0, 1) for f in subjects]                 # slowest first
    jobs += [(c, s, LATER) for s in range(6) for c in CONFIGS]
    done = set()
    if os.path.exists(f"{OUT}/results.csv"):
        done = set(map(tuple, pd.read_csv(f"{OUT}/results.csv")[["config", "seed", "fold"]].astype(str).values))
    jobs = [j for j in jobs if tuple(map(str, j)) not in done]
    print(f"{len(jobs)} jobs", flush=True)
    Parallel(n_jobs=4, verbose=10)(delayed(job)(*j) for j in jobs)
