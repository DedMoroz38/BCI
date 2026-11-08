"""Steps 1-2 of the improvement plan (2026-11-07), forest K=100, m=20, C=0.01.

configs: lag100 (baseline), lag100_stretch (per-session stretching, weight 10),
lag3_100 (order-3 augmented covariance), randlag100 (random lag 1-6 samples per tree).
LOSO seeds 0-3, later sessions seeds 0-5. lag100 rows for seeds already in
save/augment_check/results.csv are reused (same code path, same seeds).

Memory-safe: covariances cached once per view (new ones float32), memory-mapped by
<=4 workers; each job appended to results.csv as it finishes (resumable).
    python run.py OUT
"""
import os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd
from joblib import Parallel, delayed

OUT = sys.argv[1]
OLD = "/Users/egormaksimov/Documents/PROGRAMMING/BCI/save/augment_check"
sys.path.insert(0, "/Users/egormaksimov/Documents/PROGRAMMING/BCI/Brain-Decoding/track2_bci")
DATA = "/Users/egormaksimov/Documents/PROGRAMMING/BCI/data"
from bci.data import Windows, concat, load_split
from bci.evaluation import later_sessions, score
from bci.models import forest
from bci.riemann import RiemannFeatures

CONFIGS = {  # name: (augment, n_trees, stretch)
    "lag100": ("lag", 100, None),
    "lag100_stretch": ("lag", 100, 10),
    "lag3_100": ("lag3", 100, None),
    "randlag100": ("randlag", 100, None),
    "lag3_100_stretch": ("lag3", 100, 10),  # added 2026-11-08: the combination of the two winners
}
LATER = "later sessions"


def cov_path(band, aug):
    name = f"{band[0]}-{band[1]}_{aug or 'plain'}.npy"
    old = f"{OLD}/covs/{name}"  # float64 caches of the earlier run (plain + lag)
    return old if os.path.exists(old) else f"{OUT}/covs/{name.replace('@', 'at')}"


def dev_windows():
    return concat(load_split("train", DATA), load_split("val", DATA))


def dev_meta():
    dev = dev_windows()
    fields = {k: getattr(dev, k) for k in Windows.__dataclass_fields__ if k not in ("X", "ch_names")}
    return Windows(X=np.zeros((len(dev), dev.X.shape[1], 1), np.float32), **fields, ch_names=dev.ch_names)


def precompute():
    os.makedirs(f"{OUT}/covs", exist_ok=True)
    needed = {v for aug, _, _ in CONFIGS.values() for v in forest.views(forest.BANDS, aug)}
    todo = [v for v in sorted(needed, key=str) if not os.path.exists(cov_path(*v))]
    if not todo:
        return
    dev = dev_windows()
    for band, aug in todo:
        f = RiemannFeatures(band, 5, aug)
        covs = np.concatenate([f.covariances(dev.X[i:i + 200]) for i in range(0, len(dev), 200)])
        np.save(cov_path(band, aug), covs.astype(np.float32))
        print("cached", band, aug, covs.shape, flush=True)


class FoldCovs:
    def __init__(self, paths, idx):
        self.paths, self.idx = paths, idx

    def __getitem__(self, view):
        return np.asarray(np.load(self.paths[view], mmap_mode="r")[self.idx], np.float64)


def split(dev, fold):
    if fold == LATER:
        _, _, test = next(later_sessions(dev))
        key = lambda w: list(zip(w.stream, w.run, w.onset))
        test_keys = set(key(test))
        return np.array([k in test_keys for k in key(dev)])
    return dev.subject == fold


def job(config, seed, fold):
    dev = dev_meta()
    is_test = split(dev, fold)
    train, test = dev.select(~is_test), dev.select(is_test)
    augment, n_trees, stretch = CONFIGS[config]
    paths = [cov_path(b, a) for b, a in forest.views(forest.BANDS, augment)]
    t0 = time.time()
    model = forest.fit(train, n_trees=n_trees, n_channels=20, C=0.01, augment=augment, seed=seed,
                       covs=FoldCovs(paths, np.flatnonzero(~is_test)), stretch=stretch)
    P = model.tree_probas(test, covs=FoldCovs(paths, np.flatnonzero(is_test)))
    row = dict(config=config, seed=seed, fold=fold, minutes=(time.time() - t0) / 60,
               **score(test, P.mean(0).argmax(-1)))
    pd.DataFrame([row]).to_csv(f"{OUT}/results.csv", mode="a", index=False,
                               header=not os.path.exists(f"{OUT}/results.csv"))


if __name__ == "__main__":
    precompute()
    subjects = sorted(set(dev_meta().subject))
    if not os.path.exists(f"{OUT}/results.csv"):  # reuse the earlier lag100 rows
        old = pd.read_csv(f"{OLD}/results.csv")
        old[old.config == "lag100"].to_csv(f"{OUT}/results.csv", index=False)
    jobs = [("lag100", 0, subjects[0] + "#recheck")]  # same seed/fold as an old row: must match it
    for c in CONFIGS:  # slowest (LOSO, lag3) first
        jobs += [(c, s, f) for s in range(4) for f in subjects]
    jobs += [(c, s, LATER) for s in range(6) for c in CONFIGS]
    jobs.sort(key=lambda j: (j[2] == LATER, j[0] != "lag3_100"))
    done = set(map(tuple, pd.read_csv(f"{OUT}/results.csv")[["config", "seed", "fold"]].astype(str).values))
    jobs = [j for j in jobs if tuple(map(str, j)) not in done]
    print(f"{len(jobs)} jobs", flush=True)

    def run(config, seed, fold):
        if fold.endswith("#recheck"):  # reproduce an old result with the new code
            dev = dev_meta(); f = fold[:-8]; is_test = dev.subject == f
            train, test = dev.select(~is_test), dev.select(is_test)
            paths = [cov_path(b, a) for b, a in forest.views(forest.BANDS, "lag")]
            m = forest.fit(train, n_trees=100, n_channels=20, C=0.01, augment="lag", seed=seed,
                           covs=FoldCovs(paths, np.flatnonzero(~is_test)))
            s = score(test, m.tree_probas(test, covs=FoldCovs(paths, np.flatnonzero(is_test))).mean(0).argmax(-1))
            pd.DataFrame([dict(config=config, seed=seed, fold=fold, minutes=0, **s)]).to_csv(
                f"{OUT}/results.csv", mode="a", index=False, header=False)
        else:
            job(config, seed, fold)

    Parallel(n_jobs=4, verbose=10)(delayed(run)(*j) for j in jobs)
