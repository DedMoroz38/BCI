# Window SPD representation ablation

This experiment replaces only the per-window matrix R(X) in the original
cumulative log-Euclidean pipeline. It does not use the previous reference-update
ablations. All choices were fixed before results in `experiment.json`.

- `delay1_oas`: one 120-Hz sample delay (8.333 ms), OAS, paired copies of each physical electrode.
- `hsic_rbf`: exact 480-sample Gaussian-kernel channel Gram matrix with diagonal ridge.
- `envelope_oas`: Hilbert amplitude envelope followed by OAS.

The original code used OAS rather than a bare XXᵀ. OAS is retained for delay and
envelope variants. HSIC is the channel-Gram construction proposed in the user
conversation, based on HSIC; this is not a full published EEG classifier.

## Run

From `/Users/den1shh/Documents/diploma`, use the existing locked environment and
prepared PROTEUS data. Repeated commands resume completed work and reject changes
to frozen numerical code, source commits, data or environment.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/Users/den1shh/Documents/diploma/reproduction/.cache/matplotlib
reproduction/.venv/bin/python -u reproduction/representation_ablation/prepare_cache.py --workers 6
reproduction/.venv/bin/python -u reproduction/representation_ablation/run.py --initialize
reproduction/.venv/bin/python -u reproduction/representation_ablation/launch.py --workers 4
reproduction/.venv/bin/python -u reproduction/representation_ablation/analyze.py
```

The first command computes 18 exact per-window matrix sets. Its `.progress.json`
files permit resuming at a flushed chunk boundary. `--auxiliary --workers 2` can
also compute delay/envelope sets while the main process computes HSIC; per-file
locks avoid duplicate writes. `launch.py` waits for the six matrices of each
representation before starting that model group. Do not run two launchers for
the same model groups concurrently.

For a single model group, after its matrices are complete:

```sh
reproduction/.venv/bin/python -u reproduction/representation_ablation/run.py --variant delay1_oas --model forest20
```

Checks:

```sh
reproduction/.venv/bin/python -m pytest -q reproduction/representation_ablation/test_matrix_cache.py reproduction/representation_ablation/test_representations.py reproduction/literature_ablation/test_literature_references.py reproduction/reference_ablation/test_references.py reproduction/scripts
```

## Protocol and artifacts

Each representation has 29 evaluations: 8 LOSO folds × 3 models, 3 later-session
models, and public test for logreg/forest20 (test fit uses original train only).
The 29 original predictions are reused as unchanged controls. Analysis checks
all 116 files with the official competition metrics, along with window ordering,
subject splits, tree bands and physical channel selection.

The cache contains only a deterministic transformation of each individual raw
window. It uses no labels or statistics pooled across windows. Priors, scalers
and classifiers are fitted separately inside each original training split.
Model timing in JSON excludes cache preprocessing and is not end-to-end latency.

- `REPORT.md`, `comparison.png`: results and interpretation.
- `comparison.csv`: 12 representation/model summaries.
- `fold_scores.csv`: 116 individual scores; `paired_loso.csv`: 72 paired differences.
- `experiment.json`: frozen definitions, code/data/environment hashes and prior-result snapshots.
- `cache_inventory.json`: all 18 cache hashes, matrix dimensions and minimum eigenvalues.
- `results/<representation>/*_predictions.npz`: raw per-window predictions.
- `verification.json`, `verified_jobs.json`: independent verification evidence.
- `logs/`: per-model logs, cache logs, tests and analysis output.

The figure SD is sample SD across eight people, not a confidence interval.
No lag, kernel bandwidth policy, ridge or classifier regularization tuning was
performed on the public test. The delay representation increases the tangent
feature count; C remains fixed as requested. Previous reproduction, reference
and literature ablation artifacts are preserved and hash-checked.
