# Kaggle launches — 9 November 2026

Both writes confirmed by the CLI: **Kernel version 1 successfully pushed**.

- CPU: https://www.kaggle.com/code/den1shh/proteus-winner-features-cpu
  Private, 240-minute application cap,15000-second Kaggle timeout, no GPU.
  21 classic experts,6 saved-prediction variants,3 frozen geometry variants,
  all11 outer folds, plus fixed fusion comparisons. Existing SWAD weights
  attached through kernel source den1shh/proteus-regularization-wave2.
- GPU: https://www.kaggle.com/code/den1shh/proteus-causal-alignment-gpu
  Private,T4,690-minute application cap,43200-second Kaggle timeout.
  4 latent-alignment CNNs +2 TSMNet-derived causal models,2seeds,11folds:
  132 outer model fits with inner-NLL selection and refit.

Initial CPU upload was rejected because embedded weights exceeded Kaggle's
1MB source limit. No CPU run was created by those rejected writes. Moved
weight loading to the existing notebook output and successfully pushed v1.
GPU v1 was not pushed again. No redundant version was launched.

No baseline/control refits, numerical reproduction/equality gates or remote
preflight test stage. Minimal local component gradient/training smoke passed.
No competition submission, no background status monitor. Run completion,
remote execution state after launch, and final metrics have not been checked.

Outputs: winner-cpu-results and winner-gpu-results. Per-job failures and
completed predictions survive independent failures. Trainable stacking gate
remains the later dependent wave, as documented in EXECUTION.md.

## Checked 10 November 2026

Both CPU/GPU v1 COMPLETE. CPU 1684.34s,21/21 experts on11/11 folds,
no failures or budget skips. GPU 4284.87s,132/132 fits complete,no failures.
Summary outputs downloaded to downloads/cpu-v1 and downloads/gpu-v1.
114CPU +42GPU complete configurations;1254+462 prediction records.
Selected leader and example learning curves also downloaded. No reruns.

New best: latent_meanstd_p10_mean2__late_w0.2,OOF .6322938018366412,
later 0.7163945551338814;OOF gain +.506006pp. 7/10 people improve.
See RESULTS_20261110_RU.md. No background monitor or submission started.
