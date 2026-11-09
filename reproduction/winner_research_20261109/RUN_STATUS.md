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
