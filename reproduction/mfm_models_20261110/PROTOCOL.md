# Standalone MFM model experiments — 10 November 2026

Authorized scope: improve the model itself, prioritize local Mac computation;
Kaggle only if GPU required. This study is CPU-only: no CNN training/features,
no changes to old files, no submission, no background Kaggle monitor.

Data: existing public train/val/test concatenation, exactly the same all10 LOSO
and later_last indices as the recent studies. This is repeatedly reused public
CV, not a fresh hidden test. Every outer test person is excluded from fitted
source statistics, projection factors, class heads and rank selection. All
specified configurations are reported, no automatic outer-score selection.
Later sessions are reported separately. Window-local cached transforms may be
computed for everyone; each source prior/scaler is fitted on outer train only.
Online state uses current/past windows, resets separately in each session.

Primary comparison: standalone raw and causal-balanced cell BA versus existing
Lag4 forest50 predictions. A matched LR50 is fitted alongside changed heads
as the necessary classifier comparison, not a historical parity control.
No historical numerical tolerance/hash/control gates. The existing full CNN
ensemble64.58% is a separate historical reference, never mixed into results.

## Spectral models — 11 deterministic configurations ×11 folds

4–40Hz, source-prior causal EA weight5, Hann STFT120/hop60/no padding,
4–40Hz frequency bins:41×37×7. Causal log-power mean recentering perchannel,
frequency with source prior10; source feature standardization. Training weights
equalize people/classes within person. Parameters fixed before viewing output:

- Flattened LR:C.01/.1 (matched representation comparisons).
- Weighted PLS:4/8/16 components, latent standardized LR C1.
- Weighted HOPLS2:4 components/loadings4,4,2;8/loadings4,4,2;
  8/loadings8,6,3. Algorithm2/HOOI and Eq27 predictor projection from
  Zhao et al.,https://arxiv.org/abs/1207.1230; one-hot response, LR C1
  on latent features is our classification adapter. Max15 HOOI sweeps.
- Source-only HOSVD(Tucker):ranks4,4,2;8,6,3;12,8,4; LR C1.

## Lag4 classifiers — 8 policies ×50 experts ×11 folds

Keep original 6bands,20physical channels→40lag rows,50 experts,seed0,
participant bootstrap,prior5 causal log-Euclidean recentering,source scaler.
Perexpert geometry is computed once and shared by all heads.
Class factors are based on counts of positive-bootstrap source windows,
then multiplied by participant multiplicities. This explicit fixed convention
is shared by the matched LR and every changed head; historical probabilities
from another library/runtime are a separate saved reference, not assumed equal.

- Matched multinomial LR C.01.
- Nuclear-regularized multinomial LR:tau.01/.05; same L2 objective as C.01.
- Hard rank-capped symmetric-matrix LR:ranks8/16.
- Weighted PLS8 on the same scaled tangent vector→LR C1.
- HOPLS2 on the symmetric tangent matrix:4 components/loadings4,4;
  8/loadings8,8; LR C1.

Nuclear and rank-capped heads start from matched LR, max60 optimization steps;
nuclear penalty acts on correctly sqrt2-unvectorized symmetric coefficients.
These are new logistic adaptations; not claimed to reproduce the hinge-loss
Support Matrix Machine. Rank cap uses nonconvex projected optimization. No
neural encoder or CNN head. The forest has one fixed bootstrap seed, not
purported to establish seed robustness. Heads/geometry and losses are saved.

## mSSA before Lag4 — 2 fixed representations ×50 experts ×11 folds

Perwindow/each band horizontal multivariate SSA:shared temporal32-lag basis
across all41 channels, reconstructed by diagonal averaging. Rank8 with25%
denoised signal;rank16 with50%. No across-window decomposition. Compute OAS
on the resulting original Lag4 coordinates. Retrain the same LR50 forest;
source priors/scalers learned separately for the changed representation.
Spectrum/eigenvectors estimated from the current window need no target label.

Total21 learned model configurations ×11folds=231 outer model configurations;
the10 forest configurations contain50head fits each, plus121 spectral fits.
Each is reported raw and causal-balanced. No CNN blending. Any failed policy
is explicit; missing expert probabilities never silently form a smaller forest.

Output:comparison.csv requires all10LOSO,by_person.csv,raw probabilities,
training/inference timings,permodel source transforms and model weights.
Mac:2 processes,1BLAS thread each; source arrays mmap and small chunks.
No GPU needed, no Kaggle run charged. Resume completed folds after interruption.

Extra input policies were then declared in SPECTRAL_INPUT_FOLLOWUP.md after
the initial spectral pass. Overall54 learned configurations ×11folds=594
outer model configurations (5984 individual heads including forest members).
Three concurrent local CPU processes total across the two launches.
