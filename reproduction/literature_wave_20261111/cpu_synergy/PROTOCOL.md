# Frozen CPU synergy protocol — 2026-11-11

Written before generating or inspecting any new experiment score.

Hypothesis: rank-8 multivariate SSA, mixed at 25% into Lag4 covariance, removes nuisance temporal variation while a nuclear penalty controls the spatial-temporal classifier. Their combination may improve the existing mSSA LR forest and ordinary Lag4 nuclear forest.

Use exactly the ten fixed folds and ordered train/test row IDs from `reproduction/custom_validation_20261110/splits.json`: eight source people, two held-out people, each person held out twice. Metadata and cached mSSA covariances cover all 2655 windows. No new split, subject/window subsampling, target tuning or target labels in training.

Two new models only: `mssa8_mix25_nuclear001_50` and `mssa8_mix25_nuclear005_50`, nuclear tau 0.01 and 0.05. Each has the same 50 experts, selected bands, 20 physical channels (40 lag channels), source-person bootstrap and class weights as the saved matched mSSA LR50 experts. Reuse those experts' source-only log prior, source StandardScaler and LR initialization. Reconstruct source features from cached covariances with source-only prior and chronological causal normalization (prior mass 5). Fit existing `models.StructuredLR(dim=40, nuclear=tau, iterations=60)`, C=.01. L2 strength and bootstrap weights exactly follow existing `classical.py`. No LR baseline refit.

Inference uses each session's prefix including the current unlabeled window; no future window or cross-person state. Ensemble probabilities are the arithmetic mean of all 50 experts. Report plain and existing causal `__balance` marginal adjustment (prior mass 10); both forms are fixed in advance. Target labels are used only after all fitting for scoring.

Primary metric: first-session cell balanced accuracy, averaging subject/session/context cells within each held-out person, averaging the two repeat metrics within person, then equal mean across ten people. Repeated predictions are never ensembled across splits. Also report worst three people, first-session NLL, natural-stream bins 1–20, 21–60, 61+, all-session cell BA and person/repeat detail.

Comparators use existing fixed-split per-window predictions from `custom_validation_20261110/results_pair_cpu`: mSSA LR50, ordinary Lag4 matched LR50, nuclear .01 and .05, each plain and balanced. No submissions, parity/hash gates, remote jobs or monitoring service. New results are exploratory on previously reused validation participants, not independent test evidence.

Before the full run: meaningful synthetic tests of prefix invariance under future perturbations, independence from target labels, isolation of source fitted state from target covariances, nuclear objective descent and stronger regularization. At most two CPU worker processes and one BLAS/OpenMP thread each. Save all new per-window probabilities and expert heads/geometry/bootstrap references, exact source/test IDs, progress and failures. Exclude incomplete 50-expert folds and incomplete ten-fold configurations from aggregate comparison. Log exceptions; never replace failure with uniform probabilities. Report elapsed runtime and failure counts.
