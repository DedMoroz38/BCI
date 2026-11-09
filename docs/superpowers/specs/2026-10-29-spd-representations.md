# Replace EEG window covariance, retain original log-Euclidean alignment

## Request and scope

Compare the original pipeline with three window representations from the user's description: one-delay augmented covariance, an HSIC Gram SPD matrix, and covariance of Hilbert amplitude envelopes. Retain the original cumulative log-Euclidean session reference and prior mass 5. Refit the training prior, scaler and classifier on each new representation; numerical priors necessarily change with R (and with its dimension).

Same PROTEUS data, subject splits and 29-job table-2 protocol per representation: 24 LOSO, 3 later-session, 2 public test (logreg/forest20 only). Three candidates = 87 new evaluations plus the original 29 controls. C=.01, seed=0, 50 trees, physical channel subsets m=12/20, original six bands. No test-based parameter tuning.

## Predeclared representations

1. `delay1_oas`: apply the unchanged band filter to the full 4-s window. For each electrode, concatenate its current samples x[t] and one delayed copy x[t-1], using aligned t=1..479. Electrode-major order: (electrode0 current, electrode0 delayed, electrode1 current, ...). Estimate OAS covariance on the full 2h-channel embedding, preserving the original covariance estimator. One extra copy, lag=1 sample at 120 Hz (8.333 ms). Matrices: 82 full channels, 24 or 40 coordinates per forest tree, tangent dimensions 3403 / 300 / 820 for logreg/forest12/forest20.
2. `hsic_rbf`: unchanged band filter; for each channel and window set sigma_i to the median absolute pairwise difference of its 480 samples (strict upper triangle). Constant channel: sigma=1, centered kernel zero. K_i[t,s]=exp(-(x_i[t]-x_i[s])^2/(2*sigma_i^2)); G_i=H K_i H. S_ij=tr(G_i G_j)/480^2. R=S+eps I, eps=max(1e-12,1e-6*trace(S)/h). All 480 time samples used, no subsampling, RFF or Nyström. Biased T^-2 HSIC as given by user. Per-window median bandwidth is scale-adaptive and does not directly retain raw channel-power magnitude.
3. `envelope_oas`: unchanged band filter, a=abs(scipy.signal.hilbert(x,axis=-1)), then the same OAS covariance estimator on a. Full window, no cross-window filtering/state. This is envelope covariance, not normalized envelope correlation or the full FUCONE ensemble.

Important estimator detail: the original implementation uses OAS, not raw XX^T. OAS is deliberately retained for both covariance-based candidates to isolate the signal representation. HSIC has its separate diagonal SPD regularizer. Therefore these are controlled implementations of the proposed representations, not literal unregularized-Cov+eps replicas of three external papers.

## Physical channel grouping

Continue drawing physical electrodes from 41 channels with the exact upstream RNG/bootstrap code. For delays, select both copies for every chosen electrode. A scoped `_sub` adapter is used during upstream fitting; after fitting physical indices are retained as metadata and the tree's matrix indices are expanded to [2c,2c+1] for normal upstream prediction. No random sampling from 82 independent augmented channels. Bands, physical channel selections and bootstrap code stay unchanged.

## Exact caching

Representations depend only on one raw input window and its band; they use no labels or cross-window statistics. Precompute full matrices once for all 2655 windows and six bands, using deterministic per-window transforms. Use read-only .npy memory maps plus an exact float32-window hash index. Cache rows and metadata are checked. This unlabeled per-window cache does not fit priors/scalers/classifiers on test data; all learned statistics remain fold-specific. Cached and live matrix/prediction paths must agree. Cache computation is resumable by completed chunks and config/code hashes; no approximation to HSIC.

## Implementation

External RepresentationFeatures overrides only covariances(), retaining inherited fit_transform/new_session/step and original SessionRecentering. Scoped injection into source model factories; original source stays unchanged. Add delay-group mapping without copying the training/randomization logic. Up to four independent local processes. Preserve all prior experiment artifacts unchanged.

## Verification and reporting

Test exact delay indexing and grouped submatrices, pure-carrier Hilbert envelopes, HSIC against a naive centered-kernel trace implementation, nonlinear x vs x^2 example, HSIC joint-permutation invariance, SPD including constant inputs, full-vs-cached equality, original-control coefficient equality, causal per-window behavior, session reset and batch/stream prediction parity.

Before model runs freeze configuration, source/data/package hashes and cache checksums. Independently re-score all 87 new plus 29 original prediction files with official metrics. Report LOSO mean/sample SD, paired subject differences, later-session and test scores, matrix/tangent dimensions, computational choices and literature attribution. Test is exploratory public data, not a new hidden evaluation.

Sources:
- Carrara & Papadopoulo (2023), Classification of BCI-EEG based on Augmented Covariance Matrix: https://arxiv.org/abs/2302.04508
- Gretton et al. (2005), Measuring Statistical Dependence with Hilbert-Schmidt Norms: https://www.cs.cmu.edu/~arthurg/papers/GreBouSmoSch05.pdf
- Corsi et al. (2022), FUCONE, section 2.2.3 amplitude envelope correlation: https://arxiv.org/html/2111.03122v2

HSIC channel-to-channel Gram SPD construction is the user-specified assembly based on HSIC; do not attribute it as an evaluated winning method in those papers. Envelope covariance and fixed lag=1 are our declared adaptations.
