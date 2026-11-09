# Literature-driven replacement of R(X), 2026-10-29

Question: which label-free per-window SPD function preserves useful temporal/continuous-path information better than original OAS and the previous one-sample augmented covariance?

Scope: only R(X). Original source, log-Euclidean session reference (prior weight 5), scaler, LogisticRegression C=.01, seed0, 50 models, physical 12/20 electrode draws, frequency bands, splits and window order stay fixed. No baseline changes. Priors/scalers/classifiers are refitted on training representations inside each fold. Pure per-window transforms can be cached on all splits without using labels or other windows.

Controls: original OAS (29 existing jobs) and delay1 OAS (29 existing jobs). Their artifacts remain immutable and hash checked.

## Six predeclared candidates

1. `lag4_oas` (82): OAS of [x_i(t),x_i(t-4)], 476 aligned samples, electrode-major copies. Tests a 33.333-ms delay with the same dimension as previous best. ACM/Takens literature.
2. `stationary_oas` (82): center each complete channel; zero-padded shift vectors [x(t),x(t-1)] across T+1 positions, followed by OAS. Equal diagonal blocks, biased lag autocovariance; exactly SPD block-Toeplitz. Inspired by stationary autocovariance estimation, not a reproduction of TICC graphical lasso.
3. `pyramid2_oas` (82): independent OAS for first and second 240-sample halves; assemble block diagonal in electrode-major order. Tests coarse within-window nonstationarity. Level-1 temporal covariance hierarchy adaptation; no classifier change.
4. `quadrature_oas` (82): real/imaginary Hilbert analytic signal covariance, regularized by original OAS on stacked quadratures; project to complex covariance structure by averaging S and J S J^T (J is the quarter-turn matrix), then interleave physical copies. Convex average guarantees SPD. Spectral/Hermitian covariance adaptation.
5. `path_integral_oas` (41): piecewise-linear interpolation of samples; evaluate two exact Gauss-Legendre nodes per interval, then OAS. Unshrunk empirical covariance equals continuous-path covariance integral exactly. Node-based OAS changes its shrinkage with the effective sample count; this is explicitly part of R(X). Proposed continuous-path adaptation, not attributed as an existing named classifier.
6. `signature2_oas` (41): C_OAS(X) + s^2 Phi Phi^T/64. s is the single window-wide RMS after filtering (floor 1e-12); Phi concatenates four nonconstant depth-2 signature coordinates (S_x,S_tx,S_xt,S_xx) of time-augmented, piecewise-linear channel paths on 16 disjoint 30-sample patches, local time 0..1, amplitude divided by s. Exact Chen integration, no subsampling. Original covariance preserves power/sign; Gram adds nonlinear ordered-path features. This channel-Gram fusion is a proposed adaptation of signature kernels, not the full published signature-kernel classifier. For variable test fixture length use 16 equal patches (minimum two samples).

All R are strictly positive definite on nondegenerate filtered windows due to OAS; signature addition is PSD; no post-hoc eigenvalue clipping. Zero-signal fixtures are diagnosed, not interpreted as valid EEG. All doubled variants select both coordinates of each physical electrode together.

## Evaluation locked before data-dependent results

- Six candidates × original 29 jobs =174 new evaluations: 8 LOSO ×3 models, 3 later-session models, 2 public-test models per candidate.
- Run all CV/later jobs before public test. Primary selection: maximum mean session-balanced accuracy of forest20 over 8 LOSO people; deterministic alphabetical tie break. Freeze selection.json, including SHA-256 of all CV outputs, BEFORE test execution. Selection includes original and previous delay1 controls, so no new method is required to win.
- Then test all fixed candidates for complete same-format tables. Test cannot change the declared selection or trigger tuning.
- Compare same-size families separately: 41 vs original41; 82 vs previous delay1 82. Feature count and unchanged C remain limitations.
- All 232 new/control files independently rescored by official four metrics, exact subject/window pairing, tree draws, grouped copies, dimensions and caches checked.
- Candidate results are confirmatory relative to this fixed protocol; choosing a winner among six on these eight dev people remains exploratory model selection. Public test already used in past work, so no pristine holdout claim.
- Stop after these six candidates, full validation and literature synthesis. No unbounded sweep. If none beats delay1, retain it and report negative results.

## Hypotheses

H1: lag4 reduces redundancy between adjacent samples at equal dimension; predicts forest20 LOSO > delay1.
H2: stationary estimation reduces boundary variance; predicts more consistent per-person gains at equal dimension.
H3: preserving first/second-half covariance order helps nonstationary windows; predicts gains over original, possibly at a cost to small-sample estimation.
H4: quadrature covariance preserves phase-lag interactions beyond instantaneous covariance; predicts improvement over original and amplitude-only envelope.
H5: continuous-path integration removes discretization weighting artifacts at fixed dimension; predicts similar or better performance with lower path roughness sensitivity.
H6: ordered signature features add nonlinear temporal information while retaining original power; predicts improvement over original and prior power-normalized HSIC.

## Execution integrity

Protocol committed in the isolated research directory before numerical experiment results. No edits to prior numerical source or results. Current-turn monitoring plus persistent research-state/log/findings support continuation; Claude Code /loop and OpenClaw cron are unavailable in this runtime and are not fabricated.
