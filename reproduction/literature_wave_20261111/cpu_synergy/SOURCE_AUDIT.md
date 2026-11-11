# Source and inference audit

The reused `custom_validation_20261110/classical.py::matrix_job` fits `log_prior` with `la.mean(0)` over its `tr` rows and StandardScaler with source tangent features `a`. Its bootstrap is sampled from source people only. `mssa8_mix25_lr50` is trained with source `y[keep]` and class/bootstrap weights. Saved states contain band, 20 physical channels, 40x40 log prior, StandardScaler, source-length bootstrap and LR head.

This run passes only source features/source labels/source weights to `StructuredLR.fit`; target covariances are transformed only after the two new heads are fitted. Stored source state is never updated. Target covariance log references use a cumulative sum separately within person/session in run/onset order. Balance correction is likewise cumulative per person/session. Both include the current unlabeled window and have no future dependence. Source participant histories may be used offline during source fitting.

`mfm_models_20261110/prepare.py::mssa_mixed` constructs the temporal mSSA basis independently within each window; `mssa_band` filters and computes OAS Lag4 covariances window by window. All six cache bands have shape (2655,82,82). No cross-window or target-label fitted mSSA state exists. Window-internal filtering is offline within the already acquired window.

Native threadpool inspection after importing the experiment showed OpenMP with one thread; environment variables also constrain OpenBLAS, MKL, Accelerate/vecLib and NumExpr to one thread. The executor has exactly two worker processes.

Four behavioral tests are saved in tests.log. The first test run before implementation failed because the experiment module did not yet exist; after implementation all four passed. These tests target causal behavior and the mathematical objective rather than copied output equality.
