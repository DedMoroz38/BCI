# SPD Window Representation Experiment Plan

**Spec:** docs/superpowers/specs/2026-10-29-spd-representations.md
**Goal:** 87 new PROTEUS evaluations for delay, HSIC and envelope representations, with original cumulative log-Euclidean reference.

- [x] Inspect original covariance/model code, verify primary sources and predeclare all representation parameters.
- [x] Write failing mathematical/grouping/causality/control tests, implement per-window representations and upstream adapters, pass checks.
- [x] Implement and verify exact resumable per-window matrix cache; freeze code/data/config; compute 18 band/representation caches.
- [x] Run all 87 jobs with original physical channel selection and original log-Euclidean recentering.
- [x] Independently verify all 116 original/new prediction files, grouping, SPD and unchanged prior experiment artifacts.
- [x] Produce same-format tables, plots, dimensionality/estimator details, provenance and rerun commands; final checks.
