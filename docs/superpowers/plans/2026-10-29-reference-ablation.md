# Reference Matrix Ablation Implementation Plan

**Goal:** Execute 87 controlled PROTEUS evaluations and compare them with the 29 verified original control jobs.
**Spec:** docs/superpowers/specs/2026-10-29-reference-ablation-design.md
**Architecture:** External reference updater and subclass, scoped injection into original model factories; independent resumable jobs; official scoring and report outside source checkouts.
**Tech stack:** Existing reproduction/.venv and unchanged pinned commits/data from the prior run.

## Constraints

- Fixed candidate updates: cumulative Euclidean, inductive AIRM with gain 1/(5+t), normalized log-Euclidean decay .95.
- Same original training log-Euclidean prior for every method, w0=5, current-window inclusion and session resets.
- Same C=.01, seed=0, 50 trees, m=12/20, original bands and bootstrap RNG.
- Same 29 jobs per candidate; retain original baseline results unchanged; no test-set parameter tuning.
- Explain that recursive AIRM is not exact batch mean and EMA warm-up has normalized prior weight.

## Tasks

- [x] Inspect source and record design/variant formulas before computing scores.
- [x] Write and execute failing reference-update tests, implement the external adapter, pass unit and control-parity checks.
- [x] Implement resumable runner with source/data/environment guards and predeclared variant manifest.
- [x] Run all 87 jobs, preserving predictions and tree selections.
- [x] Independently re-score with official metrics; compare paired LOSO differences, check original artifacts unchanged.
- [x] Produce report, per-fold CSVs, plots, exact rerun commands and final verification.
