# Author Ensemble Comparison Implementation Plan

Goal: reproduce seven author variants and refit the submitted Lag4 on identical folds.
Architecture: unchanged author checkout; isolated runner and baseline subprocess
avoid Python module collisions. Persist component specs and predictions per fold.
Tech stack: existing Python3.12,NumPy,SciPy,scikit-learn,PyTorch CPU.
Spec: ../../../PROTOCOL.md

## Constraints and review focus
No source/data mutation,no test tuning,no change to defaults; pin all inputs.
Guard participant splits,recording order,cache tampering,resume integrity,
stream resets and model serialization. Previously seen test is descriptive.

## Task1 — runner and integrity checks
- [ ] Add tests for exact folds,metadata matching,changed manifest rejection,
  streamed vs reset/prefix behavior,and saved-model replay; observe failure.
- [ ] Implement run.py shared folds/manifest/probability evaluation and
  author component training; baseline.py independent old-code fit.
- [ ] Run tests and a real reduced-size smoke check without reporting its scores.

## Task2 — numerical experiment
- [ ] Freeze protocol and code hashes,commit before measured full runs.
- [ ] Run8 LOSO+late with two workers; persist each variant and Lag4 controls.
- [ ] Freeze CV hashes then run original public-test split.

## Task3 — analysis
- [ ] Independently rescore saved predictions,require80 total scores and exact
  baseline equality,produce tables,paired uncertainty,and Russian report.
- [ ] Verify artifacts,record timings,review final implementation and conclusions.
