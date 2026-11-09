# Quick Local Ablation Implementation Plan

Goal: finish a small local screening of mixed views,causal dispersion and balance.
Spec: ../../../PROTOCOL.md
Architecture: immutable historical probabilities as control; shared window-local
matrix cache; independent fold fits; fixed configurations; independent rescore.
Tech stack: existing NumPy/SciPy/sklearn/pyriemann Python3.12 environment.

## Constraints/review focus
Preserve exact tree draws,bootstrap weights and OAS-before-subsetting. Prevent
stream cross-contamination,target-label fitting,cache tampering and stale resume.
No new neural training. No public-test selection. Fixed50-tree family budget.

## Task1 — implementation and verification
- [ ] Tests first: causal prefix/reset for RMS and balance; exact adaptive-lag
  matrix vs DedMoroz code; fold/metadata identity; changed receipts rejected.
- [ ] Implement transforms.py,run.py with hashed inputs and resumable results.
- [ ] Run tests,freeze code/protocol and record commit before numerical scores.

## Task2 — bounded local screening
- [ ] Precompute window-local cache and time one full LOSO fold.
- [ ] Complete8LOSO+late in3workers,refit exact original predictions as a guard.
- [ ] Independently rescore45results,write comparison and paired report.
- [ ] Review implementation and verify results before conclusions.
