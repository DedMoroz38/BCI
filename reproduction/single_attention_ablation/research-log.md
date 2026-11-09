# Single-model experiment ledger

2026-11-07: User explicitly replaces the forest experiment with baseline versus
attention WITHOUT forest,after asking about end-to-end training. User handles
old notebook cancellation; API now verifies old version2 CANCEL_ACKNOWLEDGED.
Protocol e314fdb committed before any new accuracy measurements.

Scope: single Lag4 OAS baseline and BOTH previously requested attention SPD
forms,one model each. All41 channels,band4–40,lag4;all outer-training subjects,
no bootstrap. User did not request choosing a different band/channel subset.

Implemented full-batch differentiable training through train log prior,
cumulative per-session recentering,tangent map,training mean/std,and the SAME
retained final LR. No detached reference or auxiliary identity-tangent head.
Final LR initialized by sklearn once on current encoder's actual causal features,
then updated jointly.80full-batch steps fixed,not a convergence claim.

Tests: first failed on offdiagonal sqrt2 incorrectly rounded through float32;
fixed the scaler constant in float64. Tests now verify pyRiemann equivalence,
past-reference gradients,frozen-eval causal prefixes,two-session reset,
actual training,retained-head updates,save/reload,and independent NumPy streamed
head inference.8local tests pass;one full1800window82channel CUDA gradient smoke
runs on Kaggle before fitting,covering the largest actual training fold1713.

Independent read-only review found no critical/important correctness defect.
Reviewer limitations addressed: runtime now rejects library-version changes on
resume (regression test failed before implementation then passed),and CUDA smoke
increased from128 to1800 windows to cover full-fold memory. No new model metrics
yet. No prior forest metrics used to tune this architecture.
