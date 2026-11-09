# H7: stable encoding of the same temporal covariance hierarchy

Locked after six-candidate CV began, before any NEW public-test scores existed.
Motivation is numerical evidence, not test performance. `block_numerics_probe.json`
shows 400 theoretically zero cross-block tangent features acquire variance up to
4.53e-27 and are standardized to values up to22.0 by the untouched baseline scaler.
Raw block-diagonal accuracy therefore cannot be promoted as a reliable winner.

Hypothesis: the same two-half covariance information, expressed in a fixed
per-physical-channel sum/difference basis, avoids forced-zero tangent features
without changing the baseline classifier or scaler.

Let A=OAS(first 240 samples), B=OAS(last240 samples), U=(A+B)/2,V=(A-B)/2.
In block coordinate order define R_mixed=[[U,V],[V,U]], then interleave each
physical channel's two coordinates. Equivalently R_mixed=Q diag(A,B) Q^T,
Q=(1/sqrt2)[[I,I],[I,-I]]. Q is orthogonal, independent of all labels/data and
mixes only the two copies of each physical channel. Information and eigenvalues
are preserved; the original source's diagonal StandardScaler/regularization need
not be invariant to this representation. No scaler threshold or source edit.

Minimal pre-implementation numeric probe: min original scaler feature variance
.00194, zero variances below1e-20, same source matrices positive. No accuracy used
to choose the mixing matrix or any parameter.

Evaluation: the original29 jobs for this one additional R(X), with original
physical draws, source classifier, C=.01, seed0 and cumulative log-Euclidean prior5.
Keep all174 first-stage results as raw evidence. No source/result overwrites.

Quality gate established before new test scores: raw pyramid2_oas and
quadrature_oas have mathematically forced zero tangent coordinates, so they are
not eligible for final recommendation under this source scaler; retain/report
raw scores transparently. The former has direct numeric evidence; the latter's
complex structure forces each self-quadrature tangent cross-entry to zero.
No claim that a score difference is entirely caused by roundoff is made without
an ablation. We only refuse to rank a representation with amplified numerical
null coordinates as the reliable winner.

Wait for all162 first-stage CV/later and27 follow-up CV/later jobs. Freeze and
commit final selection over eligible first-stage functions, original/delay1
controls and mixed hierarchy using forest20 LOSO mean. Then run all14 new test
jobs (12 first-stage +2 follow-up). No tuning after any new public test.

Report203 new jobs plus58 controls=261 official re-scores; same-size comparison,
per-person deltas, full numerical diagnosis and selection chronology. This is an
exploratory follow-up prompted by integrity checking, not an independent
confirmatory replication. Stop after this corrective follow-up and validation.
