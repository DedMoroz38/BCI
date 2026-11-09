# Findings: temporal SPD function research

Only R(X) changed. Original source, cumulative log-Euclidean reference, C=.01,
seed0,50 models, physical channel subsets, bands, folds and data remained fixed.
203 new evaluations plus58 saved controls were verified by official metrics;
101 tests passed. Nine primary papers were checked and summarized in literature/.

## Measured outcome (forest20, session-balanced accuracy)

| R | LOSO | Public test |
|---|---:|---:|
| Original OAS | .531430 | .528622 |
| Previous delay1 | .553871 | .544350 |
| Fixed lag4 | .562801 | .555407 |
| Stationary lag covariance | .555193 | .544350 |
| Continuous integral | .531734 | .523288 |
| OAS + depth2 signature Gram | .499496 | .461485 |
| Stable mixed temporal halves | .610376 | .507398 |

The numerical-eligibility winner selected before test was mixed temporal halves.
It gained5.65 percentage points in LOSO over delay1 but lost3.70 points on test.
Its ranking survived dropping each one person from the eight-score average, but
this is descriptive stability, not nested validation. Good development-person
ranking did not establish external superiority.

Lag4 is the only new eligible configuration strictly improving both forest20
LOSO and public test over delay1: +.89 and+1.11 points. It won5/8 LOSO people.
Its practical preference is retrospective across fixed candidates, not a new
blinded selection. Lag4 was never tuned after test, but fresh data are required
to validate this recommendation; the same public test was used in earlier work.

## Numerical integrity changed the conclusion

Raw block-diagonal temporal halves reached .620296 LOSO but .481984 test.
Audit found400 theoretically zero tangent coordinates with variances up to
4.53e-27, standardized by the untouched source scaler to values up to22.0.
Quadrature realification similarly had20 forced self-quadrature null coordinates
with variances up to7.93e-28 and scaled values up to9.81. These raw scores remain
reported but are ineligible for final recommendation.

A fixed per-physical-electrode Hadamard congruence preserved both half covariances
and all eigenvalues, while removing the forced null coordinates. All29 follow-up
models had zero near-null coordinates; minimum measured variance .0015225.
No scaler threshold or baseline source was changed. This follow-up was committed
before its accuracy evaluations and all new test scores. Orthogonal congruence
also interacts with coordinate scaling/regularization, so score differences
cannot be attributed entirely to roundoff without a separate ablation.

## Continuity and ordered paths

Piecewise-linear interpolation is an explicit assumption between samples, not a
consequence of having a continuous underlying physical process. Exact integral
covariance using two Gauss nodes per interval did not improve average held-out
performance meaningfully here. This does not rule out other interpolation,
roughness, spectral, derivative or delay representations.

The depth2 signature channel Gram loses more information than the full signature
kernel. Reversal maps coordinates (Sx,Stx,Sxt,Sxx) to(-Sx,-Sxt,-Stx,Sxx), an
orthogonal transformation, so the Gram is reversal invariant. Concatenated patch
positions are also pooled by the Gram. Label-free probes confirmed reversal
invariance and shuffle sensitivity. Full signature universality cannot be claimed
for this adaptation. No frozen numerical implementation was changed after this
mechanism finding.

## Lessons and limits

- SPD validity is necessary for the geometry but does not guarantee useful information or numerical feature stability.
- Same-size comparisons matter: lag4 vs lag1 both use82x82; integral/signature vs original use41x41.
- Temporal hierarchy from human action recognition gives a strong development signal but weak public-test transfer in this setting.
- Preserved power plus nonlinear features was not enough to rescue our particular shallow signature fusion.
- Future experiments need fresh evaluation people/data before further public-test-guided optimization. Do not keep sweeping this test.
- The source scaler is sensitive to tiny nonzero variance in theoretical null tangent coordinates. Any structured SPD representation needs this check.

See FINAL_REPORT.md for exact matrix formulas, all models/protocols, literature
attribution and qualification of pre-test selection versus retrospective reading.
