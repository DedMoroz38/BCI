# Five additional literature-based SPD references

## Scope

Repeat the same causal PROTEUS table-2 protocol: 29 evaluations per new reference, 145 new jobs. Models: original one classifier, forest12 and forest20; eight LOSO folds and one later-session fold per model; public test only one classifier/forest20 fitted on six train people. All windows, source commits, C=.01, seed=0, bands, OAS, scalers, 50 trees, channel selections and bootstrap code are the same. Update is applied in training and inference, includes the current window and resets per session. Shared training prior M0=exp(mean(log R_train)), initial mass 5.

## Predeclared new candidates (before new scores)

Let n=5+t, A=(5M0+sum R_i)/n, H=((5M0^-1+sum R_i^-1)/n)^-1.

1. `harmonic_cumulative`: M=H. Harmonic/inverse-Euclidean mean; Moakher & Batchelor (2006), discussed with explicit A/H definitions in Nielsen (2024), Appendix A.
2. `poweuclid_half`: M=((5 M0^(1/2)+sum R_i^(1/2))/n)^2. Power-Euclidean metric, p=.5, Dryden/Pennec/Peyrat (2010), equations 2–4. This is not the implicit Riemannian power mean.
3. `kullback_sym`: M=A # H (AIRM geodesic midpoint). Exact symmetrized-KL/Jeffreys centroid of centered Gaussian covariance matrices, Moakher/Batchelor (2006); Nielsen (2024), Theorem A1, equation A5.
4. `logchol_cumulative`: Cholesky factors L_i=chol(R_i). Average strictly lower entries arithmetically, average log(diagonal) arithmetically; reconstruct M=Lbar Lbar^T. Lin (2019), section 4.2, equation 4.3. Channel order stays identical to control; this geometry depends on that order.
5. `wasserstein_recursive`: M_t=BW_geodesic(M_(t-1),R_t,1/(5+t)); geodesic from Bhatia/Jain/Lim (2019), equations 38–41, implemented by pyRiemann's geodesic_wasserstein. This is an inductive streaming estimator, not the exact batch Wasserstein barycenter.

For 1–4, these are exact prefix weighted means/centroid with the shared prior represented as mass 5. For 5 the same prior initializes the inductive procedure and its mass determines the step schedule. No parameter tuning on test.

## Literature attribution boundary

These are published SPD estimators, applied here as new reference updaters in the existing EEG model. We do not claim reproduction of five full published EEG systems or their reported performance. Streaming prefix/prior handling is our controlled adaptation. Synthetic identity-reference experiments remain unchanged.

Primary sources:
- https://arxiv.org/abs/1009.3045 (Dryden, Pennec, Peyrat, 2010)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11675394/ (Nielsen, 2024), Appendix A / A5
- https://arxiv.org/abs/1908.09326 (Lin, 2019), section 4.2
- https://arxiv.org/abs/1712.01504 (Bhatia, Jain, Lim, 2019), section 4
- https://link.springer.com/chapter/10.1007/3-540-31272-2_17 (Moakher & Batchelor, 2006)

## Implementation/validation

External subclass and scoped factory injection as in the previous comparison; original model files unchanged. Independent resumable processes, at most four local processes. Unit tests compare scalar hand-derived results and noncommuting weighted means with independent library definitions, SPD, causal prefix behavior, reset and exact original-control parity. Frozen code/data/environment hashes before execution; preserve old artifacts unchanged; verify 145 new plus 116 existing prediction files with official metrics; report paired LOSO changes, sample SD and all scores without test-based selection.
