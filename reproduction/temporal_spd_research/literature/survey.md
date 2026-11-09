# Literature synthesis

Question: replace only R(X), preserving the original pipeline. Literature does not establish a universally optimal SPD representation. We test a fixed, auditable subset of mechanisms on the same PROTEUS splits.

| Mechanism | Primary references | Experiment | Transfer status |
|---|---|---|---|
| Delay/state embedding | Carrara 2023 | lag4_oas | Published construction, fixed parameter; same dimension as lag1 |
| Stationary auto/cross-covariance | Wu & Xiao 2011; Hallac 2017 | stationary_oas | New finite-window Gram/OAS implementation; no TICC training |
| Local covariance hierarchy | Hussein 2013; Sanin 2013 | pyramid2_oas | One temporal level packed into a single SPD matrix |
| Complex second-order/phase information | von Sachs 2020; Chau & von Sachs 2021 | quadrature_oas | Analytic-signal adaptation with real SPD packing |
| Continuous interpolation | Király & Oberhauser 2019; Dyer 2022 | path_integral_oas | Proposed exact integral covariance with node OAS |
| Ordered nonlinear path moments | Király & Oberhauser 2019; Dyer 2022 | signature2_oas | Proposed local signature Gram added to original covariance |

## Why continuity is a modeling choice

A sampled signal does not reveal its continuous trajectory uniquely. We explicitly choose a piecewise-linear interpolant. Its covariance integral has exact two-point quadrature per interval. Signature coordinates are iterated integrals over that same interpolant and encode order. Neither construction assumes the actual physiological signal is exactly piecewise linear; both depend on the chosen resolution. A band-limited or spline interpolation would be a different R(X), requiring its own experiment.

## SPD guarantees

Ordinary lag cross-covariance is generally nonsymmetric. Arbitrarily symmetrizing it does not guarantee SPD. Each candidate instead uses an OAS covariance, a block diagonal of SPD matrices, a convex average under orthogonal congruence, or SPD plus a Gram matrix. Stationary covariance is constructed from explicit zero-padded shifts rather than enforcing Toeplitz entries after the fact.

## Exclusions

- Full TICC segmentation, learned spectral networks and new kernel classifiers would alter the baseline. Excluded.
- An unconstrained DTW similarity is not automatically a positive-definite kernel. No unproved SPD conversion is used.
- Simply summing the real part of an unweighted full-band spectrum can collapse to ordinary covariance by Parseval. The quadrature candidate explicitly retains imaginary cross-channel interactions.
- Deeper Hankel embeddings increase dimension beyond the previous best; this first comparison prioritizes equal-dimension alternatives.
- Full signature kernels and continuous covariance are distinct: covariance integrates values, signatures integrate ordered increments. Their theoretical guarantees must not be conflated.

## Primary sources

- [Classification of BCI-EEG Based on Augmented Covariance Matrix](https://arxiv.org/abs/2302.04508) — Igor Carrara; Théodore Papadopoulo, 2023. Detailed attribution: `carrara2023.md`.
- [Covariance Matrix Estimation in Time Series](https://www.stat.uchicago.edu/~wbwu/papers/cov-reviewJune15-2011-2.pdf) — Wei Biao Wu; Han Xiao, 2011. Detailed attribution: `wu_xiao2011.md`.
- [Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data](https://arxiv.org/abs/1706.03161) — David Hallac; Sagar Vare; Stephen Boyd; Jure Leskovec, 2017. Detailed attribution: `hallac2017.md`.
- [Human Action Recognition Using a Temporal Hierarchy of Covariance Descriptors on 3D Joint Locations](https://www.ijcai.org/Proceedings/13/Papers/363.pdf) — Mohamed E. Hussein; Marwan Torki; Mohammad A. Gowayyed; Motaz El-Saban, 2013. Detailed attribution: `hussein2013.md`.
- [Spatio-Temporal Covariance Descriptors for Action and Gesture Recognition](https://arxiv.org/abs/1303.6021) — Andres Sanin; Conrad Sanderson; Mehrtash T. Harandi; Brian C. Lovell, 2013. Detailed attribution: `sanin2013.md`.
- [Nonparametric Spectral Analysis of Multivariate Time Series](https://doi.org/10.1146/annurev-statistics-031219-041138) — Rainer von Sachs, 2020. Detailed attribution: `vonsachs2020.md`.
- [Intrinsic Wavelet Regression for Curves of Hermitian Positive Definite Matrices](https://arxiv.org/abs/1701.03314) — Joris Chau; Rainer von Sachs, 2021. Detailed attribution: `chau2021.md`.
- [Kernels for Sequentially Ordered Data](https://jmlr.org/papers/v20/16-314.html) — Franz J. Király; Harald Oberhauser, 2019. Detailed attribution: `kiraly2019.md`.
- [Amortised Likelihood-free Inference for Expensive Time-series Simulators with Signatured Ratio Estimation](https://proceedings.mlr.press/v151/dyer22a.html) — Joel Dyer; Patrick W. Cannon; Sebastian M. Schmon, 2022. Detailed attribution: `dyer2022.md`.
