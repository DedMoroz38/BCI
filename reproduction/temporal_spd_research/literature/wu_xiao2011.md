# Covariance Matrix Estimation in Time Series

Wei Biao Wu; Han Xiao. 2011.

Primary source: https://www.stat.uchicago.edu/~wbwu/papers/cov-reviewJune15-2011-2.pdf

Verified literature mechanism: Studies covariance and autocovariance estimation for temporally dependent observations, including Toeplitz structure and tapering. Serial dependence changes covariance estimation; arbitrary lag matrices are not automatically positive definite.

Transfer and limits: Our zero-padded shift Gram is a finite-sample PSD construction with equal diagonal blocks, then OAS regularization. This particular implementation is an adaptation, not the full estimator from the review.
