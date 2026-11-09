# Classifier-only ablation of the submitted Lag4 forest

All70 evaluations completed. Geometry,scalers,tree draws and subject bootstraps are shared. The original LR reproduced all10 prior baseline prediction files exactly.

LOSO-selected model,chosen before new public-test scores: **svm_linear**.

| Classifier | LOSO session BA mean ± SD,% | Delta,pp | Later session,% | Public test cell BA,% | Public test session BA,% |
|---|---:|---:|---:|---:|---:|
| Logistic regression (submitted baseline) | 56.28 ± 13.32 | +0.00 | 72.37 | 55.72 | 55.54 |
| Shrinkage LDA | 56.51 ± 13.08 | +0.23 | 70.90 | 55.28 | 55.17 |
| Linear SVM + grouped calibration | 57.52 ± 12.72 | +1.24 | 75.83 | 50.44 | 50.35 |
| RBF SVM + grouped calibration | 56.64 ± 14.54 | +0.36 | 66.84 | 51.94 | 51.84 |
| Ridge + grouped calibration | 51.37 ± 11.10 | -4.91 | 62.84 | 45.89 | 45.78 |
| Elastic net + grouped calibration | 52.58 ± 11.46 | -3.70 | 65.31 | 45.85 | 45.78 |
| MLP32 (PyTorch CUDA) | 54.52 ± 13.63 | -1.76 | 73.38 | 53.60 | 53.44 |

## Selection stability

The selected svm_linear remains the winner in 8/8 leave-one-person-out reaggregations and in 69.7% of20,000 paired person bootstraps. These reaggregate existing LOSO scores without retraining. They describe sensitivity to the eight observed people; they are not probabilities of superiority on a new cohort. Full counts are in selection_stability.json.

## Interpretation limits

Fixed literature-informed configurations were compared,not optimally tuned model families. There are only8 LOSO people; their training folds overlap. Paired bootstrap intervals and sign-flip/Holm values in the CSV are exploratory descriptions,not independent-cohort confirmation. The later-session estimate is one held-out session. Public test comprises two people and has already been inspected in prior research. It is not a sealed score.

The probability interface for SVM,ridge and elastic net includes subject-grouped sigmoid calibration. This is part of the classifier replacement. Geometry is frozen at the outer-training fold during inner calibration; no outer-evaluation labels enter fit.

MLP uses300 fixed full-batch Adam epochs on CUDA,not the provisional local sklearn L-BFGS fixture. Its fixed-epoch results do not assert convergence. All numerical ConvergenceWarnings from classical classifiers remain in individual job JSON files.

Evidence tiers and adaptations are documented in ../literature/survey.md. Ridge has direct fMRI tangent-connectome support; RBF has a weaker direct EEG source. The strongest direct EEG precedents are TS-LDA,TS-linear-SVM and TS-elastic net.
