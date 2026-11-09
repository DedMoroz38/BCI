# Classifier-only ablation of the submitted Lag4 forest

Date: 2026-10-29. Authorized scope: replace only the classifier after the exact
existing standardized tangent features. Compare MLP and five literature-grounded
alternatives to the submitted Riemann-Forest-Lag4, not to the original paper forest.

## Frozen geometry and evaluation

Baseline: submission_lag4/package/model_info.json; 50 trees,20 physical channels,
six original bands,lag4 OAS,40 expanded coordinates,820 tangent features per tree,
cumulative log-Euclidean reference,prior_weight=5,seed=0. Original author and
competition source, data, matrix caches, submission ZIP and previous results are
read-only. Tree draws include the original subject bootstrap RNG consumption.
Training priors and scaler follow the existing all-outer-training-window behavior.

Cache the EXACT float64 standardized train/evaluation feature arrays separately
inside each outer fold. Labels never fit evaluation preprocessing. Preserve
chronological per-session recentering and resets. A common feature cache makes
the intervention literally classifier-only. Refit the original LR on each cache
and require exact label equality with all ten previously stored baseline jobs.
For the official test additionally compare trained coefficients, priors and
scalers with the exported submission arrays. Do not silently accept mismatches.

Evaluation: forest20 only,8 LOSO folds on train+val,one later-session fold on
train+val,then original train-only public-test split (1521 training,755 test
windows). Seven variants give70 jobs. Primary CV selection remains the previous
experiment's mean session-balanced accuracy over8 LOSO subjects. Report cell-
mean leaderboard balanced accuracy,session mean,pooled BA,accuracy,log loss and
multiclass Brier score. Report per-subject paired differences and uncertainty.
Freeze the CV selection and all CV result hashes before evaluating alternatives
on public test. All seven fixed models get test scores; test cannot change the
selection or cause hyperparameter sweeps. Public test has been seen in earlier
work; no claim of an untouched holdout or a new Codabench submission.

## Predeclared classifier configurations

1. baseline: original LogisticRegression(C=.01,max_iter=2000,class_weight=balanced).
2. lda_lw: LDA(lsqr,shrinkage=auto). Exact integer repetition implements original
   subject-bootstrap frequencies; equal class priors implement weighted class
   balancing. LDA is motivated by direct TS-EEG literature; shrinkage is an
   adaptation for820 features and correlated windows, not a copied full pipeline.
3. svm_linear: SVC(kernel=linear,C=1,tol=.001); group-calibrated probabilities.
4. svm_rbf: SVC(kernel=rbf,C=1,gamma=1/820,tol=.001); group-calibrated probabilities.
   Direct EEG TS-RBF evidence is weaker than the foundational linear-SVM evidence;
   do not attribute Gaussian-tangent experiments to Barachant2013's linear TS SVM.
5. ridge: RidgeClassifier(alpha=1,solver=cholesky); group-calibrated probabilities.
   Direct tangent-connectome support is fMRI (Dadi2019), an explicit modality
   transfer rather than an EEG-specific superiority claim.
6. elasticnet: one-hot multi-output ElasticNet(alpha=2e-4,l1_ratio=.5,
   max_iter=10000,tol=1e-4,selection=cyclic); group-calibrated probabilities.
   These penalty values follow the NER2015 winner, but no extra Normalizer,
   xDAWN,meta-features or old normalize=True is imported: fixed existing features.
7. mlp: following the user's switch to Kaggle GPUs BEFORE any ablation accuracy
   measurements,replace the provisional sklearn L-BFGS implementation with
   PyTorch/CUDA,820->32(ReLU)->3,Adam lr=.001,300 full-batch epochs,seed0,
   alpha=1 for explicit L2 on weights (no bias penalty). Each independent tree
   minimizes weighted mean cross-entropy + alpha*||W||^2/(2*sum(weights)). Train
   the50 independent networks with batched matmuls,without sharing parameters.
   No random window-level early stopping or per-test tuning. Literature motivates
   the family; architecture/optimizer are our fixed adaptation,not Cai2024's full
   system. Fixed-epoch training is not a claim of optimizer convergence.

Only classifier hyperparameters change. No search over C,alpha,gamma,network
width,seed or calibration methods after viewing results. Convergence warnings
are recorded per tree. Nonconverged jobs cannot be presented as fully converged.

All alternatives use the SAME effective class-balanced subject-bootstrap
weights as installed sklearn1.9.1 LR: w_i=beta_s * sum(beta)/(L*sum_class(beta)).
This differs from old sklearn versions: class_weight=balanced now accounts for
sample_weight. Test this identity against actual LogisticRegression.

For nonprobabilistic estimators use sigmoid calibration with the pinned sklearn
_SigmoidCalibration implementation,3-fold GroupKFold(or all available groups if fewer),groups are
physical subjects among positive-bootstrap-weight windows. Every calibration
score is generated without training its classifier on that person's windows.
Class balancing is recomputed using only the inner training subjects,so even
the held-out person's label proportions cannot affect their OOF score. Fit
the final estimator on all positive-bootstrap-weight windows,then fit the
three one-vs-rest sigmoid calibrators on OOF scores with balanced outer-training
weights and normalize their probabilities. This is the ensemble=False design.
Original outer-fold geometry remains frozen during calibration; this keeps
the intervention classifier-only and is not nested refitting of the whole
geometry. Every probability vector must be finite,nonnegative and sum to1.

## Integrity and literature

Save protocol and implementation plan in an isolated research git repository
before executing numerical accuracy experiments. Use autoresearch's fixed-scope
experiment/log/findings workflow. No cron or /loop API exists here; monitoring is
in this active turn with persisted state,not a fabricated scheduled job.
Record source/data/cache/submission hashes and installed versions. Save every
ensemble/tree probability and raw prediction. Independently rescore with official
competition metrics. Verify at least one live model against feature-cache
inference and chronological/session reset behavior.

Relevant primary sources are documented in literature/survey.md with explicit
evidence levels and limitations. Publication reputation is secondary to matching
the exact representation; no citation is evidence of superiority on PROTEUS.
