# Single-model Lag4 versus end-to-end channel attention SPD

2026-11-07,user replaces forest experiment: compare baseline and attention
WITHOUT any forest. User will stop old GPU notebook. No competition submission.
Independent W_Q/W_K remain mandatory. Preserve BOTH previously requested SPD
constructions as separate single-model alternatives. Three rows,not an ensemble.

## Fixed input and evaluation
All41 original channels,one band4–40Hz,same4second120Hz window,lag4 expansion
into82 rows x476 samples. All outer-training people/windows,no bootstrap or
random channel selection. Same10 folds as previous experiments:8 LOSO on
train+val,one later-session fold,and original train-only public test.
Baseline: Lag4 OAS82x82,cumulative log-Euclidean session reference with prior5,
3403 tangent coordinates,training StandardScaler,balanced LR(C=.01,max_iter2000).
Reproduce original single Lag4 logreg CV predictions wherever saved controls
exist. Do not compare against old50tree55.716% as if it were the same baseline.

## Attention encoders
Reuse previously specified channel-token encoders with channels82,samples476,
Q/K dimension16,independent learned temporal W_Q and W_K,layer-normalized Q/K.
Training-only scalar RMS normalization after per-channel temporal centering.
Gram: V=U W_V (64 dims),A=softmax(QK^T/4),Z=V+.1 A V,feature-center Z,
R=ZZ^T/64+.01 I. Exponential: B=sym(tanh(QK^T/4))/82,R=exp(B).
No parameter sharing between the two alternatives or outer folds.

## TRUE end-to-end training
One final linear softmax classifier,not an auxiliary/discarded classifier.
At EVERY full-batch training step:
1. Encode all TRAIN windows into R and logR.
2. Compute training prior logP=mean(logR),retaining its gradient.
3. In each training session,compute logM_t=(5logP+cumsum(logR)_t)/(5+t).
   No detach/truncated gradient; cumulative path differentiates through all
   previous windows of that training session and through the training prior.
4. W_t=exp(-.5logM_t),tangent=upper(log(W_t R_t W_t));offdiag*sqrt2.
5. Compute population training mean/std of tangent features WITH gradients.
   For attention,std below1e-6 is replaced by1 to avoid amplification of
   numerically constant directions. Baseline retains original sklearn scaler.
6. Feed standardized tangent features into the SAME final LR head; minimize
   balanced weighted crossentropy +||W_head||²/(2*.01*N).
Train encoder and final head together using AdamW lr=.001,encoder decay=.001,
head decay0,clip global gradient norm5,80 full-batch updates,seed0.
Initialize the final head once with exact sklearn LR on initial encoder's
causal standardized features; keep that head and optimize it jointly. This is
initialization,not training an auxiliary identity-tangent head. No evaluation
labels/inputs in representation fitting,prior,scaler,head initialization or loss.
No minibatch ordering or truncation: all training sessions vectorized by cumsum.

Symmetric log/exp use CPU float64 eigendecomposition with analytic Loewner
backward,then eigenpairs returned to CUDA. This handles repeated eigenvalues
and avoids slow T4 dispatch. Other projections/matrix products and optimizer
use CUDA. Every final output uses frozen learned encoder+retained final head.
After last update,recompute prior/scaler from training windows under final
encoder. Do NOT refit/discard/replace the jointly trained head.
Offline training prior/scaler use all TRAIN windows,as baseline does; inference
prior/scaler are frozen from training. Evaluation state uses only current and
past evaluation windows,reset each participant-session. No future eval windows.

## Frozen reporting
Primary old mean LOSO session BA;also official cell BA,late-session BA,public
test session/cell BA,pooled BA,accuracy,logloss,Brier. Freeze CV hashes/ranking
before new test scoring. Public test was seen before;no untouched-test claim.
Report paired8person deltas and descriptive bootstrap CIs,train loss curves,
gradient norms,runtime,convergence warnings.80updates is not convergence proof.
Save encoders,retained head,prior/scaler,probabilities,input/source hashes.
Independent NumPy/pyRiemann streamed inference roundtrip must reproduce output.

## Implementation plan and checks
1. Write tests for differentiable symmetric exp,session tangents versus original
   RiemannFeatures,causal prefixes/resets,and gradients through reference history.
2. Implement single_e2e.py;test actual training/save/reload/deployed head equality.
3. Implement10fold driver,exact original single-model baseline checks,resumable
   provenance and CV freeze before test. No new architecture/hyperparameter search.
4. Independent review;new private Kaggle T4 notebook and remote tests before fit.
5. Automatic result saving,independent rescoring and Russian comparison report.
