# Learned attention SPD representations against the submitted Lag4 forest

Authorized 2026-11-07: run both previously proposed representations on Kaggle.
Q and K projections MUST be separate independently learned parameter tensors.
No competition submission. Fixed finite ablation, no open-ended optimization.

## Design and controls
Baseline: immutable submitted Lag4 forest,50 trees,20 physical channels per
tree,6 original frequency bands,lag4 at120Hz,40 expanded channels,820 tangent
coordinates,subject bootstrap seed0,cumulative log-Euclidean reference prior5,
StandardScaler and balanced LogisticRegression(C=.01,max_iter=2000).
Use the EXACT old10 outer folds:8 LOSO on train+val,one later-session split,
and original train-only public test. Reproduce all10 baseline labels exactly;
on original test verify baseline parameters against exported submission.

Each attention tree sees exactly its original band and40 lag-expanded rows
of476 time samples,centered within each row,divided by a single RMS estimated
ONLY from its positive-bootstrap-weight training windows. Each tree owns its
own encoder and is trained only on its positive-bootstrap-weight people.
No sharing learned parameters across trees or outer folds. All evaluation
labels,including later sessions,are excluded from fitting representations,
references,scalers,auxiliary heads and classifiers. No future evaluation windows
enter state. Current whole4second input window is allowed.

For channel tokens U of shape40x476:
Q=layer_norm(U W_Q), K=layer_norm(U W_K), d_q=d_k=16;
S=Q K^T/sqrt(16). W_Q and W_K are independent with different initialization.

A (attention_gram): A=softmax_row(S), V=U W_V (64 features),
Z=V+.1 A V, centered over64 features,
R=ZZ^T/64 + .01 I. W_V is learned independently per tree.
The explicit .01 ridge guarantees SPD,even for constant inputs.

B (attention_exp): B=(tanh(S)+tanh(S)^T)/(2*40), R=matrix_exp(B).
The division bounds ||B||_2<=1,so eigenvalues of R are in[e^-1,e].
This bounds the spectral range without tying W_Q and W_K or applying
row-softmax to R. Symmetrization is applied to scores,not to parameters.

These are our fixed architectures,not literal reproductions of MAtt.
Attention axes are channels/lag copies; trainable projections act on the
within-window temporal samples. They do not aggregate predictions over windows.

## Supervised representation fitting
Train end-to-end using an AUXILIARY linear3-class head on upper(log(R)) at
the identity reference. This is a training surrogate,not the deployed head.
For exp,log(R)=B exactly; Gram uses a symmetric matrix logarithm with a stable
Loewner divided-difference backward,including repeated eigenvalues.
Freeze initial training-feature mean/std (std floor .02),computed using only
positive-bootstrap-weight training windows. These auxiliary features differ
from the final causal tangent features. Save training-loss curves and gradient
checks; do not claim the full deployed pipeline was jointly optimized.

Fixed seed0,80 epochs,batch128,AdamW lr=.001,encoder weight_decay=.001,
global gradient norm clip5 per pack of trees. Auxiliary head penalty follows
LR strength: ||W_head||^2/(2*.01*sum(effective_weights)). Effective weights are
the same class-balanced subject-bootstrap weights as baseline. Tree packs of5
are computational batching only; each tree's initialization uses its tree ID,
each tree's shuffle is common across a pack,and objectives sum across trees.
No early stopping,hyperparameter search or evaluation-driven changes. Fixed
epoch count is not proof of convergence. Report nonfinite failures explicitly.
After training discard auxiliary classifier,freeze encoder,recompute ALL
training/evaluation R,fit original references/scaler,and refit original LR.
Inference probabilities are mean of50 tree probability vectors.

## Evaluation and evidence
Report old primary mean LOSO session BA,official cell BA,late session scores,
original public-test cell/session BA,pooled BA,accuracy,logloss,Brier,per-person
paired differences and descriptive paired bootstrap CI. Freeze all CV result
hashes and ranking before new public-test scoring. The old public test has
already been seen: no claim of an untouched test. No tuning after viewing it.
Preserve source/input/protocol hashes,encoders,reference/scaler/LR weights,
per-tree probabilities and runtime. Independently recompute metrics. Causal
prefix tests and reset tests must pass. Old artifacts stay unchanged.

## Implementation plan
1. Test SPD/finite gradients/repeated eigenvalues/independent QK and learnability;
   implement attention encoder and supervised trainer in attention.py.
2. Implement10fold driver using old exact baseline prepare/run,then new matrices
   through original RiemannFeatures and LR. Save resumable tree-pack checkpoints.
3. Test synthetic fit/predict,weights exclusion,stream prefix/reset,and saved-state
   roundtrip. Audit code independently before expensive GPU execution.
4. Bundle immutable source plus old inputs into a new PRIVATE GPU Kaggle notebook.
   Run mathematical tests remotely before training; persist outputs as each pack
   completes; resume validated artifacts if infrastructure fails.
5. Download completed outputs,independently rescore,write Russian metric report.

Research context: Vaswani etal NeurIPS2017 attention;
Pan,Chou,Wei MAtt NeurIPS2022 uses SPD manifold attention AFTER covariance,
not this exact covariance replacement. No literature claim of PROTEUS gain.
https://papers.neurips.cc/paper/7181-attention-is-all-you-need
https://papers.neurips.cc/paper_files/paper/2022/file/c981fd12b1d5703f19bd8289da9fc996-Paper-Conference.pdf

## Execution-only amendment before any attention evaluation (v2)
Version1's T4 cuSOLVER tiny40x40 batched eigendecompositions took about12.5sec
per epoch in the first pack,projecting beyond one notebook's runtime. No
attention fold scores were obtained or inspected. Move ONLY the float64
symmetric eigendecomposition inside the custom matrix-log forward to CPU,
then move eigenpairs back to CUDA for the SAME Loewner backward. Encoder,
loss,optimizer,epochs,batches and final matrices are unchanged. CPU benchmark
640x40 eigendecompositions ~.067sec. Add a CUDA test comparing values and
parameter gradients against native CUDA eigh at tight tolerances. Restart the
same frozen experiment as v2; stop superseded v1 when session ID is available.
