# Author ensemble vs submitted Lag4 — 2026-11-08

Authorized: reproduce the algorithms in DorinDaniil/Brain-Decoding ensemble.py
locally and compare with the user's submitted Lag4 forest.

Author source pinned at 934f423a73e1ed50d653345d27256ecfe50ab009 in a separate
unchanged checkout ../ensemble_author_20261108. Original source and experiments
remain unchanged. No Kaggle run or competition submission.

Primary evaluation matches previous experiments: train+val (1900 windows,8
participants),8 leave-one-person-out folds; one last-session fold; afterwards
train-only (1521 windows,6 participants) vs public test (755,2). Author CLI's
default all-10-person CV is deliberately not used: it would change the task.
Public test is previously seen, not an untouched holdout. No hyperparameter search.

Compare author's seven VARIANTS: forest,+balance,+bands,+routing,+deep,
all but deep,all. Exact fit_components/assemble math and defaults: 50 global
trees,20 channels,10 trees per subject,C=.01,prior5,forest seed0;
deep seed0,100 epochs,AdamW lr.001 WD.01,batch64,cosine schedule;
mix .3,geometric probability pool; balance alpha1,prior10. One deep seed
matches benchmark.py, whereas make_submission.py defaults to three.
No Lag4 is added to author algorithms. Training priors/scalers include all
outer-training windows as upstream, despite bootstrap zero-weight rows.

Baseline is existing submitted 50-tree six-band Lag4 OAS,20 physical channels,
seed0,C=.01,prior5,40 matrix coordinates/tree. Refit independently using old
unchanged temporal_features.fit_model and existing SHA-verified matrix caches;
require exact prediction equality with all10 previous baseline results.

All inference is chronological,one window at a time,reset per person-session.
Save probabilities,predictions,metadata,model specs,training/evaluation times,
warnings and source/data/environment hashes. Resume only under identical code,
protocol and inputs. Verify saved-model replay,stream reset and causal prefix.
Report session-mean BA (previous primary),cell-mean BA (leaderboard),accuracy,
log loss,Brier,and per-person paired LOSO differences vs Lag4. Bootstrap paired
people for descriptive95% CI; folds have overlapping training sets.
Freeze CV result hashes before test; test cannot affect parameter choices.

Execution: local CPU, two fold processes, two torch threads per worker,BLAS1;
no modification of upstream algorithms for speed. Save nondeep components
before deep training; finished fold artifacts survive interruption.
