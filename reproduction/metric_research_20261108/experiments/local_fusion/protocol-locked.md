# Metric research wave1 — 2026-11-08

User-authorized objective: maximize PROTEUS competition balanced accuracy;
fast work on local Mac,heavy experiments privately on Kaggle in background.
No competition submission is authorized by this protocol.

## Continuity and available compute
No agent cron/heartbeat tool is available. Do not fabricate recurring agent
execution. Persist research-state,logs,reports and runnable batch jobs. Kaggle
continues independently of chat; read-only watchers record status/logs.
Observed GPU quota:21600s total,15895.634131s used,0reserved,~5704s remaining.
Wave1 GPU wall budget75min,stop scheduling optional work after65min; preserve
partial artifacts if budget reached. No third concurrent GPU job.

## Learned constraints
Lag4 baseline56.28% legacy LOSO,55.72% old public cellBA. Causal probability
balance alpha1 improved legacy LOSO to57.57%,late78.37%; no public-test score
was measured for that new variant. Author forest+CNN raised legacy LOSO to
61.37% but public-test cellBA45.95%; geometric pooling and overconfident
100epoch CNN may be contributors,not established causes. Author extras did
not surpass Lag4. Pseudo-label adaptation had negative external repo evidence.

## H1 — local reuse of trained CNN predictions
Use the10verified saved CUDA-trained checkpoints from ensemble recoveryv3.
Extract the pure CNN's raw logits with exactly author band1–58Hz,causal
Euclidean alignment,prior5,source ShallowNet config. No new neural training.
Verify that reconstructed original forest+.3geometric CNN reproduces saved
author predictions (exact labels,probabilities tolerance1e-4); fall back toB1
if batched inference cannot meet this guard. Metadata and checkpoint hashes
must agree. Never mix checkpoints across held-out people.

Legacy evaluation:8LOSO on train+val,one late fold,then old train-only/public
test split. Baselines:Lag4 and Lag4+balance(alpha1,prior10). Fixed candidates:
arithmetic Lag4/CNN mixtures,CNN weight.1/.3,temperature1/2/4,with/without
balance1;geometric weight.3,temperature1,with/without balance1;CNN-only
diagnostic. Temperatures are fixed exploratory values,not claimed calibrated.
Select by mean8LOSO cellBA,tie-break logloss and simpler candidate; freeze CV
hashes/config before evaluating all fixed candidates on old public test.
Old public test has repeatedly been seen; it is diagnostic,not untouched.
No pseudo-nested calibration from other outer-fold predictions: those models
include the target person's training labels and would leak into inner selection.

## H2 — full public cohort local baseline
All2655public windows,10participants.10LOSO fits each train9/test1;late fold
holds last session of both multi-session participants,train earlier sessions
and other people. Exact submitted50-tree Lag4,20physical channels,6bands,
prior5,C.01,seed0. Reuse SHA-verified pure matrix cache. Save tree probabilities,
stream metadata and parameter arrays without pickling matrix caches. Include
fixed balance1 postprocessing. Primary new-wave metric:mean10LOSO cellBA;
late-cellBA secondary. These new numbers are not directly comparable to legacy8.

## H3 — nested person-held-out CNN training on Kaggle
Same full-cohort11outer folds as H2. Inner GroupShuffleSplit(test_size=.2,
random_state33),groups=outer-training people; no outer person/session/window
used for early stopping,alignment priors or optimization.
Author ShallowNet temporal/spatial/square/log design,band1–58Hz,causal EAprior5.
Three configurations,seed0 first on all folds:
- shallow40_class:n_filters40,dropout.5,class-balanced training CE.
- shallow16_class:n_filters16,dropout.5,class-balanced training CE.
- shallow16_subject:n_filters16,dropout.5,equal total weight per source person
  and per class within that person. This is a domain-generalization hypothesis;
  it is not identical to competition cell balancing.
All AdamWlr.001,WD.01,batch64,CosineAnnealing T_max100,max30selection epochs,
patience6,min_delta1e-4. Inner selection uses equal-cell/class-weighted NLL
for all configurations; report train/val NLL and cellBA curves every epoch.
Fresh initialization/refit on all outer-training people for best_epoch exactly,
refitting its EAprior; no target labels. Keep aligned tensors on GPU,assign
different T4devices to two workers,BLAS1/torch2threads. Float32,noAMP.
If budget permits after all seed0 jobs,train seed1 for both compact variants;
report seeds separately and arithmetic probability ensemble,never best seed.
Checkpoint every epoch; exact parameters,FP32 replaytol1e-4+exact labels.
Do not average away different input metadata or failed/missing folds.

## Analysis and continuation
Pair methods on the same participants and fitting cohort; primary10person
cellBA,late-cellBA and per-person differences,logloss/Brier,latency. Future
mixture choices must come from public cross-validation and be labeled
exploratory; no claims of sealed improvement or independent confirmation.
At completion synthesize calibration vs capacity vs source-person weighting.
Choose the next hypothesis from results rather than launching arbitrary grids.
