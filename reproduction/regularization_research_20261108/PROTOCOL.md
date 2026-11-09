# Regularization wave2 — 2026-11-08

User explicitly authorized previous suggestions2(label smoothing),3(augmentations,
many variants starting simple),4(weight averaging). Do not change to BA epoch
selection or add pretrained/foundation models (suggestions1/5 rejected).
No competition submission. No background polling: user reports completion.
Kaggle private T4x2; training wall budget240minutes,save partial results.
User reports25GPUhours available;240min is a per-run safety cap,not a wait.

## Fixed controls and inference
All10public people,2655windows. Same10LOSO and later-last-session fold aswave1.
Same inner person-held-out split(test_size.2,seed33);NLL selection stays equal
cell/class-weighted NLL,max30epochs,patience6,min_delta1e-4. Fresh outer refit
for selected epoch/range. No outer labels for stopping,BN statistics or choosing
averaging ranges. AuthorShallowNet16,class-weightedCE,dropout.5,AdamWlr.001,
WD.01,batch64,CosineTmax100,band1–58Hz,causalEAprior5 stay unchanged.
Evaluate clean windows only; augment training inputs after causal EA. These
41coordinates are aligned input components,not physical missing-electrode
simulation. Caches fitted separately on inner/outer training only,stored /tmp.

User steering2026-11-09: no separate control training. Neither clean-control
nor BN-refresh-only jobs are scheduled. Compare against already savedwave1
results and SHA-verified matching Lag4 and
Lag4balance1 predictions. All variants seeds0/1;report both,probability mean2,
never choose best seed. Same full cohort;primary pooledOOF officialcellBA,
mean-personBA separate,late2person secondary. Legacy8 not directly comparable.

## Twenty-nine simple configurations,then six fixed combinations
LSε=.02,.05,.1,.2 (4). RMS-scaled Gaussian noiseσ=.01,.03,.05 (3).
Global input gain uniform[1−g,1+g],g=.1,.2 (2);coordinate gain g=.1,.2 (2).
Zero exactly1/3/5 random input coordinates per window (3). Contiguous time mask
24/48samples,shared across coordinates (2). Reflection-padded temporal shifts
uniforminteger±6/±12samples (2). Global sign inversionp=.5 (1).
Cross-person same-class input mixup betaα=.2/.4 (2),identity if no eligible
partner. Training-only within-class/cross-person MixStyle on temporal log-power
features,p=.5,betaα=.1/.3 (2);identity where no partner orconstant temporalvar.
Trailing uniform weight average3/5epoch checkpoints (2). Parameter EMA after
each optimizerstep,decay.95/.99 (2). SWAD LossValley adaptation n_converge3,
n_tolerance6,ratio.05/.1 (2): dense within-epoch parameter means,innerNLL,
author scalar valley-selection logic;onno convergenceuse last segment asauthor.
Total simple configs29;no separate controls.

Six fixed combinations:LS.05+noise.03;LS.05+drop3;LS.05+globalgain.2;
noise.03+drop3;LS.05+noise.03+drop3;LS.05+trailingavg3. Total35configs,770fits.
Schedule each complete configuration/seed over all11folds in this order,
simple before combinations. Stop scheduling if<3min remain; everybatch deadline.
Partial configs are explicitly incomplete and excluded from full-CV rankings.

All augmentation randomness uses a dedicated checkpointed generator; clean
control draws no extra global RNG. Dropout remains identical control behavior.
Label smoothing affects training only;validationNLL always hard targets.
Averaging averages parameters,not optimizers. RecomputeBN using clean training
inputs only,dropout/augmentation disabled;never val/test stats. For trailing/EMA
select best averaged checkpoint by unchanged innerNLL. ForSWAD retain author
valley rule driven by NLL;record explicit epoch-range and refit only that range.
Record raw and averaged validation curves so differences remain auditable.

## Analysis fixed before runs
StandaloneCNN softmaxT1;fixed ensemble70%Lag4+30%meanCNNsoftmax(logits/4),
then causalbalance(alpha1,prior10). No sweep of ensemble weights/temperature.
All11folds per seed required;no best seed. Compare loss/confidence/Brier,cellBA,
per-person changes and clean train/val curves,not merely shrinking train-valgap.
Public CV is reused and many methods tested: rankings exploratory,not sealed
confirmation. Validate no-data mutation,eval augmentation bypass,LS loss,
parameter averaging,BN train-only behavior,resume RNG/optimizer and clean
unit-test compatibility before pushing. Preserve logits,models,alignments,
selection/refit histories and learning curves with hashes and CPU/GPU replay
diagnostics. Probability/label differences from historical runs or other
backends never block the grid. Saved/loaded parameters must remain exact.
Individual model exceptions are recorded as failed jobs;remaining jobs run.
Incomplete configuration/seeds stay out of full-CV rankings.
