# Research log

2026-10-29: Found the submitted Lag4 artifact and exact previous ten forest20
folds. Locked protocol before any new classifier accuracy results. Existing
submission weights and all earlier source/results stay untouched. No autonomous
scheduler tool is provided; use active-turn monitoring and persistent state.

Implementation workflow: autoresearch,writing-plans,executing-plans and meaningful
test-driven checks. Dedicated sibling repository avoids editing author source.

2026-10-29: User interrupted local preparation and requested Kaggle execution.
No numerical classifier-ablation jobs have started. Ten classifier interface,
weighting and group-isolation tests pass; the tiny MLP fixture reached the fixed
500-iteration limit,which is recorded rather than treated as convergence.
Installed the explicitly chosen shepsci/kaggle-skill in ~/.codex/skills/kaggle;
separate CLI environment in reproduction/kaggle_runner/.venv preserves the
original locked numerical environment. Existing OAuth den1shh accepted by Kaggle;
all three service hosts and Kaggle MCP respond. No dataset uploaded and no remote
run started during connection setup. Plan needs an explicit computational-backend
update before experiments: sklearn does not automatically use a Kaggle GPU.

2026-10-29: Locked CUDA MLP and five classical replacements before any accuracy
experiments. All16 local tests pass. Partial independent review identified
resume/selection integrity omissions; fixed with regression tests. Reviewer
subsequently stopped at its usage limit,so no complete independent code-review
claim is made. Self-review additionally removed held-out class-frequency influence
on inner calibration training weights. No numerical model selection has occurred.

Uploaded fixed inputs privately as den1shh/proteus-lag4-classifier-inputs.
Initial notebook push was rejected before execution: Kaggle's source limit is1MB.
Moved the exact source/archive receipts to a second private input dataset,
den1shh/proteus-lag4-classifier-source; notebook shrank from2.8MB to5.8KB.
Version1 of den1shh/proteus-lag4-classifier-ablation accepted and started with T4.
Source ZIP SHA256 ad744d6a8a3c36b751319833198f3154aefe37ea51a812dc6b51041f2979b15f.
Numerical implementation commit2aa61b5; fixed protocol remains unchanged.

Kaggle version1 exposed automatic ZIP expansion in dataset inputs. Version2
verified all hashes,then exposed disabled ensurepip in Debian Python. Bootstrap
now copies hash-verified unpacked sources and creates a --without-pip venv;
system pip targets that venv. Version3 runs Python3.13.15,PyTorch2.11.0+cu128 and
the same pinned numerical packages as local. All16 remote tests passed; numerical
accuracy work started with LOSO P01. Frozen source bundle is aff4681e0b7ffb459d5ae1161ff7ce96616dfb63bd91d984419d8b62c8a76265.
Only bootstrap changed across failed launches; no alternative accuracy scores
were produced before version3. pip's unrelated preinstalled-package conflicts
are not imported by this experiment; the actual numerical versions are printed
and the implementation tests pass.

Performance-only orchestration amendment after the first six P01 model outputs:
the remaining ElasticNet job blocked all subsequent folds while other CPU slots
were idle. A single persistent three-worker pool now overlaps CPU classifier
fits with preparation of later folds. All CV futures are drained and validated
before selection and any test feature preparation. Classifiers,features,seeds,
hyperparameters and metrics are unchanged. A regression test first failed on
the sequential runner,then passed with the new queue; all17 tests pass. Restart
all jobs in one new Kaggle version/config hash rather than mixing receipts.

Version4 accepted and all17 remote tests passed. Baseline and MLP P01 scores
match version3 exactly; P02 geometry preparation begins while P01 CPU jobs run.
User supplied version3 URL with scriptVersionId363509788 after the browser
access tool denied Arc. The official CancelKernelSession SDK call returned no
error and version3 reached CANCEL_REQUESTED; version4 remains RUNNING.

Independent follow-up review of0350b84 found no blocking training/evaluation
bugs. It requested independent probability-metric verification and the planned
selection-stability summary. Added manual logloss/Brier checks,paired-person
bootstrap winner frequencies,and leave-one-person-out reaggregation. These do
not refit models or change selection. New standalone analysis tests exposed a
real import collision: common prepends an upstream directory containing another
run.py. The analyzer now loads its own runner by exact path. Three analysis tests
pass and independent reviewer found no new errors. Training config hash remains
unchanged. The already-running v4 will retain its original report cell; finish
and verify its immutable70 outputs with the corrected analyzer,without retraining.

2026-10-29 final: Kaggle training v4 completed all70 jobs successfully,including
its original report cell. The anticipated import collision did not occur there:
the remote source bundle contains temporal_features.py and temporal_cache.py,
not the other upstream run.py. The exact-path fix remains needed for local
standalone analysis. No classifier configuration changed.

Downloaded158 selected output files (all70 job JSONs and70 prediction NPZs),
preserving a raw download receipt. Kaggle CPU report notebook v1 and independent
local analyze.py both verified all70 outcomes,all10 baseline label matches,
63 frozen CV result hashes,and independently recomputed probability metrics.
Public-test baseline parameters differ from the exported submission by at most
1.0193e-9;geometry differences are below4.68e-14. Both notebooks are COMPLETE.

CV selects linear SVM:57.5186% vs baseline56.2801% mean session BA,+1.2385pp.
Its exploratory paired95% interval is[-0.8644,+3.3553]pp. It wins5/8 people,
remains the winner in8/8 leave-one-person-out reaggregations,and wins69.705% of
paired person bootstraps. Public-test cell BA falls to50.4447% vs55.7164% LR.
No fixed alternative beats LR on that metric; LDA has slightly better logloss
and Brier. MLP scores54.5185% LOSO and53.6039% public-test cell BA. ElasticNet
emits2423 ConvergenceWarnings,so its low result is optimizer-limit qualified.
The experiment establishes outcomes for these fixed configurations,not optimized
families. Test has only two people and was previously seen. No new submission.
Verified summaries are in to_human/,raw model outputs in ignored results/.
