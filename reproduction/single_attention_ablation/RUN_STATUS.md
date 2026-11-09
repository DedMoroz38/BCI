# Active SINGLE-model full-gradient experiment

Actual Kaggle URL:
https://www.kaggle.com/code/den1shh/proteus-single-attention-end-to-end
Version1 pushed2026-11-07;QUEUED at first status check. Kaggle assigned slug
from title;it differs from requested `proteus-single-attention-e2e`. Local
kernel-metadata.json corrected to actual slug. BEFORE any future rebuild fix
build_notebook.py's id too;do NOT rebuild/push solely to correct that helper.
Uploaded source bundle remains notebook-build.json/notebook source SHA.

Protocol e314fdb;uploaded implementation595c178. Three SINGLE models:
Lag4 OAS+causal tangent+LR,attention Gram,and attention score exponential.
41channels,band4–40,lag4,10old folds. Both attention models train full gradient
through prior,entire session reference history,tangent,training scaler,and the
retained final logistic head. No auxiliary/discarded head,no forest/bootstrap.

Old forest notebook den1shh/proteus-attention-spd-ablation/v2 was stopped by
user;API verified CANCEL_ACKNOWLEDGED. No attempt to restart it.

8local tests passed;CUDA full1800window82channel gradient smoke runs before
accuracy jobs on Kaggle. Independent review found no blocking correctness issue.
Reviewer runtime/library limitations addressed by full-shape smoke and runtime
version checks.80full-batch steps remain fixed;no convergence claim.

Read-only log stream exec session9514:
reproduction/kaggle_runner/.venv/bin/python reproduction/kaggle_runner/follow_single.py 1
Logs single_attention_ablation/live-v1.jsonl.

Completion watcher exec session32065:
reproduction/kaggle_runner/.venv/bin/python reproduction/kaggle_runner/watch_single.py 1
Reads status every30sec;on COMPLETE downloads paginated single-results outputs
using download_single.py,then independently runs analyze_single.py locally.
Mutable receipt completion-v1.json. Download target:
reproduction/kaggle_runner/downloads/single-e2e-v1/single-results
Final LATEST_REPORT.md copied here. Remote notebook also analyzes automatically.

All models retain weights,references,scalers/probabilities and input/source hashes.
CV selection frozen before public test. Public test was seen before. No new
competition submission. Do not compare new single baseline with old submitted
forest55.716% as if identical. Final metrics pending.

Live verification: new v1 allocatedTeslaT4. All9remote tests passed,including
full1800window82channel end-to-end gradient smoke. First LOSO P01 baseline
reproduced original single-model implementation and historical predictions;
sessionBA .546875,cellBA .547043 (ONE fold,not overall metric). First Gram
end-to-end steps finite: epoch1 loss .171166,epoch10 .099831;~4.1sec/update.
Preliminary whole-run estimate1.5–2.5hours;no attention fold metrics yet.
