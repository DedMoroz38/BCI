# Active attention SPD experiment — version2

Kaggle private notebook: https://www.kaggle.com/code/den1shh/proteus-attention-spd-ablation
Version2 pushed successfully2026-11-07 and is RUNNING. Version1 was stopped by
the user; API verified CANCEL_ACKNOWLEDGED. Remote execution survives chat closure.
Protocol commit e499aba; execution-only amendment/implementation d0a5d1b; exact embedded source
manifest and notebook SHA in notebook-build.json. Do not rebuild or push a new
version while v1 is running merely to update reports/monitoring.

Two variants: independent WQ/WK attention Gram+ridge and bounded score matrix
exponential. Each50 independent trees on8 LOSO+later+old public test; baseline
refitted and checked exactly. All11 local tests passed; extra CPU-vs-CUDA equivalence test runs remotely. Independent
review important findings fixed before launch. No metrics yet.

Read-only monitor:
reproduction/kaggle_runner/.venv/bin/python reproduction/attention_spd_ablation/follow_run.py 2
Logs saved live-v2.jsonl. Monitor is not included in the original uploaded bundle.

Outputs /kaggle/working/attention-results include intermediate pack .pt/.npz/.json,
final30 model/fold evaluations,selection,REPORT.md,verification.json. Baseline
features are temporary /tmp; result provenance and prediction vectors are saved.

After completion download attention-results with paginated kernels_output
(page_size200; guarded SDK paths). Independent analyzer is analyze_attention.py;
ATTENTION_OUTPUT points to downloaded result folder. It does not refit. Preserve
raw downloads and hash receipt. Final Russian report must state metric table,
paired LOSO deltas,old public test status,and convergence/runtime limitations.
No competition submission authorized or made.


Execution-only v2 change: exact float64 matrix-log eigendecomposition offloaded
to CPU; CUDA projections/backward/optimizer unchanged. No attention fold score
was obtained in v1. All hyperparameters and folds unchanged.

Completion watcher (local exec session21216):
reproduction/kaggle_runner/.venv/bin/python reproduction/kaggle_runner/watch_attention.py 2
It reads status every30seconds,downloads all paginated outputs on COMPLETE,
then runs independent local analyzer. Receipt completion-v2.json; eventual
LATEST_REPORT.md; immutable downloads reproduction/kaggle_runner/downloads/attention-v2.
Live log stream exec session3214. These monitors use no GPU.
