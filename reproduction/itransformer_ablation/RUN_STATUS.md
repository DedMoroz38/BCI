# Author iTransformer without lags versus single Lag4 — active v1

Private Kaggle:
https://www.kaggle.com/code/den1shh/proteus-itransformer-vs-lag4
Version1 pushed successfully2026-11-07;initial stateQUEUED. Exact upload source
commit9ac9bfd;protocol7256e70;notebook/source SHA in notebook-build.json.
Official upstreamc2426e68ca13f74aaec08045c5c724d8ad328124;MIT retained.
No competition submission. Do not rebuild/restart solely for reporting changes.

Two model families: SINGLE Lag4 baseline and iTransformer WITHOUT lag copies.
iT41x480,d64,1block,4heads,FFN128,dropout.2,GELU,flatten+linear3class head.
80maxepochs,patience10,min_delta1e-4,batch64,AdamW.001/decay.001,clip5.
Three fixed seeds0,1,2;individual predictions,average METRICS only,no ensemble.
10article-aligned outerfolds;baseline10jobs+neural30jobs=40evaluations.
Nested validation held-out PEOPLE only within outer-training set. Selected epoch
sets fresh outer refit length. Outer-eval labels/windows never guide fitting.
Author optional per-channel variance norm disabled;training-only scalar RMS
preserves EEG relative powers. FullAttention/Encoder math remains upstream.

11local tests passed,CUDA smoke runs remotely before fitting. Independent review
important cache/hardware provenance issues fixed with failing→passing regressions.
Selected-checkpoint loss replay,artifact tamper,seed-config and outer split guard
also tested. Notebook cells compiled and format validated.

Read-only live monitor exec session26564:
reproduction/kaggle_runner/.venv/bin/python reproduction/kaggle_runner/follow_itransformer.py 1
Logs live-v1.jsonl here. Live PNG plots displayed/updated by notebook parent.
CSV each epoch;PNG every5+first/final epochs. Inner baseline shown as dotted
reference lines;outer refit has separate train-history CSV.

Completion watcher exec session47695:
reproduction/kaggle_runner/.venv/bin/python reproduction/kaggle_runner/watch_itransformer.py 1
Downloads all it-results files (pagination200) on COMPLETE to
reproduction/kaggle_runner/downloads/itransformer-v1/it-results,
runs independent analyze_itransformer.py and writes LATEST_REPORT.md here.
Runtime receipt completion-v1.json. Save30learning-curve PNGs+CSVs and all40
predictions/checkpoints. Previous experiments/artifacts untouched.
