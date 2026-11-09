# Author iTransformer comparison ledger

2026-11-07: User requests author iTransformer,no lag copies,previous proposed
architecture,train/val loss plots,Kaggle execution,and article-aligned local
validation because competition submissions unavailable. No submission requested.
Protocol7256e70 committed before numerical accuracy experiments.

Read original local PDF EEG protocol:6train2val2test,8LOSO andone later-session
fold,session-mean BA. Keep those outer folds;also report cell BA. Baseline is
SINGLE Lag4,not the submitted forest. Three fixed neural seeds for stability,
metrics averaged across seeds;predictions never ensembled.

Downloaded official THUML commitc2426e68ca13f74aaec08045c5c724d8ad328124,
retained MIT license and per-file original/vendored hashes. Only author-file
patch lazily imports unused Reformer dependency in its own constructor;selected
FullAttention and Encoder untouched. Classifier subclasses official Model,
replaces forecast projector with41*64→3 and exposes encoder hidden states.

Norm choice: disable author's optional per-channel variance normalization;
center each row and fit one scalar RMS from training only,preserving relative
EEG powers. Architecture1block,d64,4heads,FFN128,dropout.2,GELU. Training AdamW
.001,decay.001,batch64,clip5,max80epochs,patience10,min_delta1e-4 fixed a priori.

Nested split uses only outer-training people. Inner train/val evaluation both
in eval mode. CSV each epoch,PNG every5+first/final;plots include unweighted CE,
weighted CE and session BA,plus fixed inner-baseline lines. Best inner epoch
selects fresh all-outer-train refit length;no outer-evaluation feedback.

Independent reviewer found2 important provenance gaps: inherited matrix cache
reader lacks per-array checksum,and inherited runtime allowed device changes.
Added explicit band0 matrix/receipt/manifest checksum and strict hardware/cache
runtime fingerprint. Tests failed before implementation,then passed. Added
selected-checkpoint validation-loss replay,changed selection-config rejection,
artifact tamper rejection,and outer-person split guard. Encoder equivalence now
compares against independent original Model.forecast with captured hidden state,
not the adapter's own module twice.11CPU tests pass;CUDA smoke runs on Kaggle.

No new accuracy scores seen yet. All former single-attention source/results
remain untouched. New notebook uses same immutable private data/source datasets.
