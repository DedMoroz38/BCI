# Research log

2026-11-08: frozen protocol at commit5b064b3; implementation6ccd923.
Implementation tests:5 passed. Fresh read-only reviewer checked protocol,
folds,default arguments and streaming reset/serialization. Fixed identified
cache-pickling issue before launching: save baseline parameters only; also
pin cache index manifest and completion receipts. Author source unchanged.

Device pilot (no evaluated accuracy,not part of final scores): first dev LOSO
training windows,one/four epochs CPU and MPS. Four epochs CPU18.50s,MPS22.02s;
keep upstream default CPU in sandbox. Both full-fold jobs use torch2threads,
BLAS1,matching protocol. Local Apple GPU is available outside sandbox even
though sandbox reports MPS unavailable.

Main run launched locally,PID65040,logs/run.log. Both first Lag4 folds exactly
matched historical predictions; author forest first2 also matches old original
forest session scores exactly. No conclusions before full LOSO.

User requested adding ../forest_lag100 (only inference and weights supplied):
100experts,60 original OAS views,40 band-dependent quarter-period-lag views;
20 physical channels/expert,prior5,6bands. Training code/regularization/bootstrap
recipe and training cohort unknown. User subsequently explicitly deferred this
variant until supplying details. Do not evaluate provided weights on LOSO as
though held-out people were unseen. No changes to forest_lag100 files.
