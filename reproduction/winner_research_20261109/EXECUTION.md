# Executed research package

CPU: 21 new classic experts (coherence4, phase7, powers2, CSP4, MiniRocket4),
6 saved-prediction variants, 3 frozen-CNN geometry variants, 11 outer folds.
New experts are scored alone, with .1/.2 late fusion, and .1/.2 prior fusion
followed by the existing causal transduction. CPU wall cap 240 minutes.

GPU: four causal latent-alignment Shallow16 models and two TSMNet-derived
models; seeds0,1; 11 outer folds =132 outer fits with inner selection/refit.
Inner cell-NLL criterion, max30epochs, patience6. Two-seed means are the
primary neural comparison; individual seeds are diagnostics. GPU wall cap
690 minutes; finishes immediately when all work is done.

Latent alignment uses author-compatible ShallowNet encoder, source-only BN
recalibration frozen during prefix passes, source-shrunk causal context,
and our prior SWAD.1 averaging policy. This is a new training recipe;
effects include the BN/prefix policy, not solely an adapter layer.

TSMNet-derived models retain author4 temporal filters,40 spatial filters,
20-dimensional Stiefel BiMap, ReEig, custom spectral backward and LogEig.
Normalization is our causal log-Euclidean mean/scalar-dispersion adaptation,
prior10, reset per stream, NOT an exact reproduction of author SPDDSMBN.
SPD layers and head run in float64 on CPU as in the author implementation;
convolutions run on GPU. RiemannianAdam lr.001, decay.0001; best NLL checkpoint,
no averaging of Stiefel parameters. Input band4–36; rawLag4 variant has82rows.

Source-only feature fitting; current/past target windows only. No validation
data for BN/source statistics, no target labels for adaptation. No future-window
or full-session target statistics. No user attachments are uploaded.

Historical Lag4/CNN/62.7234% predictions are bundled, not refitted or used as
blocking tolerance controls. Numerical/runtime failures are recorded per job
and independent jobs continue. Atomic prediction receipts prevent the previous
intermediate-summary race. Reports are collected after independent workers end.

The trainable stacking gate and its new inner source-fold fits are the later
dependent wave D: not claimed as part of this initial fixed-feature/CNN run.
SPDIM, extra regularization combinations and personalization remain reserve.
No competition submission and no local background poller are started.
