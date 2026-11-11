# Current PROTEUS validation — approved10November2026

Current source/prospective splits:
[custom_validation_20261110/PROTOCOL.md](custom_validation_20261110/PROTOCOL.md).

1. Score only the first session of a completely held-out person and give
   every person equal weight. Keep official all-session cellBA as a separate
   reference. Do not let P04/P10's extra sessions triple their primary weight.
2. Fixed10fold8train/2test schedule:metadata-only random participant ring,
   seed20261110. Every participant held out twice against different source
   sets. Average metrics within person across repeats,then average10people;
   do not ensemble repeated predictions into a new predictor. All sessions of
   both held-out people are excluded from all source fitting.
3. Evaluate the first session in chronological bins1–20,21–60,61+; keep
   class/context support. Predictions are made with the natural past prefix,
   no extra resets between bins,contexts or runs. Session reset is mandatory.

Hyperparameters frozen for comparisons. Neural epoch selection uses6inner
training people/2inner validation people within the8source people,seed33.
Refit on8source people after selection. Outer pair never selects stopping,
normalization,weights,calibration,components or source PCA/TTA geometry.

Report mean first-session BA,worst3people,NLL,perperson changes and repeated
source-set sensitivity. Existing10-person LOSO can be rescored for items1/3;
it cannot supply8/2 predictions. New splits do not make previously explored
participants a fresh hidden test. No reproduction/parity gates or submission.
