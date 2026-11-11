# Approved validation — local complete; GPU version1 launched

Kaggle CLI confirmed:Kernel version1 successfully pushed.
https://www.kaggle.com/code/den1shh/proteus-custom-person-validation

Private T4,60neural outer fits(3existing families×2seeds×10source pairs),
inner stopping/refit,application cap180minutes,Kaggle timeout12600seconds.
No competition submission,baseline parity gates,old9person weights or poller.
Remote completion/metrics have not been checked after push.

CPU:20/20jobs complete,0failures,530.6seconds. Four model recipes on all10
8/2folds;50experts each. CPU predictions included in the GPU notebook are
fresh8source-person fits,not cached9person LOSO. Each target person held out
twice. Source states reset per session,stream metrics retain natural prefix.

Primary8/2first-session equal-person BA:
- Lag4 LR+balance57.5936%.
- Nuclear.01+balance58.0717%(+0.4780pp,7/10people improve).
- Nuclear.05+balance58.3194%(+0.7258pp,5/10people improve).
- mSSA8/mix25%→Lag4+balance57.6352%(+0.0416pp).

Saved old LOSO forecasts were separately rescored,not retrained:
full fixed30recipe66.6929%,seed-agreement recipe66.5453%,previous normalized
recipe65.0215%,old full recipe64.3810%. These are first-session equal-person
scores,not the old official all-session scores and not new8/2results.

Canonical future validation:../VALIDATION_PROTOCOL.md.
Frozen pairs: splits.json. Detailed prospective design:PROTOCOL.md.
Local result:results_pair_cpu/validation/comparison.csv.
Joint local report:reports/comparison.csv (scheme column separatesLOSO/8/2).
Read saved results again:python evaluate.py (no fitting or monitoring).

When user asks after Kaggle finishes,download custom-validation-results and
copy/link that directory asresults_pair_gpu,then runevaluate.py to combine.
No need to retrain or repeat local CPU work. Remote stdout isexecution.log;
progress.json records60fits. Final report isvalidation/REPORT.md. Learning
curves/checkpoints are under models/pair*/{swad_plain,point_meanstd,
early_and_filter}/seed*/selection/.
