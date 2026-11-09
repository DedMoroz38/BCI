# Quick local screening — research record

2026-11-08: user requested only experiments that finish quickly locally.
Selected three mechanisms from the supplied repo synthesis: mixed views,
causal dispersion normalization and predicted-frequency balancing. Four fixed
alternatives plus exact historical Lag4 control. No new neural fits or changes
to Kaggle job.8LOSO+late,no public-test evaluation or adaptive hyperparameter sweep.

Local tests4passed,covering exact DedMoroz RMS mechanism,independent block-order
adaptive covariance construction,OAS SPD,causal prefix/reset and cache digest.
Fresh reviewer found no blocking leakage/math/protocol issue. Additional
independent cache audit verified all11new files' family,band,shape,float64 dtype
and recorded receipt hashes in cache-verification.json. Matrices are1900window
train+val transforms only; test windows are not processed by this screening.

First full fold P01 took91.867s after18.1s shared matrix precomputation.
Every one of its50unstretched LR refits agreed with historical probabilities
to1e-7 and mean labels exactly;0optimizer warnings. P01 score differences are
not used to change configurations. Remaining8folds run with3CPUworkers,BLAS1.

Timing note:sum(total_seconds) is accumulated fold wall time,not measured CPU
time. The generated report's timing label should be read with this correction;
final verification will explicitly record wall time and cache-audit hash.

Completed:45evaluations,9exact baseline refits,0optimizer warnings. Independent
rescore passed; all11new matrix/receipt files rechecked against the initial audit.
Best fixed configuration in this screening:balance10,LOSO sessionBA57.5695%
vs56.2801%,delta+1.2894pp; descriptive paired bootstrap95%CI[+0.2105,+2.4657]pp,
5/8subject wins. Late sessionBA78.3733% vs72.3730%,oneparticipant only.
Other LOSO session means:mixed50 56.4965%,stretch50 56.8443%,balance05 57.1884%.
All configurations were retained; no additional fits or tuning after outcomes.
Summed fold wall time1570.0s includes overlapping parallel jobs; it is not
elapsed experiment time or a measurement of CPU consumption. Report label fixed.
