# Timeline

2026-11-08: user approved LS,augmentation andweightaveraging experiments;
explicitly excluded BA epoch selection and pretrainedmodels. Registered36
configs/792fits before newtrainingresults. UseKaggle background only;noagent
continuity/polling because user explicitly requested to report completion.

User clarified25hours of compute remain. Raised per-run wall cap to240min
before implementation/results;finish immediately once the fixed grid is done.

Implementation:addedBN-only control before results (37configs,814fits).
Baseline fit on syntheticdata matches frozenwave1 parameter-for-parameter.
Signal transforms use isolated checkpointed RNG,no evaluation transforms.
LS/augmentation/EMA/SWAD optimizer+RNGresume tested RED→GREEN. Scalar SWAD
selector compared against actual MIT authoraverager,including repeatedepoch
weights;outerrefit preserves that multiset. AllGPUmodels will replay onCPU
before saving completed predictions. Graphs include raw and averaged valNLL.

Fresh independent review found noCritical/Important findings. Parent upgraded
resume artifact checks and mandatoryCPUreplay to required checks and added
RED→GREEN tests. Minor deferred: confidence is derivable from saved logits;
summary currently reports NLL/Brier rather than a dedicated confidence column.
Local report regression reuses realwave1 control artifacts and exactly recovers
the60.23% mean2 controlmixture;it is not a new empirical training result.

Launch:final full local suite39passed,source bundle101files verified,private
T4settings reviewed. Kaggle skill wrapper reported authenticationfailure while
direct CLI quota succeeded(25.39h remaining) and direct kernels push succeeded.
Version1:https://www.kaggle.com/code/den1shh/proteus-regularization-wave2.
Do not ask for login based on the wrapper alone in future. Successful push is
confirmed;actual CUDA preflight/training not inspected. No polling,watcher or
agent continuity job created;user will report completion.

2026-11-09: Version1 ERROR at the historical-control probability gate before
any regularizer ran. Downloaded72 artifacts/245inventory files;7 controlseed0
selection/refit pairs trained,zero completed prediction records. Data hashes,
packageversions andT4model matchwave1;chosen epochs match on all7. Weight
drift concentrates in temporal.bias/BNrunningmean;convolution kernels differ
only~2e-7. Exact source of CUDA training drift not isolated;do not claim a
proven backend cause. Cross-run identity was an inappropriate blocking gate.
User explicitly rejected separate controls. Removed clean and BN-only control
jobs:35variants,770fits. Cross-backend probability/label checks now diagnostic;
checkpointtensor identity/data integrity remain checked. One failed model is
recorded and remaining jobs proceed;partial configs excluded from rankings.

2026-11-09: final suite40passed;Version2 successfully pushed directly via CLI.
Grid35variants×2seeds×11folds=770fits,zero separate controls. Blocking historical
and cross-device numeric gates removed;individual model errors are logged and
remaining jobs continue. Actual Version2runtime not inspected,no monitoring.

2026-11-09: user requested results. Version2 COMPLETE:all770fits,35configs,
no failures or budget-pending jobs;4946.6s trainingwall. Downloaded2890analysis
artifacts (predictions/logits/metadata allfits,176learninghistories,22SWADmodels).
AllBA metrics rescored independently and match Kaggle;data/code/reference and
prediction/logit hashes verified. CPU/GPU diagnostic records show0label
differences across770fits. No tolerance gate blocked training.

Best two-seed fixed mixture:SWADratio.1=60.6139% vs previous60.2279%,gain
.38595pp;6/10personwins,4losses;late67.7561%vs66.6991%. StandaloneCNN54.1994%
vs52.5504%,still belowLag457.4424%. EMA.99mix60.4164%;LS.05+noise.03+drop3
60.4749%;shift12mix60.4579%. LS.2NLL1.014vsold1.089,but mixture60.2329%
unchanged. Train/valgap persists. PublicCVselection among35methods exploratory.
SavedRESULTS.md,allmetrics,curves andper-person deltas. No newexperiments.

2026-11-09: user asked to inspect updated authorrepo and localprivateHTML
literature review after results. Cloned currentca5d3d0 separately,read standalone
Transducer and causalStreamModel,checked primarySPDIM/TSMNet/RPA/ACM papers
and official streamingFAQ. Audited public class counts only (no newtraining):
14sessions,13notexactlybalanced,P08first20=4/10/6. Ranked next directions:
existingLag4+SWADpriors→causaltransduction;softprefixmarginals;Lag4/common-chart
transductiongeometry;balancedpseudo-labelreference;laterTSMNet. Documented
unverified oracle/gain/novelty claims and benchmarkaggregation mismatch.
PrivateHTML itself was not modified,copied into notebooks or published.
