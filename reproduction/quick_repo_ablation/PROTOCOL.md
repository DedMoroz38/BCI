# Quick local repository ablation — 2026-11-08

Authorized: implement and run fast local experiments based on the preceding
two-repository review. Fixed scope: no neural training,no new Kaggle runs,
no modification of old sources/data/results or active Kaggle experiment.

Reuse verified submitted Lag4 forest50 tree probabilities from
classifier_ablation/results/baseline,with exact recording metadata matching
temporal_spd_research/results/lag4_oas/forest20 references. Fixed outer protocol:
8 LOSO on train+val,one later-session fold. No public-test evaluation in this
quick screening; no hyperparameter selection or claim of untouched validation.

Configurations,all50 experts except deterministic postprocessing:
- baseline: original50 Lag4 experts,20physical channels,6bands,prior5,C.01,seed0.
- mixed50: keep original experts0..16; replace17..33 by unaugmented OAS and
  34..49 by band-dependent quarter-period-lag OAS. Broad4–40Hz uses fixedLag4
  in the latter family,not a purported broadband90degree phase shift.
  Preserve each expert's exact original band,physical channels and subject
  bootstrap counts. Uniform-within-existing draws,17/17/16family allocation.
- stretch50: same50 Lag4 experts,standardize unmodified tangent features first,
  then cumulative per-session per-coordinate RMS normalization with unitRMS
  prior weight10,at both training and inference; refit same balanced LR.
- balance05 and balance10: unchanged baseline probabilities divided by smoothed
  cumulative predicted class marginal to powers.5 and1,uniform prior10windows,
  update includes current probability (as Dorin StreamModel),reset per session.

Reference sources unchanged: Dorin934f423 and DedMoroz695c29d. Matrix estimation
is OAS of all41or82 rows before physical-channel subsetting,matching the old
pipeline's shrinkage convention. Lag4 matrix cache SHA-verified. New ordinary
and adaptive-lag covariances are pure window-local float64 transforms,cached
in chunks,with data/code/parameter hashes and verified completion receipts.

All50 baseline draws must equal historical references. New preprocessing,
reference priors,scalers and classifiers fit outer training only. Bootstrap
weights and class balancing follow original all-training-geometry convention.
No target labels fit any transformation. Same chronological/session resets.
Save raw probabilities,metadata,train/eval timing and warnings per fold.

Report equal-weight mean sessionBA over8people,leaderboard cellBA,late sessionBA,
logloss/Brier,paired per-person differences and descriptive bootstrapCI.
Repeated historical validation implies exploratory findings,not confirmatory
evidence. No individual-subject score is treated as an independent seed.
Any numerical refit of unstretched Lag4 must exactly reproduce historical
predictions (and match probabilities to1e-7); otherwise stop the experiment.

Runtime plan:3 CPU processes,one BLAS thread each; first measure one real fold,
then finish remaining folds using cached pure matrix transforms. No extra
architectures,grids,seeds or expensive tests beyond this declared scope.
