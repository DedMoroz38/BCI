# Lag4: synthesis of two supplied repositories

Research-only review,2026-11-08. No new training or modification of the active
Kaggle experiment. Sources downloaded for inspection,not executed.

## Fixed source snapshots

- DorinDaniil/Brain-Decoding:934f423a73e1ed50d653345d27256ecfe50ab009,
  local ensemble_author_20261108. Read bci/ensemble.py and bci/stream.py.
- DedMoroz38/BCI:695c29d6117a363f62653f136017319170aead2a,
  local BCI_DedMoroz38. Read track2_bci/bci/{riemann.py,models/forest.py,
  models/tta.py,models/gated.py,make_submission.py},save/augment_check/check.py,
  save/next_steps/{run.py,tta.py,summary.py},and their CSV results.
- DedMoroz submissions/forest_lag100/{model.npz,submission.py} each have exactly
  the same SHA256 as the previously supplied local forest_lag100 files.
  Training code now located; user previously deferred its numerical inclusion,
  so do not interpret this review as authority to launch another experiment.

## Evidence before recommendations

Our baseline:50trees,20physical channels,6bands,fixed lag4 in every view,OAS,
prior5,cumulative log-Euclidean recentering,StandardScaler,balanced LR C.01.

Dorin adds extended bands,global+person-specific forests,session-fingerprint
routing,causally aligned ShallowConvNet and prediction-frequency correction.
Its global forest has no lag augmentation; our active comparison therefore
does not yet measure adding these modules to the stronger Lag4 backbone.

DedMoroz forest banks contain six unaugmented bands plus augmented narrow
bands (upper/lower cutoff ratio<=2). Augment lag uses a band-centre quarter
period,not fixed4:5,3,2,1,1samples for the five narrow bands. Current supplied
weights are100experts:60plain,40lag. These are different representations,
not merely a100-tree version of our Lag4. Delay copies retain physical channel
groups. Lagged dependence does not establish physiological causation.

Repository results are external logs,not independently reproduced measurements.
Use matched(seed,fold) rows; never compare unpaired means of incomplete sweeps.
- lag100 has16 complete LOSO runs:8participants x seeds0,1; cell BA55.1710%.
- lag3_100 has28 rows with an incomplete fourth seed. On16 common rows,
  cell BA difference vs lag100 is only+0.0583percentage points.
- TTA has90rows,18runs/variant including only2folds ofseed2. Restricting to
  the two complete seeds gives16paired runs/variant: no adaptation55.1710%;
  q.3/lam.1=48.0712%;q.3/lam1=49.5798%;q.6/lam.1=47.5996%;q.6/lam1=47.7469%.
  Thus these pseudo-label recipes lose5.59–7.57percentage points on average.
- stretch and randlag configurations exist in code but have no rows in the
  committed results.csv. Presence of implementation is not positive evidence.
- Final make_submission.py concatenates train,val,test by default. Packaged
  pretrained weights require provenance checking before any held-out scoring;
  valid LOSO must refit on the fold's training participants.

## Ranked hypotheses

1. **Hierarchical ensemble of views.** Keep Lag4 as anchor; add original OAS
   and band-adaptive delay as separate families. Uniform family sampling then
   band/channel sampling avoids arbitrary weight changes when one family has
   more lag values. Calibrate/choose at most a few family weights from genuinely
   person-held-out OOF predictions; same outer people never choose their own
   weights. Compare100all-Lag4 vs100mixed experts to control ensemble size.
   Expected mechanism: complementarity of power/instantaneous and lagged
   structure,not an assertion of gain from more trees. Reject if errors and
   OOF mixtures provide no benefit over Lag4 at matched tree/latency budgets.

2. **Lag4 + aligned ShallowConvNet.** Transfer Dorin's heterogeneous ensemble
   onto our Lag4 global backbone,not the weaker original forest. First use
   an arithmetic probability mixture with one OOF-selected weight. Calibrate
   neural probabilities using training-only person OOF. Geometric pooling
   in upstream lets a very small probability from one member veto a class;
   this may be fragile with an overconfident network on unseen people.
   Diversity is determined by errors/mixture BA,not architectural names.
   CNN selection must use people-held-out validation. Reject if no consistent
   gain across participants,seeds and late sessions despite improved train loss.

3. **Conservative unsupervised dispersion adaptation.** Lag4 already adjusts
   a covariance reference location. DedMoroz SessionStretch divides standardized
   tangent coordinates by a cumulative RMS with an initial unit-RMS prior.
   This is per-coordinate RMS normalization,not full Riemannian Procrustes
   alignment and not equivalent to centering a variance. Port causally and
   identically to train/eval,shrink strongly or share statistics across blocks
   to avoid estimating820 scales from early few windows. Preserve useful power
   information; downweight artifact-driven updates. Reject if early-window or
   low-performing-subject degradation cancels any late-session gain.

4. **Global fallback with selective specialization.** Dorin router matches
   causal session means to training session templates; DedMoroz gated.py
   mixes learned subject embeddings and FiLM-modulates a small classifier,
   hiding the own-person centre probabilistically during training.
   Prefer a constrained global-Lag4 + specialist mixture initially. For known
   participants use both absolute distance/OOD evidence and nearest/second
   ratio; a good ratio alone can occur far from all training domains.
   Calibrate on leave-person-out and leave-session-out examples. New people
   should retain global fallback unless specialist transfer is validated.
   Only one eligible dev participant has later sessions,so personalization
   claims remain weak. Reject if unknown people are frequently routed as known.

5. **Small guarded corrections,not unqualified self-training.** Dorin balance
   divides class probabilities by a smoothed running predicted marginal.
   This changes decisions according to past output frequencies; it is not
   averaging adjacent labels. Useful only when its marginal assumption holds;
   model outputs need not be calibrated estimates of true class frequencies.
   Shrink towards alpha0,choose training-only OOF,reset every session.
   DedMoroz TTA is not a first-priority port: top-q selection within each
   predicted class does not guarantee correct pseudo-labels or equal class
   counts,and missing predicted classes are not recovered by that rule.
   If revisited,use a frozen teacher,agreement+absolute confidence screening,
   update caps and an anchored residual head; these safeguards are hypotheses.

## Deferred candidates and why

- More delay copies:little matched evidence for benefit and higher SPD dimension.
-100instead of50trees:valid variance control,not an independent new mechanism.
- randlag1–6:useful negative control for the quarter-period hypothesis; its
  code balances augmentation-family draw probabilities,which should be preserved.
- Hilbert analytic quadrature:already explored in our earlier representation
  work; this repository does not turn it into a new untouched hypothesis.
- Rich learned subject gate:greater overfitting risk with7training domains/fold.
- Plain pseudo-label LR adaptation:negative committed empirical evidence.
- Add all modules at once:cannot identify source of gain or failure.

## Small comparison sequence after current run finishes

1. Frozen Lag4 reference vs matched-size mixed-view forest,one change at a time.
2. Add CNN to the best fixed global forest,OOF mixture and calibration only.
3. Separately test dispersion normalization and conservative routing; later
   combine only components that improve matched held-out-person predictions.
4. Full test is previously seen; freeze choices using nested person splits,
   report per-person deltas,cell/session BA and latency. Repeat seeds are not
   new independent participants. No claim of an unobserved sealed-score gain.
