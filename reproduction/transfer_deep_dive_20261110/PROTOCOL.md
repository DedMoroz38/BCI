# Prospective transfer-normalization study — 10 November 2026

Primary aim: official pooled LOSO cell BA on completely held-out people.
Also report equal-person LOSO and per-person deltas; later_last is separate.
Historical complete baseline:63.229380% OOF,71.639456% later_last.
No baseline fitting, historical parity gates or remote preflight tests.

Literature was reviewed before writing the new model: literature/REVIEW_RU.md.
These are new causal adaptations, not claims to reproduce contest scores.

## Fourteen fixed model configurations

| ID | Change | Hypothesis |
|---|---|---|
| filter_p10 |16 filter moments pooled over62 temporal positions | Less noisy target statistics |
| filter_p40 | Same, source prior40 | Stronger shrinkage for new people |
| filter_person_prior | Equal-person source mean/second moment | Avoid domination by multi-session people |
| filter_half |50% residual mixture with unnormalized hidden state | Preserve discriminative power |
| filter_clip | Limit std gain to[.5,2] | Restrict erroneous amplification |
| filter_ttn16 |16 sigmoid interpolation coefficients, source-trained,L2.01 | Learn a small amount of adaptation |
| early_and_filter | Align pre-square activations and pooled log-power | Correct shift before nonlinearities |
| multiscale12_filter | Three4-filter branches,kernels13/25/51 | Complementary temporal scales at small capacity |
| point_p40 | Existing992-coordinate policy,prior40 | Isolate prior strength |
| point_half | Existing point policy,50% residual | Isolate adaptation strength |
| point_person_prior | Point policy,equal-person source moments | Isolate source weighting |
| filter_memory50 | Exponential moment decay.98 | Adapt within-session changes |
| filter_shift12 | Filter alignment +up to100ms reflection-padded shifts | Combine two supported mechanisms |
| filter_ls005 | Filter alignment +label smoothing.05 | Limit overconfidence |

Unchanged:41channels,480samples,causal input EA on1–58Hz with prior5;
Shallow16 except the explicitly named12-filter multiscale variant;
dropout.5,AdamW.001,decay.01,cosine Tmax100,max30epochs,patience6;
class-weighted CE,inner people-held-out cellNLL and SWAD ratio.1.
Source-only BN recalibration/frozen BN during prefix passes follows the
previous positive experiment. Candidate model construction preserves RNG state.

Per-filter mean and second moment are first computed within each fully
available window, then accumulated across current/past windows. Pooling time
for statistics does not remove the62 temporal features used by the head.
Mixed variance includes the between-mean term. Target labels never update
statistics. The16 learned coefficients are frozen at inference.
Equal-person policy changes source hidden moments only; it does not pretend
to change every source weighting or the loss to a person-balanced objective.

All10LOSO +later_last,2seeds:308 outer fits; each has inner selection/refit.
For every family both seeds are required for its primary mean2 result.
No seed is selected based on outer scores. Outer participant data never enter
the feature extractor,source prior,learned alpha or inner epoch selection.

## Fixed fusion hypotheses

1. Each new two-seed expert alone.
2. Add.1/.2/.3 to the current63.2294% full baseline,using the expert's causal
   balance1 probabilities. These are fixed before the run.
3. Replace the previously added normalized CNN branch: .8*old62.7234% baseline
   +.2*new normalized expert. This is the most matched architecture comparison.
4. Two bounded,label-free gates: seed JS agreement and seed agreement with
   prefix-size shrinkage. Weight stays within[.1,.3]. Agreement is a heuristic,
   not a guarantee of correctness. No fitted stacking on reused outer OOF.
5. Previous fixed70%Lag4+30%newCNN(T4),with balance and the unchanged
   soft50/half-strength causal transducer,reported as distinct complete models.

Current/future-window distinction is enforced by cumulative computations;
batch evaluation is only a vectorization of the causal prefix equations.
Session state resets independently. Subject/session/interface/run metadata
are used only for data splits,resets and scoring; model input is EEG alone.

## Output and execution

One private Kaggle T4 notebook,two GPU workers. Application cap600min;
Kaggle timeout41400s leaves room for summaries/packaging. Stops as soon as
all jobs finish; the cap is not an estimate of expected runtime.
No local or remote background poller after launch. No competition submission.
Completed predictions and failures saved atomically per job. Reports require
all10LOSO folds and show incomplete jobs explicitly. Source/checkpoint/curves
and held-out logits/hidden states are retained.

The public CV has been reused repeatedly. Best post-hoc score is exploratory;
these data do not establish a hidden-cohort gain or significance. Comparisons
with different people/metrics in the review are never pooled with PROTEUS BA.
