# Custom validation of current leading algorithms —10November2026

Implements the user's accepted proposals1/2/3. Canonical policy:
../VALIDATION_PROTOCOL.md. Split seed20261110,chronology byrun/onset.

Freeze model recipes before new8/2 scores. Compare:
- Lag4 LR50,raw and causal balance1.
- Lag4 nuclear logistic tau.01/.05,raw and balance1,shared matched geometry.
- mSSA rank8/mix25%→Lag4 LR50,raw and balance1.
- Full pre-normalization ensemble(the old62.72% recipe).
- Full point mean/std normalization ensemble(the old63.23% recipe).
- Full early+late normalization30%branch(the old64.58% recipe).
- The same with seed-agreement branch weight(the old64.51% recipe).
- Native two-seed neural components retained as diagnostics.

The historical percentages name recipes,not predictions on the new folds.
No weak HOPLS/Tucker/iTransformer sweep. No new CNN architecture or tuning.
CPU forests are trained locally;60GPU outer fits retrain3existing neural
families ×2seeds ×10pairs,each with inner stopping and8person refit. Original
training policies are retained for each family. Only source statistics are
fitted; input alignment remains causal at target. Curves/checkpoints saved.

Full model recipes:
q0=causal_TTA(.7*Lag4+.3*mean_seed_softmax(SWAD_logits/4)),soft50/half,
Lag4running geometry,k20,shrink.5,warm20,beta1,5iterations.
q1=.8*q0+.2*balance(mean_seed_softmax(point_meanstd_logits)).
q2=.7*q1+.3*balance(mean_seed_softmax(early_filter_logits)).
qgate=(1-a)*q1+a*balance(mean_seed_softmax(early_filter_logits)),
a=.1+.2*exp(-JS(seed0,seed1)/.05). No label fit for gating.

Every outer pair's source priors,PCA/TTA source means and class heads must
be rebuilt from the8source people. Reusing9person LOSO weights could leak
the second held-out person and is not allowed. Pure perwindow covariance
caches may be reused. Do not select seeds or branches on outer labels.

LOSO rescoring is read-only on saved forecasts;8/2 is new model training.
First-session BA uses mean of per-context3class balanced accuracies; every
person equal. Stream-bin BA is3class macro recall over each bin regardless
of context;class and context counts accompany it. It is a different slice,
not a guaranteed monotonic learning curve. Missing-class bins areNaN,not
silently interpreted as failure. State was generated on the whole natural
prefix and is never restarted for a metric bin.

Two outer appearances per person are averaged as metrics,not probabilities.
Only models with all10folds/10people/2appearances enter the pair ranking.
All-session official-style BA,worst3first-session BA,NLL and source-repeat
spread are additional columns;no arbitrary weighted synthetic score.

No baseline equality/commit/hash/reproduction checks,external submissions
or background Kaggle watcher. CPU results feed the private GPU notebook;
the old9person LOSO reference forecasts are never used in its8/2 fusion.
