# PROTEUS: causal transduction over saved held-out priors

No classifier training, no new control runs, no reproduction-tolerance gates.
Private Kaggle CPU notebook; 4 workers; 240-minute wall cap. No submission.

## Inputs and evaluation

Saved Lag4 probabilities and two SWAD.1 CNN seed logits, from the prior 11
outer folds. Two priors: Lag4; and .7 Lag4 + .3 mean softmax(CNN logits / 4).
No final balance is applied before transduction. Warm-up/fallback uses our
existing causal balance1 with prior10, reset for each person/session.

Main metric: pooled OOF mean balanced accuracy per person/session/context,
all 10 LOSO people. Mean-person LOSO and later_last are separate diagnostics.
Scores are computed only after emitting predictions. Public CV has already
been reused repeatedly: this is exploratory research, not sealed confirmation.
No configuration is selected or changed in response to intermediate scores.

## Six research directions

1. Author's causal Gram-PCA -> soft-LDA -> balanced assignment transducer:
   5 bands, k20, warm20, 5 self-labelling iterations, shrinkage .5, beta1.
2. Relax equal prefix class mass: scaling exponent tau=0,.25,.5,.75,1;
   slower onset, 50/80-window warm-up, .5/1/2 beta, half-strength adaptation.
   Intermediate tau uses KL-relaxed column scaling, not a claim to reproduce
   any particular unbalanced-OT EEG method.
3. Ordinary 41-channel vs Lag4 82-dimensional covariances, and equal-weight
   two-view Gram fusion. Each view is trace-normalized on the current prefix.
4. Stored running-anchor features vs fixed training anchor vs common log-cov
   coordinates; prior5 vs prior20. Fixed and log charts have common coordinates;
   running charts reproduce the author's stored-feature convention.
5. Class-balanced reference: maintain three soft log-covariance sums/masses
   using frozen Lag4 pseudo-probabilities, average the three class means with
   equal weights, shrink to source mean with total prior5. Also test sharpened
   probabilities (p squared). This is a new hypothesis motivated by SPDIM's
   label-shift analysis, not an implementation of SPDIM.
6. Ordinary/Lag4 prediction fusion and stability weighting: mean predictions,
   half strength, or shrink toward fallback by exp(-JS/.05) / exp(-JS/.2).
   Agreement is a reliability heuristic, not a correctness guarantee.

13 geometries x 8 shared settings, plus 14 targeted robustness settings:
118 transduction configurations. Four two-view/stability combinations and
one existing fallback = 123 configurations per prior, 246 total, 2706
outer-fold evaluations. No classifier fitting is included in that count.
Settings are fixed before the run. Joint settings vary k10/20/30 and LDA
shrinkage .2/.5/.8 only for the two most relevant geometries.

## Causality and computation

Source log-covariance means exclude the outer evaluation windows. Caches are
pure single-window filter/OAS/log transforms. Pseudo reference uses only the
current/past window's frozen Lag4 probabilities, never target labels. State
resets at every person/session. Gram matrices may be precomputed as a speed
optimization, but every PCA/normalization/assignment sees only its leading
current prefix. Past emitted predictions are never revised.

CPU batching shares PCA across independent hyperparameters/priors; every case
has independent pseudo-label iterations. NumPy/SciPy handle small linear
systems; a GPU would mostly sit idle. No GPU hours are requested.
Completed folds are saved and summarized even if another fold errors or
reaches the wall cap. Rankings require all ten LOSO folds. No partial-fold
score is presented as complete. Figures show ranking and person deltas,
not train/validation loss: this experiment has no training epochs.

## Sources

- DorinDaniil/Brain-Decoding, commit ca5d3d0045120b426f77437d48c374f57d7cce45:
  https://github.com/DorinDaniil/Brain-Decoding/tree/ca5d3d0045120b426f77437d48c374f57d7cce45/track2_bci/transduction
- SPDIM, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc6cd575b8340be6ee70beaed9ca4015-Abstract-Conference.html
- Chizat et al., unbalanced transport scaling: https://arxiv.org/abs/1607.05816
- Competition streaming contract: https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md

TSMNet/SPD normalization with new neural training remains a separate more
expensive research direction; it is not represented by the pseudo-reference
or the post-hoc transducer here. No claim of running TSMNet is made.
