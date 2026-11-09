# Track 02 — BCI Decoding: data, proposed model, and Riemannian Forest review

## Data (PROTEUS / NEMAR nm000290)

- **Task**: classify cued mental command from a 4 s EEG window. 3 classes: motor imagery, mental arithmetic, word generation.
- **Subjects/sessions**: 10 subjects × 14 sessions total. Public split: 6 train, 2 val, 2 test subjects → 1521 / 379 / 755 windows.
- **Channels**: h = 41 EEG, recorded at 500 Hz, resampled to 120 Hz.
- **Preprocessing** (fixed by NeuralBench): notch at 50/60 Hz, band-pass 0.1–75 Hz, window starts 1 s after cue, values in µV, no scaling/clamping.
- **Context**: each run tagged by interface — Graz or BrainHero (PROTEUS-specific).
- **Eval protocol**: causal/streaming. `predict(X)` gets one window at a time (B=1), in recording order, per session; each session starts from a fresh `deepcopy` of the model with `reset_state()` called. No cross-session adaptation allowed.
- **Metric**: balanced accuracy per (subject, session, context) cell, averaged over cells. Chance ≈ 33.3%.
- Published baselines (cross-subject, seeds 33/34/35): Constant 33.3%, EEGNet 32.6%, CTNet 35.0%, REVE fine-tuned 36.0%.

## Proposed model: Riemannian Forest (tangent-space ensemble)

Recommend building the submission around the **Riemannian Forest** from Dorin's paper (see review below) rather than EEGNet/CTNet/REVE — it was built and evaluated on this exact PROTEUS split/preprocessing and clears the published neural baselines by a wide margin (bal. acc. 0.53 vs. 0.33–0.36, see Results below). Code: `github.com/DorinDaniil/Brain-Decoding`; relies on `pyriemann` (already in `.venv`).

**Why it fits the contract well:**
- Its reference point per session is a **running log-Euclidean mean** updated window-by-window (Eq. 2 in the paper) — this *is* `reset_state()` + causal streaming, no adaptation needed.
- No deep net training; fast to fit, fast to run per-window (covariance + matrix log + logistic regression).

**Architecture:**
1. Split 0.1–75 Hz signal into B=6 sub-bands (4–40, 4–8, 8–13, 13–20, 20–30, 30–40 Hz).
2. Each of K=50 "trees" = (random band, random m-of-41 channel subset, bootstrap subject sample). Compute channel-subset covariance, whiten/recenter against the session's running mean, project to tangent space (`vec_√2 log(...)`), fit L2-logistic regression on tangent features.
3. Forest output = average class probabilities over trees, `argmax`.
4. `reset_state()`: reset the per-session running mean to the training-set prior `M_0`.

**Tuning**: paper's rule (17) picks `K/B` from desired channel-pair coverage; use **out-of-bag** error across subjects (Prop. 2) to pick `m` and `K` — no extra CV split needed. Paper's best config: `m=20`, giving 0.529 bal. acc. on the official PROTEUS test split.

**Possible extension**: stack/average forest probabilities with REVE features for a further bump, since the two capture different structure (tangent-space geometry vs. learned embeddings).

## Thoughts on the "Riemannian Forest" PDF (Dorin, MIPT)

**What it does**: gives the first theoretical treatment of an ensemble ("forest") of tangent-space classifiers, each trained on a random channel subset + bootstrapped subjects — the structure used by several winning BCI-competition solutions, previously without theory.

**Key results, and why they matter for us:**
- **Prop. 1**: restricting to a channel subset is a norm-non-increasing "contraction" of the full whitened covariance, so existing tangent-space error bounds (geodesic distance ↔ Euclidean feature distance) carry over per-tree with `m` channels instead of `h` — justifies small-subset trees without re-deriving guarantees.
- **Thm 1**: finite-forest excess risk = (mean per-tree variance)/K, with exponential concentration vs. the infinite-forest decision — tells you diminishing returns from more trees and gives a concrete `K` sizing rule.
- **Thm 2**: closed form for expected fraction of tangent coordinates shared by two trees, `η_h(m)` — directly gives tree-to-tree correlation and the probability a given channel-pair is "covered" by at least one tree (used in rule 17).
- **Thm 3**: in the strong-regularization limit, the forest ≈ one logistic regression over *all* channels where cross-channel (connectivity) terms are penalized `(h-1)/(m-1)×` harder than per-channel power terms. This is a clean result: **random channel subsampling is literally an implicit structured regularizer that favors power features over connectivity features.** Useful diagnostic — if your signal lives mostly in connectivity rather than channel power, small `m` will hurt (confirmed in their synthetic experiment).
- **Prop. 2**: OOB error over held-out *subjects* (not just held-out windows) is a valid, unbiased estimate of new-subject error — avoids a separate cross-validation loop, directly relevant since Track 02 is cross-subject/cross-session.
- **Empirics**: synthetic data confirm the correlation/variance formulas tightly; on PROTEUS the forest beats EEGNet/CTNet/REVE by a large margin and beats a single tangent classifier by 0.04–0.12 bal. acc.

**Caveats / where to be careful:**
- Test set is tiny (2 subjects), and the reported std across subjects (0.11–0.15) is comparable to the effect sizes being claimed — treat the win over baselines as promising, not conclusive.
- Theorem 3 is a first-order-in-`1/λ` approximation; the actual PROTEUS run uses moderate regularization (`C_sk=0.01`), so the "equivalent structured regression" interpretation is only approximate there, not exact — the author flags this explicitly.
- The real implementation deviates from the clean theoretical setup in 4 ways (own log-Euclidean running reference points instead of batch Riemannian mean, feature standardization, class-balanced LR weights, covariance shrinkage via OAS) — so Props 1–3's exact guarantees don't literally apply to the shipped code, only approximately.
- Rule (17) for picking `m`/`K` is derived, not yet validated across multiple datasets (paper says so directly) — worth sanity-checking against our own OOB curve before trusting it blindly.

**Bottom line**: credible theory, matches synthetic simulation well, and the empirical PROTEUS result is the strongest public number I've seen for this track by a wide margin — good default model to start from and iterate on.
