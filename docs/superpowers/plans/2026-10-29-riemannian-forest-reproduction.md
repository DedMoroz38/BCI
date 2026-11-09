# Riemannian Forest Reproduction Implementation Plan

> Execution: inline, as authorized by the request to reproduce the complete experiment. Preserve upstream model code unchanged.

**Goal:** Run section 5 of `Riemannian_Forest (1).pdf` and compare independently computed results with both paper tables and figure 1.

**Architecture:** Isolate the upstream repository at a recorded commit and install its requirements into `reproduction/.venv`. Use the original data preparation and model/evaluation functions for EEG. If original synthetic code cannot be located, reconstruct the stated protocol separately and explicitly identify all unspecified choices; never label that a bitwise reproduction.

**Tech Stack:** Python 3.12, author's neuralbench 0.4.0 data pipeline, NumPy/SciPy, pyRiemann, scikit-learn, matplotlib.

**Spec:** Paper section 5, pages 14–17; upstream notebooks forest.ipynb and logreg.ipynb.

## Global Constraints

- Commit: 8d577af42619e5911cc5da8743c8a3e9b629014b.
- EEG splits: 1521/379/755 windows; 6/2/2 distinct subjects; (41, 480) per window.
- EEG: C=0.01, prior_weight=5, seed=0 (upstream default); K=50, m=12/20, six upstream bands.
- CV: train+val (8 people), LOSO and last-session fold; test: fit only train, baseline and m=20 forest, one evaluation each.
- Report session_mean; LOSO standard deviation uses pandas sample std, matching upstream.
- No test-set tuning and no replacement of the streaming mean.
- Published neural-network numbers are reference values, not independently reproduced results.

## Review Focus

- Train/validation/test people must be disjoint and have the stated counts.
- Covariance preprocessing must remain the official pipeline, including run ordering.
- Streaming reset must occur at each subject/session boundary.
- Stored prediction scores must agree with re-scoring and streaming/batched feature calculation.
- Missing seeds, package pins and synthetic sample choices must be disclosed.

## Tasks

- [x] Inspect paper and upstream code; record commit and protocol.
- [x] Install isolated environment; freeze packages and source/data hashes.
- [x] Prepare PROTEUS using upstream `python -m bci.data`; validate counts and ordering.
- [x] Add external runner and protocol validation; execute baseline and both forest CV protocols, save per-fold scores/predictions.
- [x] Evaluate the two selected models on official public test split once.
- [x] Locate original synthetic code; if unavailable, run a disclosed reconstruction of figure 1/table 1.
- [x] Verify saved results, compare against paper, and write final reproduction report and rerun commands.
