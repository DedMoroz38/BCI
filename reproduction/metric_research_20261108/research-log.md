# Research timeline

2026-11-08: user authorized new autonomous research,local fast experiments and
private background Kaggle computation,aiming at maximum metric. Initialized
fixed wave1 protocol before new numerical outcomes. No agent scheduling tool
available; use durable batch execution and recorded state rather than claiming
an agent heartbeat. Account quota is6hours total this week,~95min remaining,
not the earlier assumed30hours. Prioritize75min first wave and checkpointing.

2026-11-08: completed local fusion grid (170 scores). Froze 153 CV file
hashes and selected arith_w0.3_T4_a1 before old-public-test evaluation.
Legacy mean-person LOSO cell BA 64.43%, baseline 56.30%; old public-test
cell BA 56.12%, baseline 55.72%. Repeatedly observed test remains exploratory.
Independent rescore and source-ensemble replay passed for every saved result.

2026-11-08: completed all10 baseline controls (11 folds, two predictions per
fold). Pooled OOF cell BA: Lag4 57.44%, balance1 58.53%. Equal-person means
are 59.54% and 60.41%; these use a different cohort/aggregation from legacy8.

2026-11-08: pushed private Kaggle proteus-metric-wave1 version1 successfully.
Three CNN recipes, inner participant-held-out epoch selection, causal alignment,
checkpoint/replay guards and a 75-minute training budget. Runtime uninspected.
User explicitly requested no polling and to stop once local work and launch
were done. No watcher, cron or monitoring process created for this wave.

2026-11-08: user requested results check. Version1 ERROR: after 10 passing
tests, the fresh-process CUDA smoke imported bci before the path setup in
neural_training/common. Failure at ~72s, before run_gpu.py; full 92-file
inventory contains zero completed predictions. Saved raw log and inventory
under ignored downloads/. Added notebook-generated smoke subprocess regression:
RED reproduces ModuleNotFoundError; reordered imports; GREEN 11 tests pass.
Rechecked all local numerical results. Repair push failed authentication;
credentials --verify also fails. Asked user to repeat OAuth login. No new
version launch is claimed and no background monitor is running.

2026-11-08: user completed forced OAuth login. Generic skill credential
verification still failed, while the directly selected runner CLI read quota
successfully. Retried notebook-push with explicit KAGGLE_CLI_BIN pointing to
reproduction/kaggle_runner/.venv/bin/kaggle: Version2 successfully pushed.
No runtime completion claim and no polling or watcher. Keep the 75-minute
training budget and unchanged experiment configuration. Current direct CLI
quota reports 30h total, 25.56h remaining; the earlier 6h quota assumption is
not consistent with this read and should not be relied on for future planning.

2026-11-08: user requested another check. Kaggle Version2 COMPLETE; all55
fits finished,none budget-pending. Downloaded557 metric artifacts with
version-pinned inventory/hash receipt,excluding redundant optimizer/final
training states. Verified all55 predictions,source/data/artifact hashes,
metadata,splits,selection/refit epochs; reconstructed all55 saved models on
local CPU with exact labels (max delta1.252e-6). Float32 softmax/NLL rescore
differs at most1.192e-7 across Linux/ARM; BA matches at1e-12.

2026-11-08: applied only the earlier fixed70/30,T4 recipe to each trained
configuration and probability-averaged both compact seeds (no best-seed
choice). Class-loss compactmean2 mixture gives60.23% pooledOOFcell vs57.44%
Lag4 and58.53% balancedLag4;9/10personwins,later66.70%vs64.64%. Standalone
CNNs lose. Learned NLL/BA mismatch:49/50LOSO BA peaks later than chosenNLL
epoch. Saved complete comparison,diagnostics,curves and GPU_REPORT.md.
No new model training or recurring monitoring started.
