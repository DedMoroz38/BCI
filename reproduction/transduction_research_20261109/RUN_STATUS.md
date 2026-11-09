# Kaggle launch

2026-11-09: Kaggle CLI confirmed **Kernel version 1 successfully pushed**.

https://www.kaggle.com/code/den1shh/proteus-causal-transduction-research

Private CPU run, internet off, public input dataset attached. Application wall
cap 240 minutes; Kaggle process timeout 15000 seconds including startup.
123 settings per prior, two saved priors, 11 outer folds: 2706 evaluations.
No new classifier/control training, no parity gates, no competition submission.
No status poller or background monitor has been started. Actual run completion
and metrics have not been checked.

## Results checked after the user's completion notice

Kaggle reports ERROR due to an intermediate report race: summarize() saw
another worker's NPZ before its final JSON score receipt. All 11 workers
finished and saved all 2706 evaluations. No fold failed. Downloaded Version 1
outputs to downloads/v1. Rebuilt the report locally without rerunning any
experiment. Scored all saved probabilities: maximum difference from original
score receipts 1.11e-16. Runner now ignores in-progress folds until the JSON
receipt exists. No new Kaggle version or run was started.

Best OOF candidate: lag_running__soft50_half, prior lag4_swad.
OOF cell BA .627233744814203 vs previous .6061385200168313 (+2.109522 pp).
Later sessions .711102216424204 vs .677560869468587 (+3.354135 pp).
8 of 10 people improve, 2 decline. Notebook events span about 11 minutes;
no GPU accelerator was requested. Public-CV selection remains exploratory.

Artifacts: transduction-results/comparison.csv, fold_scores.csv, REPORT.md,
top20.png, best_person_deltas.png, per-fold predictions, failures.json,
completion.json, runtime.json, execution.log.
