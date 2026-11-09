# Reference-update experiment provenance

- User authorized repeating the already reproduced experiment with several reference-matrix alternatives.
- Scope: the causal PROTEUS experiment, table 2. The synthetic experiments use fixed identity references and were not changed.
- Three candidate update rules and one reused original control; identical original log-Euclidean training prior, weight 5.
- Candidate formulas, decay=.95 and source/data/environment hashes were frozen in experiment.json before any candidate scores.
- Source model code was not rewritten or modified. A subclass replaces new_session() through a scoped model factory, in separate processes.
- 87 new evaluations completed successfully; all 29 original control evaluations retained byte-for-byte.
- Official competition metrics independently verified all 116 prediction files, atol=1e-14.
- Identical subject folds, window order, frequency/channel selections and source code verified against original control.
- 21 analytic, causal, control-parity and previous protocol tests passed in the final run.
- Recursion is an inductive AIRM estimator, not exact batch Frechet mean.
- EMA uses normalized exponentially decaying prior/window weights; warm-up gain is not fixed .05, limiting gain is .05.
- No test-set parameter tuning or seed changes. All comparisons are exploratory at one seed.
- Report, full fold scores, paired differences, plot, code and per-window predictions are retained; checksums.json records their SHA-256.
