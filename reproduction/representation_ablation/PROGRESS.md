# Completed

- All 87 new evaluations finished: 29 each for delay1_oas, hsic_rbf and envelope_oas.
- Original cumulative log-Euclidean reference unchanged; all source/data/config/previous-result hashes match.
- All 18 exact per-window caches complete; 47,790 strictly positive definite matrices.
- All 116 candidate/control prediction files independently rescored with four official metrics; all match.
- All physical electrode selections, delay-copy grouping, tangent dimensions and window order checked.
- 65 tests passed; final tests in logs/final-tests.log. No runtime or convergence warnings in model logs.
- The original launcher stopped before finishing delay1_oas/forest20 public test. No experiment process remained active; only that missing evaluation was resumed via run.py --protocol test. See logs/delay1_oas_forest20_test_resume.log.
- Results: REPORT.md, comparison.csv, comparison.png; full evidence in verification.json and cache_inventory.json.
- Plot visually inspected; all plan steps complete. No changes to previous experiment artifacts.
