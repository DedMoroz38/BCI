# Five literature-based reference variants: provenance

- User request: run five further literature-based reference matrices in the same comparison format.
- Scope: same causal PROTEUS table-2 protocol; original identity-reference synthetic experiments not changed.
- Primary formulas were verified in literature before execution; definitions, citation locations and URLs frozen in experiment.json/literature_references.py.
- Candidates: harmonic cumulative, power-Euclidean p=.5, symmetrized Gaussian KL/Jeffreys centroid, Log-Cholesky cumulative, inductive Bures-Wasserstein.
- These are published SPD estimators used as controlled EEG references; not replicas of five full EEG systems.
- Shared original log-Euclidean training prior and weight 5; only session-reference construction changes. No test-set tuning.
- 145 new jobs completed in separate local processes, at most four at once; same C, seeds, bands, 50-tree subsets, bootstrap code and folds.
- 43 analytic/control/causality/protocol tests passed. Mathematical prefix results checked against scalar hand calculations and noncommuting library means, BW against independent transport formula.
- Official metrics independently verified all 261 prediction files: 145 new, 87 previous alternatives and 29 original controls; atol=1e-14.
- Original data, code, control results and previous alternative results unchanged by SHA-256 checks.
- New summary: comparison.csv (18 rows); cumulative summary: combined_comparison.csv (27 rows, nine references).
- All 261 fold scores, 120 new and 192 total paired LOSO comparisons, per-window outputs, sources, plots and rerun commands are retained.
- Single seed, eight LOSO people, one later session and two public-test people; ranking is exploratory, with repeated use of public data.
