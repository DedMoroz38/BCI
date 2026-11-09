# Literature-driven replacement of R(X)

Completed:203 new evaluations,58 unchanged controls,261 independently verified
prediction files,101 passing tests. Only R(X) changes; original source, cumulative
log-Euclidean reference, scaler, classifier, C, seed, physical channel draws,
frequency bands, folds and data remain fixed.

Read `FINAL_REPORT.md` for conclusions. The pre-test LOSO winner (mixed temporal
halves) failed to beat previous delay1 on public test. Fixed lag4 improved both
LOSO and test; that practical preference is retrospective and needs a fresh
holdout. The raw block-diagonal and quadrature candidates contain numerical null
features amplified by the original scaler; their scores remain visible but are
excluded from the numerical-eligibility ranking.

## Full execution order

From `/Users/den1shh/Documents/diploma`, use the existing locked environment and
prepared data. Completed caches/jobs resume with immutable source/data/config
hashes. Do not run duplicate launchers for the same groups.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/Users/den1shh/Documents/diploma/reproduction/.cache/matplotlib
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/prepare_cache.py --workers 4
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/run.py --initialize
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/launch.py --phase cv --workers 4
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/followup_mixed/run.py --initialize
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/followup_mixed/launch.py --phase cv --workers 2
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/followup_mixed/select_model.py
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/select_model.py
```

On a fresh run, commit both `selection.json` and
`followup_mixed/selection.json` in this isolated research repository BEFORE any
new public-test scores. Existing selections are validated on resume, never
replaced from test scores. Commit55feda5 records the current pre-test selections.
The raw first-stage selection and the final numerical-eligibility selection are
separate records. `continue_with_audit.py` automates this boundary; the initial
`continue_after_cv.py` was superseded and must not run concurrently.

```sh
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/launch.py --phase test --workers 3
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/followup_mixed/launch.py --phase test --workers 1
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/analyze.py
reproduction/.venv/bin/python -u reproduction/temporal_spd_research/followup_mixed/analyze_final.py
```

## Tests

```sh
reproduction/.venv/bin/python -m pytest -q reproduction/temporal_spd_research/followup_mixed/test_mixed_features.py reproduction/temporal_spd_research/test_temporal_features.py reproduction/temporal_spd_research/test_temporal_cache.py reproduction/representation_ablation/test_matrix_cache.py reproduction/representation_ablation/test_representations.py reproduction/literature_ablation/test_literature_references.py reproduction/reference_ablation/test_references.py reproduction/scripts
```

Tests include exact continuous quadrature, depth-2 signatures, SPD, same physical
channel subsets, session reset, streaming/batch/prefix behavior, cache/live model
agreement, interrupted-cache recovery, and orthogonal mixed encoding.

`mechanism_probe.py` is an exploratory label-free probe. It filters the same four
windows first and then bypasses only the second filter inside the isolated probe,
so reversal/shuffling checks matrix construction rather than filtering effects.
It found that depth-2 signature channel Gram loses reversal orientation. Full
signature-kernel order/universality does not transfer to this packing.

## Final artifacts

- `FINAL_REPORT.md`, `final_comparison.png`: conclusions and interpretation.
- `final_comparison.csv`:27 model/representation summaries.
- `final_fold_scores.csv`:261 individual scores; `final_verification.json`: combined checks.
- `REPORT.md`, `comparison.csv`: raw six-function stage, with numerical warnings.
- `PROTOCOL.md`, `followup_mixed/PROTOCOL.md`: protocols committed before the respective accuracy experiments.
- `selection.json`, `followup_mixed/selection.json`: pre-test choices and CV hashes.
- `temporal_features.py`, `followup_mixed/mixed_features.py`: matrix functions and adapters.
- `cache_inventory.json`:36 matrix sets,95,580 SPD windows. H7 mixes the same immutable source matrices on lookup; no additional fitted statistics.
- `literature/`:9 verified primary-source notes, metadata and synthesis.
- `block_numerics_probe.json`, `quadrature_numerics_probe.json`, `mixed_numerics_probe.json`: scaler integrity evidence.
- `findings.md`, `research-log.md`, `research-state.yaml`: trajectory and final understanding.
- `results/`, `followup_mixed/results/`:203 per-window prediction files and metadata.
- `logs/final-tests.log`, `logs/final-analysis.log`: final verification logs.

Cache transforms use only their own raw float32 window, never labels or pooled
statistics. Priors, scalers and classifiers are fitted separately inside the
original training splits. Cache fit timing excludes preprocessing and is not
end-to-end application latency. Public test has been viewed in earlier work; this
research does not replace a new independent final validation.
