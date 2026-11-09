# Reproduction provenance

- Request: reproduce the paper experiment exactly, including synthetic and EEG sections.
- Paper: ../Riemannian_Forest (1).pdf, section 5, pages 14–17.
- Source: https://github.com/DorinDaniil/Brain-Decoding
- Source commit: 8d577af42619e5911cc5da8743c8a3e9b629014b (2026-10-28).
- All reproduction runners/results live outside the source checkout.
- Original EEG notebooks contain no saved outputs and use default model seed 0.
- Requirements pin neural{bench,set,fetch,train}=0.4.0 and exca=0.5.29, but do not pin most numerical packages. Original environment is unavailable.
- No synthetic experiment implementation was found in the checkout or in historical tracked filenames. Asked user for original sources; independent EEG work continues.
- All numeric paper values in report are reference values and must not be confused with measured values.
- Exact synthetic random seeds, number of trees for figure 1a/1b, several grids and variance resampling details are not specified in the paper. Reconstructed results must be labelled accordingly.

- User supplied official competition repository; downloaded unchanged at commit 2b53aec91c111f8b5423b6674f096c48cc1a99af.
- Verified current scoring has both cell_mean (ranking) and session_mean (paper/reference baselines); reproduce the latter and retain both.
- PROTEUS loader pins NEMAR nm000290 v1.0.0, CC-BY-4.0. Download uses upstream retry wrapper, no preprocessing changes.
- Synthetic reconstruction choices are recorded in results/synthetic/manifest.json before execution. User provided the competition repository, not original synthetic scripts; synthetic results remain a disclosed reconstruction.

- Data preparation completed with exact paper counts 1521/379/755 and 6/2/2 subjects. All splits are disjoint and in recording order.
- EEG models run concurrently in three independent processes (one per model), BLAS threads=1; model mathematics and default seed unchanged.
- Separate synthetic solver-tolerance diagnostic does not change the primary reconstruction or tune the EEG test.

- Completed 29 EEG jobs; all 11 published baseline/forest values in table 2 match after rounding to three decimals.
- Independent official re-scoring verified all 29 saved prediction files at atol=1e-14.
- Final protocol/matrix checks: 8 tests passed. Re-running EEG runner resumed all 29 jobs without fitting/evaluating again.
- Original source checkouts have no tracked modifications. Final report: REPORT.md; artifact and script SHA-256: checksums.json.
- Exact synthetic numerical reproduction remains unavailable without original scripts/seeds. Primary reconstruction and supplemental solver-tolerance results are both retained, with no parameter selection to force agreement.
