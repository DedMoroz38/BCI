# Complete

All203 new jobs finished, all261 candidate/control prediction files officially
re-scored,101 tests passed. Both pre-test selections committed55feda5.

Main outcome: fixed lag4 improves both forest20 LOSO and public test over previous
delay1 (.562801/.555407 vs .553871/.544350). Its preference is retrospective;
no lag/parameter tuning followed test. Pre-test CV winner mixed temporal halves
(.610376) fell to .507398 on public test and is not an established overall optimum.

Raw block-diagonal and quadrature forms have numerical null tangent coordinates
amplified by the source scaler; both retained transparently but excluded from
recommendation. Hadamard follow-up preserves information/eigenvalues and avoids
null coordinates without changing scaler or baseline.

Final files: FINAL_REPORT.md, final_comparison.csv, final_fold_scores.csv,
final_verification.json, final_comparison.png, literature/survey.md, findings.md.
All numerical code and prior results hash-verified. Final plot visually inspected.
All native experiment/controller sessions completed; no further experiments pending.
