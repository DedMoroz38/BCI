# Launch confirmed — 10 November 2026

Kaggle CLI: **Kernel version 1 successfully pushed**.

https://www.kaggle.com/code/den1shh/proteus-transfer-normalization-deep-dive

Private T4 notebook.14 fixed small model policies x2seeds x11outer folds:
308outer fits with inner-NLL selection and refit. Application cap600min;
Kaggle timeout41400s. No baseline refits, reproduction gates, remote test
preflight, competition submission or background monitoring.

Detailed source review completed first: literature/REVIEW_RU.md.
Prospective variants/metric rules: PROTOCOL.md. Priority: high OOF cell BA
and transfer to held-out people; equal-person/per-person summaries included.
Comparison baseline63.2294% OOF/71.6395% later is bundled as saved probabilities.

Minimal local new-code gradient/causal-prefix and optimizer smoke passed;
no historical baseline was fitted or checked for numeric equality.
Actual remote run completion and metrics have not been checked.
Output directory: transfer-results; completed predictions/curves/models,
progress/failure receipts,primary_comparison.csv,transfer_by_person.csv,REPORT.md.
