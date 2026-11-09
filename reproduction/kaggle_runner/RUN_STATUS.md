# Completed run record

**COMPLETE:** training v4 and CPU report v1 both finished successfully. All70
outputs and all10 baseline label matches were independently verified on Kaggle
and locally. Final summaries: `../classifier_ablation/to_human/`; raw predictions:
`../classifier_ablation/results/`; immutable download: `downloads/training-v4/`.
CV winner is linear SVM; public-test BA is highest for the submitted LR baseline.
No requested work remains. The notes below are historical execution instructions.
The predicted reporting failure did not occur remotely because upstream run.py
was not included in the source bundle; the exact-path analyzer fix is useful locally.

Training: private `den1shh/proteus-lag4-classifier-ablation`,version4.
Implementation0350b84; frozen classifier config SHA256:
`12abff9e82c4954355840023e94af27c90981efd953793259bbd59f8eca2e827`.
Source ZIP71595fd41a3d5d00a16039b9e5e5a7584358acd3d10f9015fa8606b0e6e05864.
Source manifest3e8cf0b952a1a18ab496a89d6a31275522e8efa003ea6b8ed4b9e6467d2a3c81.
Dataset inputs version1; source dataset version3. Do not regenerate/upload source
or restart training to apply report-only changes.

Live monitor: `follow.py 4`,tool exec session16088. It reconnects automatically,
deduplicates display,and appends `live-v4.jsonl`. Original CLI log sessions
45512/22142 need no polling. Version3 cancellation acknowledged for ID363509788.

The GPU MLP takes3–6s/fold; fixed CPU ElasticNet with grouped calibration is
the bottleneck (~11–12min/10trees). Single persistent3-worker pool overlaps
CV jobs and main-process feature preparation. Drain all63 CV records before
selection and test. No hyperparameter changes or additional configurations.

All17 remote training tests passed. Current local suite20 tests passes.
Independent review found no blocking training/evaluation bug. Local analysis
commit e4ef942 fixes a standalone run.py import collision and adds independent
logloss/Brier rescoring plus selection stability. These change no classifier
configuration hash. The running v4 retains its old analysis cell: after all70
jobs finish,it is expected to fail at report import. Outputs survive errors
(verified against version2). Download/inspect v4 outputs even if status ERROR.

`build_report_notebook.py` already prepared a private CPU-only report notebook
in `report-notebook/`,slug `den1shh/proteus-lag4-classifier-report`.
After v4 completes,push it via the Kaggle skill; it attaches v4 kernel outputs,
checks every original source hash,copies immutable results,installs the same
pins,runs3 analysis tests,and independently verifies all70 outputs and all10
baseline predictions. No retraining. If attaching an ERROR kernel version is
unsupported,download outputs and publish the selected results/source privately
as a dataset for this analysis notebook. Keep original training notebook atv4.

After report completion download the classifier-results subtree and independently
rerun local analyze.py with CLASSIFIER_OUTPUT pointing at those files. Preserve
the70 JSON/prediction files; selection.json locks63 CV JSON hashes.
Read REPORT.md,comparison.csv,paired_subject_scores.csv,selection_stability.json,
verification.json. Final response in Russian with7-model metric table,LOSO-selected
winner,public-test cell BA,changes vs55.716% baseline,and limitations. Public test
was seen earlier; no sealed-score claim and no new Codabench submission.
