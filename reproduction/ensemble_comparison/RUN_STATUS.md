# GPU recovery — 2026-11-08

## Complete and locally verified

Recovery v3 COMPLETE (API independently confirmed).294output files downloaded.
Local verify_gpu_results.py independently checked80/80 scores,metadata and
probability metrics;10/10 Lag4 predictions match historical controls;63frozen
CV result hashes and46imported parent records verified.30new replay checks
passed exact parameters/labels,maximum probability delta0.0. Source training
warnings for interrupted P02 remain unavailable;no convergence claim.

Stable report:LATEST_REPORT.md. Verification:LOCAL_GPU_VERIFICATION.json.
Raw outputs:downloads/recovery-v3/ensemble-results.

LOSO sessionBA:Lag4 56.2801%,forest53.1430%,+balance53.0683%,+bands52.2848%,
+routing53.7920%,+deep61.3656%,all but deep52.5035%,all60.5515%.
Public test cellBA:Lag4 55.7164%,+deep45.9481%,all48.4297%.
Deep wins6/8LOSO people but loses9.7683pp on old public test vs Lag4.
No configuration changed or new experiment launched after inspecting results.

## Recovery history (superseded running status)

Current run:https://www.kaggle.com/code/den1shh/proteus-author-ensemble-recovery
Version3 imported6trained CUDA component checkpoints and46verified evaluation
records (remote log at78.8266s). It continues missing scores/fits only. Two T4
devices are available; workers now select distinct device indices when possible.
Read-only watchers:PIDs71750/71751,completion-recovery-v3.json and
logs/recovery-v3-{status,live}.log.

Original comparison v2 endedERROR:bitwise FP32 prefix/serialization assertion
failed on P02,max probability difference5.9846e-6. Current replay policy verifies
all serialized parameters exactly,probabilities with absolute tolerance1e-4,
and decision labels exactly. No training math/hyperparameters/folds changed.
Nine local tests pass,including rejection of decision changes even below1e-4.
Local ARM replay of downloaded P02 parameters is exact; original Linux FP32
roundoff was not reproduced locally. Alignment prior condition~1.28e4,so no
evidence of near-singular covariance as the cause. Do not claim a proven
backend-specific root cause beyond an overly strict bitwise probability check.

Saved artifacts:downloads/gpu-v2/ensemble-results (176downloaded files).
Private checkpoint dataset:den1shh/proteus-author-ensemble-checkpoints,
17.7MB archive of129necessary artifacts. Kaggle automatically unpacks ZIPs;
recovery v3 verifies pinned integrity-receipt SHA and each extracted file SHA.
Parent v2 manifest is explicitly pinned;rebound records keep original record
hashes and parent manifest IDs;resume-lineage.json records every imported hash.
Original training warnings are unavailable for interrupted P02;an empty resumed
warnings file is not evidence that its original training emitted no warnings.

Recovery v1 ended before training because Kaggle rejected failed-notebook output
as kernel source. Recovery v2 ended before training because the uploaded ZIP was
automatically unpacked. Only v3 is intended to run training. These preliminary
attempts did not repeat any trained folds. No new competition submission.

## Earlier launch record (superseded)

Local parent65040 and descendants stopped by SIGTERM at user's request.
Local status watcher65189 also stopped; ps confirmed both processes absent.
Local results retained under results/:16 completed evaluations across first2
folds at cancellation request. They are not pooled into the CUDA report.

Active private Kaggle notebook:
https://www.kaggle.com/code/den1shh/proteus-author-ensemble-vs-lag4

- Version1 CANCEL_ACKNOWLEDGED,independently API-verified after user cancelled.
- Version2 RUNNING,independently API-verified.
- Tesla T4 confirmed in runtime logs.
-5 remote protocol/serialization tests passed (one warning).
- Actual CUDA training + serialized model replay smoke check passed.
- Both initial full folds started (P01,P02),not merely queued.
-80 evaluations; seven fixed author variants and exact submitted Lag4.
-No forest_lag100 until user supplies training recipe and resumes that variant.

Version2 corrects output-relative path for freezing CV before public test;
regression test failed before fix and passed after. Local suite6passed.
Upstream author files are byte-identical,pinned at934f423a73e1ed50d653345d27256ecfe50ab009.
Only neural training device changes to CUDA; upstream inference stays CPU.
Same original splits and pinned numerical dependencies,recorded remote runtime.

Build receipt: notebook-build.json. GPU output root:/kaggle/working/ensemble-results.
Automatic remote runner independently rescores all80 prediction artifacts and
writes REPORT.md,comparison.csv,fold_scores.csv,paired.json,verification.json.
Read-only monitoring: monitor_gpu.py,completion-gpu-v2.json,live-gpu-v2.jsonl,
logs/gpu-status.log and logs/gpu-live.log. Monitoring PIDs66040,66041.
GPU ETA not asserted until a complete fold establishes runtime.
