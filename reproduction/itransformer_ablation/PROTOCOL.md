# Author iTransformer WITHOUT lags versus single Lag4 baseline

2026-11-07: user explicitly requests Kaggle notebook+run,author implementation,
previously proposed small hyperparameters,no lag copies for iTransformer,and
train/validation loss plots. No competition submission. Only2 model families.

## Article-aligned outer evaluation
Original local paper Riemannian_Forest(1).pdf,section EEG experiment:6train,
2validation,2test public participants;8LOSO on train+val,one later-session fold,
original train-only public test. Use exact existing10 folds and chronological
session resets. Report mean LOSO session BA and official cell BA,late-session
and old public-test scores. Public test already seen;not untouched or sealed.

Single baseline EXACT old Lag4:41channels,4–40Hz,4sample lag ->82x476,OAS,
log-Euclidean cumulative reference prior5,3403 tangent features,StandardScaler,
balanced LogisticRegression(C=.01,max_iter2000). No forest/bootstrap.
Reproduce old single-baseline controls and implementation parameters.

## Author code and minimal adaptation
Official https://github.com/thuml/iTransformer
Pinned commit c2426e68ca13f74aaec08045c5c724d8ad328124,MIT license retained.
Vendor original Model,DataEmbedding_inverted,Encoder/EncoderLayer,FullAttention/
AttentionLayer and masking utilities. Keep encoder forward/math verbatim.
If needed move optional Reformer import to its unused constructor,so plain
FullAttention does not require unrelated reformer-pytorch. Document exact patch.
Subclass original Model;replace forecast projector with Linear(41*64,3),override
forward to use author embedding+encoder and flatten channel representations in
fixed electrode order. No Gram/SPD/reference adaptation in iTransformer.

Input same raw4second41x480 recording window,band4–40Hz Butterworth4 forward/
backward as baseline. NO lag augmentation/truncation for iTransformer:41tokens,
480samples each. Per-window channel temporal centering and ONE training-fitted
scalar RMS;preserve relative channel powers. Set author use_norm=False to avoid
per-channel variance normalization erasing those powers. No timestamp tokens.
One encoder layer,d_model64,n_heads4,d_ff128,dropout.2,GELU,FullAttention factor5
(unused in dense attention),output_attentionFalse. All encoder+head train jointly.
AdamW lr=.001,weight_decay=.001,clip5,batch64,max80epochs,patience10,min_delta1e-4,
no LR scheduler. Fixed independent seeds0,1,2;report mean/std,do NOT ensemble or
select a seed using outer scores. Architectural values follow prior proposal;
training settings are fixed starting choices,not tuned optima.

## Nested model selection and loss curves
For EACH outer fold,GroupShuffleSplit(test_size=.2,random_state33) on OUTER TRAIN
participants only defines inner fitting/validation people. All three seeds use
that same split. No outer-evaluation person/window enters early stopping,
normalization,class weights or any training decision. For late-session protocol,
inner people are still disjoint;record overlap between inner-val people and
outer-eval known-person identities transparently (their windows remain disjoint).
Check all selected training classes exist.

Class-weighted CE uses class weights N/(3N_class) fitted on current TRAIN only;
normalize weighted CE by sum of example weights. Fixed encoder/head AdamW decay.
After EVERY epoch evaluate both inner TRAIN and VAL in model.eval()/no_grad:
record unweighted CE,weighted CE,session/cell BA and accuracy;train and val
weighted curves use SAME inner-training class weights. Optimization dropout CE
is logged separately. Plot train/val unweighted CE plus balanced accuracy;save
CSV after each epoch and PNG every5epochs plus first/final/stopped epoch,so
overfitting curves survive partial failures without excessive plotting overhead.
Also compute a single inner Lag4 baseline train/val CE/BA,draw horizontal
reference lines. It has no neural training epochs.

Early stop uses ONLY inner VAL weighted CE,patience10,min_delta1e-4;restore/save
best inner checkpoint. Then train a FRESH model from same seed on ALL outer
training people for exactly selected best_epoch,refitting scalar RMS/class
weights. This refit has NO outer-validation feedback. Save separate refit train
loss history. Do not plot outer test as validation or claim inner-best metrics
are outer scores. No post-evaluation hyperparameter changes.

Freeze all CV scores/hashes and ranking before public test. Aggregate iTransformer
outer scores by arithmetic mean over3seeds for comparison;also report all seeds
and paired person deltas. Never average their predictions in this experiment.
Save all checkpoints,curves,metadata,predictions,source/data/library hashes.
Resume must validate complete experiment provenance and individual artifacts.

## Verification and execution plan
1. Vendor pinned upstream with hashes/license and documented dependency-only patch.
2. Tests:author encoder equivalence,41token/no-lag shape,independent Q/K weights,
   finite gradients,deterministic eval/save-reload,train-only preprocessing,
   inner/outer split isolation,best-epoch selection and curves persistence.
3. Implement two-stage training and10fold driver,reuse exact old baseline code.
4. Independent review;private T4 Kaggle notebook,remote CUDA smoke before fits.
5. Plot curves in notebook as fitting proceeds;automatic final independent report.
