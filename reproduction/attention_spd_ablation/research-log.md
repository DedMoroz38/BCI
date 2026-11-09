# Experiment ledger

2026-11-07: User authorized both attention SPD replacement experiments on Kaggle,
then explicitly excluded tied Q/K. Protocol e499aba committed before scores.
Implementation follows the two previously selected constructions with bounded
score exponential and fixed ridge Gram. No new architecture search.

Independent code review found two important gaps before execution: resume did
not bind inputs/imported dependencies; synthetic tests did not cover trained
save/reload LR inference or mixed bootstrap packs. Added frozen runtime/raw input
hashes,all imported implementation hashes,fold/spec/metadata fingerprints in every
pack,result; changed-input rejection test failed before implementation then
passed. Added trained encoder+geometry+LR roundtrip,explicit two-session reset,
and mixed-pack zero-weight exclusion tests. Gram product+ridge computed in
float64. Eleven tests pass locally. No accuracy experiments have run yet.

Ruling: Continue directly within user's already-authorized Kaggle experiment
scope; no redundant design/launch confirmation. New notebook is private and
uses weekly GPU quota; no competition submission or existing notebook update.

Execution amendment: v1 first baseline reproduced exactly; first Gram pack
T4 eigendecomposition is ~12.5sec/epoch (epochs1→10:112.68sec). Projected Gram
training exceeds24h. No attention evaluation score yet. Float64 CPU eigh on
640x40 test matrices locally takes .067sec. Offload this operation only,without
changing architecture/objective/optimizer or evaluation. Added CUDA equivalence
test; run remotely before v2 training. v1 cancellation needs session ID; browser
Arc access was rejected by tool,API metadata lacks ID. Asked user for v1 URL or
manual cancellation. Other work continues while waiting.
