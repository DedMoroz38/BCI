# Fixed full-ensemble forest substitutions

Implements the parent wave's frozen `../PROTOCOL.md`. No training, refitting,
historical equality gate, weight selection, or new validation splits.

Use the ten accepted 8/2 split records, saved two-seed plain/point/early CNN
logits, and raw saved ordinary Lag4 nuclear .01/.05 and mSSA8/mix25 nuclear
.01/.05 probabilities. Keep the saved original full ensemble as the comparator.

For each replacement, form .7 forest + .3 mean-seed softmax(plain logits/4).
Run the inherited causal `lag_running` geometry TTA with tau .5, beta 1,
k20, five iterations, shrinkage .5, warm20, strength .5, no ramp. Source
logcovariance means use exactly the eight source people. All variants share
only label-free prefix geometry. Then .8 TTA + .2 balanced point CNN, followed
by .7 of that + .3 balanced early CNN. No balancing of forests before TTA.
State resets at each participant/session and follows run/onset order.

Per-window OAS caches use original temporal covariance band indices
1/2/3/4/0, checked against the manifest, and the existing window identity
mapping. Each mapped raw window's BLAKE2 identity and all forecast metadata
are checked before numerical work. These are row alignment checks, not model
score equality requirements.

Primary first-session cell BA: average each participant's two repeat metrics,
then average the ten participants equally. Preserve per-person changes, NLL,
worst3, stream bins, official-style all-session cell BA and repeat spread.

One process, BLAS1. Run only after the CPU synergy completion receipt.

Commands:

```sh
python -m pytest -q reproduction/literature_wave_20261111
python reproduction/literature_wave_20261111/fusion/fusion_wave.py --validate-only
python -u reproduction/literature_wave_20261111/fusion/fusion_wave.py
```
