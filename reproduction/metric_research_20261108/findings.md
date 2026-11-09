# Starting insights

1. Probability balance on Lag4 improved legacy CV with negligible inference
cost. Check transfer to observed public-test people only after freezing choices.
2. A fixed100epoch CNN is useful on many dev people but degrades held-out2person
public test. This does not distinguish overconfidence,geometric pooling,capacity
and optimization. Reuse logits to isolate pooling before retraining.
3. Existing original-CNN mixes used the unaugmented author forest,not Lag4.
4. All10public participants may be used for final training; future primary CV
should leave one of10people out,rather than spending every decision on8people.
5. Repeated use of validation means future results are exploratory. Keep inner
epoch selection independent of outer people and avoid stacked-OOF leakage.
6. Source-person weighting and cell weighting are distinct: multi-session people
contribute more cells in the competition metric,while equal-person loss is a
separate domain-generalization hypothesis. Do not conflate these justifications.

## Completed local evidence

An arithmetic mixture with a softened CNN and causal probability balancing
is a stronger candidate than the previous sharp geometric mixture. The fixed
grid's CV winner uses 70% Lag4 and 30% softmax(CNN logits / 4), followed by
balance alpha1 with uniform prior10. Legacy LOSO cell BA rises from 56.30%
to 64.43%, while the previously observed public test rises only from 55.72%
to 56.12%. This does not establish sealed-set improvement. It shows that the
CNN's value depends strongly on how its probabilities enter the ensemble;
the ordinary geometric version had only 45.81% on that old public test.
The grid changes multiple interacting factors, so it does not isolate a
single causal explanation or prove that the CNN probabilities are calibrated.

All10 controls establish a separate reference for the pending Kaggle wave:
pooled OOF cell BA is 57.44% for Lag4 and 58.53% with balance1. These numbers
must not be ranked against legacy8 scores as if they shared a protocol.

Pending Kaggle training tests compact capacity, source-person balancing,
and inner validation stopping. Analyze its saved outer logits against the
all10 baselines only after the user reports completion; no monitoring.

## Completed GPU evidence

Version2 completed all55 fits in520s (training package,excluding setup).
Independent rescoring,source/data/artifact hashes,person split and selected
epoch/refit guards passed. All55 saved models replayed on CPU with fresh
causal alignment,exact labels and max probability delta1.252e-6.

All10 OOF cell BA is60.23% for Lag4 plus the two-seed compact class-loss CNN
ensemble,using the prior70/30,T4,balance1 recipe. Lag4 is57.44%,balancedLag4
58.53%. The mixture improves9/10people and later cells64.64%→66.70%.
CNNs alone remain weak: CNN40seed0=49.42%,compactclassmean2=52.55%,
compactsubjectmean2=52.30%. Class balancing vs person balancing has no clear
large separation. Holding seed0 fixed favors the40filter CNN mixture60.09%
over compactclass59.69%/compactsubject59.56%,so compact capacity is not
established as the source of ensemble improvement.

Early stopping by NLL appears misaligned with BA:49/50 LOSO histories peak
in observed inner BA later than the NLL-selected epoch. In51/55 histories
trainCE falls while valCE rises by stopping. This supports an overconfidence
and criterion-mismatch hypothesis,not a claim that later checkpoints improve
outer generalization. The next hypothesis would select epochs by inner cellBA
and handle probability softness separately; no new training was launched.
All candidate rankings use reused public CV and do not establish sealed gain.
