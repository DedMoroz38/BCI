# Regularization Wave2 Implementation Plan

> **For agentic workers:** implement natively using superpowers:executing-plans.

**Goal:** launch the user-authorized regularization grid on Kaggle.
**Architecture:** isolated repo; frozen wave1 helpers; new training transforms
and averaged checkpoint policies; versioned private notebook.
**Tech Stack:** PyTorch,numpy/scipy,pytest,nbformat,Kaggle CLI.
**Spec:** PROTOCOL.md.

## Global Constraints
NoBA-based epoch selection,no pretrained models,no submissions,no monitoring.
Thirty-one simple configs followed by six fixed combinations;two seeds,11folds.

## Review Focus
Control RNG and clean predictions must reproducewave1.
Mixing must use training people/labels only and not modify source tensors.
BN recalibration must not consume validation/test or enable dropout.
SWAD ranges must exactly match outer-refit averaging and survive resume.
Expired budget and partial folds must not masquerade as complete CV.

### Task1: Regularizers and policies
Create src/regularizers.py,tests/test_regularizers.py; preserve frozen source.
- [ ] RED: LS0 equals oldweightedCE; transforms preserve shape/no mutation;
  evalbypass;cross-person same-class pairing;window/EMAaverage exact;BN guard;
  SWAD loss valley syntheticsequence selects expected epochs.
- [ ] Implement augment(),RegularizedNet,WeightAverage,LossValley.
- [ ] GREEN: ../.venv/bin/python -m pytest -q tests/test_regularizers.py.

### Task2: Training and orchestration
Create src/training.py,run_gpu.py,summarize.py,tests/test_training.py.
- [ ] RED: cleancontrol exact originalfit on syntheticdata;interrupted
  optimizer/augmentationRNGresume equals uninterrupted;averaged selection
  refits declaredrange;no target BN;everybatchbudget.
- [ ] Implement checkpointed NLL selection/refit,fold caches and grid order.
- [ ] GREEN: ../.venv/bin/python -m pytest -q.

### Task3: Notebook and launch
Create build_notebook.py,notebook/regularization.ipynb,metadata and source refs.
- [ ] Test generated smoke in isolated process,including LS,augment,averaging.
- [ ] Bundle pinned authorcode,wave1controls,Lag4predictions;validate everySHA.
- [ ] Run whole suite,review,fix consequential findings and regeneratebundle.
- [ ] Dryrun then authorized push;persist launchreceipt;end without polling.
