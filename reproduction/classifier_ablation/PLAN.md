# Classifier Ablation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution.

**Goal:** Run the submitted Lag4 forest with its original LR,MLP,and five fixed
classifier alternatives,then deliver verified paired final metrics.

**Architecture:** Read-only original code and matrix caches; exact per-fold
standardized tangent feature caches shared by all classifiers. Each alternative
retains the tree draws,subject weights,50-member probability mean,and protocols.

**Tech Stack:** Existing locked Python3.12,numpy,scipy,sklearn,pyriemann environment.

**Spec:** PROTOCOL.md (the user explicitly requested execution and final metrics).

## Global Constraints

Only classifier changes; no edits to previous work or submitted ZIP. Fixed
configurations,seed0,50 trees,20 electrodes,lag4,820 features,prior5. Eight LOSO,
one later-session,one original train-only public test. CV selection frozen first.

## Review Focus

Effective class weights match installed LR; calibration splits exclude entire
physical subjects; expanded channels and RNG draws match submission; feature
cache respects stream reset and all train-only fitted statistics; convergence
and baseline equality failures are surfaced rather than hidden.

## Tasks

- [ ] Task1: Tests for balanced weights,subject-isolated calibration,simplex
  probabilities and deterministic tree specs. Observe failures,implement
  classifiers.py,run tests. Document verified primary-source evidence.
- [ ] Task2: Build common.py and prepare_features.py using original feature
  methods. Test training/evaluation transform equality and stream reset; verify
  immutable hashes. Cache nine CV/later folds first,then public-test features.
- [ ] Task3: Implement run.py and launcher.py; save per-tree probabilities,
  convergence and runtime. Reproduce all baseline predictions before trusting
  candidate scores. Run all six candidate CV/later folds.
- [ ] Task4: Freeze selection.json with all CV hashes and protocol commit before
  alternatives' public test. Run seven train-only public test models.
- [ ] Task5: analyze.py independently recomputes official metrics,paired subject
  effects,uncertainty and winner stability. Write REPORT.md,CSV,figure,literature
  notes,and provenance. Run meaningful integrity tests and fresh final review.

## Execution decisions

User already fixed the intervention and requested execution; no additional
design approval round. Dedicated sibling research repository is isolated without
altering any existing source repository. Scope is70 evaluations,not an open-ended
optimization project. No GPU,network-dependent packages,or external submission.
