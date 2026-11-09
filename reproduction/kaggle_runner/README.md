# Kaggle connection for the classifier experiment

Installed skill: `/Users/den1shh/.codex/skills/kaggle/SKILL.md`
Source: https://github.com/shepsci/kaggle-skill/tree/main/skills/kaggle
Skill version:3.0.1. Separate local Python environment:
`reproduction/kaggle_runner/.venv` (Python3.12.12,kaggle2.2.5,kagglehub1.0.3).
The original experiment environment is unchanged.

Verified on2026-10-29: existing OAuth credentials accepted as `den1shh`; Kaggle
web/API/storage and MCP reachable. No credential values are in this directory.
Setup used the existing login; no credential values are in experiment artifacts.

From `/Users/den1shh/Documents/diploma`, if login ever needs refreshing:

```sh
reproduction/kaggle_runner/.venv/bin/kaggle auth login --force
```

Safe connection check:

```sh
PATH="$PWD/reproduction/kaggle_runner/.venv/bin:$PATH" reproduction/kaggle_runner/.venv/bin/python /Users/den1shh/.codex/skills/kaggle/scripts/kaggle_skill.py doctor --verify
```

No need to create another token when the verified OAuth login works. For
dataset publishing use the skill's `--via cli` option with OAuth.
Check GPU availability/phone verification in https://www.kaggle.com/settings.

The private input datasets are `den1shh/proteus-lag4-classifier-inputs` and
`den1shh/proteus-lag4-classifier-source`. Version1 of the private GPU notebook
`den1shh/proteus-lag4-classifier-ablation` was launched on2026-10-29.
`build_notebook.py` packages only the explicitly selected source and baseline
receipts; `notebook-build.json` records source and notebook hashes.
Kaggle limits notebook source to1MB,so the2.1MB code ZIP is an attached input.
This Kaggle workspace is compute infrastructure; competition submissions remain
on the original Neural Interface Challenge Codabench.

GPU acceleration needs a CUDA-capable implementation. sklearn's current LR,
LDA,SVC,ridge,ElasticNet and MLP do not gain CUDA simply by enabling a GPU.
Keep the submitted baseline exact; implement the requested MLP on PyTorch/CUDA
and document its fixed training configuration before evaluating it.
