# Primary sources and experiment mappings

- Müller,Kornblith,Hinton. When Does Label Smoothing Help? NeurIPS2019.
  https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html
  Supports studying softened training labels/calibration;the fixed fourepsilon
  values are our exploratory grid,not a literature guarantee for PROTEUS.
- Rommel,Paillard,Moreau,Gramfort. Data augmentation for learning predictive
  models on EEG:a systematic comparison,2022. https://arxiv.org/abs/2206.14483
  Adequate augmentations are task-specific. Our operations are on causally
  aligned inputs;coordinate dropout is not a faithful physical-electrode loss.
- Zhou,Yang,Qiao,Xiang. Domain Generalization with MixStyle,ICLR2021.
  https://openreview.net/pdf?id=6xHJ37MVxxp
  Train-only feature-statistic mixing;our same-class cross-person log-power
  adaptation is a hypothesis for EEG,not an established result of theirpaper.
- Cha etal. SWAD:Domain Generalization by Seeking Flat Minima,NeurIPS2021.
  https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html
  Source:https://github.com/khanrc/swad. Archived authorLossValley/averaging
  source+MITlicense and hashes under vendor/. Our epoch-based dense-segment
  adaptation preserves numeric averaging multiplicities,checked against author
  source on four synthetic loss histories. BN recalibration uses clean training
  only. This is adapted to our fixedNLL stopping/refit protocol,not a claim of
  reproducing the paper's complete optimization or domain benchmarks.
- Guo etal. On Calibration of Modern Neural Networks,ICML2017.
  https://proceedings.mlr.press/v70/guo17a.html
  Fixed inferenceT4 carried fromwave1;not fitted on outerlabels,not claimed a
  validated calibrated temperature. Userexcluded changing epochselection.
