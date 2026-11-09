# Primary sources informing wave1

- Guo,Pleiss,Sun,Weinberger. On Calibration of Modern Neural Networks,ICML2017.
  https://proceedings.mlr.press/v70/guo17a.html
  Supports temperature as a low-dimensional posthoc calibration mechanism;
  does not establish transfer to EEG or justify fitting on evaluation labels.
  Here first wave uses fixed temperatures1/2/4 as an exploratory softness ablation.
- Gong et al. Confidence Calibration for Domain Generalization under Covariate
  Shift,ICCV2021. Official CVF PDF search retrieved section4.1 but fullopen403:
  https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_Confidence_Calibration_for_Domain_Generalization_Under_Covariate_Shift_ICCV_2021_paper.pdf
  Multiple source calibration domains motivate person-aware validation;no
  assumption that one source-trained temperature is automatically valid on EEG.
- Junqueira,Aristimunha,Chevallier,deCamargo. A Systematic Evaluation of Euclidean
  Alignment with Deep Learning for EEG Decoding,2024,arXiv2401.10746v4.
  https://arxiv.org/abs/2401.10746
  Evaluates EA/shared neural models and voting;alignment can help while extra
  ensembling need not outperform a shared aligned model. Causal streaming EA
  in our protocol is a task-specific adaptation,not their full-session procedure.
- Official PROTEUS scientific FAQ and Track02 description,retrieved2026-11-08:
  https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md
  https://neural-interfaces26.github.io/tracks.html
  Public10people/14sessions;sealed later+new participants,exact cohort private;
  metric participant×session×context cellBA. No target labels/calibration in fit.
- Competition rules,retrieved2026-11-08:
  https://neural-interfaces26.github.io/rules.html
  Training uncapped;inference60min on oneA100;sealed submission quota5total,
  max1/day. This research does not submit to Codabench.
