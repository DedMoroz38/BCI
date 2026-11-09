# Updated author code and private literature review: research synthesis

Inputs: local reproduction/BCI_literature_review.html,read without modifying or
publishing it; fresh read-only clone reproduction/ensemble_author_20261109,
commit ca5d3d0045120b426f77437d48c374f57d7cce45 (2026-11-09).
No new model experiments,benchmark run,dependency installation or submission.

Current measured candidate:Lag4+twoShallow16classSWAD.1,arithmetic70/30,T4,
balance1;all10pooledOOFcell60.6139%,previous60.2279%. Residual overfitting
remains. This motivates target-session adaptation as a next hypothesis,not a
proof that further architectural learning cannot work.

## Highest priority
1. Plug standalone Transducer.step(z,p) into our existing Lag4 or Lag4+SWAD
priors. Author module accepts arbitrary probabilities and uses session-prefix
GramPCA→softLDA→Sinkhorn,returning only the current row;warm20,k20,5iterations,
LDAshrink.5,beta1. Reuse saved outer-fold priors and causal features;no retraining
needed for first comparison. Pass priors before our final balance,matching
authorStreamModel logic;avoid stacking balance corrections unintentionally.
Authorall also changes bands,routing,CNN/geometricpooling,so moving the
transducer alone better preserves the current known strong components.
2. Relax prefix class marginals. Public label audit:14streams,onlyP10/session1
has exactly64/64/64;other streams are close but notexactlybalanced. P08first20
is4/10/6. Thus equal mass on every prefix is a heuristic,not known targettruth.
Try soft/unbalancedOT and gradual adaptation strength/warmup,using no labels.
3. Improve transduction geometry with Lag4 features or joint Lag4/CNN features.
The prior model already has lags;authortransduction has ordinary41chcovs in5
bands. These are different design choices. GramPCA keeps decomposition in n×n
even if feature dimension increases,but computing lagged covariances still costs.

Code-specific hypothesis:trans_feats stores eachz_i using its then-currentM_i.
LaterPCA treats them as a fixed common vector space. Test rawlogcov/fixedanchor
features or recompute past features relative to current commonM_t (past/current
only). This may reduce chart drift;it is not a demonstrated defect or gain.

## Next priority
4. Class-balanced pseudo-label reference,with source-prior shrinkage and soft
weights. SPDIM supports the general issue that marginal alignment can hurt
under labelshift;it does not prove this particular heuristic solvesPROTEUS.
5. Adaptation weight depends on stability across views/PCA dimensions/recent
prefixes;reduce the change when pseudo-label solutions disagree. Label-free
stability does not guarantee correctness,so treat as a separate hypothesis.
6. More expensive structural model:TSMNet/SPDdomain-specific normalization,
trained from scratch. Explicit session-specific normalization and covariance
pooling have relevant cross-session/person evidence. Streaming implementation
must estimate target stats on current/pastwindows and reset per session.
RPArotation frompseudo-labels is research-interesting but riskier:wronglabels
can reinforce alignment error;arbitrary tangent-space rotation from3centroids
is underdetermined and requires strong constraints/shrinkage.

## Critical reading
- OracleLDA withtrue sessionlabels does not prove easily recoverable unlabeled
clusters. HTML oracle/BOTS gains are not reproduced metrics of our model.
- Full-session offline plots must not be compared as streaming competition
scores. Current authorStreamModel is causal;full history is prefixonly.
- Review claims like "all successful TTA needs marginals","AdaBN almost always
safe","no one used this" are stronger than established by its cited papers.
- GeometricRPA distance invariance under a commoncongruence does not itself
align different domains or preserve classifier decisions.
- benchmark.py averages LOSO people then laterprotocol equally into a
"sealed-like mean". This is a local diagnostic,not official pooledOOFcellBA;
exact sealed cohortweights are undisclosed. Use raw predictions for ourmetric.

## Verified primary sources
- Author method:
  https://github.com/DorinDaniil/Brain-Decoding/blob/ca5d3d0045120b426f77437d48c374f57d7cce45/track2_bci/transduction/method.py
- Competition causal/sessionreset contract:
  https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md
- SPDIM,Li,Kawanabe,Kobler,ICLR2025:
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/fc6cd575b8340be6ee70beaed9ca4015-Abstract-Conference.html
- TSMNet,Kobler etal,NeurIPS2022:
  https://papers.neurips.cc/paper_files/paper/2022/hash/28ef7ee7cd3e03093acc39e1272411b7-Abstract-Conference.html
- RPA,Rodrigues etal,IEEE TBME2019:https://pubmed.ncbi.nlm.nih.gov/30596565/
- ACM,Carrara/Papadopoulo,IEEE TBME2024:https://pubmed.ncbi.nlm.nih.gov/38587944/
- UnbalancedOT,Chizat etal,MathComp2018:https://arxiv.org/abs/1607.05816
- GOPSA,Mellot etal,NeurIPS2024 (age regression,not our classification task):
  https://papers.nips.cc/paper_files/paper/2024/hash/399888844ffc8fa95c2db196f3f1b60b-Abstract-Conference.html
