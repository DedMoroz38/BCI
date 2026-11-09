# Primary-source rationale for classifier-only alternatives

These are families used on tangent vectors,not five new SPD representations.
Results in other studies motivate the candidates; they do not establish that
any candidate beats the submitted LR forest on cross-subject PROTEUS. We do not
copy upstream xDAWN,reference estimation,feature selection,normalization or
metadata features,because the intervention is strictly the final classifier.

## LDA with covariance shrinkage

Barachant,Bonnet,Congedo,Jutten (2012),Multiclass Brain-Computer Interface
Classification by Riemannian Geometry,IEEE Transactions on Biomedical Engineering
59(4):920-928,DOI10.1109/TBME.2011.2172210,establishes tangent-space linear
classification for EEG. Eder,Xu,Grosse-Wentrup (2024),Journal of Neural Engineering
21:044002,DOI10.1088/1741-2552/ad6793,benchmarks TGSP+LDA and TGSP+SVM against
CNNs. The authors' classification_pipelines.py explicitly implements
Covariances('oas')->TangentSpace('riemann')->LDA().

Our shrinkage='auto' is a deliberate classifier-only adaptation to820 correlated
features. It is not the exact LDA hyperparameterization in Eder's code.

https://doi.org/10.1109/TBME.2011.2172210
https://doi.org/10.1088/1741-2552/ad6793
https://github.com/ederm42/BCI-riemannian-vs-eegnet/blob/master/riemannian_vs_eegnet/classification_pipelines.py

## Linear SVM

Barachant,Bonnet,Congedo,Jutten (2013),Classification of covariance matrices
using a Riemannian-based kernel for BCI applications,Neurocomputing112:172-178,
DOI10.1016/j.neucom.2012.12.039. Section3.2 explicitly establishes equivalence
between the proposed kernel and a linear SVM on vectorized whitened matrix
logarithms. Motor-imagery experiments report benefits over CSP+LDA. Eder2024
uses SVC(kernel='linear') in its TS pipeline. These are direct EEG precedents.
We add subject-grouped sigmoid calibration because the forest averages class
probabilities rather than SVM margins.

https://doi.org/10.1016/j.neucom.2012.12.039
https://www.researchgate.net/publication/257352207_Classification_of_covariance_matrices_using_a_Riemannian-based_kernel_for_BCI_applications

## RBF SVM

Determining Levels of Affective States with Riemannian Geometry Applied to EEG
Signals (2025),Applied Sciences15(19):10370,DOI10.3390/app151910370,Section3.8
explicitly uses1711 standardized tangent features and an RBF SVM. This is a
direct representation match,but a different EEG task and a less persuasive
evidence source than the foundational TS-linear-SVM papers. The paper uses
C=10,gamma=.01;we predeclare C=1,gamma=1/820 as a fixed dimensionality-aware
adaptation. Do not call this a reproduction of that paper or attribute its
Gaussian SVM to Barachant2013,whose successful TS classifier was linear.

https://www.mdpi.com/2076-3417/15/19/10370

## Ridge classification

Dadi,Rahim,Abraham,Chyzhyk,Milham,Thirion,Varoquaux et al. (2019),Benchmarking
functional connectome-based predictive models for resting-state fMRI,NeuroImage
192:115-134,DOI10.1016/j.neuroimage.2019.02.062. This benchmark covers tangent
covariance parametrization and several classifiers,including ridge,across
six cohorts and roughly2000 people. It supports simple nonsparse linear
predictors as strong defaults. The evidence is fMRI,not EEG;our use transfers
the classifier family to the same mathematical type of covariance-derived
tangent vector. Alpha=1 follows its stated default. Sigmoid calibration is
needed for the forest's probability interface.

https://doi.org/10.1016/j.neuroimage.2019.02.062
https://fileserver-az.core.ac.uk/download/184908460.pdf

## Elastic-net least-squares classification

Barachant,Cycon,Gouy-Pailler's NER2015 winning solution uses tangent features,
elastic-net regression and subject-wise bagging. The published no-leak
configuration sets l1_ratio=.5,alpha=2e-4. Carrara&Papadopoulo2023 explicitly
includes COV+EN (tangent-space elastic net) as a comparison pipeline in its
augmented-covariance EEG evaluation. We retain those penalty values while
using three one-hot targets and subject-grouped sigmoid calibration. Original
NER row normalization,normalize=True,meta-features and xDAWN are not imported.
This is an adaptation of the classifier,not the complete winning system.

https://github.com/alexandrebarachant/bci-challenge-ner-2015
https://arxiv.org/html/2302.04508

## MLP

Cai,Zhang,Yang,Huang,Ma (2024),Manifold Learning-Based Common Spatial Pattern
for EEG Signal Classification,IEEE Journal of Biomedical and Health Informatics
28(4):1971-1981,DOI10.1109/JBHI.2024.3357995. Their MLCSP-TSE-MLP combines
tangent embedding with an MLP and reports benefits on three EEG datasets.
Because their preceding representation also changes,their results motivate the
family but do not isolate the MLP's effect. Our820->32->3 CUDA network is a fixed
classifier-only adaptation,not their architecture or complete pipeline.

https://pubmed.ncbi.nlm.nih.gov/38265900/

## Evidence limits

No evidence found that all five alternatives are uniquely specialized to TS,
or that all outperform LR for unseen PROTEUS people. They are established
vector classifiers with published TS applications. The strongest direct EEG
precedents are TS-LDA,TS-linear-SVM and TS-elastic net. Ridge's modality transfer
and the weaker direct RBF source must remain visible in the final interpretation.
