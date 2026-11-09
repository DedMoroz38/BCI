"""Classifiers on the exact frozen standardized tangent feature vectors."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ElasticNet,LogisticRegression,RidgeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

VARIANTS=('baseline','lda_lw','svm_linear','svm_rbf','ridge','elasticnet','mlp')

def balanced_weights(y,w):
    y=np.asarray(y);w=np.asarray(w,dtype=float).copy()
    classes=np.unique(y)
    factors=compute_class_weight('balanced',classes=classes,y=y,sample_weight=w)
    return w*factors[np.searchsorted(classes,y)]

def calibration_splits(y,groups):
    n=len(np.unique(groups))
    if n<2:raise ValueError('Calibration requires at least two real subjects')
    splits=list(GroupKFold(n_splits=min(3,n)).split(np.zeros(len(y)),y,groups))
    for train,test in splits:
        if set(groups[train])&set(groups[test]):raise RuntimeError('Subject leakage')
        if len(np.unique(y[train]))!=3:raise ValueError('Missing calibration-training class')
    return splits


class ElasticNetClassifier(ClassifierMixin,BaseEstimator):
    """Independent elastic-net least squares for three one-hot targets."""
    def __init__(self,alpha=2e-4,l1_ratio=.5,max_iter=10000,tol=1e-4):
        self.alpha,self.l1_ratio,self.max_iter,self.tol=alpha,l1_ratio,max_iter,tol

    def fit(self,X,y,sample_weight=None):
        self.classes_=np.unique(y);self.n_features_in_=X.shape[1]
        target=np.eye(len(self.classes_))[np.searchsorted(self.classes_,y)]
        self.regressor_=ElasticNet(alpha=self.alpha,l1_ratio=self.l1_ratio,
                                  max_iter=self.max_iter,tol=self.tol,selection='cyclic')
        self.regressor_.fit(X,target,sample_weight=sample_weight)
        return self

    def decision_function(self,X):return self.regressor_.predict(X)

    def predict(self,X):return self.classes_[self.decision_function(X).argmax(axis=1)]


class SubjectSigmoidClassifier(ClassifierMixin,BaseEstimator):
    """Subject-isolated OOF scores; class balancing refitted inside each fit."""
    def __init__(self,estimator):self.estimator=estimator

    def fit(self,X,y,*,sample_weight,groups):
        self.classes_=np.unique(y);self.n_features_in_=X.shape[1]
        w=np.asarray(sample_weight,dtype=float)
        self.oof_scores_=np.empty((len(y),3))
        self.calibration_groups_=[]
        for train,test in calibration_splits(y,groups):
            model=clone(self.estimator)
            model.fit(X[train],y[train],sample_weight=balanced_weights(y[train],w[train]))
            self.oof_scores_[test]=model.decision_function(X[test])
            self.calibration_groups_.append(dict(train=np.unique(groups[train]).tolist(),test=np.unique(groups[test]).tolist()))
        effective=balanced_weights(y,w)
        self.calibrators_=[_SigmoidCalibration().fit(self.oof_scores_[:,k],y==cl,sample_weight=effective)
                           for k,cl in enumerate(self.classes_)]
        self.estimator_=clone(self.estimator).fit(X,y,sample_weight=effective)
        return self

    def predict_proba(self,X):
        scores=self.estimator_.decision_function(X)
        p=np.stack([cal.predict(scores[:,k]) for k,cal in enumerate(self.calibrators_)],axis=1)
        total=p.sum(1,keepdims=True)
        return np.divide(p,total,out=np.full_like(p,1/3),where=total!=0)

    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(1)]

def fit_classifier(name,x,y,w,groups):
    if name not in VARIANTS:raise ValueError(name)
    x=np.asarray(x,dtype=float);y=np.asarray(y);w=np.asarray(w,dtype=float).copy()
    groups=np.asarray(groups)
    if np.any(w<=0):raise ValueError('Pass only positive-bootstrap-weight windows')
    if name=='baseline':
        return LogisticRegression(C=.01,max_iter=2000,class_weight='balanced').fit(x,y,sample_weight=w)
    effective=balanced_weights(y,w)
    if name=='lda_lw':
        if not np.array_equal(w,w.astype(int)):raise ValueError('LDA requires integer bootstrap counts')
        repeated=np.repeat(np.arange(len(y)),w.astype(int))
        return LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(3)/3).fit(x[repeated],y[repeated])
    if name=='mlp':
        raise ValueError('Use gpu_mlp.fit_predict for the fixed batched CUDA experiment')
    if name=='svm_linear':base=SVC(kernel='linear',C=1.,tol=.001)
    elif name=='svm_rbf':base=SVC(kernel='rbf',C=1.,gamma=1/x.shape[1],tol=.001)
    elif name=='ridge':base=RidgeClassifier(alpha=1.,solver='cholesky')
    else:base=ElasticNetClassifier()
    return SubjectSigmoidClassifier(base).fit(x,y,sample_weight=w,groups=groups)
