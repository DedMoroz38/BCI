"""Behavioral tests: detect future leakage, label access, and wrong nuclear objective."""
import unittest
from pathlib import Path
import sys
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0,str(Path(__file__).resolve().parent))
import synergy as s


def meta(n, person='a'):
    return dict(subject=np.repeat(person,n),session=np.repeat('1',n),run=np.zeros(n),onset=np.arange(n),context=np.repeat('ctx',n),y=np.arange(n)%3)


def covariances(n,d=3):
    a=np.random.default_rng(72).normal(size=(n,d,d))
    return a@a.transpose(0,2,1)+np.eye(d)

class Causality(unittest.TestCase):
    def test_future_covariance_cannot_change_prefix(self):
        c=covariances(12);m=meta(12);prior=np.zeros((3,3))
        expected=s.causal_features(c[:5],{k:v[:5] for k,v in m.items()},prior)
        changed=c.copy();changed[5:]*=100
        actual=s.causal_features(changed,m,prior)[:5]
        np.testing.assert_allclose(actual,expected,atol=1e-12)

    def test_target_labels_never_enter_transform(self):
        c=covariances(12);m=meta(12);prior=np.eye(3)*.1
        expected=s.causal_features(c,m,prior)
        m['y'][:]=-999
        np.testing.assert_allclose(s.causal_features(c,m,prior),expected,atol=1e-12)

    def test_target_changes_do_not_alter_source_features_or_saved_state(self):
        from sklearn.preprocessing import StandardScaler
        source=covariances(15);target=covariances(10);sm=meta(15);tm=meta(10,'b')
        prior=s.spectral(source,'log').mean(0)
        raw=s.causal_features(source,sm,prior);scaler=StandardScaler().fit(raw)
        before=(prior.copy(),scaler.mean_.copy(),scaler.scale_.copy())
        a,b=s.transform_pair(source,target,sm,tm,prior,scaler)
        target*=55;tm['y'][:]=-123
        aa,bb=s.transform_pair(source,target,sm,tm,prior,scaler)
        np.testing.assert_allclose(a,aa,atol=1e-12)
        self.assertGreater(np.max(abs(b-bb)),1e-3)
        for old,new in zip(before,(prior,scaler.mean_,scaler.scale_)):np.testing.assert_array_equal(old,new)

class Objective(unittest.TestCase):
    def test_nuclear_fit_improves_full_penalized_objective(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(120,6));y=np.argmax(x[:,:3],axis=1);w=np.ones(120)
        init=dict(coef=np.zeros((3,6)),intercept=np.zeros(3),weight_sum=120.)
        model=s.StructuredLR(3,nuclear=.05,iterations=250).fit(x,y,w,init)
        def objective(coef,intercept,tau):
            logits=x@coef.T+intercept
            return np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])+np.sum(coef**2)/(2*.01*120)+tau*np.linalg.svd(s.unupper(coef,3),compute_uv=False).sum()
        self.assertLess(objective(model.coef,model.intercept,.05),objective(init['coef'],init['intercept'],.05)-.01)
        self.assertTrue(np.isfinite(model.predict_proba(x)).all())
        strong=s.StructuredLR(3,nuclear=.3,iterations=250).fit(x,y,w,init)
        weaknorm=np.linalg.svd(s.unupper(model.coef,3),compute_uv=False).sum()
        strongnorm=np.linalg.svd(s.unupper(strong.coef,3),compute_uv=False).sum()
        self.assertLess(strongnorm,weaknorm)

if __name__=='__main__':unittest.main(verbosity=2)
