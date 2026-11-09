import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import classifiers as c


def fixture_data():
    rng=np.random.default_rng(14)
    y=np.tile(np.repeat(np.arange(3),8),6)
    groups=np.repeat(np.arange(6),24)
    x=rng.normal(size=(len(y),8));x[:,:3]+=np.eye(3)[y]*1.2
    w=np.repeat([1,2,0,3,1,1],24)
    return x,y,groups,w


def test_effective_weights_match_actual_installed_logreg():
    x,y,g,w=fixture_data();keep=w>0;x,y,w=x[keep],y[keep],w[keep]
    # Deliberately unbalance one class to expose weighting convention errors.
    w=w.copy();w[y==0]*=2
    actual=LogisticRegression(C=.01,max_iter=2000,class_weight='balanced').fit(x,y,sample_weight=w)
    expected=LogisticRegression(C=.01,max_iter=2000).fit(x,y,sample_weight=c.balanced_weights(y,w))
    np.testing.assert_allclose(actual.coef_,expected.coef_,atol=1e-12,rtol=1e-12)
    for cl in range(3):
        assert np.isclose(c.balanced_weights(y,w)[y==cl].sum(),w.sum()/3)


def test_calibration_is_subject_isolated_and_complete():
    x,y,g,w=fixture_data();keep=w>0
    splits=c.calibration_splits(y[keep],g[keep])
    seen=np.zeros(keep.sum(),int)
    for train,test in splits:
        assert not set(g[keep][train])&set(g[keep][test])
        assert set(y[keep][train])=={0,1,2}
        seen[test]+=1
    np.testing.assert_array_equal(seen,1)


@pytest.mark.parametrize('name',[n for n in c.VARIANTS if n!='mlp'])
def test_every_classifier_preserves_simplex_and_class_order(name):
    x,y,g,w=fixture_data();keep=w>0
    model=c.fit_classifier(name,x[keep],y[keep],w[keep],g[keep])
    p=model.predict_proba(x)
    assert p.shape==(len(y),3)
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    np.testing.assert_array_equal(model.classes_,[0,1,2])


def test_calibration_requires_two_real_subjects():
    with pytest.raises(ValueError,match='subjects'):
        c.calibration_splits(np.tile([0,1,2],3),np.zeros(9))


def test_oof_scores_do_not_use_heldout_person_class_frequencies():
    x,y,g,w=fixture_data();keep=w>0;x,y,g,w=x[keep],y[keep],g[keep],w[keep]
    a=c.fit_classifier('ridge',x,y,w,g)
    changed=y.copy();changed[g==0]=0
    b=c.fit_classifier('ridge',x,changed,w,g)
    # Calibrator itself may use these labels; the underlying OOF score must not.
    np.testing.assert_allclose(a.oof_scores_[g==0],b.oof_scores_[g==0],atol=1e-12,rtol=0)
