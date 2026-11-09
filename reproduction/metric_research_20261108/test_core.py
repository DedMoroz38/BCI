import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
import pytest
import common as c

def test_subject_weights_equalize_people_and_classes():
    y=np.array([0,1,2,0,1,2,0,0,1,2]);s=np.array(['a']*3+['b']*7)
    w=c.loss_weights(y,s,'subject')
    assert w.mean()==pytest.approx(1)
    for person in ('a','b'):
        for label in range(3):assert w[(s==person)&(y==label)].sum()==pytest.approx(len(y)/6)

def test_mixture_preserves_simplex_and_temperature_argmax():
    p=np.array([[.8,.1,.1],[.1,.7,.2]])
    z=np.array([[8.,-2.,0.],[-3.,5.,2.]])
    for temp in (1,2,4):
        q=c.softmax(z/temp);np.testing.assert_array_equal(q.argmax(1),z.argmax(1))
        for pool in ('arithmetic','geometric'):
            r=c.mix(p,q,.3,pool);np.testing.assert_allclose(r.sum(1),1);assert np.all(r>=0)
    np.testing.assert_allclose(c.mix(p,c.softmax(z),0,'arithmetic'),p)

def test_causal_balance_does_not_use_future_or_other_stream():
    p=np.array([[.8,.1,.1],[.6,.3,.1],[.1,.2,.7]])
    out=c.balance(p,[np.arange(2),np.arange(2,3)],1)
    np.testing.assert_array_equal(out[:1],c.balance(p[:1],[np.arange(1)],1))
    np.testing.assert_array_equal(out[2:],c.balance(p[2:],[np.arange(1)],1))

def test_all10_protocol_and_inner_people():
    jobs=c.folds('all10')
    assert len(jobs)==11
    for _,tr,te in jobs[:10]:
        assert len(set(tr.subject))==9 and not set(tr.subject)&set(te.subject)
        a,b=c.inner_split(tr.subject)
        assert not set(tr.subject[a])&set(tr.subject[b])
    assert len(set(jobs[-1][2].subject))==2

def test_author_geometric_pool_preserves_probability_floor():
    import local_fusion as lf
    p=np.array([[.4,.3,.3]]);q=np.array([[1e-20,.1,.9]])
    z=.7*np.log(p+1e-9)+.3*np.log(q+1e-9)
    np.testing.assert_allclose(lf.author_geo(p,q),c.softmax(z,axis=1),atol=1e-14)
