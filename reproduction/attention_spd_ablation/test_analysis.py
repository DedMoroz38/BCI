import numpy as np
from analyze_attention import probability_metrics

def test_independent_probability_metrics():
    y=np.array([0,1,2]);p=np.array([[.8,.1,.1],[.2,.5,.3],[.2,.2,.6]])
    r=probability_metrics(y,p)
    np.testing.assert_allclose(r['log_loss'],-np.log([.8,.5,.6]).mean())
    np.testing.assert_allclose(r['brier'],np.square(p-np.eye(3)).sum(1).mean())
