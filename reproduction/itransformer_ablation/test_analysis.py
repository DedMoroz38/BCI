import numpy as np
from analyze_itransformer import rescore

def test_rescore_probability_and_macro_metrics():
    y=np.array([0,1,2,0,1,2]);p=np.eye(3)[y]*.7+.1
    a=dict(y=y,proba=p,pred=y,subject=np.repeat(['a','b'],3),session=np.repeat('1',6),context=np.repeat('Graz',6))
    r=rescore(a)
    np.testing.assert_allclose(r['cell_mean'],1)
    np.testing.assert_allclose(r['log_loss'],-np.log(.8))
    np.testing.assert_allclose(r['brier'],.06)
