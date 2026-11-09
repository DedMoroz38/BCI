import numpy as np
from analyze_single import independently_score

def test_metrics_probability_and_cell_macro():
    y=np.array([0,1,2,0,1,2]);p=np.eye(3)[y]*.7+.1
    a=dict(y=y,proba=p,pred=y,subject=np.repeat(['a','b'],3),session=np.repeat('1',6),context=np.repeat('Graz',6))
    score=independently_score(a)
    np.testing.assert_allclose(score['cell_mean'],1)
    np.testing.assert_allclose(score['log_loss'],-np.log(.8))
    np.testing.assert_allclose(score['brier'],.06)
