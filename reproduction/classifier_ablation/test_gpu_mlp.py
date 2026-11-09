import numpy as np
import gpu_mlp


def test_gpu_mlp_cpu_smoke_and_zero_weight_invariance():
    rng=np.random.default_rng(0)
    x=rng.normal(size=(2,24,5));y=np.tile([0,1,2],8)
    weights=np.ones((2,24));weights[:,12:]=0
    eval_x=x[:,:4].copy()
    a=gpu_mlp.fit_predict(x,y,weights,eval_x,device='cpu',epochs=3)
    x[:,12:]+=100
    b=gpu_mlp.fit_predict(x,y,weights,eval_x,device='cpu',epochs=3)
    np.testing.assert_allclose(a['tree_proba'],b['tree_proba'],rtol=0,atol=0)
    p=a['tree_proba'];assert p.shape==(2,4,3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(-1),1,atol=1e-6)
    assert a['epochs']==3 and a['device']=='cpu'
