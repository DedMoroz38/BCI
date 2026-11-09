from pathlib import Path
import numpy as np
import torch
from training import subject_split,EarlyStopper,train_selection,refit,predict
torch.set_num_threads(1)

def test_inner_split_is_subject_disjoint_and_deterministic():
    groups=np.repeat(['a','b','c','d','e','f','g'],3)
    a,b=subject_split(groups);c,d=subject_split(groups)
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    assert not set(groups[a])&set(groups[b])
    assert len(a)+len(b)==len(groups)

def test_best_epoch_uses_only_validation_loss():
    e=EarlyStopper(patience=2,min_delta=.001)
    assert e.update(.9,1)
    assert e.update(.8,2)
    assert not e.update(.8001,3)
    assert not e.stopped
    assert not e.update(.81,4)
    assert e.stopped and e.best_epoch==2

def test_training_curves_and_fresh_refit(tmp_path):
    torch.set_num_threads(1)
    rng=np.random.default_rng(0)
    x=rng.normal(size=(18,41,480)).astype('float32');y=np.tile(np.arange(3),6)
    groups=np.repeat(['a','b','c'],6)
    selected=train_selection(x[:12],y[:12],groups[:12],x[12:],y[12:],groups[12:],
              seed=0,folder=tmp_path/'selection',device='cpu',max_epochs=2,patience=10)
    assert selected['best_epoch'] in [1,2]
    assert (tmp_path/'selection/curves.csv').exists()
    assert (tmp_path/'selection/curves.png').exists()
    assert (tmp_path/'selection/best.pt').exists()
    fitted=refit(x,y,groups,seed=0,epochs=selected['best_epoch'],folder=tmp_path/'refit',device='cpu')
    p=predict(fitted['model'],x,device='cpu')
    assert p.shape==(18,3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    assert len(fitted['history'])==selected['best_epoch']


def test_selected_checkpoint_restores_actual_validation_loss(tmp_path):
    from classifier import EEGiTransformer
    from training import class_weights,evaluate
    rng=np.random.default_rng(4)
    x=rng.normal(size=(15,41,480)).astype('float32');y=np.tile(np.arange(3),5)
    selected=train_selection(x[:9],y[:9],np.repeat('a',9),x[9:],y[9:],np.repeat('b',6),
                            seed=1,folder=tmp_path,device='cpu',max_epochs=2)
    m=EEGiTransformer();m.load_state_dict(torch.load(tmp_path/'best.pt',weights_only=True)['state'])
    measured=evaluate(m,torch.tensor(x[9:]),torch.tensor(y[9:]),np.repeat('b',6),class_weights(torch.tensor(y[:9])))
    np.testing.assert_allclose(measured['weighted_ce'],selected['best_validation_weighted_ce'],atol=1e-7)
    import pytest
    with pytest.raises(RuntimeError,match='configuration'):
        train_selection(x[:9],y[:9],np.repeat('a',9),x[9:],y[9:],np.repeat('b',6),seed=2,folder=tmp_path,device='cpu',max_epochs=2)
    p=tmp_path/'best.pt';p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(RuntimeError,match='artifact'):
        train_selection(x[:9],y[:9],np.repeat('a',9),x[9:],y[9:],np.repeat('b',6),seed=1,folder=tmp_path,device='cpu',max_epochs=2)
