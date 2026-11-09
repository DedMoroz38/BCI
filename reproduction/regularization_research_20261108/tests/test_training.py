import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
import numpy as np
import torch
import training as t
import wave1_training as old
from configurations import BY_NAME,BASE
from bci.data import synthetic


@pytest.fixture
def windows():
    torch.set_num_threads(1)
    w=synthetic(n_subjects=3,multi_session=0,runs=2,per_run=6)
    tr=w.select(w.subject!='sub-2');va=w.select(w.subject=='sub-2')
    x,v,_=old.aligned_pair(tr,va)
    return tr,va,torch.as_tensor(x),torch.as_tensor(v)


def test_clean_control_training_matches_frozen_wave1(windows,tmp_path):
    tr,va,x,v=windows
    a,ar=old.fit(x,tr,dict(n_filters=16,loss='class'),7,tmp_path/'old','cpu',val_x=v,val_windows=va,epochs=2)
    b,br=t.fit(x,tr,dict(BASE,name='unit_test_clean'),7,tmp_path/'new','cpu',val_x=v,val_windows=va,epochs=2)
    assert ar['best_epoch']==br['best_epoch']
    for k,value in a.state_dict().items():assert torch.equal(value,b.state_dict()[k]),k


@pytest.mark.parametrize('name',['ls05_noise03_drop3','mixstyle0.1','window3','ema0.95','swad0.05'])
def test_regularized_fit_refit_save_and_replay(windows,tmp_path,name):
    tr,va,x,v=windows;cfg=BY_NAME[name]
    net,r=t.fit(x,tr,cfg,0,tmp_path/'selection','cpu',val_x=v,val_windows=va,epochs=3)
    assert r['best_epoch']>0 and r['refit_epochs']>0
    assert r['averaging_epochs'] and max(r['averaging_epochs'])<=r['stopped_epoch']
    refit,rr=t.fit(x,tr,cfg,0,tmp_path/'refit','cpu',epochs=r['refit_epochs'],fixed_average_epochs=r['averaging_epochs'])
    again,ar=t.fit(x,tr,cfg,0,tmp_path/'refit','cpu',epochs=r['refit_epochs'],fixed_average_epochs=r['averaging_epochs'])
    np.testing.assert_array_equal(old.logits(refit,v),old.logits(again,v))
    assert rr['averaging_epochs']==r['averaging_epochs']


@pytest.mark.parametrize('name',['ls05_noise03_drop3','mixstyle0.1','ema0.95','swad0.05'])
def test_interrupted_training_resumes_rng_optimizer_and_average(windows,tmp_path,monkeypatch,name):
    tr,va,x,v=windows;cfg=BY_NAME[name]
    full,r=t.fit(x,tr,cfg,7,tmp_path/'full','cpu',val_x=v,val_windows=va,epochs=3)
    original=t.atomic_save
    def interrupt(value,path):
        original(value,path)
        if Path(path).name=='latest.pt' and value['epoch']==1:raise RuntimeError('interrupt')
    monkeypatch.setattr(t,'atomic_save',interrupt)
    with pytest.raises(RuntimeError,match='interrupt'):
        t.fit(x,tr,cfg,7,tmp_path/'partial','cpu',val_x=v,val_windows=va,epochs=3)
    monkeypatch.setattr(t,'atomic_save',original)
    resumed,rr=t.fit(x,tr,cfg,7,tmp_path/'partial','cpu',val_x=v,val_windows=va,epochs=3)
    assert r['averaging_epochs']==rr['averaging_epochs']
    for k,value in full.state_dict().items():assert torch.equal(value,resumed.state_dict()[k]),k


def test_expired_budget_cannot_start_training(windows,tmp_path):
    tr,va,x,v=windows
    with pytest.raises(old.BudgetStop):t.fit(x,tr,BY_NAME['ls0.05'],0,tmp_path,'cpu',deadline=time.time()-1)
    assert not (tmp_path/'complete.json').exists()
