import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
import numpy as np
import torch
import pytest,time
import neural_training as nt

def test_early_stop_keeps_best_epoch_not_last():
    s=nt.Stopper(patience=2)
    assert s.update(1.,1)
    assert s.update(.8,2)
    assert not s.update(.9,3) and not s.stopped
    assert not s.update(.85,4) and s.stopped and s.best_epoch==2

def test_compact_network_is_smaller_and_save_roundtrip(tmp_path):
    torch.set_num_threads(1)
    large=nt.new_net(nt.CONFIGS['shallow40_class'],0,'cpu')
    small=nt.new_net(nt.CONFIGS['shallow16_class'],0,'cpu').eval()
    assert sum(p.numel() for p in small.parameters())<sum(p.numel() for p in large.parameters())/4
    x=torch.randn(3,41,480)
    with torch.no_grad():p=small(x).numpy()
    torch.save(small.state_dict(),tmp_path/'weights.pt')
    restored=nt.new_net(nt.CONFIGS['shallow16_class'],4,'cpu').eval()
    restored.load_state_dict(torch.load(tmp_path/'weights.pt',weights_only=True))
    with torch.no_grad():q=restored(x).numpy()
    np.testing.assert_array_equal(p,q)

def test_training_checkpoint_and_selection_smoke(tmp_path):
    from bci.data import synthetic
    torch.set_num_threads(1)
    w=synthetic(n_subjects=3,multi_session=0,runs=2,per_run=6)
    tr=w.select(w.subject!='sub-2');va=w.select(w.subject=='sub-2')
    a,b,_=nt.aligned_pair(tr,va)
    net,record=nt.fit(torch.as_tensor(a),tr,nt.CONFIGS['shallow16_subject'],0,tmp_path,'cpu',val_x=torch.as_tensor(b),val_windows=va,epochs=2)
    assert record['best_epoch'] in (1,2)
    assert (tmp_path/'latest.pt').exists() and (tmp_path/'curves.csv').exists()
    restored,again=nt.fit(torch.as_tensor(a),tr,nt.CONFIGS['shallow16_subject'],0,tmp_path,'cpu',val_x=torch.as_tensor(b),val_windows=va,epochs=2)
    np.testing.assert_array_equal(nt.logits(net,torch.as_tensor(b)),nt.logits(restored,torch.as_tensor(b)))

def test_expired_budget_does_not_start_training(tmp_path):
    from bci.data import synthetic
    w=synthetic(n_subjects=3,multi_session=0,runs=2,per_run=6)
    with pytest.raises(nt.BudgetStop):
        nt.fit(torch.as_tensor(w.X),w,nt.CONFIGS['shallow16_class'],0,tmp_path,'cpu',epochs=1,deadline=time.time()-1)
    assert not (tmp_path/'complete.json').exists()

def test_interrupted_optimizer_rng_resume_matches_full_fit(tmp_path,monkeypatch):
    from bci.data import synthetic
    torch.set_num_threads(1)
    w=synthetic(n_subjects=3,multi_session=0,runs=2,per_run=6)
    tr=w.select(w.subject!='sub-2');va=w.select(w.subject=='sub-2')
    a,b,_=nt.aligned_pair(tr,va);x=torch.as_tensor(a);v=torch.as_tensor(b);cfg=nt.CONFIGS['shallow16_class']
    full,_=nt.fit(x,tr,cfg,7,tmp_path/'full','cpu',val_x=v,val_windows=va,epochs=2)
    original=nt.atomic_save
    def interrupt_after_checkpoint(value,path):
        original(value,path)
        if Path(path).name=='latest.pt' and value['epoch']==1:raise RuntimeError('simulated interruption')
    monkeypatch.setattr(nt,'atomic_save',interrupt_after_checkpoint)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        nt.fit(x,tr,cfg,7,tmp_path/'resumed','cpu',val_x=v,val_windows=va,epochs=2)
    monkeypatch.setattr(nt,'atomic_save',original)
    resumed,_=nt.fit(x,tr,cfg,7,tmp_path/'resumed','cpu',val_x=v,val_windows=va,epochs=2)
    for p,q in zip(full.state_dict().values(),resumed.state_dict().values()):assert torch.equal(p,q)
