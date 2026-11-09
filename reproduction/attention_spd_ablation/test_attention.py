import numpy as np
import pytest
import torch
from attention import AttentionSPD, log_spd, train_pack

torch.set_num_threads(1)

@pytest.mark.parametrize('mode',['attention_gram','attention_exp'])
def test_spd_independent_qk_and_gradients(mode):
    model=AttentionSPD(mode,[0,1],channels=6,samples=20)
    x=torch.randn(2,4,6,20)
    assert model.wq.data_ptr()!=model.wk.data_ptr()
    assert not torch.equal(model.wq,model.wk)
    r=model(x)
    assert r.shape==(2,4,6,6)
    torch.testing.assert_close(r,r.transpose(-1,-2),atol=1e-6,rtol=1e-6)
    assert torch.linalg.eigvalsh(r).min()>0
    torch.testing.assert_close(model.log_matrix(x),log_spd(r),atol=1e-5,rtol=1e-5)
    model.log_matrix(x).square().sum().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.wq.grad.abs().sum()>0 and model.wk.grad.abs().sum()>0
    constant=torch.zeros_like(x)
    assert torch.linalg.eigvalsh(model(constant)).min()>0
    model.zero_grad();model.log_matrix(constant).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())

def test_log_repeated_eigenvalues_and_gradcheck():
    eye=torch.eye(4,dtype=torch.float64,requires_grad=True)
    log_spd(eye).sum().backward()
    torch.testing.assert_close(eye.grad,torch.ones_like(eye))
    a=torch.randn(3,3,dtype=torch.float64)
    r=(a@a.T+torch.eye(3)).requires_grad_()
    assert torch.autograd.gradcheck(log_spd,(r,),atol=1e-5,rtol=1e-3)

@pytest.mark.parametrize('mode',['attention_gram','attention_exp'])
def test_pack_tree_independence_and_roundtrip(mode):
    a=AttentionSPD(mode,[7,8],channels=4,samples=12)
    b=AttentionSPD(mode,[8],channels=4,samples=12)
    x=torch.randn(2,3,4,12)
    torch.testing.assert_close(a(x)[1],b(x[1:])[0])
    restored=AttentionSPD(mode,[7,8],channels=4,samples=12)
    restored.load_state_dict(a.state_dict())
    torch.testing.assert_close(a(x),restored(x))

def test_zero_weight_training_windows_excluded():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(1,18,4,12)).astype('float32');y=np.tile(np.arange(3),6)
    weights=np.ones((1,18));weights[:,12:]=0
    kwargs=dict(mode='attention_exp',tree_ids=[1],device='cpu',epochs=3,batch_size=6)
    a=train_pack(x,y,weights,**kwargs)
    modified=x.copy();modified[:,12:]*=10000
    changed=y.copy();changed[12:]=(changed[12:]+1)%3
    b=train_pack(modified,changed,weights,**kwargs)
    for key in a['model'].state_dict():
        torch.testing.assert_close(a['model'].state_dict()[key],b['model'].state_dict()[key],rtol=0,atol=0)
    assert all(np.isfinite(v['objective']) for v in a['history'])
    assert any((a['model'].wq.detach()-AttentionSPD('attention_exp',[1],channels=4,samples=12).wq).flatten()!=0)


def test_zero_weight_pairs_inside_mixed_pack_excluded():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(2,18,4,12)).astype('float32');y=np.tile(np.arange(3),6)
    weights=np.ones((2,18));weights[0,12:]=0
    kwargs=dict(mode='attention_gram',tree_ids=[1,2],device='cpu',epochs=3,batch_size=6)
    a=train_pack(x,y,weights,**kwargs)
    modified=x.copy();modified[0,12:]*=10000
    b=train_pack(modified,y,weights,**kwargs)
    for key in a['model'].state_dict():
        torch.testing.assert_close(a['model'].state_dict()[key],b['model'].state_dict()[key],rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA comparison runs on Kaggle')
def test_cpu_eigh_offload_matches_cuda_forward_and_gradients():
    a=torch.randn(3,6,8,device='cuda',dtype=torch.float64)
    r=(a@a.transpose(-1,-2)+torch.eye(6,device='cuda')*.1).requires_grad_()
    vals,vec=torch.linalg.eigh(r)
    direct=(vec*vals.log().unsqueeze(-2))@vec.transpose(-1,-2)
    actual=log_spd(r)
    torch.testing.assert_close(actual,direct,atol=1e-9,rtol=1e-9)
    g=torch.randn_like(r)
    old=torch.autograd.grad((direct*g).sum(),r,retain_graph=True)[0]
    new=torch.autograd.grad((actual*g).sum(),r)[0]
    torch.testing.assert_close(new,old,atol=1e-8,rtol=1e-8)
