import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'Brain-Decoding/track2_bci')]
from single_e2e import exp_symmetric,session_tangents,SingleE2E,fit_single
from bci.riemann import SessionRecentering

torch.set_num_threads(1)

def test_exp_repeated_eigenvalues_and_gradcheck():
    z=torch.zeros(3,3,dtype=torch.float64,requires_grad=True)
    out=exp_symmetric(z);torch.testing.assert_close(out,torch.eye(3,dtype=torch.float64))
    out.sum().backward();torch.testing.assert_close(z.grad,torch.ones_like(z))
    a=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(exp_symmetric,(a,),atol=1e-5,rtol=1e-4)

def test_causal_tangents_match_original_and_past_gradient():
    torch.manual_seed(0)
    a=torch.randn(8,4,6,dtype=torch.float64)
    r=a@a.transpose(-1,-2)+.1*torch.eye(4)
    from single_e2e import log_spd
    lr=log_spd(r).detach().requires_grad_();prior=lr.detach().mean(0)
    streams=[np.arange(4),np.arange(4,8)]
    actual=session_tangents(r,lr,prior,streams)
    expected=[]
    for idx in streams:
        state=SessionRecentering(prior.numpy(),5)
        expected.extend([state(r[i].numpy())[0] for i in idx])
    np.testing.assert_allclose(actual.detach().numpy(),expected,atol=1e-10,rtol=1e-9)
    actual[2].square().sum().backward()
    assert lr.grad[0].abs().sum()>0
    assert lr.grad[3:].abs().sum()==0
    prefix=session_tangents(r[:3],lr[:3],prior,[np.arange(3)])
    torch.testing.assert_close(prefix,actual[:3])
    reset=session_tangents(r[4:],lr[4:],prior,[np.arange(4)])
    torch.testing.assert_close(reset,actual[4:])

@pytest.mark.parametrize('mode',['attention_gram','attention_exp'])
def test_full_gradient_through_prior_scaler_and_final_head(mode):
    torch.manual_seed(1)
    model=SingleE2E(mode,channels=4,samples=12)
    x=torch.randn(12,4,12);streams=[np.arange(6),np.arange(6,12)]
    logits,stats=model.train_logits(x,streams)
    stats['prior'].retain_grad()
    loss=torch.nn.functional.cross_entropy(logits,torch.arange(12)%3);loss.backward()
    assert model.encoder.wq.data_ptr()!=model.encoder.wk.data_ptr()
    for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    assert model.encoder.wq.grad.abs().sum()>0
    assert model.encoder.wk.grad.abs().sum()>0
    assert model.head.weight.grad.abs().sum()>0
    assert stats['prior'].grad.abs().sum()>0

@pytest.mark.parametrize('mode',['attention_gram','attention_exp'])
def test_fit_roundtrip_streaming_and_retained_head(mode,tmp_path):
    rng=np.random.default_rng(0)
    x=rng.normal(size=(18,4,12)).astype('float32');y=np.tile(np.arange(3),6)
    streams=[np.arange(9),np.arange(9,18)]
    result=fit_single(x,y,streams,mode=mode,device='cpu',epochs=3)
    model=result['model'];assert len(result['history'])==3
    p=model.predict_proba(x,streams)
    assert p.shape==(18,3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1),1)
    torch.testing.assert_close(model.head.weight.detach().cpu(),result['retained_head_weight'])
    assert not torch.equal(result['initial_head_weight'],result['retained_head_weight'])
    path=tmp_path/'single.pt';torch.save(model.state_dict(),path)
    restored=SingleE2E(mode,channels=4,samples=12)
    restored.load_state_dict(torch.load(path,weights_only=True))
    np.testing.assert_allclose(restored.predict_proba(x,streams),p,atol=1e-10)
    np.testing.assert_allclose(p[:3],model.predict_proba(x[:3],[np.arange(3)]),atol=1e-10)
    np.testing.assert_allclose(p[9:],model.predict_proba(x[9:],[np.arange(9)]),atol=1e-10)
    with torch.no_grad():r=model.encoder(torch.as_tensor(x)[None])[0].cpu().numpy()
    from scipy.special import softmax
    for idx in streams:
        state=SessionRecentering(model.log_prior.cpu().numpy(),5)
        for i in idx:
            f=(state(r[i])[0]-model.feature_mean.cpu().numpy())/model.feature_scale.cpu().numpy()
            q=softmax(f@model.head.weight.detach().cpu().numpy().T+model.head.bias.detach().cpu().numpy())
            np.testing.assert_allclose(q,p[i],atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Kaggle CUDA full-shape check')
def test_cuda_full_shape_end_to_end_gradients():
    model=SingleE2E('attention_gram').cuda()
    x=torch.randn(1800,82,476,device='cuda')
    logits,_=model.train_logits(x,[np.arange(900),np.arange(900,1800)])
    torch.nn.functional.cross_entropy(logits,torch.arange(1800,device='cuda')%3).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder.wq.grad.abs().sum()>0
