import numpy as np
import pytest
import torch

def make(kind,strength=.5):
    from novel import NovelNet
    torch.manual_seed(4)
    return NovelNet(dict(name=kind,kind=kind,alignment_strength=strength,
                        family='context',prior=10,scope='filter',early=True)).eval()

@pytest.mark.parametrize('kind',['spectral','covariance'])
def test_future_windows_cannot_change_earlier_features(kind):
    net=make(kind);x=torch.randn(7,41,480)
    y=net.latent(x)
    x2=x.clone();x2[4:]=30*torch.randn_like(x2[4:])
    torch.testing.assert_close(net.latent(x2)[:4],y[:4],atol=3e-5,rtol=3e-5)
    torch.testing.assert_close(net.latent(x[:4]),y[:4],atol=3e-5,rtol=3e-5)

@pytest.mark.parametrize('kind',['spectral','covariance'])
def test_gradient_survives_real_signal_and_source_refresh(kind):
    net=make(kind);x=torch.randn(8,41,480)
    groups=[np.arange(4),np.arange(4,8)]
    net.refresh_source(x,groups,groups)
    loss=net(x[:4]).square().mean();loss.backward()
    grad=net.temporal.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())

def test_spectral_adjustment_preserves_perwindow_power_and_mean():
    net=make('spectral');h=torch.randn(5,16,468)
    diagonal=super(type(net),net).align_early(h)
    adjusted=net.align_early(h)
    torch.testing.assert_close(adjusted.mean(-1),diagonal.mean(-1),atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(adjusted.var(-1,unbiased=False),diagonal.var(-1,unbiased=False),atol=3e-5,rtol=3e-5)

@pytest.mark.parametrize('kind',['spectral','covariance'])
def test_zero_strength_retains_diagonal_backbone(kind):
    net=make(kind,0.);h=torch.randn(3,16,468)
    torch.testing.assert_close(net.align_early(h),super(type(net),net).align_early(h))

@pytest.mark.parametrize('kind',['spectral','covariance'])
def test_flat_and_nearly_collinear_channels_are_finite(kind):
    net=make(kind);h=torch.ones(4,16,468)
    h[:,0]+=1e-8*torch.randn(4,468)
    assert torch.isfinite(net.align_early(h)).all()

@pytest.mark.parametrize('kind',['spectral','covariance'])
def test_single_window_adapter_matches_session_prefix(kind):
    from novel import SessionPredictor
    net=make(kind);source=torch.randn(8,41,480)
    net.refresh_source(source,[np.arange(4),np.arange(4,8)],[np.arange(4),np.arange(4,8)])
    x=torch.randn(5,41,480)
    expected=net(x).detach()
    predictor=SessionPredictor(net)
    actual=torch.cat([predictor.predict_logits(a[None]) for a in x])
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
    predictor.reset_state()
    torch.testing.assert_close(predictor.predict_logits(x[:1]),expected[:1],atol=3e-5,rtol=3e-5)

def test_flat_spectral_input_has_finite_backward():
    net=make('spectral');h=torch.zeros(3,16,468,requires_grad=True)
    net.align_early(h).square().mean().backward()
    assert torch.isfinite(h.grad).all()
