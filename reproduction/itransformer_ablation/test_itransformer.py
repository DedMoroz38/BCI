import numpy as np
import torch
from classifier import EEGiTransformer,config,fit_scale

torch.set_num_threads(1)

def test_author_encoder_equivalence_and_no_lags():
    torch.manual_seed(0);m=EEGiTransformer().eval()
    x=torch.randn(3,41,480);fit_scale(m,x)
    normalized=(x-x.mean(-1,keepdim=True))/m.input_scale
    from classifier import AuthorModel
    author=AuthorModel(config()).eval()
    author.enc_embedding.load_state_dict(m.enc_embedding.state_dict())
    author.encoder.load_state_dict(m.encoder.state_dict())
    captured=[]
    hook=author.encoder.register_forward_hook(lambda module,args,out:captured.append(out[0]))
    author.forecast(normalized.transpose(1,2),None,None,None)
    expected=captured[0];hook.remove()
    actual=m.hidden(x)
    assert actual.shape==(3,41,64)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert m(x).shape==(3,3)
    att=m.encoder.attn_layers[0].attention
    assert att.query_projection.weight.data_ptr()!=att.key_projection.weight.data_ptr()
    assert config().seq_len==480 and config().e_layers==1
    assert config().n_heads==4 and config().d_ff==128

def test_eval_sample_independence_and_roundtrip(tmp_path):
    torch.manual_seed(0);m=EEGiTransformer().eval();x=torch.randn(4,41,480)
    fit_scale(m,x[:2]);scale=m.input_scale.clone()
    p=m(x)
    torch.testing.assert_close(p[:1],m(x[:1]),atol=1e-6,rtol=1e-6)
    changed=x.clone();changed[1:]*=100
    torch.testing.assert_close(m(changed)[:1],p[:1],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(m.input_scale,scale)
    path=tmp_path/'m.pt';torch.save(m.state_dict(),path)
    restored=EEGiTransformer().eval();restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(p,restored(x),atol=0,rtol=0)

def test_encoder_and_head_receive_finite_gradients():
    m=EEGiTransformer();x=torch.randn(6,41,480);fit_scale(m,x)
    torch.nn.functional.cross_entropy(m(x),torch.arange(6)%3).backward()
    for p in m.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.enc_embedding.value_embedding.weight.grad.abs().sum()>0
    assert m.projector.weight.grad.abs().sum()>0


import pytest
@pytest.mark.skipif(not torch.cuda.is_available(),reason='Kaggle CUDA smoke')
def test_cuda_forward_backward_and_eval():
    m=EEGiTransformer().cuda();x=torch.randn(64,41,480,device='cuda');fit_scale(m,x)
    torch.nn.functional.cross_entropy(m(x),torch.arange(64,device='cuda')%3).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    torch.testing.assert_close(m(x[:1]),m(x)[:1],atol=1e-5,rtol=1e-5)
