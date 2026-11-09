import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
import torch
import torch.nn.functional as F
import regularizers as r

def test_zero_smoothing_matches_weighted_baseline_and_positive_smoothing():
    z=torch.tensor([[4.,1.,-1.],[-2.,3.,.5]],requires_grad=True);y=torch.tensor([0,1]);w=torch.tensor([.5,2.])
    assert torch.equal(r.weighted_loss(z,y,w,0),(F.cross_entropy(z,y,reduction='none')*w).mean())
    q=.1/3+F.one_hot(y,3)*.9
    expected=(-(q*z.log_softmax(1)).sum(1)*w).mean()
    torch.testing.assert_close(r.weighted_loss(z,y,w,.1),expected)

@pytest.mark.parametrize('config',[dict(noise=.03),dict(global_gain=.2),dict(coordinate_gain=.2),dict(drop=3),dict(mask=24),dict(shift=6),dict(flip=.5),dict(mix=.2)])
def test_augmentation_shape_reproducibility_and_no_mutation(config):
    x=torch.randn(12,41,480);original=x.clone();y=torch.arange(12)%3;s=torch.arange(12)//3
    a=r.augment(x,y,s,config,torch.Generator().manual_seed(9))
    b=r.augment(x,y,s,config,torch.Generator().manual_seed(9))
    assert a.shape==x.shape and torch.isfinite(a).all()
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(a,x)

def test_drop_exact_coordinates_mask_shared_across_coordinates():
    x=torch.ones(8,41,480);y=torch.arange(8)%3;s=torch.arange(8)
    a=r.augment(x,y,s,dict(drop=3),torch.Generator().manual_seed(1))
    assert torch.all((a.abs().sum(-1)==0).sum(1)==3)
    a=r.augment(x,y,s,dict(mask=24),torch.Generator().manual_seed(1))
    assert torch.all((a==0).sum(-1)==24)
    assert torch.equal(a[:,0],a[:,40])

def test_partners_same_class_other_person_or_identity():
    y=torch.tensor([0,0,1,1,2]);s=torch.tensor([0,1,0,1,0])
    p=r.partners(y,s,torch.Generator().manual_seed(1))
    assert torch.equal(y[p],y) and torch.all(s[p[:4]]!=s[:4]) and p[-1]==4

def test_clean_control_preserves_rng_and_author_network():
    import wave1_training as old
    torch.set_num_threads(1)
    a=old.new_net(dict(n_filters=16),4,'cpu').eval()
    b=r.new_net(dict(n_filters=16,augment={}),4,'cpu').eval()
    x=torch.randn(5,41,480)
    torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    g=torch.Generator().manual_seed(1);state=g.get_state().clone()
    assert r.augment(x,torch.arange(5)%3,torch.arange(5),{},g) is x
    assert torch.equal(g.get_state(),state)

def test_mixstyle_disabled_in_eval_and_single_person_training():
    config=dict(n_filters=16,augment=dict(style=.1));net=r.new_net(config,1,'cpu')
    x=torch.randn(6,41,480);y=torch.arange(6)%3;s=torch.zeros(6,dtype=torch.long)
    plain=r.new_net(dict(n_filters=16,augment={}),1,'cpu');plain.load_state_dict(net.state_dict())
    net.eval();plain.eval();torch.testing.assert_close(net(x,y,s),plain(x),rtol=0,atol=0)
    h=torch.ones(6,16,1,20)
    torch.testing.assert_close(r.mix_style(h,y,s,.1,torch.Generator().manual_seed(1)),h)

def test_parameter_averages_and_bn_recalibration_clean_only():
    a={'w':torch.tensor([1.,3.]),'counter':torch.tensor(2)};b={'w':torch.tensor([3.,5.]),'counter':torch.tensor(4)}
    avg=r.mean_states([a,b]);torch.testing.assert_close(avg['w'],torch.tensor([2.,4.]));assert avg['counter']==4
    ema=r.StateAverage('ema',.5);ema.update(a);ema.update(b)
    torch.testing.assert_close(ema.result()['w'],avg['w'])
    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bn=torch.nn.BatchNorm1d(2);self.drop=torch.nn.Dropout(.9)
        def forward(self,x):
            assert not self.drop.training
            return self.drop(self.bn(x))
    net=Probe().eval();state=torch.get_rng_state().clone();x=torch.full((8,2),7.)
    r.refresh_bn(net,x,batch=4)
    torch.testing.assert_close(net.bn.running_mean,torch.tensor([7.,7.]));assert not net.training
    assert torch.equal(torch.get_rng_state(),state)

def test_loss_valley_range_excludes_dead_tail_and_returns_last_if_no_convergence():
    v=r.LossValley(3,6,.05)
    for i,loss in enumerate([1.,.8,.7,.71,.72,.73,1.,1.1,1.2,1.3,1.4,1.5],1):v.update(i,loss)
    selected=v.final_epochs();assert selected[0]==3 and max(selected)<12 and v.dead
    v=r.LossValley(3,6,.05)
    for i,loss in enumerate([1.,.9,.8],1):v.update(i,loss)
    assert v.final_epochs()==[3]

def test_config_count_and_fixed_user_scope():
    from configurations import CONFIGS
    assert len(CONFIGS)==35 and len({c['name'] for c in CONFIGS})==35
    assert all(c['n_filters']==16 and c['loss']=='class' for c in CONFIGS)
    assert [c['stage'] for c in CONFIGS]==[1]*29+[2]*6
    assert not {'control','bn_refresh'} & {c['name'] for c in CONFIGS}

@pytest.mark.parametrize('losses',[[1.,.8,.7,.71,.72,.73,1.,1.1,1.2,1.3,1.4,1.5],[1.,.9,.8],[1.,1.01,1.02,1.03,1.04,1.05,1.06],[1.,.8,.7,.71,.72,.69,.68,.7,.73,.75]])
def test_loss_valley_matches_author_parameter_averaging(losses):
    from types import SimpleNamespace
    vendor=Path(__file__).resolve().parents[1]/'literature/vendor'
    namespace={};exec((vendor/'swa_utils.py').read_text(),namespace)
    class CPUAverage(namespace['AveragedModel']):
        def cuda(self,*args,**kwargs):return self
    source=(vendor/'swad.py').read_text().replace('from domainbed.lib import swa_utils','')
    namespace={'swa_utils':SimpleNamespace(AveragedModel=CPUAverage)};exec(source,namespace)
    evaluator=SimpleNamespace(logger=SimpleNamespace(error=lambda *args:None))
    oracle=namespace['LossValley'](evaluator,n_converge=3,n_tolerance=6,tolerance_ratio=.05)
    adapted=r.LossValley(3,6,.05)
    for epoch,loss in enumerate(losses,1):
        model=torch.nn.Linear(len(losses),1,bias=False)
        with torch.no_grad():model.weight.zero_();model.weight[0,epoch-1]=1.
        segment=CPUAverage(model);segment.start_step=segment.end_step=epoch
        oracle.update_and_evaluate(segment,0.,loss,lambda *args:None);adapted.update(epoch,loss)
    expected=oracle.get_final_model().module.weight.detach()
    epochs=adapted.final_epochs();actual=torch.zeros_like(expected)
    for epoch in epochs:actual[0,epoch-1]+=1/len(epochs)
    torch.testing.assert_close(actual,expected)
