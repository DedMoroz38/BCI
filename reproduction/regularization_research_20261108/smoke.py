"""Fresh-process CPU/CUDA gradient and durable fit checks before real training."""
import argparse,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from regularizers import new_net,augment,weighted_loss
from configurations import CONFIGS,BY_NAME
from training import aligned_pair,fit,logits
from runner_support import replay_diagnostics
from bci.data import synthetic
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--out',type=Path,default=Path('/tmp/proteus-reg-smoke'));a=p.parse_args()
    torch.set_num_threads(2)
    for c in CONFIGS:
        net=new_net(c,1,a.device).train();g=torch.Generator(device=a.device).manual_seed(2)
        x=torch.randn(12,41,480,device=a.device);original=x.clone()
        y=torch.arange(12,device=a.device)%3;s=torch.arange(12,device=a.device)//3
        z=net(augment(x,y,s,c['augment'],g),y,s,g)
        loss=weighted_loss(z,y,torch.ones(12,device=a.device),c['ls']);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
        assert torch.equal(x,original)
    print('ALL35 GRADIENT SMOKES PASSED',a.device,flush=True)
    w=synthetic(n_subjects=3,multi_session=0,runs=2,per_run=6)
    tr=w.select(w.subject!='sub-2');va=w.select(w.subject=='sub-2');x,v,_=aligned_pair(tr,va)
    x=torch.as_tensor(x,device=a.device);v=torch.as_tensor(v,device=a.device)
    for name in ('ls05_noise03_drop3','window3','ema0.95','swad0.05'):
        _,r=fit(x,tr,BY_NAME[name],0,a.out/name/'selection',a.device,val_x=v,val_windows=va,epochs=3)
        net,record=fit(x,tr,BY_NAME[name],0,a.out/name/'refit',a.device,epochs=r['refit_epochs'],fixed_average_epochs=r['averaging_epochs'])
        restored,again=fit(x,tr,BY_NAME[name],0,a.out/name/'refit',a.device,epochs=r['refit_epochs'],fixed_average_epochs=r['averaging_epochs'])
        np.testing.assert_array_equal(logits(net,v),logits(restored,v))
        cpu=new_net(BY_NAME[name],0,'cpu');cpu.load_state_dict({k:t.detach().cpu() for k,t in net.state_dict().items()});cpu.eval()
        expected=logits(net,v);actual=logits(cpu,v.cpu())
        replay_diagnostics(expected,actual)
    print('REGULARIZED FIT/REFIT/RELOAD PASSED',flush=True)

if __name__=='__main__':main()
