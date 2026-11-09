"""Simple-first two-GPU regularization grid; no polling or remote services."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,fcntl,hashlib,importlib.metadata,json,sys,time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context,current_process
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import ROOT,AUTHOR,sha,write_json,inner_split,save_prediction
from configurations import CONFIGS,BY_NAME
from training import fit,aligned_pair,align_with_prior,logits,BudgetStop,new_net
from metrics import load,softmax
from runner_support import fold_tags,split_outer,schedule,validate_existing,replay_diagnostics
from bci.data import concat,load_split

OUT=Path(os.environ.get('REG_OUTPUT',str(HERE/'results')))
CACHE=Path(os.environ.get('REG_CACHE','/tmp/proteus-regularization-cache'))
_FULL=None;_PREPARED=None

def cohort():
    global _FULL
    if _FULL is None:_FULL=concat(*(load_split(s,ROOT/'data') for s in ('train','val','test')))
    return _FULL

def initialize():
    assert torch.cuda.is_available(),'CUDA required';OUT.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((AUTHOR.parent/'SOURCE_RECEIPT.json').read_text())
    assert receipt['commit']=='934f423a73e1ed50d653345d27256ecfe50ab009'
    for name,digest in receipt['files'].items():assert sha(AUTHOR.parent/name)==digest
    refs=json.loads((HERE/'reference/manifest.json').read_text())
    for name,digest in refs.items():assert sha(HERE/'reference'/name)==digest
    files=list((HERE/'src').glob('*.py'))+[HERE/'run_gpu.py',HERE/'runner_support.py',HERE/'PROTOCOL.md']
    manifest=dict(code={str(p.relative_to(HERE)):sha(p) for p in files},configs=CONFIGS,
        data={s:sha(ROOT/'data'/f'proteus_{s}.npz') for s in ('train','val','test')},
        references=refs,versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','scikit-learn')},
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        max_epochs=30,patience=6,seeds=[0,1],criterion='inner_cell_NLL',inner_seed=33)
    path=OUT/'study-manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:write_json(path,manifest)
    return fold_tags(cohort())

def prepare(tag,device):
    global _PREPARED
    if _PREPARED is not None and _PREPARED['tag']==tag:return _PREPARED
    _PREPARED=None;torch.cuda.empty_cache()
    train,test=split_outer(cohort(),tag);a,b=inner_split(train.subject)
    inner_train=train.select(a);inner_val=train.select(b)
    assert not set(inner_train.subject)&set(inner_val.subject)
    assert not set(train.stream)&set(test.stream)
    if tag.startswith('loso_'):assert not set(train.subject)&set(test.subject)
    cache=CACHE/tag;cache.mkdir(parents=True,exist_ok=True)
    expected=dict(study_sha256=sha(OUT/'study-manifest.json'),tag=tag)
    with (cache/'cache.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (cache/'ready.json').exists():
            xi,xv,inner_prior=aligned_pair(inner_train,inner_val);xt,xe,prior=aligned_pair(train,test)
            for key,value in dict(xi=xi,xv=xv,xt=xt,xe=xe,prior=prior,inner_prior=inner_prior).items():np.save(cache/(key+'.npy'),value)
            write_json(cache/'ready.json',expected)
        assert json.loads((cache/'ready.json').read_text())==expected
    prior=np.load(cache/'prior.npy');rebuilt=align_with_prior(test,prior)
    np.testing.assert_allclose(rebuilt,np.load(cache/'xe.npy',mmap_mode='r'),rtol=0,atol=1e-6)
    tensors={name:torch.as_tensor(np.load(cache/(name+'.npy')),device=device) for name in ('xi','xv','xt','xe')}
    _PREPARED=dict(tag=tag,train=train,test=test,inner_train=inner_train,inner_val=inner_val,
        prior=prior,replay_x=torch.as_tensor(rebuilt,device=device),replay_cpu=torch.as_tensor(rebuilt),**tensors)
    return _PREPARED

def worker_impl(args):
    name,seed,tag,deadline=args
    if time.time()>deadline-180:return dict(name=name,seed=seed,tag=tag,status='budget_pending')
    torch.set_num_threads(2);identity=current_process()._identity
    gpu=((identity[0]-1) if identity else 0)%torch.cuda.device_count();torch.cuda.set_device(gpu);device=f'cuda:{gpu}'
    folder=OUT/'models'/tag/name/f'seed{seed}'
    if (folder/'prediction.json').exists():
        previous=validate_existing(folder/'prediction.json')
        assert previous['study_sha256']==sha(OUT/'study-manifest.json')
        return dict(name=name,seed=seed,tag=tag,status='existing')
    with threadpool_limits(limits=1,user_api='blas'):
        p=prepare(tag,device);cfg=BY_NAME[name]
        try:
            net,selection=fit(p['xi'],p['inner_train'],cfg,seed,folder/'selection',device,
                val_x=p['xv'],val_windows=p['inner_val'],epochs=30,deadline=deadline-30)
            del net
            net,refit=fit(p['xt'],p['train'],cfg,seed,folder/'refit',device,
                epochs=selection['refit_epochs'],fixed_average_epochs=selection['averaging_epochs'],deadline=deadline-30)
        except BudgetStop:return dict(name=name,seed=seed,tag=tag,status='budget_pending')
        z=logits(net,p['xe']);prob=softmax(z,axis=1)
        np.save(folder/'logits.npy',z)
        torch.save(dict(config=net.config,regularization=cfg,state={k:v.detach().cpu() for k,v in net.state_dict().items()}),folder/'model.pt')
        np.savez(folder/'alignment.npz',cov_prior=p['prior'],prior_weight=5,band=[1,58])
        restored=new_net(cfg,seed,device)
        restored.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state']);restored.eval()
        replay=softmax(logits(restored,p['replay_x']),axis=1)
        for key,value in net.state_dict().items():
            assert torch.equal(value.detach().cpu(),restored.state_dict()[key].detach().cpu()),key
        gpu_diagnostic=replay_diagnostics(prob,replay)
        cpu_restored=new_net(cfg,seed,'cpu')
        cpu_restored.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state']);cpu_restored.eval()
        cpu_replay=softmax(logits(cpu_restored,p['replay_cpu']),axis=1)
        cpu_diagnostic=replay_diagnostics(prob,cpu_replay)
        r=save_prediction(folder/'prediction.npz',p['test'],prob,tag=tag,config=name,seed=seed,
            best_epoch=selection['best_epoch'],refit_epochs=selection['refit_epochs'],averaging_epochs=selection['averaging_epochs'],
            parameters=refit['parameters'],study_sha256=sha(OUT/'study-manifest.json'),
            cpu_replay_diagnostic=cpu_diagnostic,gpu_replay_diagnostic=gpu_diagnostic,
            model_sha256=sha(folder/'model.pt'),alignment_sha256=sha(folder/'alignment.npz'),logits_sha256=sha(folder/'logits.npy'),
            train_people=sorted(set(p['train'].subject)),inner_train_people=sorted(set(p['inner_train'].subject)),
            inner_val_people=sorted(set(p['inner_val'].subject)))
        print('REG_DONE',name,seed,tag,'GPU',gpu,'cell',round(r['cell_mean'],5),flush=True)
        del net,restored,cpu_restored
    return dict(name=name,seed=seed,tag=tag,status='complete')

def worker(args):
    try:return worker_impl(args)
    except Exception as e:
        import traceback
        name,seed,tag,_=args
        failure=dict(name=name,seed=seed,tag=tag,status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
        write_json(OUT/'models'/tag/name/f'seed{seed}'/'failure.json',failure)
        print('MODEL_FAILED_CONTINUING',name,seed,tag,type(e).__name__,str(e)[:250],flush=True)
        return {k:v for k,v in failure.items() if k!='traceback'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=float,default=240);args=parser.parse_args()
    tags=initialize();start=time.time();deadline=start+args.minutes*60;state=[]
    jobs=schedule(tags)
    with ProcessPoolExecutor(max_workers=min(2,torch.cuda.device_count()),mp_context=get_context('spawn')) as pool:
        state.extend(pool.map(worker,[(*j,deadline) for j in jobs]))
    write_json(OUT/'execution.json',dict(seconds=time.time()-start,budget_seconds=args.minutes*60,
        jobs=state,planned_jobs=len(jobs),failed_jobs=[j for j in state if j['status']=='failed'],
        budget_pending=[j for j in state if j['status']=='budget_pending']))
    import subprocess
    subprocess.run([sys.executable,str(HERE/'summarize.py'),'--root',str(OUT)],check=True)

if __name__=='__main__':main()
