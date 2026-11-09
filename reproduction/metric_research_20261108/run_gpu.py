"""Budgeted two-GPU nested-person CNN experiments; checkpointed per fit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,importlib.metadata,json,sys,time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context,current_process
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import ROOT,AUTHOR,sha,write_json,folds,inner_split,metadata,save_prediction,softmax
from neural_training import CONFIGS,aligned_pair,align_with_prior,fit,logits,new_net,BudgetStop
OUT=Path(os.environ.get('METRIC_OUTPUT',str(HERE/'experiments/neural_nested/results')))

def initialize():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((AUTHOR.parent/'SOURCE_RECEIPT.json').read_text())
    assert receipt['commit']=='934f423a73e1ed50d653345d27256ecfe50ab009'
    for name,digest in receipt['files'].items():assert sha(AUTHOR.parent/name)==digest
    files=list((HERE/'src').glob('*.py'))+[HERE/'run_gpu.py',HERE/'PROTOCOL.md']
    record=dict(code={str(p.relative_to(HERE)):sha(p) for p in files},configs=CONFIGS,
        data={s:sha(ROOT/'data'/f'proteus_{s}.npz') for s in ('train','val','test')},
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','torch')},
        gpu=[torch.cuda.get_device_name(k) for k in range(torch.cuda.device_count())],
        max_epochs=30,patience=6,seeds_primary=[0],seeds_optional=[1],inner_split_seed=33,
        cohort='all10_public',selection='inner_cell_weighted_NLL',author=receipt)
    p=OUT/'study-manifest.json'
    if p.exists() and json.loads(p.read_text())!=record:raise RuntimeError('Changed study sources/inputs')
    if not p.exists():write_json(p,record)

def worker(args):
    tag,seed,names,deadline=args
    torch.set_num_threads(2)
    identity=current_process()._identity;gpu=((identity[0]-1) if identity else 0)%torch.cuda.device_count()
    device='cuda:'+str(gpu);torch.cuda.set_device(gpu)
    pending=[name for name in names if not (OUT/'models'/tag/name/f'seed{seed}'/'prediction.json').exists()]
    if not pending:return dict(tag=tag,seed=seed,done=0,pending=[])
    if time.time()>deadline-600:return dict(tag=tag,seed=seed,done=0,pending=pending)
    _,train,test=next(f for f in folds('all10') if f[0]==tag)
    a,b=inner_split(train.subject);inner_train=train.select(a);inner_val=train.select(b)
    assert not set(inner_train.subject)&set(inner_val.subject)
    assert not set(train.stream)&set(test.stream)
    if tag.startswith('loso_'):assert not set(train.subject)&set(test.subject)
    print('NEURAL_FOLD',tag,'seed',seed,'GPU',gpu,'inner_people',len(set(inner_train.subject)),len(set(inner_val.subject)),flush=True)
    with threadpool_limits(limits=1,user_api='blas'):
        xi,xv,inner_prior=aligned_pair(inner_train,inner_val)
        xt,xe,outer_prior=aligned_pair(train,test)
        xi=torch.as_tensor(xi,device=device);xv=torch.as_tensor(xv,device=device)
        xt=torch.as_tensor(xt,device=device);xe=torch.as_tensor(xe,device=device)
        done=0;remaining=[]
        for name in pending:
            if time.time()>deadline-600:remaining.append(name);continue
            folder=OUT/'models'/tag/name/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            try:
                selection_net,selection=fit(xi,inner_train,CONFIGS[name],seed,folder/'selection',device,val_x=xv,val_windows=inner_val,epochs=30,deadline=deadline-30)
                del selection_net
                net,refit=fit(xt,train,CONFIGS[name],seed,folder/'refit',device,epochs=selection['best_epoch'],deadline=deadline-30)
            except BudgetStop:
                remaining.extend(pending[pending.index(name):]);break
            z=logits(net,xe);p=softmax(z,axis=1)
            np.save(folder/'logits.npy',z)
            torch.save(dict(config=net.config,state={k:v.detach().cpu() for k,v in net.state_dict().items()}),folder/'model.pt')
            np.savez(folder/'alignment.npz',cov_prior=outer_prior,prior_weight=5,band=[1,58])
            restored=new_net(CONFIGS[name],seed,device)
            restored.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state']);restored.eval()
            for first,second in zip(net.state_dict().values(),restored.state_dict().values()):assert torch.equal(first,second)
            with np.load(folder/'alignment.npz') as alignment:
                rebuilt=align_with_prior(test,alignment['cov_prior'])
            replay=softmax(logits(restored,torch.as_tensor(rebuilt,device=device)),axis=1)
            np.testing.assert_allclose(p,replay,rtol=0,atol=1e-4);np.testing.assert_array_equal(p.argmax(1),replay.argmax(1))
            record=save_prediction(folder/'prediction.npz',test,p,tag=tag,config=name,seed=seed,
                best_epoch=selection['best_epoch'],parameters=refit['parameters'],study_sha256=sha(OUT/'study-manifest.json'),
                model_sha256=sha(folder/'model.pt'),alignment_sha256=sha(folder/'alignment.npz'),
                logits_sha256=sha(folder/'logits.npy'),
                train_people=sorted(set(train.subject)),inner_train_people=sorted(set(inner_train.subject)),inner_val_people=sorted(set(inner_val.subject)))
            print('NEURAL_DONE',tag,name,seed,'epochs',selection['best_epoch'],'cell',record['cell_mean'],flush=True)
            del net,restored;done+=1;torch.cuda.empty_cache()
    return dict(tag=tag,seed=seed,done=done,pending=remaining)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=float,default=75);a=parser.parse_args()
    initialize();start=time.time();deadline=start+a.minutes*60
    tags=[tag for tag,_,_ in folds('all10')];state=[]
    primary=[(tag,0,list(CONFIGS),deadline) for tag in tags]
    workers=min(2,torch.cuda.device_count())
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:state+=list(pool.map(worker,primary))
    if not any(r['pending'] for r in state) and time.time()<deadline-900:
        secondary=[(tag,1,['shallow16_class','shallow16_subject'],deadline) for tag in tags]
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:state+=list(pool.map(worker,secondary))
    write_json(OUT/'execution.json',dict(seconds=time.time()-start,budget_seconds=a.minutes*60,jobs=state,budget_pending=[r for r in state if r['pending']]))
    import subprocess
    subprocess.run([sys.executable,str(HERE/'summarize.py'),'--root',str(OUT)],check=True)

if __name__=='__main__':main()
