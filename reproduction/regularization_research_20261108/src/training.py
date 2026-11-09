"""Unchanged inner-NLL stopping with training-only regularization policies."""
from copy import deepcopy
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from common import cells,loss_weights,sha,write_json
from wave1_training import (BudgetStop,Stopper,cpu_state,evaluate,logits,plot,
                            atomic_save,save_history,aligned_pair,align_with_prior)
from regularizers import (new_net,augment,weighted_loss,StateAverage,LossValley,
                          mean_states,refresh_bn)


def check_budget(deadline):
    if deadline is not None and time.time()>=deadline:raise BudgetStop('Wall budget reached;complete-epoch checkpoint preserved')


def candidate(net,state,x,deadline):
    check_budget(deadline)
    model=deepcopy(net);model.load_state_dict(state)
    refresh_bn(model,x)
    check_budget(deadline)
    return model


def fit(x,windows,cfg,seed,folder,device,*,val_x=None,val_windows=None,epochs=30,
        deadline=None,fixed_average_epochs=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    selecting=val_windows is not None;policy=cfg.get('average','none')
    settings=dict(config=cfg,seed=seed,epochs=epochs,selecting=selecting,
        people=sorted(set(windows.subject)),windows=len(windows),
        val_people=sorted(set(val_windows.subject)) if selecting else [],
        fixed_average_epochs=fixed_average_epochs,
        source={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('regularizers.py'))},
        lr=.001,decay=.01,batch=64,schedule_Tmax=100,criterion='cell_NLL')
    token=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    if (folder/'complete.json').exists():
        r=json.loads((folder/'complete.json').read_text())
        assert r['settings_hash']==token and sha(folder/'final.pt')==r['final_sha256']
        net=new_net(cfg,seed,device);net.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True))
        return net.eval(),r
    check_budget(deadline);net=new_net(cfg,seed,device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    shuffle=torch.Generator().manual_seed(seed)
    ag=torch.Generator(device=device).manual_seed(seed+89271)
    y=torch.as_tensor(windows.y,dtype=torch.long,device=device)
    _,groups=np.unique(windows.subject,return_inverse=True);subjects=torch.as_tensor(groups,device=device)
    weight=torch.as_tensor(loss_weights(windows.y,cells(windows),'class'),dtype=torch.float32,device=device)
    stopper=Stopper();ema=StateAverage('ema',cfg.get('decay',.99));valley=LossValley(3,6,cfg.get('ratio',.05))
    snapshots={};history=[];best_state=None;start_epoch=1;begin=time.monotonic()
    if (folder/'latest.pt').exists():
        s=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True);assert s['settings_hash']==token
        net.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        shuffle.set_state(s['shuffle']);ag.set_state(s['augment_rng']);torch.set_rng_state(s['rng_cpu'])
        if str(device).startswith('cuda'):torch.cuda.set_rng_state(s['rng_cuda'],device)
        start_epoch=s['epoch']+1;history=s['history'];snapshots=s['snapshots'];best_state=s['best_model']
        stopper.best=s['best'];stopper.best_epoch=s['best_epoch'];stopper.bad=s['bad']
        ema.load_state_dict(s['ema']);valley.load_state_dict(s['valley'])
    for epoch in range(start_epoch,epochs+1):
        if selecting and stopper.stopped:break
        check_budget(deadline);net.train();total=0.;seen=0;segment=StateAverage()
        permutation=torch.randperm(len(y),generator=shuffle).to(device)
        for idx in permutation.split(64):
            check_budget(deadline)
            batch=augment(x[idx],y[idx],subjects[idx],cfg.get('augment',{}),ag)
            z=net(batch,y[idx],subjects[idx],ag)
            loss=weighted_loss(z,y[idx],weight[idx],cfg.get('ls',0.))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
            if policy=='ema':ema.update(net.state_dict())
            elif policy=='swad':segment.update(net.state_dict())
            total+=float(loss.detach())*len(idx);seen+=len(idx)
        scheduler.step();check_budget(deadline)
        if policy=='swad':snapshots[epoch]=segment.result()
        elif policy=='window':snapshots[epoch]=cpu_state(net)
        if policy=='none':monitored=net
        elif policy=='window':monitored=candidate(net,mean_states([snapshots[k] for k in range(max(1,epoch-cfg['window']+1),epoch+1)]),x,deadline)
        elif policy=='ema':monitored=candidate(net,ema.result(),x,deadline)
        elif policy=='swad':monitored=candidate(net,snapshots[epoch],x,deadline)
        else:raise ValueError(policy)
        validation=evaluate(monitored,val_x,val_windows) if selecting else None
        raw_validation=evaluate(net,val_x,val_windows) if selecting and policy!='none' else validation
        if selecting:
            if stopper.update(validation['cell_nll'],epoch):
                best_state=cpu_state(monitored);atomic_save(best_state,folder/'best.pt')
            if policy=='swad':valley.update(epoch,validation['cell_nll'])
        monitor=epoch==1 or epoch%5==0 or (selecting and stopper.stopped) or epoch==epochs
        train_eval=evaluate(monitored,x,windows) if monitor else None
        row=dict(epoch=epoch,optimization_ce=total/seen)
        for prefix,values in (('train',train_eval),('val',validation),('raw_val',raw_validation)):
            row.update({prefix+'_'+key:(values[key] if values else None) for key in ('ce','cell_nll','cell_ba')})
        history.append(row);save_history(folder/'curves.csv',history)
        state=dict(settings_hash=token,epoch=epoch,model=cpu_state(net),best_model=best_state,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),shuffle=shuffle.get_state(),
            augment_rng=ag.get_state(),rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
            history=history,snapshots=snapshots,ema=ema.state_dict(),valley=valley.state_dict(),
            best=stopper.best if selecting else 0.,best_epoch=stopper.best_epoch,bad=stopper.bad)
        atomic_save(state,folder/'latest.pt')
        if monitor:
            if selecting:plot(folder,history,stopper.best_epoch)
            print('REG_EPOCH',cfg['name'],seed,epoch,'trainCE',round(train_eval['ce'],4),
                  'valCE',round(validation['ce'],4) if validation else '-',flush=True)
        if policy!='none':del monitored
    selected=stopper.best_epoch if selecting else epochs
    if selecting:
        if policy=='swad':average_epochs=valley.final_epochs()
        elif policy=='window':average_epochs=list(range(max(1,selected-cfg['window']+1),selected+1))
        elif policy=='ema':average_epochs=list(range(1,selected+1))
        else:average_epochs=[selected]
    else:
        average_epochs=list(fixed_average_epochs or [epochs])
        assert max(average_epochs)==epochs
    if policy=='swad':
        net=candidate(net,mean_states([snapshots[k] for k in average_epochs]),x,deadline)
    elif selecting:net.load_state_dict(best_state)
    elif policy=='window':net=candidate(net,mean_states([snapshots[k] for k in average_epochs]),x,deadline)
    elif policy=='ema':
        assert average_epochs==list(range(1,epochs+1));net=candidate(net,ema.result(),x,deadline)
    atomic_save(cpu_state(net),folder/'final.pt')
    r=dict(settings_hash=token,settings=settings,best_epoch=selected,stopped_epoch=len(history),
        refit_epochs=max(average_epochs),averaging_epochs=average_epochs,seconds=time.monotonic()-begin,
        parameters=sum(p.numel() for p in net.parameters()),final_sha256=sha(folder/'final.pt'))
    write_json(folder/'complete.json',r);return net.eval(),r
