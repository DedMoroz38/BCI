"""Fixed and label-free bounded expert weights. No fitted OOF stacking."""
from pathlib import Path
import numpy as np
from common import *
from nets import CONFIGS

def combine_outputs(out,cache):
    out=Path(out)
    with np.load(Path(cache)/'meta.npz') as z:meta={k:z[k] for k in META}
    for tag in TAGS:
        tr,te=train_test(meta,tag);mt=subset(meta,te);ref=reference(tag);raw=[];names=[]
        for cfg in CONFIGS:
            folders=[out/'models'/tag/cfg['name']/f'seed{s}' for s in (0,1)]
            if not all((p/'logits.npy').exists() and (out/'predictions'/tag/f"{cfg['name']}_seed{s}.json").exists() for s,p in enumerate(folders)):continue
            logits=[np.load(p/'logits.npy') for p in folders];prob=[softmax(z,axis=1) for z in logits]
            p=np.mean(prob,axis=0);name=cfg['name']+'_mean2';save_expert(out,tag,name,mt,p,seeds=[0,1])
            ep=balanced(mt,p)
            save_prediction(out,tag,name+'__replace_norm',mt,.8*ref['old_champion']+.2*ep)
            mean=(prob[0]+prob[1])/2
            js=.5*np.sum(prob[0]*np.log(np.maximum(prob[0],1e-15)/np.maximum(mean,1e-15))+
                          prob[1]*np.log(np.maximum(prob[1],1e-15)/np.maximum(mean,1e-15)),axis=1)
            agreement=np.exp(-js/.05)
            for mode in ('seed_agreement','prefix_agreement'):
                a=.1+.2*agreement
                if mode=='prefix_agreement':
                    count=np.empty(len(p))
                    for ids in streams(mt):count[ids]=np.arange(1,len(ids)+1)
                    a=.1+.2*agreement*count/(20+count)
                q=(1-a[:,None])*ref['champion']+a[:,None]*ep
                save_prediction(out,tag,name+'__'+mode,mt,q,fusion='label_free_gate',bounds=[.1,.3])
            cnn=np.mean([softmax(z/4,axis=1) for z in logits],axis=0)
            mixed=.7*ref['lag_raw']+.3*cnn
            save_prediction(out,tag,name+'__lag4_mix_balance',mt,balanced(mt,mixed))
            raw.append(mixed);names.append(name+'__lag4_mix_tta')
        if raw:
            try:
                result=apply_tta_many(tta_grams(cache,meta,tr,te),mt,np.stack(raw))
                for name,p in zip(names,result):save_prediction(out,tag,name,mt,p)
            except Exception as e:
                import traceback
                write_json(out/'combination_failures'/f'{tag}.json',dict(error=str(e),traceback=traceback.format_exc()))
