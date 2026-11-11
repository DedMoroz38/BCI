"""Actual 6/2 inner-source training smoke; scores do not select policies."""
import time,json
from pathlib import Path
import torch
from core import *
from novel import CONFIGS,new_net,SessionPredictor
from neural_training import fit_early,predict

def main():
    torch.set_num_threads(2);cache=ROOT/'mfm_models_20261110/cache';m=metadata(cache)
    x=np.load(cache/'X.npy',mmap_mode='r');tag=pair_tags()[0]
    rec=json.loads((HERE/'splits.json').read_text())['folds'][tag]
    tr=np.asarray(rec['inner_train_ids']);va=np.asarray(rec['inner_val_ids'])
    a,b,prior=get_aligned(x,m,tr,va)
    a=torch.from_numpy(a);b=torch.from_numpy(b);mt=subset(m,tr);mv=subset(m,va)
    receipts=[]
    for cfg in (CONFIGS[0],CONFIGS[2]):
        folder=HERE/'smoke'/cfg['name'];begin=time.monotonic()
        net,selection=fit_early(cfg,0,a,mt,folder/'selection','cpu',time.monotonic()+180,
                                val_x=b,val_meta=mv,epochs=2)
        net,refit=fit_early(cfg,0,a,mt,folder/'refit','cpu',time.monotonic()+180,
                            epochs=selection['refit_epochs'],fixed_average=selection['averaging_epochs'])
        z=predict(net,b,mv);ids=next(streams(mv))[:8];stream=SessionPredictor(net)
        q=np.concatenate([stream.predict_logits(b[i:i+1]).numpy() for i in ids])
        assert np.allclose(q,z[ids],atol=2e-4,rtol=2e-4)
        receipts.append(dict(name=cfg['name'],source_people=np.unique(mt['subject']).tolist(),
                             validation_people=np.unique(mv['subject']).tolist(),selection=selection,refit=refit,
                             b1_max_error=float(np.abs(q-z[ids]).max()),seconds=time.monotonic()-begin))
        write_json(HERE/'smoke/receipt.json',receipts)
    print('REAL SOURCE SELECTION/REFIT SMOKE PASSED',flush=True)

if __name__=='__main__':main()
