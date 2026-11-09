"""Legacy fixed-weight fusion using existing trained CNNs; no neural training."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import sosfiltfilt
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import ROOT,AUTHOR,sha,write_json,folds,metadata,balance,mix,scores,softmax,save_prediction
from bci.ensemble import aligned_windows,DEEP_BAND
from bci.stream import bandpass,load_spec,ShallowNet
SOURCE=ROOT/'ensemble_comparison/downloads/recovery-v3/ensemble-results'
OUT=HERE/'experiments/local_fusion/results';CACHE=HERE/'experiments/local_fusion/cache'
CONFIGS=[dict(name='lag4',weight=0,temp=1,alpha=0,pool='arithmetic'),dict(name='lag4_balance1',weight=0,temp=1,alpha=1,pool='arithmetic'),dict(name='cnn_only',weight=1,temp=1,alpha=0,pool='arithmetic')]
CONFIGS += [dict(name=f'arith_w{weight}_T{temp}_a{alpha}',weight=weight,temp=temp,alpha=alpha,pool='arithmetic') for weight in (.1,.3) for temp in (1,2,4) for alpha in (0,1)]
CONFIGS += [dict(name=f'geo_w0.3_T1_a{alpha}',weight=.3,temp=1,alpha=alpha,pool='geometric') for alpha in (0,1)]

def author_geo(p,q):return softmax(.7*np.log(p+1e-9)+.3*np.log(q+1e-9),axis=1)

def extract(tag,test):
    path=CACHE/(tag+'.npz');source=SOURCE/'results'/tag/'parts.npz';digest=sha(source)
    if path.exists():
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['checkpoint']!=digest or receipt['prediction']!=sha(path):raise RuntimeError('Changed CNN extraction cache')
        with np.load(path) as f:
            for k,v in metadata(test).items():np.testing.assert_array_equal(v,f[k])
            return f['logits'].copy()
    parts=load_spec(source);deep=parts['deep'];assert len(deep['nets'])==1
    xf=sosfiltfilt(bandpass(DEEP_BAND,120),np.asarray(test.X,np.float64),axis=-1)
    aligned=aligned_windows(test,xf,deep['cov_prior'],deep['prior_weight'])
    aligned=np.ascontiguousarray(aligned,dtype=np.float32)
    member=deep['nets'][0];net=ShallowNet(**member['config'])
    net.load_state_dict({k:torch.as_tensor(v) for k,v in member['state'].items()});net.eval()
    def inference(batch):
        with torch.no_grad():return torch.cat([net(x) for x in torch.as_tensor(aligned,dtype=torch.float32).split(batch)]).numpy()
    logits=inference(32)
    with np.load(SOURCE/'results'/tag/'variant0.npz') as f:forest=f['proba']
    with np.load(SOURCE/'results'/tag/'variant4.npz') as f:reference=f['proba'];labels=f['pred']
    reproduced=author_geo(forest,softmax(logits,axis=1))
    if not np.array_equal(reproduced.argmax(1),labels) or np.max(np.abs(reproduced-reference))>1e-4:
        logits=inference(1);reproduced=author_geo(forest,softmax(logits,axis=1))
    np.testing.assert_array_equal(reproduced.argmax(1),labels)
    np.testing.assert_allclose(reproduced,reference,rtol=0,atol=1e-4)
    CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,logits=logits,**metadata(test))
    write_json(path.with_suffix('.json'),dict(checkpoint=digest,prediction=sha(path),original_ensemble_replayed=True,max_delta=float(np.max(np.abs(reproduced-reference)))))
    return logits

def evaluate(tag,test):
    z=extract(tag,test)
    with np.load(SOURCE/'results'/tag/'baseline.npz') as f:
        for k,v in metadata(test).items():np.testing.assert_array_equal(v,f[k])
        base=f['proba'].copy()
    records=[]
    for cfg in CONFIGS:
        q=softmax(z/cfg['temp'],axis=1)
        p=author_geo(base,q) if cfg['pool']=='geometric' else mix(base,q,cfg['weight'],cfg['pool'])
        if cfg['alpha']:p=balance(p,list(test.streams()),cfg['alpha'])
        record=save_prediction(OUT/cfg['name']/(tag+'.npz'),test,p,tag=tag,config=cfg)
        records.append(dict(variant=cfg['name'],tag=tag,**{k:record[k] for k in ('cell_mean','session_mean','log_loss','brier')}))
    print('FUSION_DONE',tag,flush=True);return records

def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',dict(configs=CONFIGS,protocol_sha256=sha(HERE/'PROTOCOL.md'),source_manifest=sha(SOURCE/'manifest.json')))
    rows=[]
    with threadpool_limits(limits=1,user_api='blas'):
        for tag,_,test in folds('legacy'):rows+=evaluate(tag,test)
        cv=pd.DataFrame(rows);lo=cv[cv.tag.str.startswith('loso_')]
        ranking=lo.groupby('variant').agg(cell_mean=('cell_mean','mean'),log_loss=('log_loss','mean')).reset_index().sort_values(['cell_mean','log_loss','variant'],ascending=[False,True,True])
        selection=dict(winner=ranking.iloc[0].variant,scores=ranking.to_dict('records'),cv_files={str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*/*.json')})
        write_json(OUT/'selection-before-test.json',selection)
        for tag,_,test in folds('legacy',test=True):rows+=evaluate(tag,test)
    df=pd.DataFrame(rows);df.to_csv(OUT/'fold_scores.csv',index=False)
    df['protocol']=df.tag.map(lambda t:'loso' if t.startswith('loso_') else ('test' if t=='test_official' else 'later'))
    table=df.pivot_table(index='variant',columns='protocol',values=['cell_mean','session_mean','log_loss','brier'],aggfunc='mean')
    table.columns=['_'.join(v) for v in table.columns];table=table.sort_values('cell_mean_loso',ascending=False)
    table.to_csv(OUT/'comparison.csv')
    print('FROZEN_CV_WINNER',selection['winner'],flush=True)
    print(table[['cell_mean_loso','cell_mean_later','cell_mean_test']].to_string(),flush=True)

if __name__=='__main__':main()
