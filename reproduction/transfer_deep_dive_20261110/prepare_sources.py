"""Reuse working research plumbing, adding only the new prospective policies."""
from pathlib import Path
import json,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'winner_research_20261109'

def main():
    for name in ('averaging.py','tta_core.py'):shutil.copyfile(OLD/name,HERE/name)
    common=(OLD/'common.py').read_text().replace('-.627233744814203','-.6322938018366412')
    common=common.replace('Historical full baseline: OOF62.7234%, later71.1102%.','Historical full baseline: OOF63.2294%, later71.6395%.')
    common=common.replace('for w in (.1,.2):','for w in (.1,.2,.3):')
    common+='''

_original_summary=summarize
def summarize(out):
    _original_summary(out)
    import pandas as pd
    out=Path(out)
    if not (out/'comparison.csv').exists():return
    df=pd.read_csv(out/'comparison.csv')
    df=df[~df.name.str.contains(r'_seed[01]$')].copy()
    folds=pd.read_csv(out/'fold_scores.csv');baseline={}
    for tag in TAGS[:10]:
        r=reference(tag);baseline[tag]=score({k:r[k] for k in META},r['champion'])['cell']
    extra=[];person=[]
    for name in df.name:
        rows=folds[folds.name.eq(name)&folds.tag.str.startswith('loso_')]
        delta=[]
        for _,r in rows.iterrows():
            d=100*(r.cell-baseline[r.tag]);delta.append(d)
            person.append(dict(name=name,tag=r.tag,cell=r.cell,baseline=baseline[r.tag],delta_pp=d))
        extra.append(dict(name=name,wins=int(np.sum(np.array(delta)>0)),median_person_delta_pp=float(np.median(delta)),
                          worst_person_delta_pp=float(np.min(delta)),worst3_person_mean=float(np.mean(np.sort(rows.cell.to_numpy())[:3]))))
    df=df.merge(pd.DataFrame(extra),on='name').sort_values('oof_cell',ascending=False)
    df.to_csv(out/'primary_comparison.csv',index=False);pd.DataFrame(person).to_csv(out/'transfer_by_person.csv',index=False)
    text='# Transfer normalization: held-out people\\n\\nBaseline OOF63.2294%,later71.6395%. No baseline refits.\\n'
    text+='Two-seed complete configurations only; individual seeds are diagnostics in comparison.csv.\\n'
    text+='Public-CV selection remains exploratory. Later sessions are separate.\\n\\n'+df.head(40).to_string(index=False)
    (out/'REPORT.md').write_text(text+'\\n');print(text,flush=True)
'''
    (HERE/'common.py').write_text(common)
    gpu=(OLD/'gpu.py').read_text()
    gpu=gpu.replace("groups=list(streams(meta));y=", "groups=list(streams(meta));people=[np.flatnonzero(meta['subject']==p) for p in np.unique(meta['subject'])];y=")
    gpu=gpu.replace('refresh_source(x,groups)','refresh_source(x,groups,people)')
    gpu=gpu.replace("z=net(x[idx]);yy=", "batch=x[idx]\n            if cfg.get('shift'):\n                width=cfg['shift'];offset=torch.randint(-width,width+1,(len(idx),1,1),device=x.device)\n                positions=torch.arange(batch.shape[-1],device=x.device)[None,None,:]+width+offset\n                batch=F.pad(batch,(width,width),mode='reflect').gather(-1,positions.expand(-1,batch.shape[1],-1))\n            z=net(batch);yy=")
    gpu=gpu.replace("F.cross_entropy(z,yy,reduction='none')*ww", "F.cross_entropy(z,yy,reduction='none',label_smoothing=cfg.get('ls',0.))*ww")
    gpu=gpu.replace("if not torch.isfinite(loss)","loss=loss+net.regularization()\n            if not torch.isfinite(loss)")
    gpu=gpu.replace("monitored=new_net(cfg,seed,device);monitored.load_state_dict", "monitored=deepcopy(net);monitored.load_state_dict")
    gpu=gpu.replace('from cpu import prep_job','from preprocess import prep_job')
    gpu=gpu.replace("default='/tmp/winner-gpu-cache'","default='/tmp/transfer-cache'")
    gpu=gpu.replace("default='/kaggle/working/winner-gpu-results'","default='/kaggle/working/transfer-results'")
    gpu=gpu.replace('default=690','default=600')
    gpu=gpu.replace('selection=selection,refit=refit)',"selection=selection,refit=refit,learned_alpha=net.alpha_logit.sigmoid().detach().cpu().tolist() if hasattr(net,'alpha_logit') else None)")
    # Replace dependent fusion reporting; true base-model fitting stays unchanged.
    start=gpu.index('def combine(out,cache):');end=gpu.index('\ndef main():',start)
    gpu=gpu[:start]+'''def combine(out,cache):
    from fusion import combine_outputs
    combine_outputs(out,cache)
''' + gpu[end:]
    (HERE/'run.py').write_text(gpu)
    refs=HERE/'reference';refs.mkdir(exist_ok=True)
    for p in sorted((OLD/'reference').glob('*.npz')):
        with np.load(p) as z:values={k:z[k] for k in z.files}
        values['old_champion']=values['champion'].copy()
        folder=OLD/'downloads/gpu-v1/winner-gpu-results/predictions'/p.stem
        with np.load(folder/'latent_meanstd_p10_mean2__late_w0.2.npz') as z:values['champion']=z['proba']
        with np.load(folder/'latent_meanstd_p10_mean2.npz') as z:values['previous_norm_cnn']=z['proba']
        np.savez_compressed(refs/p.name,**values)
    origins=dict(source='winner_research_20261109; current best63.2294%; frozen predictions reused',
                 model_design='14 causal normalization/multiscale adaptations; no pretrained weights',
                 review='literature/REVIEW_RU.md',selection='Inner-person-held-out cell-NLL; SWAD.1',seeds=[0,1])
    (HERE/'ORIGINS.json').write_text(json.dumps(origins,indent=2))

if __name__=='__main__':main()
