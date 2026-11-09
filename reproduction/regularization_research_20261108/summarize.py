"""Rank complete all10 configuration/seeds; report partial results explicitly."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import sha,write_json
from metrics import load,score,balanced_mix,softmax
from configurations import CONFIGS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root
    records={};status=[]
    for path in (root/'models').glob('*/*/seed*/prediction.json'):
        r=json.loads(path.read_text());parent=path.parent
        for name,key in [('prediction.npz','prediction_sha256'),('model.pt','model_sha256'),('alignment.npz','alignment_sha256'),('logits.npy','logits_sha256')]:assert sha(parent/name)==r[key]
        meta=load(parent/'prediction.npz');baseline=load(HERE/'reference/baselines'/r['tag']/'lag4.npz')
        for key in ('y','subject','session','context','run','onset'):np.testing.assert_array_equal(meta[key],baseline[key])
        np.testing.assert_array_equal(meta['pred'],meta['proba'].argmax(1))
        np.testing.assert_allclose(score(meta,meta['proba'])['cell_mean'],r['cell_mean'],atol=1e-12)
        records.setdefault((r['config'],int(r['seed'])),{})[r['tag']]=(meta,np.load(parent/'logits.npy'),baseline['proba'])
    expected=sorted(p.name for p in (HERE/'reference/baselines').iterdir() if p.is_dir());assert len(expected)==11
    groups={}
    for c in CONFIGS:
        for seed in (0,1):
            available=records.get((c['name'],seed),{})
            complete=sorted(available)==expected
            status.append(dict(config=c['name'],seed=seed,folds=len(available),complete=complete))
            if complete:groups[(c['name'],str(seed))]={tag:[available[tag]] for tag in expected}
        if all((c['name'],str(seed)) in groups for seed in (0,1)):
            groups[(c['name'],'mean2')]={tag:[records[c['name'],seed][tag] for seed in (0,1)] for tag in expected}
    rows=[];foldrows=[]
    for (name,seeds),fits in groups.items():
        cv=[];personal=[]
        for tag,parts in fits.items():
            meta=parts[0][0];z=[part[1] for part in parts];base=parts[0][2]
            cnn=np.mean([softmax(v,axis=1) for v in z],axis=0);mixed=balanced_mix(meta,base,z)
            values=dict(cnn_cell=score(meta,cnn)['cell_mean'],mix_cell=score(meta,mixed)['cell_mean'])
            foldrows.append(dict(config=name,seeds=seeds,tag=tag,**values))
            if tag.startswith('loso_'):cv.append((meta,cnn,mixed));personal.append(values)
            else:late=values
        merged={k:np.concatenate([v[0][k] for v in cv]) for k in ('y','subject','session','context','run','onset')}
        assert len(merged['y'])==2655 and len(set(zip(merged['subject'],merged['session'],merged['run'],merged['onset'])))==2655
        cs=score(merged,np.concatenate([v[1] for v in cv]));ms=score(merged,np.concatenate([v[2] for v in cv]))
        rows.append(dict(config=name,seeds=seeds,cnn_oof_cell=cs['cell_mean'],mix_oof_cell=ms['cell_mean'],
            cnn_mean_person=np.mean([p['cnn_cell'] for p in personal]),mix_mean_person=np.mean([p['mix_cell'] for p in personal]),
            cnn_late_cell=late['cnn_cell'],mix_late_cell=late['mix_cell'],cnn_nll=cs['log_loss'],mix_nll=ms['log_loss'],cnn_brier=cs['brier']))
    pd.DataFrame(status).to_csv(root/'completeness.csv',index=False)
    pd.DataFrame(foldrows).to_csv(root/'fold_scores.csv',index=False)
    df=pd.DataFrame(rows)
    if not df.empty:df=df.sort_values(['seeds','mix_oof_cell'],ascending=[True,False]);df.to_csv(root/'comparison.csv',index=False)
    ranking=df[df.seeds=='mean2'].sort_values('mix_oof_cell',ascending=False) if not df.empty else df
    lines=['# PROTEUS regularization wave2','',f'Completed fits: {sum(len(v) for v in records.values())}/770.',
        f'Complete two-seed configurations: {len(ranking)}/35. Incomplete configurations are excluded from ranking.',
        'NLL epoch selection unchanged. Fixed fusion:70%Lag4+30%mean softmax(CNNlogits/4),then balance1/prior10.',
        'OOF cell BA pools all10people;mean person is separate. Public CV is reused: exploratory,not sealed confirmation.','',
        ranking.to_string(index=False) if len(ranking) else 'No complete two-seed configurations yet.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'summary.json',dict(fits=sum(len(v) for v in records.values()),planned_fits=770,
        complete_mean2_configs=len(ranking),all_complete=len(ranking)==35))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
