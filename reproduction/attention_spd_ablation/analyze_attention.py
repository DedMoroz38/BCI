"""Independent saved-output metric verification, no model fitting."""
import csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'Brain-Decoding/track2_bci'))
from bci.evaluation import balanced_accuracy
OUT=Path(os.environ.get('ATTENTION_OUTPUT',str(HERE/'results')))

def probability_metrics(y,p):
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
    return dict(log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(float).eps)).mean()),
                brier=float(np.square(p-np.eye(3)[y]).sum(1).mean()))

def independent_metrics(data):
    y=data['y'];p=data['proba'];pred=p.argmax(1)
    np.testing.assert_array_equal(pred,data['pred'])
    np.testing.assert_allclose(p,data['tree_proba'].mean(0)/data['tree_proba'].mean(0).sum(1,keepdims=True),atol=1e-12)
    sessions=np.array([f'{s}/{t}' for s,t in zip(data['subject'],data['session'])])
    cells=np.array([f'{s}/{c}' for s,c in zip(sessions,data['context'])])
    mean=lambda group:float(np.mean([balanced_accuracy(y[group==g],pred[group==g]) for g in np.unique(group)]))
    return dict(session_mean=mean(sessions),cell_mean=mean(cells),pooled=balanced_accuracy(y,pred),
                accuracy=float((pred==y).mean()),**probability_metrics(y,p))

def main():
    records={};verified=0;exact=0;foldrows=[]
    for mode in ['baseline','attention_gram','attention_exp']:
        directory=OUT/'baseline_reproduction/baseline' if mode=='baseline' else OUT/mode
        records[mode]={}
        paths=sorted(directory.glob('*.json'));assert len(paths)==10,(mode,len(paths))
        for path in paths:
            r=json.loads(path.read_text());npz=path.with_name(path.stem+'_predictions.npz')
            assert hashlib.sha256(npz.read_bytes()).hexdigest()==r['predictions_sha256']
            with np.load(npz) as data:
                recomputed=independent_metrics(data)
                for k,v in recomputed.items():np.testing.assert_allclose(r[k],v,atol=1e-12,rtol=0)
                if mode=='baseline':
                    protocol=r['protocol'];fold=r['fold'].replace('/','_')
                    ref=ROOT/'temporal_spd_research/results/lag4_oas'/f'forest20_{protocol}_{fold}_predictions.npz'
                    with np.load(ref) as old:
                        for key in ['y','subject','session','context','run','onset','pred']:
                            np.testing.assert_array_equal(old[key],data[key])
                    exact+=1
            verified+=1;records[mode][path.stem]=r
            foldrows.append(dict(model=mode,fold=path.stem,**recomputed))
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['cv_hashes'].items():
        assert hashlib.sha256((OUT/rel).read_bytes()).hexdigest()==digest
    summary=[];paired=[]
    for mode,rs in records.items():
        loso=[rs[k]['session_mean'] for k in sorted(rs) if k.startswith('loso_')]
        np.testing.assert_allclose(np.mean(loso),selection['scores'][mode],atol=1e-14)
        summary.append(dict(model=mode,loso_session_mean=float(np.mean(loso)),
                            loso_cell_mean=float(np.mean([rs[k]['cell_mean'] for k in sorted(rs) if k.startswith('loso_')])),
                            later_session_mean=rs['later_last']['session_mean'],
                            test_cell_mean=rs['test_official']['cell_mean'],
                            test_session_mean=rs['test_official']['session_mean'],
                            test_log_loss=rs['test_official']['log_loss'],test_brier=rs['test_official']['brier']))
        if mode!='baseline':
            diffs=np.array([rs[k]['session_mean']-records['baseline'][k]['session_mean'] for k in sorted(rs) if k.startswith('loso_')])
            rng=np.random.default_rng(0);boot=diffs[rng.integers(0,len(diffs),size=(10000,len(diffs)))].mean(1)
            paired.append(dict(model=mode,mean_difference=float(diffs.mean()),wins=int((diffs>0).sum()),
                               ci95=np.quantile(boot,[.025,.975]).tolist(),per_person_differences=diffs.tolist()))
    for name,rows in [('comparison.csv',summary),('fold_scores.csv',foldrows)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(verified_jobs=verified,exact_baseline_jobs=exact,independent_metric_recomputation=True),indent=2))
    lines=['# Attention SPD vs Lag4: verified comparison','',
           '| Model | LOSO session BA | Later session BA | Public test cell BA | Public test session BA |',
           '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['loso_session_mean']:.6f} | {r['later_session_mean']:.6f} | {r['test_cell_mean']:.6f} | {r['test_session_mean']:.6f} |")
    lines+=['',f"Frozen CV winner: {selection['winner']}. All30 outputs verified; all10 baseline label vectors reproduce exactly.",'',
            'Auxiliary identity-tangent linear head trains each independent encoder; deployed head remains original causal tangent LR. W_Q and W_K are independently learned. Fixed80epochs is not a convergence guarantee. Public test was used previously; no sealed-test claim or competition submission. Paired bootstrap CIs are descriptive; LOSO training sets overlap.']
    for r in paired:lines.append(f"\n{r['model']}: paired LOSO delta {100*r['mean_difference']:+.3f}pp;95% descriptive CI [{100*r['ci95'][0]:+.3f},{100*r['ci95'][1]:+.3f}]pp;wins{r['wins']}/8.")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('VERIFIED ALL30 ATTENTION/BASELINE JOBS',flush=True)
if __name__=='__main__':main()
