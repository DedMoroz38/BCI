"""Read-only independent scoring of the completed35-variant research wave."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import sha,write_json,softmax
from configurations import CONFIGS
from metrics import load,balanced_mix

ROOT=HERE/'downloads/gpu-v2/regularization-results'
OUT=HERE/'to_human'

def main():
    manifest=json.loads((ROOT/'study-manifest.json').read_text())
    build=json.loads((HERE/'notebook-build.json').read_text())
    provenance=json.loads((ROOT/'bundle-provenance.json').read_text())
    assert provenance['regularization_bundle_sha256']==build['bundle_sha256']
    for name,digest in manifest['code'].items():assert sha(HERE/name)==digest
    for split,digest in manifest['data'].items():assert sha(HERE.parent/'data'/f'proteus_{split}.npz')==digest
    for name,digest in manifest['references'].items():assert sha(HERE/'reference'/name)==digest
    execution=json.loads((ROOT/'execution.json').read_text())
    assert len(execution['jobs'])==770 and not execution['failed_jobs'] and not execution['budget_pending']
    spec=importlib.util.spec_from_file_location('independent_scorer',HERE.parent/'ensemble_comparison/analyze.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer);score=scorer.independent_scores
    rows={};curves=[];nll_diffs=[];softmax_diffs=[];cpudiff=0;gpudiff=0;cpumax=0.;count=0;best_models=0
    expected_configs={c['name'] for c in CONFIGS};expected_tags=sorted(p.name for p in (HERE/'reference/baselines').iterdir())
    baseline={tag:load(HERE/'reference/baselines'/tag/'lag4.npz') for tag in expected_tags}
    for path in sorted((ROOT/'models').glob('*/*/seed*/prediction.json')):
        r=json.loads(path.read_text());parent=path.parent;meta=load(parent/'prediction.npz');z=np.load(parent/'logits.npy')
        assert r['config'] in expected_configs and r['seed'] in (0,1)
        assert sha(parent/'prediction.npz')==r['prediction_sha256'] and sha(parent/'logits.npy')==r['logits_sha256']
        assert r['study_sha256']==sha(ROOT/'study-manifest.json')
        for key in ('y','subject','session','context','run','onset'):np.testing.assert_array_equal(meta[key],baseline[r['tag']][key])
        assert not set(r['inner_train_people'])&set(r['inner_val_people'])
        assert set(r['inner_train_people'])|set(r['inner_val_people'])==set(r['train_people'])
        if r['tag'].startswith('loso_'):assert not set(meta['subject'])&set(r['train_people'])
        np.testing.assert_array_equal(meta['pred'],meta['proba'].argmax(1))
        for key,value in score(meta).items():np.testing.assert_allclose(value,r[key],rtol=0,atol=1e-12)
        p=softmax(z,axis=1);softmax_diffs.append(float(np.max(np.abs(p-meta['proba']))))
        nll=float(-np.log(np.maximum(meta['proba'][np.arange(len(p)),meta['y']],1e-15)).mean())
        nll_diffs.append(abs(nll-r['log_loss']))
        cpu=r['cpu_replay_diagnostic'];gpu=r['gpu_replay_diagnostic']
        cpudiff+=cpu['label_disagreements'];gpudiff+=gpu['label_disagreements'];cpumax=max(cpumax,cpu['max_probability_delta'])
        if (parent/'model.pt').exists():
            assert sha(parent/'model.pt')==r['model_sha256'] and sha(parent/'alignment.npz')==r['alignment_sha256'];best_models+=1
        rows.setdefault((r['config'],r['seed']),{})[r['tag']]=(meta,z,r)
        history=parent/'selection/curves.csv'
        if history.exists():
            h=pd.read_csv(history);complete=json.loads((parent/'selection/complete.json').read_text())
            assert r['best_epoch']==complete['best_epoch'] and r['averaging_epochs']==complete['averaging_epochs']
            first,last=h.iloc[0],h.iloc[-1];selected=h[h.epoch==r['best_epoch']].iloc[0]
            curves.append(dict(config=r['config'],seed=r['seed'],tag=r['tag'],best_epoch=r['best_epoch'],
                refit_epochs=r['refit_epochs'],stopped_epoch=len(h),train_ce_first=first.train_ce,train_ce_last=last.train_ce,
                val_ce_first=first.val_ce,val_ce_last=last.val_ce,train_ba_last=last.train_cell_ba,val_ba_last=last.val_cell_ba,
                val_nll_best=selected.val_cell_nll,val_ba_selected=selected.val_cell_ba,
                raw_val_ce_last=last.raw_val_ce,raw_val_ba_last=last.raw_val_cell_ba,
                overfit_curve=bool(last.train_ce<first.train_ce and last.val_ce>first.val_ce)))
        count+=1
    assert count==770 and len(rows)==70 and all(sorted(g)==expected_tags for g in rows.values())
    summaries=[];fold_scores=[]
    for c in CONFIGS:
        for seeds in ([0],[1],[0,1]):
            name='mean2' if len(seeds)==2 else str(seeds[0]);cv=[];person=[]
            for tag in expected_tags:
                parts=[rows[c['name'],seed][tag] for seed in seeds];meta=parts[0][0]
                cnn=np.mean([softmax(p[1],axis=1) for p in parts],axis=0)
                mixed=balanced_mix(meta,baseline[tag]['proba'],[p[1] for p in parts])
                def scored(p):return score(dict(meta,proba=p,pred=p.argmax(1)))
                values=dict(cnn_cell=scored(cnn)['cell_mean'],mix_cell=scored(mixed)['cell_mean'])
                fold_scores.append(dict(config=c['name'],seeds=name,tag=tag,**values))
                if tag.startswith('loso_'):cv.append((meta,cnn,mixed));person.append(values)
                else:late=values
            merged={k:np.concatenate([p[0][k] for p in cv]) for k in ('y','subject','session','context','run','onset')}
            assert len(merged['y'])==2655 and len(set(zip(merged['subject'],merged['session'],merged['run'],merged['onset'])))==2655
            pc=np.concatenate([p[1] for p in cv]);pm=np.concatenate([p[2] for p in cv])
            sc=score(dict(merged,proba=pc,pred=pc.argmax(1)));sm=score(dict(merged,proba=pm,pred=pm.argmax(1)))
            summaries.append(dict(config=c['name'],seeds=name,cnn_oof_cell=sc['cell_mean'],mix_oof_cell=sm['cell_mean'],
                cnn_mean_person=np.mean([v['cnn_cell'] for v in person]),mix_mean_person=np.mean([v['mix_cell'] for v in person]),
                cnn_late_cell=late['cnn_cell'],mix_late_cell=late['mix_cell'],
                cnn_nll=float(-np.log(np.maximum(pc[np.arange(2655),merged['y']],1e-15)).mean()),
                mix_nll=float(-np.log(np.maximum(pm[np.arange(2655),merged['y']],1e-15)).mean()),
                cnn_brier=float(np.mean(np.sum((pc-np.eye(3)[merged['y']])**2,axis=1))),cnn_mean_confidence=float(pc.max(1).mean())))
    result=pd.DataFrame(summaries);original=pd.read_csv(ROOT/'comparison.csv')
    brier_diffs=[]
    for _,row in original.iterrows():
        actual=result[(result.config==row['config'])&(result.seeds==row.seeds)].iloc[0]
        for key in ('cnn_oof_cell','mix_oof_cell','cnn_mean_person','mix_mean_person','cnn_late_cell','mix_late_cell'):
            np.testing.assert_allclose(actual[key],row[key],rtol=0,atol=1e-12)
        brier_diffs.append(abs(float(actual.cnn_brier)-float(row.cnn_brier)))
    result.sort_values(['seeds','mix_oof_cell'],ascending=[True,False]).to_csv(OUT/'REGULARIZATION_COMPARISON.csv',index=False)
    pd.DataFrame(fold_scores).to_csv(OUT/'REGULARIZATION_FOLD_SCORES.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'REGULARIZATION_LEARNING.csv',index=False)
    write_json(OUT/'REGULARIZATION_VERIFICATION.json',dict(fits=count,configs=35,seeds=2,folds=11,
        failed_jobs=0,budget_pending=0,source_data_reference_hashes=True,independent_metric_rescore=True,
        prediction_and_logit_hashes=True,metadata_matches=True,inner_outer_people_separated=True,
        downloaded_best_model_hashes=best_models,learning_histories=len(curves),
        gpu_recorded_cpu_label_differences=cpudiff,gpu_recorded_gpu_label_differences=gpudiff,
        max_recorded_cpu_probability_delta=cpumax,max_logit_softmax_rescore_delta=max(softmax_diffs),
        max_float32_nll_rescore_delta=max(nll_diffs),max_brier_recomputed_softmax_delta=max(brier_diffs),seconds=execution['seconds']))
    print(result[result.seeds=='mean2'].sort_values('mix_oof_cell',ascending=False).head(10).to_string(index=False),flush=True)
    print('VERIFIED',count,'fits;',len(curves),'histories',flush=True)

if __name__=='__main__':main()
