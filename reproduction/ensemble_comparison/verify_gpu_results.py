"""Local independent validation of a completed recovered Kaggle run."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    target=args.directory
    spec=importlib.util.spec_from_file_location('independent_ensemble_metrics',HERE/'analyze.py')
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    manifest=sha(target/'manifest.json');records=[];baseline_matches=0
    for path in sorted((target/'results').glob('*/*.json')):
        if path.stem!='baseline' and not path.stem.startswith('variant'):continue
        r=json.loads(path.read_text());p=path.with_suffix('.npz')
        assert r['manifest_sha256']==manifest
        assert r['prediction_sha256']==sha(p)
        tag=path.parent.name
        suffix='loso_Dreyer2026Proteus_'+tag[5:] if tag.startswith('loso_') else tag
        reference=ROOT/'temporal_spd_research/results/lag4_oas'/f'forest20_{suffix}_predictions.npz'
        with np.load(p) as f,np.load(reference) as control:
            for key in ('y','subject','session','context','run','onset'):np.testing.assert_array_equal(f[key],control[key])
            np.testing.assert_array_equal(f['pred'],f['proba'].argmax(1))
            for key,value in metrics.independent_scores(f).items():np.testing.assert_allclose(value,r[key],rtol=0,atol=1e-12)
            proba=f['proba'];y=f['y']
            np.testing.assert_allclose(proba.sum(1),1,atol=1e-10)
            assert np.isfinite(proba).all() and np.all(proba>=0)
            np.testing.assert_allclose(-np.log(np.maximum(proba[np.arange(len(y)),y],1e-15)).mean(),r['log_loss'],atol=1e-12)
            np.testing.assert_allclose(np.mean(np.sum((proba-np.eye(3)[y])**2,axis=1)),r['brier'],atol=1e-12)
            if path.stem=='baseline':
                np.testing.assert_array_equal(f['pred'],control['pred']);baseline_matches+=1
        records.append(r)
    assert len(records)==80 and baseline_matches==10
    for name in {r['variant'] for r in records}:
        jobs=[r for r in records if r['variant']==name]
        assert len(jobs)==10 and len({r['tag'] for r in jobs})==10
    freeze=json.loads((target/'cv-freeze.json').read_text());assert freeze['manifest']==manifest
    assert len(freeze['results'])==63
    for name,digest in freeze['results'].items():assert sha(target/name)==digest
    lineage=json.loads((target/'resume-lineage.json').read_text())
    old=HERE/'downloads/gpu-v2/ensemble-results'
    assert lineage['recovery_manifest']==manifest
    assert lineage['parent_manifest']==sha(target/'parent-manifest.json')==sha(old/'manifest.json')
    assert sum(lineage['records'].values())==46 and len(lineage['records'])==6
    for name,digest in lineage['files'].items():assert sha(old/name)==digest
    replay=[json.loads(p.read_text()) for p in (target/'results').glob('*/replay*.json')]
    assert all(r['parameters_exact'] and r['labels_exact'] and r['max_probability_delta']<=r['atol'] for r in replay)
    pairs={name:metrics.paired(records,name) for name in {r['variant'] for r in records} if name!='Lag4'}
    reported=json.loads((target/'paired.json').read_text())
    for name in pairs:
        np.testing.assert_allclose(pairs[name]['mean_delta'],reported[name]['mean_delta'],atol=1e-12)
        assert pairs[name]['wins']==reported[name]['wins']
    result=dict(evaluations=80,baseline_matches=10,independent_probability_and_label_rescore=True,
                metadata_matches=80,cv_freeze_checked=63,parent_imports_checked=sum(lineage['records'].values()),
                parent_manifest=lineage['parent_manifest'],manifest=manifest,replay_checks=len(replay),
                max_replay_probability_delta=max((r['max_probability_delta'] for r in replay),default=0),
                original_training_warnings_unavailable=lineage.get('original_training_warnings_unavailable',[]))
    (HERE/'LOCAL_GPU_VERIFICATION.json').write_text(json.dumps(result,indent=2))
    (HERE/'LATEST_REPORT.md').write_text((target/'REPORT.md').read_text()+'\nЛокальная независимая проверка:80/80оценок,10/10совпаденийLag4,63CV-хеша и46импортированныхоценок проверены. Исходные предупреждения обученияP02недоступны.\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
