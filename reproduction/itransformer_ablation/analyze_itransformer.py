"""Independent probability/metric verification; seed means are NOT ensembles."""
import csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'single_attention_ablation'))
from analyze_single import independently_score as rescore
OUT=Path(os.environ.get('IT_OUTPUT',str(HERE/'results')))
SEEDS=[0,1,2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    baseline={};neural={};foldrows=[];verified=0
    for p in sorted((OUT/'baseline').glob('*.json')):
        r=json.loads(p.read_text());assert r['exact_original_single_model']
        prediction=p.with_name(p.stem+'_predictions.npz')
        assert sha(prediction)==r['predictions_sha256']
        assert sha(p.with_name(p.stem+'_weights.npz'))==r['checkpoint_sha256']
        with np.load(prediction) as a:
            for k,v in rescore(a).items():np.testing.assert_allclose(r[k],v,atol=1e-12,rtol=0)
        baseline[p.stem]=r;verified+=1
    assert len(baseline)==10
    for p in sorted((OUT/'itransformer').glob('*_seed*.json')):
        r=json.loads(p.read_text());tag=r['tag'];seed=r['seed']
        for rel,digest in r['artifact_sha256'].items():assert sha(OUT/rel)==digest
        with np.load(OUT/r['predictions_file']) as a,np.load(OUT/'baseline'/f'{tag}_predictions.npz') as old:
            for k in ['y','subject','session','context','run','onset']:np.testing.assert_array_equal(a[k],old[k])
            for k,v in rescore(a).items():np.testing.assert_allclose(r[k],v,atol=1e-12,rtol=0)
        assert not set(r['split']['train_subjects'])&set(r['split']['validation_subjects'])
        if tag.startswith('loso_'):
            assert not set(r['split']['outer_evaluation_subjects'])&(set(r['split']['train_subjects'])|set(r['split']['validation_subjects']))
        neural[tag,seed]=r;verified+=1
        foldrows.append(dict(model='itransformer',tag=tag,seed=seed,selected_epoch=r['selected_epoch'],
                             **{k:r[k] for k in ['session_mean','cell_mean','log_loss','brier','inference_ms_per_window']}))
    assert len(neural)==30
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['cv_hashes'].items():assert sha(OUT/rel)==digest
    tags=sorted(k for k in baseline if k.startswith('loso_'));assert len(tags)==8
    def summary(name,get):
        return dict(model=name,loso_session_mean=float(np.mean([get(t)['session_mean'] for t in tags])),
                    loso_cell_mean=float(np.mean([get(t)['cell_mean'] for t in tags])),
                    later_session_mean=get('later_last')['session_mean'],test_cell_mean=get('test_official')['cell_mean'],
                    test_session_mean=get('test_official')['session_mean'],test_log_loss=get('test_official')['log_loss'],test_brier=get('test_official')['brier'])
    summaries=[summary('Lag4 single',lambda t:baseline[t])]
    seedrows=[summary(f'iTransformer seed{seed}',lambda t,s=seed:neural[t,s]) for seed in SEEDS]
    average=dict(model='iTransformer mean3seeds',**{k:float(np.mean([r[k] for r in seedrows])) for k in seedrows[0] if k!='model'})
    summaries += seedrows+[average]
    np.testing.assert_allclose(selection['scores']['baseline'],summaries[0]['loso_session_mean'])
    np.testing.assert_allclose(selection['scores']['itransformer'],average['loso_session_mean'])
    diffs=np.array([np.mean([neural[t,s]['session_mean'] for s in SEEDS])-baseline[t]['session_mean'] for t in tags])
    rng=np.random.default_rng(0);boot=diffs[rng.integers(0,8,size=(10000,8))].mean(1)
    paired=dict(mean_difference=float(diffs.mean()),wins=int((diffs>0).sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),per_person_differences=diffs.tolist(),
                seed_loso_std=float(np.std([r['loso_session_mean'] for r in seedrows],ddof=1)))
    for name,rows in [('comparison.csv',summaries),('fold_seed_scores.csv',foldrows)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(verified_jobs=verified,exact_baseline_jobs=10,independent_metric_recomputation=True,seed_predictions_ensembled=False,inner_person_disjoint_validation=True),indent=2))
    lines=['# Author iTransformer без лагов против одиночного Lag4','',
           '| Модель | LOSO session BA | Поздние сессии BA | Public test cell BA |',
           '|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['model']} | {r['loso_session_mean']:.6f} | {r['later_session_mean']:.6f} | {r['test_cell_mean']:.6f} |")
    lines+=['',f"Зафиксированный по CV победитель: {selection['winner']}. Независимо проверены40 оценок.",'',
            'iTransformer: авторский encoder,41токен,480отсчётов,без лагов;1блок,d64,4головы,FFN128,dropout.2,GELU,линейная голова. Бейзлайн Lag4 имеет82лаговых ряда и адаптируется к сессии. Сравниваются полные решения.',
            'Для каждого внешнего разбиения эпоха выбрана по внутренним отложенным людям;затем выполнено новое обучение на всей внешней тренировочной части. Train/val кривые в itransformer/<fold>/seedN/selection/curves.png;CSV сохраняется после каждой эпохи. Внешние test-окна не использованы для ранней остановки.',
            'Среднее3seed — среднее метрик отдельных моделей,не ансамбль предсказаний. Public test использовался ранее;это не sealed test. Сабмишен не создавался.',
            f"Парная LOSO разница {100*paired['mean_difference']:+.3f}п.п.;победы{paired['wins']}/8;описательный95%CI [{100*paired['ci95'][0]:+.3f},{100*paired['ci95'][1]:+.3f}]п.п. Обучающие выборки LOSO пересекаются."]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('VERIFIED ALL40 ITRANSFORMER/BASELINE JOBS',flush=True)
if __name__=='__main__':main()
