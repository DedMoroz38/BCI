"""Independent rescore of all saved single-model predictions; no training."""
import csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'Brain-Decoding/track2_bci'))
from bci.evaluation import balanced_accuracy
OUT=Path(os.environ.get('SINGLE_OUTPUT',str(HERE/'results')))
MODES=['baseline','attention_gram','attention_exp']

def independently_score(a):
    y=a['y'];p=a['proba'];pred=p.argmax(1)
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
    np.testing.assert_array_equal(pred,a['pred'])
    sessions=np.array([f'{s}/{t}' for s,t in zip(a['subject'],a['session'])])
    cells=np.array([f'{s}/{c}' for s,c in zip(sessions,a['context'])])
    grouped=lambda group:float(np.mean([balanced_accuracy(y[group==g],pred[group==g]) for g in np.unique(group)]))
    return dict(session_mean=grouped(sessions),cell_mean=grouped(cells),pooled=balanced_accuracy(y,pred),
                accuracy=float((pred==y).mean()),log_loss=float(-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(float).eps)).mean()),
                brier=float(np.square(p-np.eye(3)[y]).sum(1).mean()))

def main():
    records={};foldrows=[];baseline_exact=0;historical=0;verified=0
    for mode in MODES:
        paths=sorted((OUT/mode).glob('*.json'));assert len(paths)==10,(mode,len(paths))
        records[mode]={}
        for path in paths:
            r=json.loads(path.read_text());npz=path.with_name(path.stem+'_predictions.npz')
            assert hashlib.sha256(npz.read_bytes()).hexdigest()==r['predictions_sha256']
            checkpoint=path.with_name(path.stem+'_weights.npz') if mode=='baseline' else path.with_suffix('.pt')
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==r['checkpoint_sha256']
            with np.load(npz) as a:
                values=independently_score(a)
                for k,v in values.items():np.testing.assert_allclose(r[k],v,atol=1e-12,rtol=0)
                if mode=='baseline':
                    assert r['exact_original_single_model'];baseline_exact+=1
                    if r['historical_prediction_match']:
                        if path.stem.startswith('loso_'):protocol='loso';fold='Dreyer2026Proteus_'+path.stem.removeprefix('loso_')
                        elif path.stem=='later_last':protocol='later';fold='last'
                        else:protocol='test';fold='official'
                        reference=ROOT/'temporal_spd_research/results/lag4_oas'/f'logreg_{protocol}_{fold}_predictions.npz'
                        with np.load(reference) as old:
                            for k in ['y','subject','session','context','run','onset','pred']:np.testing.assert_array_equal(old[k],a[k])
                        historical+=1
                else:
                    assert r['retained_final_head'] and r['full_history_gradient']
                    assert r['independent_numpy_inference_max_difference']<=1e-6
            verified+=1;records[mode][path.stem]=r;foldrows.append(dict(model=mode,fold=path.stem,**values))
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['cv_hashes'].items():assert hashlib.sha256((OUT/rel).read_bytes()).hexdigest()==digest
    summary=[];paired=[]
    for mode,rs in records.items():
        tags=sorted(k for k in rs if k.startswith('loso_'));loso=[rs[k]['session_mean'] for k in tags]
        np.testing.assert_allclose(np.mean(loso),selection['scores'][mode],atol=1e-14)
        row=dict(model=mode,loso_session_mean=float(np.mean(loso)),loso_cell_mean=float(np.mean([rs[k]['cell_mean'] for k in tags])),
                 later_session_mean=rs['later_last']['session_mean'],test_cell_mean=rs['test_official']['cell_mean'],
                 test_session_mean=rs['test_official']['session_mean'],test_log_loss=rs['test_official']['log_loss'],test_brier=rs['test_official']['brier'])
        summary.append(row)
        if mode!='baseline':
            diffs=np.array([rs[k]['session_mean']-records['baseline'][k]['session_mean'] for k in tags])
            rng=np.random.default_rng(0);boot=diffs[rng.integers(0,8,size=(10000,8))].mean(1)
            paired.append(dict(model=mode,mean_difference=float(diffs.mean()),wins=int((diffs>0).sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),per_person_differences=diffs.tolist()))
    for name,rows in [('comparison.csv',summary),('fold_scores.csv',foldrows)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(verified_jobs=verified,original_single_baseline_jobs=baseline_exact,historical_label_matches=historical,independent_metrics=True),indent=2))
    lines=['# Одиночный Lag4 и сквозной attention: проверенное сравнение','',
           '| Модель | LOSO session BA | Поздние сессии BA | Public test cell BA | Public test session BA |',
           '|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['model']} | {r['loso_session_mean']:.6f} | {r['later_session_mean']:.6f} | {r['test_cell_mean']:.6f} | {r['test_session_mean']:.6f} |")
    lines+=['',f"Победитель по зафиксированной CV: {selection['winner']}. Проверены30 оценок;10 одиночных бейзлайнов совпали с исходной реализацией.",'',
            'Леса нет. Все модели используют41 канал,полосу4–40Гц и лаг4. Attention обучен сквозным градиентом через prior,историю опорной матрицы,касательные признаки,стандартизацию и сохранённый конечный логрег. W_Q и W_K независимы.80шагов не гарантируют сходимость. Public test использовался ранее;это не sealed test. Сабмишен не создавался.']
    for r in paired:lines.append(f"\n{r['model']}: LOSO delta {100*r['mean_difference']:+.3f}п.п.;описательный95%CI [{100*r['ci95'][0]:+.3f},{100*r['ci95'][1]:+.3f}]п.п.;победы{r['wins']}/8. Обучающие выборки LOSO пересекаются.")
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('VERIFIED ALL30 SINGLE-MODEL JOBS',flush=True)
if __name__=='__main__':main()
