"""Independent local verification and human summary of completed CPU work."""
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'src'))
from common import ROOT,sha,write_json

def scorer():
    path=ROOT/'ensemble_comparison/analyze.py'
    spec=importlib.util.spec_from_file_location('independent_metric_checker',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.independent_scores

def check(path,score):
    r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['prediction_sha256']
    with np.load(path) as f:
        values={k:f[k].copy() for k in f.files}
        np.testing.assert_array_equal(values['pred'],values['proba'].argmax(1))
        for key,value in score(values).items():np.testing.assert_allclose(value,r[key],rtol=0,atol=1e-12)
    return r,values

def main():
    score=scorer();fusion=HERE/'experiments/local_fusion/results';source=ROOT/'ensemble_comparison/downloads/recovery-v3/ensemble-results'
    selection=json.loads((fusion/'selection-before-test.json').read_text())
    assert len(selection['cv_files'])==153
    for name,digest in selection['cv_files'].items():assert sha(fusion/name)==digest
    cfg=json.loads((fusion/'protocol.json').read_text())
    for commit in subprocess.check_output(['git','log','--format=%H','--','PROTOCOL.md'],cwd=HERE,text=True).splitlines():
        data=subprocess.check_output(['git','show',commit+':PROTOCOL.md'],cwd=HERE)
        if hashlib.sha256(data).hexdigest()==cfg['protocol_sha256']:
            (HERE/'experiments/local_fusion/protocol-locked.md').write_bytes(data);break
    else:raise RuntimeError('Cannot recover pre-result protocol')
    count=0
    for path in fusion.glob('*/*.npz'):
        _,values=check(path,score);count+=1
        with np.load(source/'results'/path.stem/'baseline.npz') as old:
            for key in ('y','subject','session','context','run','onset'):np.testing.assert_array_equal(values[key],old[key])
    assert count==170
    reconstructed=0
    for path in (HERE/'experiments/local_fusion/cache').glob('*.npz'):
        with np.load(path) as cached,np.load(source/'results'/path.stem/'variant0.npz') as f,np.load(source/'results'/path.stem/'variant4.npz') as old:
            from scipy.special import softmax
            q=softmax(cached['logits'],axis=1);p=softmax(.7*np.log(f['proba']+1e-9)+.3*np.log(q+1e-9),axis=1)
            np.testing.assert_allclose(p,old['proba'],rtol=0,atol=1e-4);np.testing.assert_array_equal(p.argmax(1),old['pred'])
        reconstructed+=1
    assert reconstructed==10
    baseline_root=HERE/'experiments/all10_baseline/results';baseline_rows=[]
    for name in ('lag4','lag4_balance1'):
        cv=[];personal=[];late=None
        for folder in sorted(baseline_root.iterdir()):
            if not folder.is_dir():continue
            complete=json.loads((folder/'complete.json').read_text())
            for file,digest in complete['files'].items():assert sha(folder/file)==digest
            record,values=check(folder/(name+'.npz'),score)
            if folder.name.startswith('loso_'):cv.append(values);personal.append(record['cell_mean'])
            else:late=record['cell_mean']
        assert len(cv)==10 and late is not None
        merged={key:np.concatenate([v[key] for v in cv]) for key in cv[0]}
        assert len(merged['y'])==2655
        keys=list(zip(merged['subject'],merged['session'],merged['run'],merged['onset']));assert len(set(keys))==2655
        pooled=score(merged)
        baseline_rows.append(dict(variant=name,official_oof_cell=pooled['cell_mean'],mean_person_cell=float(np.mean(personal)),late_cell=late))
    table=pd.read_csv(fusion/'comparison.csv').set_index('variant')
    winner=selection['winner'];assert winner==table.cell_mean_loso.idxmax()
    baseline=pd.DataFrame(baseline_rows);baseline.to_csv(HERE/'to_human/ALL10_BASELINE.csv',index=False)
    lines=['# Первая исследовательская волна: локальные результаты','',
        'Метрики независимо пересчитаны:170 legacy-оценок,22 новых all10-контроля;10 исходных ансамблей воспроизведены из CNN-логитов.',
        'Legacy LOSO здесь — среднее по8 исключённым людям. Старый public test уже использовался ранее.','',
        '| Вариант | Legacy LOSO cell BA | Поздняя cell BA | Старый public-test cell BA |','|---|---:|---:|---:|']
    for name in ('lag4','lag4_balance1','geo_w0.3_T1_a0',winner):
        r=table.loc[name];lines.append(f'| {name} | {100*r.cell_mean_loso:.2f}% | {100*r.cell_mean_later:.2f}% | {100*r.cell_mean_test:.2f}% |')
    lines += ['',f'Выбран до тестовой оценки по legacy LOSO:{winner}. Формула:70%Lag4+30%softmax(CNNlogits/4),затем причинная коррекция накопленных предсказанных частот сprior10,alpha1.',
        'Это исследовательский выбор из17фиксированных вариантов на неоднократно использовавшейся валидации;он не является подтверждённым улучшением sealed-метрики.','',
        '## Новые контроли: все10публичных людей','',
        'Official OOF cell — объединение всех2655исключённых предсказаний с официальным усреднением по ячейкам. Mean person — дополнительное равновесное среднее10людей.','',baseline.to_string(index=False),'',
        'Новый Kaggle:три CNN-рецепта,внутренний выбор эпох по исключённым людям,двеGPU,75минут бюджета. Фоновый опрос отключён по запросу пользователя.']
    (HERE/'to_human/LOCAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(HERE/'to_human/LOCAL_VERIFICATION.json',dict(legacy_scores=170,all10_scores=22,reconstructed_teacher_ensembles=10,cv_hashes=153,
        winner=winner,source_manifest=sha(source/'manifest.json'),local_protocol_snapshot_sha256=cfg['protocol_sha256']))
    print('\n'.join(lines))

if __name__=='__main__':main()
