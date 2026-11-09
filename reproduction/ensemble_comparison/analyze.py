"""Rescore raw predictions independently; paired comparison, no parameter selection."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def paired(rows,variant):
    base={r['tag']:r['session_mean'] for r in rows if r['variant']=='Lag4' and r['tag'].startswith('loso_')}
    other={r['tag']:r['session_mean'] for r in rows if r['variant']==variant and r['tag'].startswith('loso_')}
    if base.keys()!=other.keys() or not base:raise ValueError('Unpaired participants')
    delta=np.array([other[k]-base[k] for k in sorted(base)])
    rng=np.random.default_rng(0)
    boot=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    return dict(n=len(delta),mean_delta=float(delta.mean()),wins=int(np.sum(delta>0)),
                ci95=np.quantile(boot,[.025,.975]).tolist(),per_person={k:float(v) for k,v in zip(sorted(base),delta)})

def independent_scores(f):
    def ba(idx):
        y,p=f['y'][idx],f['pred'][idx]
        return float(np.mean([np.mean(p[y==c]==c) if np.any(y==c) else 0 for c in np.union1d(y,p)]))
    stream=np.char.add(np.char.add(f['subject'],'/'),f['session'])
    session=np.mean([ba(np.flatnonzero(stream==s)) for s in set(stream)])
    cell=np.mean([ba(np.flatnonzero((stream==s)&(f['context']==c))) for s,c in set(zip(stream,f['context']))])
    return dict(session_mean=float(session),cell_mean=float(cell),pooled=ba(np.arange(len(stream))),
                accuracy=float(np.mean(f['y']==f['pred'])))

def main():
    rows=[]
    manifest=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()
    for path in sorted((HERE/'results').glob('*/*.json')):
        if path.stem!='baseline' and not path.stem.startswith('variant'):continue
        r=json.loads(path.read_text());p=path.with_suffix('.npz')
        if r['manifest_sha256']!=manifest:raise RuntimeError('Changed manifest')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['prediction_sha256']:raise RuntimeError('Changed predictions')
        with np.load(p) as f:
            np.testing.assert_array_equal(f['pred'],f['proba'].argmax(1))
            for k,v in independent_scores(f).items():np.testing.assert_allclose(v,r[k],atol=1e-12)
        r['protocol']='loso' if r['tag'].startswith('loso_') else ('test' if r['tag']=='test_official' else 'later')
        rows.append(r)
    if len(rows)!=80:raise RuntimeError(f'Expected80 evaluations,got{len(rows)}')
    df=pd.DataFrame(rows)
    if df.groupby(['variant','protocol']).size().tolist()!=[1,8,1]*8:
        # Explicit checks avoid relying on alphabetic group ordering.
        for _,g in df.groupby('variant'):
            if g.protocol.value_counts().to_dict()!={'loso':8,'later':1,'test':1}:raise RuntimeError('Incomplete evaluation')
    names=['Lag4','forest','+balance','+bands','+routing','+deep','all but deep','all']
    values=['session_mean','cell_mean','accuracy','log_loss','brier']
    summary=df.pivot_table(index='variant',columns='protocol',values=values,aggfunc='mean').reindex(names)
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(HERE/'comparison.csv');df.to_csv(HERE/'fold_scores.csv',index=False)
    pairs={name:paired(rows,name) for name in names[1:]}
    (HERE/'paired.json').write_text(json.dumps(pairs,indent=2))
    lines=['# Авторский ансамбль против Lag4','',
           'Локальное воспроизведение commit934f423. Все80 оценок пересчитаны по сохранённым предсказаниям.',
           'В8 LOSO-фолдах обучаемся на7 участниках, проверяемся на одном. Поздние сессии представлены одним участником.',
           'Публичный тест уже использовался ранее; параметры по нему не подбирались.','',
           '| Модель | LOSO session BA | LOSO cell BA | Поздняя session BA | Test cell BA | Δ LOSO к Lag4 | Победы |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        s=summary.loc[name];d=s['session_mean_loso']-summary.loc['Lag4','session_mean_loso']
        wins='—' if name=='Lag4' else f"{pairs[name]['wins']}/8"
        lines.append(f"| {name} | {100*s.session_mean_loso:.2f}% | {100*s.cell_mean_loso:.2f}% | {100*s.session_mean_later:.2f}% | {100*s.cell_mean_test:.2f}% | {100*d:+.2f} п.п. | {wins} |")
    lines+=['','95% интервалы описательные: бутстрап8 парных разностей по людям; обучающие выборки фолдов пересекаются.',
            'Все10 переобучений Lag4 точно совпали по предсказаниям с прежним бейзлайном.',
            'У автора одна ShallowConvNet seed0,100 эпох, без early stopping (как в benchmark.py); финальный make_submission по умолчанию использует3 seeds.',
            'Сохранены предупреждения оптимизации; сравнение не доказывает сходимость каждого классификатора.',
            'Маршрутизация проверяется отдельно на новых людях и поздней сессии. Одна поздняя сессия даёт очень ограниченную оценку.',
            'Гиперпараметры авторских методов и Lag4 не подбирались. Результат сравнивает полные методы, а не только эффект ансамблирования.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (HERE/'verification.json').write_text(json.dumps(dict(evaluations=80,baseline_matches=10,independent_rescore=True,manifest=manifest),indent=2))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
